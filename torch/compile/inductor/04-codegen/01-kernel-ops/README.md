# LoopBody 重放：把逐位置计算翻译成当前后端的表达式

[上一层：Codegen](../README.md) · [下一篇：CPU C++ 与 GPU/Triton 后端](../02-backends/README.md) · [返回总览](../../README.md)

Q34 把普通或融合 scheduler node 交给 device backend。以一个普通节点为例，它此时已经带着：

- 一份 `LoopBody`：逐位置计算的 FX 小图、索引式、循环范围和操作摘要；
- 经过 Scheduler 调整后的迭代轴、规约轴与顺序；
- 当前 backend 建立的 kernel 上下文，其中已经安装了负责生成目标表达式的 handler。

本篇观察的是 backend 生成 kernel 时发生的一次重放。比如节点表示
`out[i]=relu(x[i]+y[i])*2`，LoopBody 保存“load、add、relu、mul、store”的关系；CPU
handler 把它们翻译成 C++ 临时值，Triton handler 把它们翻译成 Triton 表达式与语句。
重放得到的是编译期代码对象，不是真实 tensor 的计算结果。

这项协作分成四部分：

```text
带 LoopBody 的 scheduler node + 当前 kernel/handler
  │
  ├─ 1. LoopBody 保存逐位置值流、索引和特殊子图
  ├─ 2. 当前循环位置代入索引，FX interpreter 调用当前 handler
  ├─ 3. backend 生成候选表达式，CSE 决定建立或复用临时值
  └─ 4. load/store/reduction/特殊控制经过缓存与删除判断，留下必要的 kernel 动作
         │
         └─ Q42 同时组织这些动作所在的 CPU loop 或 Triton grid
```

Q41 与 Q42 不是前后两个编译批次。Q42 所讲的 kernel 生成过程会安装 handler、安排循环或
grid，并在其中调用本篇的 LoopBody 重放；本篇只是把镜头对准每一项 ops 怎样变成表达式。

## 第一项：认识节点里已经保存好的 LoopBody

### 它在到达 Q34 之前怎样形成

Lowering 为 pointwise 或 reduction 先保存一个 `inner_fn`。pointwise 的 inner_fn 接收当前
逻辑位置，计算并返回一个元素值；它本身不负责给最终输出命名和写回。

随后，`ComputedBuffer.get_store_function()` 把这条值计算包装成完整规则：

```text
pointwise inner_fn(index) → value
                         + 输出名、输出 indexer
                         + ops.store(output, index, value)

reduction inner_fn(index, reduce_index) → value
                                      + ops.reduction(...)
                                      + ops.store_reduction(...)
```

Scheduler 建立 `SchedulerNode` 时会取得这份完整规则。它先通过 `CaptureIndexing` 执行
一次规则，捕获默认轴顺序下的 LoopBody；再依据其中的访存索引决定怎样简化、合并或重排
循环；最后复制 FX graph，并把旧索引映射到新轴，形成节点保存的 LoopBody。因此 Q34
交来的节点已经有调整后的 LoopBody；Codegen 此刻不再从 Lowering 的 inner_fn 重新开始。

循环形态在到达本篇前还可能改变：

1. SchedulerNode 初建时先捕获默认 LoopBody，再按访存地址简化和重排轴，最后复制图并
   重映射索引；这一步不会再次运行原始 store 规则；
2. 融合判断为了让一对依赖的访问位置一致，可以调用 `reorder_iter_loops`；
3. 融合后，适用的 CUDA 或 CPU Halide 路径可以继续 `merge_loops`，并刷新读写依赖。

所以本篇重放的是调整后的 LoopBody 与循环变量，不是假定仍使用 Lowering 最初的轴顺序。

### LoopBody 保存三类内容

用 `relu(x+y)*2` 作为 pointwise 例子。完整的 store 规则可以理解成：

```python
def store_rule(i):
    a = ops.load("arg0_1", i)
    b = ops.load("arg1_1", i)
    c = ops.add(a, b)
    d = ops.relu(c)
    e = ops.mul(d, ops.constant(2.0, torch.float32))
    return ops.store("buf0", i, e)
```

建立 LoopBody 时，`CaptureIndexing` 安装一个 FX tracer handler，用符号位置执行这条规则。
load 并不读取真实输入；tracer 返回 FX proxy，后续 add、relu、mul 使用这些 proxy，从而记录
节点之间的值流。

捕获后，LoopBody 保存：

| 内容 | 本例中的值 | 后续消费者 |
| --- | --- | --- |
| root FX graph | `load x → load y → add → relu → constant → mul → store` | Codegen 重放完整值流 |
| `indexing_exprs` | `index0 = i` | 重放和依赖分析把当前循环变量代入 |
| operation summary | `LOAD x[index0]`、`LOAD y[index0]`、`STORE buf0[index0]` | 依赖分析快速建立 ReadWrites |

operation summary 不只包含内存动作。它还可能记录 `LOAD_SEED`、`STORE_REDUCTION`、
`INDEX_EXPR`、`CHECK_BOUNDS` 和 `BUCKETIZE`。add、relu、mul 之间的值流保存在 FX graph，
不会重复列进这个摘要。

本例等价的 root graph 是：

```python
def root(self, ops):
    index = self.get_index("index0")
    a = ops.load("arg0_1", index)
    b = ops.load("arg1_1", index)
    c = ops.add(a, b)
    d = ops.relu(c)
    two = ops.constant(2.0, torch.float32)
    e = ops.mul(d, two)
    return ops.store("buf0", index, e)
```

图中的 `ops` 是 placeholder。捕获时它只表示“以后会传入一个解释这些操作的对象”；真正
重放时才传入 CPU、Triton 或其他当前 handler。

已有 LoopBody 做依赖分析时通常不重放 add/relu/mul。`extract_read_writes()` 直接遍历上述
operation summary，把 LOAD/STORE 等项还原成 MemoryDep；它也处理 INDEX_EXPR 和 BUCKETIZE，
并有意跳过只做检查的 CHECK_BOUNDS。输入仍是普通 callable 时，才临时安装
`RecordLoadStore` 并运行 callable 来收集读写。

## 第二项：代入当前索引，用当前 handler 重放 FX 图

`SchedulerNode.codegen(index_vars)` 接收 Q42 当前 kernel 安排的循环变量。它安装两层动态
上下文：

- `SimplifyIndexing` 根据本节点的范围化简访存索引；
- current-node context 告诉 kernel 当前表达式属于哪个 scheduler node。

随后调用 `LoopBody(*index_vars)`。LoopBody 把当前变量代入 `indexing_exprs`，再让 root
`LoopBodyBlock` 运行。

`LoopBodyBlock` 使用 `InterpreterShim` 逐个执行 FX node，并把当前
`V.get_ops_handler()` 作为 root graph 的 `ops` 实参：

```text
Q42 安排当前 index_vars
  → SchedulerNode.codegen(index_vars)
  → LoopBody 把 index_vars 代入 index0、index1……
  → InterpreterShim.run(root FX graph, 当前 handler)
  → load/add/relu/store 依次调用 handler 的同名方法
```

`V.set_ops_handler(handler)` 把 handler 放在线程局部的动态作用域；退出上下文时，即使内部
抛错，也会恢复先前 handler。模块级 `ops` 经过 `OpsWrapper`：它把参数中的 `OpsValue`
解包，调用当前 handler，再把有值的结果包装回来，使规则能写成 `a*b+c`。`V.ops` 则直接
取得未包装的当前 handler。

`OpsHandler` 约定 load、add、reduction、store 等方法的输入形式。产生值的操作返回当前
后端定义的值对象；本篇实验中的具体 store handler 返回 `None`，同时把写入语句加入 kernel。

## 第三项：backend 先给候选表达式，CSE 再决定临时值

进入具体 kernel 后，当前 handler 通常是包在 backend override 外面的 `CSEProxy`。一项
普通有值操作经历：

```text
LoopBody 中的 ops.add(a, b)
  → CSEProxy.add(a, b)
  → 当前 backend override 产生候选表达式
  → CSE 表在当前作用域按候选代码文本查询
       ├─ 未命中：建立临时变量，发出定义，登记候选式
       └─ 命中：返回已有临时变量
```

### pointwise：同一值流在 CPU 与 Triton 中的出口

输入：

```text
x = [-2, 1, 3]
y = [ 1, 4,-1]
out = relu(x+y)*2
```

当前 CPU 标量轨迹是：

| ops 调用 | 返回或动作 |
| --- | --- |
| `load(arg0_1,i)` | `CppCSEVariable tmp0` |
| `load(arg1_1,i)` | `CppCSEVariable tmp1` |
| `add(tmp0,tmp1)` | `tmp2` |
| `relu(tmp2)` | `tmp3` |
| `constant(2.0,float32)` | `tmp4` |
| `mul(tmp3,tmp4)` | `tmp5` |
| `store(buf0,i,tmp5)` | 向当前 kernel 加入写语句；本例返回 `None` |

对应 C++ 片段为：

```cpp
auto tmp0 = in_ptr0[x0];
auto tmp1 = in_ptr1[x0];
auto tmp2 = tmp0 + tmp1;
auto tmp3 = std::max(tmp2, 0.0f);
auto tmp4 = 2.0f;
auto tmp5 = tmp3 * tmp4;
out_ptr0[x0] = tmp5;
```

位置 0、1、2 的结果分别是 0、10、4。实验 12 实际记录了这条 CPU handler 轨迹并执行生成
代码；它没有运行 GPU。

同一 LoopBody 在默认 CUDA Triton 路径中仍按相同次序调用 load、add、relu、mul、store。
差别是 handler 返回 `TritonCSEVariable`，load/store 产生类似 `tl.load(pointer+index)`
与 `tl.store(pointer+index,value)` 的 Triton 表达式和语句。具体 pointer、mask 和 program
id 由 Q42 的 Triton kernel 组织；这里没有先产生一份通用 tmp 清单再交给 Triton。

若同一 kernel 的同一 CSE 作用域再次产生完全相同的候选代码，例如第二次
`maximum(tmp0,tmp1)`，CSE 表可以直接返回第一次建立的临时值。换 dtype、mask 或执行条件
时，backend 可能生成不同代码，masked/subblock 也会使用隔离的 CSE 作用域；这时不能只因
高层算子名称相同就复用临时值。

load/store 还有一份本 kernel 的 store cache。普通 store 会记录“逻辑 buffer 名 → 刚写入的
CSE 临时值”。随后在同一作用域按这个逻辑名进行非间接 load 时，可以直接返回该临时值，
不再增加一次内存读取。例如：

```text
v = add(a,b)
store("buf0", i, v)   → store_cache["buf0"] = v
w = load("buf0", i)   → 返回 v；后续计算直接使用同一个临时值
```

这条快速路径固定为普通 store、同一逻辑名、当前 CSE 作用域和非间接 load。atomic 等 store
mode 不写这项缓存；带临时符号的间接 load 直接走间接读取；masked 子图有自己的 CSE 作用域。
mutation 输出还会把明确声明为被修改的旧名字指向同一个值，普通 view alias 不会仅因共享
物理存储就自动获得同一 cache 项。

被删除的中间 buffer 也分两个时点。若重放前它已经在图的 removed 集合中，store 仍把值留在
store cache 供融合体内的后继使用，但不生成目标内存写入。若重放结束后才确认它只在当前
kernel 内使用，原先加入的 deferred store line 会在最终渲染时被过滤。两条路径都省去中间
存储写回，同时保留后继计算需要的值。

### reduction：重放产生累计动作，Q42 把动作放进规约循环

对 2×128 输入计算 `(x+1).sum(dim=1)`，LoopBody 的核心值流是：

```python
value = ops.load("arg0_1", 128*x0 + x1)
value = ops.add(value, ops.constant(1.0, torch.float32))
acc = ops.reduction(torch.float32, torch.float32, "sum", value)
ops.store_reduction("buf0", x0, acc)
```

`x0` 选择输出行，`x1` 遍历该行 128 项。对一个标量 CPU handler，可以把语义推演为：

```text
进入 x1 循环前：acc = 0
x1=0：读取 x[0,0]=0，加1得到1，acc=1
x1=1：读取 x[0,1]=1，加1得到2，acc=3
...
x1=127：读取127，加1得到128，acc=8256
离开 x1 循环：store_reduction buf0[0] = 8256
```

第二行重新初始化，结果为 24640。`ops.reduction` 让 handler 建立或更新累计对象，
`store_reduction` 表示规约循环结束后的写回，并不是每个 x1 都写一次输出。

这段是标量 handler 的语义推演。实验 12 的最终 CPU kernel 实际选择 16-lane vector
reduction，再做横向合并；编译时也为了候选生成多次重放 reduction LoopBody。协议保证的是
同一规约语义，不保证只重放一次或固定采用标量循环。

## 第四项：特殊索引和子图仍在本次重放中解决

### 间接索引：先算索引值，再把它代入后续访存

考虑：

```python
j_value = ops.load("index", i)
j = ops.indirect_indexing(j_value, data_size, check=True)
v = ops.load("data", j)
ops.store("out", i, v)
```

捕获时不能把 `j_value` 当成已知整数。CaptureIndexing 建立一个间接索引符号，例如
`indirect0`，让 data 的索引式先引用它；FX graph 还保存一个负责“把刚加载的 j_value 绑定
到 indirect0”的子模块调用。

重放顺序是：

```text
load index[i] → 当前后端值 tmp_j
set_indirect0(tmp_j)
get_index(data_index) 将 indirect0 替换为 tmp_j 对应表达式
load data[data_index]
store out[i]
```

CPU handler 因而生成用临时整数索引指针的读取；Triton handler 生成用当前 Triton 值计算
地址的 load。`check=True` 所需的范围检查也由当前后端表达。若后端不能实现该检查或索引
形式，异常向上终止编译。

### masked：把受保护的计算保存成子 LoopBodyBlock

考虑 `i<n` 时读取 `x[i]+1`，否则使用 0：

```python
mask = ops.lt(i, n)
value = ops.masked(mask, masked_body, 0.0)
ops.store("out", i, value)
```

捕获时，`masked_body` 单独成为一个 `LoopBodyBlock`。LoopBody 为它建立一个 masked shim，
root graph 记录的是对这个 shim 子模块的调用，并传入 mask 与 other。重放该子模块时，shim
才调用当前 handler 的 `masked(mask, subblock, other)`；handler 决定怎样生成条件，以及何时
调用 subblock。subblock 仍通过同一个 interpreter 和当前 handler 生成 load、add 等表达式，
返回值再交给 root graph 的 store。

CPU 可以把这条关系实现为受条件保护的表达式或语句，Triton 可以把条件变成 load/store 的
mask 或选择表达式；精确形式由各 backend 决定。handler 未实现 masked 或无法生成这个
subblock 时，错误直接传播，本层没有另一个自动实现。

其他复合操作沿用相同原则：

| 操作 | LoopBody 中额外保存什么 | 重放时谁接住 |
| --- | --- | --- |
| `scan` | combine 函数对应的子模块/shim | 当前 handler 的 scan 实现 |
| `sort` | 多返回值关系，FX proxy 分别代表 tuple 各项 | 当前 handler 的 sort 实现 |
| `bucketize` | offsets 及 BUCKETIZE 摘要项 | handler 生成查找表达式；依赖快速路径也读取该摘要 |
| `index_expr` | 非常量索引对应的 INDEX_EXPR 摘要 | handler 生成当前 dtype 的索引值 |

这些操作没有一个通用 fallback。当前 backend 不支持相应方法、store mode 或 reduction type
时，通常抛出 `NotImplementedError` 或其他编译异常。`SchedulerNode.codegen()` 会记录
出错节点并重新抛出，不会改回 eager。

mutation/scatter 也通过同一 store 接口进入 handler：store 的 mode 可以表示
`atomic_add` 等写法，原地输出还会结合 kernel 的复用和 mutation 状态。CPU、Triton 或扩展
backend 必须明确支持该 mode；未知 mode 会失败。

普通 materialized `ComputedBuffer` 的规则由 `get_store_function()` 包含 store，即使迭代
范围为 0，捕获的规则仍有写回动作，只是运行时循环没有位置可执行。人工或特殊 LoopBody
可以没有 store，甚至 operation summary 为空；重放只产生图中实际存在的表达式，不会凭空
补一条输出写入。

重放正常结束后会清除本次索引代入状态。当前实现若 root block 在中途抛错，handler 的动态
上下文仍会恢复，但 LoopBody 内部的 `indexing` 临时字段可能来不及清空；编译此时已经沿
异常路径终止。

## 本篇出口：表达式生成与 kernel 组织发生在同一次调用里

正常重放不会返回 tensor，也不会建立一份跨后端通用的中间代码。load 可以命中已有 CSE
值或 store cache，被删除 buffer 的 store 也可以跳过；其余必要内容才进入当前 kernel：

| 本次重放留下的内容 | 具体形式 | 同时工作的 Q42 职责 |
| --- | --- | --- |
| 有值操作 | CPU/Triton CSE variable | 安排它所在的 loop、vector lane 或 program instance |
| load/store | 当前目标语言的读取与写入动作 | 计算 pointer、mask、循环或 grid |
| reduction | 累计对象与写回动作 | 安排初始化、规约循环、并行合并和最终写回位置 |
| masked/scan/sort/indirect | backend 特定表达式、子图或控制动作 | 把这些动作放入合法 kernel 结构 |

随后 Q42 从另一个观察角度解释同一次 kernel 生成：CPU 怎样形成循环和向量 kernel，Triton
怎样形成 program grid、索引与 mask。Q41 的 LoopBody 值流与 Q42 的 kernel 空间结构合在
一起，才是 Q34 backend 最终提交给 wrapper 的 kernel 源码。

## 用新操作推算

若同一个 pointwise LoopBody 在相同 dtype、mask 和执行条件下包含：

```text
t0 = load(x, i)
t1 = constant(0, float32)
t2 = maximum(t0, t1)
t3 = maximum(t0, t1)
t4 = add(t2, t3)
store(y, i, t4)
```

捕获结果是一个 load、一个 store 的摘要，以及包含两次 maximum 的完整 FX 值流。重放时，
第一次 maximum 的候选式建立临时值；第二次候选式相同，CSE 返回已有值；add 接到的是同一
临时值两次。CPU 关键代码可写成：

```cpp
auto tmp0 = in_ptr0[x0];
auto tmp1 = 0.0f;
auto tmp2 = max_propagate_nan(tmp0, tmp1);
auto tmp3 = tmp2 + tmp2;
out_ptr0[x0] = tmp3;
```

若第二次 maximum 位于不同 masked/subblock 作用域，就不能仅凭高层表达式相同认定为
同一值。判断一个新案例时，应依次找：节点中的 LoopBody 保存了什么值流和子图；Q42 传入
哪些当前索引；当前 handler 为每项 ops 产生什么对象；CSE 条件是否相同；动作最终写进当前
kernel 的哪个结构。

## 继续阅读实现

[LoopBody 重放源码：handler、CSE 与规约状态](01-replay-source/README.md)
