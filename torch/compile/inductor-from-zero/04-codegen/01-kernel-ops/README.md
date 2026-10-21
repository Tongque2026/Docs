# 虚拟化 `ops`：同一串读取、计算和写入怎样变成不同目标代码

[上一层：Codegen](../README.md) · [返回总览](../../README.md) · 下一项：CPU C++ 与 GPU/Triton 后端

Lowering 保存的计算规则不是一段预先写死的 C++。例如 `out[i]=x[i]+1` 被保存为“load x 的位置 i、加常数 1、store 到 out 的位置 i”。运行这份规则时，当前动态安装的 handler 决定每一步返回什么：依赖分析 handler 在集合中登记一次读写，CPU handler 返回 C++ 表达式，Triton handler 返回 Triton 表达式。

```text
同一份 LoopBody
      │
      ├─ 捕获 handler            → 运行 callable 时把 load/add/store 记成 FX 小图与索引表
      ├─ 依赖分析 handler       → MemoryDep / ReadWrites
      ├─ CPU handler            → C++ 标量/向量表达
      ├─ Triton handler         → tl.load / tl.* / tl.store
      └─ 分析 handler           → 每个值可能的范围、操作数量、格式化文本等
```

这是一种 define-by-run 表示：计算过程保存在一个可调用对象里；依赖分析器或代码生成器运行它，并分别让 load/add/store 产生依赖记录或代码表达式，而不是每个阶段再解析一套新语法。

先固定三个容易混淆的东西：

| 名称 | 里面实际有什么 | 什么时候存在 |
| --- | --- | --- |
| 原始 callable | 调用 `ops.load/add/store` 的 Python 计算过程 | Lowering 构造计算规则时 |
| `LoopBody` | FX 操作小图、SymPy 索引表、循环范围、内存读写；masked 等嵌套计算另存为子图 | 捕获一次后保存到计算节点 |
| 当前 handler | 同一项 ops 调用在本次分析/生成中怎样执行 | 进入动态上下文时安装，退出时恢复 |

因此“重放 LoopBody”不是重新执行用户模型，也不是拿真实 tensor 再算一遍；它用符号循环位置执行保存的小图，让 handler 产生依赖记录或目标代码对象。

## `OpsHandler[T]` 规定每项操作接收什么、返回什么

`OpsHandler` 是一个 Python Protocol，也就是一组方法要求：实现者要提供 `load(name,index)`、`add(a,b)`、`store(name,index,value)` 等方法，并返回后续方法能够继续使用的对象。类型参数 `T` 表示这些中间值的类型。它包含几类操作：

| 类别 | 例子 | 作用 |
| --- | --- | --- |
| 常量与算术 | `constant`、`add`、`mul`、`relu` | 描述单个逻辑元素的计算 |
| 类型与索引 | `to_dtype`、`index_expr`、`indirect_indexing` | 生成计算值或可用于访存的索引 |
| 内存 | `load`、`store` | 从 buffer 的线性位置读取或写入 |
| 规约 | `reduction`、`store_reduction` | 在规约轴累计，再写最终值 |
| 复合控制 | `masked`、`scan`、`sort`、`bucketize` | 表达需要后端特殊实现的结构 |

这些操作通常是标量语义。`ops.add(a,b)` 表达当前输出位置上的两个值相加；外层循环遍历哪些位置由 kernel backend 安排。ATen 分解和 Lowering 此前已经完成 dtype promotion，handler 不负责重新执行完整 PyTorch dtype 规则。

当 `T=str` 时，一个 handler 可以返回代码字符串；当 `T=CSEVariable` 时，它返回包含临时名字、表达式、dtype 和复用信息的对象，相同表达式可直接复用已有临时值；依赖分析也可让 `T` 只是占位文本，同时把 load/store 记入集合。

## `V.ops` 是动态作用域中的当前解释器

`Virtualized` 用线程局部键保存当前 handler。`V.set_ops_handler(handler)` 返回上下文管理器：进入时安装新 handler，退出时恢复旧 handler，即使内部抛异常也会恢复。

```python
with V.set_ops_handler(cpu_handler):
    body(indices)     # ops.add 被解释成 C++ 表达

with V.set_ops_handler(dependency_handler):
    body(indices)     # ops.load/store 被记录为依赖
```

调用点只访问统一的 `V.ops` 或模块级 `ops`，不需要把 CPU/GPU/分析器参数层层传递。动态作用域还允许局部再包一层 `SimplifyIndexing` 或记录器，离开后不会污染下一次解释。

模块级 `ops` 是 `OpsWrapper`。它会解包参数中的 `OpsValue`，调用当前 `_ops` handler，再把结果包装回 `OpsValue`，从而让 Lowering 规则可以写成 `a*b+c`。`OpsValue.__add__`、`__mul__` 等最终仍转回当前 handler 的标量操作。

## 计算规则先被捕获成 `LoopBody` 的 FX 小图

`LoopBody` 接收原始 callable、循环变量和范围。第一次建立时，`LoopBodyBlock` 用 FX tracer 创建一个名为 `ops` 的 placeholder，并安装 `CaptureIndexing` handler：

1. 每次 load/store 的 SymPy 索引先登记到 `indexing_exprs`。
2. 索引在 FX 图中变成对 `get_index(name)` 的调用。
3. 算术操作记录为对 `ops` placeholder 的方法调用。
4. `masked` 内部的计算另建一张 subblock 小图，例如只在 mask 为真时执行的 load/add；间接索引则保存一个先计算新索引、再交回外层图使用的小函数（shim）。

pointwise 实验的真实 LoopBody 图为：

```python
def op0(self, ops):
    index = self.get_index("index0")
    a = ops.load("arg0_1", index)
    b = ops.load("arg1_1", index)
    c = ops.add(a, b)
    d = ops.relu(c)
    two = ops.constant(2.0, torch.float32)
    e = ops.mul(d, two)
    return ops.store("buf0", index, e, None)
```

这里是按实际图整理的等价写法；原始 `debug_str()` 中每个索引读取有独立临时名，但都引用同一 `index0`。图没有出现具体 C++ 语法，也没有真实读取 tensor。

捕获前后的对象变化可以写成：

```text
callable(x0)
  说：读取两个 buffer 的 x0，做 add/relu/mul，写结果
        │ CaptureIndexing + FX proxy
        ▼
LoopBody
  indexing_exprs = {index0: x0}
  memory_usage   = 两次 LOAD + 一次 STORE
  root FX graph  = ops.load → ops.add → ops.relu → ops.mul → ops.store
```

`memory_usage` 只保存内存动作及其索引名；add、relu、mul 的因果顺序在 FX 图里。后面做依赖分析时可以只消费前者，生成代码时则重放整张图。

## 重放时，FX Interpreter 把当前 handler 当作参数

`LoopBody.__call__(*indices)` 先把登记的索引式代入当前循环变量，然后调用 root block。`LoopBodyBlock.__call__()` 使用 `InterpreterShim` 执行这张 FX 小图，并把 `V.get_ops_handler()` 作为图中 `ops` placeholder 的实参。

所以同一张图里的 `ops.add` 会调用传入对象的 `add` 方法。换 handler 就换解释结果，FX 图自身不变。`V.ops` 直接返回未包装的当前 handler；模块级 `ops` 则经过 `OpsWrapper`，会把普通返回值包成支持 `+`、`*` 等运算符的 `OpsValue`。两者指向同一动态选择，但外层接口并不相同。

SchedulerNode 进入 codegen 时还会包一层 `SimplifyIndexing`，把已知循环范围用于简化索引；同时 `V.kernel.set_current_node(self)` 让 handler 知道当前节点、输出和每个中间值可能的最小/最大范围（bounds）。这些范围由此前的值域分析根据常量、循环上下界和算术传播得到。

## CPU pointwise 如何从 ops 变成 `tmp0…tmp5`

输入：

```text
x=[-2,1,3]
y=[1,4,-1]
return relu(x+y)*2
```

本次实际重放顺序和返回对象是：

| ops 调用 | CPU codegen 返回 |
| --- | --- |
| `load(arg0_1, x0)` | `tmp0` |
| `load(arg1_1, x0)` | `tmp1` |
| `add(tmp0,tmp1)` | `tmp2` |
| `relu(tmp2)` | `tmp3` |
| `constant(2.0,float32)` | `tmp4` |
| `mul(tmp3,tmp4)` | `tmp5` |
| `store(buf0,x0,tmp5)` | 无返回值 |

这些 `tmp` 实际是 `CppCSEVariable`，携带名称、dtype、值域 bounds、是否向量以及依赖哪些迭代变量。生成代码对应为：

```cpp
auto tmp0 = in_ptr0[x0];
auto tmp1 = in_ptr1[x0];
auto tmp2 = tmp0 + tmp1;
auto tmp3 = std::max(tmp2, 0.0f);
auto tmp4 = 2.0f;
auto tmp5 = tmp3 * tmp4;
out_ptr0[x0] = tmp5;
```

位置 0 得到 `max(-2+1,0)*2=0`；位置 1 得到 10；位置 2 得到 4。LoopBody 决定操作顺序，CPU handler 决定 `relu` 用 C++ `max`、load/store 用指针索引，外层 CPU kernel 决定生成 `x0=0..2` 的循环。

## `CSEProxy` 在 backend override 外再包一层公共逻辑

进入具体 kernel context 时，`Kernel.__enter__()` 先取得后端 override handler，再把 `CSEProxy` 安装成当前 ops handler。大多数普通操作经历：

```text
LoopBody ops.add
    -> CSEProxy.add
    -> CPU/Triton override.add 生成候选表达式
    -> CSE cache 查重并分配临时变量
    -> CSEVariable
```

`CSEProxy` 还完成多后端共有的工作：

- 用预先分析或 `ValueRangeAnalysis` 附加 bounds；
- 相同表达式只生成一次，复用临时变量；
- load 可复用已知 store cache，间接索引 load 放到 compute 区；
- store 更新 cache，并跳过已被删除的 buffer；
- 统计 load、reduction，处理 mutation 的同存储名字；
- 间接索引例如 `idx[i]` 先产生实际位置 p；若已知 `-N<=p<N`，负值可修正为 `p+N`，随后检查修正结果是否落在 `[0,N)`，越界则触发断言。

因此 `tmp4=constant(2)` 后续若再次出现可直接复用；规约实验中三次重放也各自在自己的 kernel/CSE 作用域内生成临时名，不跨 kernel 错误复用。

## 规约多出“输入位置”和“累计状态”

使用 2×128 输入计算 `(x+1).sum(dim=1)`，真实 LoopBody 图为：

```python
value = ops.load("arg0_1", 128*x0 + x1)
value = ops.add(value, ops.constant(1.0, torch.float32))
acc = ops.reduction(torch.float32, torch.float32, "sum", value)
ops.store_reduction("buf0", x0, acc)
```

`x0` 选择输出行，`x1` 遍历这一行的 128 个元素。第一次 CPU scalar 重放记录：

```text
load -> tmp0
constant -> tmp1
add -> tmp2
reduction(sum,tmp2) -> tmp_acc0
store_reduction(buf0,x0,tmp_acc0)
```

同一 LoopBody 又为向量版本重放，load/add 返回对象的 `is_vec=True`。生成 C++ 最终采用宽度 16 的向量循环：

```cpp
for x0 in [0,2):
    tmp_acc0_vec = Vectorized<float>(0)
    for x1 in [0,128) step 16:
        tmp0 = loadu(input + 128*x0 + x1, 16)
        tmp3 = tmp0 + Vectorized<float>(1)
        tmp_acc0_vec += tmp3
    tmp_acc0 = vec_reduce_all(add, tmp_acc0_vec)
    output[x0] = tmp_acc0
```

第 0 行是 0 到 127，各加 1 后总和为 8256；第 1 行是 128 到 255，结果为 24640。`ops.reduction` 保留“求和”的共同语义，CPU 后端把它实现成向量累计与横向归并；GPU 后端可以采用不同程序结构。

## 同一份信息也服务于 Scheduler 分析

在 Q31 中，`extract_read_writes()` 得到了 MemoryDep。当前 `LoopBody` 已保存 `memory_usage` 和索引名，因此该函数有快速路径：先把新循环变量代回索引表，再直接遍历 LOAD、LOAD_SEED、STORE、STORE_REDUCTION、INDEX_EXPR、BUCKETIZE 记录并交给 `_RecordLoadStoreInner`，不必再次执行整张 FX 图；`CHECK_BOUNDS` 在这条快速路径中被有意跳过。对于普通 callable 的慢路径，则安装 `RecordLoadStore` handler 并实际 define-by-run。

两条路径交出的语义相同：哪些 buffer 被哪个索引读取或写入。这里需要准确区分“统一 handler 协议允许重放”和“当前优化后的快速路径是否真的重放整张图”。

`KernelFormatterHandler` 是另一个例子：它在 handler 返回的每条表达式前分配教学/调试用 `tmpN`，可把 callable 格式化成伪 IR 文本。`OpCounterCSE` 则统计使用了哪些 ops、读取哪些 buffer。这些都复用同一接口。

## 为什么不直接让 Lowering 拼 C++ 字符串

如果 Lowering 直接返回 C++：

- Scheduler 无法用同一规则提取精确依赖和索引；
- GPU/Triton、Halide 或扩展 backend 必须重新理解 C++；
- bounds、CSE、mask、间接索引和向量化难以在共同层处理；
- 同一语义无法为 scalar/vector 两种 CPU kernel 重放。

虚拟化 ops 把“做什么”留在 LoopBody，把“用什么目标表达”和“怎样组织循环”交给 kernel/backend。两者边界清楚，才让前面统一的 Lowering/Scheduler 能接到多种后端。

## 用新操作推算

若 LoopBody 是：

```text
t0 = load(x, i)
t1 = constant(0)
t2 = maximum(t0, t1)
store(y, i, t2)
```

可以先不关心目标语言，得到共同数据流“读 x[i]，与 0 取最大，写 y[i]”。然后分别判断：

- 依赖 handler 记录一项 x 的读和 y 的写；
- CPU handler 返回 C++ 临时变量与 `std::max`；
- 向量 CPU handler 返回向量值和向量 maximum；
- Triton handler 返回 `tl.load`、Triton maximum 与 `tl.store`；
- 若同一 maximum 在相同作用域重复，CSEProxy 可复用表达式。

这就是读懂后续任何 backend codegen 的入口：先找 LoopBody 发出的 ops，再看当前 handler 如何解释它们。

## 继续阅读实现

[LoopBody 重放源码：handler、CSE 与规约状态](01-replay-source/README.md)
