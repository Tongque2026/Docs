# 虚拟化 `ops`：同一串读取、计算和写入怎样变成不同目标代码

[上一层：Codegen](../README.md) · [返回总览](../../README.md) · 下一项：CPU C++ 与 GPU/Triton 后端

Lowering 保存的计算规则不是一段预先写死的 C++。例如 `out[i]=x[i]+1` 被保存为“load x 的位置 i、加常数 1、store 到 out 的位置 i”。这份规则先被捕获成 `LoopBody`；后续代码生成再用当前 handler 解释其中的每一步。

```text
Lowering 建立的 callable
      │ CaptureIndexing 运行 callable
      ▼
LoopBody = FX 操作图 + 索引表 + 内存动作摘要
      │
      ├─ 代码生成重放 FX 图
      │      ├─ CPU handler      → C++ 表达式
      │      └─ Triton handler   → Triton 表达式
      │
      └─ 依赖分析读取内存动作摘要 → MemoryDep / ReadWrites
```

这里有两种不同的“运行”。捕获阶段运行原始 callable，目的是建立 `LoopBody`；代码生成阶段执行 `LoopBody` 中保存的 FX 图，目的是产生目标表达式。对于已经建好的 `LoopBody`，依赖分析通常直接读取内存动作摘要，并不执行 add、relu、mul 等算术操作。

先固定三个容易混淆的东西：

| 名称 | 里面实际有什么 | 什么时候存在 |
| --- | --- | --- |
| 原始 callable | 调用 `ops.load/add/store` 的 Python 计算过程 | Lowering 构造计算规则时 |
| `LoopBody` | FX 操作小图、SymPy 索引表、循环范围、内存读写；masked 等嵌套计算另存为子图 | 捕获一次后保存到计算节点 |
| 当前 handler | 同一项 ops 调用在本次分析/生成中怎样执行 | 进入动态上下文时安装，退出时恢复 |

因此“重放 LoopBody”不是重新执行用户模型，也不是拿真实 tensor 再算一遍；它用符号循环位置执行保存的小图，让代码生成 handler 产生目标代码对象。依赖分析对已有 `LoopBody` 走摘要路径，只有收到普通 callable 时才用记录 handler 运行它。

## `OpsHandler[T]` 规定每项操作接收什么、返回什么

`OpsHandler` 是一个 Python Protocol，也就是一组方法要求：实现者要提供 `load(name,index)`、`add(a,b)`、`store(name,index,value)` 等方法。产生值的操作返回后续操作可以消费的对象；store 等动作型操作可以没有返回值。类型参数 `T` 表示这些中间值的类型。它包含几类操作：

| 类别 | 例子 | 作用 |
| --- | --- | --- |
| 常量与算术 | `constant`、`add`、`mul`、`relu` | 描述单个逻辑元素的计算 |
| 类型与索引 | `to_dtype`、`index_expr`、`indirect_indexing` | 生成计算值或可用于访存的索引 |
| 内存 | `load`、`store` | 从 buffer 的线性位置读取或写入 |
| 规约 | `reduction`、`store_reduction` | 在规约轴累计，再写最终值 |
| 复合控制 | `masked`、`scan`、`sort`、`bucketize` | 表达需要后端特殊实现的结构 |

这些操作通常是标量语义。`ops.add(a,b)` 表达当前输出位置上的两个值相加；外层循环遍历哪些位置由 kernel backend 安排。ATen 分解和 Lowering 此前已经完成 dtype promotion，handler 不负责重新执行完整 PyTorch dtype 规则。

当 `T=str` 时，一个 handler 可以返回代码字符串；代码生成时，`T` 可以是承载临时值信息的对象；依赖分析时，`T` 也可以只是占位值，同时把 load/store 记入集合。CPU 代码生成中这个对象具体怎样形成，将在第一个完整案例中说明。

## `V.ops` 是动态作用域中的当前解释器

`Virtualized` 用线程局部键保存当前 handler。`V.set_ops_handler(handler)` 返回上下文管理器：进入时安装新 handler，退出时恢复旧 handler，即使内部抛异常也会恢复。

```python
with V.set_ops_handler(codegen_handler):
    body(indices)     # ops.add 被解释成 C++ 表达

with V.set_ops_handler(dependency_handler):
    ordinary_callable(indices)  # 慢路径把 load/store 记录为依赖
```

调用点只访问统一的 `V.ops` 或模块级 `ops`，不需要把 CPU/GPU/分析器参数层层传递。在具体 kernel 中，安装为当前 handler 的通常是 `CSEProxy`：它先请 CPU 或 Triton 后端生成候选表达式，再为表达式分配或复用临时值。动态作用域还允许局部再包一层索引简化或记录器，离开后不会污染下一次解释。

模块级 `ops` 是 `OpsWrapper`。它会解包参数中的 `OpsValue`，调用当前 `_ops` handler，再把结果包装回 `OpsValue`，从而让 Lowering 规则可以写成 `a*b+c`。`OpsValue.__add__`、`__mul__` 等最终仍转回当前 handler 的标量操作。

## 计算规则先被捕获成 `LoopBody` 的 FX 小图

`LoopBody` 接收原始 callable、循环变量和范围。以 `relu(x+y)*2` 为例，原始 callable 可以先理解成下面的等价伪代码：

```python
left_name = "arg0_1"
right_name = "arg1_1"
output_name = "buf0"

def body(x0):                         # x0 是调用者传入的逻辑位置
    a = ops.load(left_name, x0)       # 三个名字由 body 从外层记住
    b = ops.load(right_name, x0)
    c = ops.add(a, b)
    d = ops.relu(c)
    e = ops.mul(d, ops.constant(2.0, torch.float32))
    return ops.store(output_name, x0, e)
```

`body` 的参数只有本次要计算的位置 `x0`。三个 buffer 名由 Python 闭包从外层变量中取得；它们不是作为真实 tensor 传给 `body`。建立 `LoopBody` 时，`LoopBodyBlock` 用 FX tracer 创建一个名为 `ops` 的 placeholder，并安装 `CaptureIndexing` handler：

1. 每次 load/store 的 SymPy 索引先登记到 `indexing_exprs`。
2. 索引在 FX 图中变成对 `get_index(name)` 的调用。
3. 算术操作记录为对 `ops` placeholder 的方法调用。
4. callable 返回后，tracer 已经拿到从 load 到 store 的完整节点关系，`LoopBody` 保存这张图。

FX proxy 是 tracer 返回的替身对象：它此刻没有真实数值；后续操作使用它时，tracer 据此记录“新节点读取哪个旧节点”。以第一次 `load` 和随后的 `add` 为例，捕获时的对象交接如下：

```text
LoopBodyBlock 用符号 x0 调用 body(x0)
  body 调用模块级 ops.load("arg0_1", x0)
  → OpsWrapper 把调用转给当前 handler：CaptureIndexing
  → CaptureIndexing 把 x0 登记成 index0，并建立 get_index("index0")
  → 内层 FX proxy 建立 ops.load("arg0_1", get_index(...)) 节点
  → OpsWrapper 把这个 FX proxy 包成 OpsValue，交回变量 a

body 随后调用 ops.add(a, b)
  → OpsWrapper 先从 a、b 中取出两个 FX proxy
  → 当前 handler 建立 ops.add(load_a, load_b) 节点
  → 新 FX proxy 包成 OpsValue，交回变量 c
```

`ops` placeholder 是这些节点共同引用的接收者。捕获时它只是 FX 图的一个参数；重放时才由 `InterpreterShim` 把当前代码生成 handler 传给它。

这个案例捕获出的等价 FX 图为：

```python
def op0(self, ops):
    index = self.get_index("index0")
    a = ops.load("arg0_1", index)
    b = ops.load("arg1_1", index)
    c = ops.add(a, b)
    d = ops.relu(c)
    two = ops.constant(2.0, torch.float32)
    e = ops.mul(d, two)
    return ops.store("buf0", index, e)
```

图没有出现具体 C++ 语法，也没有真实读取 tensor。`ops` 是一个待传入的参数：重放时传 CPU handler，方法调用就产生 C++ 侧的值；传其他 handler，则得到另一种解释。

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

SchedulerNode 进入 codegen 时还会用已知循环范围简化索引，并把当前节点放入动态作用域。这样 handler 能把本次产生的表达式归到正确节点。具体索引化简和值域传播属于后续实现篇。

## CPU 表达式经过 backend override 和 `CSEProxy` 两步形成

进入具体 kernel 时，当前 handler 通常是 `CSEProxy`，而不是裸 CPU 后端。一次普通操作经历：

```text
LoopBody 中的 ops.add
    → CSEProxy.add
    → CPU backend override 生成候选表达式，例如 tmp0 + tmp1
    → CSEProxy 查询公共子表达式表
    → 新表达式得到临时名；相同表达式复用已有临时值
```

最终返回的 `CppCSEVariable` 不只是字符串；它至少记录临时名和类型，供后面的操作继续使用。下表中的 `tmp0…tmp5` 都是这种对象。

例如处理 `add(tmp0,tmp1)` 时，两层的状态依次是：

```text
CPU backend override 交出候选式：tmp0 + tmp1
CSEProxy 用候选式查询当前 kernel 的 CSE 表
  未命中：建立 CppCSEVariable("tmp2")
          发出 auto tmp2 = tmp0 + tmp1;
          记录 “tmp0 + tmp1 → tmp2”
          把 tmp2 交给下一项 relu
```

如果同一个 kernel 的同一位置随后再次请求完全相同的 `add(tmp0,tmp1)`，查询会命中 `tmp2`，直接返回它，不再发出第二条加法定义。CSE 表还要区分类型和执行条件；下一层源码篇再展开这些条件。本篇先看清“后端给候选式，`CSEProxy` 决定新建还是复用，后续 ops 接收变量对象”这条交接。

## CPU pointwise 如何从 ops 变成 `tmp0…tmp5`

输入：

```text
x=[-2,1,3]
y=[1,4,-1]
return relu(x+y)*2
```

按本例采用的标量 CPU 生成轨迹，操作顺序和返回对象是：

| ops 调用 | CPU codegen 返回 |
| --- | --- |
| `load(arg0_1, x0)` | `tmp0` |
| `load(arg1_1, x0)` | `tmp1` |
| `add(tmp0,tmp1)` | `tmp2` |
| `relu(tmp2)` | `tmp3` |
| `constant(2.0,float32)` | `tmp4` |
| `mul(tmp3,tmp4)` | `tmp5` |
| `store(buf0,x0,tmp5)` | 无返回值 |

CPU backend 负责把 relu 表成 C++ `max`、把 load/store 表成指针索引；`CSEProxy` 负责为这些候选表达式命名和查重。生成代码对应为：

```cpp
auto tmp0 = in_ptr0[x0];
auto tmp1 = in_ptr1[x0];
auto tmp2 = tmp0 + tmp1;
auto tmp3 = std::max(tmp2, 0.0f);
auto tmp4 = 2.0f;
auto tmp5 = tmp3 * tmp4;
out_ptr0[x0] = tmp5;
```

位置 0 得到 `max(-2+1,0)*2=0`；位置 1 得到 10；位置 2 得到 4。LoopBody 决定操作和值流顺序，backend override 决定目标语言表达，`CSEProxy` 管理临时值，外层 CPU kernel 决定生成 `x0=0..2` 的循环。

## 规约多出“输入位置”和“累计状态”

使用 2×128 输入计算 `(x+1).sum(dim=1)`，捕获出的核心操作为：

```python
value = ops.load("arg0_1", 128*x0 + x1)
value = ops.add(value, ops.constant(1.0, torch.float32))
acc = ops.reduction(torch.float32, torch.float32, "sum", value)
ops.store_reduction("buf0", x0, acc)
```

`x0` 选择输出行，`x1` 遍历这一行的 128 个元素。用标量 CPU handler 解释时，可以得到：

```text
load -> tmp0
constant -> tmp1
add -> tmp2
reduction(sum,tmp2) -> tmp_acc0  # 跨 x1 循环维护的累计状态
store_reduction(buf0,x0,tmp_acc0)
```

这几行是编译期生成代码的动作，不是说一次 `ops.reduction` 调用在编译期真的读取了 128 个数。后端把它们分别放到运行时规约循环的不同位置：

```text
处理第 0 行（x0=0）
  进入 x1 循环前：acc = 0
  x1=0：读取 x[0,0]=0，加 1 得 1，acc = 0+1 = 1
  x1=1：读取 x[0,1]=1，加 1 得 2，acc = 1+2 = 3
  ...
  x1=127：读取 127，加 1 得 128，acc = 8128+128 = 8256
  离开 x1 循环后：把 acc 写到 buf0[0]
```

`ops.reduction(..., "sum", value)` 让 handler 建立并更新这一累计状态；`store_reduction` 登记规约完成后的写回，不是在每个 `x1` 上写一次输出。处理第 1 行时会重新从 0 开始累计，最后写 `buf0[1]=24640`。

第 0 行是 0 到 127，各加 1 后总和为 8256；第 1 行是 128 到 255，结果为 24640。换成向量 handler 时，load、add 和累计对象会变成向量值，最后再横向合成一个标量；换成 GPU handler 时，循环和累计可以采用另一种程序结构。Q42 再解释后端怎样选择并组织这些完整循环，本篇只关心同一 `reduction` 语义能够交给不同 handler。

## 同一份信息也服务于 Scheduler 分析

Scheduler 需要用 `extract_read_writes()` 得到 MemoryDep。当前 `LoopBody` 已保存 `memory_usage` 和索引名，因此该函数有快速路径：先把新循环变量代回索引表，再把其中的 load/store 记录变成读写依赖，不必再次执行整张 FX 图。对于普通 callable 的慢路径，则安装 `RecordLoadStore` handler 并实际运行 callable。

对本篇的 pointwise 案例，两条路径最终都交出：读 `arg0_1[index0]`、读 `arg1_1[index0]`、写 `buf0[index0]`。区别只在信息来源：快速路径读取捕获时留下的三条摘要；慢路径运行普通 callable，在遇到两个 load 和一个 store 时现场登记。

两条路径交出的语义相同：哪些 buffer 被哪个索引读取或写入。这里需要准确区分“统一 handler 协议允许重放”和“当前优化后的快速路径是否真的重放整张图”。

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
t1 = constant(0, float32)
t2 = maximum(t0, t1)
t3 = maximum(t0, t1)       # 同一 kernel、同一位置、同一 dtype、同一执行条件
t4 = add(t2, t3)           # 两个 maximum 的结果都被使用
store(y, i, t4)
```

先自己填写捕获结果，再与下面核对：

```text
indexing_exprs = {index0: i}
memory_usage   = LOAD x[index0] + STORE y[index0]
root FX graph  = load → constant → 两个相同输入的 maximum → add → store
```

两个 maximum 和最后的 add 没有增加内存动作，所以摘要仍然只有一读一写。本例明确固定两次 maximum 的输入、dtype、所在 kernel 和执行条件都相同；backend 因而交出相同候选式，CSE 查询所需条件也相同。代码生成重放完整 FX 图时，CPU 路径继续向下走：

```text
load(x,index0)       → tmp0
constant(0)          → tmp1

第一次 maximum(tmp0,tmp1)
  backend 候选式：max_propagate_nan(tmp0, tmp1)
  CSE 表未命中：建立 tmp2，发出定义并登记候选式 → tmp2

第二次 maximum(tmp0,tmp1)
  backend 给出相同候选式
  CSE 表命中：直接返回 tmp2，不再发出新定义

add(tmp2,tmp2)       → tmp3
store(y,index0,tmp3)
```

因此标量 C++ 的关键部分是：

```cpp
auto tmp0 = in_ptr0[x0];
auto tmp1 = 0.0f;
auto tmp2 = max_propagate_nan(tmp0, tmp1);
auto tmp3 = tmp2 + tmp2;
out_ptr0[x0] = tmp3;
```

再看另外两位消费者：

- 如果已经有 `LoopBody`，依赖快速路径直接读取 `memory_usage`，得到 x 的一项读和 y 的一项写，不执行 maximum；
- 如果输入还是普通 callable，依赖慢路径才安装记录 handler，运行 callable 并在 load/store 时登记读写；
- 换成向量 CPU 或 Triton handler 时，FX 值流和内存摘要不变，候选目标表达式由相应 backend 决定。

这就是读懂后续任何 backend codegen 的入口：先找 LoopBody 发出的 ops，再看当前 handler 如何解释它们。

## 继续阅读实现

[LoopBody 重放源码：handler、CSE 与规约状态](01-replay-source/README.md)
