# GraphLowering：逐个读取 FX 节点并建立 Inductor 的编译状态

[返回 Lowering](../README.md) · [下一篇：计算 IR](../02-compute-ir/README.md) · [返回总览](../../README.md)

实现叶子：[输入登记](01-node-entry-source/README.md)、[属性与算子分派](02-dispatch-source/README.md)、[输出与输入修改](03-output-source/README.md)、[`run_node` 外围处理](04-run-node-source/README.md)。

上一篇交给 `GraphLowering` 的是一张 FX 图和一组样例输入。FX 图中的参数仍是对其他
FX 节点的引用，例如 `add(arg0_1, arg1_1)`；Scheduler 需要的却是输入、常量、计算工作、
修改和返回值组成的 Inductor 状态。本篇解释这一步怎样完成。

`GraphLowering` 继承 `torch.fx.Interpreter`。父类解释器按照 FX 拓扑顺序取节点，
`GraphLowering.run_node(node)` 包住当前节点的处理：它先建立“正在处理哪个节点”的上下文，
再按 `node.op` 进入相应分支，最后补上布局、是否要生成独立 buffer、来源和符号约束等
全图要求。

```text
Interpreter.run() 取出下一个 FX node
        ↓
GraphLowering.run_node(node) 建立当前节点上下文
        ↓
按 node.op 选择一个分支
  ├─ placeholder：把函数参数登记为编译后函数的输入
  ├─ get_attr：取得 GraphModule 上的常量或子图
  ├─ call_function：调用当前 target 对应的 lowering
  └─ output：整理返回值，并把输入修改写回原输入
        ↓
run_node 补充本节点的布局、来源、后续读取关系和符号约束
        ↓
父类把本节点结果暂存进 env，继续下一个节点
```

因此，每个 FX 节点根据 `node.op` 进入 `placeholder`、`get_attr`、`call_function` 或
`output` 其中一个处理分支；`run_node` 包在所选分支外面，每个节点都会经过。

## 四项职责怎样接力

本模块的四项直接职责与四个实现子篇对应：

1. **登记函数输入。** `placeholder` 把样例输入的 shape、stride、dtype 和 device 变成
   `InputBuffer` 等输入描述。
2. **取得属性并处理调用。** `get_attr` 把权重或子图变成 IR 对象；`call_function`
   从 `env` 取得参数，再选择普通 lowering、容器取项、特殊处理器或 fallback。
3. **固定图的出口。** `output` 让返回 Tensor 拥有可引用的存储，记录返回顺序，并把
   图中对输入的修改写回调用者传入的原存储。
4. **维护每个节点周围的全图要求。** `run_node` 负责当前节点来源、布局要求、是否需要
   提前得到独立 buffer，以及运行时符号检查。

前两项不断产生本节点的结果；第四项在结果写进 `env` 前检查它；第三项在 output 节点
结束整张图。下面用同一张图观察这四项怎样更新状态。

## 解释过程中同时更新哪些状态

`GraphLowering` 在遍历期间同时维护下面几组记录：

| 状态 | 本例中的内容 | 后面谁会使用 |
| --- | --- | --- |
| `env` | 当前仍可能被后续 FX 节点引用的“FX node → Lowering 结果” | 父类解释器为后续节点解引用参数；最后一次使用后可删除旧项 |
| `graph_inputs` | 输入名 → `TensorBox` 或 int/float 对应的符号表达式 | Scheduler 计算依赖；wrapper 解包运行时参数 |
| `graph_inputs_original` | Tensor 输入名 → 最初的 `InputBuffer` | `output` 在输入被修改时找回调用者原存储 |
| `constants` 等属性记录 | 常量名 → 原 Tensor；子图则直接成为当前节点的 `Subgraph` 结果 | Scheduler、kernel 或 wrapper 读取常量；控制流 lowering 使用子图 |
| `buffers` | 按登记顺序保存有名字的 buffer | 逐项决定布局及后续生成；另有 `name_to_buffer` 负责按名字查找 |
| `operations` | 按登记顺序保存 Scheduler 要安排的工作 | `Scheduler(self.operations)` 的直接输入 |
| `graph_outputs` | 按返回顺序保存 IR 外层对象或常量结果 | Scheduler 保留输出依赖；wrapper 生成返回值 |
| 符号与运行时检查记录 | 新产生的无实际样例值符号及其范围检查 | 后续生成运行时检查代码 |

`buffers` 和 `operations` 各自是有顺序的列表。普通逐元素结果变成 `ComputedBuffer`
时，同一个对象既是 buffer，也是一项 operation，所以两个列表会一起增加。两张表仍有
不同用途：前者供逐项决定布局，配套的 `name_to_buffer` 才负责按名字查找；后者直接交给
Scheduler 安排执行。

view、共享存储和修改关系主要保存在 IR 对象及 operation 之间，并不存在一张能概括所有
情况的独立 alias 表。输入修改最终会形成写回 operation，之后 Scheduler 再从这些工作中
识别被修改的输入。

## 主案例：`add → relu → mul` 怎样走完整张图

当前实验 [`04_graph_lowering_nodes.py`](../../experiments/04_graph_lowering_nodes.py) 编译：

```python
def fn(x, y):
    a = x + y
    b = torch.relu(a)
    return b * 2
```

逐节点观察结果如下：

| FX node | `run_node` 返回前的结果 | buffers | operations |
| --- | --- | ---: | ---: |
| `arg0_1` placeholder | `TensorBox → StorageBox → InputBuffer` | 0 | 0 |
| `arg1_1` placeholder | `TensorBox → StorageBox → InputBuffer` | 0 | 0 |
| `aten.add.Tensor` | `TensorBox → StorageBox → Pointwise` | 0 | 0 |
| `aten.relu.default` | `TensorBox → StorageBox → Pointwise` | 0 | 0 |
| `aten.mul.Tensor` | `TensorBox → StorageBox → ComputedBuffer` | 1 | 1 |
| `output` | 返回 `None`；`graph_outputs` 含 `StorageBox → ComputedBuffer` | 1 | 1 |

两个 placeholder 先建立输入。add 和 relu 只留下“给定位置 `i` 时怎样算”的规则，因此
还没有独立 buffer。mul 是图的 Tensor 输出；它的 `run_node` 发现 output 用户提出了
具体 stride 和可引用存储的要求，于是在返回前把整条逐元素规则变成 `ComputedBuffer
buf0`。随后 output 将包着 buf0 的 `StorageBox` 放入 `graph_outputs`，没有再新增一项
operation。

这里有一个容易忽略的细节：`env` 是临时工作表，不是整张图的历史档案。父类解释器知道
每个 FX 节点还剩几次使用；某项完成最后一次使用后，可以从 `env` 删除。因而在处理 add
时能从 env 取回两个输入，在处理 relu 时能取回 add 的结果，但遍历结束时不能认为所有
旧节点仍留在 env 中。

## 第一项：placeholder 建立运行时输入说明

Tensor placeholder 读取样例 Tensor 的属性，不读取其真实元素：

```text
样例或 FakeTensor
shape=(3,), stride=(1,), dtype=float32, device=cpu
        ↓
FixedLayout(cpu, float32, size=[3], stride=[1])
        ↓
InputBuffer("arg0_1", layout)
        ↓
TensorBox(StorageBox(InputBuffer(...)))
```

外层 `TensorBox` 进入 `env` 和 `graph_inputs`，最初的 `InputBuffer` 还进入
`graph_inputs_original`。两份记录故意指向不同层次：前者可以随着 lowering 改成输入的
新值，后者始终保留“调用者传进来的那块存储”，供输出阶段写回。

动态尺寸可以用 `s0` 这样的符号表示。`int` 和 `float` 输入会变成 SymPy 表达式并进入
`graph_inputs`；`None` 和 `BackwardState` 不登记为正式图输入。当前实现虽然会在
placeholder 分支接受 Python `bool`，但若让它继续经过普通 output 收尾，会因输出阶段
只接受 `TensorBox` 或 `sympy.Expr` 类型的正式输入而报错，不能把 bool 与 int/float
概括成同一条成功路径。

## 第二项：get_attr 和 call_function 产生当前节点结果

### get_attr 从 GraphModule 取得不是函数参数的对象

FX 图中的权重、常量和控制流子图通常由 `get_attr` 节点引用。先看默认配置且该 Tensor
可以作为生成代码的输出时，当前实现会按取得的对象选择表示：

```text
GraphModule 属性
├─ 标量 Tensor              → Constant
├─ 可内联的小型一维 Tensor   → 包着 Pointwise 的 TensorBox
├─ 其他 Tensor              → 包着 ConstantBuffer 的 TensorBox
├─ 子 GraphModule           → Subgraph
└─ TorchScript 对象         → TorchBindObject，并登记到 torchbind_constants
```

这张图不是只看形状就能无条件套用。若启用了 AOTInductor 的运行时常量折叠、配置要求始终
保留 Tensor 常量，或者该 Tensor 不适合直接作为生成代码的输出，标量和小型一维 Tensor
也会作为 `ConstantBuffer` 保存。

父类解释器同样把这个结果暂存进 `env`。后续 `call_function` 如果引用该 get_attr 节点，
取得的已经是右侧 IR 对象，而不是再次读取 Python 属性。哪些 Tensor 会被内联、哪些会保留
为常量 buffer，留给[属性与算子分派实现](02-dispatch-source/README.md)展开。

### call_function 先解引用，再选择处理办法

处理 add 时，FX 参数最初只是节点引用。解释器先从 env 取出它们的 Lowering 结果：

```text
FX：add(arg0_1_node, arg1_1_node)
        ↓ 从 env 解引用
IR：add(TensorBox(InputBuffer arg0_1),
        TensorBox(InputBuffer arg1_1))
        ↓ 查 lowerings[aten.add.Tensor]
TensorBox(StorageBox(Pointwise add_rule))
```

贯穿案例随后形成可以互相调用的位置规则：

```text
add  → i ↦ load(x,i) + load(y,i)
relu → i ↦ max(add_rule(i), 0)
mul  → i ↦ relu_rule(i) * 2
```

并非所有 `call_function` 都查同一张普通 lowering 注册表：

- `operator.getitem` 的第一个参数若已是 list、tuple 或 dict，父类直接取出其中一项；
  这个动作本身不登记 operation。
- 带 `_inductor_lowering_function` 标记的 target 直接调用自己的处理器，并把它返回的 IR
  对象作为当前节点结果。
- 普通 ATen target 查 `lowerings`。没有登记时，允许的 target 或开启的隐式 fallback
  会先建立 fallback lowering；既不能 fallback 又找不到 lowering 时，本节点报出带 target
  和参数信息的错误。
- 即使 target 已有 lowering，若当前输入或输出 Tensor 不受生成后端直接处理，`run_node`
  仍可在普通分派前改走 fallback。例如 complex Tensor，或者关闭 CPU C++ codegen 后出现
  CPU 输出，都可能进入这条路；它不只是检查 dtype。这与“注册表里没有 target”是两个
  不同原因。
- 表示符号运算的 Python magic method 可以直接得到 SymPy 表达式，不必构造 Tensor IR。

fallback 通常会形成 Scheduler 需要安排的外部 operation；容器 getitem 则只是从已有结果
中选一项。这组对照说明“call_function 已处理”不等于“operations 一定增加一项”。

## 第三项：output 固定返回值并写回输入修改

output 先从 env 取得 FX 图声明的返回项，再逐项确保 Tensor 结果有可引用的存储，并尽量
让输出 stride 与 FX 元信息一致。结果按顺序写入 `graph_outputs`；`output()` 自身返回
`None`，因为后面已经没有 FX 节点需要引用 output 节点的结果。

贯穿案例的返回项不是裸 `ComputedBuffer`，而是：

```text
graph_outputs[0]
└─ StorageBox
   └─ ComputedBuffer(name=buf0, data=Pointwise(...))
```

若图修改了输入，`graph_inputs[name]` 中的当前值已经不再是最初那个同名 `InputBuffer`。
output 会把当前值写入 `graph_inputs_original[name]` 指向的原存储。例如：

```python
def mutate(x, y):
    old = x + 1
    x.copy_(y)
    return old, x
```

Lowering 把 `old` 作为普通返回值保留，同时登记把 x 的新值写回调用者原存储的工作。如果
返回列表中正好含有 x 当前值对应的 `StorageBox`，output 会把该项换成原 `InputBuffer`，
使生成的 wrapper 返回调用者认识的输入存储。用户层更复杂的重复 alias 还要由外层
AOTAutograd 调用约定恢复，不能只靠本页这一处替换解释全部情况。

每个节点的 `run_node` 已经陆续加入输出 stride、固定输入布局等要求；最后的
`finalize()` 遍历已登记的 buffer，调用各自的 `decide_layout()` 完成剩余布局选择。
到这里，FX 图的遍历结束。

## 第四项：run_node 处理跨越单个 opcode 的要求

`run_node` 每次先记录 `buffers` 和 `operations` 的起始长度，再处理 opcode，因此它知道
当前节点新增了哪些对象。它在主案例中完成三类工作：

- 把当前 FX node 记录为新 IR 的来源，方便调试和错误定位；
- 根据 output 用户的 stride 与独立存储要求，让最终 mul 在当前节点返回前变成 buf0；
- 记录结果读取了哪些 buffer，以及当前节点新产生的符号和范围检查。

另一些图还会进入不同分支。例如，某个算子要求固定输入布局时，布局处理可能用一个新 IR
对象替换原参数；若该参数在算子 schema 中可写，`run_node` 还要把新对象上的修改传播回
原参数。view 输出、`as_strided` 输入、多用户且后继要求真实 buffer、读取关系过于复杂的
Pointwise，也可能在这里提前获得独立 buffer。具体条件和每个分支的状态变化见
[`run_node` 外围处理](04-run-node-source/README.md)。

## 本层出口有两部分

解释完成后，主案例留下：

```text
graph_inputs：arg0_1、arg1_1 的输入描述
buffers：buf0
operations：生成 buf0 的一项 Pointwise 工作
graph_outputs：StorageBox(ComputedBuffer buf0)
```

`GraphLowering.codegen()` 用 `operations` 创建 Scheduler；Scheduler 还会通过当前
`GraphLowering` 读取输入、常量、buffer 和输出，建立依赖、保留返回值并识别输入修改。
与此同时，wrapper 生成会读取 `graph_inputs`、`graph_outputs`、常量和 Scheduler 得出的
修改信息，生成参数解包、返回值和输入写回。因此本层不是只交出一张 operation 列表，
而是交出“一组待调度工作 + 生成完整调用函数所需的图级状态”。

如果函数改成 `a=x+y; return a,relu(a)`，add 本身也成为可观察输出，需要自己的稳定
存储；这会改变 buffers、operations 和 graph_outputs，而 add 的数学公式没有改变。

[下一篇](../02-compute-ir/README.md)会回到本篇产生的 `Pointwise`、`Reduction` 和 `View`
内部，解释它们怎样表示“给定位置时读哪里、算什么”。这是阅读顺序上对 call_function
结果的展开；实际编译主线从本篇出口继续进入 Scheduler。
