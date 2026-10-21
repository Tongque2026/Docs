# Lowering：把 FX 的算子请求变成 Inductor 的执行描述

[上一篇：图优化](../01-graph-optimization/README.md) · [返回总览](../README.md) · 下一篇：[Scheduler](../03-scheduler/README.md)

Lowering 不是把高级算子简单换成低级算子名字。先看 `a=x+1; b=a*2`：编译器可以暂时只保存“位置 i 的 a 等于 x[i]+1”，处理 b 时继续接上乘 2，不必真的建立完整的 a。若后面改成 `mm(a,w)`，矩阵乘函数必须收到一份可以传入的张量数据，编译器才为 a 登记一个有名字的结果及其产生方法。Lowering 要做的，就是为每个 FX 节点决定采用哪一种表示：可继续组合的位置规则、需要单独保存的数据、共享原数据的视图，或已有函数调用。

## GraphLowering 怎样连接一张图

图里 `b=relu(a)` 只保存了“b 的输入引用 a”。Lowering 必须在读到 b 时找到先前为 a 建立的计算方法。`GraphLowering` 用一个环境保存“FX node → 当前结果描述”，按拓扑顺序逐节点完成这次连接；它不会在编译时算出 eager Tensor 数值。

```text
读到输入 x      → 环境记下：x 的 shape、dtype，以及怎样读取 x
读到 a = x + 1 → 环境记下：给定位置 i，a[i] = x[i] + 1
读到函数返回 a  → 登记一项产生有名结果的工作，并把它列入返回值
```

因此，处理后一个节点时可以从环境取回前一个节点留下的内容。输入、普通计算、已有函数调用和返回值留下的内容不完全相同，但都要让后继节点知道“我现在引用的是什么”。下节把这项“逐节点读图并维护结果表”的职责编号为 Q21。

## 四项直接职责

先把四项放在同一张图里。读图职责反复处理 FX nodes；另外三项不是依次只运行一次，而是被当前节点按需调用，并把结果重新交回环境：

```text
合法 FX GraphModule + 输入属性
             │
             v
Q21 逐节点读图：从结果表取输入，处理当前节点，再把结果放回表
       │             │                 │
       │             │                 └──────────────┐
       v             v                                v
Q22 写出位置计算    Q23 决定共享原数据还是保存结果  Q24 接入已有函数和控制流
       │             │                                │
       └─────────────┴────────────────────────────────┘
                             │
                             v
Q21 把结果放回表中，继续下一个 FX node
                             │
                             v
             整理待执行工作、数据记录和返回值
                             │
                             v
                         Scheduler
```

例如 add 先经过 Q22 得到位置规则，再由 Q21 记成 a；mm 处理器属于 Q24，但它需要 Q23 先把参数变成可传入的真实张量；copy_ 同时需要 Q23 安排覆盖前后的数据版本，并由 Q21 记录输入已被修改。四项协作的共同点是：每个 FX 结果都必须变成后继节点能引用、Scheduler 能消费的 IR 或图级状态。

### 1. 读图并连接结果

这里的“解释器”就是按 FX 节点顺序重复三个动作的 `GraphLowering`。FX 节点的 `target` 说明当前要调用谁：例如 `op="call_function", target=aten.add.Tensor, args=(x,y)` 表示用 x、y 调用 ATen 加法。解释器从环境取出 args 指向的前驱结果，调用 target 对应的处理器，再把结果放回环境。对于 `a=x+y; b=relu(a)*2`，关系可以写成：

| 处理后 | FX 名字对应的内容 | 消费者 |
| --- | --- | --- |
| x、y | 两份输入的 shape、stride、dtype、device 与读取能力 | add |
| add | 给定位置读取 x、y 并相加的规则 | relu |
| relu | 在 add 规则后应用 max(value,0) | mul |
| mul | 再乘 2 的最终位置规则 | output |

### 2. 表达计算和索引

先用求和例给字段赋值：输出有两行，所以 `ranges=[2]`，合法输出坐标 row 是 0、1；每行组合三个输入，所以 `reduction_ranges=[3]`，组内坐标 column 是 0、1、2；`inner_fn(row,column)` 返回当前输入值，`reduction_type="sum"` 指定把三项相加。当前 IR 把普通逐位置规则称为 `Pointwise`；把同时含输出坐标和组内坐标的规则称为 `Reduction`。

例如 `x` 是 2×3，`x.sum(dim=1)` 的教学表达是：

```text
输出范围 row = 0..1
规约范围 column = 0..2
inner_fn(row,column) = load x[row,column]
reduction_type = sum
```

输出 `row=1` 时依次组合 3、4、5，得到 12。它与 pointwise 的差别不是“函数名叫 sum”，而是一个输出位置还拥有独立的组内位置范围。

这个 2×3 例子只用于把两个坐标算清楚。当前实现会把某些很小的静态规约展开成普通位置计算，因此不能用它证明实际保留了 `Reduction` IR。先定义下面对象链中的三层包装：TensorBox 是 Lowering 交给后续节点的 tensor 值入口；StorageBox 让同一 tensor 值的底层表示可以从延迟计算换成有名 buffer，而持有者不必换引用；ComputedBuffer 则表示“这条计算规则已经登记成一份需要生成代码并拥有布局的结果”。独立实验另用 `x=torch.arange(256.).reshape(2,128)`，实际观察到：

```text
aten.sum.dim_IntList
  → TensorBox
  → StorageBox
  → ComputedBuffer(size=[2])
  → Reduction(ranges=[2], reduction_ranges=[128], type=sum)
```

对输出 `row=1`，输入范围是原张量的 128 到 255，数学和为 `(128+255)×128/2=24512`。这里 `[2]` 决定有两项输出，`[128]` 决定每项组合多少个输入；`sum` 决定起始状态和组合方式。三份信息共同构成 Scheduler/Codegen 后续需要的规约工作。小例子的坐标关系与真实 IR 例相同，尺寸不同是为了避免小静态展开消除目标机制。

把 `row=1` 真正代入这份描述：规约索引 `column` 从 0 走到 127，读取位置是 `128+column`。累计状态从 0 开始，前三步为 `0+128=128`、`128+129=257`、`257+130=387`；最后一步把 255 加入此前的 24257，得到 24512。换成 `row=0` 时，读取位置改为 `column`，范围和组合规则不变，结果为 8128。由此可以分别判断三种变化：改行号只改变读取位置；把轴 1 改成轴 0 会同时改变输出范围和规约范围；把 sum 改成 max 会改变起始状态与组合操作。

视图则不建立累计状态。对 2×3 连续输入，原 stride 是 `[3,1]`；permute 后的 size 为 `[3,2]`、stride 为 `[1,3]`。新位置 `[2,1]` 读取的元素偏移是 `2×1+1×3=5`，所以仍读原输入 `[1,2]`。这份坐标变换直接引用输入 storage，因此实验中 buffers 和 operations 都保持 0；若后继只接受连续输入，Q23 才需要把这份视图物化成真实数据。

### 3. 安排存储、布局、别名与修改

对 `a=x+1; b=a*2`，编译器起初可让 a 指向一条规则：“给定位置 i，读取 x[i] 再加 1”。处理 b 时把乘 2 接在这条规则后，于是同一循环可直接算 `(x[i]+1)*2`。

若后面改成 `z=torch.mm(a,w)`，已有矩阵乘实现需要一份能作为参数传入的数据，不能接收一条 Python 位置规则。Lowering 此时给 a 登记有名字的结果 buffer，并保留产生它的计算：

```text
a 的结果引用
  原来指向：给定 i 计算 x[i]+1
  物化以后：读取 buf0[i]；op0 负责按原规则产生 buf0
```

`StorageBox` 保存这份可改变的底层表示，使已经引用 a 的对象继续找到更新后的表示；`TensorBox` 是图环境中传递 tensor 值的外层对象。当前源码把“延迟规则变成有名结果”称为 `realize()`，由 `ComputedBuffer` 连接 buffer 身份、布局与产生它的计算。视图可以继续共享来源；mutation 还要区分覆盖前后的数据版本。

独立实验把普通消费者和 mm 消费者分开观察，`operations` 的变化如下：

| 请求处理到哪里 | a 当前是什么 | operations | 为什么此时变化 |
| --- | --- | --- | --- |
| `(x+1)*2` 读完 add | 未物化的 Pointwise | `[]` | mul 可以继续组合位置规则 |
| 同一请求读完 mul | `ComputedBuffer buf0` 中的最终 Pointwise | `[op0]` | mul 是图输出，需要稳定的返回结果 |
| `(x+1)@weight` 读完 add | 未物化的 Pointwise | `[]` | 尚未遇到需要真实张量的消费者 |
| 同一请求读完 mm | a 的生产 operation 加 mm operation | `[op0,op1]` | mm 必须接收真实张量，因此 a 被物化；mm 自身保留外部工作身份 |

因此“FX add 已经处理完”并不等于“a 已经分配 buffer”。决定发生在消费者提出要求时；若消费者仍能组合规则，就可以继续延迟。

这解释了贯穿案例为什么没有 add 和 relu 的独立完整数组：最终 C++ 循环用局部变量 `tmp2`、`tmp3` 接住中间值，只分配最终 `buf0`。

### 4. 接入普通循环以外的实现

不是每个 FX target 都适合展开成 `Pointwise`。以 mm 为例，Lowering 可以比较“调用现成 ATen mm”和“把输入、输出 layout 及分块参数填入专用 kernel 模板”等候选，选中后留下已有调用记录或模板记录。histc 没有原生位置规则时，可保留“运行时调用 aten.histc”的 fallback 记录。`cond`、`while_loop` 则分别编译子图，并在父图保留运行时选择或循环。

这些结果仍须向 Scheduler 说明输入、输出、device、布局和依赖，但内部计算不一定由同一套逐元素 ops 生成。

三类实际请求经过 Q21、Q23、Q24 后留下的对象不同：

| FX 请求 | Q23/Q24 做的关键工作 | 当前 CPU 实验的根图 operations | 后继怎样取得结果 |
| --- | --- | --- | --- |
| `torch.mm(a,b)` | 两个输入先满足可传入布局；Q24 建立候选并在本次配置选中 ATen extern | `[ExternKernelOut buf0]` | buf0 本身就是 mm 的张量结果 |
| `torch.histc(x,...)` | 输入先实现为可调用参数；Q24 保存 operator 与常数参数 | `[FallbackKernel buf0, MultiOutput buf1]` | buf0 表示一次 packed 调用，buf1 从其返回结构取出 Tensor |
| `torch.cond(pred,t,f,[x])` | predicate/operands 物化；两支各自 Lower，再统一接口 | `[Conditional buf0, MultiOutput buf1]` | getitem 读取 buf1，不再新增 operation |

mm 的 `ExternKernelOut` 是本次 CPU 配置的实际选择，不是 mm 在所有设备上的固定表示。当前 lowering 可以比较多类实现：ATen 是现成 PyTorch operator；Triton 是为 GPU 生成的 kernel；CUTLASS 是 NVIDIA GPU 矩阵乘模板库；CK 是 AMD GPU 的 Composable Kernel 模板库；C++ packed 是 CPU 上把权重预先整理后调用的 C++ 实现。被选中的候选最终建立相应 template 或 extern IR；没有有效候选时，只有配置允许才退回 ATen，否则明确抛出 `NoValidChoicesError`。

## 四条路径怎样在 Lowering 内完整交接

把本层直接职责放回具体请求，可以看见它们不是四份互不相干的清单。

| 请求 | 读图取得什么 | 处理职责交回什么 | output 后的本层出口 |
| --- | --- | --- | --- |
| `return (x+y)*2` | x、y 的输入说明；add 的结果引用 | 组合后的 Pointwise 位置规则 | 一项最终 ComputedBuffer operation；add 不必单独物化 |
| `return x.sum(1)` | x 的 size/stride 与轴 1 | 输出范围 `[2]`、组内范围 `[128]`、sum 组合规则 | 实验实际保留 Reduction operation |
| `return x.permute(1,0)` | x 的 storage 与原布局 | size `[3,2]`、stride `[1,3]` 的 ReinterpretView | 共享输入 storage 的 graph output；0 个新 operation |
| `return relu(x@w)` | x、w 及 mm/relu 的结果引用 | mm 的 Extern/候选实现结果，relu 的位置规则 | 至少一个外部调用边界，后续 pointwise 可继续调度 |

mutation 再增加数据版本。对 `old=x+1; x.copy_(y); return old,x`，AOT 交给本层的 post-grad 图返回 old，并用 `copy_(x,y)` 表达对输入 x 的副作用。当前实验中的状态是：

| 处理阶段 | operations | mutated_buffers | 本层保存的含义 |
| --- | --- | --- | --- |
| add 后 | `[op0]` | `[]` | old 是图的显式输出，因此已经成为 `ComputedBuffer buf0` |
| copy_ 后 | `[op0,op1]` | `[arg0_1]` | op1 的 mutation layout 指向原输入 x；x 是被覆盖的真实 storage |
| output 后 | 仍为 `[op0,op1]` | 仍含 `arg0_1` | graph_outputs 只含 old；输入 x 的更新由 mutation 状态而非第二个普通返回 buffer 表达 |

这里不能把 old 的物化单独归因于 copy_：它本来就是 post-grad 图的输出。源码中的通用保护发生在 `mark_buffer_mutated(x_name)`：它查找此前登记的所有旧 x reader，并对这些 reader 调用 `realize()`，然后 mutation operation 才指向 x 的 storage。这样，即使某个旧 reader 不是直接图输出，它也不会在覆盖之后重新展开并误读新 x。

运行时，buf0 得到 `[1,2,3]`，copy operation 把调用者 x 改为 `[9,9,9]`；用户看到的第二项返回值仍与调用者 x 使用同一地址。当前生成的 Inductor `call` 直接返回 `(buf0,)`，原输入的可见更新由写回 x 的副作用实现；外层编译代码再按用户函数原来的返回形式补回第二项 x。Scheduler 接下来从 op0、op1 和 mutation target 建立“旧读完成后才能覆盖”的依赖。

特殊路径也必须从 FX 引用走到同一个本层出口，而不是停在“有 fallback/cond”这个名称上：

- `torch.histc([0.5,1.5,1.8,3.2], bins=4, min=0, max=4)` 没有变成 Pointwise。Q21 从 FX 节点取得输入和常数参数，Q24 保留 `aten.histc.default` 的 operator 身份、参数结构和 packed 返回，建立 `FallbackKernel buf0`；该 Tensor 输出再由 `MultiOutput buf1` 引用。Q21 把 buf1 交给 output，Scheduler 因而看见一次有输入、有输出的外部工作。结果为 `[1,2,0,1]`。
- `torch.cond(pred, true_fn, false_fn, [x])` 中，Q21 取得 predicate、两个子 GraphModule 和 operands；Q24 分别建立子 GraphLowering，true 分支的 `x+1` 与 false 分支的 `x-1` 各自形成子图出口，并检查 size/stride/device/dtype/offset 能否统一。父图登记 `Conditional buf0`，用 `MultiOutput buf1` 代表统一结果，再由 Q21 连接父图后继或 output。输入 x=`[2,4]` 时，pred 真得到 `[3,5]`，假得到 `[1,3]`；Lowering 编译两支接口，不等于运行时同时执行两支。

若 cond 两支的输出数量、size、stride、device、dtype 或 offset 任一不一致，Q24 就无法为父图建立一套统一的 MultiOutput，路径在 Lowering 明确失败；分支输出存在不支持的共享存储关系也会拒绝。Scheduler 不会等到运行时 predicate 已知后再猜接口。若 fallback schema 表明算子会修改输入，而且不能自动改写成函数式形式，`FallbackKernel` 会抛出 `NotImplementedError`，而不是把它当成没有副作用的外部函数。

这三条路径的共同出口仍是“有名字的 operation、输入引用、输出引用和明确的接口要求”。差别在于普通 mm 保留算法选择结果，fallback 保留任意已注册 operator 的调用结构，cond 还保留两张子图及统一的多输出形式。

逐元素和规约交出计算规则，视图交出共享与坐标关系，已有调用交出调用身份，mutation 交出数据版本顺序；`GraphLowering` 的环境把每种结果继续接到后继节点和 output。这才是四项职责在一张图里协作的方式。

## 综合案例：六个 FX 节点为什么只交出两个 operation

沿用根篇已实测的 `permute→add→relu→sum→unsqueeze→mm`，并同时返回 sum 结果与 mm 结果。Q21 的 env 每处理一步都保存当前引用，但 `operations` 的变化不是“一节点加一项”：

| 处理后 | env 中当前结果 | operations | 原因 |
| --- | --- | --- | --- |
| permute | 引用输入 storage 的 `ReinterpretView [3,128]` | `[]` | Q22 只改坐标，Q23 保留共享 |
| add | `Pointwise [3,128]` | `[]` | 仍是可组合的位置规则 |
| relu | 包含 add 的 `Pointwise [3,128]` | `[]` | 继续延迟，不需独立数组 |
| sum | `ComputedBuffer buf0 [3]` | `[buf0]` | Q22 建立规约；作为 mm 参数和最终输出，Q23 给它稳定身份 |
| unsqueeze | buf0 上的 `ReinterpretView [1,3]` | `[buf0]` | 只改变 mm 看见的逻辑形状 |
| mm | `ExternKernelOut buf1 [1,2]` | `[buf0,buf1]` | Q24 保留已有调用及其输出 |
| output | graph outputs 为 buf0、buf1 | `[buf0,buf1]` | Q21 整理双输出，交 Scheduler |

这个表把四项职责接在同一份状态上：Q21 负责结果引用不断档，Q22 提供 view/pointwise/reduction 的计算内容，Q23 根据后继的实际需要决定共享还是物化，Q24 建立 mm 外部工作。若删去“返回 rows”但保留 mm，buf0 仍然需要，因为 mm 必须取得真实输入；若删去 mm 但仍返回 rows，buf0 仍因图输出身份而需要；两种消费者都删去后，不能沿用这个理由断言仍有同样的 operation，必须重新看新的后继和 output。

## 算子处理器怎样接回读图过程

`register_lowering(aten_fn, ...)` 最终把 target 与一个“算子处理包装函数”放进 `lowerings` 字典。这个注册 wrapper 只在编译期先广播输入、做类型提升，再调用真正的 IR 构造函数；它不是前文那个在运行时检查输入、分配内存并调用 kernel 的模块 wrapper。

例如当前 `aten.mul` lowering 会处理 bool 特例，否则用 `make_pointwise` 构造乘法规则；`aten.sum` 选择 `make_reduction("sum")`。注册表解决“这个 target 交给谁”，但图环境、布局、物化、mutation 和输出处理由 `GraphLowering` 与 IR 对象共同完成。

## 本层出口

Lowering 完成后，关键产物不是一张新的 FX 图，而是：

- `operations`：已物化、需要调度的 IR 操作；
- buffers / graph inputs / constants：计算读写的存储身份；
- graph outputs：返回值及其布局、alias 关系；
- 图级状态：mutation、device、symbolic sizes、外部调用等信息。

下一篇的 Scheduler 不再关心 `aten.add` 这个名字本身；它主要读取这些操作的循环范围、读写集合、布局和 device，决定执行与融合。若把上表的 `permute` 后接一个只接受连续输入的 extern 调用，Lowering 必须在“安排存储/布局”这一步准备好连续数据，不能把问题留给 Scheduler 猜测。

继续深入：

1. [GraphLowering 怎样逐节点解释 FX 图](01-graph-interpreter/README.md)
2. [Pointwise、Reduction 与 View 怎样表达索引计算](02-compute-ir/README.md)
3. [realize、layout、alias 与 mutation 怎样决定存储](03-storage-layout/README.md)
4. [模板、外部调用、fallback 与控制流](04-special-calls/README.md)
