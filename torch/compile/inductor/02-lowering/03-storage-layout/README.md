# 存储与数据版本：计算规则何时成为 buffer，结果写到哪里

[返回 Lowering](../README.md) · [上一篇：计算 IR](../02-compute-ir/README.md) · [下一篇：特殊调用](../04-special-calls/README.md) · [返回总览](../../README.md)

实现叶子：[物化、布局固定与 mutation 源码](01-storage-source/README.md)。

上一篇产生的 `Pointwise`、`Reduction` 和 `View` 已经能回答“某个逻辑位置怎样得到值”。
编译后程序还需要确定四件事：

1. 这条计算规则是否需要一项有名字的结果；
2. 结果的逻辑位置怎样映射到物理位置；
3. 新名字是拥有一块存储，还是借用已有存储；
4. 写入发生时，哪些计算必须继续看到旧内容，谁必须看到新内容。

本篇分别把这四项职责称为物化、layout、view/alias 和 mutation。它们共同把计算 IR
变成 Scheduler 能安排的 operation、读写关系和存储身份；运行时真正分配或复用内存仍由
后续 wrapper/codegen 完成。

```text
延迟计算规则
   │
   ├─ 仍可安全组合进后继 loader ───────────────→ 保持规则
   │
   ├─ 需要独立结果 ─→ realize ─→ ComputedBuffer + operation
   │                                  │
   │                                  └─ 先带 FlexibleLayout
   │                                         ↓ 消费者约束或图收尾
   │                                     FixedLayout
   │
   ├─ 只换读取坐标 ─────────────────────────→ ReinterpretView，共享旧 storage
   ├─ 新计算直接写已有区域 ─────────────────→ NonOwningLayout，借用目标 storage
   └─ 修改已有值
       ├─ 内部临时值 ───────────────────────→ StorageBox 改指向新 data
       └─ 输入或 module buffer ─────────────→ 写回原 storage 的 mutation operation
```

## 第一项：realize 给计算结果建立名字和 operation

先看同一个 `StorageBox` 怎样变化。假设 `a=x+1` 尚未物化：

```text
TensorBox
└─ StorageBox A
   └─ Pointwise(i ↦ x[i]+1)
```

调用 `A.realize()` 后，`StorageBox A` 本身仍是同一个对象，只是它内部的 `data` 被替换：

```text
TensorBox
└─ 同一个 StorageBox A
   └─ ComputedBuffer(name=buf0, layout=FlexibleLayout, data=Pointwise(...))
```

`buf0` 同时登记到 `buffers` 和 `operations`。已有 TensorBox 或后继 loader 仍引用
StorageBox A，所以它们会自然改成读取 buf0。这个编译动作没有执行 `x+1`，也没有在运行时
申请内存；它只规定生成的程序必须产生一项名为 buf0 的结果。

### 后续使用方式决定哪条规则先成为 buffer

以下案例都从 `a=x+1` 开始，但提出独立结果要求的人不同。

| 后续请求 | 谁作决定 | 哪条规则成为 buffer |
| --- | --- | --- |
| `return (a*2)` | 最终 mul 是图输出，需要可返回的存储 | a 继续内联；最终 mul 成为 `ComputedBuffer` |
| `return mm(a,w)` | mm lowering 要求可传递的真实 Tensor 参数 | a 先成为 `buf0/op0`，mm 再登记 extern operation |
| a 有多个用户，重复计算 a 已经较贵 | `mark_reuse(user_count)` 检查读取数、表达式大小及部分 CPU 重算成本 | 命中启发式后，a 直接 realize |
| a 有多个用户，其中一个属于通常需要真实输入的算子 | `run_node` 对 a 调用 `realize_hint()` | 只有 a 的规则包含不止一次非平凡读取时才立即 realize；否则 a 暂时保持规则，具体消费者仍可在 lowering 中强制物化 |
| a 的读取累计过多，或 Pointwise 表达式过大 | `run_node` 的复杂度检查 | 先尝试 `realize_hint()`；过大的 Pointwise 仍未物化时还有直接 realize 的兜底 |
| 输入即将被覆盖，但尚未执行的旧值规则仍读取它 | mutation 的旧读保护 | 覆盖前先把这些旧值规则各自物化，再登记写入 |

当前实验的普通链中，add 后仍是 Pointwise，直到最终 mul 才登记 op0；mm 案例则在进入 mm
时先登记 add 的 op0，mm 是 op1。`realize()` 对仍是计算规则的 StorageBox 是明确物化动作；
`realize_hint()` 只在当前规则包含不止一次非平凡读取时执行该动作。当前源码把 index
不是 Python `int` 或 SymPy 整数常量的读取计为“非平凡读取”；这通常是随循环位置变化的
索引，符号表示的运行时固定索引也会计入。重复得到同一分析临时值的读取只计一次。
reuse/read-complexity
的具体阈值和遍历细节属于实现叶子，父篇只保留它的因果：重复内联可能改变成本，所以
“能继续组合”不等于“一定继续组合”。

普通 `Reduction` 还有上一篇已经说明的主动路径：若 reduction 构造结果仍是
`Reduction`，对应 lowering 会立即 realize；zero/one/small 退化成 Pointwise 时不在该处
主动登记。

## 第二项：layout 决定逻辑位置对应哪个物理位置

layout 记录 device、dtype、size、stride 和 offset。realize 新建的 `ComputedBuffer` 先带
`FlexibleLayout`：size 已定，stride 仍可根据消费者的读取方式选择。显式 stride 要求或
图收尾时的 `finalize()` 才调用布局决定逻辑，把它变成 `FixedLayout`。

一般元素位置公式是：

```text
元素位置 = offset + Σ(index[d] × stride[d])
```

offset 和 stride 的单位都是元素；换成 byte 地址时还要整体乘以该 dtype 的元素字节数。
例如 stride `[3,1]`、offset=4 时，逻辑 `[1,2]` 位于元素位置 `4+1×3+2×1=9`。

下面用 size `[2,3]`、offset=0 比较两种 layout。先给六个逻辑位置按行连续编号：
`[0,0]→0`、`[0,1]→1`、
`[0,2]→2`、`[1,0]→3`、`[1,1]→4`、`[1,2]→5`。

| FixedLayout stride | 逻辑 `[row,column]` 的元素位置 | 按物理地址从小到大看到的逻辑编号 |
| --- | --- | --- |
| `[3,1]` | `3×row+column` | `[0,1,2,3,4,5]` |
| `[1,2]` | `row+2×column` | `[0,3,1,4,2,5]` |

### 没有外部硬性要求时，读取方式提供选择线索

`ComputedBuffer` 可以从自己的读取关系中取得 stride 线索。当前实验直接构造两个同为
size `[2,3]` 的 Pointwise：

```text
row_read([r,c])    = load(x, 3*r+c)
column_read([r,c]) = load(x, r+2*c)
```

第一项沿 c 连续读取，得到 fill order `[1,0]`，最后选 stride `[3,1]`；第二项沿 r 连续
读取，得到 fill order `[0,1]`，最后选 stride `[1,2]`。没有可用读取线索的常量结果使用
默认连续 stride `[3,1]`。

这里的 fill order 表示循环维从内到外怎样排列；它是选择 layout 的线索，不是运行时另外
搬运一次数据。实现怎样从 `MemoryDep` 提取 stride hint、怎样在多份线索中排序，留给源码
叶子。

### 消费者给出的明确要求优先

若外部调用、图输出或 `as_strided` 等消费者要求确定的 stride/order，结果必须先满足该
要求：

- 仍是 `FlexibleLayout` 时，可以在原 buffer 上直接冻结成要求的 stride；
- 已是匹配的 `FixedLayout` 时，继续使用原对象；
- 已固定但不匹配时，不能把既有物理解释悄悄改掉，需要产生满足新要求的转换结果。

实验中，size `[2,3]` 的 flexible buffer 接受列连续要求后原对象变成 stride `[1,2]`；
已经固定为 `[3,1]` 的 buffer 面对同一要求时，返回了另一个 stride `[1,2]` 的结果。
因此“先 realize”和“layout 已冻结”是两个不同时间点；一旦固定，后来的启发式不能任意
改写它。

## 第三项：两种共享 storage 的关系用途不同

一个逻辑名字不必拥有一块独立 storage。本层需要区分“用新坐标读取旧数据”和“把新计算
写进已有区域”。

### ReinterpretView：新坐标读取同一份已有数据

连续输入 x 的 size/stride 是 `[2,3]` / `[3,1]`。`x.permute(1,0)` 建立：

```text
ReinterpretView(size=[3,2], stride=[1,3], offset=0)
└─ 原 InputBuffer x
```

`view[2,1]` 的元素位置是 `2×1+1×3=5`，也就是 `x[1,2]`。view 没有自己的 producer
operation，也没有复制数据；它的 loader 用新 layout 计算地址，再读取 x 的 storage。

### NonOwningLayout：producer 直接写进目标的一段区域

考虑把两项结果拼成一个 concat 输出。系统可以先建立完整目标 storage 以及分别指向两段
区域的 slice view，再让每个可重定向的 producer 直接写自己的 slice：

```text
concat destination storage D
├─ slice0 view ─← producer0 的 NonOwningLayout
└─ slice1 view ─← producer1 的 NonOwningLayout
```

`NonOwningLayout(slice0)` 保存目标 view，表示 producer0 的逻辑 buffer 没有自己的最终
分配；它生成的 store 地址来自 slice0。这个优化只在 producer 仍可改变 layout、且不是
必须自行分配输出的 extern kernel 时直接成立；否则先建立一项 copy Pointwise，再把 copy
写进目标区域。

二者都涉及共享 storage，但方向不同：ReinterpretView 是“已有数据换一种坐标去读”；
NonOwningLayout 是“新计算把结果写到别人拥有的区域”。后续 Scheduler 通过 layout 报告的
alias 关系建立依赖，wrapper 只为真正拥有 storage 的对象安排分配。

## 第四项：mutation 区分内部换值与外部可见写回

`copy_`、`fill_` 等修改请求首先进入 `mutate_to(changed, val)`。它必须判断 changed 的
storage 身份是否能被程序外部观察。

### 内部临时值：让 StorageBox 改指向新 data

若 changed 是图内部的普通 `StorageBox`，不是函数输入、module buffer 或特殊
`NopKernel`，外部没有人要求“原来那块 storage 必须被改”。Lowering 可以：

```text
changed StorageBox ─→ old internal data
val StorageBox     ─→ new data（先 realize）

fast path 后：
changed StorageBox ─→ 同一份 new data
```

这里只改变编译 IR 中 StorageBox 的 `data` 指针；不会建立写回旧 storage 的 mutation
layout，也不会把旧名字加入 `mutated_buffers`。以后引用 changed 的节点看到新 data，
旧内部 storage 身份可以消失。

### 输入或 module buffer：必须写回原 storage

函数输入 x 和 module buffer 的 storage 身份对调用者可见，不能换成另一块 data 冒充修改。
这条慢路径建立指向原目标的 mutation operation：

```text
source = 新值 val
target = 原 x InputBuffer（或原 module buffer）
layout = MutationLayoutSHOULDREMOVE(target)  # 当前源码类名
```

建立这项 operation 前，`mark_buffer_mutated(target_name)` 先做两件事：把目标名加入
`mutated_buffers`，并让当前已登记的旧值读者物化。随后 source 被包装为写入 target 的
operation。Scheduler 从 operation 的 mutation target 建立先读后写顺序；
`mutated_buffers` 还帮助后续阶段保留输入修改，但它不是所有 mutation 的通用名单——内部
fast path 不会进入它。

## 两个案例分开证明“输出物化”和“旧读保护”

### 案例 A：old 本身是输出，add 已经先物化

```python
old = x + 1
x.copy_(y)
return old, x
```

在当前 post-grad 图中，old 是普通图输出；add 的 `run_node` 结束时已经因输出要求成为
`buf0/op0`。copy_ 随后登记写入原 x 的 operation，并把 `arg0_1` 加入
`mutated_buffers`。因此这个案例证明输入写回和返回 alias，却不能用来证明“copy_ 首次
迫使 old 物化”。

该 post-grad 图的 `graph_outputs` 只包含普通输出 old；x 的修改作为输入副作用交给外层
AOTAutograd 调用约定，外层再让用户看到返回的 x 与调用者 x 是同一 storage。实验中调用
后 x 变为 y，返回 x 与调用者 x 的 `data_ptr` 相同。

若某张 GraphLowering 图直接把被修改输入的当前 `StorageBox` 放入 `graph_outputs`，output
收尾还会把该项换成 `graph_inputs_original[name]` 指向的原 `InputBuffer`。这样 wrapper
返回的仍是调用者认识的 storage，而不是写回前临时产生的新逻辑结果。

### 案例 B：旧值 reader 被排到写入之后，copy_ 首次保存旧版本

为单独观察旧读保护，实验把一张合法图安排成：

```text
add:  old = x + 1       # 此时 old 仍是 Pointwise，operations=[]
copy: x.copy_(y)        # 写 x 之前发现 old 仍会被 mul 读取
mul:  out = old * 2
```

copy_ 到来时，`mark_buffer_mutated(x)` 找到 old 的延迟读者，先把 old 保存成 op0，再登记
写 x 的 op1；之后 mul 读取 op0 并形成 op2。于是运行时即使 x 已被覆盖，mul 仍得到覆盖前
的 `2*(x+1)`。这里才是 mutation 首次触发旧值物化的直接案例。

删除 old 的输出并不自动消除这项保护：关键条件是覆盖之后是否还有某个后继需要旧值。
若既没有输出，也没有跨写入的旧 reader，这个物化理由才消失。

## 本层交出的对象

| 情况 | 本层结果 | 谁继续使用 |
| --- | --- | --- |
| 延迟规则被物化 | 同一 StorageBox 内变成 `ComputedBuffer(FlexibleLayout, rule)`，并登记 operation | layout 决定、Scheduler |
| layout 被确定 | `FixedLayout(size,stride,offset)`；不匹配的后继要求可能产生转换结果 | kernel 地址生成、外部调用 |
| 纯 view | `ReinterpretView` 共享原 storage，无独立 operation | 后继 loader、graph output |
| 定向写已有区域 | producer 使用 `NonOwningLayout(target_view)` | Scheduler alias 依赖、wrapper 分配所有者 |
| 内部 mutation | StorageBox 改指向新 data，不写回旧 storage | 后继引用该 StorageBox 的节点 |
| 输入/module mutation | `MutationLayoutSHOULDREMOVE(target)` operation、旧 reader 保护和 mutated target | Scheduler 写入顺序、wrapper/AOT 外部状态恢复 |

下一篇[特殊调用](../04-special-calls/README.md)会使用已经物化并满足 layout 要求的输入，
建立 extern、fallback、template 和控制流 operation。它接住的是本篇确定的 buffer、layout
和共享/修改关系；运行主线随后仍由 Scheduler 统一安排这些 operations。
