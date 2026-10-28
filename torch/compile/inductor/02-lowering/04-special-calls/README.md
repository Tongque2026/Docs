# 特殊调用：给现成实现、外部算子和控制流建立 IR

[返回 Lowering](../README.md) · [上一篇：存储与数据版本](../03-storage-layout/README.md) · [返回总览](../../README.md)

实现叶子：[候选、fallback 与控制流构造源码](01-special-source/README.md)。

上一篇处理了 `Pointwise`、`Reduction` 和 view：它们可以用“输出位置怎样读取输入并计算”
来描述。现在看三类需要保留更多信息的请求：

```text
矩阵乘等专用计算        需要从若干现成实现中选一个
没有专门 lowering 的算子 需要在编译后的程序里调用原 ATen operator
cond / while_loop       需要保存多张子图以及它们之间的接口
```

本篇从 Lowering 已经取得算子参数、并按上一篇要求准备好输入 buffer 的位置开始。三条路径
分别建立一种带名字的 IR operation。Scheduler 随后读取 operation 的输入、输出和存储关系，
安排它与前后工作的顺序。

```text
专用计算 ─────→ 选实现 ───────→ ExternKernelOut 或 TemplateBuffer
ATen fallback ─→ 保留一次调用 ─→ FallbackKernel 及其各项输出
控制流 ───────→ Lower 子图 ───→ Conditional 或 WhileLoop 及其各项输出
```

## 矩阵乘：从可用实现中选出图里的一项工作

以 `out = torch.mm(x, y)` 为例。对同一组输入，Inductor 可能有现成 ATen 库调用，也可能有
一个或多个专用模板。Lowering 先建立候选列表。每个候选都记录：

- 它读取哪些输入 buffer；
- 它承诺产生怎样的输出，包括 device、dtype、size 和 stride；
- 怎样测试这份实现的耗时；
- 选中后怎样建立真正的输出 IR。

源码把这样一份候选称为 `ChoiceCaller`。它不是已经登记进图的矩阵乘 operation，而是
“如果选我，怎样测试并建立那项 operation”的记录。

### 通常路径：Lowering 当场选定

假设候选表中有下面两项：

```text
候选 A：调用现成 ATen mm
候选 B：使用一份专用模板
```

在这条不保留 `MultiTemplateBuffer` 的路径中，只有一个候选时，选择器可以直接使用它。
存在多个候选时，选择器通常取得缓存中的
测量结果，或者实际测量后选耗时最小的一项。timings 为空或第一项候选没有 timing 时，
当前实现使用列表中的第一个候选；timings 非空但全部为非有限值时，它会报
`NoValidChoicesError`；其余情况在已有 timing 的候选中取耗时最小的一项。选中 A 时，
`A.output_node()` 建立 `ExternKernelOut`；选中 B 时，
`B.output_node()` 建立 `TemplateBuffer`。因此普通路径离开 Lowering 时，图里已经只有一个
确定的实现：

```text
x、y → 候选 A / 候选 B
           │ 选择
           ▼
      ExternKernelOut          或          TemplateBuffer
      （现成库调用）                       （专用模板工作）
```

候选列表为空会直接报 `NoValidChoicesError`。此时没有矩阵乘 operation 可以交给 Scheduler。

当前 CPU 实验对两个 4×4 float32 输入实际得到：

```text
TensorBox → StorageBox → ExternKernelOut(name=buf0)
operations = [ExternKernelOut]
```

所以这次编译的 Scheduler 接到的是一项 extern 工作。实验脚本见
[`experiments/07_special_lowering_paths.py`](../../experiments/07_special_lowering_paths.py)。

### 延迟路径：把多个候选保留到 Scheduler

默认选择入口在 `benchmark_epilogue_fusion` 打开、同时启用 `max_autotune` 或
`max_autotune_gemm` 时，才允许建立 `MultiTemplateBuffer`；CPU 输出或候选测量需要特制输入
数据时，当前实现关闭这条延迟路径。这个对象已经以一项 operation 的身份进入 Lowering
结果，同时保存多份候选和一个“需要时取得各候选耗时”的入口。此时最终实现身份尚未确定，
因为 Scheduler 还可能看到紧接在矩阵乘后的普通计算。

例如图中有 `relu(mm(x, y))`。如果一份 Triton 模板能够把 relu 放进同一个 kernel，
Scheduler 会比较带 relu 的模板与分开执行的方案：

```text
方案 1：模板 mm 与 relu 一起生成
方案 2：最快的独立 mm，再单独执行 relu
```

方案 1 更合适时，`MultiTemplateBuffer` 被固定为那份 Triton 模板，并把 relu 作为模板的
后处理计算。无法合并或者分开执行更合适时，Scheduler 取不带 relu 的最快候选：Triton
候选会把模板生成入口固定下来；extern 候选会建立真实 extern operation，并用它替换原来
的多候选 operation。之后的调度节点类型也随最终 operation 改变。

Scheduler 第一次需要这些候选耗时时，才调用保存的测量入口。没有 timing 的候选不参加
后续最小值比较；timings 非空但全部为非有限值时，测量入口抛 `NoValidChoicesError`；若
得到空 timing 表，Scheduler 也没有可供定型的最小候选，编译失败。普通路径中“timings
为空便采用第一候选”的分支位于 `MultiTemplateBuffer` 返回之后，因此不适用于这里。

当前环境没有实际跑到这条延迟模板路径；这里的行为来自当前源码及测试。名为
`template_mm` 的实验 11 案例实际也选择了 `ExternKernelOut`，所以它只能证明那次选择结果，
不能证明模板已经执行。

## fallback：把一次 ATen 调用留在编译图中

有些 ATen 算子没有专门的 Inductor 计算实现，但生成的程序能够直接调用它。Lowering 为
这次调用建立 `FallbackKernel`。例如：

```python
x = torch.tensor([0.5, 1.5, 1.8, 3.2])
out = torch.histc(x, bins=4, min=0, max=4)
```

Lowering 需要知道两类信息：运行时应把哪些参数传给 `aten.histc.default`，以及后续节点
会收到什么结构的结果。处理过程如下：

```text
原调用参数
args=(x,), kwargs={bins:4, min:0, max:4}
        │
        ├─ Tensor 参数：x
        └─ 非 Tensor 参数：4、0、4，并保存原来的嵌套位置
        │
        ▼
准备 x 的 buffer，固定它当前的 layout
        │
        ▼
用对应的 FakeTensor 调用 aten.histc.default
只取得结果的 size、stride、dtype 和容器结构
        │
        ▼
登记一次 FallbackKernel(aten.histc.default, 全部运行时参数)
        │
        ▼
为 Tensor 结果建立 MultiOutput，供后继节点引用
```

FakeTensor 携带张量的 device、dtype、size 和 stride，但没有真实元素数据。这次调用用来回答
“结果长什么样”，不会在编译时计算直方图。运行时仍由 `FallbackKernel` 保存的 operator
和参数完成真正计算。

当前案例的 IR 是：

```text
FallbackKernel(name=buf0, op=aten.histc.default)
  └─ MultiOutput(name=buf1, 路径=[])
```

`buf0` 代表一次共同调用。`buf1` 表示从这次调用的返回值中取出 Tensor 结果；读取 `buf1`
不会再次执行 histc。实际运行结果为 `[1, 2, 0, 1]`。

返回值也可以是 list、tuple 或 dict。Lowering 保存每个 Tensor 所在的路径，例如“tuple 的
第 1 项、其中 dict 的键 `values`”，并为该 Tensor 建立一个 `MultiOutput`。Python `int`、
`SymInt` 和 `None` 可以直接留在相同返回结构中；其他尚未支持的叶子类型会在这里报错。

算子的 schema 还说明输入与输出是否共享 storage，以及哪些输入会被修改。函数式算子可
直接建立上述调用。受支持的 alias 或修改关系会记进 fallback operation，并在需要时增加
`MutationOutput`；可变算子若无法自动改写成受支持的函数式接口，会抛
`NotImplementedError`，不会产生半完整的 operation。

因此 fallback 成功时，算子仍位于同一张编译图中：它的生产者、这次调用及其消费者都由
Scheduler 排序。若 fallback 构造失败，本次 Lowering 不会留下 `FallbackKernel`，而是把
错误交回调用方。

## 控制流：把子图接成父图中的一项工作

`cond` 和 `while_loop` 都携带子 GraphModule。Lowering 会为每张子图另建一份
`GraphLowering`，先把子图内部计算变成 IR，再检查子图交给父图的接口。接口成立后，父图
登记一项控制流 operation；父图的后续节点通过 `MultiOutput` 读取它的各项结果。

### cond：两支必须交出相同形状的结果槽位

```python
def true_fn(v):
    return v + 1

def false_fn(v):
    return v - 1

out = torch.cond(pred, true_fn, false_fn, [x])
```

Lowering 从 FX 节点取得一组 fake operands，并用同一组 operands 分别处理 true 和 false
子图。假设 `x` 是 shape `[2]` 的 float32 Tensor，子图出口为：

```text
true 子图： ComputedBuffer(size=[2], dtype=float32, stride=[1], offset=0)
false 子图：ComputedBuffer(size=[2], dtype=float32, stride=[1], offset=0)
```

父图逐项比较输出数量、size、stride、device、dtype 和 offset。两项接口一致后，父图登记：

```text
predicate + operands
        │
        ▼
Conditional(buf0，保存 true/false 两张已 Lower 的子图)
        │
        └─ MultiOutput(buf1，第 0 个结果槽位)
```

运行时 predicate 为真，`buf1` 取得 true 子图结果；为假则取得 false 子图结果。对
`x=[2,4]`，两条运行分别得到 `[3,5]` 和 `[1,3]`。若一支返回 shape `[2]`，另一支返回
shape `[1,2]`，父图无法给 `buf1` 确定一个共同接口，Lowering 在登记 `Conditional` 前失败。

当前实现还检查分支输出之间是否互相 alias。它能确认 true 分支内部的重复 alias 会被
拒绝；当前这项检查不能保证仅发生在 false 分支的重复 alias 也在这里被拒绝。具体判断
语句留到实现叶子逐行说明。

### while_loop：只有 carried values 进入下一轮

`while_loop` 的输入分成两组：

- `carried_inputs` 是循环状态；body 的第 i 项输出成为下一轮第 i 项 carried input；
- `additional_inputs` 是 cond/body 每轮都能读取的附加值，本轮读取后仍把原值传给下一轮。

设 carried 初值为标量 Tensor `v=0`，附加输入 `limit=3`：

```python
def cond(v, limit):
    return v < limit

def body(v, limit):
    return (v + 1,)
```

两张子图每次都接收 `[v, limit]`，但 body 只返回新的 `v`：

```text
进入循环：v=0，limit=3
第 1 次判断：0<3 → body 返回 v=1；limit 仍为 3
第 2 次判断：1<3 → body 返回 v=2；limit 仍为 3
第 3 次判断：2<3 → body 返回 v=3；limit 仍为 3
第 4 次判断：3<3 为假 → 循环结果是 v=3
```

Lowering 给 cond/body 使用相同的 `carried_inputs + additional_inputs`。cond 必须返回恰好
一个 dtype 为 bool、shape 为 `[]` 的标量 Tensor。body 的输出数量必须等于 carried input
数量，并逐项保持 size、stride、device、dtype 和 offset；这些输出之间也不能互相 alias。

检查通过后，父图登记一项 `WhileLoop`，并按 body 输出建立 `MultiOutput`。附加输入保存在
`WhileLoop` 的输入中供每轮读取，不会成为父图返回的循环状态。

循环也可能执行零轮。例如初值 `v=3`、`limit=3` 时，第一次判断就是假，结果直接是调用者
传入的那个 `v`。若这项 carried input 同时是整个编译图的输入，父图结果可能与调用者输入
使用同一块 storage。Lowering 因此把对应输出加入 `never_reuse_buffers`：后续内存规划不能
拿这项输出的 storage 覆盖别的值，否则零轮路径可能改坏调用者的输入。

本机没有独立运行 while_loop 案例；上述接口和零轮处理来自当前 `ir.py` 与控制流测试。

## 交给 Scheduler 的结果

三条路径都把原 FX 节点变成了 Scheduler 能识别的 operation，但各自保留的信息不同：

| FX 请求 | Lowering 完成的变化 | Scheduler 接到的对象 |
| --- | --- | --- |
| `mm` 等专用计算 | 测量并选定一项实现；特定配置保留多候选供后续结合后处理定型 | extern、template 或 multi-template operation |
| 允许 fallback 的 ATen 调用 | 保存 operator 与完整参数，推导返回结构，记录 alias/mutation | `FallbackKernel` 与各项 `MultiOutput` |
| `cond` | Lower 两支子图并建立共同结果槽位 | `Conditional` 与各项 `MultiOutput` |
| `while_loop` | Lower cond/body，区分下一轮会更新的状态与每轮照常传入的附加输入，固定循环接口 | `WhileLoop` 与表示最终 carried outputs 的各项 `MultiOutput` |

Scheduler 根据这些 operation 的读写关系安排顺序；其中只有 `MultiTemplateBuffer` 还需要在
Scheduler 中确定最终实现。`FallbackKernel`、`Conditional`、`WhileLoop` 和
`MultiOutput` 都属于 `ExternKernel` 这一族，Scheduler 分别把每项 operation 包成一个
`ExternKernelSchedulerNode`。`MultiOutput` 的节点只生成“从父 operation 的返回值中取第
几项”的 wrapper 语句，不分配输出，也不会再次执行父 operation。

下一篇从这些已经登记好的 operation 开始，说明 Scheduler 怎样建立依赖、排序、融合并分派
生成工作。各对象在源码中怎样一步步构造，见
[实现叶子](01-special-source/README.md)。
