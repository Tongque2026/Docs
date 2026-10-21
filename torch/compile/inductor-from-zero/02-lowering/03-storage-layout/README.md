# 存储与数据版本：什么时候必须有 buffer，谁和谁共享

[返回 Lowering](../README.md) · [上一篇：计算 IR](../02-compute-ir/README.md) · [返回总览](../../README.md)

计算规则回答“一个位置怎样得到值”，存储规则还要回答四个不同问题：这项结果是否需要完整保存；若保存，shape/stride 是什么；一个 view 是否共享已有 storage；写入发生时，哪些计算必须继续看到旧版本。

先固定两种排布记录的差别。输入 x 的 size=`[2,3]`、stride=`[3,1]`、offset=`0` 已不能随意更改，位置 `[1,2]` 必须读取偏移 5；源码称它为 `FixedLayout`。新建结果 buf0 时可以先只确定 size=`[2,3]`，让 stride 等后续读取方式明确后再选；源码称它为 `FlexibleLayout`。当某个已有函数要求连续输入，或结果即将生成代码时，编译器选定 stride，之后不再允许改动这项排布。

```text
Loops 计算规则
   │ realize
   ▼
ComputedBuffer + 尚可选择 stride 的 FlexibleLayout
   │ 后继读取方式提出要求，编译器选定 stride
   ▼
可索引的 FixedLayout

已有 storage ── view ──→ 同一 storage 的新 size/stride/offset

旧 storage ── 原地写入 ──→ 同一块数据中的新值
       └── 仍需旧值的计算必须先完成或保存结果
```

## TensorBox 与 StorageBox 为什么是两层可变包装

`TensorBox.create(data)` 实际返回：

```text
TensorBox(StorageBox(data))
```

外层 TensorBox 是 lowering 在 FX 环境中传递的 tensor 值；StorageBox 让同一个逻辑值的底层表示可以从延迟 Loops 切换成 ComputedBuffer，而持有这个 StorageBox 的引用不必全部更换。

例如：

```text
realize 前：StorageBox.data = Pointwise
realize 后：StorageBox.data = ComputedBuffer(name=buf0,
                                             layout=FlexibleLayout(...),
                                             data=原 Pointwise)
```

后续 loader 再访问这项结果时，就可以从 `buf0` 读取，而不是继续内联整条 Pointwise 规则。

## realize：从“怎样算”变成“必须产生一份结果”

当前 `StorageBox.realize()` 对已经是 InputBuffer、ComputedBuffer、ReinterpretView、TemplateBuffer 等具有存储/调用身份的对象直接返回名字。对 `Pointwise/Reduction/Scan/Sort`，它执行：

1. 保留原计算的 origin 和 traceback；
2. 建立 `ComputedBuffer(name=None, FlexibleLayout(device,dtype,size), data=loops)`；
3. `register_buffer` 分配 `bufN`；
4. `register_operation` 分配 `opN`；
5. 把 StorageBox.data 原地换成该 ComputedBuffer。

所以 realize 不是运行计算，也不是马上分配运行时内存。它在编译 IR 中登记：“后续生成的程序必须执行这项计算，并把结果放进名为 bufN 的数据空间。”实际分配由 wrapper/codegen 落实。

### 四类常见触发原因

| 触发 | 为什么不能只保留延迟规则 |
| --- | --- |
| graph output | 调用者需要一份具有 shape/stride/storage 合同的结果 |
| extern/template 或要求 realized input 的消费者 | 消费者需要可传递的真实 buffer 参数 |
| 多用户、读取过多或 inner_fn 过大 | 反复内联会重复计算或让表达式过深 |
| mutation 前仍有旧值 readers | 必须固定旧版本，防止它们以后读到新值 |

`realize_hint()` 比无条件 realize 更保守：它统计计算规则中不是常量、也不是简单索引值的实际数据读取；例如同时读取 `x[i]` 和 `y[i]` 得到 2。当前 Pointwise/Reduction 只有这个计数大于 1 时才立即物化。`mark_reuse(users)` 也走启发式；CPU 上 exp、sigmoid 等重操作被多次使用时更倾向物化。

## FlexibleLayout：先允许全局选择，再冻结实际索引

ComputedBuffer 刚建立时通常得到 `FlexibleLayout(device,dtype,size)`。默认 stride 是连续 stride，但“flexible”表示在正式生成索引前仍允许根据消费者和读访问顺序改变排列或加入允许的 padding。

当前 `ComputedBuffer.decide_layout()`：

1. 从 `get_read_writes()` 收集实际 MemoryDep；
2. 对与输出同规模的读取分析各维 stride hints；
3. 让 `pick_loop_order` 建议 fill order；
4. 有建议时 `freeze_layout_with_fill_order`，否则普通 `freeze_layout()`；
5. 结果成为 FixedLayout。

### 用两组真实 stride hints 推一次选择

先固定输出 shape `[2,3]`，循环变量为行 `r` 和列 `c`。假设消费者按输入的连续排列读取：

```text
读取地址 = 3r+c
r 增加 1，地址增加 3；c 增加 1，地址增加 1。
stride_hints = [3,1]
```

`pick_loop_order([[3,1]],[2,3])` 在当前源码中返回 `[1,0]`。这里的 fill order 表示先填充维 1，让列 c 成为物理上最快变化的维；`FlexibleLayout.fill_ordered` 因而得到 stride `[3,1]`：

```text
物理顺序：(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
地址：       0     1     2     3     4     5
```

若唯一读取式改成 `r+2c`，hints 为 `[1,2]`，当前函数返回 `[0,1]`，让行 r 最快变化，得到 stride `[1,2]`：

```text
物理顺序：(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)
地址：       0     1     2     3     4     5
```

多个读取可能偏好不同顺序。当前比较器逐维统计：对多少读取而言，某维的绝对 stride 更小，或另一维是广播 stride 0；票数更多的维优先。两边打平时回到连续顺序。size 为 1 的维不影响地址变化，被移到末尾。这个启发式建议只在 layout 仍为 FlexibleLayout 时有效；`require_exact_strides` 等外部合同可以更早固定布局。

因此决定链不是“看到 transpose 就选列连续”，而是：从本 operation 的实际 `MemoryDep.index` 对循环变量求 stride hints，用当前启发式得到 fill order，再把选择冻结为后续统一使用的 FixedLayout。

外部约束也可更早冻结：

- `require_exact_strides`：必须使用给定 stride；
- `require_stride_order`：保持维度快慢顺序，具体 stride 可按 size 生成；
- `freeze_layout_with_same_order`：参考另一份 stride 的排序；
- `finalize()`：GraphLowering output 收尾时让仍未决定的 buffer 执行 `decide_layout()`。

### size 相同不代表物理排列相同

shape `[2,3]` 的连续 FixedLayout stride `[3,1]`：

```text
index [r,c] → 元素位置 3r+c
```

同 shape 也可以有 stride `[1,2]`：

```text
index [r,c] → 元素位置 r+2c
```

两者都容纳六个元素，但遍历 row/column 的物理顺序不同。Scheduler 和 Codegen 需要 layout indexer 才能把逻辑 loads/stores 变成地址表达式。

## View alias：新坐标，共享旧 storage

Q22 的 `x.permute(1,0)` 实际结果为：

```text
TensorBox(
  ReinterpretView(
    StorageBox(InputBuffer x),
    FixedLayout(size=[3,2], stride=[1,3], offset=0)))
```

它没有注册 buffer 或 operation。view 的 storage 身份仍来自 x；改变的是逻辑坐标到物理位置的公式。完整位置图见上一节的 [transpose-layout.html](../02-compute-ir/transpose-layout.html)。

这种 alias 有可观察后果：写 `v[2,1]` 实际写 x 的元素位置 5，也就是 x[1,2]。若接口只要求读取，view 可一直作为 ReinterpretView；若某个消费者要求独立连续存储，就必须通过 clone/layout conversion 产生新 ComputedBuffer。

`NonOwningLayout` 也表达“输出不拥有独立 storage，而是某个 view 的别名”。它的 `get_inputs_that_alias_output()` 让后续阶段知道输出与哪个 buffer 共享。

一个当前源码中的具体用途是 concat 写入。`ExternKernelAlloc` 表示已有函数自行分配并返回一份独立输出，它的 layout 不能改成别人的切片。`ConcatKernel.realize_into(src,dst_slice)` 若发现 src 是可改 layout 的 FlexibleLayout，且不属于这种独立分配结果，就把 src 的 layout 改成 `NonOwningLayout(dst_slice)`。src 的计算直接写 concat 输出的一段 slice，不再为 src 分配一块独立最终 storage。post-grad 的某些 cat pattern 也让 extern/template 输出采用指向 concat slice 的 NonOwningLayout。

因此逻辑上的 src buffer 名仍可用于依赖和生成定位，但 `should_allocate`/alias 信息告诉后续阶段它并不拥有另一块最终存储。

## Mutation：存储身份不变，数据版本改变

独立实验 [`experiments/06_mutation_storage.py`](../../experiments/06_mutation_storage.py) 编译：

```python
def fn(x, y):
    old = x + 1
    x.copy_(y)
    return old, x
```

输入：

```text
x=[0,1,2]
y=[9,9,9]
```

正确结果要求：old 保留覆盖前的 `[1,2,3]`；调用者原来的 x storage 被写成 `[9,9,9]`；返回的 x 与调用者 x 仍 alias。

具体物理时间线见 [mutation-timeline.html](mutation-timeline.html)。图把 x 的三个原地址与独立 old buffer 并列，展示覆盖前后的两个数据版本。

### copy_ 怎样进入 mutate_to

`copy_` lowering 先让 src 转到 dst 的 device/dtype，并 expand 到相同 size，然后调用 `mutate_to(dst,src)`。

对于普通内部临时值，如果可以安全替换 StorageBox.data，fast path 会让 changed_data 直接指向已经 realize 的新数据。但输入 buffer 或 module buffer 具有外部可观察身份，不能只在编译器内部“换指针”，必须写回原 storage。

fast path 的条件很具体：changed 必须包着 StorageBox，而且该 StorageBox 不是 graph input、不是 module ConstantBuffer，也不是只维护依赖却不生成计算代码的 NopKernel。`val` 先被确保为 StorageBox 并 realize，然后 `changed_data.data = val.data`。这改变的是编译期逻辑值接下来指向哪份已实现数据；不会登记一条向外部 storage 写回的 mutation operation。

相反，input/module buffer 的地址身份会被调用者或模块继续观察。它们必须走 `MutationLayoutSHOULDREMOVE.realize_into`，让 operation 的 layout 指向原 target，后端才能真的向原 storage 写入。两条路径都让后续 FX env 看见 changed 的新值，但只有后一条承担外部可见的原地更新合同。

实验中的 x 是 InputBuffer，所以进入 `MutationLayoutSHOULDREMOVE.realize_into(src,dst)`。

### 为什么先保护旧 readers，再建立写入

`realize_into` 的关键顺序是：

```text
1. dst.realize()                         确认目标 storage 身份
2. V.graph.mark_buffer_mutated(dst.name) 物化所有仍读取旧 dst 的结果
3. 将 src 包成一条复制 Pointwise          防止无意 alias src 与 dst
4. src.realize()
5. 把该 operation 的 layout 标成 MutationLayout(dst)
```

`mark_buffer_mutated` 查 `name_to_users[dst.name]` 并对每个旧 reader 调用 `realize()`。这是正确顺序的来源：old 原本只是 `x+1` 的延迟规则；如果不先物化，copy_ 后再展开它就可能读取已经变成 9 的 x。

实验 06 实际观察：

| FX node 后 | buffers/operations | mutated_buffers |
| --- | --- | --- |
| add | `buf0/op0`，FixedLayout Pointwise | 空 |
| copy_ | 新增 `buf1/op1`，结果仍指 InputBuffer | `arg0_1` |

这里必须分清因果：old 本身就是 post-grad graph 的普通输出，所以 `run_node` 在处理 add 时已因输出 storage/stride 合同将它物化为 buf0；这张表**不能**证明 copy_ 首次迫使 add 物化。copy_ 仍会登记 mutation、写回原 x，并保持已经存在的 old buffer 先于覆盖。

为单独观察旧 reader 保护，实验 18 另取 `old=x+1; x.copy_(y); return old*2`，并在独立实验 pass 中把合法拓扑顺序从 `add→mul→copy_` 调整为 `add→copy_→mul`。这个调整不改变依赖边：mul 仍读 add，copy_ 与 mul 之间没有数据边；它只让 GraphLowering 在看到最终 reader mul 之前先处理 copy_。

实际节点状态为：

| 节点后 | add 所在表示 / operations | mutated |
| --- | --- | --- |
| add | 延迟 Pointwise；`[]` | 空 |
| copy_ | add 被旧 reader 登记保护并成为 op0；mutation 为 op1 | `arg0_1` |
| mul | 读取已保护的 add buffer，输出成为 op2 | `arg0_1` |

copy_ 处理时，`mark_buffer_mutated(arg0_1)` 遍历此前由 `register_users_of(add_result)` 登记的 reader，并调用其 `realize()`；所以 add 在 copy_ 节点期间从延迟规则变成 op0。随后 mul 虽在线性顺序上位于 copy_ 后，仍读取覆盖前已经保存的 add，数值保持正确。这个独立重排只用于暴露当前机制，不声称默认 post-grad 总把 reader 放在 mutation 后。

### MutationLayout 表达“写到谁”，而非另分配 buf1

复制 operation 在 IR 注册时有自己的 buf/op 名字，便于依赖和调度；其 layout 指向 mutation target。`MutableBox` 是拥有可替换 `data` 字段的外层盒子，StorageBox 就继承它。`MutationLayout.get_buffer()` 逐层取出盒子中的 data，并沿 view 的来源继续走，直到找到最终目标 InputBuffer；真实 stride/storage size 都委托给这个目标的 layout。

因此 buf1 不是运行时另一块最终存储。生成 wrapper 调用：

```python
cpp_fused_add_0(arg0_1, arg1_1, buf0, arg0_1)
```

最后一个参数就是原输入 x，kernel 把 y 写进它。

## 同一个 kernel 仍能保留先后关系

实验生成一个 C++ function，但里面有两个有序循环：

```text
循环 1：读 in_ptr0（旧 x）→ +1 → 写 out_ptr0（old/buf0）
循环 2：读 in_ptr1（y）   → 写 out_ptr1（原 x storage）
```

“融合成一个 kernel/function”不等于任意交错语句。对实验 06，Scheduler 的 mutation dependency 保证覆盖必须排在 old 输出生产之后；CPU codegen 在同一函数中保留两个循环边界。对上面的重排实验，旧 reader 已在 Lowering 的 copy_ 节点处物化，后续调度再依据读写依赖安排这些 operations。

如果错误地先执行循环 2，循环 1 会读到三个 9，old 变成 `[10,10,10]`。实际实验得到 `[1,2,3]`。

## 为什么 post-grad graph 只返回 old，用户仍拿到 x

实验的 post-grad GraphModule 是：

```text
add = arg0_1 + 1
copy_(arg0_1, arg1_1)
return (add,)
```

被修改的输入通过 side effect 交付，不必作为 Inductor 单图的一项普通返回值重复返回。外层 AOTAutograd/调用包装知道原用户函数返回了 x，因而用调用者持有的已更新输入重建用户返回结构。

运行证据：

```text
old=[1,2,3]
caller_x_after=[9,9,9]
returned_x=[9,9,9]
returned_x.data_ptr() == caller_x.data_ptr()
```

这正是数值相同之外的 alias 合同。

## 把 realize、layout、alias、mutation 放回同一判断框架

遇到一个新中间值，依次问：

1. 它只是可内联规则，还是已有消费者/输出要求真实 buffer？
2. 如果物化，layout 仍可选择，还是被输出、extern 或 view stride 固定？
3. 它是拥有 storage 的 buffer，还是指向另一 storage 的 view/non-owning output？
4. 如果有人写入该 storage，哪些 reader 必须继续看到旧版本？
5. 写入是内部逻辑值替换，还是必须保持外部输入/module buffer 的身份？

回答完这五项，Scheduler 才能看到正确的 operations、读写依赖、alias 和 mutation target。

下一项 Q24 将覆盖普通 loops 之外的出口：template/algorithm selection、ExternKernel/fallback，以及 cond/while_loop 子图如何作为特殊调用交给后续阶段。

## 继续阅读实现

[物化与布局决定源码：完整方法与具体索引状态](01-storage-source/README.md)
