# Scheduler 节点与依赖：谁读了哪个位置，谁必须先完成

[上一层：Scheduler](../README.md) · [返回总览](../../README.md) · 下一项：排序、mutation 版本与生存期

Lowering 交给 Scheduler 的每项 operation 都描述一项待执行工作。例如普通计算可能写 buf0，已有矩阵乘调用可能读取 buf0 并写 buf1。Scheduler 首先把每项工作包成统一的调度节点，并回答两个问题：这项工作会读写哪些 buffer；其中哪些读取尚未有生产者完成。普通计算、模板生成和已有函数调用对应的内部类名分别是 `ComputedBuffer`、`TemplateBuffer`、`ExternKernel`，但本篇只关心它们交出的读写记录。

本篇先从 `Scheduler.__init__` 收到的 operation 列表建立依赖图，不提前讨论怎样排序或融合。

## 一项 operation 为什么还要再包一层

IR operation 保存“怎样计算”。Scheduler 还需要附加只服务于全局安排的状态，例如未满足依赖、输出使用者、祖先、最后使用位置和是否已经生成。当前实现由 `Scheduler.create_scheduler_node()` 按 operation 类型选择包装：

| Lowering operation | Scheduler 节点 | 依赖从哪里取得 |
| --- | --- | --- |
| `ComputedBuffer`、`TemplateBuffer` | `SchedulerNode` | 重放循环体，记录其中的 load/store；模板调用自身的提取接口 |
| `ExternKernel` | `ExternKernelSchedulerNode` | operation 已保存的 `get_read_writes()` |
| no-op operation | `NopKernelSchedulerNode` | operation 已保存的 `get_read_writes()` |

所以 FX 节点数、Lowering operation 数和 Scheduler 节点数不能互换。在案例 `(x+1).sum(dim=1)*2` 中，FX 图包含多步数学运算，Scheduler 入口则只有一个 `ComputedBuffer`：逐元素加法和乘法已经进入规约的计算体。

## `ReadWrites` 是一项工作的读写清单

每个 Scheduler 节点都有一份 `ReadWrites`，主要包含 `reads` 和 `writes` 两组依赖。对可生成循环的 `ComputedBuffer`，`extract_read_writes()` 用符号循环变量重放计算体；此时 load/store 不访问真实数据，而是记录 buffer 名、索引式和变量范围。

本次规约输入为：

```python
x = torch.arange(6.0).reshape(2, 3)
result = (x + 1).sum(dim=1) * 2
```

实际记录可简化为：

```text
读 arg0_1[3*c0]、arg0_1[3*c0+1]、arg0_1[3*c0+2]，c0 ∈ [0, 2)
写 buf0[c0]，c0 ∈ [0, 2)
```

把 `c0=0` 代入，三个读取位置是 0、1、2，写输出 0；代入 `c0=1`，读取 3、4、5，写输出 1。数值过程为：

| `c0` | 读取 x | 加 1 后求和 | 乘 2 | 写入 |
| --- | --- | --- | --- | --- |
| 0 | 0、1、2 | 1+2+3=6 | 12 | `buf0[0]` |
| 1 | 3、4、5 | 4+5+6=15 | 30 | `buf0[1]` |

这种带精确索引和迭代范围的记录叫 `MemoryDep`。它不仅说明“读 x”，还说明本循环如何访问 x；后面的融合和循环顺序判断会使用这些信息。

## 三种依赖分别记录什么

### `MemoryDep`：知道访问哪个位置

`MemoryDep(name, index, var_names, size, mode)` 描述一个按索引访问的 buffer。`index` 是线性物理索引，`var_names` 与 `size` 给出符号变量范围。普通读取/写入的 mode 为默认值；原子累加等特殊写入会记录相应 mode，后端据此选择普通赋值还是原子操作，融合也不能把不兼容的访问方式当成同一次普通读写。上例的 `MemoryDep(arg0_1, 3*c0+1, {c0: 2})` 表示两次循环分别访问物理位置 1、4。

两个依赖都叫 `MemoryDep`，不代表索引相同。连续访问、广播、转置和间接索引会留下不同的表达式；Scheduler 因而可以比较循环是否访问兼容，而不是只比较 buffer 名。

### `StarDep`：只知道依赖整个 buffer

外部调用未必向 Scheduler 暴露逐元素循环。案例：

```python
buf0 = torch.mm(x, weight)
buf1 = torch.sin(buf0)
```

这个 CPU 案例得到两个入口节点：

```text
op0 ExternKernelSchedulerNode
  reads  = StarDep(arg0_1), StarDep(arg1_1)
  writes = StarDep(buf0)

op1 SchedulerNode
  reads  = MemoryDep(buf0, c0, {c0: 8})
  writes = MemoryDep(buf1, c0, {c0: 8})
```

`StarDep(buf0)` 表示对整块 `buf0` 的依赖，没有可供比较的逐位置索引。`sin` 的循环则明确读取展平位置 0 到 7。两种表示仍通过同一个名字 `buf0` 相连：`buf0` 的生产者是 op0，使用者是 op1。

### `WeakDep`：只维持 mutation 顺序

`WeakDep` 不表示一次真实内存读取；它只要求修改发生在先前读者之后。之所以是“弱”的，是因为先前读者若被删除，或者其结果被融合掉，这条人为顺序边也可随之裁剪，不必强迫一个无用 buffer 物化。

## 普通生产者—消费者边怎样建立

创建节点时，`read_writes.reads` 已经存在，`unmet_dependencies` 初始取同一组读取，再删去已经可用的图输入和常量。

先看一条完全由普通逐元素计算组成的链：

```text
op0: buf0 = x + 1
  reads  = MemoryDep(arg0_1, c0, {c0: 8})
  writes = MemoryDep(buf0,   c0, {c0: 8})

op1: buf1 = buf0 * 2
  reads  = MemoryDep(buf0, c0, {c0: 8})
  writes = MemoryDep(buf1, c0, {c0: 8})
```

图输入 `arg0_1` 已在 `available_buffer_names` 中，所以它从 op0 的 `unmet_dependencies` 删除。`buf0` 是内部结果，op1 的未满足读取仍保留：

```text
op0.unmet_dependencies = {}
op1.unmet_dependencies = {MemoryDep(buf0, c0, {c0: 8})}
users[buf0]             = {op1}
users[buf1]             = {OUTPUT}
```

`name_to_buf[buf0].defining_op` 指回 op0。于是 op1 中尚未满足的 `buf0` 读取有了明确生产者，形成 `op0 → op1`；后面的排序必须把 op0 放在 op1 前面。这里两端都是精确的 `MemoryDep`，依赖名字确定生产—消费关系，索引式则继续供融合等判断使用。

在 mm 案例中，op0 读取的两个输入一开始就在 `available_buffer_names`，所以它们不是未满足依赖。op1 读取内部结果 `buf0`，因此保留：

```text
op1.unmet_dependencies = MemoryDep(buf0, c0, {c0: 8})
users[buf0]             = op1
users[buf1]             = OUTPUT
```

`compute_dependencies()` 遍历每个节点的读取，把当前节点登记为相应 buffer 的使用者。输出还会登记一个虚拟的 `OUTPUT` 使用者，防止最终结果被死节点删除。

在这条 mm→sin 链中，同样由 `name_to_buf[buf0].defining_op` 找到 extern op0。生产端的 `StarDep` 和读取端的 `MemoryDep` 精度不同，但相同的 buffer 名仍形成 `op0 → op1`。是否把两项合成一个生成单元由后面的融合阶段处理。

## mutation 会增加两类依赖记录

看下面的旧值读取与写回：

```python
old = x + 1
x.copy_(y)
return old, x
```

输入 `x=[0,1,2]`、`y=[9,9,9]`。Lowering 已经交出两个 operation：

```text
op0：读 arg0_1，写 buf0          # old
op1：读 arg1_1，写 buf1
     buf1 mutations=['arg0_1']   # 写回调用者的 x
```

只看 op1 的数学循环，它读 y、写 buf1，与 op0 看似无关。但 `buf1` 最终覆盖 `arg0_1`，而 op0 需要旧的 `arg0_1`。`compute_dependencies()` 因而为 op1 增加：

```text
StarDep(arg0_1)                  # 记录被写回的目标版本
WeakDep(buf0, mutating_buf=buf1) # 修改必须晚于先前读者 op0
```

采集到的关键状态是：

```text
op1.unmet_dependencies = WeakDep(buf0, mutating_buf=buf1)
users[buf0] = op1(weak=True), OUTPUT
mutation_renames = {'arg0_1': 'buf1'}
mutation_real_name = {'buf1': 'arg0_1'}
```

`StarDep(arg0_1)` 记录写回目标；若目标由一个尚未完成的内部节点产生，它也会约束写回等待该生产者。本例的目标是已经可用的图输入，所以经过 `prune_deps()` 后，它不留在 `unmet_dependencies`，也不形成另一条 op0→op1 边。真正保护旧值读取的是未满足的 `WeakDep(buf0)`，它明确连接 op0 与 op1。后续读原名 `arg0_1` 时，依赖图会把它重命名为最新版本 `buf1`；生成代码仍通过 `mutation_real_name` 知道实际被写回的是原输入。

由这条边可推出正确顺序：先用旧 x 算得 `old=[1,2,3]`，再把 y 写入 x，最终 x 为 `[9,9,9]`。若反过来，old 会错误地变成 `[10,10,10]`。

### mutation 后的值可能已被上游改写成新值来源

对照函数：

```python
def read_after_mutation(x, y):
    x.copy_(y)
    return x * 2
```

这里进入 Scheduler 的后继乘法已经被改成直接读取 `y`，因此入口是：

```text
op0：读 arg1_1(y)，写 buf0 = y*2       # 普通输出
op1：读 arg1_1(y)，写 buf1，mutations=[arg0_1]
```

`compute_dependencies()` 后，op1 只有用于表达写回目标版本的 `StarDep(arg0_1)`；因为 arg0_1 是已可用输入，它不留在 unmet。op0 根本不读取旧 arg0_1，所以 op1 不需要指向 op0 的 WeakDep。实际状态为：

```text
mutation_renames = {'arg0_1': 'buf1'}
mutation_real_name = {'buf1': 'arg0_1'}
users[buf0] = OUTPUT
users[buf1] = OUTPUT
```

运行得到输出 `[8,10,12]`，调用者 x 更新为 `[4,5,6]`。这里的两个 OUTPUT 含义不同：buf0 是普通函数结果，buf1 代表必须保留的输入更新。版本表仍登记“原名 arg0_1 的最新逻辑版本是 buf1”；当前图的后继计算直接读 `y`，没有节点再读取这个逻辑新版本。

是否需要 WeakDep，取决于 mutation 前是否已经登记了必须完成的旧版本 reader。这个案例没有旧 `x` reader，因此不需要 WeakDep。

## 别名为什么共用使用者集合

如果一个输出与另一个 buffer 共享存储，按两个独立名字维护使用者会漏掉真实冲突。`compute_dependencies()` 先读取各输出的 alias 名，并让别名键指向同一份 `name_to_users` 列表。随后无论从哪个名字登记读取或 mutation，都能看到共享存储上的既有使用者。

例如 logical names `view_out` 与 `base` alias 同一 storage 时，若消费者登记为读 `view_out`，之后 mutation 声明覆盖 `base`，二者仍落入同一 users 列表；mutation 不会因为名字不同漏掉 view reader。

这一步只建立保守而正确的关系。哪些边可删除、mutation 版本怎样参与拓扑排序、节点何时成为最后使用者，放在下一项继续讲。

## 用新案例判断依赖

假设 `buf0=x+1` 已物化，随后两个节点分别计算 `buf1=buf0*2` 和 `buf2=buf0.sin()`：

- 两个消费者都各有一条读取 `buf0` 的 `MemoryDep`；
- `users[buf0]` 同时包含两个消费者；
- 两个消费者之间没有由这组读写自然产生的先后边；
- 若第三项要覆盖 `buf0`，它必须排在这两个旧读者之后，并会获得相应 mutation 顺序依赖。

改变某个消费者的布局会改变 `MemoryDep.index`，但不会取消它对 `buf0` 生产者的依赖。改变访问方式与改变执行先后，是依赖记录提供给后续阶段的两类不同信息。

若第一个消费者经 view 名读取、第二个经 base 名读取，而这两个名字已声明 alias，mutation 仍必须看见两名消费者。若其中一个消费者随后被死节点删除，对应 weak 顺序边才可能在依赖裁剪时消失；本篇只负责把当前已知读者完整登记出来。

## 继续阅读实现

[依赖构造核心源码：完整方法与状态推演](01-dependency-source/README.md)
