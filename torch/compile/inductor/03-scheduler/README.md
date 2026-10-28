# Scheduler：把 IR 工作排成可生成的执行计划

[上一篇：Lowering](../02-lowering/README.md) · [返回总览](../README.md) · [节点与依赖](01-dependencies/README.md) · [排序与生存期](02-order-lifetime/README.md) · [融合](03-fusion/README.md) · [后端分派](04-backend-dispatch/README.md) · 下一篇：[Codegen](../04-codegen/README.md)

Lowering 已把需要运行时执行的工作登记成 IR operation。Scheduler 接住这些 operation，
找出依赖、删除无用工作、安排顺序并形成最终节点——即完成融合、模板定型和最终整理后，
真正进入生成循环的调度单元；随后它按这个顺序直接调用相应后端和 wrapper 的生成接口。
这里的 wrapper 是 `GraphLowering` 在创建 Scheduler 前准备的、正在填写的图级代码与内存
规划对象：设备后端向它登记 kernel 定义和调用，Scheduler 也会直接向它登记 extern 调用、
设备切换和引用结束等动作，最后由 `GraphLowering.codegen()` 让它拼成完整源码。Scheduler
决定“现在把哪个节点交给哪个入口”，CPU/Triton 后端与 wrapper 决定“具体写出什么代码”。
两者在调用边界上协作，不存在一份先返回、再由 Codegen 另行读取的“动作列表”。

这一步不计算 Tensor 的值，也不决定 `relu` 的数学含义。它处理的是“哪些工作必须先做、哪些可以合在一起、每份中间结果最后在哪里使用、每项工作交给谁生成”。

## 先认识 Scheduler 操作的对象

假设 Lowering 留下四项工作：

```text
op0：普通逐元素计算，读取 x，写 a
op1：普通逐元素计算，读取 a，写 b
op2：调用外部 mm，读取 b 和 weight，写 c
op3：普通逐元素计算，读取 c，写 output
```

Scheduler 为每项 operation 建立一条 scheduler node。记录中保留原 operation，还加入本阶段需要的信息：读哪些 buffer、写哪些 buffer、在哪个 device 上运行，以及它属于普通计算、外部调用还是模板计算。

| Lowering 留下的 operation | Scheduler 中的节点 | 后续生成方式 |
| --- | --- | --- |
| `ComputedBuffer` | `SchedulerNode` | 交给设备后端生成普通计算 |
| `TemplateBuffer` | `SchedulerNode`，保留 template 身份 | 交给 template 分支 |
| `ExternKernel` | `ExternKernelSchedulerNode` | 生成已有算子的调用点 |
| no-op operation | `NopKernelSchedulerNode` | 维持必要关系，不生成普通 kernel |

这里的 no-op 是“运行时不需要普通 kernel，但在编译关系中仍要占一个完成位置”的
operation；它不是“可以随意删除的死计算”。若 operation 不属于当前支持的这些类型，
节点创建会抛 `NotImplementedError`，不会猜一个节点类型。

`Scheduler._init()` 建立这些节点以及按 operation 名、buffer 名查找节点的索引。此后 Scheduler 改变的是节点的关系、顺序和分组，不改 operation 所表示的数学计算。

## 四项职责怎样形成并执行一条计划

本篇对应四个直接子模块。第二项“排序与生存期”发生在两个时点：融合前先排序、删除和
计算祖先；最终节点整理结束后，再计算 `last_usage`。这不是第五项独立职责。

流程图还会出现 collective。它是 `all_reduce`、`all_gather` 等多设备通信工作：Lowering
把相应 FX 调用登记成保存通信算子、输入和输出/写回关系的 `_CollectiveKernel` IR
operation；Scheduler 将它包装成 extern 类调度节点。融合前的通信排序读取这些节点并在
相邻通信之间补顺序约束；融合后的 overlap 重排再读取包含通信、wait 和普通计算的节点
集合，决定它们怎样穿插。没有 collective 的图不会产生通信之间的附加顺序约束，内建
overlap 策略也没有 collective/wait 可供普通计算穿插；配置开关开启时，这两个函数本身
仍会运行。

四项工作会经历下面的状态变化：

```text
IR operations
      │
      ▼
建立 scheduler nodes 和读写边
      │  op0 → op1 → op2 → op3
      ▼
拓扑排序并删除无用节点
      │  每个消费者仍在生产者之后
      │  若启用通信重排，可先确定 collective 的全局先后关系
      ▼
准备专用分组、融合普通节点，并做最终节点整理
      │  [op0, op1] → op2 → op3
      │  还可能建立 foreach/combo 分组、合并循环、定型多候选模板、
      │  再按 compute/communication overlap 调整最终位置
      ▼
在最终序列上记录哪些 buffer 到当前节点后可结束生存期
      │  同时考虑读、写、alias 和输出身份
      ▼
按 device 和节点类型直接调用后端/wrapper 生成入口并发出释放动作
```

这里要特别注意：首次排序和删除发生在融合前；若启用 compute/communication overlap，
祖先关系确定后可以先决定 collective 的全局先后。前一步只固定通信工作彼此的次序；
融合后的第二步才把计算与通信穿插到最终位置，目的是让二者尽量重叠。foreach 分组、第二次拓扑排序和
pre-fusion custom pass 也发生在正式融合前。融合之后才合并循环、定型多候选模板，并按
compute/communication overlap 再调整节点位置，然后建立 combo nodes，并拆开此前通信排序
阶段建立的 grouped nodes，最后计算 `last_usage`。这样记录的生存期边界对应最终节点，
而不是融合前已经被合并或替换的节点。

这里的“最终节点”不是刚包装好的 scheduler node，而是融合、模板定型、通信重排及
combo/grouped 整理都完成以后，真正进入 `_codegen()` 循环的节点。

下面沿同一组 `op0` 到 `op3` 看每一步交出什么。

## 1. 建立依赖：由 buffer、alias、mutation 与符号得到先后关系

`op0` 写 `a`，`op1` 读 `a`，所以 `op1` 必须在 `op0` 之后。相同规则继续得到：

```text
op0 --a--> op1 --b--> op2 --c--> op3
```

这些边既限制执行顺序，也给融合提供关系依据。例如 `op0` 和 `op1` 是直接相连的普通逐元素工作，可以继续检查是否适合融合；`op2` 是外部调用，节点类型已经表明它不能被当作普通逐元素循环处理。

写回同一份存储时，Scheduler 还要区分修改前后的逻辑版本。例如：

```text
m0：读取旧 x，写 old_result
m1：读取 y，把结果写回 x
m2：读取写回后的 x，写 output

m0 --保护旧值读取--> m1 --提供 x 的新版本--> m2
```

第一条边保证 `m0` 读完旧值以后才能写回，第二条边保证 `m2` 看到写回后的值。Scheduler
用 mutation 的名字映射和附加依赖保存这两个次序，避免把“同一存储”误当成“同一时刻
的值”。被修改的是图输入或常量时，还会增加通向 OutputNode 的保活关系；输入 mutation
另登记被修改的输入与参数位置，因此即使没有普通数据消费者，也不会被当成死节点删除。

此外，两种关系不是普通 buffer 边却仍会限制顺序：共享 storage 的 alias 会合并相关
users；unbacked SymInt 的定义与使用会建立只用于调度的 fake dependency。这些边不表示
运行时多搬一份数据，而是避免把依赖同一存储或符号值的工作排错。具体构造留到
[节点与依赖](01-dependencies/README.md)。

## 2. 排序与生存期

### 融合前：先得到一条合法的活节点序列

依赖边建立后，拓扑排序保证每个生产者都排在消费者之前。若一项工作没有使用者、不是图输出，也没有必须保留的副作用，死节点删除会把它移走；若删除它以后，它的生产者也变得无用，删除会继续向前传播。

对当前案例，四项工作都通向 `output`，因此得到的活节点序列仍是：

```text
op0, op1, op2, op3
```

这一步还会计算祖先关系，供后面的候选判断使用：`op3` 依赖 `op2`，就不能为了凑一个融合组而越过 `op2`。排序规则、死节点删除和祖先信息在[排序与生存期](02-order-lifetime/README.md)展开。

## 3. 融合与最终节点整理

### 融合前：建立 foreach 分组并允许定制调整

首次排序、DCE 和祖先关系完成后，启用通信重排的流程会先决定 collective 的全局顺序。
这里决定的是各项通信工作的相对先后，还不是融合后根据 compute/communication overlap
安排最终位置。随后当前流程可以建立 foreach nodes，再做一次拓扑排序；配置的
pre-fusion custom pass 也在正式 `fuse_nodes()` 之前运行。这些工作会改变接下来参加融合
的节点或顺序，因此不能归到“融合后的整理”。

foreach node 把图中已经登记为同一列表、彼此没有数据依赖的多项工作包装成一个可并行
调度单位。融合后可选的 combo kernel 也把可并行节点组成一个生成单位，但会经过自己的
组合收益判断。grouped node 则是通信路径使用的临时围栏：组内节点可以彼此融合，其他
节点不能插进组内；在最终生成前它又被拆回组内融合后的普通节点。因此三者都会改变
最终序列的形成：foreach/combo 成功建组会改变调度单元；grouped node 至少会约束中间
排序和融合范围，若组内节点可以融合，还会进一步改变最终节点。即使本例没有进入这些
分支，它们也不能从完整流程中省略。

### 融合：把可共同生成的工作变成一个节点

Scheduler 检查候选节点是否有可利用的数据关系、循环范围是否兼容、device 和后端是否
一致，以及节点类型是否允许融合。这只是条件类别的入口；真实判断还要处理祖先与成环、
alias/mutation、规约方向、template epilogue、融合规模和可选 benchmark 等限制。

在这个案例中，假设 `op0` 与 `op1` 都是范围相同的普通逐元素计算，后端也允许融合，它们会变成一个融合节点：

```text
融合前：op0 → op1 → op2 → op3
融合后：[op0, op1] → op2 → op3
```

这不只是抽象假设。CPU 融合实验实际观察到相同范围的 `add → mul` 从两个普通节点变成
成员为 `[op0, op1]` 的一个 `FusedSchedulerNode`；`dependent_mixed` 分派实验又观察到
`Fused(op0,op1) → extern op2 → op3` 的最终序列。当前主案例借用的正是这个已经观测到的
节点形状；至于任意一对新节点是否接受，仍要重新运行上述检查。

融合节点内部先由 `op0` 算出 `a[i]`，紧接着由 `op1` 使用它算出 `b[i]`。`a` 不必成为两个独立 kernel 之间的完整中间结果。`b` 仍要作为融合节点的输出，因为外部的 `op2` 需要读取它。

`op2` 保持独立：它是 extern node。`op3` 虽然也是普通计算，却依赖 `op2` 的结果，不能跨过这个边界并入前面的组。候选怎样产生、哪些检查会接受或拒绝融合，在[融合](03-fusion/README.md)展开。

已确定的 `TemplateBuffer` 进入 Scheduler 后保留模板身份，可以按后端规则吸收合法
epilogue，但不会被当作任意普通节点处理。例如模板 matmul `op0` 产生 `buf0`，普通
pointwise `op1` 读取 `buf0` 做 `relu`；若模板的输出接口允许这项后处理且其余融合检查
通过，最终融合节点保留 template `op0`，并把 `op1` 作为 epilogue 列表交给后端的
`codegen_template(op0, [op1])`。这里 epilogue 就是接在模板结果后的普通计算，不是另一种
模板。CPU 实验在冻结权重的 matmul 案例中实际观察到了这条
`template matmul → relu` 分派；非冻结版本在该实验里选择的是 extern mm，不能当作模板
路径的证据。

`MultiTemplateBuffer` 仍包含模板和 extern 等
多个候选，以及由 Lowering 的选择算法提供的一份延迟测时入口；Scheduler 第一次读取时
可能才触发 autotune 或缓存查询并得到各候选耗时。没有 epilogue 需要联合比较时，最终
定型步骤取未融合候选中耗时最小的一项。若它不是 Triton template，Scheduler 就用该候选
产生的 `OperationBuffer` 重建节点。`OperationBuffer` 是“某项 IR operation 产生的、有
名字和 layout 的结果 buffer”这一共同基类；重建后的具体节点类型由所选 operation 决定，
例如 extern 候选会成为 `ExternKernelSchedulerNode`，之后不再走 template 分派。带 epilogue 的联合
benchmark 还可能更早选定 Triton 候选。Scheduler 在这里不创造候选，只消费测时结果并
定型节点。模板 kernel 怎样生成属于 Codegen，本篇只追踪定型后节点身份怎样影响出口。

### 融合后：整理最终节点并记录生存期

融合并非最后一次改变节点。之后还会按配置合并循环、定型多候选模板、重排
compute/communication、建立 combo kernel，并处理 grouped nodes。不是每次请求都会
进入每项分支，但只有这些整理完成后，下面的 `last_usage` 才对应真正进入生成循环的节点。

融合和节点整理结束后，Scheduler 才调用 `compute_last_usage()`。对于当前的最终序列：

```text
[op0, op1] → extern op2 → op3
```

`last_usage` 不是“本节点最后读取了谁”的日志。对每个最终节点，它根据该节点使用、写入
或通过 alias 关联的 buffer 名，扣除后续节点仍需要的名字，得到“执行到本节点之后可以
结束生存期”的集合。融合节点还保留组级与成员级信息。

对这条示意链：`a` 已成为融合 kernel 内部值，可能根本没有独立 wrapper buffer；若
`b` 是物化 buffer 且只供 extern `op2` 读取，它可在 op2 之后结束；`c` 只供 op3 使用时
可在 op3 之后结束。这个 a/b/c 状态是按当前算法作的推演，不冒充教程实验直接打印的
结果。若 `b` 同时是 graph output，它仍被返回值引用，不能在 op2 后释放。

Scheduler 在自己的 `_codegen()` 循环中消费这些集合：把名字加入待释放候选；边界
`flush()` 调用后端的统一边界接口，再由 `free_buffers()` 排除已删除、已经释放或当前
不能释放的 buffer，只为满足条件的名字调用 wrapper 的 `codegen_free`。因此它不仅把
信息“交给以后再处理”，而是在驱动生成时尝试把生存期决定落实进 wrapper 状态。
这里“当前不能释放”的一个直接例子是图输出：它还要由 wrapper 返回；没有真实分配空间的
`NoneLayout` 也无需生成 free。alias 已在计算 `last_usage` 时把相关名字一同考虑，它主要
影响候选何时出现，并不是到 `free_buffers()` 才第一次处理。

## 4. 后端分派：按最终顺序驱动生成

`Scheduler._codegen()` 按最终节点序列前进，并直接调用生成接口：普通节点交给当前 device
的普通后端；已定型 template 调用 template 入口；extern 节点先处理可能的原地复用，
再 `mark_run` 并向 wrapper 写已有调用；foreach/combo 节点有专门入口；nop 通常只
`mark_run()`，不生成普通 kernel。device、extern、template 或后端主动
`ready_to_flush()` 的边界都会调用统一的 `flush()` 接口。

`flush()` 对不同后端不是同一内部动作。CPU 后端可以先把多个节点积累在 kernel group
中，到 flush 才封口并生成 kernel 定义/调用；当前 Triton/SIMD 后端则在
`codegen_node()` 时已经生成 kernel，`flush()` 本身为空。统一的是边界调用，不是所有
后端都必须缓存一批节点。

当前案例的交接顺序是：

```text
普通融合节点 [op0, op1]
        │
        ├─ 先结束当前普通生成组
        ▼
extern op2
        ▼
新的普通生成组 op3
```

这里没有返回一串“生成动作”。这次调用的可观察出口是：普通后端已经接收
`[op0,op1]` 和 `op3`，wrapper 已写入 extern op2 的调用与释放动作，kernel/wrapper
代码缓冲区已经按上述顺序被填充；`Scheduler.codegen()` 本身返回 `None`。随后
`GraphLowering.codegen()` 才让 wrapper 生成完整源码。

若某个 device 没有已注册后端，或 GPU 环境缺少所需 Triton/版本条件，后端创建会明确
报错，不会把节点静默交给错误设备。当前 CPU 实验的直接观察包括：普通 `op0/op1` 进入
同一融合节点；extern `op2` 令前一 CPU group flush；冻结权重的 template matmul 与 relu
进入 `codegen_template(op0, [op1])`；空 no-op 只标记完成且不生成普通 kernel。跨 device
行为是依据当前 device guard/flush 分支的源码推演，并非本机多设备实测。当前版本还有
一个已复现限制：nop 在开启 Triton kernel debug-sync 的特定路径会因 device 状态未建立
而抛 `UnboundLocalError`，不能概括成所有 nop 配置都顺利完成。

不同节点如何选择后端、何时结束当前组以及 device 切换如何处理，在[后端分派](04-backend-dispatch/README.md)展开。

## 用条件变化检查整条链

仍看 `[op0, op1] → extern op2 → op3`。

如果增加一项读取 `a` 的旁路工作，依赖图会多出 `op0 → 旁路节点`。Scheduler 必须重新排序，并重新判断旁路节点能否与原来的普通节点共同融合；最终节点改变以后，`a` 的生存期边界也随之重算。这是按依赖和融合算法作的迁移推演，不是现有实验的直接输出。

如果把 `op3` 放到另一 device，`op2 → op3` 的数据依赖不变，但两者不能组成同一 device 上的生成组。分派时会在边界结束当前组，再切换到另一设备的后端；这同样是源码支持的迁移推演，不是当前 CPU 实验。

再看一个实际采集的输入修改案例。函数先计算旧 `x` 的值，再把 `y` 写回 `x`：

```python
def f(x, y):
    old = x + 1
    x.copy_(y)
    return old, x
```

Lowering 交来的两项工作和 Scheduler 建立的关系如下。这里的 `arg0_1` 是输入 `x`，
`buf0` 保存 `old`，`buf1` 是写回后的 `x`。

| 时刻 | 当前状态 | 为什么需要它 |
| --- | --- | --- |
| 包装并建边 | `op0` 生成 `buf0`；`op1` 写 `buf1`，并把 `arg0_1` 标为被修改的输入 | 先把“算旧值”和“覆盖输入”表示成两项调度工作 |
| 补 mutation 顺序 | `op1` 对 `arg0_1` 有 `StarDep`，还对 `buf0` 有 `WeakDep(buf0, mutating_buf=buf1)` | `op1` 必须等 `op0` 读完旧 `x`，否则 `old` 会读到新值 |
| 记录名字关系 | `mutation_renames[arg0_1] = buf1`，而 `mutation_real_name[buf1] = arg0_1` | 后续看到新版本 `buf1` 时，仍知道它写回的是输入 `x` 的那块存储 |
| 排序与删除 | 实际顺序为 `[op0, op1]`，两项都保留 | `old` 是图输出；对输入的写回也是可观察结果，不能当作无用工作删除 |
| 计算最终生存期 | 实验中 `op0.last_usage = []`，`op1.last_usage = [arg0_1, arg1_1]` | 旧 `x` 在 `op0` 之后仍要被 `op1` 写回，因此不能提前结束；写回完成后两个输入都不再被后续节点使用 |
| 驱动生成 | `op0`、`op1` 的先后语义保留，并以最终节点交给 CPU 后端；最终节点可能是包含二者的融合节点 | wrapper 同时保留 `old` 的返回值并落实输入写回，本层出口因而保住修改前后的两个可观察结果 |

这次运行得到 `old=[1,2,3]`、修改后的 `x=[9,9,9]`。表中依赖、排序和
`last_usage` 来自实验 08、09 的实际输出；“交给 CPU 后端并由 wrapper 落实写回”来自
当前 Scheduler 生成路径。现有实验没有单独打印这两个节点在融合后的身份，因此这里不把
它们宣称为“必然融合”或“必然不融合”。关键是：无论最后是否融合，mutation 建立的先后
关系和两个可观察结果都必须保留到本层出口。

模板路径则先保留候选身份，融合阶段只吸收合法 epilogue；多候选模板在最终整理时选定
template 或 extern/其他 operation；`last_usage` 基于重建后的最终节点计算，分派才按已
定型身份进入 template 或其他入口。

判断一个新案例时，可以沿实际状态变化追问：谁生产、谁读取；依赖允许哪些顺序；哪些活节点能组成一个生成单元；最终节点上每份数据最后在哪里使用；每个节点交给哪个后端。[节点与依赖](01-dependencies/README.md)、[融合](03-fusion/README.md)和[后端分派](04-backend-dispatch/README.md)分别展开相应工作；[排序与生存期](02-order-lifetime/README.md)同时解释融合前的排序和融合后的最后使用记录。
