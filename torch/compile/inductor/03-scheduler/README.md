# Scheduler：把 IR 的读写关系变成执行与融合计划

[上一篇：Lowering](../02-lowering/README.md) · [返回总览](../README.md) · [节点与依赖](01-dependencies/README.md) · [排序与生存期](02-order-lifetime/README.md) · [融合](03-fusion/README.md) · [后端分派](04-backend-dispatch/README.md) · 下一篇：[Codegen](../04-codegen/README.md)

Scheduler 接收 `GraphLowering.operations`。它不再决定 `aten.relu` 的数学定义，而是回答：每项操作读什么、写什么、依赖谁、能否删除、能否融合、何时释放结果、交给哪个 device backend。

## 从 operation 到 scheduler node

先看两项 Lowering 已经登记的工作：

```text
op0：读取 x、y，写 buf0，位置范围 0..7
op1：读取 buf0，写 buf1，位置范围 0..7
```

Scheduler 为每项工作建立一条记录。以 op1 为例，这条记录保存：它包装哪项 IR operation、读取 `buf0`、写入 `buf1`、循环范围是 0..7，而且必须等产生 `buf0` 的 op0 完成。后文把这条带读写和先后关系的记录称为 scheduler node。Scheduler 操作的是这些记录，不是在编译时计算 buf0 或 buf1 的数值。

`Scheduler._init` 首先调用 `create_scheduler_node`：

| Lowering 留下的工作 | Scheduler 建立的记录 | 记录怎样生成代码 |
| --- | --- | --- |
| `ComputedBuffer` / `TemplateBuffer` | `SchedulerNode` | 可按循环或模板生成的计算 |
| `ExternKernel` | `ExternKernelSchedulerNode` | 已有调用，生成调用点而非普通融合循环 |
| no-op operation | `NopKernelSchedulerNode` | 维护关系但不生成普通 kernel |

包装后，每条记录分析自己读取和写入的名字，并删除已经确定不影响顺序的关系。Scheduler 再建立“操作名找到对应节点”和“buffer 名找到对应记录”的索引表，然后比较全图的先后关系。

## 四项直接职责

```text
IR operations
   │
   ├─ Q31 节点与依赖：记录每项 operation 读谁、写谁、由谁产生、被谁使用
   ├─ Q32 顺序与生存期：保证生产者在使用者之前，删除无用工作，找出每份数据最后由谁读取
   ├─ Q33 融合：产生垂直/水平候选，检查合法性、分组与后端能力
   └─ Q34 交给代码生成后端：按 device 与工作类型成组提交，结束一组后再开始下一组
   │
   ▼
按顺序排列的工作记录，逐项交给 CPU、Triton 等代码生成器
```

这四项有严格的数据交接。Q31 先把“谁读写谁”变成边；Q32 才能据此删除死节点、排列活节点并知道一份 storage 最后在哪用；Q33 在不破坏这些边和生存期的前提下把兼容节点组成更大生成单元；Q34 遍历最终节点，遇到 extern、template 或 device 边界时先结束当前后端 group，再提交下一类工作。

图中的“垂直候选”是有生产者—使用者关系的一对节点，例如 op1 读取 op0 写出的 buf0；“水平候选”是彼此无依赖、但循环范围与后端兼容的两项工作，例如两条分支都读取 x。group 是后端暂存、准备合成一个 kernel 的节点集合；extern、template 或 device 改变会结束当前 group，源码把这次提交动作称为 `flush()`。last usage 是最终顺序中最后读取某份数据的节点，例如 op2 是 buf0 的最后读取者，op2 之后才能安排释放。

### 依赖建模为何先发生

假设有两个已经物化的操作：

```text
buf0 = x + y       读 x,y；写 buf0
buf1 = relu(buf0)  读 buf0；写 buf1
```

`buf1` 必须在 `buf0` 可用后执行。这条关系也允许 Scheduler 判断垂直融合：如果两个循环范围兼容、后端允许，第二项可以直接消费第一项当前位置的值，不必把完整 `buf0` 留给另一个 kernel。

如果两个节点都读 x、互不依赖，可能成为水平融合候选；如果有不兼容 device、布局、mutation、模板边界或后端限制，就保持分开。

### 三类依赖各自保护什么

Scheduler 用两个集合记录“生成计划已经推进到哪里”。`available` 保存后续工作现在可以引用的数据名；图输入一开始就在其中。`unmet` 保存某个节点仍在等待的数据名；如果 op1 读取 buf0，而 buf0 尚未加入 available，op1 的 unmet 就包含 buf0。等写出 buf0 的 op0 被安排在前面，这条等待关系也就成为 `op0→op1` 的依赖边。

`MemoryDep(name,index,ranges)` 同时说明读写哪份数据、怎样计算一维位置、循环变量取哪些值。例如写记录 `MemoryDep(buf0,index=row,ranges={row:2})` 表示 row=0、1 时分别写 buf0[0]、buf0[1]；读表达式 `3*row+0/1/2` 表示同一 row 下读取输入的三个位置。独立实验把三种差异放在一起：

| 请求 | producer 的写 | consumer 的读/顺序要求 | 形成的关系 |
| --- | --- | --- | --- |
| 2×3 按行求和 | `MemoryDep(buf0, index=row, ranges={row:2})` | 输入三次精确读取 `3*row+0/1/2` | 输入已 available，op0 无 unmet；buf0 的 user 是 OUTPUT |
| `mm→sin` | mm 用 `StarDep` 写整个 buf0 | sin 用 `MemoryDep(buf0,index=i,ranges={i:8})` | op1 的 unmet 指向 buf0，因此 op0 必须先完成 |
| `old=x+1; x.copy_(y)` | op0 写保存旧读的 buf0；op1 写 mutation buf1 | op1 增加 `StarDep(arg0_1)` 和 `WeakDep(buf0, mutating_buf=buf1)` | WeakDep 强制旧 reader 在覆盖前完成 |

`MemoryDep` 能说出具体索引和范围，适合普通循环；extern 的内部访问不按 Inductor 位置表达时用 `StarDep` 保守表示整份 buffer。`WeakDep` 不表示 op1 要读取 buf0 的数值，它只保护顺序：buf0 所代表的旧 x 读取必须先结束，op1 才能覆盖同一真实 storage。

mutation 后，逻辑名字更新为 `mutation_renames={'arg0_1':'buf1'}`，同时保存 `mutation_real_name={'buf1':'arg0_1'}`。后继读取“新 x”应跟随 buf1 这个最新逻辑版本；释放存储时又要知道 buf1 仍对应调用者的 arg0_1，而不是一块可任意释放的独立内存。

边并不是由 `MemoryDep`、`StarDep` 的类型名字直接决定，而是由它们指向的 buffer 名建立：Scheduler 先记录哪个节点写出 buf0；后续节点出现尚未满足的 `read(buf0)` 时，就把后者登记为 buf0 的 user。于是 mm 的 `StarDep write(buf0)` 与 sin 的 `MemoryDep read(buf0)` 虽然类型不同，仍形成 `op0→op1`。图输入 arg0_1、arg1_1 一开始已经 available，所以读取它们不会形成 unmet producer。

mutation 对照把 `compute_dependencies()` 前后列开：

| 请求 | mutation node 原有 reads | 依赖补全后新增 | 最终边与版本 |
| --- | --- | --- | --- |
| `old=x+1; x.copy_(y); return old,x` | 精确读取 y | `StarDep(arg0_1)`、`WeakDep(buf0, mutating_buf=buf1)` | buf0 的旧读必须先完成；arg0_1 的最新逻辑名变为 buf1 |
| `x.copy_(y); return x*2` | 当前优化结果的两个 operation 都直接读取 y | copy 节点只增加 `StarDep(arg0_1)`，没有 WeakDep | 没有旧 x reader 要保护；结果为 `[8,10,12]`，调用者 x 为 `[4,5,6]` |

第二例当前已经把“读取新 x 再乘 2”等价改写为“直接读取 y 再乘 2”，因此不能用它证明后继一定显式读取 buf1；它能证明的是：不存在旧版本 reader 时，不会凭空添加 WeakDep。若仍有旧 reader，第一例中的 WeakDep 才把 mutation 排在它之后。

具体到第一例，buf0 的 users 从只有 OUTPUT 变成 `op1(weak=True)` 加 OUTPUT。weak user 不表示 op1 消费 buf0 的数值，却足以禁止 op1 越到 op0 前面。buf1 又被记作 arg0_1 的最新逻辑版本和真实 storage 映射，供后续排序、生存期与 wrapper 写回继续使用。

### 排序和删除改变什么

`compute_dependencies()` 后进行拓扑排序和 dead node elimination。mutation 用重命名表区分同一存储的不同逻辑版本，避免依赖图形成假环。之后计算 ancestors，为融合合法性提供快速判断。

这些工作改变的是执行计划中的节点与边，不重新定义算子数值。

### 分叉、死链和最后使用的具体状态

three-mm 案例先计算 `buf0=mm(x,w0)`，再由两个独立 mm 分别读取 buf0。合法拓扑是 `op0,op1,op2`；op1、op2 都把 op0 记为 ancestor，但彼此没有祖先关系，所以两者的相对顺序可在不破坏依赖时调整。buf0 的最后使用在较后的 op2，因而 `last_usage[op2]` 包含 buf0；不能在 op1 结束时释放。

实验把初始节点反转后再排序，得到 `op0,op2,op1`：这不是结果不稳定，而是具体展示了唯一硬约束是 op0 必须在两个消费者之前。以正常顺序 `op0,op1,op2` 为准，last usage 是：

| 节点 | 本节点后不再需要的名字 | 原因 |
| --- | --- | --- |
| op0 | arg0_1、arg1_1 | 首次 mm 已最后读取 x 和 first_weight |
| op1 | arg2_1 | 左分支最后读取 left_weight；buf0 还有右分支使用 |
| op2 | arg3_1、buf0 | 右分支最后读取 right_weight，也完成 buf0 的最后一次消费 |

若改用合法顺序 `op0,op2,op1`，buf0 的最后使用也随较后的消费者移动到 op1。last usage 不是 buffer 自身的固定属性，而是“在当前最终顺序中，未来节点是否还会使用它”的反向计算结果。

Scheduler 的死代码删除只处理它已经建立了 users 与副作用信息的调度节点，源码称 controlled DCE。案例含 `dead_parent→dead_leaf`、一个有副作用的 effect 节点和一个活输出节点。整理后只剩 `effect, live`，删除 dead_parent、dead_leaf 及其三个临时 buffer。删除从无 user 的 leaf 反向传播到 parent；effect 即使没有普通输出 user 也必须保留，因为移除它会改变程序可观察行为。

本例的 effect 节点代表一次写入操作：即使没有节点读取它的返回值，它仍会修改调用者能够观察的数据。若删除它，函数结束后的输入内容不同，所以不能按“无 user”删除。

mutation 案例则展示“逻辑版本”和“真实 storage”共同影响 last usage：写入 buf1 mutation 后，真实名字仍是 arg0_1。最后使用计算不能把 buf1 当成全新临时 storage 提前回收，否则会破坏调用者仍持有的 x。

单次 mutation 的实际集合为 `last_usage(op0)=[]`、`last_usage(op1)={arg0_1,arg1_1}`。buf0 是 graph output，所以 op0 后仍活跃；op1 最后读取 y，并完成对调用者 x 的写回。`mutation_real_name[buf1]=arg0_1` 使反向扫描按真实 storage 计算，而不是把 buf1 当成独立临时结果。

double-mutation 请求 `before=x+1; x.copy_(y); middle=x*2; x.copy_(z)` 到 Scheduler 时只有三个节点：op0 产生 before；op1 直接从 y 产生 middle；op2 把 z 写回 x。第一个 copy 的值传播已经使 middle 直接读取 y，因此 Scheduler 没有第二份 input-mutation 版本可供排列。最终只有 `mutation_renames[arg0_1]=buf2`，last usage 分别是：op0 为空，op1 完成 arg1_1，op2 完成 arg0_1 与 arg2_1。这里必须按 Scheduler 实际输入讲版本，不能按用户源码的 copy_ 数量猜节点数。

### 融合如何出现在贯穿案例中

实验的 post-grad FX 图有 add、relu、mul 三节点，生成代码却是单循环：

```text
同一个 x0：load x[x0], y[x0]
          → add
          → relu
          → mul 2
          → store output[x0]
```

Lowering 可以把逐元素规则延迟组合；对于已形成的多个 SchedulerNode，`fuse_nodes` 还会按依赖、分组、后端能力和配置尝试融合。最终应以实际 pre/post fusion 节点或生成代码确认路径，不能仅凭 Python 源码相邻就断言融合。

候选分数有四项：是否满足模板优先并能省访问、两边是否同为规约或同为非规约且能省访问、估计减少的字节数、原始顺序距离的负数。分数只在多个合法候选之间决定先尝试谁，不会把非法候选变合法。独立实验同时观察了接受和拒绝，而不把“尺寸一样”当成充分条件：

| 场景 | 候选关系 | 实际决定 |
| --- | --- | --- |
| vertical | op1 读取 op0，二者 CPU group 都是 8 项 | 候选分数 `(True,True,32,-1)`，合成 `op0_op1` |
| horizontal | 两条独立 8 项分支 | 先形成各自垂直候选，也形成 op0/op2 水平候选，最终四成员同组 |
| shifted index | 两节点 group 都是 8 项，但第二项按移位索引取数 | 拒绝：没有可用的 shared data 关系 |
| extern boundary | mm extern 后接 sin | 拒绝普通融合：一方是 extern/nop 类型 |
| `max_fusion_size=1` | 原本可融合的依赖链 | 拒绝：超过配置允许的融合规模 |

垂直融合利用 producer→consumer 的数据流；水平融合让相同后端和兼容范围的独立工作共用一次生成/启动。两者都还要通过依赖、索引、节点类型、规模和后端检查。拒绝只意味着保留多个生成单元，不改变数学结果。

以 vertical 的两个 8 元素 float32 节点为例，实际候选分数是 `(True,True,32,-1)`。四项依次表示：当前模板优先条件满足且能省内存访问、两节点同为 reduction 或同为非 reduction 且能省访问、估计省下 32 字节数据移动、原始顺序距离为 1。分数用于多个合法候选冲突时排序；它不能让一个非法候选越过 `can_fuse` 检查。融合前是 op0、op1，融合后是含两个成员的 `op0_op1`。

horizontal 案例原来有四项已物化 operation：add 的生产与输出、mul 的生产与输出。候选先包含两组垂直关系 `(op0,op1)`、`(op2,op3)`，也包含共享输入 x 的水平关系 `(op0,op2)`；本次循环最终把四项并成一个 fused node。branched 案例则把共同 producer 和两个消费者的五项 operation 合成一组，说明“一个结果有两个 user”不会自动禁止融合，仍需看索引、输出身份和后端能力。

三条拒绝路径分别改变不同条件：

- roll 的 consumer 虽然也是 8 项，却按移位索引读取 producer；读写依赖不能按同一位置匹配，memory score 为 0，当前非 aggressive 配置报告 `no shared data`。
- mm 是 `ExternKernelSchedulerNode`，在普通融合检查的节点类型关口直接报告 `node1 is extern or nop`；相同 CPU device 和 8 个输出元素都不能绕过这一边界。
- `max_fusion_size=1` 保留了合法 producer→consumer 关系，但两个成员超过配置上限，因此报告 `exceeds max fusion`。

读者据此可以区分“没有数据共享”“节点类型不允许”和“启发式规模限制”：三者最终都保留多个节点，但原因和改变条件后的结果不同。

### 生存期怎样影响 wrapper

`compute_last_usage()` 标出某个 buffer 最后被谁使用。codegen 遍历节点时把 `node.last_usage` 加入待释放集合。后端 flush kernel 后，wrapper 可在安全位置复用或释放存储。

贯穿案例没有中间完整 buffer，只需分配最终 `buf0`。若一个真实中间结果被两个不能融合的后继读取，它的最后使用位置就决定何时可以回收。

## codegen 顺序中的分支

template 的 epilogue 是接在主计算结果后的逐元素计算，例如 mm 模板内部算出累加结果后紧接着做 relu，从而不必先写完整 mm 输出再启动一个 relu kernel。foreach/combo 节点则把多项结构相同的工作交给一次组合 kernel 生成。`Scheduler._codegen()` 按最终 `self.nodes` 遍历：

```text
template node → backend.codegen_template(node, epilogue)
extern node   → codegen_extern_call(node)
foreach node  → backend.codegen_combo_kernel(node)
普通/融合节点 → backend.codegen_node(node)
nop node      → 只标记完成
```

device 改变、extern 或 template 边界可能先 `flush()` 当前后端；有 device guard 的设备会生成进入/退出守卫。节点完成后，其输出加入 available buffer，operation 名加入 completed 集合。

Scheduler 因而既做全局计划，也承担向设备后端有序提交工作的控制循环。后端怎样把一个融合节点写成 C++ 或 Triton，是下一篇的职责。

## 一个混合序列怎样走到 Scheduler 出口

独立实验构造了“两个普通 pointwise → mm extern → 一个普通 pointwise”的节点序列。按本层四项职责推进：

```text
op0: 普通 pointwise，写 a
op1: 普通 pointwise，读 a，写 b
mm : extern，读 b 和 weight，写 c
op2: 普通 pointwise，读 c，写 output
```

1. 依赖建模得到 `op0→op1→mm→op2`；如果 op1 与 op0 的循环和后端条件兼容，它们可成为同一融合组。
2. 拓扑与删除保留这条有用链；ancestor 信息让候选判断知道 op2 不能越过 mm。
3. codegen 遍历前两个普通节点时，CPU backend 可暂存它们组成的 kernel group。
4. 遇到 mm extern，Scheduler 先 flush 当前 group，生成 mm 调用；否则尚未生成的 b 不可供 mm 使用。
5. op2 在 extern 之后进入新的 group，遍历结束再 final flush。

本层出口因此不是一张静态“融合后图”，而是一串按边界提交的动作：`CPU group(op0,op1) → extern mm → CPU group(op2)`，同时附带各 buffer 的最后使用位置。Codegen 接到的顺序已经足以生成分配、调用和释放时间线。

实际 dispatch 实验中的节点名是融合的 `op0_op1`、extern `op2`、普通 `op3`。`completed` 保存已经交给生成逻辑的 operation 名；`pending_free` 保存已经到最后使用、但要等当前 group 真正 flush 后才能写出释放动作的数据名。关键状态按下面推进：

| 时刻 | 动作 | available / completed 的新增 | 为什么要这样做 |
| --- | --- | --- | --- |
| 访问 `op0_op1` | `CppScheduling.codegen_node` 暂存两个成员 | buf0、buf1 可作为后续逻辑结果；op0、op1 completed | 当前 CPU group 尚可继续收普通节点 |
| 访问 extern op2 前 | `flush()` CPU group | pending free 中到期的 buf0/arg0_1 在 flush 后处理 | mm 必须读取已经生成的 buf1，不能越过未提交 group |
| 提交 extern op2 | wrapper 记录 mm 调用 | buf2 available；op2 completed | extern 不进入普通 C++ loop group |
| 访问 op3 | 新的 `CppScheduling.codegen_node` | buf3 available；op3 completed | extern 后重新开始普通 group |
| 遍历结束 | final flush | 处理 buf2 的最后使用 | 确保最后一组也真正生成 |

这里的 available 与“机器代码已经执行”不是一回事。普通 fused node 交给 backend 后，Scheduler 就把其输出 buf0、buf1 加入 available，并把成员 op0、op1 加入 completed；它们表示后续生成动作可以引用这些逻辑结果。直到 extern 边界触发 `flush()`，暂存的 CPU group 才写成完整 kernel。pending_free 同样先收集 last usage，flush 后由 wrapper 生成逻辑消费并清空。

以 mixed 实验的实际集合为例：

```text
开始：available={三个输入}，completed={}，pending_free={}
提交 op0_op1 后：available 增加 buf0、buf1；completed 增加 op0、op1；pending_free={buf0,arg0_1}
extern 前 flush 后：pending_free 清空
提交 mm op2 后：available 增加 buf2；completed 增加 op2
提交 op3 后：available 增加 buf3；completed 增加 op3；pending_free={buf2}
final flush 后：pending_free 清空
```

buf0 是融合组内部已到最后使用的中间结果，buf1 是 mm 需要的 pointwise 输出；二者同时进入 available，不代表二者具有相同生存期。

FallbackKernel 和它的 MultiOutput 在当前实验中都按 extern scheduler node 分派：先提交 fallback buf0，再提交从 packed result 取 Tensor 的 buf1。它们没有因为 device 同为 CPU 就混入普通 C++ pointwise group；node type 本身就是边界。

模板分支必须单独看证据。本机尝试对 16×16 mm 打开 `max_autotune` 并只指定 CPP，实际仍得到 `ExternKernelOut op0`，relu 是普通 op1；事件中没有 `codegen_template`。因此本篇只能依据当前 `_codegen` 源码说明：真正的 template node 会带可融合 epilogue 交给 `backend.codegen_template`。本次运行不能证明模板路径已经执行，也不能把配置意图写成选择结果。

把 op1 的另一个结果也作为图输出时，它的存储不能在 mm 读完 b 后立即回收；`last_usage` 会延长到返回端。把 mm 换成跨 device 节点时，除 flush 外还会切换 device guard。这两种变化分别来自生存期和后端边界，不改变前面的数学定义。

本层四项职责的内部按以下顺序展开：[节点与依赖](01-dependencies/README.md)、[排序与生存期](02-order-lifetime/README.md)、[融合](03-fusion/README.md)、[后端分派](04-backend-dispatch/README.md)。

## 改变一个条件，沿四项职责重新判断

仍看 `op0→op1→extern mm→op2`。如果 op1 产生的 b 除了给 mm，还直接作为图输出：

1. Q31 为 b 增加 OUTPUT user；producer→mm 的数据边不消失。
2. Q32 不能只按 mm 的读取把 b 判为可回收；图输出要求把真实 storage 保留到 wrapper 返回。
3. Q33 即使仍能融合 op0/op1，也不能把 b 仅当作融合函数中转瞬即逝、没有输出身份的局部值；融合节点必须产生可供 mm 和返回共同引用的结果。
4. Q34 仍在 mm 前 flush 前一 CPU group，提交 mm 后再开始 op2；返回端根据延长的生存期处理 b。

如果改变的不是输出，而是把 op2 放到另一 device，Q31/Q32 的数学依赖仍是 mm→op2；Q33 会因 device 不同拒绝普通融合；Q34 在跨 device 时 flush 当前 backend，并切换相应 guard/backend。哪个决定改变，取决于新增条件作用在依赖、生存期、融合合法性还是生成边界，不能统一回答“多一个 kernel”。

## 判断新案例

看到两个逐元素操作时，不要只问“数学上能否合并”。依次检查：后者是否依赖前者；循环分组是否兼容；中间结果是否还有别的使用者；是否跨 device/extern/template 边界；目标后端是否允许融合。只有这些条件共同满足，才会形成一个生成单元。

例如在上面的混合序列中再插入一个只读 a 的旁路 op3，不能只说“op0 和 op1 相邻所以融合”。先检查 op0 的结果是否可在融合组内同时供 op1/op3 使用，再决定分组；a 的最后使用也由 op3 的实际位置决定。
