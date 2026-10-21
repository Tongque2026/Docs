# 存储、布局与修改篇作者记录

## 所在层、输入与出口

- 队列位置：Q23，第 2 层，父项 Q20；第五轮已按技能重新复核，保留“本层完成”。本状态不替代 Q231 叶子的源码回修。
- 读者前置：Q22 已能从 Pointwise/Reduction/View 推导逻辑访问；本篇回答这些规则何时获得 storage、怎样决定地址排列、怎样共享和怎样保护数据版本。
- 本篇输入：延迟 Loops/View、消费者与输出布局合同、已有 storage/alias、读者关系和 mutation 请求。
- 本篇出口：拥有名字与 operation 的 ComputedBuffer/调用结果，冻结或受约束的 layout，显式 view/non-owning alias，以及 mutation target/旧读依赖；Scheduler 据此构造真实读写顺序，wrapper/codegen 据此准备或复用存储。
- 本篇直接职责：物化身份、布局选择、alias/view、mutation/version 四者协作。Q231 是下一层叶子，负责 realize 与布局决定的完整源码；mutation/alias 的完整实现仍在本篇当前层讲合同，不为每个函数机械拆目录。
- 边界：本篇不把“登记 buffer”说成运行时已分配；不把 logical buffer 名说成独立物理 storage；不在此展开 Scheduler 依赖算法和 wrapper 复用计划。

## 篇内知识点队列

| 顺序·编号 | 目标与完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| 1·K01-R | 展示同一 StorageBox 从 Loops 原地切换到 ComputedBuffer，区分编译登记与运行分配 | pointwise consumer/output/extern 对照 | 完成，已实测并复读 |
| 2·K02-R | 从实际 MemoryDep stride hints 推到 fill order/FixedLayout，并说明外部 constraint 优先级 | `[2,3]` 两种 stride 公式 | 完成，已按当前源码复核 |
| 3·K03-R | 区分 owning buffer、ReinterpretView 和 NonOwningLayout 的 storage 身份 | transpose 实验 + concat realize_into 源码 | 完成，已复读 |
| 4·K04-R | 解释内部 fast-path mutation 与外部输入写回的不同身份合同 | copy_/mutate_to/realize_into | 完成，已按当前源码复核 |
| 5·K05-R | 走通旧读、覆盖、普通 graph output 和用户返回 alias，并分开各自物化原因 | 实验 06 + 实验 18 重排例 | 完成，错误因果已替换并实测 |
| 6·K06-R | 综合判断新增/删除输出、改变 extern consumer、经 view 写入时哪些决定变化 | 只凭正文回答具体 storage/version | 完成，已复读 |

## 处理情况与案例覆盖

| 情况与条件 | 状态变化 | 本层出口 | 证据 | 当前深度 |
| --- | --- | --- | --- | --- |
| graph output 需要稳定存储 | Loops 不能只保持内联规则 | ComputedBuffer + layout/output identity | 实验 04/18 | 已实测并讲解 |
| extern/realized-input consumer | 参数必须是可传递 buffer | producer 在调用前物化 | 实验 18 mm | 已实测并讲解 |
| reuse/read complexity heuristic | 多次内联成本过高 | realize_hint/mark_reuse 可物化 | 当前源码 | 已讲条件边界并复读 |
| layout 仍 flexible | operation 有实际 MemoryDep | stride hints→fill order→FixedLayout | 当前源码与公式推演 | 已讲机制并复读 |
| exact/order/output constraint | 消费者/接口指定 stride | 提前冻结/转换；heuristic 不再自由选择 | 当前源码 | 已讲边界 |
| storage-backed view | 新 size/stride/offset，共享旧 storage | ReinterpretView，0 operation | 实验 05 | 已实测并讲解 |
| non-owning output/alias | 输出直接写另一 storage 的 view/slice | NonOwningLayout 指向 concat destination slice | 当前 ConcatKernel/post-grad 源码 | 已讲具体对象，未独立运行 |
| 内部临时值可安全替换 | 外部不可观察旧 storage 身份 | StorageBox.data 可改指向已 realize 新数据，不登记外部写回 | 当前 mutate_to 源码 | 已讲对照 |
| 输入/module buffer mutation | 外部必须观察原 storage 被改 | MutationLayout 指向原 target，生成写回 | 实验 06 | 已实测并讲解 |
| 旧 reader 跨 mutation | reader 仍引用旧版本 | copy_ 节点处将延迟 add 物化为 op0，再登记 mutation op1 | 实验 18 old_reader_reordered | 已实测并讲解 |
| old 同时是普通 graph output | output 本身已要求物化 | 不能用该例单独证明 copy_ 首次触发 realize | 实验 06 | 错误已修正 |

## 依据与已发现的错误

- 当前源码：`ir.py` 的 StorageBox/ComputedBuffer/FlexibleLayout/View/MutationLayout，`lowering.py` 的 copy_/mutate_to，`graph.py` 的 reader/mutation/output 交接。
- 实验 06 的 old 本身就是 post-grad graph 普通输出，所以它在 add node 后已经是 buf0。原正文“add 在 copy_ lowering 过程中被迫成为 buf0”与实际节点轨迹矛盾，必须删除。
- 实验 18 新增 `old_reader_reordered`：独立 post-grad pass 将合法顺序变成 add→copy_→mul；add 后 0 operation，copy_ 后已有 op0/op1，mul 后新增 op2。它与 `mark_buffer_mutated` 源码共同证明旧 reader 在 mutation 处被提前物化。
- 两个逻辑循环生成在一个 CPU function 是本例证据，不推广所有 backend 或所有 mutation 融合。
- `MutationLayoutSHOULDREMOVE` 是当前类名；不从名字推断未来设计。
- 实际读者反馈尚无。

## 作者复读与迁移题

1. 从实验 06 删除 old 的普通 graph output 后，能否继续断言 add 会在 add 节点立即物化？答：不能；输出理由已消失，需看剩余消费者、reuse heuristic 或后续 mutation 保护。
2. 在重排例中 mul 在线性顺序上位于 copy_ 后，为什么仍读到旧 x？答：add 作为读取 x 的 StorageBox 已登记到 `name_to_users`；copy_ 的 `mark_buffer_mutated` 先将它物化为 op0，再建立写入 op1，mul 后来读取的是已保存的 add buffer。
3. 内部临时 changed 与 graph input changed 接收同一新值，为什么 IR 不同？答：内部值可安全摆动 StorageBox.data；graph input 的 storage 身份外部可见，必须用 MutationLayout 生成对原 target 的写回。
4. ReinterpretView 与 NonOwningLayout 都不拥有独立最终 storage，差别是什么？答：前者是读/写已有 storage 的坐标解释；后者作为某个 producer 的输出 layout，表示该 producer 直接写另一 destination view/slice。
5. `require_exact_strides` 已固定 layout 后，`decide_layout` 能否按另一个 consumer 的 stride hints 任意改列/行主序？答：不能；启发式只对仍为 FlexibleLayout 的结果选择，外部精确合同优先。
6. 两个 logical buffers 出现在 operation 列表，为什么不能直接推断运行时有两块独立分配？答：MutationLayout/NonOwningLayout 可能把逻辑结果写到原输入或 destination slice；物理拥有关系必须看 layout/alias 合同。

答案均指出具体 StorageBox、layout、reader 或消费者变化。实际读者反馈尚无。

## 当前恢复点

- K01-R 至 K06-R 完成，Q23 达到第 2 层本篇完成条件。
- Q231 仍在第 3 层待完整源码回修；本篇没有用源码概括替代叶子完成。
- 根队列下一项为同层 Q24。

## 第五轮重新复核

- 范围：物化身份、布局选择、view/alias、mutation/version 四项职责从延迟 Loops 走到 Scheduler 可消费的 operation/layout；运行时物理分配明确留给 wrapper/codegen。
- 前提：正文先用同一 StorageBox 从 Pointwise 变 ComputedBuffer 的过程解释两层包装用途，没有从类名清单起步。
- 过程：output、extern、reuse、旧 reader 四类物化触发；两组 stride hints 的 fill order；ReinterpretView/NonOwningLayout；内部 fast path 与外部输入写回；旧值保护均有具体状态。
- 综合：实验 06 与重排实验 18 分开证明“output 已物化”和“mutation 首次保护旧 reader”，消除了旧稿把相关性写成因果的错误；mutation 的逻辑 buf 名也没有被误写成独立物理分配。
- 理解：只凭正文可判断删除 old 输出、增加跨 mutation reader、精确 stride 已冻结、内部临时修改和输入修改时，哪些 storage/layout/version 决定改变。实际读者反馈仍为空。
