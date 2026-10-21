# Q431 wrapper 复用规划完整源码备课记录

## 读者、层次与边界

- 读者已从 Q43 能沿两次 mm 案例区分逻辑 buffer、Python 名和 storage，也知道默认 reuse planner 与 inference pool planner 的外部结果；尚不知道 planning line 怎样扫描、live range 怎样编号、时间/空间树怎样放置 allocation。
- 本篇是第 3 层叶子。入口是 Scheduler 顺序写入的普通 wrapper 行、`AllocateLine`、`FreeIfNotReusedLine`、`ReuseLine` 以及图输出集合；出口是已经决定普通复用或 pool offset、可以逐行 codegen 的 wrapper 时间线。
- allocator 内部、kernel 执行和 cache/load 不属于本篇。Python/C++ wrapper 的字符串语法差异只在 `make_buffer_*` 消费端解释，规划算法共享同一组 line/buffer 对象。
- 依据限定为 `/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`、现有测试和本目录实验 14；不使用旧教程。

## 声明范围

本叶子按两个闭合职责保留完整源码：

1. **默认 LIFO 复用**：`buffer_reuse_key`，`MemoryPlanningState` 的 `__init__/__contains__/pop/push`，`MemoryPlanningLine.plan/codegen/__str__`，`AllocateLine.plan/codegen`，`FreeIfNotReusedLine.plan/codegen`，`ReuseLine.plan/codegen`，`WrapperCodeGen.memory_plan_reuse`、`make_buffer_reuse`、`codegen_allocation`、`codegen_free`、`can_reuse`、`codegen_inplace_reuse`。
2. **inference pool 规划**：完整保留 `LiveRange`、`LiveRanges`、`AllocationTreeNode`、`Allocation`、`Empty`、`ClearCacheOnAllocateMixin`、`TemporalSplit`、`SpatialSplit`、`AllocationPool`、`AllocationPools`、`BufferGroup`、`PoolMemoryPlanningLine`、`AllocFromPoolLine`、`DeallocFromPoolLine` 和 `MemoryPlanner` 的全部方法。

共 87 个函数/方法：默认 planner 20 个，pool planning 相关 15 个类共 67 个。原 49 项范围漏掉了分配树实际调用的 `Allocation.mark_allocated/finalize`、空闲节点、缓存失效和聚合 getter；K05 研究时扩为完整类，避免正文展示 `_allocate` 却隐藏它依赖的状态更新。它们共同覆盖“产生 planning line—默认复用”与“转换成 pool line—计算生存期—时间/空间放置—生成 view”两条完整出口。

## 篇内知识点队列

| 编号 | 完成条件 | 案例与状态 | 状态 |
| --- | --- | --- | --- |
| K01 | 建立 planning line、reuse key、LIFO pool 与两遍扫描的用途，走完 A 释放后被 C 接管 | 三个同规格 buffer 教学状态 + 实验 14 对照 | 完成，已复读 |
| K02 | 完整讲 allocation/free/inplace 入口的身份过滤、removed/alias/multi-output 与 exact/reinterpret 出口 | 输入、图输出、不同 stride 对照 | 完成，已复读 |
| K03 | 完整讲默认 planner 声明范围的第一组源码并完成具体状态推演 | 两次 mm 默认 wrapper | 完成，20/20 AST，已复读 |
| K04 | 建立 BufferGroup 与离散 live-range 时间，解释普通行怎样形成时间边界 | buf0/buf1、buf2/buf3 | 完成，已复读 |
| K05 | 完整推演 TemporalSplit/SpatialSplit 的无重叠复用、重叠分区、扩容与 offset | A=64/[1,4)、B=32/[4,6)、C=48/[2,5) | 完成，52/52 AST，已复读 |
| K06 | 完整讲 pool planner pass、pool 首末使用和输出保活源码 | 实验 14 pool wrapper | 完成，15/15 AST，已复读 |
| K07 | 87/87 AST、实验、链接和正文迁移题验收，更新根队列 | 完成审计 | 完成，已复读 |

## 处理情况覆盖

| 情况 | 关键状态/出口 | 正文安排 |
| --- | --- | --- |
| removed buffer | planning line 变 `NullLine` | K02/K06 |
| 普通 free 后同 key allocate | free 压栈、allocate 弹最近项、两行分别变“已复用”和 `ReuseLine` | K01/K03 |
| alias output / MultiOutput | free 不进入普通 reuse pool | K02 |
| 输入、常量、输出/never-reuse 身份 | `can_reuse` 或输出生存期阻止错误接管 | K02/K04 |
| 同容量同布局 / 异布局 | exact alias / reinterpret view | K02/K03 |
| subgraph 嵌套 | 独立 `MemoryPlanningState`，退出后不与外层栈混用 | K03 |
| inplace 多名字 | 一个 `BufferGroup` 合并 names | K04/K06 |
| 生存期不重叠 | `TemporalSplit` 共用 offset | K05 |
| 生存期重叠 | `SpatialSplit` 安排相邻 offset | K05 |
| 输出 | live range 延至末尾，并按 pool 配置单独或合并放置 | K04/K06 |
| 动态 symbolic bytes | hint 用于选择，symbolic size 决定运行时 pool/view | K05/K06 |

## 证据边界与当前恢复点

- 实验 14 实际证明两次 mm pipeline 的默认 exact reuse、pool 分组、输出保活及结果正确；不证明异布局 reinterpret、非零 offset 或动态 symbolic pool。
- 本轮重跑实验 14：默认与 pool 两条结果均 `matches_eager=True`；默认产物仍为两次真实 allocation 加两条显式 inplace reuse，pool 产物仍为 `pool1` 中间组与 `pool0` 输出组。该观测已与 K03 的 line 类型边界对应。
- 后三项采用当前源码和独立小状态推演；若增加实验，须观察 planning line、group/live range 或实际 wrapper，不能只看最终数值。
- K01–K07 全部完成。根队列下一项 Q441。
- 实际读者反馈尚无；完成状态只记录作者复读。

## 第五轮事实与理解验收

- **源码范围**：正文 Python fence 无语法错误；默认 planner 20/20、pool planning 67/67，合计 87/87 方法 AST 与 commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402` 相同。教学标识注释不改变语义。
- **实验**：重跑 `experiments/14_wrapper_memory.py`，默认与 pool 结果均和 eager 一致；实际 wrapper 仍为两次 allocation、两条显式 inplace reuse，pool wrapper 仍为 pool1 中间组、pool0 输出组。实验未覆盖非零 offset，正文已明确将 112-byte tree 标为源码推演。
- **结构与导航**：本篇 44 个 fence 平衡；全教程检查 159 个 Markdown 链接。简易正则仅把 Q212 正文中的两处函数签名 `(*args, **kwargs)` 误认成链接，实际本地导航无断链。
- **作者迁移 1**：B live 从 `[4,6)` 改为 `[3,6)` 后与 A 重叠，必须由 temporal 共址改为空间分支；因果由半开区间 overlap 决定。
- **作者迁移 2**：G3 加入 pool1 且结束更晚，G1 dealloc 不能销毁 pool；反向标记必须等 aggregate end 到达。G3 若为输出，生存期延到规划末尾。
- **作者迁移 3**：同容量异 stride 只改变 reuse 的 tensor view；dtype 或符号 capacity 改变则 key/断言更早阻止复用。
- **作者迁移 4**：函数额外返回 first 时，不能保留覆盖 buf0 的 inplace ReuseLine；group names、live range 和 allocation tree 都必须随输出合同重建。

以上为作者只依据正文完成的复读，不代表已经取得实际读者反馈。
