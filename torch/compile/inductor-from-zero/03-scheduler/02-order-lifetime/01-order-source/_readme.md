# Q321 排序、删除与最后使用源码备课记录

第五轮状态：补齐 Scheduler 构造期真实调用顺序，以及 `WeakDep(buf0)` 怎样经 DFS 把旧 reader 排在 mutation 前之后，叶子复核完成；根队列转 Q331。

## 读者、层次与边界

- 读者已从 Q31/Q311 获得 nodes、unmet dependencies、users、WeakDep、mutation rename/real-name；从 Q32 知道排序、DCE、ancestors 和 last usage 的结果。
- 本篇是第 3 层叶子。输入是依赖图和输出/副作用状态；出口是合法节点顺序、删除集合、ancestors/order 以及普通与 fused 节点的 last_usage。
- 本篇不展开融合选择，也不把 last_usage 写成 wrapper 已释放或复用内存；融合属于 Q331，实际内存计划属于 Q431。
- 产品基线：`/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。

## 声明范围

| 产品位置 | 声明 | 职责 |
| --- | --- | --- |
| BaseSchedulerNode | `set_last_usage`、`prune_weak_deps` | 映射真实 storage 后计算末次使用；删除失效弱边 |
| FusedSchedulerNode | `set_last_usage` | 同时计算跨 kernel 与 kernel 内部末次使用 |
| Scheduler | `dead_node_elimination` | 反向删除 dead buffer/node 后裁 weak deps |
| Scheduler | `topological_sort_schedule` | 对给定节点集合做稳定 DFS 拓扑排序 |
| Scheduler | `compute_ancestors` | 递推全部祖先并初始化 order 范围 |
| Scheduler | `compute_last_usage` | 以图输出为 future set 反向传播 |

共 7 个方法，最终须 AST 7/7 等价。每个类内保持产品声明顺序；长函数不省略分支。

## 篇内知识点队列

| 编号 | 完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| K01 | 能手工执行 DFS 排序，区分强制 producer 顺序与稳定但非唯一的兄弟顺序 | 实验 09 three-mm/reverse | 完成，已复读 |
| K02 | 能从反向 DCE 判断 buffer/node/side effect/weak-only user 的保留和裁剪 | 实验 09 controlled DCE；weak-only 源码推演 | 完成，已复读 |
| K03 | 能由直接依赖递推 ancestors，并解释 min/max order 初值及后续消费者 | three-mm + 三层 DAG 推演 | 完成，已复读 |
| K04 | 能反向推进 future set，计算普通/fused last_usage，并把 mutation version 映回真实 storage | three-mm、input mutation、fused 源码 | 完成，已复读 |

## 处理覆盖与恢复点

| 情况 | 关键状态 | 出口 | 归属 |
| --- | --- | --- | --- |
| 单链/分叉/反向候选 | seen、递归 producer、result | 合法且稳定的拓扑序 | K01 |
| mutation 旧读 / 旧 reader 被删除 | WeakDep 先参与 DFS；DCE 后按 removed operation 裁边 | 覆盖晚于活旧读；死旧读不再制造顺序 | K01/K02 |
| 集合外依赖 | dep 不在 name_to_node | 保持为外部已知条件，不递归 | K01 |
| dead chain/side effect/weak-only | active_buffers、removed sets | live nodes + weak dep pruning | K02 |
| 多级依赖 | 直接 producer 与其 ancestors 合并 | ancestors/min/max | K03 |
| graph output/shared read/mutation | future_used_buffers 与 real-name 映射 | node.last_usage | K04 |
| fused group | 全局 future 与组内空 future 两次反向 | 跨 kernel/组内 last usage | K04 |

K01–K04 均完成，Q321 叶子完成。最终审计以“按产品声明顺序连读七个方法”为范围：顺序为 BaseSchedulerNode `set_last_usage→prune_weak_deps`、FusedSchedulerNode `set_last_usage`、Scheduler `dead_node_elimination→topological_sort_schedule→compute_ancestors→compute_last_usage`，AST 7/7 等价；实验 09 重跑通过 three-mm、input/double mutation、controlled DCE；3 个导航链接存在。事实与理解分别验收，没有把 last_usage 扩大成 wrapper 已释放/复用。下一项转 Q331，先建立融合主循环叶子的声明范围与篇内计划。

## 第五轮叶子验收

- **声明完整性：** 7 个方法按所属类中的产品顺序完整出现，去除固定教学注释后 AST 7/7 等价；嵌套的 `visit`、`can_eliminate_user` 与 `should_prune` 均未省略。
- **真实阶段顺序：** 正文已经连接依赖构造、首次拓扑排序、反向 DCE/弱边裁剪、ancestors、组合节点后的再次排序、融合后 last usage；不会把每个函数误讲成互不相干的工具。
- **分支与状态：** 覆盖单链、分叉、逆序候选、集合外依赖、mutation WeakDep、dead chain、弱 user、副作用、多输出、多级祖先、graph output、mutation real-name 和 fused 内外两级 last usage。three-mm 的 `future` 表、controlled DCE 的 removed 集合和逆序 DFS 均给出逐步状态。
- **作者迁移复读：** 逆序 `[op1,op0]` 中 WeakDep 仍得到 `[op0,op1]`；删除 op0 后弱边被裁而真实 MemoryDep 不会；三层 DAG 的后继取得传递祖先闭包；交换两个合法 sibling 会把共享 buf 的 last usage 交给新的较晚 consumer；图输出不会成为内部释放候选。答案均可仅凭正文推出。
- **证据边界：** three-mm、input/double mutation 与 controlled DCE 来自实验 09；fused 内层 future、多输出和弱边裁剪的若干对照来自当前源码推演。实际读者尚未作答，只登记作者复读。
