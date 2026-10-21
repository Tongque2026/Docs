# Scheduler 排序与生存期篇作者记录

## 所在层、输入与出口

- 队列位置：Q32，第 2 层，父项 Q30；第五轮已按技能重新复核，保留“本层完成”。本状态不替代 Q321 叶子的源码回修。
- 读者前置：Q31 已得到 nodes、unmet deps、users、OUTPUT、mutation rename/real-name。
- 本篇输入：依赖图与候选节点顺序、side-effect/weak user 标记、图输出、mutation versions。
- 本篇出口：合法拓扑序、删除后的 live nodes/removed sets、ancestors 与原始 order 范围，以及融合后每节点 last_usage；Q33 使用 order/ancestors，Q34/codegen 使用 last_usage/free 集合。
- 直接职责：拓扑排序、反向 DCE/弱边裁剪、祖先与 order、反向 last usage 四项协作。Q321 下一层叶子逐行解释排序与 last-usage 核心源码。
- 边界：本篇计算“何时最后使用”，不声称 wrapper 已执行释放；buffer 物理复用归 Q43。融合前后 last usage 的重新计算按 Scheduler 实际顺序解释。

## 篇内知识点队列

| 顺序·编号 | 目标与完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| 1·K01-R | 用分叉 DAG 与反向候选证明拓扑必要顺序及兄弟非唯一顺序 | three-mm | 完成，已实测并复读 |
| 2·K02-R | 解释 rename/real-name 在排序、last usage 和真实 storage 中的不同方向 | input mutation/double mutation | 完成，已按优化后 IR 修正 |
| 3·K03-R | 从叶向根走通 DCE、side effect 节点/无用输出、weak users/prune | controlled DCE | 完成，局部当前方法证据 |
| 4·K04-R | 从直接依赖递推全部 ancestors，并说明 min/max order 的消费者 | three-mm + 延长链迁移 | 完成，已实测并复读 |
| 5·K05-R | 从 outputs 初始 future set 反向计算 last_usage，处理共享 buffer 与 mutation real name | three-mm/input mutation | 完成，已实测并复读 |
| 6·K06-R | 区分 last_usage、free_buffers、kernel-local removal 和 wrapper reuse，并完成综合迁移 | 当前源码 + 分叉图 | 完成，已讲边界并复读 |

## 处理情况与案例覆盖

| 情况与条件 | 状态变化 | 本层出口 | 证据 | 当前深度 |
| --- | --- | --- | --- | --- |
| 单链/分叉 DAG | producer 递归先放入结果 | 所有依赖满足；兄弟顺序随候选稳定变化 | 实验 09 three-mm/reverse | 已实测并讲解 |
| mutation weak edge | mutation 必须晚于旧 reader | 反转候选仍恢复 op0→op1 | 实验 09 mutation | 已实测并讲解 |
| 纯 dead chain | leaf 无有效 users，删除后 parent 也死 | operations/buffers 进入 removed sets | controlled DCE | 已调用当前方法并讲解 |
| side-effect node 的无用输出 | 节点必须执行但 output 无 consumer | 保留 node，output buffer 可 removed | controlled DCE | 已调用当前方法并讲解 |
| weak-only user | 不独自使 producer live | 删除后 survivor prune weak dep | 当前源码 | 已讲机制，未单列运行 |
| ancestors/order | 当前节点合并直接 producer 与 producer ancestors | 查询集合与 min/max | three-mm | 已实测并讲解 |
| shared intermediate | 多个后继使用同一 buf | 排在更后的实际 user 持有 last_usage | three-mm buf0 | 已实测并讲解 |
| graph output | 跨函数边界仍存活 | 初始 future-used 保护，不进入内部 last_usage | three-mm buf1/buf2 | 已实测并讲解 |
| mutation logical version | buf version 映回原 storage | last_usage 使用 arg0_1，不释放逻辑 buf1 | input mutation | 已实测并讲解 |
| Python 连续两次 copy_ | 上游优化可能消除中间写回版本 | 当前只保留最后 mutation；middle 直接读 y | double mutation | 已实测并讲明边界 |

## 依据与边界

- 当前源码：`scheduler.py` 的 init 阶段顺序、topological_sort_schedule、dead_node_elimination、compute_ancestors、compute_last_usage、free_buffers/remove_kernel_local_buffers。
- 实验 09 的 reverse probe 只额外调用纯排序函数；实际 Scheduler 仍使用正常候选顺序。
- controlled DCE 使用最小 duck-typed nodes，但直接调用当前 Scheduler 方法；证明方法局部行为，不冒充完整 torch.compile 图。
- double mutation 实际只有最后 copy_ 留作 mutation operation；不能由 Python 两次写推断两级 mutation chain。
- last_usage 已实测；wrapper free/reuse 未由本实验捕获，不能写成运行时已释放。
- 实际读者反馈尚无。

## 作者复读与迁移题

1. reverse probe 为什么仍必须以 op0 开头，却可交换 op1/op2？答：两者都依赖 buf0 producer op0，彼此无边；拓扑只固定必要先后。
2. double mutation 为什么版本表只有 arg0_1→buf2？答：上游已让 middle 直接读 y，并消除第一次独立写回；Scheduler 只看优化后的三个 operations。
3. side-effect node 保留时，其无用 output 为什么仍可加入 removed_buffers？答：执行副作用与分配无人读取的结果是两个决定；DCE 分别判断 active buffer 和 node side effects。
4. 图输出为何不出现在内部 last_usage？答：future_used_buffers 初始含所有 output names，表示它们离开函数后仍存活。
5. three-mm 的 buf0 为什么归 op2 而不是 op1 的 last_usage？答：反向顺序先遇到 op2；op1 再看到 buf0 已被未来使用，只有更后的实际 consumer op2 结束它的生存期。
6. mutation 例为什么 last_usage 写 arg0_1 而不是 buf1？答：set_last_usage 先通过 mutation_real_name 将逻辑版本映回真实输入 storage。
7. last_usage 出现能否证明 wrapper 已释放或复用该地址？答：不能；还需 codegen 累计、flush、can_free、别名/mutation/must-keep 和 wrapper 计划。
8. 一个 fused-kernel 内部 buffer 何时可从全局参数移除？答：所有非弱 users 都在同一 fused group，且不属于 must_keep、kernel input、mutation rename/real-name 等排除集合。

答案均指出依赖、反向状态集合或实际 storage 映射。实际读者反馈尚无。

## 当前恢复点

- K01-R 至 K06-R 完成，Q32 达到第 2 层本篇完成条件。
- wrapper 实际 free/reuse 保留给 Q43；本篇未用 last_usage 代替运行时证据。
- Q321 保留第 3 层待回修；根队列下一项为同层 Q33。

## 第五轮重新复核

- 范围：拓扑排序、反向 DCE/弱边裁剪、ancestors/order、反向 last usage 四项职责都使用 Q31 的依赖图，并分别交给融合合法性和生成释放；物理复用明确留给 Q43。
- 过程：three-mm 的正常/反转合法顺序、controlled DCE、单/双 mutation、共享 buf0 的反向生存期均有逐节点状态；图输出从 future set 起始就被保护。
- 事实：reverse probe 只证明排序方法局部行为；controlled DCE 使用当前方法和最小节点；double mutation 按实际优化后 operations 解释，没有从 Python 两次 copy_ 猜两级版本。
- 边界：last_usage 只产生可释放计划，不能证明 wrapper 已释放或复用；kernel-local buffer 删除也列出 must-keep、输入和 mutation 排除条件。
- 理解：只凭正文可推算分叉兄弟换序后的 last usage、dead leaf 变 output、增加第三个 consumer、alias reader 和副作用节点对生存期的不同影响。实际读者反馈仍为空。
