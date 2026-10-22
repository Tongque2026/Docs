# FX 图不变量篇作者记录

## 2026-10-22 独立读者审阅后的结构回修

- 审阅依据：`/work/learning/1021/56sol/inductor-fresh-reader-review/04_GRAPH_INVARIANTS_REVIEW.md`，81/81 个语义单元。
- 正文固定为顺序、node meta、`forward`、结构合法性四项，不再加入 DCE 作为第五项。
- stable sort 与 FakeTensorUpdater 只保留父层可观察变化，内部队列、hash 和传播状态下放两个实现叶子。
- 删除 pre/joint/post 收尾清单；阶段差异归阶段编排篇。
- 综合 transpose 案例用最终表明确交出可由下一 pass 或 GraphLowering 读取的 GraphModule。

## 所在层、输入与出口

- 队列位置：Q13，第 2 层，父项 Q10；第五轮已按技能重新复核，保留“本层完成”。本状态不替代 Q131/Q132 两个叶子的源码回修。
- 读者前置：Q11 已说明三个 pass 处理阶段，Q12 已说明局部 matcher 如何插入、重接和删除节点。
- 本篇输入：局部 pass 修改后的 `GraphModule`/FX `Graph`，其中线性 node 顺序、`meta["val"]`、生成的 `forward`、users 或死节点可能与新依赖不一致。
- 本篇出口：恢复当前阶段需要的拓扑与 fake meta，重新生成可执行 `forward`，用 lint 验证结构；DCE 只按 purity/user 合同删除节点。通过后才交给下一 pass 或 GraphLowering。
- 直接子职责：Q131 负责 stable topological sort；Q132 负责 FakeTensorUpdater 增量更新。`recompile`、`lint`、DCE 是本篇必须讲清的相邻 FX 图合同，但不另建函数级子目录。
- 层次边界：本篇讲四类状态为何不同、怎样协作以及各出口；Q131/Q132 才逐行解释循环、hash、storage/符号分支。

## 篇内知识点队列

| 顺序·编号 | 目标与完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| 1·K01-R | 建立“依赖边”和“线性 node 顺序”两个对象，走通稳定排序入口到合法出口 | `y,add,x` 人工乱序 | 完成，已实测并复读 |
| 2·K02-R | 解释 FakeTensorUpdater 的变更发现、下游传播、相同属性停止、跳过出口 | 改写 add 输入导致 shape 变化；add→sub 等属性对照 | 完成，已实测并复读 |
| 3·K03-R | 解释 Graph 与安装在 GraphModule 上的 Python forward 是两份状态 | add target 改 mul，recompile 前后 | 完成，已实测并复读 |
| 4·K04-R | 区分 lint、修复、DCE；说明非法结构、纯死节点和副作用死节点 | 当前 lint/DCE 源码；sin/copy_ 对照 | 完成，关键路径已实测 |
| 5·K05-R | 比较 pre/joint/post 三个阶段实际采用哪些收尾及顺序原因 | 当前三个 pass 入口 | 完成，已按当前源码核对 |
| 6·K06-R | 综合改写走完 topology→meta→recompile→lint/DCE，并完成迁移题 | transpose 共享 users 推演与 add→sub 对照 | 完成，已复读 |

## 处理情况与案例覆盖

| 情况与条件 | 进入状态 | 本层变化与出口 | 证据 | 当前深度 |
| --- | --- | --- | --- | --- |
| 消费者在线性顺序中早于输入 | 边正确，链表顺序错误 | stable sort 只移动满足依赖所需节点；lint 通过 | 实验 03 | 已实测并讲解 |
| 新增节点或 args/target 改变且 fake 属性变化 | updater 初始 hash 不再命中 | 重算当前 node，更新 meta，将 users 纳入传播 | 实验 03 `[2,3]→[4,3]` | 已实测并讲解 |
| 节点改变但 fake 属性等价 | hash 变化，重算结果满足当前等价合同 | 在该 node 停止，不传播 users | 实验 03 add→sub | 已实测并讲解 |
| 节点不是受支持 call_function 或 fake 参数不完整 | updater 无法安全执行 target | 跳过该 node，meta 不由本 updater 补齐 | 当前源码 | 已讲边界 |
| 只改 Graph node target | Graph 已变，旧 forward 尚在 | recompile 后执行新 target | 实验 03 | 已实测并讲解 |
| lint 遇到非法顺序/所有权/opcode/target | 图合同被破坏 | 报错，不自动修复 | 当前源码；顺序已实测 | 已讲机制 |
| 无 user 的纯节点 | 合法拓扑图 | DCE 逆序删除并更新上游 users | 实验 03 sin | 已实测并讲解 |
| 无 user 的副作用节点 | mutation/其他 impure op | DCE 保留 | 实验 03 copy_ | 已实测并讲解 |
| 三个 pass 阶段 | 阶段改写集合与消费者不同 | 各自按当前实现选择 sort/meta/recompile/lint 等 | pre/joint/post 当前源码 | 已核对并讲解 |

## 依据与边界

- 当前源码：`torch/_inductor/pattern_matcher.py:stable_topological_sort`、`torch/_inductor/fx_utils.py:FakeTensorUpdater`、`torch/fx/graph.py:lint/eliminate_dead_code`，以及三个 FX pass 入口。
- 独立实验 `experiments/03_graph_invariants.py` 证明乱序修复、recompile 前后、fake shape 向 user 传播、相同属性停止传播，以及 DCE 删除纯 sin/保留 copy_。所有权、opcode、target 等其他 lint 分支仍以当前源码为证据，没有伪装成逐项运行。
- Q131/Q132 已有叶子页面，但将在第 3 层按完整源码、原顺序、重要分支和具体状态重新验收；本篇不提前把它们标为叶子完成。
- 实际读者反馈仍为空；本轮只能登记作者复读结果。

## 作者复读与迁移题

1. add 的 args 改接 `[4,3]` 输入，为什么 relu 的 args 容器没改也会重算？答：add 新 fake 与旧 meta 不同，updater 将 add 的 users 放入传播集合；relu 因被点名处理而重算。
2. add 改成同 shape 的 sub，为什么 FakeTensorUpdater 可以不调用 relu，但仍必须 recompile？答：sub 输出的 tensor 属性合同没变，所以 meta 传播停止；实际 Python forward 仍是旧代码，只有 recompile 才改变数值操作。
3. 一个 `copy_` 没有 users，为什么不能据此删除？答：users 只描述数据结果消费；copy_ 的写入副作用仍会改变输入，DCE 的 impurity 判断使它保留。
4. 新 transpose 暂时插在 producer 前，先 recompile 能否修好顺序？答：不能。recompile 按当前 Graph 生成代码，不负责恢复依赖拓扑；应先 stable sort，再进行后续更新与验收。
5. lint 通过是否证明 fake meta 正确、没有无用纯节点？答：不能。lint 检查结构合同，不重新执行 fake target，也不主动 DCE。

答案均要求指出具体状态、消费者和因果，不能只摘抄四个工具名称。实际读者反馈尚无。

## 当前恢复点

- K01-R 至 K06-R 完成，Q13 达到第 2 层本篇完成条件。
- Q10 的三个第 2 层直接职责 Q11/Q12/Q13 均已完成第五轮重新复核。
- 根队列下一项为同层 Q21；Q131/Q132 仍按第 3 层顺序待回修。

## 第五轮重新复核

- 范围：本篇区分依赖拓扑、fake meta、GraphModule `forward`、结构合法性和 users/purity 五类状态；Q131/Q132 只打开排序和 meta 两个直接实现职责，recompile/lint/DCE 作为相邻图合同留在本篇。
- 过程：人工乱序、shape 传播、属性等价停止、target 改变后的旧 forward、纯 sin 与副作用 copy_ 分别展示不同工具实际修复或验证的对象。
- 综合：新增 transpose 并重接两个 users 的案例依次推进 sort→meta→recompile→lint/DCE；add→sub 对照说明 meta 不传播仍不能省略 recompile。
- 失败边界：unsupported call、fake 参数不完整会被 updater 跳过而非“成功补齐”；lint 只报错不修复；DCE 的默认 purity 覆盖也没有被写成完备证明。
- 理解：只凭正文可判断 shape 改变为何传播到未改 args 的 user、属性等价为何停止传播、无 user 的 copy_ 为何保留以及 lint 通过不能证明哪些事实。实际读者反馈仍为空。
