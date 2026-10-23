# Q331 融合主循环源码备课记录

## 第六轮：隔离冷读与源码一致性回修

- 第一名无历史读者只读正文，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/113_FUSION_SOURCE_COLD_REVIEW_1.md`，评级 B。主循环成立，但对象关系、branched 初始读写、WeakDep 方向、loop reorder 契约、template/aggressive 分支、cycle 手算和 backend 新组交接均有实质缺口。
- 实验 10 增加每个当前组的 buffers、used buffer、精确 reads/writes、unmet dependencies 和 ancestors 捕获，并重新运行通过。正文据此加入 branched 五节点状态表、vertical 融合前后 unmet dependency 和 backend 聚合组状态，不再从候选日志反推字段。
- 正文开头先建立原 operation、buffer、当前组、ancestor、unmet dependency 与旧候选端点；修正 K01 的停止时点和十轮上限用途；补齐 score 第一项 truth table、三类 buffer-name 口径和同分候选发现顺序。
- WeakDep 案例改为严格对应源码：weak name 是 node1 输出，mutating buffer 由 node2 写，real name 用于筛 node1 的旧值 reads。第二名读者报告 `/work/learning/1021/56sol/inductor-fresh-reader-review/117_FUSION_SOURCE_COLD_REVIEW_2.md` 进一步发现 `all([])==True`；正文现明确 relevant reads 为空时当前实现也放行，不再把正例的一项旧读误写成必要条件。
- cycle 增加 `X→A`、`B→Y` 且第三方已融合为 `{X,Y}` 的完整手算：`combined_names={A,B}`、`combined_ancestors={X}`，沿 X 映射到 `{X,Y}` 后命中其 ancestor B。裁边增加 `op1.unmet={buf0}` 到融合组 unmet 为空的前后状态。
- 第二名读者还发现 multi-template 示例把未融合 choice timing 与另测 fused timing 混写。正文现区分 `choice_timings[choice]`、单独测得的 `ms2` 与逐 choice 测得的 `ms_fused`；普通 GPU 例单独使用 `ms1/ms2/ms_fused`。
- 第三名无历史读者只读修订稿，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/118_FUSION_SOURCE_COLD_REVIEW_3.md`，评级 A。读者能闭卷重建 branched grouping/score、template 翻向、aggressive 无共享路径、WeakDep 三个出口、cycle、旧候选映射、新组聚合与裁边、两种 timing 及三类外层停止。
- 本轮事实证据为当前 Scheduler 源码与扩充后的实验 10；cycle、WeakDep 空集合和 multi-template 分支按当前源码推演，未冒充实验实测。Q331 隔离冷读闭环，整改队列进入 Q341。

第五轮状态：旧恢复点 K04 已完成。补入原先缺失的 `speedup_by_fusion`、`fuse_nodes_once`、`will_fusion_create_cycle` 三段完整源码，修正七段 docstring 的 AST 差异，并完成 branched 当前组状态推演后，Q331 叶子复核完成；根队列转 Q341。

## 读者、层次与边界

- 读者已从 Q33 理解纵向/横向、候选、score、静态拒绝、cycle 与 benchmark 门次；尚不知道这些判断在循环中以什么顺序执行、一次 merge 后旧候选怎样映射到新融合组。
- 本篇是第 3 层叶子。输入是 Q321 交出的合法节点序列、依赖/祖先/order 和 backend；出口是重新拓扑排序并裁冗余边的 SchedulerNode/FusedSchedulerNode 列表。
- 本篇不展开 CPU/Triton backend 如何生成融合 kernel，也不把静态融合成功写成性能提升；后端生成属于 Q421，wrapper/运行属于 Q431/Q441。
- 产品基线：`/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。

## 声明范围

| 产品顺序 | 声明 | 职责 |
| --- | --- | --- |
| 1 | `Scheduler.fuse_nodes` | 最多十轮，直到节点数不变或只剩一个 |
| 2 | `Scheduler.speedup_by_fusion` | CPU/配置/GPU benchmark 收益门 |
| 3 | `Scheduler.fuse_nodes_once` | 当前组映射、复查、cycle、收益、backend merge 与收尾 |
| 4 | `Scheduler.get_possible_fusions` | 按共享 buffer/可选 group 聚集、去重、方向、priority/score 排序 |
| 5 | `Scheduler.will_fusion_create_cycle` | 检查合并祖先是否引入回边 |
| 6 | `Scheduler.can_fuse` | 类型、设备、数据、规模与纵横方向静态门 |
| 7 | `Scheduler.can_fuse_vertical` | producer-consumer 精确依赖与 backend 判断 |
| 8 | `Scheduler.fusable_weak_dep/fusable_read_and_write` | 判断纵向 weak 与精确读写是否可在同一循环满足 |
| 9 | `Scheduler.score_fusion` | 模板、纵向、共享字节、距离排序键 |
| 10 | `Scheduler.dep_size_hint/score_fusion_memory` | 估算共同精确依赖可省的字节 |
| 11 | `Scheduler.get_possible_fusions_with_highest_priority` | 仅保留 backend priority 最优组 |
| 12 | `Scheduler.score_fusion_key` | 将 pair 接到列表排序接口 |

共 14 个方法；最终按产品物理顺序建立连续源码区，AST 14/14 等价。`backend.fuse/can_fuse_vertical/can_fuse_horizontal` 是消费者/扩展点，不纳入本叶子完整声明范围；loop-reorder shared-data 与 peak-memory 是独立启发式接口，本篇解释它们在门次中的作用，不继续递归扩张源码范围。

## 篇内知识点队列

| 编号 | 完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| K01 | 能推演外层最多十轮、停止条件和每轮输入输出，解释为何一次扫描后仍可能继续 | vertical/horizontal/branched + 源码 | 完成，已复读 |
| K02 | 能从共享 buffer/group 建候选、处理 template/foreach 反向并按 priority/score 排序 | horizontal/branched | 完成，已复读 |
| K03 | 能逐门判断 can_fuse 的类型、设备、共享精确依赖、规模和纵横条件 | vertical/shifted/extern/max-size | 完成，已复读 |
| K04 | 能推演候选映射到当前融合组、cycle/收益复查、backend merge 与一轮收尾 | branched 四次 merge | 完成，已复读 |

## 处理覆盖与恢复点

| 条件 | 状态变化 | 出口 | 归属 |
| --- | --- | --- | --- |
| 一轮有 merge / 无 merge / 剩一个 | old_len/new_len | 继续或停止，最多十轮 | K01 |
| 共享 buffer / aggressive 同 group | 分组内两两配对、seen 去重 | 候选表 | K02 |
| template/foreach 方向 | 正向失败后仅这两类尝试反向 | 定向候选或拒绝 | K02 |
| 纵向/横向/索引错配/extern/规模 | 静态门依次执行 | 可融合或 WhyNoFuse | K03 |
| 旧候选成员已进入融合组 | name_to_fused_node 映射当前 node | 扩大的融合对 | K04 |
| cycle / benchmark 拒绝 | 不改 fused_nodes | 保留原组 | K04 |
| merge 接受 | 移两项、加新组、更新成员映射 | 重排、拓扑、裁边 | K04 |

K01–K04 已完成。实验 10 实际捕获 vertical/horizontal/branched merge、shifted-index `no shared data`、extern 类型拒绝和 max-size 拒绝；template、device、weak mutation、peak-memory、cycle 与 GPU benchmark 边支限定为当前源码。理解验收能由具体 Dep 判断 computed/remaining、定位首个拒绝门，并把每个旧候选解析到当时的融合组。

## 第五轮叶子验收

- **声明完整性：** 正文中的 14 个目标方法各恰有一个代码块定义；用当前 Python 导入的 `Scheduler` 方法源码逐一比较，去除教学注释后 AST 14/14 等价。旧稿七个类内 docstring 多保留了四格缩进，本轮已修正，未继续沿用错误的“等价”记录。
- **分支覆盖：** 外层覆盖固定点、单节点与十轮上限；候选覆盖共享名、aggressive group、去重、template/foreach 反向、backend priority 与 score；静态门覆盖 grouped/extern/template/no-fuse/device/shared/max-size/纵横向；一轮执行覆盖当前组映射、同组拒绝、cycle、CPU/atomic/template/multi-template/GPU benchmark、backend merge、重排与冗余边裁剪。
- **具体状态：** branched 候选从五个单节点依次变为 `op0_op1`、`op0_op1_op2`、`op3_op4`、五成员组，最后一个旧候选映成同组对自身；shifted/extern/max-size 各停在不同首个拒绝门；32-byte memory score 与距离分量均有实际候选记录。
- **作者迁移复读：** 旧 A+C 合法不保证映射后的 AB+C 仍合法；benchmark 的严格 `<` 使相等耗时拒绝；priority 1 候选不会留在本轮 priority 0 列表尾部；十轮耗尽不等于固定点；cycle/收益拒绝均不修改当前组。答案均可只凭正文推出。
- **证据边界：** 实验 10 为 CPU 且关闭 benchmark，支持候选、静态门和 merge 状态；cycle、multi-template、GPU timing、atomic-add 放行等结论来自当前源码逐句推演，未冒充实测。实际读者尚未作答。

下一项 Q341：先在 `03-scheduler/04-backend-dispatch/01-dispatch-source/` 建立叶子计划，再解释 backend 选择、调度循环、kernel 定义和 wrapper 调用的交接。
