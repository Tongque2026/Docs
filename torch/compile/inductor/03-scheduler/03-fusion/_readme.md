# Scheduler 融合篇作者记录

## 第六轮：独立初学者逐篇审阅整改

- 本轮只审阅本篇 70/70 个语义单元；报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/14_FUSION_REVIEW.md`。
- 入口明确为 topo/DCE 后仍存活、已有 ancestors 与 order 范围的 scheduler nodes。
- 横向案例补全 op0–op3 的 reads/writes/ancestors，从两条纵向边和共同读取 x 推出横向候选。
- 教学分类拆成正确性/后端可生成条件与收益/资源策略；源码 `can_fuse()` 可以同时实现两组判断，但正文不再把它们统称数学合法性。
- 测试脚手架、实验日志口吻和反向辩护移出读者正文；待同一读者只复查本篇。
- 复查覆盖 71/71，S0/S1/S2 均为 0；本篇通过后才可进入后端分派篇。

## 本篇位置与教学目标

- 全系列编号：Q33；层次 2；父项 Q30；前置 Q31、Q32；同层下一项 Q34。
- 读者已知：会读调度节点的读写依赖，知道祖先关系决定拓扑顺序；不知道“共享数据”为何还不足以融合，也不知道候选怎样在一轮内变成不断扩大的融合组。
- 入口：Q32 交出的、已满足依赖顺序的调度节点列表。
- 出口：仍保持合法拓扑顺序的 `SchedulerNode / FusedSchedulerNode` 列表，交给 Q34 按设备后端生成。本篇不展开 kernel 内部代码生成。
- 学完标准：能从两个节点的依赖、精确索引、循环组、原顺序和类型判断纵向/横向候选，指出拒绝门次，并沿一次候选序列更新当前融合组。
- 资料边界：只用 `/work/pytorch` 提交 `8332c5622aabd9535969e9a8d0ed60e4a90e4402`、当前测试与独立 CPU 实验；未运行 GPU benchmark fusion。
- 第五轮状态：已按技能重新复核，保留“本层完成”；Q331 叶子仍未完成 K04，不能由本篇状态代替。

## 本层职责与下一层边界

1. 候选发现：按 `used_buffer_names()` 聚集可能共享数据的节点；aggressive 模式才额外按循环 `group` 聚集。
2. 静态选择：`can_fuse()` 检查方向、类型、设备、共享精确依赖、规模及纵向/横向条件；backend 再检查目标循环组合。
3. 冲突处理：backend priority 先筛组，`score_fusion()` 再排序；执行每对前映射到当前融合组，重查静态合法性和成环风险。
4. 合并与收尾：可选 benchmark 收益检查通过后，由 backend 构造融合节点；一轮末重排、拓扑排序并删冗余依赖，外层最多重复十轮。

Q331 叶子以后逐句讲这些函数。本篇只展开理解候选流转所需的字段和状态。

## 处理情况覆盖

| 情况 | 进入状态与关键变化 | 本层出口 | 证据 | 深度/状态 |
| --- | --- | --- | --- | --- |
| 纵向生产者→消费者 | B 的祖先含 A；B 的读与 A 的写精确匹配 | A、B 成组 | vertical 2→1 | 机制已讲解并复读 |
| 横向共同读取 | 两边互不依赖，共读 x；检查距离与 backend 能力 | 两分支可成组 | horizontal 4→1 | 机制已讲解并复读 |
| 多候选冲突 | priority、score 排序；旧成员名映射到当前组 | 融合组逐次扩大 | branched 四次 merge | 机制已讲解并复读 |
| 索引错配 | 同 buffer 名但精确 `MemoryDep` 不同 | 保留两节点 | roll：`no shared data` | 边界已讲解并复读 |
| 调用/规模边界 | 普通 extern 或成员数超限 | 保留原节点 | mm+sin；size=1 | 边界已讲解并复读 |
| template/foreach 方向 | 正向失败时只对这两类尝试反向；template 只接 epilogue | 合法方向或拒绝 | 当前源码 | 条件已讲，未声称实测 |
| aggressive 无共享数据 | 同 `group` 的非规约节点可越过无共享数据启发式 | 继续 backend 判断 | 当前源码 | 条件已讲，未声称收益 |
| 成环风险 | 先前融合引入组合祖先关系；执行候选前 DFS | 拒绝该对 | 当前源码 | 作用与状态已讲解 |
| benchmark 收益 | 静态合法且不成环；CPU 当前直接允许 | 接受或保留 | 当前源码；实验关闭 | 事实边界已讲解 |

未运行覆盖 device mismatch、距离 >64、mutation/weak dep、template、foreach、cycle、GPU 寄存器溢出；正文只按源码解释条件。

## 篇内知识点队列

| 编号 | 目标与完成条件 | 案例/正文 | 状态 |
| --- | --- | --- | --- |
| K01 | 建立节点、融合组、成员名映射和出口；说明融合不是改 FX 图 | 开篇与四类信息 | 完成 |
| K02 | 从祖先关系区分纵向/横向，走完 vertical | 纵向融合 | 完成 |
| K03 | 解释共读 x 的横向候选和同轮三次 merge | horizontal 状态表 | 完成 |
| K04 | 按候选聚集、priority、score 说明 branched 次序 | 候选与排序 | 完成 |
| K05 | 用精确索引解释 roll 拒绝，而非只看 shape/名字 | shifted index | 完成 |
| K06 | 将 extern、size、template、cycle、benchmark 放到正确门次 | 边界与总流程 | 完成 |
| K07 | 只依赖正文完成迁移题并核对证据边界 | 用新案例判断 | 完成 |

## 独立实验与案例事实

- 脚本：`../../experiments/10_scheduler_fusion.py`；输出：`../../experiments/artifacts/scheduler_fusion.txt`。
- CPU，`force_disable_caches=True`、`benchmark_fusion=False`；`_inductor_test.realize` 只用于保留可观察节点。
- 采集：`get_possible_fusions` 记录候选/score/priority；`CppScheduling.fuse` 记录每次实际 merge；`fuse_nodes_once` 记录整轮前后；`WhyNoFuse` 记录拒绝；`compute_last_usage` 前记录最终节点。

| 案例 | 实际过程 | 实际出口 |
| --- | --- | --- |
| vertical | `(op0,op1)`，score `(True,True,32,-1)`；一次 merge | 一个二成员组 |
| horizontal | `op0+op1`、`op2+op3`，再合并当前两个二成员组 | 一个四成员组 |
| branched | 五个候选，实际四次 merge | 一个五成员组 |
| shifted index | 拒绝 `no shared data` | 两节点 |
| extern | 拒绝 `node1 is extern or nop` | 两节点 |
| max size | 拒绝 `exceeds max fusion` | 两节点 |

`32` 是 8×float32 精确公共依赖的字节估计，不是固定权重。数值结果均与 eager 核对；没有测量性能。

## 当前源码依据与关键对应

- `torch/_inductor/scheduler.py`：`fuse_nodes`、`fuse_nodes_once`、`get_possible_fusions`、`can_fuse`、`can_fuse_vertical`、`will_fusion_create_cycle`、`score_fusion*`、`speedup_by_fusion`。
- `torch/_inductor/codegen/cpp.py`：`CppScheduling._why_fuse_nodes`、横纵融合检查、priority、`fuse`。
- `test/inductor/test_torchinductor.py`：horizontal/vertical、forced realize、`max_fusion_size`；`test_benchmark_fusion.py`：benchmark 范围。

候选生成先执行一次 `can_fuse`；执行候选时用 `name_to_fused_node` 映射当前组，再执行 `can_fuse` 和独立的 `will_fusion_create_cycle`，随后才是 `speedup_by_fusion` 与 backend `fuse`。所以成环检查不属于 `can_fuse()` 内部。

## 作者复读：事实与理解分别验收

事实：新增逐 merge 采集后脚本重跑成功；vertical 2→1、horizontal 4→1、branched 5→1，三项拒绝原因均由输出支持。template、cycle、GPU benchmark 只由源码支持，未写成运行结论。

只依赖正文回答：

1. `a=realize(x+1); b=a*2` 是纵向，因为 b 的祖先含 a，且未满足读可与 a 的写按名字、索引、范围匹配。
2. 两分支共读 x 只使它们进入候选；仍需距离、成员数、backend、成环和收益检查。
3. horizontal 第三对保存原节点名，但执行前映射到已形成的两个二成员组，因此一次合并四个成员。
4. roll 改成 `producer*3` 后，移位读取变为同位置读取，共同精确依赖出现。
5. 静态合法不等于最终融合；还须不成环、收益策略接受并由 backend 合并。

作者复读通过；实际读者反馈尚无。

## 续写位置

Q33 第五轮重新复核完成。根队列转到同层 Q34；Q331 叶子仍按广度优先顺序留到下一层。

## 第五轮重新复核

- 范围：候选发现、静态选择、priority/score 冲突处理、当前组映射/cycle/收益、backend merge 与多轮收尾形成完整决定链；kernel 内部生成留给后端篇。
- 过程：vertical、horizontal、branched 三个接受例和 shifted-index、extern、max-size 三个拒绝例都展示候选、关键状态和最终节点组；template/aggressive/cycle/benchmark 保留源码证据边界。
- 因果：共同 buffer 名只缩小搜索范围，精确 Dep/后端能力证明合法性，score 只处理合法候选冲突，不能越过拒绝门；静态合法后仍须 cycle 与收益策略接受。
- 事实：实验 10 关闭 benchmark，只支持候选、merge 与拒绝原因，不支持性能提升；GPU benchmark fusion 未运行。
- 理解：只凭正文可判断 roll 改为同位置读取、提高 max size、把 extern 换普通 pointwise、前一 merge 扩大当前组时，哪些门次需重新检查。实际读者反馈仍为空。
