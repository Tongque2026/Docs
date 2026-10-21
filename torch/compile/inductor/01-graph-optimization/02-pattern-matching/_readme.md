# Pattern Matcher 篇作者记录

## 所在层、输入与出口

- 队列位置：Q12，第 2 层，父项 Q10；第五轮已按技能重新复核，保留“本层完成”。本状态不替代 Q121/Q122 两个叶子的源码回修。
- 读者前置：已从 Q10/Q11 理解 FX node、结果引用、三组 pass 所处的处理阶段，以及一次局部改写必须恢复图不变量。
- 本篇输入：一份 `GraphModule`，按 `(node.op, node.target)` 建索引的 pattern entries，以及节点的 meta、user 和 mutation-region 信息。
- 本篇出口：未命中时原图不变并返回 0；命中时 entry 执行局部替换或 handler，重接外部 users，只删除已无 users 的旧节点，并返回应用次数。稳定拓扑、fake meta、DCE、lint/recompile 等阶段级维护由 Q13 继续解释。
- 直接子职责：Q121 负责候选扫描、递归匹配、通用门和 entry 应用；Q122 负责解释 replacement graph、插入新节点、重接出口和删除旧节点。两者是下一层叶子；本篇只讲它们怎样共同完成一轮替换。

## 篇内知识点队列

| 顺序·编号 | 目标与完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| 1·K01-R | 说明 Q12 输入、出口、两个直接子职责及边界 | 本篇总图、Q121/Q122 源码定位 | 完成 |
| 2·K02-R | 解释 registry key、PatternExpr 递归、Match 捕获及同一 PatternExpr 重复约束 | distribute-mul、常量不符、同一/不同输入 | 完成，已实测并复读 |
| 3·K03-R | 解释无候选、结构失败、不支持节点、mutation region 和 `extra_check` 拒绝出口 | relu-only、乘 3、float/int、跨 copy_ | 完成；除 unsupported type 外均实测，后者按源码界定 |
| 4·K04-R | 解释 replacement 的 placeholder 绑定、新节点 meta、users、共享内部节点和多输出出口 | 共享 add 实测；packed 多输出按源码出口 | 完成，已复读 |
| 5·K05-R | 比较 Replacement/Graph/Lowering 三类 entry；解释具体 shape 重匹配 | 当前源码对象与出口对照 | 完成，已复读 |
| 6·K06-R | 用成功、拒绝、共享和副作用组成综合迁移题并作者复读 | 只能依据正文作答，答案含对象变化与因果 | 完成，已复读 |

## 处理情况与案例覆盖

| 情况与条件 | 入口和关键状态 | 本层出口 | 案例与证据 | 当前深度 |
| --- | --- | --- | --- | --- |
| 无锚点候选 | registry 对应 bucket 查不到图中节点 | 图不变，应用次数 0 | 实验 02 `relu(x+y)` | 已实测并讲解 |
| 根 target 相同但内部结构/常量不同 | 候选进入递归，子 PatternExpr 返回 MatchFailure | 尝试下一 entry/候选 | 实验 02 `(x+y)*3` | 已实测并讲解 |
| 结构命中且附加条件接受 | 捕获 x/y，float32 条件为真 | 嵌入 replacement，应用次数 1 | 实验 02 float32 | 已实测，需补状态表 |
| 结构命中但附加条件拒绝 | Match 已建立，int64 条件为假 | 原图保留，应用次数 0 | 实验 02 int64 | 已实测，需补因果迁移 |
| 跨 mutation region | 匹配节点 region id 不唯一 | 通用门拒绝，不调用 entry | 实验 02：add=0、copy_/mul=1 | 已实测并讲解 |
| 内部 add 另有外部 user | pattern 允许多 users；mul 可替换，但 add 不能删除 | relu 继续读旧 add；旧 mul users 改读新结果 | 实验 02 shared_add | 已实测并讲解 |
| replacement 多输出 | 旧 packed output 由 getitem users 拆用 | 每个 getitem user 重接相应新输出；非 getitem 报错 | 当前源码 `replace_with_graph` | 已讲机制，未独立运行 packed 例 |
| 三类 entry | 相同匹配入口，不同 apply 出口 | 嵌入图 / 自定义手术 / 插入 Lowering handler call | 当前三类 entry 的 `apply` | 已讲机制并复读 |
| `register_replacement` 具体重匹配 | 通用 pattern 命中后读取真实 meta，再 trace specific pattern/replacement | 当前真实 shape/dtype 通过才替换 | 当前源码 `check_fn` | 已讲机制并复读 |

## 依据与边界

- 当前实现依据：`/work/pytorch/torch/_inductor/pattern_matcher.py` 的 `Match`、`MatchContext`、三类 entry、`register_replacement`、mutation region helpers 与 `PatternMatcherPass.apply`。
- 当前测试依据：`/work/pytorch/test/inductor/test_pattern_matcher.py` 及直接相关 pass 测试；只采用本版本可定位的行为。
- 独立实验：`experiments/02_pattern_matcher.py` 和 `experiments/artifacts/pattern_matcher.txt` 证明 float32 成功、int64 条件拒绝、无根候选、常量结构失败、共享 add 保留、跨 mutation region 拒绝，以及重复 PatternExpr 的对象身份约束。
- 教学 distribute-mul replacement 是本目录独立规则，不是 PyTorch 当前默认注册的产品优化；正文必须一直保留这个边界。
- 源码能证明实现条件和出口，局部实验能证明已测对象变化；两者都不能替代读者理解。本轮尚无实际读者逐项反馈。

## 作者复读与迁移题

1. 图里有正确的 mul target，但常数由 2 改为 3，为什么 `extra_check` 不会运行？答：递归到常量参数时已经返回 `FailedMatch`，尚未形成可交给通用门和 `extra_check` 的 Match。
2. 共享 add 例为什么能替换，又为什么旧 add 仍在？答：pattern 用 `_users=MULTIPLE` 允许形成 Match；替换只重接旧 mul 的 users，relu 仍引用旧 add，所以 `erase_nodes` 不满足“无 user”条件。
3. 把 `copy_` 从 add 与 mul 之间移到 mul 之后，会先改变哪项判断？答：add/mul 重新同属一个 mutation region，跨区门不再拒绝；结构和 dtype 仍须继续检查。
4. packed 多输出旧节点若有一个不是 getitem 的直接 user，为什么不能照单输出路径重接？答：replacement 给出的是多个独立新节点，直接 user 没有索引说明该读哪一个；当前实现明确报 `can't handle`。
5. 两次规则命中结构相同但实际 shape 不同，为什么 replacement graph 可能不同？答：`register_replacement.check_fn` 用本次 Match 节点的真实 meta 重 trace specific search，确认后再以本次参数 trace replacement。

上述答案都能从正文的具体对象、顺序和出口推出；不是标题摘抄。实际读者反馈仍为空，不能写成读者已理解。

## 当前恢复点

- K01-R 至 K06-R 均完成，Q12 达到第 2 层本篇完成条件。
- packed 多输出没有新增独立运行实验，其机制结论限定为当前源码的明确出口；叶子 Q122 回修时逐行推演该分支。
- 根队列下一项为同层 Q13；不得提前进入 Q121/Q122 叶子回修。

## 第五轮重新复核

- 范围：Q121 候选扫描/应用与 Q122 replacement 嵌入是两个直接子职责；稳定拓扑、fake meta、DCE、lint/recompile 明确交给 Q13，没有把阶段维护并入 matcher。
- 过程：无锚点、常量结构失败、unsupported type、跨 mutation region、extra_check 拒绝、成功替换、共享内部 user、多输出和三类 entry 都有各自进入条件与出口。
- 案例：distribute-mul 明确是独立教学规则，不冒充产品默认优化；同一案例同时展示 placeholder 捕获、新节点插入、外部 user 重接、旧 mul 删除与共享 add 保留。
- 事实：实验覆盖成功、拒绝、共享与副作用边界；packed 多输出和 unsupported type 仅按当前源码陈述，并明确未作为独立运行结果。
- 理解：只凭正文可区分“user 约束导致无法形成 Match”和“替换后因外部 user 无法删除旧节点”，也可判断移动 copy_ 后只会消除 mutation-region 这一道拒绝而非保证替换。实际读者反馈仍为空。
