# 图优化篇作者记录

## 2026-10-22 独立读者审阅后的结构回修

- 审阅依据：`/work/learning/1021/56sol/inductor-fresh-reader-review/01_GRAPH_REVIEW.md`。
- 原断点：先展开 pre/joint/post，随后才声明三项直接职责；阶段内部分类、模块职责和图状态分类连续出现，读者无法判断层级。
- 新主线：输入/出口 → 同一 matcher 案例依次经过运行位置、节点替换、状态同步 → 再分别打开三个职责。
- `sin/add_/copy_` 改为阶段编排的局部案例；float32 分配律替换承担父篇贯穿案例。
- 正文删去针对旧稿分类错误的防误读话术；作者为何这样重排只保存在本记录。

## 所在层、输入与出口

- 队列项：Q10，第 1 层，父项 Q00；当前状态为第二轮回修。
- 读者已知：FX node 通过结果引用连成图；根篇四阶段边界；FX replacement 与 scheduler fusion 不是同一层。
- 本篇输入：Dynamo/AOTAutograd 在不同处理阶段交来的 FX GraphModule，以及节点 meta、配置和 pass 注册。
- 本篇出口：GraphLowering 可按拓扑解释的、meta 与 GraphModule 代码一致的 FX 图。
- 直接子职责只有三项，并与目录一致：Q11 安排 pre/joint/post-grad 的处理阶段；Q12 完成模式匹配与替换；Q13 在改图后维护拓扑、fake meta、可执行代码和合法性。规范化、简化、reinplace 等具体 pass 属于 Q11 各阶段组织的工作，不另冒充第四个直接子模块。
- 下一层打开三项职责内部；本篇不讲 matcher 递归算法、FakeTensorUpdater 字段或具体 pass 函数的完整源码。

## 篇内知识点队列

| 编号 | 完成条件 | 案例 | 状态 |
| --- | --- | --- | --- |
| K01-R | 三项直接职责为何缺一不可；能从输入图走到 GraphLowering 入口 | 系统图，明确每项收到和交出什么 | 完成，已复读 |
| K02-R | 同一个 mutation 请求为何在三个阶段呈现不同图 | `add_`：pre 保留 mutation，joint/post 为 functional add + copy_ | 完成，已实测并复读 |
| K03-R | replacement 怎样改变 Lowering 将读到的边，拒绝时为何保留原图 | float32 接受、int64 拒绝、共享 add 保留 | 完成，已复读 |
| K04-R | 拓扑/meta/recompile/lint 怎样恢复可用 GraphModule | 乱序插入并重接 user 的独立案例 | 完成，已实测并复读 |
| K05-R | 综合迁移验收 | mutation 图中匹配子图后，区分三项职责的变化 | 完成，已复读 |

Q10 第二轮回修完成。根队列转同层 Q20，不提前进入 Q11。

## 处理情况与案例覆盖

| 情况 | 进入对象 | 本层变化与出口 | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| pass 阶段差异 | 含 `add_` 的请求 | pre 仍见 mutation；AOT 后以函数式计算和显式写回表达 | `experiments/16_pass_stage_graphs.py` | 机制已讲并复读 |
| pattern 接受 | float32 `mul(add(x,y),2)` | 新建子图、重接 output、清除无用户旧节点 | `experiments/02_pattern_matcher.py` | 机制已讲并复读 |
| pattern 拒绝 | 同结构 int64 | extra_check 拒绝，原图合法流出 | 同上 | 机制已讲并复读 |
| 旧节点共享 | add 另有 user | 只删除无用户部分，共享 add 保留 | 当前源码与迁移题 | 机制已讲并复读 |
| 不变量恢复 | 故意乱序插入依赖并重接 | stable sort、meta、recompile、lint 恢复可用图 | `experiments/03_graph_invariants.py` | 机制已讲并复读 |

## 依据、边界与反馈

依据：`compile_fx.py` 的调用位置；`pre_grad.py:pre_grad_passes`、`joint_graph.py:joint_graph_passes`、`post_grad.py:post_grad_passes`。正文只承诺本层协作，不展开 pattern matcher 内部。

上一版把“规范化、局部简化、模式替换、不变量维护”画成四个直接职责，却只建立三个子目录，模块树与协作图不一致；第二轮回修改为三项直接子职责，具体 pass 行为归入相应处理阶段。旧“本层及 Q11–Q13 均完成”结论撤销。

既有 pattern 复读：float32 的 `mul(add(x,y),2)` 会新建两项 mul 和一项 add，并把 output 从旧 mul 重接到新 add；int64 因 extra_check 保留原图。若旧 add 还有第二个用户，它不能随旧 mul 一起删除。本轮仍须把它与处理阶段和不变量案例贯通。实际读者尚未回答。

## 第二轮理解验收

- 只凭正文可答：若把只理解函数式 add/mul 的 replacement 搬到 pre-grad，它会遇到仍保留的 `add_` 与 alias/mutation 语义，不能假定已有纯函数结构。
- 只凭正文可答：float32 replacement 接受且旧 add 还有第二输出时，新分配律子图和旧 add 会同时流出；旧 mul 可删，旧 add 不可删。
- 只凭正文可答：stable sort 后若不 recompile，Graph 对象顺序可以正确，但 `GraphModule.forward` 仍可能执行旧代码；若不更新 meta，后续又可能用旧 shape/dtype/stride 作决定。
- 综合迁移：若 post-grad mutation 图中的函数式 add 命中一项 replacement，阶段职责先保证它可以按纯计算匹配；matcher 改相应边；不变量维护恢复图、meta 与 forward；显式 copy_ 仍负责写回，除非规则明确且安全地连同它一起改写。
- 上述是作者仅依据正文的复读。实际读者反馈仍为空，不报告读者已经理解。

