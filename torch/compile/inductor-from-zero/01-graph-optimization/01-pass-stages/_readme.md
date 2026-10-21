# 三个 pass 处理阶段作者记录

## 所在层、输入与出口

- 队列项：Q11，第 2 层，父项 Q10；第五轮已按技能重新复核，保留“本层完成”。本状态不替代 Q111 递归入口的叶子源码回修。
- 读者已经掌握：FX node 与引用、Q10 的 pass 阶段/pattern/invariant 三项职责，以及 mutation 请求在 pre 与 AOT 后表示不同。
- 本篇输入：Dynamo 交出的 pre-grad GraphModule，以及 AOTAutograd 形成的 inference graph 或 training joint/forward/backward graphs。
- 本篇出口：进入 GraphLowering 前的每张 normalized/functionalized 单图；其拓扑、Python forward 和必要 meta 已按当前阶段收尾。
- 本篇打开 Q10 的“pass 处理阶段”职责。篇内解释 pre-grad、joint、post-grad 各自接收怎样的图、能采用哪些前提、处理后交给谁；唯一直接实现子模块 Q111 打开三个 recursive entry 的完整源码。具体 pattern 替换算法属于同层 Q12，图排序/meta 维护属于同层 Q13，不能在本篇重复展开。
- 本层承诺：读者能根据变换需要的语义信息选择阶段，能比较 inference 与 training 路由，能说明嵌套子图何时处理及各阶段交给谁。下一层 Q111 才逐行解释递归入口。

## 篇内知识点队列

| 编号 | 问题、对象与完成条件 | 案例 | 状态 |
| --- | --- | --- | --- |
| K01-R | 三个阶段各自能假定什么、接收和交出什么 | `sin/add_` 请求的三张快照 | 完成，已复读 |
| K02-R | functionalization 怎样把计算新值与外部写回分开 | pre `add_` 对照 joint/post `add+copy_` | 完成，已实测并复读 |
| K03-R | inference 与 training 为什么都经过 joint、但 post 图数量不同 | inference 一张图；training joint 后拆为 forward/backward | 完成，已实测并复读 |
| K04-R | cond/while 子图怎样递归处理，example inputs 与 is_inference 怎样传递 | 当前 recursive entry 源码；嵌套控制流 | 完成，已作递归顺序与参数迁移复读 |
| K05-R | no-change/config/失败出口与阶段选择迁移 | pattern 关闭、joint count=0、post reinplace、跨阶段需求 | 完成，已作条件出口与职责边界复读 |

## 处理情况与案例覆盖

| 情况 | 本层入口 | 关键变化 | 本层出口 | 证据 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 未函数化 mutation | pre 图含 `add_` | pass 必须自行保护 alias/mutation | pre 收尾后的 GraphModule 交 AOT | 实验 16、pre_grad.py | 机制已讲并复读 |
| AOT 后显式写回 | joint/post 图含 add 与 copy_ | 新值计算和 storage 写回分开 | joint 交 partition 或 inference compiler；post 交 Lowering | 实验 16 | 机制已讲并复读 |
| inference | AOT inference graph | fw compiler 主动调用 joint；post 一张 inference graph | 单张图进入 Lowering | 实验 19、compile_fx.py | 机制已讲并复读 |
| training | joint forwards+backwards graph | joint 后 min-cut partition；forward/backward 分别 post | 两张图分别进入 Lowering | 实验 19、compile_fx.py | 机制已讲并复读 |
| nested cond/while | 外图引用子 GraphModule | 每个阶段先递归子图再处理外图 | 保留 HOP 节点与子图接口 | recursive entry 源码 | 机制已讲并复读 |
| pass 无候选/配置关闭 | 合法 GraphModule | 某组 pattern 可跳过，阶段仍按自身收尾规则退出 | 原计算关系或其他 pass 后关系 | 三个 pass 源码 | 机制已讲并复读 |
| inference-only locality | post inference 图且配置开启 | `reorder_for_locality` 只在 is_inference 分支执行 | 调整后的合法 post 图 | post_grad.py | 机制已讲并作条件迁移 |

## 依据与证据边界

- 源码：`compile_fx.py` 的三个 recursive entries 及 AOT compiler/partition 路由；三个 `fx_passes/*.py` 阶段主体。
- 实验 16：同一 mutation 请求在 pre 为 `sin+add_`，joint/post 为 `sin+add+copy_`；证明该请求在这些 hook 位置的表示，不推广为所有图逐字相同。
- 实验 19：inference 捕获一次 joint、一次 post；training 捕获一张含前向与反向算子的 joint 图，随后捕获 forward post 与 backward post 两张图。目标列表证明图范围差异，不独自证明每个 pass 内部算法。
- AOTAutograd 内部不作为本篇源码展开范围；本篇仅沿当前 `compile_fx.py` 的调用、partition 接口和实际 hook 图解释交接。
- 实际读者反馈为空；以下均为作者仅依正文复读，不声称读者已经理解。

## 第三轮严格复读进度

- K01/K02：若规则只匹配函数式 `aten.add.Tensor`，pre 图中的 `add_` 不满足前提；joint/post 才出现本例的 add 与 copy_。若改写破坏 copy_ 顺序，仅检查 old 数值不够，还必须检查调用者 x。
- K03：inference 没有 backward partition，但 fw compiler 仍主动运行 joint passes；training 的 joint 图同时含 sin/mul/sum 和 cos/expand/add 等反向工作，partition 后 post 分别看到三项 forward targets 与七项 backward targets。
- K04：三个 recursive entry 均先子图后外图；pre 子图明确收到 `example_inputs=None`，post 子图继承同一 `is_inference`，joint 无额外参数。需要样例的规则不能假定嵌套图也拥有外图 inputs。
- K05：pattern matcher 关闭只跳过相应 matcher 主体；pre/post 仍有各自收尾和其他条件工作，joint 的末尾重排/recompile 由 count 控制。候选拒绝保留原节点，异常则向上失败。
- 阶段迁移：inference-only locality 属于 post 的 is_inference 分支；重新引入 mutation 的工作放在 post 末尾；同时需要 backward 使用关系与 kernel loop range 的优化必须拆分职责。
- Q11 本层主要情况均有入口、阶段状态、出口和迁移推演，达到第 2 层完成条件。Q111 递归入口源码仍待第三层回修；完成 Q11 不代替叶子完成。实际读者反馈仍为空。

## 第五轮重新复核

- 范围：本篇只打开 Q10 的“pass 处理阶段”职责，pre、joint、post 是篇内阶段；唯一直接实现子模块是 Q111。matcher 算法和图不变量分别留在同层 Q12、Q13，没有越层重复。
- 过程：同一 `sin/add_` 请求展示 pre 的原地语义与 joint/post 的 `add+copy_`；inference/training 展示 joint 都存在而 post 图数量不同；cond/while 展示子图先于外图且不摊平。
- 出口：无匹配、配置关闭、附加条件拒绝和 pass 异常均有不同出口；“关闭 matcher”没有被写成跳过整个阶段。
- 事实：实验 16/19 只支持相应 hook 中观察到的表示和图数量；AOTAutograd 内部算法不在本篇冒充完整源码依据。
- 理解：只凭正文可判断依赖 example inputs、只适用于 inference locality、重新引入 in-place、同时需要 backward 与最终 loop range 的规则应放在哪里或为何需要拆分。实际读者反馈仍为空。

## 2026-10-21 读者反馈回修

- 原反馈指出，旧稿用一个抽象词代替了流程位置。缺口不是单个词的措辞问题，而是读者尚未看到 pre-grad、joint、post-grad 分别位于编译流程的哪里。
- 实质修订：标题和导航统一改为“三个处理阶段”；正文先画出 `Dynamo 图 → pre-grad → AOTAutograd/joint → partition → post-grad → GraphLowering`，再解释每一阶段能看到的图和可采用的语义假设。
- 复读结果：读者能根据三个位置实际保留的图内容，回答 pass 为什么不能在 pre-grad、joint、post-grad 之间随意搬动。
- 第二条原反馈指出，旧稿用作者自己的分类词概括三个 pass 位置，读者无法据此想象图发生了什么。正文现改为三个可观察事实：pre-grad 仍有 `add_`；joint 同时有前向与反向；post-grad 拿到拆分后的单图。
- 读者再次指出，“规则需要看见什么”仍然把判断留给读者。本轮进一步写明三类规则及其位置：检查 `add_` 的规则在 pre-grad；比较保存与重算的规则在 joint；处理普通 ATen 节点的规则在 post-grad。
