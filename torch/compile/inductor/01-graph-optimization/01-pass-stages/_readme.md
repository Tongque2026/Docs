# 三个 pass 处理阶段作者记录

## 第十一轮：按教材本身的首次阅读顺序重建入口逻辑

- 用户指出，作者与读者智能体之间的交流可能提前传递知识；即使智能体能回答验收问题，也必须判断答案是否由正文按顺序教出。此前冷读 A 不覆盖真实读者的“仍然看不懂”。
- 新读者只获知基础 torch、torch.compile 和四个组件名称，只读本篇正文，不读取源码、作者记录、旧报告或用户的具体断点。报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/122_PASS_STAGES_AFTER_FABLE5_COLD_REVIEW_1.md`，评级 B。
- 最早断点在旧稿第 7 行：同时宣布 AOTAutograd 和三个入口，直到九十多行后才展示图如何变化。正文现把 `add_` 请求的四种图状态和三个入口位置移到开头，再进入 pre-grad 实例。
- pre-grad 现明确区分命中时的节点替换与无论是否命中都执行的入口收尾；batch ReLU 当场说明当前 pre-grad 规则为何命中，并交代同一优化还有 ATen 形式的 post-grad 实现。
- joint 现按事实拆开两部分：`exp` 与反向使用者展示 joint graph 独有的可见关系；当前默认 `add(x, zero)` 清理展示实际节点改写。正文不虚构当前不存在的内置跨前反向 replacement。
- 第二名不继承历史的读者只读修订稿，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/123_PASS_STAGES_REWRITE_COLD_REVIEW_2.md`，评级 A。它能闭卷复述三个入口并完成三类新规则的迁移判断。
- 该读者发现开篇简图漏画推理也调用 joint、“五项门槛”有歧义、文末把单条规则未命中与整个入口无变化混在一起。正文现补出推理路径，改为“至少五个候选节点”，并按当前源码区分单条规则未命中与入口继续执行其他清理、整理和 reinplace。

## 第十轮：真实读者反馈优先于冷读评级

- 用户再次确认：fable5 写出的版本在没有经过读者反馈迭代时仍然看不懂。审阅者与作者之间的提示、问答和既有批注可能提前传递知识，因此不能把审阅者答对问题直接算作教材教会了读者；评价对象只能是正文按顺序建立认识的逻辑。
- 第一名新读者只读当前 README，提示中包含真实读者的失败现象和验收问题。报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/114_PASS_STAGES_COLD_REVIEW_1.md`，评级 A。该结果只能证明正文中能够找到 pre-grad 输入、pass 动作、入口收尾和 AOTAutograd 出口，不能单独证明自然阅读已经顺畅，因为问题清单本身会引导读者搜索答案。
- 第二名新读者不接收具体断点和机制答案，只按从头到尾的顺序记录每段当时形成的认识、悬空项与预期，再合上正文自然复述。报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/115_PASS_STAGES_COLD_REVIEW_2.md`，评级 A。它确认当前稿的 batch ReLU 已从候选检查走到节点替换、入口收尾和 AOTAutograd；`add_`/`view` 段也已经明确承担“相关改写的安全约束”，没有再冒充 pre-grad 的处理动作。
- 第二份报告仍发现三项局部负担：joint 的独有信息与实际改写分别由两个案例承担，需要读者拼接；reinplace 只有定义和放置理由，没有前后过程；数处否定式澄清服务于防止旧误读，而不是直接推进正面流程。这些问题记录在本篇后续精简项中，不能用 A 评级掩盖。
- 根据这份逐段记录，正文没有继续追加解释性补丁，而是改写原有句子：将五项门槛与语义条件正面分工；直接说明 `add_`/`view` 案例展示相关改写必须保持的顺序；把 functionalization、训练/推理的 joint 输入、局部 joint 清理和 post-grad 规则改为正面过程；删除多处“不是另一种优化”“不表示”“不能只看到”等作者式防误读措辞。
- 第三名新读者只读上述修订稿并自然顺读，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/116_PASS_STAGES_COLD_REVIEW_3.md`，评级 A。它能闭卷复述 pre-grad 入口、batch ReLU pass、`add_`/`view` 安全要求、AOTAutograd 交接和入口选择。它仍指出 joint 常量折叠把“已经得到全零 Tensor”作为前提，未展开常量折叠本身；该算法不属于本篇阶段协作的核心过程，保留为相应规则实现篇的深入项。
- 今后的读者验收分成两轮：第一轮只做无提示的顺序阅读和闭卷复述；只有读者自己暴露断点后，第二轮才用针对性问题核验。任何冷读 A 都不能覆盖真实用户的“不懂”；用户再次指出断点时，必须回到相应段落重建对象、动作和交接。

## 第九轮：真实读者再次判定未讲透

- 用户指出：即使 fable5 已改写，未经真实读者反馈迭代的正文仍可能让人看不懂；审读必须只评价教材逻辑，不能把作者与读者智能体之间交流传递的知识算作教材效果。
- 新建无历史冷读者，只允许阅读本篇 README；提示仅规定基础 torch/torch.compile 的认知边界和逐段审查方法，不提供三个入口的正确答案。报告保存为 `/work/learning/1021/56sol/inductor-fresh-reader-review/82_PASS_STAGES_COLD_REVIEW_10.md`，评级 C。
- 它确认 pre-grad 的入口、batch ReLU 改写、未命中出口已经能从正文复述；剩余结构缺口是：functionalization 在 mutation/alias 之前出现，解释迟到；joint/post 案例只证明会改图，没有证明阶段位置；最终两条件不足以判断新规则。
- 本轮重排为“pre-grad 实际改写 → pre-grad 输入图的安全难点 → AOTAutograd 表示变化 → joint → post-grad → 入口选择”。补足 batch ReLU 的 shape 变化、逐元素等价性、matcher/replacement 分工和入口收尾。
- 对 joint 不编造当前不存在的跨前反向默认案例。正文明确区分“入口提供 partition 前完整图范围”和“当前局部规则实际使用的信息”。对 joint/post 都不再声称局部规则理论上只能在该处运行。
- 入口选择改成五步判断：原始调用表示、functionalization 前提、partition 前全图、partition 后单图边界，以及多个入口均可时的注册/覆盖/顺序问题。
- 下一步由另一名无历史冷读者只读修订稿；不得把本轮批注、作者记录或正确答案交给它。

### 第九轮第二次冷读

- 第二名无历史读者只读重排后的 README，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/83_PASS_STAGES_COLD_REVIEW_11.md`，评级 B。
- 它已能闭卷回答 pre-grad 的完整入口职责、batch ReLU 等价性和改写、mutation/alias 的作用、functionalization 前后变化，以及 joint/post 的图范围与入口筛选。
- 剩余理解负担来自不服务本篇主线的实现名词，以及几个需要后文倒推或只能背结论的位置。本轮在 AOTAutograd 首次出现处直接交代其用途；解释修改图后为何必须重生成 `forward`；区分五项收益门槛与语义条件；给出参数写法例子；明确函数化图仍可为边界写回保留 `copy_`。
- 删除 joint/post 的目录式实现清单和未定义的 fake tensor；只保留当前案例所需的常量折叠、局部清理、pattern replacement 与 reinplace 因果。
- 下一步仍由新的无历史读者仅凭当前 README 复验。

### 第九轮第三次冷读

- 第三名无历史读者报告为 B。主线和三个入口均可闭卷复述；剩余缺口是 joint 的跨前反向能力与 post-grad 的最终单图边界仍只有判据，没有具体对象，读者必须自行发明案例才能迁移。
- 核对当前源码后，没有把内置的局部 joint 清理伪装成跨前反向规则。正文增加当前 `joint_custom_pre_pass`/`joint_custom_post_pass` 接口能够承载的成对规则示例：前向 `exp` 结果 y 同时被反向 `grad_y * y` 使用，只有 joint 保留这对关系。该例用于入口选择，不声称当前内置 pass 已注册这种替换。
- post-grad 使用当前 `remove_noop_ops` 的真实输入/输出存储检查：`input → clone → output` 不能删除 clone，否则新增输入输出 alias；clone 仅作图内中间值时才可在其余条件成立后绕过。这样“最终单图边界”有了实际读取对象、条件与两种出口。
- 将“边界上显式写回”改为“保留调用者能够观察到的写回”，避免把边界误解成图末端。

### 第九轮第四次冷读：通过

- 第四名无历史读者只读当前 README，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/85_PASS_STAGES_COLD_REVIEW_13.md`，评级 A。
- 读者能闭卷复述三个入口的输入、动作与出口，完整推演 batch ReLU，解释 mutation/alias 与 functionalization，并分别用 F.relu、joint custom pair 和 clone 输入输出 alias 三个案例判断入口。
- 读者准确区分当前内置规则、custom pass 放置示例和当前源码真实条件，没有从示例误推出不存在的内置优化。
- 本篇回修闭环。后续逐篇审读继续使用全新无历史读者；提示只规定认知背景和审查方法，不提供教材应教出的答案。

## 第八轮：fable5 版本仍未被真实读者理解

- 真实读者明确反馈：即使第七轮由 fable5 改写，仍未理解 pre-grad 到底做什么。最终标准是教材文字本身；审阅提示或作者交流传递的知识不能算教材教会读者。
- 新建无会话历史的冷读者，只允许阅读本篇 README。报告保存为 `/work/learning/1021/56sol/inductor-fresh-reader-review/63_PASS_STAGES_COLD_REVIEW_3.md`，评级 C。
- 这次确认的结构问题：正文虽把 `batch_relu` 动作写完整，却没有先说明 pre-grad 是 pass 运行位置；把当前 matcher 的放置理由写成优化本身的唯一时机；把 mutation/alias 约束写得像统一处理动作；又用不具阶段独占性的 joint/post 示例暗示阶段职责。
- 本轮正文先回答 `pre_grad_passes` 实际做什么，再完整运行 `batch_relu`；明确 ATen 目标的 post-grad 对应实现存在，因此只说明当前规则实现的放置，不声称优化思想只能在 pre-grad。mutation/alias 明确改为规则限制，`sin + add_` 明确是未改图反例。
- 下一步必须由另一名无历史冷读者只读修订后的 README；不得把本轮报告或正确答案交给它。

### 第八轮第二次冷读

- 第二位无历史读者只读修订稿，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/64_PASS_STAGES_COLD_REVIEW_4.md`，评级 B。
- 读者已能准确复述 pre-grad 主线和 batch ReLU 全过程；剩余问题集中在首次出现的基础名词、joint 推理出口、alias 具体关系、post-grad 案例安全条件和总结后突然展开的控制流层级。
- 本轮按首次阅读顺序补齐最低定义，正面陈述阶段职责，删除大部分回应旧误解的否定句；控制流只保留下一篇导航，不在当前篇展开半套流程。
- 下一步仍须换一名无历史读者，只以当前 README 验收。

### 第八轮第三次冷读

- 第三位无历史读者报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/65_PASS_STAGES_COLD_REVIEW_5.md`，仍为 B；主线已经成立，剩余问题是四个首次出现或旁枝术语。
- AOTAutograd 的首次介绍改为本文实际展示的 `add_ → add + copy_` 与训练合图再拆分；删除“规范化”和量化调整清单；正面说明训练与推理复用 `joint_graph_passes`，但输入图范围和出口不同。

### 第八轮第四次冷读

- 第四位无历史读者报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/66_PASS_STAGES_COLD_REVIEW_6.md`，评级 B。
- 修正开头把 partition 画成推理必经步骤的实质错误；训练与推理从 joint 出口分叉，只有训练进入 partition。
- 收窄三个案例的教学任务：batch ReLU 固定 `inplace=False` 并说明收益阈值；joint 不再暗示当前局部清理体现 joint 独有能力；post-grad 用同 dtype 空转换替换需要额外 stride/alias 论证的 view。

### 第八轮第五次冷读

- 第五位无历史读者报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/67_PASS_STAGES_COLD_REVIEW_7.md`，评级 B。
- 剩余核心歧义是 functionalization 后为何仍出现 `copy_`。正文现区分普通新值计算与外部可观察输入写回，不把函数化解释成删除全部副作用。
- functionalization、joint graph、partition 均移到总图之前定义；推理复用 joint 函数名的事实也与训练合图语义分开。

### 第八轮第六次冷读

- 第六位无历史读者报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/68_PASS_STAGES_COLD_REVIEW_8.md`，评级 B。
- 总图现于 joint 函数调用前分开训练 joint graph 与推理单图；训练与推理只复用函数，不再共享错误的图种类标签。
- mutation 与 alias 合入同一个 `view/sin/add_/cos` 案例，并画出函数化后的完整读写顺序。

### 第八轮第七次冷读：通过

- 第七位无历史读者报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/69_PASS_STAGES_COLD_REVIEW_9.md`，评级 A。
- 读者仅凭正文准确复述三个入口、训练/推理分流、batch ReLU、mutation/alias 在 functionalization 前后的具体关系和规则放置依据。
- 本轮真实读者反馈已在当前篇闭环。后续逐篇冷读仍沿用相同隔离规则：提示只规定认知背景和审查方法，不传递教材应当教出的答案；每轮换无历史读者。

## 第七轮：真实读者仍未理解后的重构

- 实际读者反馈：fable5 改写后仍无法说明 pre-grad 到底执行什么；hook no-op 案例和“必须考虑 mutation/alias”只解释约束，没有展示合法改图动作。
- 独立逐句审阅报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/15_PASS_STAGES_DEEP_REVIEW.md`。该报告只部分关闭原反馈，并指出三条新增“典型规则”无当前 pre-grad 源码支撑。
- 本轮删除自定义 LSTM、连续原地操作合并成 kernel、删除同存储冗余操作三条无落点陈述。pre-grad 主案例改用当前默认注册的 `BatchReLuPreGradFusion`：默认至少五个同组节点，完整展示 stack→relu→unbind/getitem、重接 users、删除旧节点。
- `sin + add_` 降为 no-op 与 mutation 顺序约束对照；明确 functionalization 是 AOTAutograd 的动作，不冒充 pre-grad pass。
- joint 使用当前 `constant_fold_uniform_value` 后的 `remove_no_ops` 展示 `add(x, zero)` 的引用替换和节点删除；post-grad 使用当前 `remove_noop_ops` 展示同 shape view 删除。
- 验收隔离：原审阅智能体已接触作者讨论，不承担最终理解验收。下一步创建不继承会话历史的冷启动读者，只提供读者知识边界和当前教材；禁止给它正确答案或旧反馈。
- 首次冷启动报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/16_PASS_STAGES_COLD_REVIEW.md`。读者能闭卷复述 pre-grad 的输入、完整 batch ReLU 改写、出口及 functionalization 归属；仍指出 FX/Dynamo/ATen/alias 未定义、mutation 图省略外部写入、joint 案例未声明其不使用合并图特性、推理 joint 出口含糊、post-grad reinplace 无案例。
- 第二次修订补齐上述定义和边，明确 joint 的具体清理动作并非依赖全部 joint 信息，写清推理直接进入 post-grad；删除无案例的 reinplace 提前介绍。下一步由另一名不继承历史的冷启动读者重新验收。
- 第二名冷启动读者报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/17_PASS_STAGES_COLD_REVIEW_2.md`。它能闭卷复原 pre-grad 的输入、五 ReLU 前后图、六项节点动作、no-op 对照、functionalization 归属和出口，判定阶段概念主线已可仅凭教材理解。
- 仍有意保留的边界：本篇教“判断阶段需要哪些资料”，不承诺仅凭任意 Python 源码确定注册位置；匹配算法、shape 来源和复杂 alias 分析分别属于后续实现篇。joint 当前案例只证明 joint 会真实改节点，正文明确声明它没有利用前反向共同关系，不把 partition 的保存/重算冒充 joint pass。

## 2026-10-22 独立读者审阅后的结构回修

- 审阅依据：`/work/learning/1021/56sol/inductor-fresh-reader-review/02_PASS_STAGES_REVIEW.md`，75/75 个语义单元。
- 正文统一用“输入图—独有信息—代表规则—出口”比较 pre、joint、post。
- 删除 joint/post 的内部 pass 清单、FSDP replacement 和图不变量算法；这些材料分别属于 matcher、具体规则或图状态篇。
- mutation 的 post custom hook 明确为中间快照；本篇只用最终运行结果验证旧值读取与写回语义持续成立，不把 hook 冒充最终 FX 出口。
- 作者验证方式、旧稿防误读句和课程推进说明只保存在本记录。

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
