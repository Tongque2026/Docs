# Q111 作者记录：递归 pass 入口

> 第五轮状态：叶子完成。已重新对照当前 `compile_fx.py:247-289`，并按完整源码、重要分支、具体状态和理解迁移四项复核。

## 读者、层次与完成目标

- 所在层：深度 3，Q11 的实现叶子；不再拆函数子目录。
- 读者前提：已从 Q11 学过 pre-grad、joint、post-grad 三个处理阶段，知道 FX `GraphModule`、节点引用和 `cond`/`while_loop` 子图接口；此前未学递归包装器实现。
- 本叶子职责：枚举当前图直接引用的控制流子图，并让同一阶段按后序遍历覆盖整棵图树。
- 入口：父 `GraphModule`；pre 另收 `example_inputs`，post 另收 `is_inference`。
- 出口：pre 返回阶段函数交回的当前图并逐层写回子图；joint/post 通过原地修改完成，包装器隐式返回 `None`。
- 承诺深度：声明范围全部源码、原顺序、教学行内注释，以及普通、cond、while、嵌套、对象替换和参数传播的具体状态推演。

## 声明范围与资料边界

- 产品文件：`/work/pytorch/torch/_inductor/compile_fx.py:247-289`。
- 完整范围：`_get_subgraph_names`、`_recursive_pre_grad_passes`、`_recursive_joint_graph_passes`、`_recursive_post_grad_passes` 四个连续函数。
- 源码基线：commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 允许资料：上述当前源码、当前仓库调用点、独立实验 `../../../experiments/16_pass_stage_graphs.py` 及其产物。
- 本篇不把递归包装器扩写成 pass 内部算法，也不声称它决定运行时分支选择或循环次数。

## 篇内知识点队列

| 顺序·编号 | 问题、对象与完成条件 | 案例与正文 | 状态 |
| --- | --- | --- | --- |
| 1·K01 | 由 cond/while 节点参数得到直接子图属性名，并解释排序对象 | 单 cond、根层 while 后接 cond；“先看输入”“嵌套次序” | 完成并复读 |
| 2·K02 | 推演后序递归中 `subgraph_name`、`subgraph` 和当前 `gm` 的变化 | 单 cond 四时刻表、9 图混合树 | 完成并复读 |
| 3·K03 | 区分 pre 的返回替换与 joint/post 原地修改合同 | 三入口状态对照与对象身份实验 | 完成并复读 |
| 4·K04 | 覆盖无子图、参数传播、删除分支和漏写回的迁移情况 | mutation hook、参数表、两道迁移题 | 完成并复读 |

当前点：K04 已完成。篇内下一项：无。模块达到叶子完成条件后，根队列转 Q121。

## 处理情况覆盖

| 情况与条件 | 关键状态与出口 | 正文位置 | 证据 | 深度与状态 |
| --- | --- | --- | --- | --- |
| 没有 cond/while | 生成器为空，循环零次；直接处理当前图 | “普通图没有子图” | mutation 编译 hook + 产品源码 | 叶子机制完成 |
| 一个 cond | 依次 true、false，递归返回后写回，最后父图 | “一个 cond” | 产品源码 + 状态推演 | 叶子机制完成 |
| 一个 while | 依次 cond graph、body graph，最后父图 | “while、cond 与嵌套图” | 产品源码 + 构造图实验 | 叶子机制完成 |
| 同层多个 HOO | 按 FX 节点排序后枚举；每个节点内部参数顺序固定 | 9 图混合树 | 实验实际 `direct_names` | 叶子机制完成 |
| 子图继续嵌套 | 深度优先、后序处理，根最后 | 9 图访问序列 | 三包装器实际 trace | 叶子机制完成 |
| pre 返回新图 | 子图由 `setattr` 写回；根由调用者接返回值 | pre 状态表与对照表 | 替换对象实验 | 叶子机制完成 |
| joint/post 原地修改 | 不保存阶段函数返回值，包装器返回 `None` | 三入口状态差异 | 对象 id 与返回值实验 | 叶子机制完成 |
| 额外参数 | pre 仅根有样例输入；post 每层传同一推理标志 | 状态对照表 | 实际 trace | 叶子机制完成 |

声明范围内没有 `else` 失败出口，也没有处理其他 higher-order op 的通用分支。本篇据此只说明 cond 与 while_loop；不能从生成器函数名推广到所有子图携带者。

## 案例事实与实验结果

实验 16 有两个互补部分：

1. 实际 `torch.compile` mutation 请求在 pre hook 保留 `add_`，joint/post hook 为函数式 `add` 加 `copy_`；结果 `old=[0,sin(1),sin(2)]`，调用者输入变为 `[1,2,3]`。该部分只证明无控制流路径和三个处理阶段的表示差异。
2. 独立构造含 while、cond 与两处嵌套 cond 的 FX 图；调用产品递归包装器，mock 的只是阶段函数体。实际结果为：

```text
direct_names=['while_cond', 'while_body', 'true_graph', 'false_graph']
order=wc_true -> wc_false -> while_cond -> while_body
      -> t_true -> t_false -> true_graph -> false_graph -> root
pre child inputs: 全部 None；root 保留 sentinel inputs
pre root replaced: True；四个直接 child replaced: 全部 True
joint return: None；全部 named_modules id 不变
post flags: 九项全部 True；return: None
```

产物：`../../../experiments/artifacts/pass_stage_graphs.txt`。该构造实验不执行控制流算子，不证明运行时 HOO 语义或真实 pass 对某一复杂子图一定产生替换；它直接核对递归顺序、参数、返回和对象连接。

## 源码完整性审计

- 从 `def _get_subgraph_names` 到 `post_grad_passes(gm, is_inference)` 逐行对照产品源码。
- 两个 `find_nodes`、外层 `sorted/chain`、cond/while 两个分支、六个 yield/name 赋值均保留。
- pre 的产品注释、递归参数 `None`、`setattr` 和 return 均保留。
- joint/post 的无 return 形式、post 默认参数与递归传参均保留。
- 正文新增注释统一以 `# 教学：` 标记；未改写产品语句，未使用省略号。
- 完成审计再次从正文 Python fence 提取四个函数，去掉固定教学注释后与当前 `compile_fx.py` 逐函数比较 AST；四项均等价（4/4）。

源码顺序完整只证明没有漏行；完成判断还依赖上面的处理覆盖、具体状态和理解复读。

## 只凭正文的作者理解验收

1. **为何根中 while 写在 cond 前时，直接名字不是把两个 cond 分支先列出？** `sorted` 排的是当前图的两个 HOO 调用节点；先遇到 while 后，按其参数先产生 `while_cond`、`while_body`，随后 cond 才产生 true/false。
2. **为何 `wc_true` 先于 `while_cond`？** 取得 `while_cond` 后立即递归；其内部 cond 的两支完成后，才调用当前 `while_cond` 的阶段函数。
3. **若删除 while，哪些状态消失？** `wc_true`、`wc_false`、`while_cond`、`while_body` 四项消失；true 分支内部后序关系及根最后处理不变。
4. **pre 若返回新 true 图，谁保持连接？** 子层返回给父层局部变量 `new_subgraph`，随后 `setattr(gm, 'true_graph', new_subgraph)` 改写父模块属性；最外层新根由调用者接住。
5. **为何不能说 joint/post 什么都不返回，所以没有效果？** 包装器不消费阶段函数返回值，其合同是让阶段函数原地修改传入对象；实验显示对象身份不变而记录属性确实写入。
6. **子层 `example_inputs=None` 能否推出子图没有输入？** 不能；它只说明递归入口没有对应子图的样例输入列表，子图本身仍可有 placeholder/operands。

这些答案均能由正文的源码、状态表和混合树推出，不依赖未写出的 pass 内部知识。实际读者反馈仍未取得；不得把作者复读写成读者已理解。

## 本轮整改与续写位置

原稿虽保留源码，却只把真实 mutation 实验代入零子图路径，嵌套控制流完全依赖文字假设；作者记录也缺少所在层、入口/出口、覆盖表、证据边界和可迁移理解验收。本轮重建计划，扩展实验实际调用三种产品递归包装器，并据结果重写正文。另删除“joint/post 会更新 graph、meta 和 generated forward”的过宽表述，改成源码真正保证的原地修改合同。

## 第五轮叶子验收

- **完整源码**：声明范围仍是四个连续函数；产品语句、产品注释、空行和顺序全部保留，新增行只使用 `# 教学：`。重新对照当前源码未发现漂移。
- **分支与出口**：cond/while 两个 target 分支、没有子图的零次循环、嵌套递归、pre 返回替换、joint/post 原地修改、pre/post 参数传播均有进入条件和出口。
- **具体状态**：单 cond 的四时刻表解释 `subgraph_name/subgraph/new_subgraph/gm`；9 图混合树给出实际后序序列；对象身份实验区分替换与原地修改。
- **理解迁移**：正文可独立推出删除 while 后的序列、漏掉 `setattr` 后新子图失联、joint/post 返回值被忽略但原地效果保留，以及 `example_inputs=None` 不代表子图无 placeholder。
- **证据边界**：构造实验实际调用产品递归包装器但 mock 阶段函数，只支持遍历、参数、返回和对象连接；不外推真实 pass 算法或运行时 HOO 分支。

Q111 第五轮叶子复核完成。全系列下一项是同一深度的 Q121：matcher 候选扫描与应用完整源码。

## 第一次隔离冷读整改

- 报告：`../../../../inductor-fresh-reader-review/54_RECURSIVE_ENTRY_COLD_REVIEW_1.md`，评级 B。
- 读者能复述主线和三条路径，但无法只凭正文解释生成器暂停/恢复、`get_attr` 节点到模块属性的四层引用链，以及替换对象与原地修改的身份差异。
- 源码核对：产品实现确实读取 HOO 参数中 `get_attr` 节点的 `.name`；当前实验用 `Graph.get_attr("true_graph")` 等简单属性名构图，节点 `name/target` 均对应同名父模块属性。正文将这一点限定为本实现接收的输入表示合同，没有推广到任意 get_attr。
- 本轮增加单 cond 的 yield 时间线、混合根的 find/chain/sorted/yield 中间值、pre 与 joint/post 的对象身份对照、子图参数签名为何不能直接沿用根样例，以及两道覆盖生成器和返回合同的迁移题。

## 第二次隔离冷读验收

- 报告：`../../../../inductor-fresh-reader-review/55_RECURSIVE_ENTRY_COLD_REVIEW_2.md`，评级 A。
- 读者可独立推导单 cond 的生成器暂停与恢复、mixed root 的直接名字、九图后序、pre 漏写回和 joint 返回新对象时的后果，以及 `example_inputs=None` 的边界。
- 必须修改项：无。
- 当前状态：Q111 通过隔离冷读验收；下一篇为同层 Q121。
