# GraphLowering 解释器篇作者记录

## 所在层、输入与出口

- 队列位置：Q21，第 2 层，父项 Q20；第五轮已按技能重新复核，保留“本层完成”。本状态不替代 Q211–Q214 四个叶子的源码回修。
- 读者前置：Q20 已建立 Lowering 的四个直接职责以及延迟规则、物化存储、特殊调用之间的交接；Q13 已保证进入图满足 FX 结构合同。
- 本篇输入：normalized/functionalized 的 FX `GraphModule`、example/fake inputs、ShapeEnv 与编译配置。
- 本篇出口：`graph_inputs`、`graph_outputs`、IR env、已登记 `buffers/operations`、mutation/符号约束及最终布局，供 Scheduler 和 wrapper/codegen 使用；失败时交出带 target/参数上下文的 lowering 异常或明确 unsupported 出口。
- 四个直接子职责与目录一致：Q211 输入/placeholder 登记；Q212 `get_attr`/`call_function` 分派；Q213 output 与输入 mutation 合同；Q214 `run_node` 的全图上下文和节点后处理。
- 本层只讲四者怎样协作并走完主要 FX 节点类型；四个叶子的完整源码、字段和逐分支状态留到第 3 层重新验收。旧记录“Q211–Q214 均已完成”不符合 BFS 与叶子完成标准，已撤销。

## 篇内知识点队列

| 顺序·编号 | 目标与完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| 1·K01-R | 区分 Interpreter env、graph_inputs、graph_inputs_original、buffers/operations、graph_outputs 五份状态及消费者 | add→relu→mul 轨迹 | 完成，已实测并复读 |
| 2·K02-R | 走通 Tensor、symbolic scalar、普通标量/None 等 placeholder 分支 | 实验 04 + 当前 placeholder 源码 | 完成，已实测并讲明 bool 直返边界 |
| 3·K03-R | 比较 getitem、pattern handler、已注册 lowering、fallback 与 missing operator 出口 | add、cond getitem、histc、源码失败分支 | 完成，已复读 |
| 4·K04-R | 解释 run_node 如何围绕单节点处理 origin、unsupported fallback、layout/realize、users 和 unbacked symbols | 单输出、共享/extern、控制流与符号路径 | 完成，已复读 |
| 5·K05-R | 解释 output 如何形成 graph_outputs、修正输出 stride、写回 mutation 并 finalize layout | 单输出、cond 多输出、copy_ alias | 完成，已复读 |
| 6·K06-R | 用普通链、外部调用/控制流、mutation/多输出组成迁移验收 | 实验 04/06/07/18 与正文 | 完成，已复读 |

## 处理情况与案例覆盖

| 情况与条件 | 进入对象 | 本层出口 | 证据 | 当前深度 |
| --- | --- | --- | --- | --- |
| Tensor placeholder | fake/example tensor 的 device/dtype/size/stride | InputBuffer→StorageBox→TensorBox，并登记正式输入 | 实验 04 | 已实测并讲解 |
| symbolic scalar placeholder | SymInt/SymFloat/SymBool 或 scalar meta | SymPy 表达式；直接返回时包装为 shape constant | 实验 04 unbacked SymInt | 已实测并讲解 |
| None/BackwardState/普通值 | 非 Tensor placeholder | int/float 登记 SymPy 常量；None/BackwardState 返回 None | 实验 04 + 当前源码 | 已讲普通值、None 和边界 |
| 已注册普通 ATen lowering | env 中 IR 参数与 target | 延迟/物化 IR，写回 env | 实验 04 add/relu/mul | 已实测并讲解 |
| getitem/pattern lowering handler | Python container 或带标记 target | 直接取项或调用 handler | 实验 07 cond + 当前源码 | 已讲对象对照 |
| 无 lowering 的允许 fallback | target、IR 参数、fallback policy | FallbackKernel/ExternKernel operation | 实验 07 histc | 已实测，待接入本篇 |
| 无 lowering 且不允许 fallback | target 未注册/无 decomposition | MissingOperatorWith/WithoutDecomp | 当前源码 | 已讲失败出口，未篡改注册表运行 |
| 延迟 pointwise 直接输出 | Pointwise 是 output 输入 | run_node/output 触发物化并登记 ComputedBuffer | 实验 04 | 已实测并讲解 |
| 多 user/需要真实输入 | 一份延迟 IR 被输出或 extern 等消费者共享 | realize/realize_hint，后续读稳定 buffer | 实验 18 与 Q20 | 已接入并讲解 |
| 输入 mutation | graph_inputs 当前值不再是原 InputBuffer | 保护旧读、写回原输入 storage、外层重建 alias | 实验 06 与当前 output 源码 | 已讲具体状态 |
| 多输出/控制流 | tuple/list、MultiOutput/Conditional | graph_outputs 各项与父/子图接口 | 实验 07 cond | 已接入并讲解 |
| unbacked symbol/runtime assert | node 前后新增符号或 deferred assert | AssertScalar operation/绑定信息 | 当前源码 | 已讲本层边界 |

## 依据与边界

- 当前源码：`torch/_inductor/graph.py:GraphLowering` 的 `placeholder/get_attr/call_function/run_node/output/finalize` 及调用的 lowering/IR 接口。
- 独立实验 04 现证明 Tensor 输入、int/float/None/unbacked SymInt placeholder、三项普通 lowering、output 物化和 graph_outputs；bool 直返只证明人工最小图的当前失败边界。
- 交叉证据：实验 06 覆盖输入 mutation；07 覆盖 histc fallback、mm extern、cond 父子图；18 对比 pointwise 消费者继续引用计算规则，而 mm 消费者促使输入成为独立 buffer。正文需要把这些证据整理为本层解释，不能只列链接。
- 本篇的“解释器执行”指编译时沿 FX 节点构造 IR，不是用真实 tensor 逐节点执行数值。
- 实际读者反馈尚无；作者复读必须在 K02-R 至 K06-R 完成后重做。

## 作者复读与迁移题

1. 普通整数 placeholder 为什么不产生 InputBuffer？答：placeholder 将它 sympify 为表达式并登记到 graph_inputs；只有 Tensor 才建立 FixedLayout/InputBuffer/TensorBox。
2. cond 父图的 getitem 为什么不查 `lowerings[operator.getitem]`？答：cond handler 已返回 Python list，`call_function` 的第一分支让 Interpreter 直接从容器取项。
3. add 改成 histc 后，为什么不能仍说 env 里只是延迟 Pointwise？答：实际 histc fallback 已登记 FallbackKernel/MultiOutput operation，env 指向这些已物化对象。
4. 用户函数返回 `(old,x)`，为什么实验里的 GraphLowering 普通 outputs 只有 old？答：AOT 图把 x 更新表示为输入 mutation 副作用；GraphLowering 写回原输入，外层合同再构造用户可见 alias。
5. 删除 old 的返回后，copy_ 是否仍必然因“old 是输出”而物化它？答：不是；该理由已经消失。只有仍跨 mutation 读取旧版本的其他消费者才需要由 reader/mutation 关系保护。
6. 一项 Pointwise 既作为输出又送入 mm，哪两个条件都可能促使物化？答：输出需要稳定 storage/stride；mm 属于需要真实输入的消费者。任一存在都可能改变仅内联规则的安排。

答案均可从正文中的具体 env、IR、operation、graph input/output 与 mutation 状态推出。实际读者反馈尚无。

## 当前恢复点

- K01-R 至 K06-R 完成，Q21 达到第 2 层本篇完成条件。
- Q211–Q214 有既有正文，但在第五轮仍保持待回修；本篇“本层完成”不代替叶子验收。
- 根队列下一项为同层 Q22。

## 第五轮重新复核

- 范围：Q211 输入登记、Q212 属性/算子分派、Q213 output/mutation、Q214 run_node 外围四项直接职责与目录一致；计算 IR、存储策略和特殊调用内部仍分别属于 Q22–Q24。
- 前提：正文先区分 Interpreter env、graph_inputs、graph_inputs_original、operations/buffers、graph_outputs 五份状态，再引入 InputBuffer、TensorBox 等名称。
- 过程：Tensor/symbolic/普通标量/None placeholder、普通 lowering、getitem、pattern handler、fallback、missing operator、输出物化、mutation、多输出和 unbacked symbol 都有明确入口与出口。
- 综合：普通链、histc/cond 和 `old=x+1; copy_` 三类案例分别展示 env 与 operation 不同步增长、Python 容器取项、输入写回不等同普通 graph output。
- 理解：只凭正文可判断增加 add 的独立输出、删除 old 返回、把 add 换成 histc、让 Pointwise 同时送 output/mm 时，哪份状态和物化原因改变。实际读者反馈仍为空。
