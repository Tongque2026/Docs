# Lowering 篇作者记录

## 2026-10-22 独立读者审阅后的结构回修

- 审阅依据：`/work/learning/1021/56sol/inductor-fresh-reader-review/06_LOWERING_REVIEW.md`，64/64 个语义单元。
- 正文清除 Q21–Q24 作者队列号，统一使用读图、计算与索引、存储安排、特殊调用四个职责名。
- target 到 lowering 处理器的接口并回读图职责；包装、广播和类型提升细节下放解释器子篇。
- 计算节只保留 ranges、reduction_ranges 和组合规则；TensorBox/StorageBox/ComputedBuffer 只在存储节解释。
- 特殊调用父层只交 mm、fallback、cond 的 operation 和接口；后端候选目录与失败策略下放子篇。
- mutation 父层止于旧读保护结果和 Scheduler 接口，reader 遍历与 realize 机制下放存储子篇。

## 所在层、输入与出口

- 队列项：Q20，第 1 层，父项 Q00；第五轮已按技能重新复核，保留“本层完成”。本状态只覆盖四项直接职责的协作，不替代 Q21–Q24 的内部回修。
- 读者已经掌握：FX node 与结果引用；图优化交出的 GraphModule；根综合案例里 view/pointwise/reduction/mm 最终只形成 `ComputedBuffer buf0` 和 `ExternKernelOut buf1` 两项 operation。
- 本篇输入：一张合法 FX GraphModule、example/fake 输入提供的 size/stride/dtype/device，以及当前 Lowering 注册和图级上下文。
- 本篇出口：`operations`、buffers/inputs/constants、graph outputs，以及 mutation、device、symbolic shape、特殊调用等图级状态；Scheduler 读取这些 IR 接口，不再按 FX 算子名重新解释数学语义。
- 四个直接子职责与目录一致：Q21 逐节点读图并连接结果；Q22 表达普通计算与索引；Q23 决定物化、布局、alias 和 mutation 数据版本；Q24 接入 template/extern/fallback/control flow。
- 下一层分别打开四项内部；本篇只解释它们怎样协作，不提前逐行讲 `run_node`、`Reduction.create`、`StorageBox.realize` 或候选选择函数。

## 篇内知识点队列

| 编号 | 问题、对象与完成条件 | 案例 | 状态 |
| --- | --- | --- | --- |
| K01-R | 四项职责各收到、改变、交出什么；系统图与四个子目录一致 | `add→relu→mul` 的 env 与最终 operation | 完成，已复读 |
| K02-R | 普通计算的三种表示为何不同 | 2×3 pointwise；2×128 real Reduction；2×3 transpose view | 完成，已作具体状态与迁移复读 |
| K03-R | 消费者怎样迫使延迟规则物化并保护数据版本 | `(x+1)→mm`；`old=x+1; copy_` | 完成，已作证据因果与迁移复读 |
| K04-R | 普通循环以外怎样回到同一图出口 | mm extern/template、histc fallback、cond 两支子图 | 完成，已作对象状态、失败出口与迁移复读 |
| K05-R | 多机制综合与迁移验收 | 根篇 `permute→add/relu→sum→mm` 双输出；改变一个消费者 | 完成，已作全链状态与消费者迁移复读 |

Q20 的五个知识点均已完成并在第五轮重新复核；根队列转同层 Q30，不提前进入 Q21。

## 处理情况与案例覆盖

| 情况 | 本层入口 | 四项职责中的关键交接 | 本层出口与消费者 | 证据 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 逐元素链 | add/relu/mul FX 引用 | Q21 连接 env；Q22 组合位置规则；Q23 只物化最终结果 | 一个 ComputedBuffer operation 交 Scheduler | 实验 01/04/17 与生成代码 | 机制已讲并复读 |
| 真实规约 | 2×128 sum(dim=1) | Q22 保留输出范围、规约范围、sum；Q23 赋予结果存储 | Reduction/ComputedBuffer operation | 实验 05 | 机制已讲并作累计状态复读 |
| 纯视图 | 2×3 permute | Q22 改坐标；Q23 保留共享 storage | graph output，0 个新 operation | 实验 05 | 机制已讲并作地址迁移复读 |
| extern/template | mm 及后继 pointwise | Q24 保留候选/调用合同；Q23 准备真实输入 | Extern/Template operation 与后继计算 | 实验 07/17 | 机制已讲并复读 |
| fallback | histc | Q24 保留 operator、参数结构和输出取项 | FallbackKernel + MultiOutput | 实验 07 | 机制已讲并复读 |
| mutation/alias | old 读取后 copy_ 覆盖 x | Q23 保留旧 reader、写回与 storage identity；Q21 区分普通 graph output 和输入 mutation | 旧值 operation、mutation target；用户层 alias 由外层合同恢复 | 实验 06/16、当前源码 | 机制已讲并作因果复读 |
| control flow | cond predicate、两支子图、operands | Q21 取得子图；Q24 分别 Lower 并统一输出接口 | Conditional + MultiOutput | 实验 07、当前源码 | 机制已讲并复读 |
| 空/退化与失败 | size 0/1 reduction、分支接口不一致、unsupported target | 构造器返回恒等/pointwise、明确拒绝或 fallback；不伪造普通成功路径 | 合法 IR、fallback 或异常出口 | 当前源码/测试，Q22/Q24 深入 | 本篇介绍边界，下一层讲实现 |

## 依据、旧状态撤销与反馈

- 源码依据：`graph.py:GraphLowering` 与 `run_node`；`lowering.py` 的注册与普通算子处理；`ir.py` 的 Pointwise/Reduction/View、realize、ExternKernel 和控制流节点。
- 实验证据：01、04、05、06、07、17 分别覆盖生成代码、节点交接、计算 IR、mutation、特殊路径和多机制综合。每项只证明观测到的对象与行为，不自动证明正文讲透。
- 上一版直接把 K01–K04 和 Q21–Q24 标为完成，但没有模块级覆盖表，规约的 2×3 数学例与 2×128 实际 IR 在同一张表中混写，也没有让 fallback/control-flow 从本层入口走到出口；旧完成结论撤销。
- 旧作者复读只覆盖“规则变 buffer、纯 view、覆盖前物化”三项，不能代表整篇。第二轮必须补齐重要路径、综合例和改变条件后的迁移题。
- 实际读者反馈：尚无；不得用作者复读或实验成功替代。

## 第二轮事实与理解验收

- 事实核对：实验 05 的 2×128 sum 实际为 `ComputedBuffer(size=[2]) → Reduction(ranges=[2], reduction_ranges=[128], type=sum)`；正文把 2×3 数学坐标例与 2×128 保留 IR 的证据明确分开。
- 事实核对：实验 07 的 histc 是 `FallbackKernel → MultiOutput`，cond 的两支分别 Lower，父图为 `Conditional → MultiOutput`；正文没有把 fallback 说成 graph break，也没有把“编译两支”说成“运行两支”。
- 事实核对：实验 17 的 operations 状态依次为 `[] → [] → [] → [ComputedBuffer] → [ComputedBuffer] → [ComputedBuffer, ExternKernelOut]`，与综合表一致。

## 第三轮严格复读进度

- K02-R 只凭正文可算：2×128 输入的 `row=1,column=2` 读取扁平位置 130；前三项累计为 387，最终为 24512。改成 `row=0` 后读取 0..127，得到 8128。
- 视图迁移：2×3 输入 permute 后新 `[1,1]` 的偏移为 `1×1+1×3=4`，对应原 `[1,1]`；正文给出的 size、stride 和共享来源足以推出该结论。
- 事实边界：实验 05 直接观察到 Reduction 的 `[2]`、`[128]`、`sum` 以及 permute 的 `[3,2]`、`[1,3]`、0 个 operation；累计数值和地址是基于这些已观察描述的数学推演。
- Q20 仍未完成。当前知识点转到 K03-R：消费者合同怎样迫使物化，以及 mutation 怎样保护旧数据版本。实际读者反馈仍为空。
- 作者只凭正文可答：若删掉综合例的 rows 输出但保留 mm，buf0 仍要成为 mm 可传入参数；若删掉 mm 但保留 rows 输出，buf0 仍要有稳定返回身份；两种消费者都删掉后，正文不再支持沿用同一物化结论。
- 作者只凭正文可答：cond 两支 shape 相同但 dtype 不同，Q24 无法统一父图输出合同，失败在 Lowering；Scheduler 不负责猜运行时选择。
- 作者只凭正文可答：2×3 sum 能教 row/column 坐标，却不能证明当前实现保留 Reduction；2×128 案例通过实际返回类型支持后一个主张。
- 本篇只完成 Q20 的四职责协作，Q21–Q24 的内部仍保持待回修；实际读者反馈仍为空。

## 第五轮重新复核

- 范围：系统图与目录都只有 Q21 读图连接、Q22 计算/索引、Q23 存储/布局/数据版本、Q24 特殊调用四项直接职责；正文没有把辅助函数提升为同层模块。
- 过程：pointwise、真实 reduction、纯 view、extern/template、fallback、mutation/alias、control flow 均从 FX 引用进入，展示四职责中的具体交接，并走到 operations、graph outputs 或明确异常；空/退化行为只在本层承诺出口类别，内部条件留给 Q22/Q24。
- 综合：`permute→add→relu→sum→unsqueeze→mm` 的同一状态表解释六个 FX 节点为何只产生 buf0、buf1 两项 operation，并展示两个消费者分别删除时物化理由怎样变化。
- 事实：实验只支持实际返回类型、operation 序列和数值；2×3 求和明确只是坐标教学例，2×128 才支持当前版本保留 Reduction 的主张。
- 理解：只凭正文可推出 `rows` 仅供 mm、仅供 output、两者都删除三种条件下的不同结论；也可推出 cond 两支 dtype 不同会在建立统一父接口时失败。实际读者反馈仍为空。

### K03-R 事实与理解验收

- 实验 18 隔离消费者：普通链在 add 后仍是未物化 Pointwise、operations 为空；mm 案例在处理 mm 后同时出现 a 的生产 operation 与 mm operation。该证据说明：是否在处理当前消费者时把 a 登记成独立 buffer，取决于消费者是否要求真实存储。
- 实验 06 的 post-grad 图只返回 add，copy_ 表达输入副作用；GraphLowering 在 add 后已有 op0，copy_ 后增加 op1 并把 `arg0_1` 加入 mutated buffers。生成的 Inductor `call` 返回 `(buf0,)`，同时把结果写入原 `arg0_1`。
- 因果修正：实验 06 不能证明 old 是被 copy_ 首次物化，因为 old 同时是图输出。旧正文的该表述已删除；“其他旧 reader 也会先物化”改由当前 `mark_buffer_mutated` 源码的 user 遍历支持。
- 迁移一：把 mm 换成可继续组合的乘 2，add 后 operations 仍为空；最终输出才登记一项 operation。
- 迁移二：函数修改 x 但不把 x 作为普通 graph output，mutation operation 仍必须写回原 storage；返回表中没有 x 不会取消副作用。
- K03-R 完成。当前知识点转到 K04-R：让 mm/template、histc fallback 和 cond 从 FX 入口分别走到本层出口，并补齐拒绝路径。Q20 尚未完成。

### K04-R 事实与理解验收

- 实验 07 直接观察到：本机 mm 为一个 `ExternKernelOut`；histc 为 `FallbackKernel buf0 → MultiOutput buf1`；cond 两支各有一个 ComputedBuffer，父图为 `Conditional buf0 → MultiOutput buf1`，后续 getitem 不增加 operation。
- 源码边界：mm 候选集合依设备与配置变化；无有效候选且不允许 ATen fallback 时失败。cond 逐项检查两支输出数量、size、stride、device、dtype、offset，并拒绝不支持的输出 alias。mutable 且不能 auto-functionalize 的 fallback schema 明确失败。
- 迁移一：把 cond 假分支改成返回两个 Tensor，输出数量检查在父 Conditional 建立前失败，根图不会得到一个“部分有效”的 Conditional。
- 迁移二：把 histc 换成返回两个 Tensor 的 functional fallback，packed 调用仍是一项，但会有两份 MultiOutput 引用；不能按“一个 fallback 永远恰好两个 operation”推断。
- K04-R 完成。当前知识点转到 K05-R：复读综合案例中六个 FX 节点、四项职责与两个 operation 的完整因果，并改变消费者合同验收。Q20 尚未完成。

### K05-R 与 Q20 完成审计

- 实验 17 重新采集的 operations 序列为 `[] → [] → [] → [ComputedBuffer buf0] → [buf0] → [buf0, ExternKernelOut buf1]`；graph_outputs 为 buf0、buf1，数值分别是 `[24512,24640,24768]` 与 `[[49280,49408]]`。
- 四项职责在同一案例中的因果可由正文推出：Q21 维持六步结果引用；Q22 建立 view/pointwise/reduction；Q23 在输出与 mm 合同出现处给 rows 稳定身份；Q24 建立 mm 外部工作。
- 消费者迁移：删去 rows 的普通返回但保留 mm，buf0 仍因 mm 参数合同存在；删去 mm 但保留 rows，buf0 仍因图输出存在；两者都删去后，没有依据继续断言相同的 buf0，必须按新后继重新 Lower。
- 当前层主要情况逐项复读：pointwise、真实 reduction、纯 view、extern/template、fallback、mutation/alias、control flow 均有入口、关键状态和本层出口；空/退化与 unsupported 在本篇只承诺合法 IR、fallback 或异常三类边界，构造条件留给 Q22/Q24。
- 事实证据和理解验收分开：实验 05/06/07/17/18 支持实际对象及状态；上述迁移题只依据正文回答。实际读者反馈仍为空。
- Q20 达到第 1 层完成条件；Q21–Q24 的内部算法和叶子源码仍保持待回修，不能由 Q20 完成状态代替。

