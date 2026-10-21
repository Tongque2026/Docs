# 特殊调用与控制流篇作者记录

## 所在层、输入与出口

- 队列位置：Q24，第 2 层，父项 Q20；第五轮已按技能重新复核，保留“本层完成”。本状态不替代 Q241 叶子的源码回修。
- 读者前置：Q21 能追踪 GraphLowering env/operation，Q22 理解普通 Loops，Q23 理解物化、layout、alias 和 mutation。
- 本篇输入：不适合普通 Pointwise/Reduction 表达的 ATen/HOP node、已物化参数、候选后端/模板、fallback schema 或控制流子 GraphModule。
- 本篇出口：已选 ExternKernel/TemplateBuffer/MultiTemplateBuffer，FallbackKernel+MultiOutput，或 Conditional/WhileLoop+子 GraphLowering/MultiOutput；Scheduler 只需根据其接口和依赖调度。
- 直接职责：算法选择/模板、fallback 外部调用、控制流子图三类出口。Q241 下一层叶子展开特殊调用分派核心源码；本篇讲清三类协作和接口，不按每个算子建目录。
- 边界：本机只实际选择 CPU extern mm；template/multi-template 来自当前源码和测试，不能写成已执行模板。while_loop 本机未独立运行，必须保留源码/测试证据边界。

## 篇内知识点队列

| 顺序·编号 | 目标与完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| 1·K01-R | 从 mm 参数/layout 走到候选列表、选择与具体 IR 出口 | 4×4 CPU mm + 当前候选源码 | 完成，已实测并复读 |
| 2·K02-R | 区分 ChoiceCaller、ExternKernelOut、TemplateBuffer、MultiTemplateBuffer 以及 epilogue 消费者 | mm→relu 三种可能出口 | 完成；extern 实测，模板按源码界定 |
| 3·K03-R | 走通 fallback fake execution、参数重组、packed result/MultiOutput 和 schema 副作用 | histc | 完成，已实测并复读 |
| 4·K04-R | 走通 cond 两支 Lowering、接口等价检查、父图 Conditional/MultiOutput 和运行选择 | true/false add/sub | 完成，已实测并核对失败出口 |
| 5·K05-R | 解释 while cond/body/carried/additional inputs、稳定循环接口、零轮存储与失败条件 | 当前 ir.py + control-flow tests | 完成，源码机制；未本机运行 |
| 6·K06-R | 比较 graph break/fallback、extern/template、cond/while 并完成综合迁移题 | mm+relu、stride mismatch、mutable fallback | 完成，已复读 |

## 处理情况与案例覆盖

| 情况与条件 | 本层处理 | 出口 | 证据 | 当前深度 |
| --- | --- | --- | --- | --- |
| 当前 CPU mm 选择 extern | 候选选择得到 ATen mm out | ExternKernelOut operation | 实验 07 | 已实测并讲解 |
| GPU/配置允许 template 候选 | 建立 tile/config/benchmark caller | TemplateBuffer | 当前 kernel/select_algorithm 源码 | 已讲机制，未本机运行 |
| 延迟多个模板选择 | operation 已入图但 choice timings/render 尚未最终确定 | MultiTemplateBuffer，Scheduler 结合 epilogue finalize | 当前源码 | 已讲具体状态 |
| template 后有 pointwise epilogue | Scheduler 将合法 epilogue nodes 交给 template render | 一个 template kernel 或独立后继 | 当前源码 | 已讲接口，未运行 |
| 允许 fallback 的函数式 op | fake 执行推导输出，保存 operator/参数结构 | FallbackKernel + MultiOutput | 实验 07 histc | 已实测并讲解 |
| fallback 有 alias/mutation | 读取 schema 建模或拒绝不支持组合 | mutation/alias output 或 NotImplementedError | 当前源码 | 已讲边界 |
| cond 两支合同一致 | 两支分别 Lower，逐项比较接口 | Conditional + MultiOutput | 实验 07 true/false | 已实测并讲解 |
| cond 输出数量/layout/dtype 不一致 | 无法建立统一父输出 | 逐项 assert failure | 当前 `Conditional.create` | 已讲失败条件，未逐项运行 |
| cond branch output alias | 当前实现的 alias 检查两次都读取 true_outputs | true branch 重复 alias 被拒绝；不能据此外推 false-only alias | 当前 `Conditional.create` | 已按当前源码纠正表述 |
| while 合同合法 | Lower cond/body，检查 carried 循环不变量 | WhileLoop + MultiOutput | 当前源码/tests | 已讲机制，未本机运行 |
| while cond/body/carried 不合法 | bool scalar/数量/layout/alias 检查失败 | 明确 assertion/unsupported | 当前源码/tests | 已讲失败条件 |

## 依据与边界

- 当前源码：`kernel/mm.py`、`select_algorithm.py`、`ir.py` 中 template/fallback/Conditional/WhileLoop，以及相关 lowering 注册。
- 实验 07 实际证明 mm→ExternKernelOut、histc→FallbackKernel/MultiOutput、cond 两个子图与父图接口和两种运行结果。
- 实验 11 中命名为 `template_mm` 的 case 实际仍选 `ExternKernelOut`；它不能作为 template 执行证据。正文若暗示本机已运行模板必须修正。
- cond 第二次调用复用已编译函数，所以 Lowering event 只在首次编译记录两支；true/false 数值运行都已核对。
- while_loop 依据当前实现和 `/work/pytorch/test/inductor/test_control_flow.py`；未声称本机硬件/后端执行。
- 实际读者反馈尚无。

## 作者复读与迁移题

1. 图中已有 MultiTemplateBuffer 是否表示 Triton 模板已经最终选定？答：不是；operation 已登记，但 choice timings/render 仍可等 Scheduler 看到 epilogue 后再 finalize。
2. CPU 4×4 mm 和实验 11 的 `template_mm` 名称能否证明 template codegen？答：不能；两者实际 IR 都是 ExternKernelOut。
3. histc fallback 为什么不是 graph break？答：它仍是同一 GraphLowering 的 FallbackKernel operation，输入输出、schema 副作用和前后依赖都留在同一编译图。
4. cond 两支 shape/dtype 相同但 stride 不同，失败发生在哪？答：两支都 Lower 完后，Conditional.create 的结构等价检查在父 Conditional 注册前拒绝。
5. 能否从当前 alias 循环断言 false-only 重复 alias 一定在同一位置被拒绝？答：不能；当前代码两次检查的都是 true_outputs，应按实际实现保留边界。
6. while 的 carried graph input 为什么对应输出禁止复用？答：循环可能零轮，输出仍是原调用者输入；复用该 output buffer 可能破坏外部可见存储身份。
7. mm+relu 出现时怎样判断最终是一体模板还是两个节点？答：检查选择后的 IR、MultiTemplateBuffer finalize/生成记录和 epilogue 合法性，不能从 Python 算子相邻直接推断。

答案均指出候选状态、IR 类型、子图接口或运行时边界。实际读者反馈尚无。

## 当前恢复点

- K01-R 至 K06-R 完成，Q24 达到第 2 层本篇完成条件。
- Q20 的四个第 2 层直接职责 Q21–Q24 均完成第五轮重新复核。
- Q241 保留第 3 层待回修；根队列下一项为同层 Q31。

## 第五轮重新复核

- 范围：算法选择/template、fallback、cond/while 三类直接职责均接收普通 Loops 无法表达的请求，并统一交出带输入、输出、layout、device、依赖和副作用的特殊 IR 接口。
- 过程：当前 CPU mm 实际走 extern；template/MultiTemplateBuffer 明确只按源码和测试说明；histc 从 fake execution 到 packed result/MultiOutput；cond/while 从子图 Lowering 到父接口均走到出口。
- 失败出口：无候选且不允许 fallback、mutable fallback schema、cond 数量/layout/dtype/alias、while bool/数量/carried/layout/alias 都有明确拒绝位置；当前 cond false-only alias 检查的实现边界也被如实保留。
- 事实：实验 07 支持 mm extern、histc fallback 和 cond；实验 11 的所谓 template case 实际仍是 ExternKernelOut，因此未被当作 template 运行证据；while 仅采用当前源码和相关测试。
- 理解：只凭正文可判断 mm+relu 的三种可能出口、cond stride 不一致、while body 改 shape/数量、零轮 carried input 的不可复用原因。实际读者反馈仍为空。
