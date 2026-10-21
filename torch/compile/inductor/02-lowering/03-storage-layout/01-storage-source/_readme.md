# realize 与布局决定源码叶子作者记录

> 第五轮状态：叶子完成。第五轮修正正文末尾仍指向“待审计”的陈旧状态，并重新按物化、冻结、访问启发式、外部合同和 27 个方法验收。

## 本篇定位、读者与边界

- 队列：Q231，深度 3，父项 Q23；负责从延迟计算到具名 buffer，并把 FlexibleLayout 冻结为实际布局的源码合同。
- 读者起点：Q221 已能读懂 Pointwise/Reduction/View；Q23 已理解物化、布局、alias、mutation 的本层协作。
- 学完目标：能判断一次结果是否已具备 storage 身份、启发式是否触发 realize、何时还能改 layout、读访问怎样形成 fill order，以及外部 stride 合同怎样优先于启发式。
- 边界：mutation/NonOwningLayout 仅作为限制条件引用；Scheduler 依赖和 wrapper 物理分配归后续模块。

## 声明范围与篇内队列

| 编号 | 完成条件 | 当前源码 | 案例 | 状态 |
| --- | --- | --- | --- | --- |
| K01 | 完整解释 StorageBox 身份判断、realize 与三类启发式 | `is_input_buffer` 至 `num_reads` | 延迟 add、已存储输入、多 reader/heavy op | 完成并复读 |
| K02 | 完整解释 Buffer 的普通冻结及 order/fill/exact/same-order 接口 | `freeze_layout*` | `[2,3]` 的多种 stride 合同 | 完成并复读 |
| K03 | 从 ComputedBuffer 的 MemoryDep 推到 fill order 与 decide_layout | `get_fill_order/decide_layout` | 行读、列读、广播、无 read | 完成并复读 |
| K04 | 完整解释 extern stride constraint 如何复用、冻结或复制输入 | `ExternKernel.require_strides` 及直接入口 | contiguous、transpose、view/非 view | 完成并复读 |
| K05 | 整篇 AST、处理覆盖、实验与理解验收 | 全声明范围 | 综合改变消费者案例 | 完成 |

## 依据

- `/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402` 的 `torch/_inductor/ir.py`。
- 既有实验 18 覆盖 chain、mm consumer、旧读跨 mutation；本叶子按目标机制补 direct probes，不用最终数值代替 IR 状态。
- 不使用旧教程。运行、局部真实 IR、mock 和源码/数学推演分别登记。

## K01 事实与理解验收

- StorageBox 从 `is_input_buffer` 到 `num_reads` 的八个方法完整保留，逐方法与当前产品 AST 8/8 等价。
- 正文分开已存储对象的幂等出口和 Loops→ComputedBuffer 的登记出口，并保留 origin/traceback/origins 的状态转交。
- 三种启发式分别给出条件与不动作出口；没有把 hint 或 reuse 写成无条件物化。
- 延迟 add 的实际类型变化已有实验 18 支持；CPU exp/sigmoid 分支目前按源码推演，不登记为本轮运行实测。
- 迁移题要求区分重复 realize、hint 不触发、reuse 不触发和非法 IR 失败；正文足以回答。实际读者反馈尚无。

## K02 事实与理解验收

- 完整保留 FlexibleLayout 的九个 stride 构造/转换方法和 Buffer 的五个冻结入口，方法级 AST 14/14 等价。
- 独立实验 21 对 `[2,3]` 直接调用产品接口，实际得到 default/fill-row/stride-order/same-order=`[3,1]`、fill-column=`[1,2]`、exact=`[10,2]`。
- 同一实验确认 FixedLayout 上再次调用定向冻结触发 AssertionError；这证明接口前置条件，不外推 padding 行为。
- 正文区分 fill order、stride order、same order、exact 数值合同，并说明 NonOwning/MultiOutput 为什么跳过普通冻结。
- 迁移题覆盖三维 fill order、参考顺序、二次冻结失败和非拥有 layout；正文足以回答。实际读者反馈尚无。

## K03 事实与理解验收

- 完整保留 ComputedBuffer 的 `get_fill_order/decide_layout`，与当前产品 AST 2/2 等价。
- 实验 21 使用真实 Pointwise/ComputedBuffer/MemoryDep，覆盖行读、列读、广播和无读取：fill order 分别为 `[1,0]`、`[0,1]`、`[1,0]`、None。
- 正文从四个具体 index 算到 stride hints 和最终 stride，区分“相对访问偏好”与“照抄输入 stride 数值”。
- 明确 StarDep 排除、reduction vars 置 0、Scan/Sort reindex 和已固定 layout 不动作四个条件。
- 迁移题覆盖 FixedLayout 优先、仅 StarDep、成比例 stride 和 Scan/Sort；正文足以回答。实际读者反馈尚无。

## K04 事实与理解验收

- 完整保留 `require_strides/require_exact_strides/require_stride_order`，与当前 ExternKernel AST 3/3 等价。
- 实验 21 覆盖零元素、Flexible 原地冻结、Fixed 已满足、Fixed 不满足复制、size-1 exact 无关差异五条真实 IR 路径。
- MutationLayout 与 Generic BaseView 分支按当前源码解释，明确 view 转换后仍需递归检查；本轮不冒充运行实测。
- 正文区分“copy 分支”与“最终独立 copy kernel”，并保留 Flexible size-1 歧义时复用当前 order 的产品注释。
- 迁移题覆盖有效 exact 差异、view 转换后不满足和零元素短路；正文足以回答。实际读者反馈尚无。

## 整篇严格审计

| 要求 | 权威证据 | 结果 |
| --- | --- | --- |
| 声明源码完整 | 五个类中的 27 个声明方法逐项 AST 比较 | 27/27 等价 |
| 物化两类出口 | 实验 18 与 K01 源码状态 | 已存储幂等；Loops 登记 ComputedBuffer |
| 布局冻结合同 | 实验 21 六种 stride 与失败断言 | 与正文表格一致 |
| 访问启发式 | 实验 21 四种真实 MemoryDep | fill order/最终 stride 一致 |
| 外部约束优先 | 实验 21 五条 require_strides 路径 | 复用、原地冻结、复制均覆盖 |
| 理解迁移 | K01–K04 每节迁移题 | 作者仅凭正文可回答；读者反馈尚无 |

源码按职责跨越 `FlexibleLayout`、`Buffer`、`ComputedBuffer`、`ExternKernel`、`StorageBox`，不是一个连续类范围；每个类内均保持产品方法顺序，正文按“先产生 storage，再冻结，再选择，再施加外部合同”的运行因果组织。

## 当前点

## 第五轮叶子验收

- **完整源码**：StorageBox、FlexibleLayout、Buffer、ComputedBuffer、ExternKernel 五个类中的 27 个声明方法逐项 AST 等价；各类内保持产品方法顺序。
- **分支与出口**：已存储/延迟 realize、三种启发式，普通/定向/特殊 layout 冻结，有/无 MemoryDep、StarDep、Scan/Sort、FixedLayout，extern 的零元素/Flexible/Fixed/Mutation/View/copy 全覆盖。
- **具体状态**：`[2,3]` 六种 stride 合同有明确中间规则；四类真实读取式从物理 index 算 stride hints/fill order；五条 require_strides probe 区分复用、原地冻结与复制。
- **理解迁移**：正文能判断重复 realize 幂等、hint 不触发、三维 fill order、FixedLayout 优先、成比例 stride 不照抄、有效 exact 差异必须 copy、view 转换后仍需递归检查。
- **证据边界**：实验 18 支持实际类型变化，实验 21 支持真实 IR 接口；CPU heavy-op 启发式和未跑分支按源码推演，不把接口 probe 写成最终 kernel 形态。
- **续写一致性**：正文末尾已改为审计完成并转 Q241，与本记录和根队列相同。

Q231 第五轮叶子复核完成。根队列转 Q241：特殊调用分派核心源码。
