# 计算 IR 构造源码叶子作者记录

> 第六轮状态：通过隔离冷读 A 级验收。七名全新读者只读本篇 README，评级依次为 B、B、B、B、B、B、A；每轮由主智能体按读者断点回修，读者不参与写作。

## 第六轮真实读者反馈与整改

- 评审报告：`91_IR_CONSTRUCTION_COLD_REVIEW_1.md` 至 `97_IR_CONSTRUCTION_COLD_REVIEW_7.md`，保存在教程外的 `inductor-fresh-reader-review/`。
- 第一轮发现 K03 编号与物理顺序冲突、关键对象迟到、跨千行缺状态账本、全常量与 existing-ranges 等案例未闭合；正文增加统一对象图和 K03-1～K03-5 账本，并按阅读顺序重编号。
- 第二至第四轮继续补齐 TensorBox/StorageBox 包装、realize 前后状态、existing-ranges 搜索、Generic View loader、动态 reshape 连续化与两层 dtype；对 0 项 dtype 的源码参数差异保留精确说明。
- 第五至第六轮纠正两项过度概括：existing-ranges 只检查首次发现 Reduction 的 frontier，不检查整个依赖闭包；Triton 低精度累加提升受 `config.triton.codegen_upcast_to_fp32` 控制。补出最终 Reduction 由 K02 调用方物化，以及全常量通过 `BaseConstant.get_size()=()` 构造标量 Pointwise。
- 第六轮还发现对象层级表述过宽：`TensorBox.create` 产生 StorageBox 包装，但并非所有 TensorBox 永远有双层 data。正文补入 `MutableBox` 转发，并区分 helper 可接收的 `TensorBox(ComputedBuffer)`、普通 Loops 包装、BaseView、ReinterpretView 与 copy_input 路径。
- 第七位读者闭卷恢复全部 K 的输入、动作、出口和指定案例，确认主要分支可迁移，评级 A。两处实验观测仅作为旁证，正文机制不依赖读者读取实验文件。

## 本篇定位、读者与边界

- 全系列队列：Q221，深度 3，父项 Q22；这是 Pointwise、Reduction、View 构造职责的源码叶子。
- 读者起点：已从 Q22 理解位置规则、规约的两套坐标和 view 共享语义，但还不能从 lowering 参数推到具体 IR 构造出口。
- 学完目标：能代入具体 shape/index，判断 Pointwise loader、Reduction 的退化/单层/多层出口，以及 View 的 ReinterpretView/GenericView 出口，并沿完整源码解释原因。
- 边界：buffer 何时物化及布局算法归 Q231；调度怎样排列这些 IR 归 Q31/Q32；本篇只讲构造阶段交出的 IR 与坐标规则。

## 范围与依据

- 产品版本：`/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 最终职责：Pointwise、Reduction、View 从 lowering 参数到 IR/TensorBox 出口的构造核心。
- 当前资料：`torch/_inductor/lowering.py`、`torch/_inductor/ir.py`、父篇实验 `experiments/05_compute_ir_paths.py`。
- 不使用旧教程。运行证据与源码/数学推演分开记录。

## 处理情况覆盖

| 职责 | 主要情况与失败出口 | 具体案例 | 证据深度 |
| --- | --- | --- | --- |
| Pointwise | 常量提升、dtype、broadcast、空/size-1、loader/store | 2×3 加常数与广播 | 实验 05 + 源码/坐标推演 |
| reduction 参数准备 | 负轴、空轴、重复轴、keepdim、dtype | 2×3 dim=1 与多轴迁移题 | helper 源码 + 数学推演 |
| Reduction.create | 0/1/小静态/普通/已有 ranges/多层 | 0、1、3、128、1000 项 | 实验 05 证明普通 Reduction；其余按源码推演 |
| split/multilayer | 禁止 split、inner/outer、mask、两层 buffer、dtype 提升 | 16384 公式例与 1000 项 split=32 | 当前源码与具体状态；不冒充设备实测 |
| Permute/View | 固定 layout、延迟规则、空/动态、合并/拆分/fallback | transpose、两类 reshape、三组 direct reindex | 完整编译 + 产品 helper 直接探针 |

## 篇内队列

| 编号 | 完成条件 | 主要源码 | 案例 | 状态 |
| --- | --- | --- | --- | --- |
| K01 | 完整解释常量提升、loader 捕获、ranges、Pointwise 包装与 store | `promote_constants`、`make_pointwise`、`Loops.create/_index`、`Pointwise` | 2×3 tensor+2、bool/fp64/低精度/空范围 | 完成 |
| K02 | 完整解释 axis 规范化、kept/reduced 坐标重组及 make_reduction 出口 | `_validate_reduction_axis`、`_make_reduction_inner`、`make_reduction`、`sum_` | 2×3 dim=1、keepdim | 完成 |
| K03 | 完整解释 Reduction 对象、create 的空/1/小静态/拆分/普通路径 | K03a 对象与组合；K03b create/unroll；K03c split/multilayer | 0、1、3、128 项规约 | 完成并复读 |
| K04 | 完整解释固定 layout 快速路径与延迟 reindex 路径 | `PermuteView.create/make_reindexer`、`View.create` 及直接 helper | transpose、reshape、动态 size | 完成并复读 |

### K04 子队列

| 子点 | 完成条件 | 状态 |
| --- | --- | --- |
| K04a | 完整保留 PermuteView 与 View 出口选择，解释 storage 快速路径、空/动态/延迟路径 | 完成 |
| K04b | 完整保留动态 reshape indexer，逐步推演合并维、拆分维与 fallback | 完成 |

## K01 完整性与复读

- 完整保留 `promote_constants`、`make_pointwise`、`Loops.create`、`Loops._index`、`nop_loader_fn` 与整个 `Pointwise` 类。
- 具体 `[1,2]` 位置从两个 loader 走到 store；没有把 Lowering 阶段写成已读取真实数值。
- 复杂分支覆盖 alpha、bool、GPU float64、低精度 barrier、Triton fallback、size 0/1。
- 迁移题：rank 不同且未广播的第二输入在维数断言失败；正文足以推出。
- 第六轮读者反馈：对应缺口已纳入 91–97 号报告，第七位读者评级 A。
- AST 有序覆盖审计通过：`promote_constants` 37 行、`make_pointwise` 79 行、`Loops.create` 9 行、`Loops._index` 5 行、`nop_loader_fn` 5 行、`Pointwise` 18 行，均为去空行和类级缩进后的非教学产品源码行数。

## K02 完整性与复读

- 增补直接 helper `_validate_reduction_axis`，完整覆盖负轴、空 axis、标量和重复轴失败出口。
- 2×3 row=1 逐项展示 index/reduction_index 回填成原坐标，不只列 ranges 名称。
- keepdims 与 int32→int64 各有完整状态变化；Reduction/Pointwise 出口按实际类型区分。
- 迁移题：shape `[2,3,4]` 规约 `[0,2]` 时 loader 槽位为 reduction、kept、reduction；正文足以推出。
- 第六轮读者反馈：对应缺口已纳入 91–97 号报告，第七位读者评级 A。
- AST 有序覆盖审计通过：`_validate_reduction_axis` 16 行、`_make_reduction_inner` 43 行、`make_reduction` 16 行、`sum_` 7 行；`sum_` 的注册 decorator 另行确认存在。

## K03 子队列

| 子点 | 完成条件 | 状态 |
| --- | --- | --- |
| K03a | 完整保留组合函数与 Reduction 字段/执行接口，推演 sum、argmax tie、Welford | 完成 |
| K03b | 完整保留 `_unroll_reduction_fn` 与 `Reduction.create`，解释全部出口 | 完成 |
| K03c | 完整保留 `num_splits` 与 multilayer 直接 helpers，推演 split 状态 | 完成并复读 |

### K03a 完整性与复读

- 完整保留 `REDUCTION_COMBINE_FN`、`get_reduction_combine_fn` 和 Reduction 类从字段到 `constant_to_device` 的连续前缀。
- sum 用两套变量走到只含外层 index 的 store；argmax 用相等值和索引 2/5 推演 tie；Welford 代入两组具体状态。
- 明确 dtype、src_dtype、hint 各自用途，未把 hint 写成数学组合规则。
- 迁移题：ranges/reduction_ranges 中 size=1 的索引均变 0；正文足以推出。
- 第六轮读者反馈：对应缺口已纳入 91–97 号报告，第七位读者评级 A。
- AST 有序覆盖审计通过：组合函数 50 行；Reduction 连续前缀的十个方法分别通过；简单组合表 any/max/min/prod/sum/xor_sum 六个 key 均存在。

### K03b 完整性与复读

- 完整保留 `_unroll_reduction_fn` 与 `Reduction.create` 的全部条件、产品注释和出口。
- 0/1/3/128 四种项数分别走到单位元 Pointwise、唯一位置 Pointwise、静态展开和 split 决策。
- argmax 多维索引用 `[2,3]` 的 `[1,2]→5` 展示 flatten，不只说“保留索引”。
- 明确 2×128 CPU 实验只证明实际对象类型，不用它推断其他设备的 split。
- 迁移题：小静态规约但 scalar output 不展开；正文足以推出。
- 第六轮读者反馈：对应缺口已纳入 91–97 号报告，第七位读者评级 A。
- AST 有序覆盖审计通过：`_unroll_reduction_fn` 33 行、`Reduction.create` 128 行；六类关键出口条件另行检索全部存在。

### K03c 子队列

| 子点 | 完成条件 | 状态 |
| --- | --- | --- |
| K03c1 | 完整保留 `num_splits`，推演 gates、inner/outer 公式和读索引投票 | 完成 |
| K03c2a | 完整保留 default、second-step hint 与普通 wrap loader，推演第一层 ranges/mask | 完成 |
| K03c2b | 完整保留 multilayer helper、普通入口与 existing-ranges 变体，推演两层连接 | 完成 |

#### K03c1 完整性与复读

- 完整保留 224 行 `num_splits` 方法，包含两套内部公式、已有 Reduction ranges 复用和 layout 改变后的二次采集。
- 以假定 80 SM 明确标注公式推演，不冒充当前设备实测。
- reader 投票用实际 stride hint 解释 inner/outer，包含广播无完整索引和平票出口。
- 迁移题：argmax 或 unbacked 动态规模不会 split；正文足以推出。
- 第六轮读者反馈：对应缺口已纳入 91–97 号报告，第七位读者评级 A。

#### K03c2a 完整性与复读

- 完整保留 `default_accumulator`、`default_value`、`_multilayer_second_step_hint` 和 `_multilayer_wrap_loader`，分别说明单位元、第二层 hint 与第一层坐标重组。
- 用 1000 项、split=32 推演 block 2/r=5 到线性 69，以及最后一块 992..1023 中 24 个越界位置怎样经 mask 取 sum 单位元 0。
- 明确 `dynamic_reshape_indexer` 把线性位置还原成原多维规约坐标；split 改分组而不改原 loader 的逻辑元素。
- 迁移题：1001 项最后一块 9 项有效；prod 必须以 1 填越界；正文足以推出。
- AST 方法级核对：四个方法与当前产品实现逐一等价（4/4），只增加教学注释并去掉类级缩进。
- 本子点是源码与数学状态推演，尚未构造真实设备触发 multilayer 的运行实验；不登记为实测。

#### K03c2b 完整性与复读

- 完整保留 `_multilayer_wrap_loader_existing_ranges`、`create_multilayer_helper`、`create_multilayer`、`create_multilayer_existing_ranges`；方法级 AST 与产品逐一等价（4/4）。
- 用 `ranges=[4]`、1000 项、split=32 从第一层 `[4,32]×[32]` 推到中间 buffer `[4,32]`，再推到第二层 `[4]×[32]`；具体展示 `[row=2,block=7]` 的读取。
- 解释半精度中间值提升到 fp32、`intermediate.realize()` 建立层间边界，以及第二层为何只规约 `new_ranges` 去掉原输出前缀后的部分。
- existing-ranges 用原 `[2,3,4]` 重排为新外层 `[2]`、新规约 `[3,4]`，明确 `split=-1` 只保留 hint，不表示负 split。
- 迁移题覆盖多维原输出前缀、缺少物化、dtype 提升和 original_ranges 断言；正文足以推出。
- 本子点仍是当前源码与具体状态推演，不声称当前 CPU 实验实际触发 multilayer。

## K04a 完整性与复读

- 完整保留 `PermuteView` 的 `create/_map_neg_dims/get_size/make_reindexer`，以及 `View.handle_negative_index/create/resolve_negative_size`；七个方法与当前产品 AST 7/7 等价。
- 实验 05 新增连续 reshape 和延迟 Pointwise reshape：前者在 create 时成为同 InputBuffer 的 ReinterpretView；后者 create 时为 `View→Pointwise`，被 output-view 合同物化后才成为 `ReinterpretView→ComputedBuffer`。
- transpose 完整编译仍观测为 size `[3,2]`、stride `[1,3]`、同 InputBuffer、0 个新增 buffer/operation；正文用新 `[2,1]` 与旧 `[1,2]` 的同 offset=5 解释原因。
- 空 shape 分支按源码解释 fake reindex 的接口用途；该 case 在当前编译管线没有留下 GraphLowering 事件，因此不登记为内部类型实测。
- 迁移题覆盖负 dims、`-1` 推导、无效 view 消除和非连续延迟 reshape；正文足以推出。
- 第六轮读者反馈：对应缺口已纳入 91–97 号报告，第七位读者评级 A。

## K04b 完整性与复读

- 完整保留 `dynamic_reshape_indexer` 与 `_dynamic_reshape_indexer`，两个方法与当前产品 AST 2/2 等价。
- 分别推演 `[6]←[2,3]` 的新维合并、`[2,3]←[6]` 的 ModularIndexing 拆分，以及 `[2,3]←[3,2]` 经 flat `[6]` 的 fallback 组合。
- 实验 05 增加直接产品方法探针：输入 `[1,2]` 得旧线性 `(5,)`，flat 5 得旧 `(1,2)`，fallback 新 `[2,1]` 也得旧 `(1,2)`；三项断言通过。
- size-1 维用旧 `[2,1,3]` 的中间坐标恒为 0 解释，没有把 view 写成数据复制。
- 迁移题覆盖三维合并/拆分、size-1 和总元素数失败边界；正文足以推出。
- 第六轮读者反馈：对应缺口已纳入 91–97 号报告，第七位读者评级 A。

## 整篇严格审计（完成）

- 已把 K01–K04 的教程代码块按声明归属解析：lowering 顶层函数、Loops、Pointwise、Reduction 和 View 共 36 个函数/方法，逐项 AST 36/36 等价。
- `Pointwise` 整个类 AST 等价；`REDUCTION_COMBINE_FN` 带类型标注的完整表也与产品 AST 等价。
- Reduction 的四个字段和全部 19 个方法均已出现，没有缺失方法。正文已重排为产品类顺序：prefix → `num_splits` → `_unroll_reduction_fn` → `create` → default/multilayer；程序化提取的方法名序列与产品完全相同。
- 重排后已修正“后续 K03c”等方向性引用。稳定编号不因重排改变；正文明确阅读顺序以物理位置为准。

## 当前衔接

## 第五轮叶子验收（第六轮冷读前的作者复核）

- **完整源码**：Pointwise、Reduction、View 的 36 个声明函数/方法逐项 AST 等价；Pointwise 整类、Reduction 四字段/19 方法及组合表完整，物理顺序与产品类一致。
- **分支与出口**：常量提升与 dtype/broadcast，axis/keepdim，0/1/小静态/普通/split/existing-ranges 规约，固定 storage/空/动态/延迟/fallback view 均有出口。
- **具体状态**：2×3 pointwise 与 row reduction 展开坐标；0/1/3/128/1000 项分别对应退化、展开、普通与两层公式；transpose、reshape 合并/拆分/fallback 都代入合法位置。
- **理解迁移**：正文能推出 rank 未广播断言、多轴 loader 槽位、小静态 scalar output 不展开、1001 项 split 尾块、prod mask 单位元、三维 reshape 合并/拆分与总元素数失败。
- **证据边界**：CPU 实验只证明实际保留的 Pointwise/Reduction/View 路径；GPU split/multilayer 按当前源码和数学状态推演，不标成设备实测。
- **续写一致性**：正文末尾已从“下一步审计”改为“审计完成，转 Q231”，与本记录和根队列一致。

Q221 第六轮隔离冷读完成并通过 A 级验收。根队列转 Q231：realize 与布局决定完整源码。
