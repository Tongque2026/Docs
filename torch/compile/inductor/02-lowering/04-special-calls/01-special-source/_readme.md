# Q241 特殊调用分派核心源码备课记录

> 第六轮状态：经五位相互隔离的冷读者逐轮反馈与改写，最终达到 A 级验收。

## 读者、层次与边界

- 读者已经从 Q24 知道三类本层出口：候选选择产生 extern/template，fallback 保留 operator，cond/while 保留子图接口；尚不知道这些出口由哪些语句建立。
- 本篇是第 3 层叶子。输入是已经建立的 caller、operator 参数或控制流子图；出口是登记进 `GraphLowering.operations` 的具体 IR operation 及可供父图引用的 `TensorBox`/`MultiOutput`。
- 本篇完整解释“从分派决定到特殊 IR 出口”的核心实现，不展开候选 benchmark cache 的全部算法、每个 mm template、wrapper codegen 和 Scheduler epilogue 融合循环；它们分别属于 Q24 上游候选构造、Q42/Q34/Q43。
- 产品基线：`/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。

## 声明范围与产品顺序

声明范围按产品文件和行序固定，共 9 个声明；正文最终必须逐个保留完整源码，禁止省略号。

| 顺序 | 当前源码声明 | 职责 |
| --- | --- | --- |
| 1 | `select_algorithm.py:TritonTemplateCaller.output_node` | template caller 变成 TemplateBuffer |
| 2 | `select_algorithm.py:ExternKernelCaller.output_node` | extern caller 三路建立具体 operation |
| 3 | `select_algorithm.py:autotune_select_algorithm` | 建立/复用 selector，并补延迟模板配置 |
| 4 | `lowering.py:make_fallback` | 把 operator 注册为 FallbackKernel lowering |
| 5 | `ir.py:MultiTemplateBuffer` 全类 5 个方法 | 保存 lazy timings，临时/最终安装 render，取最小候选 |
| 6 | `ir.py:FallbackKernel.create` | fake execution 后重建嵌套输出与 MultiOutput |
| 7 | `ir.py:Conditional.__init__/create/codegen` | 两支 Lower、合同检查、父 operation 与输出 |
| 8 | `ir.py:WhileLoop.__init__/create/codegen` | cond/body Lower、循环不变量、零轮复用保护 |
| 9 | `ir.py:_has_aliased_buffers` | control-flow 输出 alias 判定辅助；正文放在首次消费者之前 |

物理文件顺序不同于教学依赖时，正文先给阅读地图，源码附录仍分别保持各文件产品顺序；不能把语句重排后冒充完整源码。

## 篇内知识点队列

| 编号 | 完成条件 | 案例与证据 | 状态 |
| --- | --- | --- | --- |
| K01 | 能由同一 layout/inputs 追踪 Triton caller、extern caller 和 MultiTemplateBuffer 的三种状态，解释何时已有 operation、何时 render 尚未固定 | 实验 07 CPU mm；产品 caller/MultiTemplateBuffer 源码；实验 22 timings 推演 | 完成，已复读 |
| K02 | 能从 `make_fallback(histc)` 追踪注册、fake output、packed operation、递归输出重建和失败出口 | 实验 07 histc；实验 23 sort tuple；`FallbackKernel.create` | 完成，已复读 |
| K03 | 能逐句追踪 cond 两支图、接口比较、Conditional/MultiOutput，并判断 stride/alias 失败发生前后的状态 | 实验 07 cond；`_has_aliased_buffers`/Conditional 完整源码 | 完成，已复读 |
| K04 | 能逐句追踪 while 的 cond/body、carried 不变量和零轮 `never_reuse_buffers`，并与 cond/fallback 比较 | 实验 24 CPU 三轮/零轮；当前完整源码和 `test_control_flow.py` | 完成，已复读 |

## 处理情况覆盖

| 条件 | 状态变化 | 出口/失败 | 归属 |
| --- | --- | --- | --- |
| Triton caller 已选 | caller 中的 layout/inputs/render 复制进 TemplateBuffer | `TensorBox(TemplateBuffer)` | K01 |
| extern caller，ABI fallback / 自定义 creator / 普通 extern | 三个互斥分支各建一种 inner IR | `TensorBox(inner)` 或断言 | K01 |
| 多模板延迟决定 | timings 首次读取才计算；render 可临时交换或最终安装 | MultiTemplateBuffer 保持/完成 | K01 |
| fallback 输出为 None/Tensor/嵌套容器/int/SymInt | fake 结果决定 layout，递归建立输出引用 | packed FallbackKernel + 输出树；未知类型 assert | K02 |
| cond 合同一致/不一致 | 两支 Lower 后比较数量、layout、device、dtype、offset | Conditional+MultiOutput / assert | K03 |
| while 合同一致/不一致 | cond 标量 bool；body 对齐 carried；图输入 carried 加复用保护 | WhileLoop+MultiOutput / assert | K04 |

## 证据与验收

- 事实：实验 07 只证明当前 CPU 4×4 mm 选择 `ExternKernelOut`，histc 产生 `FallbackKernel/MultiOutput`，cond 两支产生 add/sub 子图且运行结果正确；不证明 Triton template 和 while 已在本机执行。
- 源码：九项声明用于证明分支条件、对象字段、登记顺序和失败出口。最终用 AST 去除教学注释后逐声明核对等价与顺序。
- 理解：每个 K 点至少含一个改变输入或条件的迁移题，并在本记录中仅依据正文作答。K01 未完成前不进入 K02。

## 第六轮隔离冷读整改

- 第一位读者评级 C：K01 缺 selector 到最终候选的闭环，extern 收束被写成不存在的 `finalize_as_extern_caller`；K03 将只检查 true 输出误写成两支检查。
- 第二至第四位读者连续评级 B：依次补齐 extern 替换的名字与 users 交接、packed/MultiOutput 的包装层、Subgraph 首次定义、additional input 完整案例、wrapper 运行时分支证据，以及 FX node/FakeTensor 的源码差异。
- 第五位全新读者只读最终 README，四项闭卷推演均通过，评级 A。其最后一条精修意见是：当前 alias 循环第二轮不可能单独抛错，正文已按真实控制流修正。
- 完整报告保存为 `102_SPECIAL_CALLS_COLD_REVIEW_1.md` 至 `106_SPECIAL_CALLS_COLD_REVIEW_5.md`。所有读者均未读取本记录、其他教程、源码、实验、Git 或前轮报告，也未修改教程。

## 当前恢复点

## 第五轮叶子验收

- **完整源码**：两个 caller、selector、make_fallback、MultiTemplateBuffer、FallbackKernel.create、alias helper、Conditional、WhileLoop 九项声明 AST 9/9 等价；同文件声明保持产品顺序。
- **分支与出口**：Triton/extern 三分支、多模板 lazy/swap/finalize、fallback packet/overload/HOO 与递归输出树、cond 两支合同、while cond/body/carried 合同及 codegen 出口全覆盖。
- **具体状态**：4×4 mm 从 caller 到 ExternKernelOut；两项 timing 只求值一次；sort tuple 由一次 packed 调用拆两个 MultiOutput；cond add/sub 两支与 while 三轮/零轮均走到父 operation 和输出。
- **理解迁移**：正文能判断 get_min_choice 尚未安装 render、字符串 fallback leaf 失败前 packed 可能已建、cond stride 不同在父节点前失败、while graph input carried 的输出为何进入 never_reuse。
- **当前限制与证据边界**：Conditional alias 循环实际始终检查 true_outputs；实验 07/22/23/24、源码失败分支和未运行 Triton 分别登记，不用 CPU 数值证明 GPU template。

Q241 第六轮隔离冷读 A 级验收完成。下一项按根广度优先队列转 Q311，进入依赖构造核心源码。
## 读者复读后的实现观察

- 当前 Conditional alias 循环会遍历 true/false 名称，但 `_has_aliased_buffers` 的实参始终是 `true_outputs`。这不是旁支审计信息，而是决定 false-only alias 是否会被拒绝的当前实现行为，正式教程已直接说明。
