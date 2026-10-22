# GraphLowering 输出合同源码叶子作者记录

## 第六轮：隔离冷读整改

- 第一名无历史冷读者只读原 README，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/86_OUTPUT_SOURCE_COLD_REVIEW_1.md`，评级 B。
- 读者能闭卷推演普通 tensor、无关 stride、输入修改并返回、修改不返回和 finalize 主路径，但指出完整源码先于对象模型与案例：首次读 `realize_input`、两次 `.data`、stride helper 和 `finalize` 时只能翻译语法，真正的对象链在后文才建立。
- 本轮按执行依赖重排为“出口任务图 → 普通/非 tensor 输出 → stride → mutation 三引用 → finalize → 完整源码 → 边界迁移”。源码仍保持三个方法的产品顺序和全部语句。
- 补齐三个时点的类型区分：env 解引用结果、`realize_input` 后对象、是否进入 tensor stride；明确 `ir.Constant` 形成 `TensorBox(StorageBox(ConstantBuffer))`，符号输入跳过 mutation 检查，未修改输入沿两次 `.data` 落到同名 InputBuffer。
- 新增重复返回实验：`x.add_(10); return x, x` 的两个运行时结果均与调用者 x 共享 `data_ptr()`；正文明确该外部合同还包含上层输出去重与 wrapper，不能用本方法单次 `.index()` 独自解释。
- 下一步由新的无历史冷读者只读重排稿，逐分支闭卷验收；不提供本轮问题清单或正确答案。

### 第六轮第二次冷读与事实纠正

- 第二名无历史读者只读重排稿，指出 mutation 返回替换仍存在对象层级断点：正文把 `graph_outputs` 项写成 TensorBox，却用内部 StorageBox 搜索，读者无法解释 `.index()` 为何命中，评级 B。
- 回查 `ir.ExternKernel.realize_input` 后确认旧正文事实错误：普通 `TensorBox(StorageBox(Pointwise))` 会递归到 `.data`，物化后返回 `StorageBox(ComputedBuffer)`，不是重新包装后的 TensorBox；因此普通计算输出也不会进入只接收 TensorBox/BaseView 的 stride 分支。
- 正文现逐层命名 `current_tensor`、`current_storage`、`current_ir` 和 `original_x`：返回表与输入检查持有同一个 StorageBox，`.index()` 直接匹配该对象；`realize_into` 接收的是 StorageBox.data 的底层 IRNode。
- `ir.Constant` 分支由 `add_tensor_constant` 直接返回 TensorBox，不再递归解包，所以它仍进入 stride helper。BaseView 路径按 `realize_input` 的实际返回类型判断。
- 原实验 04 已观察普通 graph output 为 `StorageBox->ComputedBuffer`；本轮修订与该 artifact 及当前 `ir.py` 源码一致。下一轮必须由全新读者重新验收，不能沿用第二名读者已获得的纠正信息。

### 第六轮第三次冷读：通过

- 第三名无历史读者只读事实修正版，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/88_OUTPUT_SOURCE_COLD_REVIEW_3.md`，评级 A。
- 读者能闭卷区分普通 TensorBox 输出、`ir.Constant`、标量/None 的 `realize_input` 层级，完整推演 stride 三出口、mutation 四层对象、写回/返回替换、未修改/符号/异名输入和 finalize。
- 最后收紧两处轻微问题：把“非 tensor”改成“不同类型的返回值”；按当前 `copy_input` 源码写明 `EffectfulKernel` 形成已物化 Pointwise 的 TensorBox并进入 stride helper。
- `experiments/06_mutation_storage.py` 增加 `x.add_(10); return x, x` 完整编译检查；两个返回值与调用者 x 均共享 `data_ptr()`。artifact 已重新生成。本篇仅用它验证外部合同，不把上层输出去重和 wrapper 行为归因给本方法的一次 `.index()`。
- Q213 隔离冷读闭环；下一篇按 BFS 顺序为 Q214（`GraphLowering.run_node` 外围收尾源码）。

> 第五轮状态：叶子完成。第五轮修正了 wrapper 消费者的队列归属，并重新按完整源码、输出类型、stride、mutation 与 finalize 状态验收。

## 本篇定位与读者

- 全系列队列：Q213，深度 3，父项 Q21；这是职责完整的源码叶子，不再按函数另建目录。
- 读者起点：已从 Q211/Q212 知道 placeholder、算子和常量怎样进入 `GraphLowering.env`，但还不知道最后一个 FX `output` 怎样形成函数接口。
- 本篇学完：能从 `result` 追到 `graph_outputs`，逐维判断无关 stride，追踪修改后当前值、原输入存储和返回别名，并说明 `finalize` 的调用边界。
- 下一层边界：布局决定的内部算法归 Q23；wrapper 如何消费 `graph_outputs` 归 Q43。本篇只讲 GraphLowering 出口合同。

## 范围与版本依据

- 产品版本：`/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 完整源码范围：`torch/_inductor/graph.py` 中相邻的 `output`、`finalize`、`try_match_insignificant_strides`；正文保持产品顺序和全部语句，只增加 `# 教学：` 注释并去掉类级缩进。
- 相邻实现依据：`torch/_inductor/ir.py::ExternKernel.realize_input`，用于核对 tensor、标量、`None` 和常量在出口处的实际包装。
- 实验依据：`experiments/04_graph_lowering_nodes.py` 与 artifact 验证普通/标量输出；`experiments/06_mutation_storage.py` 与 `artifacts/mutation_storage.txt` 验证两种 mutation、两种 stride 和 finalize 接口调用。
- 资料限制：未使用旧教程；正文事实只来自上述当前源码、当前测试思路和独立实验。

## 篇内知识点队列

| 编号 | 问题与完成条件 | 案例 | 状态 |
| --- | --- | --- | --- |
| K01 | 解释为何统一返回序列、检查类型并将各类结果变成出口 IR | tensor、`3`、`None` | 完成并复读 |
| K02 | 用逻辑索引算出 size 0/1 维为何能换 stride，并识别真实差异 | `[2,1,3]` 对照 `[2,2,3]` | 完成并复读 |
| K03 | 追踪修改值怎样写回原 InputBuffer，返回引用怎样保持调用者别名 | `old=x+1; x.copy_(y); return old,x` | 完成并复读 |
| K04 | 说明修改输入却不返回时，写回为何仍发生 | `x.copy_(y); return y+1` | 完成并复读 |
| K05 | 说明 finalize 收到什么、按何顺序处理、为何位于出口末尾 | 两个 layout 接口探针 | 完成并复读 |
| K06 | 对三个声明方法做完整源码等价和重要分支审计 | AST + 覆盖表 | 完成 |

## 处理情况覆盖

| 情况 | 进入时对象 | 关键变化与出口 | 正文位置 | 证据与承诺深度 |
| --- | --- | --- | --- | --- |
| singleton / tuple-list 输出 | 父 Interpreter 解引用的值 | singleton 统一为序列，与 FX 输出逐项配对 | 完整源码、案例一 | 当前源码；叶子实现 |
| tensor 延迟结果 | `TensorBox/StorageBox` | `realize_input` 物化，必要时匹配无关 stride，进入 `graph_outputs` | 案例一 | 实验 04 + 源码；机制 |
| 标量、布尔表达式、`None` | 原始值或 SymPy 值 | 分别形成 `ShapeAsConstantBuffer` / `NoneAsConstantBuffer`，跳过 tensor stride | 案例一后的表 | 实验 04 + `ir.py`；机制 |
| `ir.Constant` / `ConstantBuffer` | 内联常量或已有常量存储 | 登记 tensor 常量或复用原对象 | 同表 | `ir.py` 源码；用途与分支，未单独端到端测常量 output |
| stride 全相同 | 已物化 tensor + meta stride | 返回同一对象 | 完整源码 | 当前源码；叶子实现 |
| 差异只在 size 0/1 维 | 固定 layout | 同 storage/size/offset 建 `ReinterpretView`，仅换对应 stride | 案例二 | 直接真实 IR 实验；机制 |
| 有效维 stride 不同 | size > 1 的维有差异 | 原对象返回，不伪造布局转换 | 案例二 | 直接真实 IR 实验；机制 |
| 输入未修改 | 当前底层仍是同名 `InputBuffer` | 不登记写回 | 案例一 | 源码 + 实验 04；机制 |
| 输入修改并返回 | 当前 StorageBox 不再指向同名原输入 | `realize_into` 写回；返回项替换成原 InputBuffer | 案例三 | 完整编译运行 + 源码；机制 |
| 输入修改但不返回 | 写回条件相同，返回表找不到当前 StorageBox | 写回保留，吞掉 `ValueError`，返回表不替换 | 案例三 | 完整编译运行支持外部合同，源码支持具体分支；机制 |
| 符号输入 | `sympy.Expr` | 跳过 tensor 写回检查 | 完整源码 | 当前源码；叶子实现 |
| finalize | 已登记 buffers | 逐个调用 `decide_layout` | finalize 节 | 接口 mock + 源码顺序；接口机制，不声称测布局算法 |
| 非法出口/输入类型、长度不等 | 不满足断言 | 立即失败，不形成完整出口 | 完整源码 | 当前源码断言；失败出口已保留 |

## 案例选择与证据边界

1. 普通 tensor 与标量输出复用实验 04，因为它直接观察 `GraphLowering.graph_outputs`，能证明物化后的具体类型；最终数值本身不足以证明这一点。
2. stride 使用局部真实 IR 输入，直接调用产品方法并检查 storage 名、size、stride、offset 和对象身份。它精确对应目标分支，但不是一次完整 `torch.compile`。
3. mutation 使用两次完整编译，分别检查调用者输入值、返回值和 `data_ptr()`；这证明外部写回/别名合同。`ValueError` 被内部捕获，运行结果不能单独证明异常分支，因此正文同时引用当前源码。
4. finalize 探针是 mock 接口对象，只证明遍历和调用次序。布局究竟怎样决定仍由 Q23 讲解，不能从该探针外推。

## 完整源码与事实检查

- AST 核对结果：解析正文代码块与产品 `GraphLowering`，`output`、`finalize`、`try_match_insignificant_strides` 均逐一等价（3/3）；差异只有注释和类级缩进。
- 重要路径均有条件、状态与出口：singleton、允许类型、返回长度、tensor/非 tensor、相同/无关/显著 stride、tensor/符号输入、未修改/修改且返回/修改未返回、finalize 与日志。
- 已修正旧稿事实错误：`int` 与 `None` 不是原值直接留在 `graph_outputs`，而是先经 `realize_input` 包装。
- 正文没有把 `ReinterpretView` 写成数据复制；地址公式明确 size 1 维索引只能为 0。
- 正文没有用完整运行结果冒充 `ValueError` 的内部观测，也没有用 mock 冒充真实布局选择。

## 作者理解复读

仅依据正文回答，不以源码表中原句复述替代因果：

1. 若 shape `[2,1,3]` 的第二维 meta stride 从 99 改成 123，为什么值不变？因为该维唯一合法逻辑索引为 0，地址贡献始终是 `0×stride`；新 view 仍引用同一 storage、size 和 offset。
2. 若 shape 改成 `[2,2,3]`，为什么不能照做？第二维索引 1 已合法，地址贡献会从 `1×3` 变为 `1×99`，这不再是同一逻辑 tensor 的无关表示差异。
3. `x.copy_(y); return y+1` 为什么仍必须写回 x？mutation 合同由 `graph_inputs` 当前底层是否仍为同名原 InputBuffer 决定，与 x 是否在 `graph_outputs` 中无关；返回表查找只负责别名替换。
4. 返回 `(tensor, 3, None)` 时 graph_outputs 中是什么？tensor 是已物化且检查过 stride 的存储表示；另外两项分别是 `ShapeAsConstantBuffer(3)` 与 `NoneAsConstantBuffer`。
5. 为何不能提前 finalize？输出物化和 mutation 写回可能在它之前新增 buffer 或约束；最后遍历才能让当前全部已登记 buffer 参与布局决定。

上述五题均能由正文中的对象变化和实例推出；实际读者反馈尚未取得，不能登记为“读者已理解”。

## 续写位置

## 第五轮叶子验收

- **完整源码**：`output`、`finalize`、`try_match_insignificant_strides` 三个相邻方法完整保留；产品顺序、断言、日志与内部 helper 无省略。
- **分支与出口**：singleton/序列、允许与非法类型、tensor/标量/None/常量、stride 相同/仅无关维差异/有效维差异、未修改/修改返回/修改未返回、符号输入和 finalize 全覆盖。
- **具体状态**：普通 pointwise 从延迟规则到 ComputedBuffer；`[2,1,3]` 用地址贡献证明 stride 99 无关，`[2,2,3]` 对照拒绝；mutation 案例同时追踪当前 StorageBox、原 InputBuffer 和 graph_outputs。
- **理解迁移**：正文能判断 `[0,5]` 哪一维可改 stride、名字不同的 InputBuffer 为何仍触发写回，以及 ValueError 只表示返回表没有该引用而不取消 mutation。
- **证据边界与层次**：stride 是局部真实 IR，mutation 是完整编译，finalize 是接口探针；wrapper 消费 graph_outputs 正确归 Q43，布局内部归 Q23。

Q213 第五轮叶子复核完成。根队列转到同层 Q214：`GraphLowering.run_node` 外围收尾及直接 mutation helper。Q221/K03c2 仍保持原恢复位置，等广度优先队列轮到 Q221 时继续。
