# GraphLowering run_node 源码叶子作者记录

## 第六轮：隔离冷读整改

- 第一名无历史冷读者只读原 README，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/89_RUN_NODE_COLD_REVIEW_1.md`，评级 B。
- 主体时间线、分派、stride、物化、origin/readers 与水位均可闭卷复述；主要缺口是 K05 的集合推演与源码顺序不一致。旧文把 `u0<=6` 写成立即生成，但 `bound_unbacked_symbols` 在整个 new-def 循环后才 union，本轮计算 missing 时当前新符号仍可能缺失。
- 正文现用 `bound={s_old}`、`new_defs=[s0,s1]`、`ras[s0]=[s0<s1]` 逐行代入：missing 为 `{s0,s1}`，记录重新排到 s0，本次调用不会自动重访；循环结束后才把 s0/s1 加入 bound。范围 AssertScalar 与 deferred assertion 分成两条路径。
- 补充 `inductor_realize_to_strides` 的实际布尔结合、tensor result 接口不变量；补充无 MKLDNN/MKL 时列表怎样缩减但共同多用户逻辑仍执行；集中列出显式断言、schema/范围查表和 binding 覆盖失败。
- 下一步由新的无历史读者仅凭修订稿复验完整方法，不能向其提供本轮答案或问题清单。

### 第六轮第二次冷读：通过

- 第二名无历史读者只读修订稿，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/90_RUN_NODE_COLD_REVIEW_2.md`，评级 A。
- 读者能闭卷拼接 helper 与 run_node 四段源码，复原四类分派、stride/平台条件、物化、origin/readers、水位、deferred queue、binding 和所有本层关键失败出口。
- 对同 key requeue 的生命周期，正文进一步收窄为当前方法可证明的事实：本轮 union 后不重扫，离开时 assertion 仍在队列且尚未生成；不暗示当前源码保证未来一定再次消费。
- Q214 隔离冷读闭环；下一篇按 BFS 顺序为 Q221（Pointwise、Reduction、View 构造核心源码）。

> 第五轮状态：叶子完成。重新按完整长方法、连续状态、水位切片、layout/mutation、物化与 unbacked assertion 分支验收。

## 本篇定位、读者与边界

- 全系列队列：Q214，深度 3，父项 Q21；这是 GraphLowering 节点外围合同的源码叶子，不再按函数或条件建立子目录。
- 读者起点：Q211–Q213 已解释输入登记、算子分派和输出合同；本篇回答一次具体 lowering 前后，谁补齐 layout、物化、来源、读者和符号断言。
- 学完目标：能按源码顺序追踪一个节点的水位、分派、结果表示、图级索引和符号状态，并能对条件改变后的新案例作判断。
- 边界：具体 Pointwise/Reduction/View 构造归 Q221；layout 决策算法归 Q231；调度如何使用 readers 归 Q311/Q321。本篇解释 `run_node` 怎样形成这些交接。

## 范围与依据

- 产品版本：`/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 声明源码：`torch/_inductor/graph.py` 中完整 `propagate_mutation` 与 `run_node`。长方法分为四个相邻代码块，拼接顺序与产品一致。
- 独立实验：`experiments/20_run_node_contracts.py`；artifact 为 `experiments/artifacts/run_node_contracts.txt`。
- 辅助事实：Q213 的实验 06 提供真实输入修改与旧值结果；本篇只引用其 run_node reader/mutation 交接，不重复声称观察内部调度。
- 未使用旧教程。unbacked-symbol 的深分支当前没有稳定的完整编译样例，按源码和具体集合状态推演，明确不登记为运行实测。

## 篇内知识点队列

| 编号 | 完成条件 | 案例 | 状态 |
| --- | --- | --- | --- |
| K01 | 依据 schema 和对象身份判断 layout 副本的修改是否传播 | `copy_` 的可写 self、只读 src、未替换 self | 完成并复读 |
| K02 | 追踪水位、origins 和四条初始 result 分派 | add、unsupported、layout constraint、symbolic magic | 完成并复读 |
| K03 | 判断输出/as_strided stride 合同与四类物化条件 | relu 同时交给 mm 和 output | 完成并复读 |
| K04 | 追踪 origin、reader 和水位后的新符号定义 | `buf0/op0` 与抽象 `s0/s1` | 完成并复读 |
| K05 | 推演值域断言、延迟断言、binding 覆盖与失败出口 | 新符号 s0、两符号断言 | 完成并复读 |
| K06 | 核对声明源码完整性、证据边界和跨段综合案例 | AST + 实验 20 | 完成 |

## 处理情况覆盖

| 情况 | 条件与状态变化 | 出口/消费者 | 正文 | 证据深度 |
| --- | --- | --- | --- | --- |
| layout 参数替换且可写 | 新旧对象不同且 schema `is_write` | `copy_` lowering 传播回旧引用 | K01 | helper 直接 mock，证明调用参数 |
| 参数相同或只读 | 身份相同，或无写 alias | 不传播 | K01 | helper 直接 mock + schema |
| unsupported 类型 | 非 getitem 且类型检查要求 fallback | fallback handler 的 result 进入共同后处理 | K02 | 当前源码；分支实现 |
| layout constraint | 约束复制参数、正常 lowering、再传播 mutation | result 继续共同后处理 | K02 | 当前源码 + K01 probe；机制 |
| symbolic magic | meta 为 SymInt/Float/Bool | 取 SymPy expr；否则父 Interpreter | K02 | 当前源码；分支实现 |
| 普通节点 | 无前三类特殊条件 | `super().run_node` | K02 | 实验 20 覆盖普通 relu/mm |
| `inductor_realize_to_strides` | meta 标志且 result 为 TensorBox | 先 realize，再按静态 stride order 约束 | K03 | 当前源码 + 状态推演 |
| output view | result 为 BaseView | realize 以保持返回 alias | K03 | 当前源码；分支实现 |
| output/as_strided stride | 静态 stride，区分 view、dense、padding、channels-last | order 或 exact stride 要求 | K03 | 当前源码 + 具体 stride 推演 |
| 多 user 且有真实输入消费者 | user 在 `needs_realized_inputs` | realize hint，部分 target 固定布局 | K03 | 实验 20 覆盖 relu→mm/output |
| 多 user 且 output | Pointwise/Reduction 结果 | realize 并 mark reuse | K03 | 实验 20：ComputedBuffer buf0 |
| 读次数或表达式过大 | `has_exceeded_max_reads` / large pointwise | hint 或强制 realize | K03 | 当前源码 + 状态推演 |
| origin 与 readers | 非 view 的 Loops/Buffer/MultiOutput；任意 result | best-effort origin；register users | K04 | 当前源码；接口机制 |
| 本节点新定义符号 | 只扫描水位后的 buffer/op | `new_unbacked_defs` | K04 | 水位状态推演；实验 20 验证 buf/op 切片，不含符号 |
| 有限符号值域 | 新符号范围比默认更窄且边界可转 int | AssertScalar buffer/operation | K05 | 当前源码 + 具体 `[2,10]` 推演 |
| 延迟断言依赖未齐 | assertion 尚引用未绑定符号 | 按最小缺失符号重新排队 | K05 | 当前源码 + 两符号集合推演 |
| FX/IR binding 对不上 | IR 新定义集合不覆盖重命名后 FX binding | 带新增对象信息的断言失败 | K05 | 当前源码；失败出口 |
| placeholder | `n.op == placeholder` | 跳过 runtime assert/binding 检查 | K05 | 当前源码；分支实现 |

## 案例与证据边界

1. 实验 20 的 helper probe 使用真实 `aten.copy_.default` schema，但 mock 了 `GraphLowering.call_function`。它能证明可写参数筛选、身份短路和传入 copy_ 的 `(old,new)`，不能证明生成代码是否含 copy kernel。
2. 同一实验完整编译 `activation=relu(x); product=mm(activation,w); return activation,product`。hook 记录每个 `run_node` 进入前后的 buffers/operations、users 与 result 类型；relu 实际有 mm/output 两个 users，并形成 `ComputedBuffer buf0/op0`。
3. hook 是观测包装，调用的仍是当前产品 `GraphLowering.run_node`。数值断言另外确认 activation/product；数值正确本身不用于证明内部物化类型。
4. unsupported、as_strided、large expression、unbacked assert 等未由该运行案例覆盖；正文对这些项只作完整源码与具体状态推演，不写成“实测”。

## 源码完整性核对

- 从正文六个 Python 代码块中取：第 1 块为 `propagate_mutation`；第 2、3、5、6 块按顺序拼接为 `run_node`；第 4 块是教学案例，不属于产品源码。
- 解析 AST 后与当前 `GraphLowering` 两个方法比较：`propagate_mutation` 1/1 等价，`run_node` 1/1 等价。此前只做“369 条非空行有序出现”的检查太弱，现已替换为方法级 AST 等价。
- 两个方法的 docstring、断言、平台条件 target 列表、注释承载的行为边界、失败断言与 return 均保留；只增加 `# 教学：` 注释并去掉类级缩进。

## 作者理解复读

1. constraint 同时复制 dst 和 src，为何只传播 dst？对象不同只是必要条件；schema 还必须把对应参数标为可写。
2. relu 只交给 mm、不再返回时，哪项证据会消失？`num_users>1` 与 output-user 分支不再成立；mm 作为真实输入消费者仍可要求物化，但不能沿用“双 user + output”实验结论。
3. 为什么水位不能被理解为调度顺序？它只保存进入节点前列表长度，用切片识别本节点新登记对象；执行先后由后续依赖/调度处理。
4. 断言 `s0+s1<10` 在只绑定 s0 时发生什么？`missing` 仍含 s1，于是 assertion 重新挂到 s1；等 s1 定义后才建立 AssertScalar。
5. FX binding 为重命名前 u0、fake mode 记录 u0→u1、IR 定义 u1 时为何通过？检查先应用 `unbacked_renamings`，比较的是 `{u1}` 是否被本节点新定义集合覆盖。

这些答案均能由正文的对象与状态变化推出。作者复读已完成；尚无实际读者反馈，不能将其登记为读者已理解。

## 续写位置

## 第五轮叶子验收

- **完整源码**：`propagate_mutation` 完整方法与分段展示的 `run_node` 合并后保持产品顺序；上下文、平台 target 条件、长注释、断言和 return 无省略。
- **分支与出口**：unsupported、layout constraint、magic/普通分派，output/as_strided stride，多用户/读次数/大表达式物化，origin/readers、水位新符号、值域/延迟断言、binding 覆盖与 placeholder 跳过全覆盖。
- **具体状态**：copy_ schema 表区分可写 dst/只读 src；relu→mm+output 实测形成 buf0/op0；buffer/operation 水位示例只收 s0/s1；`s0+s1<10` 在缺 s1 时重新排队。
- **理解迁移**：正文能判断只给 mm 时哪条 output 证据消失、BaseView 不写 origin 但仍登记 readers、`[0,+∞]` 只生成下界断言，以及重命名后 FX/IR binding 集合如何比较。
- **证据边界**：mutation helper 为 schema+mock 接口，relu/mm 为完整 CPU 编译；unsupported/as_strided/large/unbacked 分支由当前源码和具体集合推演支持，不标成实测。

Q214 第五轮叶子复核完成。根队列恢复 Q221/K03c2，继续 Pointwise、Reduction、View 构造核心源码；Q231 仍排在其后。
