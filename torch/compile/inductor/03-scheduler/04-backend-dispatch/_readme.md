# Scheduler 后端分派篇作者记录

## 第七轮：教材自身的状态机回修

- 无上下文冷读报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/20_BACKEND_DISPATCH_FINAL_COLD_REVIEW.md`。结论为 C：调用顺序可以跟随，但正文只能让读者抄固定状态增量，不能独立推演 pending-free；`buf0` 也没有计算含义。
- 这轮不把报告中透露的知识留给下一位读者。正文自身新增统一转移顺序：前置边界 flush → 登记当前 `last_usage` → 按身份分派 → 更新 available/completed → backend 主动 flush；明确 `free_buffers()` 交付后清空 pending，而 available/completed 跨 flush 和 device 保留。
- 对照本轮重新生成的 `ir_post_fusion.txt` 与 `output_code.py`，补齐实际案例映射：op0 计算 `matrix+1→buf0`，op1 的 realize 读取 buf0 写 buf1；二者逻辑上是两个结果，wrapper 中 buf1 复用 buf0 的存储。
- 主案例改为每一步给出 group、available、completed、pending 的完整快照，并区分 extern 调用后的立即 free 与普通组等待 flush。
- 补齐 backend `ready_to_flush()` 是 Scheduler 可观察的返回信号、device 切换发生在当前节点登记 last_usage 之前，以及普通/extern/template/combo/nop 各自何时清空 pending。
- 待办：再启用一名不继承上下文的新读者，只读当前正文重新做迁移推演；不得把本轮批注或答案放进提示词。

## 第八轮：按 backend 区分普通节点出口

- 冷启动复读：`/work/learning/1021/56sol/inductor-fresh-reader-review/21_BACKEND_DISPATCH_COLD_REVIEW_AFTER_REWRITE.md`，评级 B。主案例、同设备普通变体和跨 device 变体均可独立推演；未通过项是 combo 前置边界不明、template 公共状态特例被推到叶子、guard 与最终 flush 的关系不明。
- 重新核对当前源码后确认，问题根源不是“漏列 combo flush”，而是正文把 CPU `CppScheduling` 的暂存行为泛化给了所有普通节点。当前 SIMD `codegen_node` 在内部直接定义和调用 kernel，并调用 `scheduler.free_buffers()`；combo 也只允许 SIMD/CUDA combined backend，并在 `codegen_combo_kernel` 写调用后 free。因此 combo 身份不触发 `_codegen` 前置 flush，也没有待提交的 CPU 式旧组。
- 正文现在分别说明 CPU C++ 普通节点与 CUDA SIMD 普通节点的出口和 pending 清空时机；跨 device 案例也明确当前 CUDA SIMD 调用在 guard 内当场写出，最终 flush 没有待提交的 CUDA 普通组。
- template 特例已留在本篇：backend 对主体和 epilogue 均 `mark_run()`，但 `_codegen` 局部变量重绑定后只按模板主体更新 available/completed。叶子只解释实现语句，不再承担本篇状态结论。
- 待办：第三名无上下文读者复读当前正文，检查 CPU/SIMD 分流是否建立得足够早，以及五类身份是否都能按本层状态推演。

## 第九轮：template、nop 与确定性变体

- 第三份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/22_BACKEND_DISPATCH_THIRD_COLD_REVIEW.md`，评级 B。它能完整重建主线和六类出口；剩余断点为 `mark_run` 与 completed 的关系、nop 为何延迟 free，以及普通替换变体没有唯一轨迹。
- 正文定义 `mark_run()` 为输出进入 wrapper 分配状态，区别于 Scheduler 的 `completed_operations` 历史集合。template backend 对主体和 epilogue 均 mark_run 并生成同一调用；公共集合只加入主体是 `_codegen` 局部重绑定后的状态，不会漏掉 epilogue 生成。
- nop 用 concat 型无 kernel IR 具体化：它完成输出分配登记，但该分支没有调用 `free_buffers()`，所以 pending 走下一次或最终 flush 的共享清理入口；不再虚构“等待 nop kernel”的安全因果。
- CPU 普通替换变体先固定节点兼容且 `ready_to_flush()` 全程为 false，给出唯一最终 flush 轨迹；再只改变返回值，给出 mm 后组满的对照轨迹。
- 待办：用第四名无上下文读者验收本篇，重点检查上述三处是否由正文自身闭环。

## 第十轮：template epilogue 的消费者闭环

- 第四份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/23_BACKEND_DISPATCH_FOURTH_COLD_REVIEW.md`，评级 B。除 template epilogue 外，主案例、两条普通 mm 轨迹、CPU→CUDA、六类出口、nop 和 SIMD→combo 均可推演。
- 断点来自正文把 available 解释成“后续可以引用”的近似门禁，却没有说明 epilogue 未加入 available 后如何被后继读取。当前源码中 available 用于 Scheduler 依赖修剪并在生成中累计记录；分派时的合法节点序列已经确定，后继 backend 使用 IR 里的 buffer 名生成读取，不由 available 二次授权。
- 正文补齐 `mark_run()` 的实际出口：对每个输出调用 wrapper allocation 或 inplace reuse 接口；template 对主体 `bufT` 和 epilogue `bufE` 均 mark_run，并写产生 bufE 的同一 kernel。后继 opN 从自己的 IR 读取 bufE，所以公共集合只记录主体不会丢失空间、生产或消费。
- nop 同时补了一组具名输入 `part0/part1`、结果 `joined`、allocation、available/completed 与 pending 延迟清理，避免只讲控制流。
- 待办：第五名无上下文读者验收；若仍为 B，继续按新断点回修，不用前轮报告替代当前正文评价。

## 第十一轮：普通 CPU 输出分配与物理存储

- 第五份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/24_BACKEND_DISPATCH_FIFTH_COLD_REVIEW.md`，评级 B。template 的 bufE 链路已可推演；新主缺口是普通 CPU 节点何时把 IR 输出落实成 wrapper allocation/reuse。
- 当前源码事实：`CppScheduling.codegen_node` 进入 `CppKernelProxy.codegen_nodes`，对每个成员依次 `decide_inplace_update()`、`mark_run()`，随后才把 kernel 生成工作留在 group。`mark_run()` 对输出调用 wrapper allocation/inplace reuse；flush 负责后来写 kernel 调用。
- 正文将 CPU 第 3 步补成“决定复用、mark_run 安排输出空间、暂存生成工作”，并说明 completed 会被原地复用判断用来排除已经处理的 users。
- `buffer_names_to_free` 改称释放候选：`free_buffers()` 会过滤输出、已处理名字和没有独立存储的逻辑名，然后清空集合。主案例按实际生成代码修正：buf1 复用 buf0；buf3 接管 buf2；最终不会在 sin 后物理释放返回值空间。
- combo 补具名例子，说明 `get_buffer_names()`/`get_operation_names()` 汇总子节点结果和 operation 后更新公共集合。
- 待办：第六名无上下文读者复核完整空间—调用—释放链，特别检查逻辑名结束与物理存储释放是否仍混淆。

## 第十二轮：available/completed 的实际消费者

- 第六份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/25_BACKEND_DISPATCH_SIXTH_COLD_REVIEW.md`，评级 B。所有案例链路可重建；剩余断点是 template 漏记 opE 对 completed 消费者的影响，以及分派期 available 累加后的用途。
- 正文明确 available 的直接消费发生在建立调度节点时，用于裁剪已经存在的输入/常量依赖；进入 `_codegen` 后继续维护集合，但后续节点生成不再以它作为结果读取门禁。
- completed 会被 `decide_inplace_update()` 用来过滤已经处理的 users。template 未加入 opE 时，后续复用判断可能保守地把 opE 当成剩余 user，放弃一次可行复用；不会提前覆盖数据、漏生成 epilogue 或重复分派。
- 待办：第七名无上下文读者做本篇最终冷验收；通过后在作者记录注明实际读者仍待用户阅读，而不是把 AI 复读冒充真人理解。

## 第十三轮：template 漏记的具体复用推演

- 第七份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/26_BACKEND_DISPATCH_SEVENTH_COLD_REVIEW.md`，评级 B；除 template 漏记 opE 的复用影响没有代入集合外，其余承诺均可推演。
- 正文新增 `old.users={opE,opN}`：completed 若含 opE，过滤后只剩 opN，可继续选择原地复用；当前特例漏记 opE 时过滤后仍有两个 users，复用条件失败并为 opN 另行安排空间。由此具体证明影响是保守少复用，而非提前覆盖。
- 待办：第八名无上下文读者验收；AI 冷读通过只证明正文自包含性，不记作用户本人已理解。

## 第十四轮：nop 等待的是下一清理点

- 第八份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/27_BACKEND_DISPATCH_EIGHTH_COLD_REVIEW.md`，评级 B。唯一实质断点是正文称 nop pending 等待 flush，却允许后继 SIMD/extern/template/combo 的分支内 `free_buffers()` 更早清空同一全局集合。
- 正文统一为“等待下一清理点”：后继直接生成分支在自身末尾一并检查 nop pending；后继 CPU C++ 普通节点则继续积累到边界或最终 flush。nop 自身始终不生成或等待 kernel。
- 待办：第九名无上下文读者做本篇收尾验收。

## 第十五轮：无上下文冷读通过

- 第九份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/28_BACKEND_DISPATCH_NINTH_COLD_REVIEW.md`，评级 A。
- 该读者只读当前 `README.md`，能够独立推演 CPU 主案例、ready 两条变体、CPU→CUDA、六类出口、template epilogue 的状态消费者与保守复用影响、nop 的两类后继清理点，以及 SIMD→combo；未发现实质因果断点。
- 这只说明当前正文对设定认知背景的 AI 冷读者自包含，不等于用户本人已经理解。用户后续实际阅读反馈优先级更高，发现断点继续回修。
- 当前篇冷读整改完成；按全系列顺序进入 Q41 Codegen 父篇审查。

## 第六轮：冷启动读者回修

- 冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/18_BACKEND_DISPATCH_COLD_REVIEW.md`。读者只能还原主要分支，无法证明原混合案例中 extern 依赖前一普通组；template/nop/状态/device guard 分散，父篇主线被 backend 注册、实验取证和实现细节挤占。
- 原 `mixed` 实验的 pointwise 与 mm 是独立分支，不能支撑“extern 必须引用前一组结果”。已在 `experiments/11_scheduler_backend_dispatch.py` 新增 `dependent_mixed`：`realize(matrix+1) → mm → sin`，实际节点为 `op0_op1` 普通融合组、`op2` extern、`op3` 普通节点；mm 的 `last_usage` 明确包含前组输出 `buf1`。
- 新正文只用这条真实依赖链贯通首节点空 flush、普通组暂存、extern 前提交、extern wrapper 调用、新普通组和末尾 flush。
- available/completed/pending-free 统一定义为编译期生成状态，停止在“交给 wrapper 生成逻辑”；真实内存释放与复用继续属于 Codegen 内存计划。
- template、foreach、nop、device guard 降为主线后的身份分支；删除 registry 类名矩阵、fallback 数值、template 选择实验和当前实现的局部变量重绑定细节，保留于本作者记录或叶子实现。
- 待办：由不继承本会话的新读者只读当前正文，验证能否独立走完 dispatch 时间线。

## 本篇位置与教学目标

- 全系列编号：Q34；层次 2；父项 Q30；前置 Q31、Q32、Q33；完成后转同层 Q41。
- 读者已知：最终调度节点已有合法顺序与融合组；不知道 kernel backend 与 wrapper 各负责什么，也不知道为什么同 device 的 extern/template 仍形成提交边界。
- 入口：Q33 交出的最终节点序列，以及每节点的 device、类型、成员、`last_usage`。
- 出口：backend 收到普通循环组/template/combo 请求；wrapper 收到 extern、kernel 调用、释放和 device guard。Q41-Q44 再打开后端 kernel 与 wrapper 内部。
- 学完标准：给出一个混合节点序列，能逐项写出 flush、guard、分派入口、三组 Scheduler 状态的变化，并区分普通融合组和 template 分支的状态更新差异。
- 资料：当前提交 `8332c5622aabd9535969e9a8d0ed60e4a90e4402`、当前测试、独立 CPU 实验；没有 GPU/跨设备运行。
- 第五轮状态：已按技能重新复核，保留“本层完成”；Q341 尚未创建，仍按第 3 层队列待讲。

## 本层职责和协作

1. 注册与按需创建：device type 查 `device_codegens`，规范化 device 查 `self.backends` 缓存。
2. 提交边界：device 改变、extern、template、backend `ready_to_flush()`、遍历结束会触发 flush；空 flush 合法。
3. 类型分派：template、extern、foreach、普通计算、nop 按源码顺序进入不同接口。
4. 状态兑现：访问节点时登记待释放，分派后更新 available/completed；backend flush 后生成 wrapper 释放；需要时切换 device guard。

本篇解释这些职责怎样交接，不展开 `CppKernelProxy`、Triton ops 翻译、wrapper 内存复用算法。

## 处理情况覆盖

| 情况 | 当前层过程与出口 | 证据 | 深度/状态 |
| --- | --- | --- | --- |
| 普通/融合节点 | `backend.codegen_node` 累积组；边界时 backend flush | pointwise、mixed | 机制已讲解并复读 |
| 普通 extern | 前置 flush；`codegen_extern_call` 直接写 wrapper 并自行 free | mm | 机制已讲解并复读 |
| fallback/MultiOutput extern | backend 可在首 extern 之后才因 `ready_to_flush` 查询创建 | histc | 边界已讲解并复读 |
| CPU template+epilogue | 前置 flush；`codegen_template(template, epilogues)` 立即生成并 free | frozen module 实际命中 | 机制已讲解并复读 |
| template 回退 extern | Q24 已确定身份；Q34 不重新选择 | runtime weight 实际回退 | 对照已讲解并复读 |
| foreach/combo | 仅接受 `SIMDScheduling/CUDACombinedScheduling`，进 `codegen_combo_kernel` | 源码/当前测试，无本机运行 | 条件与出口已讲，未声称实测 |
| nop | 不切 device；`mark_run()`，不走 backend 普通生成 | 源码，无独立运行 | 条件与出口已讲 |
| device 改变/guard | 先 flush 旧工作，再退出旧 guard、进入新 guard | 源码/测试；无 GPU 运行 | 机制按源码讲解 |
| backend 主动 flush | 普通节点分派后查询 `ready_to_flush()` | C++ 参数数门槛源码 | 条件已讲；未触发门槛 |
| 最终收尾 | 先退出最外 guard，再最终 flush 全部 backend 和 pending free | `_codegen` 源码；CPU 实验 | 机制已讲解并复读 |

## 篇内知识点队列

| 编号 | 完成条件 | 案例/正文 | 状态 |
| --- | --- | --- | --- |
| K01 | 区分 registry、device-keyed backend 实例和 wrapper | 注册与创建 | 完成 |
| K02 | 从首节点解释第一次 flush 为什么无输出、何时创建 backend，以及节点怎样加入当前组 | pointwise | 完成 |
| K03 | 走通普通→extern→普通，写出两次非空 C++ flush | mixed | 完成 |
| K04 | 比较 fallback、template 回退与真实 template | 三组对照 | 完成 |
| K05 | 按源码顺序覆盖五类节点的入口和失败边界 | 分派表 | 完成 |
| K06 | 推演 available/completed/pending-free，解释 template 特例 | 状态表 | 完成 |
| K07 | 解释 device guard、主动 flush 与最终收尾顺序 | 收尾与迁移题 | 完成 |

## 独立实验与实际事实

- 脚本：`../../experiments/11_scheduler_backend_dispatch.py`；输出：`../../experiments/artifacts/scheduler_backend_dispatch.txt`。
- 包装 Scheduler 与 `CppScheduling` 当前方法但调用原实现；采集 backend 创建、节点访问、Scheduler/CPP flush、普通/extern/template 分派及 available/completed/pending-free。
- pointwise：空 flush → `codegen_node(op0_op1)` → 末尾非空 flush `[op0,op1]`。
- mixed：普通融合组 → 非空 flush → mm extern → sin 普通组 → 末尾非空 flush。
- fallback：两个 extern；首个前 `self.backends` 可为空，首个完成后的 `ready_to_flush` 查询才创建 CPU backend。
- runtime-weight template 尝试：实际回退 `ExternKernelOut + SchedulerNode`，并有 `No choices for GEMM` 警告。
- frozen-template：冻结模块 buffer `constant0`，实际节点为 `op0_op1:FusedSchedulerNode`，命中 `cpp.codegen_template op0 epilogues=['op1']`；所有数值与 eager 对照通过。

真实 template 还证明一项当前实现细节：`_codegen` 在 template 分支把局部 `node` 重绑定成 template 主节点。`CppScheduling.codegen_template` 内部会对主节点和 epilogue 全部 `mark_run()`，但随后 Scheduler 的 `available_buffer_names/completed_operations` 只按重绑定后的主节点更新。本例末尾状态为 available 新增 `buf0`、completed 新增 `op0`，不是把 epilogue `op1` 也加入这两个集合。

## 当前源码依据

- `codegen/common.py`：`DeviceCodegen`、注册表、注册/查询/初始化。
- `scheduler.py`：`flush`、`codegen_extern_call`、backend 创建/缓存、`enter_context`、`_codegen`。
- `codegen/cpp.py`：`codegen_node`、`codegen_template`、`ready_to_flush`、`flush`、`KernelGroup`。
- `codegen/wrapper.py`：device guard enter/exit。
- `test_cpu_select_algorithm.py`：冻结权重、CPP template、epilogue；`test_benchmark_fusion.py`：DeviceGuard 生成检查。

## 作者复读：事实与理解分验

事实：实验重新运行成功；新增 frozen-template 明确命中 `codegen_template`。普通、extern、fallback、template 及回退的节点身份、调用顺序、状态集合都有原始输出。foreach/nop/GPU 仍只由源码或测试支持。

只依赖正文回答：

1. 首个 CPU 普通节点前为何 flush？`current_device=None`，device 改变条件成立；空 backend 组不产生 kernel。
2. mixed 中 mm 前为何同为 CPU 仍 flush？extern 本身就是边界，不能留进普通 C++ loop group。
3. fallback 首个 extern 前为何可能没有 `CppScheduling.flush`？Scheduler 只遍历已创建 backend；extern 自身不需先创建普通 backend。
4. template 为什么在 extern 之前判断？template 节点可能也是融合节点，必须先拆出主模板和 epilogue 走专用入口。
5. frozen-template 的 epilogue 是否运行？会，被 `codegen_template` 内的 `mark_run()` 处理；但 Scheduler 两集合在当前实现只更新重绑定后的主模板节点。
6. CPU A、extern B、CUDA C 的顺序？A 积累；B 前 flush A；C 前因换 device flush、切 guard，再交 CUDA backend；末尾先退出 guard再最终 flush。

作者复读通过；实际读者反馈尚无。

## 续写位置

Q34 第五轮重新复核完成；Scheduler 的四个第二层职责完成。根队列转 Q41，保持深度 2；Q341 叶子仍按广度优先队列留到下一层。

## 第五轮重新复核

- 范围：backend 注册/实例缓存、提交边界、五类节点分派、available/completed/pending-free 与 device guard/收尾形成完整控制循环；kernel 和 wrapper 内部留给 Q41–Q44。
- 过程：普通、mixed、fallback、runtime-weight template 回退、frozen-template 五组案例展示 backend 创建、空/非空 flush、extern/template 入口和最终状态；foreach/nop/GPU 明确只按源码/测试。
- 状态：available/completed 是后续生成可引用的逻辑状态，不等于机器代码已运行；template 分支当前只按重绑定后的主节点更新两集合，而 epilogue 由 backend 内部 mark_run。
- 事实：frozen-template 实际命中 `codegen_template`，修复了早先只有 extern 回退却泛称模板实测的证据缺口；跨设备和 GPU guard 仍未本机运行。
- 理解：只凭正文可推演 CPU A/B→extern C→CPU D→CUDA E 的 flush/guard/分派顺序，并解释 fallback 首次 extern 前为何可能尚无 CPU backend。实际读者反馈仍为空。
