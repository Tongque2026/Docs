# Q341 backend 分派循环核心源码备课记录

## 第六轮：隔离冷读与状态契约回修

- 第一名无历史读者只读正文，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/119_BACKEND_DISPATCH_COLD_REVIEW_1.md`，评级 B。普通、extern、guard 和 flush 主线成立；template epilogue 已生成却未进入外层 available/completed 的契约没有闭合，nop 跳过 `device` 赋值后仍进入 debug-sync 分支则形成代码疑点。
- 核对当前源码后，正文开头补齐四份状态及消费者：backend cache、current device、pending free、available/completed。`available_buffer_names` 在调度节点构造时供 `prune_deps` 使用；`completed_operations` 在后续 inplace 判断中过滤已处理使用者。它们是 Scheduler 外层记账，不是 backend 实际生成范围或设备完成事件。
- template 段现说明两层状态：当前 C++ backend 对主模板和 epilogue 都 `mark_run()` 并生成同一 kernel；Python 局部 `node` 重绑定后，外层集合只登记主模板。少记 epilogue 使 completed 的后续过滤更保守，不会撤销 backend 已分配和生成的 epilogue。
- 当前 `_codegen` 的 debug-sync 分支未排除 nop，却读取只在非 nop 条件块中赋值的 `device`。实验 11 新增 `empty_strided` 形成 NopKernelSchedulerNode：默认配置成功；以另一函数开启 `triton.debug_sync_kernel` 时捕获 `BackendCompilerFailed`，根因是 `UnboundLocalError: local variable 'device' referenced before assignment`。正文将其列为当前失败/疑点出口，没有编造外部安全前提。
- foreach 与 nop 分别补齐到当前层出口：foreach 经 backend 类型门、combo 分派、集合更新和 ready 查询；nop 只 mark_run、无 kernel/wrapper 调用、更新集合并跳过 ready，debug-sync 例外单列。
- 第二名无历史读者只读修订稿，报告为 `/work/learning/1021/56sol/inductor-fresh-reader-review/120_BACKEND_DISPATCH_COLD_REVIEW_2.md`，评级 A。读者能闭卷重建五路分派、边界 flush、guard、last usage、两层 template 状态、nop 失败、ready 与 final flush。
- 实验 11 全部数值和状态断言通过；新增 nop 的默认 shape 与 debug-sync 错误均写入 artifact。Q341 隔离冷读闭环，整改队列进入 Q411。

## 读者、层次与出口

- 读者已从 Q34 能按最终节点序列推算普通、extern、template、foreach、nop 的分派与 flush 边界；尚不知道这些判断如何由同一个循环、局部变量重绑定和共享状态集合实现。
- 本篇是第 3 层叶子。输入为 Scheduler 的最终 `self.nodes`、backend 缓存、当前 device、last usage 与 wrapper；出口为 backend 已接收的生成任务、wrapper 中的 extern/guard/free 请求，以及更新后的 available/completed 状态。
- 本篇完整解释 Scheduler 控制循环；不展开 `CppScheduling.codegen_node/flush`、Triton 指令翻译或 wrapper 内存复用算法，它们分别属于 Q411/Q421/Q431。
- 产品基线：`/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。事实来源仅为当前源码、测试与实验 11。

## 声明范围

| 产品顺序 | 声明 | 本篇职责 |
| --- | --- | --- |
| 1 | `Scheduler.flush` | flush 所有已创建 backend，再兑现 pending free |
| 2 | `Scheduler.codegen_extern_call` | 在临时 kernel handler 中决定 inplace/mark_run，写 extern wrapper 调用并 free |
| 3 | `Scheduler.create_backend` | 规范化约束、device 注册、实现查找与 Triton 可用性失败出口 |
| 4 | `Scheduler.get_backend` | 按完整 device 延迟创建并缓存 backend |
| 5 | `Scheduler.enter_context` | 从融合成员 origins 选择图顺序最晚者作为 wrapper 上下文 |
| 6 | `Scheduler.codegen` | 计时外壳，进入 `_codegen` |
| 7 | `Scheduler._codegen` | 节点循环、边界、guard、五类分派、状态更新、主动与最终 flush |

共 7 个方法。正文须按产品顺序完整保留，去除固定教学注释后 AST 7/7 等价；`BaseScheduling` 抽象接口和具体 backend 实现作为消费者边界，不扩大本叶子声明范围。

## 篇内知识点队列

| 编号 | 问题、案例与完成条件 | 状态 |
| --- | --- | --- |
| K01 | 以首个 CPU pointwise 为例，解释 backend 延迟创建、空 flush、普通组积累和最终 flush；能写出 `backends/current_device/pending-free` 的逐步状态 | 完成，已复读 |
| K02 | 以 pointwise→mm extern→sin 为例，逐句推演 `flush/codegen_extern_call`，区分 backend flush、extern 自行 free 与 Scheduler 状态更新 | 完成，已复读 |
| K03 | 对照 frozen template、fallback extern、foreach、nop，按源码顺序推演五类分派及 template 局部 `node` 重绑定 | 完成，已复读 |
| K04 | 以 CPU→CUDA 假定序列解释 guard、`ready_to_flush`、debug sync、最终退出与 flush；覆盖失败出口并完成 7/7 总审计 | 完成，已复读 |

## 处理情况覆盖

| 条件 | 关键状态变化 | 出口/失败 | 归属与证据 |
| --- | --- | --- | --- |
| backend 首次/重复请求 | `self.backends[device]` 缺失时创建，之后复用 | BaseScheduling 实例 | K01；实验 11 + 源码 |
| 不支持 device / GPU 无 Triton / 旧 CUDA | add device 后在创建阶段报错 | RuntimeError | K01；源码 |
| 普通节点 / device 首次进入 | 边界 flush、设 current device、积累 backend group | 后端任务 | K01；pointwise 实测 |
| extern | 前置 flush；临时 Kernel 中 inplace/mark_run；直接写 wrapper；free | extern 调用 | K02；mixed/fallback 实测 |
| template | 前置 flush；拆主节点与 epilogue；局部 node 重绑定 | template 调用及主节点状态更新 | K03；frozen template 实测 |
| foreach | 仅 SIMD/CUDACombined backend 可接 combo | combo 调用或 AssertionError | K03；源码/测试 |
| nop | 不切 device；mark_run；仍更新状态；不查 ready_to_flush | 无普通 kernel | K03；源码 |
| debug sync | 分派后按配置调用当前 backend `codegen_sync` | 同步代码请求 | K04；源码 |
| backend ready / device/extern/template / 结束 | 中途或最终 flush；pending free 清空由 `free_buffers` 完成 | 已提交任务与释放请求 | K01/K02/K04 |
| guard device 切换 | flush 后退出旧 guard、进入新 guard；末尾先 exit 请求再 final flush | wrapper guard 请求 | K04；源码/测试，无本机 GPU |
| origin 空/多项 | 空则不 enter；多项去重并选图序最大项 | wrapper context | K04；源码 |

## 案例事实与边界

- 主实验：`../../../experiments/11_scheduler_backend_dispatch.py` 与 artifacts。pointwise、mixed、fallback、runtime-weight template 回退、frozen template 均已在 Q34 采集并数值核对，本轮重新运行通过。
- frozen template 实际命中 `codegen_template(op0, epilogues=[op1])`；backend 内部对两者 `mark_run`，而 Scheduler 后续 available/completed 因局部 `node` 重绑定只加入主节点。这是 K03 的关键状态，不以概括替代。
- foreach、nop、跨 device、GPU guard、debug sync 和失败出口仅用当前源码/测试推演，正文必须标明证据边界。
- 实际读者反馈尚无；每个 K 点完成后仅登记作者依据正文完成迁移推演。

## 当前恢复点

K01–K04 均完成，Q341 叶子完成。根队列转 Q411。

## 第五轮叶子验收

- **声明完整性：** 7 个目标方法在正文各出现一次，按 `flush→codegen_extern_call→create_backend→get_backend→enter_context→codegen→_codegen` 的产品顺序连读；脚本从 Markdown Python fence 解析 AST，并与当前导入的 Scheduler 方法比较，7/7 为 True。
- **分支覆盖：** backend 首次/复用与三类失败，origin 空/多项，device/extern/template 边界，template/extern/foreach/ordinary/nop 五路分派，debug sync，available/completed 更新，ready-to-flush、guard 切换和 final flush 都有条件与出口。
- **具体状态：** pointwise 展示空 flush 后延迟创建、积累与 final flush；mixed 展示普通组提交→extern 自行 free→新普通组；fallback 展示首 extern 前无 backend、完成后的 ready 查询才创建；frozen template 展示 `node` 从融合组重绑定为主模板并只按主节点更新两集合。
- **作者迁移复读：** 连续 extern 不会进入普通组；同 device template flush 但不重进 guard；ordinary GPU→template→ordinary 的两次组边界可按正文推演；foreach 后端类型不符明确失败；缓存存在后空 flush 仍会访问 backend。答案均可仅凭正文推出。
- **证据边界：** 实验 11 本轮通过并支持五个 CPU 案例；foreach、nop、跨 device、GPU guard/debug sync 与创建失败仅由当前源码/测试支持。实际读者尚未作答。

下一项 Q411：在 `04-codegen/01-kernel-ops/01-replay-source/` 先建立 LoopBody 重放叶子计划，再写完整源码与状态推演。
