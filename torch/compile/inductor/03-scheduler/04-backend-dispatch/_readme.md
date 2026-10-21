# Scheduler 后端分派篇作者记录

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
