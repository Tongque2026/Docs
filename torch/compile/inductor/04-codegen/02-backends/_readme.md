# Q42 作者记录：CPU C++ 与 GPU/Triton 后端

> 当前状态：收到逐篇隔离冷读要求后重新打开。本轮第一位冷读者评为 C，已完成结构重写，等待新读者复验。

## 第六轮：隔离冷读后的结构重写

- 冷读边界：读者只看到本篇 README，认知起点为会基础 torch/torch.compile、刚理解 LoopBody 与 handler，不读取作者记录、源码或旧报告。
- 报告：`inductor-fresh-reader-review/39_BACKENDS_COLD_REVIEW_1.md`，评级 C。
- 已通过部分：N=19 的 CPU 16+3 和 Triton 3×8 能闭卷复算，block/lane/mask/thread 的边界基本清楚。
- 实质断点：2×128 规约没有 accumulator、分块、横向归并和写回状态；persistent、MultiKernel、Disable/EnableReduction 只是名词；kernel 定义与调用缺形实参、grid、device、stream 的完整交接；模板/extern 只有边界声明；正文没有作者记录声称的迁移题。
- 重写内容：用“逻辑域由谁覆盖”统一全文；保留 CPU N=19 实际循环并解释尾向量长度；把 Triton grid=`ceil_div(N,XBLOCK)` 接到 program 内索引；逐项展开两端 ops；用 2×128 走完 CPU 八次向量累计和 Triton `XBLOCK=1/RBLOCK=64` 两块累计；把 persistent 等放在基础规约完成之后；补 CPU/Triton 定义与 wrapper 调用合同；新增 N=21 与 2×130 迁移题。
- 证据边界：CPU N=19 仍是当前机器实测；Triton 只使用当前源码公式和当前测试形态作具体推演，不声称 GPU 实跑、实际选中 XBLOCK/RBLOCK 或性能。
- 验收：等待新的隔离冷读，不能沿用作者复读结论。

### 第二次隔离冷读

- 报告：`inductor-fresh-reader-review/40_BACKENDS_COLD_REVIEW_2.md`，评级 B。
- 已通过：pointwise 的 CPU 16+3、Triton 3×8、ops 对照和两道尺寸迁移均可由正文推出。
- 剩余断点：2×128 没声明输入值；Triton 把真实的 64-lane 累计对象压成标量 `acc`，缺跨块逐 lane 更新与最终横向归并；stream 表述可能暗示 program 串行；template/extern 二分过硬；MultiKernel 与规约区域标记仍偏抽象。
- 修改：声明 `arange(256)` 输入；显式推演 `acc_lane[0..63]` 在两个/三个 r block 中的更新和最终 `sum`；说明 MultiKernel 首次基准测试并缓存选择；用“规约→Disable→后缀 pointwise→Enable”展示生成状态；修正 device/stream/program 的关系；将 template/extern 改成非完备的阅读导航。
- 当前状态：等待第三位隔离冷读者复验。

### 第三次隔离冷读

- 报告：`inductor-fresh-reader-review/41_BACKENDS_COLD_REVIEW_3.md`，评级 B。
- 主线状态：N=19、ops 对照、2×128/2×130 规约、wrapper launch、template/extern 导航与迁移题均已可推演。
- 必改断点：`torch.arange(256)` 默认类型与后文 float32 SIMD/`float*` 合同不一致；规约区域标记被画得像每次规约必然出现的固定序列。
- 修改：输入显式指定 float32；说明 Disable/Enable 仅在 schedule 需要临时关闭规约循环生成外部节点时成对出现；补 persistent 执行稿边界、资源约束与 MultiKernel 选择缓存粒度；将 device/stream 明确归入 wrapper launch 上下文。
- 当前状态：等待第四位隔离冷读者复验。

### 第四次隔离冷读

- 报告：`inductor-fresh-reader-review/42_BACKENDS_COLD_REVIEW_4.md`，评级 B。
- 首要断点：用“状态保留方式”区分 persistent 与 non-persistent，和前文 non-persistent 累计器跨块保留直接冲突。
- 其他缺口：float32 pointer 合同缺 layout/index/guard 前提；MultiKernel 复用范围过宽；64-lane 累计器需限定为 XBLOCK=1；codegen 标记需明确不是设备指令；2×130 题只是正文复述。
- 修改：用“单个 reduction block 覆盖完整 r 域 vs 多 roffset 扫描”重定义两种规约；补 dtype、连续 stride、索引式与 guard 的指针合同；限定 MultiKernel 为同一缓存条目/专门化条件；补 XBLOCK>1 的二维累计器；增加 3×130、XBLOCK=2/RBLOCK=64 的双轴尾部题和 stride=2 地址题。
- 当前状态：等待第五位隔离冷读者复验。

### 第五次隔离冷读

- 报告：`inductor-fresh-reader-review/43_BACKENDS_COLD_REVIEW_5.md`，评级 A。
- 结论：pointwise、两种规约形态、MultiKernel 边界、codegen 状态标记、wrapper 合同和迁移题均可只凭正文推演，未发现错误教学模型。
- 轻微补强：为 3×130 写出 `xmask[:,None] & rmask[None,:]` 的二维组合、地址式和有效元素计数；说明 stride=2 时逻辑切分问题仍在，但连续 `loadu` 指令形态未必保留。
- 当前状态：Q42 通过隔离冷读验收。A 只表示指定冷读角色能够从正文建立逻辑，不代替真实用户反馈；若用户仍有断点，继续以实际断点重开。

## 本篇位置与教学目标

- 全系列编号：Q42；层次 2；父项 Q40；前置 Q41；下一项 Q43。
- 读者已知：LoopBody 会通过当前 handler 产生位置级表达；不知道 CPU 循环、SIMD lane、Triton program/block 与逻辑索引分别是什么。
- 入口：同一类 Scheduler node、LoopBody、迭代范围与规约范围。
- 出口：CPU C++ kernel 定义/调用，或 Triton kernel 的 range tree、program grid、block/mask 与调用参数；wrapper 细节留给 Q43。
- 学完标准：能把 N=19 的逻辑域分别映射到 CPU 16+3 和 Triton 3×8 masked block；能区分融合、线程、SIMD、program/block；能说明规约为何需要输出轴与规约轴，以及何时结束规约区域。
- 证据边界：CPU 在当前机器实际生成并运行；Triton 只依据当前源码、当前测试片段和独立公式求值，因为 `torch.cuda.is_available()==False` 且无可运行 Triton 环境。

## 本层职责与边界

1. CPU：先用 `CppKernel` 建立标量 loop nest，再尝试 tiling/vector kernel，选择主段和尾段实现，最后输出选中的循环体。
2. SIMD/Triton 调度：从融合成员得到 numel/rnumel，建立 node schedule，选择 tiling、index dtype、规约 hint 和 kernel 类型。
3. Triton kernel：range tree 生成 program id、block range、mask；handler 生成 `tl.load/store/reduction` 表达。
4. 调用：CPU binding 传 pointer 参数；Triton 额外组织 numel、grid、device、stream/block 配置。

本篇比较执行几何和本层交接，不逐句解释后端函数；Q421 以后完成源码实现。

## 处理情况覆盖

| 情况 | 对象与状态变化 | 证据 | 深度/状态 |
| --- | --- | --- | --- |
| CPU pointwise 整块 | 标量底稿 → 选宽 16 vector main loop | N=19 实际生成代码 | 机制已讲解并复读 |
| CPU 非整除尾部 | 剩余 3 项 → masked vector tail；条件不满足可回 scalar | 实际 3-lane tail + 源码分支 | 机制已讲解并复读 |
| CPU 线程与 SIMD | loop nest 可选并行深度；一步处理多个 lane | 当前源码；小例未触发 OpenMP | 概念边界已讲 |
| Triton pointwise | pid×XBLOCK+arange，mask 去掉越界 lane | 当前源码、测试、N=19 公式 | 机制按证据讲解 |
| mask 可省略 | 静态 size=1 或整除 BLOCK 等条件 | `_has_constant_mask` 源码 | 条件已介绍 |
| CPU reduction | 外层输出行、内层宽 16 规约、横向归并 | Q41 2×128 实际产物 | 机制已讲解 |
| Triton reduction | x/r range tree；persistent 或分块 r loop；结束规约后才能用结果 | 当前源码/测试，无运行 | 机制按源码讲解 |
| 多 kernel 规约选择 | persistent 时可生成 non-persistent 候选并由 MultiKernel 处理 | SIMD 源码，无本机运行 | 条件与出口已介绍 |
| 定义与调用 | CPU pybinding vs Triton grid/device kernel call | CPU 产物、Triton `call_kernel` 源码 | 协议差异已讲 |
| template/extern | 不走普通 LoopBody 对照线 | Q24/Q34 | 边界已链接 |

## 篇内知识点队列

| 编号 | 完成条件 | 案例/正文 | 状态 |
| --- | --- | --- | --- |
| K01 | 从逻辑域建立两种执行几何，不把 block 当单 thread | 开篇总图 | 完成 |
| K02 | 推演 CPU 标量底稿、向量主段和尾段选择 | N=19 | 完成 |
| K03 | 代入 pid/lane/mask 并验证覆盖、不重不漏 | N=19,XBLOCK=8 | 完成 |
| K04 | 比较同一 load/add/store 的目标表达 | 对照表 | 完成 |
| K05 | 走通 CPU 与 Triton 规约的 x/r 状态 | 2×128 | 完成 |
| K06 | 比较 kernel 定义与 wrapper 调用参数 | CPU 产物/Triton 源码 | 完成 |
| K07 | 区分融合、线程、SIMD、block、mask | 三个判断 | 完成 |
| K08 | 完成 N=20 与整除条件迁移题 | 作者复读 | 完成 |

## 独立实验与事实表

- 脚本：`../../experiments/13_backend_models.py`。
- 输出：`../../experiments/artifacts/backend_models.txt`；CPU 代码：`backend_models_cpu_output.py`。
- CPU：静态 float32 长度 19，实际生成宽 16 主向量循环和 3 元素尾向量循环；输出逐项等于 eager。
- Triton：独立求值当前源码/测试中的 `xoffset=pid*XBLOCK; xindex=xoffset+arange; xmask=xindex<xnumel`。N=19、XBLOCK=8 时 active 集合依次是 0..7、8..15、16..18，拼接严格等于 `range(19)`。
- 这不是同一图在 GPU 的实测对照，不报告 GPU 生成代码、性能或真实 XBLOCK 选择。

CPU 源码先构造 scalar kernel 和 loop nest，再在 ISA/tiling 合法时构造 vector main/tail 并替换相应 loop 上的 kernel。故“生成过程中有标量底稿”不等于最终源文件必含标量循环；N=19 产物最终只有两个 vector 循环。

## 当前源码与测试依据

- `codegen/cpp.py`：`CppKernelProxy.codegen_functions/codegen_nodes`、标量/vector/tail 分支、load/store/reduction。
- `codegen/simd.py`：`generate_node_schedule`、`codegen_node_schedule*`、tiling/persistent/MultiKernel、Disable/EnableReduction。
- `codegen/triton.py`：range tree、persistent heuristic、index/mask、kernel codegen 与 `call_kernel`。
- `test_triton_heuristics.py`：pointwise program_id/arange/xmask 当前实例。
- `test_metrics.py`：XBLOCK/RBLOCK/xmask/rmask 规约实例。

## 作者复读：事实与理解分验

事实：CPU 实验已重跑，产物与正文循环边界一致；Triton 表只由公式求值支持，并明确标为非运行证据。规约实际 CPU 证据来自 Q41；Triton persistent/non-persistent 来自当前源码与测试。

只依赖正文回答：

1. 为什么 N=19 是 16+3？当前 ISA/tiling 选 16，主段覆盖 0..15，尾段剩 3；当前 dtype 和 tail-vector 配置允许 3-lane masked vector。
2. scalar kernel 是否一定出现在最终代码？不一定；它先建立通用 loop nest，合法 vector kernel 可替换主段和尾段。
3. N=20、XBLOCK=8 时第三个 Triton program 哪些 lane 有效？候选 16..23，前四 lane 对应 16..19 有效。
4. program block 是否等于 8 个 CUDA thread？不能这样推出；Triton block tensor 到硬件线程的映射由编译器完成。
5. reduction 结果为什么不能立刻给后缀 pointwise？累计值只在规约范围完成后有效，schedule 用 Disable/EnableReduction 划开区域。
6. mask 什么时候可能省略？源码能静态证明对应范围无需防越界，例如 size=1 或满足相关整除条件；不是因为运行时碰巧没越界。

作者复读通过；实际读者反馈尚无。

## 续写位置

## 第五轮全量复核

- **范围**：正文覆盖普通 pointwise 的 CPU 主段/尾段与 Triton program/block/mask，覆盖 CPU/Triton reduction 的 `x/r` 两类范围、规约区结束条件，以及 kernel 定义与 wrapper 调用协议；template/extern 明确交回 Q24/Q34，不混入普通 LoopBody 路线。
- **过程**：同一份 `LoopBody` 先保持 `load → add/relu/mul → store` 的位置级语义，再分别映射为 CPU induction variable + SIMD lane，或 Triton `program_id*XBLOCK+arange`。规约例从输出行 `x0` 进入组内位置 `r0`，累计完成后才允许后缀 pointwise 使用结果。
- **事实证据**：CPU N=19 在当前机器实际生成宽 16 主循环和 3-lane masked-vector 尾循环且与 eager 相等；Triton N=19/XBLOCK=8 只由当前源码与测试公式求值，得到三个候选块和最后五个 false mask，不声称 GPU 实跑或真实 block 选择。
- **理解迁移**：正文足以推出 N=20 时 CPU 在同一向量宽度假设下为 16+4，Triton 第三个 8-lane block 仅 16..19 有效；也能解释 block tensor 不等于 8 个 CUDA thread、mask 只裁掉补齐位置，以及只有静态证明无需防越界时才可省略 mask。
- **层次**：本篇只讲后端协作和可观察生成形态；`CppKernelProxy`、range tree、mask/index 生成函数的完整源码仍属于 Q421 叶子，不以本篇源码片段冒充叶子完成。

Q42 第五轮回修完成。根队列转同层 Q43；Q421 仍按广度优先队列留到下一层。
