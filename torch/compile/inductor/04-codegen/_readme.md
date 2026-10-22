# Codegen 篇作者记录

## 第六轮：无上下文读者结构回修

- 冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/29_CODEGEN_PARENT_COLD_REVIEW_1.md`，评级 B。读者能拼出四项职责，但原正文的 3 元素 `relu(x+y)*2` 只连续走到 Q43；Q44 依赖其他实验和通用流程人工接回。reduction 三次 replay、19 元素 tail/mask、memory planner 反例和 FX cache 边界提前占据父篇，形成越层与篇幅失衡。
- 本轮重写父篇而非局部补句。唯一主案例现在从 op0 LoopBody、handler 局部值、C++ kernel、wrapper source 连续走到 PyCodeCache 内容 key/path、loaded `module.call` 和运行结果 `[0,10,4]`。
- 独立实验 `experiments/12_codegen_virtualized_ops.py` 新增 `GraphLowering.compile_to_module` 观察，实际记录本例的 `cache_key`、`cache_path`、`module.__file__` 与 callable 名 `call`；脚本重新运行通过，产物同步到 `experiments/artifacts/codegen_virtualized_ops.txt`。
- 父篇明确 Q41/Q42 是 replay 期间逐项协作，不是假想的两个完全分离批处理阶段；五种对象分别为 LoopBody、handler 局部值、kernel 定义、wrapper source、loaded callable。
- 新增三条分类轴：pointwise/reduction 是计算形态，CPU/Triton 是目标 backend，普通 LoopBody/template/extern 是生成路径，避免把它们写成同层互斥选项。
- reduction、混合 extern/reuse 和 FX Graph Cache 降为主线后的职责变化；移除三次 replay、CPU tail 具体 load 数、Triton pid 表、两张重复 storage 表、counter、cond/AOT/零计算等子篇内部展开和实验报告口吻。
- 待办：启用一名不继承上下文的新读者只读重写正文，验证同案 Q41→Q44 是否自然闭合，并检查删减后是否遗漏父篇必须交代的主要路径。

## 第七轮：kernel 绑定进入 wrapper 模块

- 第二份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/30_CODEGEN_PARENT_COLD_REVIEW_2.md`，评级 B。主案例、分类轴和扩展案例均可推演；唯一主要断点是 C++ kernel 源码如何成为 `call(args)` 中的 `cpp_fused_*` Python 可调用名。
- 对照本例实际生成源码补入模块顶层最小交接：`AsyncCompile()`、`async_compile.cpp_pybinding(参数类型, C++ 源码)`、赋值给 `cpp_fused_add_mul_relu_0`、`async_compile.wait(globals())`。模块加载时等待编译并更新全局命名空间，随后 `call` 才能调用该名字。
- 对象表从五项改为六项：增加 loaded module，并把 kernel 定义改准为 kernel 源码；wrapper source 明确包含 kernel 编译绑定和 `def call`。
- source line map 在首次出现时定义；删除“独立实验记录到”的作者核对句，正文直接讲对象过程。FX cache 候选条件与 wrapper 运行时 shape/stride 检查分名；template renderer 明确属于生成路径轴，不是新的设备 backend 轴。
- 待办：第三名无上下文读者复读完整 kernel source→binding→module→callable 交接，并继续寻找其他局部断点。

## 第八轮：Q42/Q43 接口与定义顺序

- 第三份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/31_CODEGEN_PARENT_COLD_REVIEW_3.md`，评级 B。所有主链与变化均可推演；剩余主要问题是正文一处称 Q42“登记进完整模块”，与 Q43 生成模块源码的职责冲突。
- 统一接口：Q42 只交出 kernel 源码、参数类型和调用要求；Q43 写模块顶层 `cpp_pybinding`、`wait(globals())` 和 `def call`，形成完整源码。
- `boxed` 定义前移到总图之前，首次出现即说明接收单个参数列表；对象计数统一为六种；`last_usage` 首次出现直接解释成“某值在哪个节点最后一次被读”，并说明 backend/wrapper 据此形成分配和复用安排。
- kernel 代码块改称“省略函数签名后的计算体”，不再让参数类型与片段展示产生矛盾。
- 待办：第四名无上下文读者验收职责边界和定义顺序。

## 第九轮：last usage 到 wrapper 复用代码

- 第四份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/32_CODEGEN_PARENT_COLD_REVIEW_4.md`，评级 B。主链与分类均可推演；唯一主要缺口是 last usage 已定义，却没有代入一次“最后使用→允许覆盖→wrapper 代码变化”。
- 使用现有实测混合图而不虚构图输入复用：第一 mm 分配 buf0/S0；relu 是旧 buf0 值的最后消费者并允许原地；wrapper 因此写 `buf1=buf0; del buf0`，不为 buf1 新分配，再用 in-place relu kernel 更新 S0。
- 正文同时给两个边界：若还需返回 relu 前的 first，旧值在 relu 后仍可观察，必须另分配 relu 输出；若返回 relu 后的 activated，原地 relu 仍可成立，但 S0 的生存期延长到函数出口。
- 待办：第五名无上下文读者验收 last usage→storage 接管→逻辑名结束以及两个返回变体。

## 第十轮：无上下文冷读通过

- 第五份冷启动报告：`/work/learning/1021/56sol/inductor-fresh-reader-review/33_CODEGEN_PARENT_COLD_REVIEW_5.md`，评级 A。
- 该读者只读当前父篇，能够独立走通同案 Q41→Q44、kernel binding、boxed callable、三条分类轴、reduction/extern/FX cache，以及 last usage→S0 接管→两类额外返回的复用差异；未发现新的实质断点。
- AI 冷读 A 只说明正文对设定背景自包含，不代替用户实际阅读反馈。用户后续指出的新断点仍优先回修。
- 当前 Q40 父篇冷读整改完成；按顺序进入 Q41 `04-codegen/01-kernel-ops/README.md`。

## 所在层、输入与出口

- 队列项：Q40，第 1 层，父项 Q00；第五轮已按技能重新复核，保留“本层完成”。本状态只覆盖四项直接职责的协作，不替代 Q41–Q44 的内部回修。
- 读者已经掌握：Scheduler 交出按 device/node type 排列的普通、融合、extern/template 节点和 last usage；mixed 案例已形成 `CPU group→extern→CPU group`。
- 本篇输入：调度节点的 LoopBody/调用合同、目标 device/backend、layout、last usage 与 wrapper 图级状态。
- 本篇出口：带 kernel/extern 定义、guards、分配/复用、调用、释放和返回的模块，以及可从缓存写入、编译、加载后取得的 boxed `call(args)`。
- 四个直接子职责与目录一致：Q41 将 LoopBody 的虚拟 ops 重放成后端可处理的操作；Q42 用 CPU C++、Triton 或模板等目标模型发射 kernel；Q43 生成 wrapper 与内存时间线；Q44 写入缓存、编译、加载并恢复 callable。
- 下一层打开四项内部；本篇不逐行讲 LoopBody、索引生成器、memory planner 或 FxGraphCache 查找。

## 篇内知识点队列

| 编号 | 问题、对象与完成条件 | 案例 | 状态 |
| --- | --- | --- | --- |
| K01-R | 四项职责怎样把调度动作连续变成 boxed callable | 系统图，明确 kernel/backend/wrapper/cache 的交接 | 完成，已复读 |
| K02-R | 同一虚拟 ops 怎样在后端形成具体局部值、循环和规约 | add-relu-mul；2×128 sum | 完成，已作 tmp、累计与多次 replay 复读 |
| K03-R | CPU vector tail 与 Triton block mask 如何覆盖合法位置 | 19 元素 CPU 实测；19 元素 Triton 公式推演 | 完成，已作逐位置与证据边界复读 |
| K04-R | wrapper 怎样落实 guards、两类调用、storage 复用与 last usage | `mm→relu→mm→sin` | 完成，已作逻辑名/storage 与生存期迁移复读 |
| K05-R | 编译、加载和多层缓存怎样恢复 callable | 两独立进程 miss→hit、cond bypass；无计算快速出口 | 完成，已作缓存层次、三出口与 guard 复读 |
| K06-R | 综合迁移验收 | 禁止一次复用、改变 shape/stride、cache guard 不满足 | 完成，已作四职责贯通与证据分层复读 |

Q40 的六个知识点均已完成并在第五轮重新复核；第 1 层结束，根队列按 BFS 转 Q11。

## 处理情况与案例覆盖

| 情况 | 关键交接 | 本层出口 | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| pointwise/reduction LoopBody | load/ops/store 或 reduction/store_reduction 被 handler 重放 | CSE 临时值、循环/累计与 kernel 定义 | 实验 12 | 机制已讲并作连续状态复读 |
| CPU C++ | 向量主循环、标量/向量尾部、ABI binding | C++ kernel + Python binding | 实验 13/output code | 机制已讲并作 19 元素位置复读 |
| GPU/Triton | program id、block offsets、mask、launch grid | Triton kernel/launch | 当前源码公式；无 CUDA 实跑 | 公式机制已讲；明确未实机执行 |
| extern/template | 不重写内部数学，wrapper 生成调用或模板 launch | 与生成 kernel 同一模块的调用点 | 实验 14、当前源码 | extern 已实测；template 仍为源码路径 |
| wrapper 内存 | guards、alloc/reuse、last usage、tuple return | `call(args)` 时间线 | 实验 14 | 机制已讲并作 storage 状态复读 |
| 首次编译加载 | source→内容 key→模块→kernel 编译/等待→call | current callable | 当前源码与实验 15 | 机制已讲并复读 |
| FX graph cache hit/miss/bypass | key 候选与 guards 决定复用或完整编译 | 恢复 callable 或重新生成 | 实验 15、当前源码 | 三路径已实测/核对并复读 |

## 依据、旧状态撤销与反馈

- 源码依据：`graph.py:codegen/_compile_to_module`、`scheduler.py:_codegen`、`codegen/`、`codecache.py` 及实际生成的 Python/C++ 源码。
- 实验 12–15 分别覆盖虚拟 ops、后端模型、wrapper 内存与跨进程 cache。CPU 是实际执行；Triton 仅当前源码和公式求值，明确未实机执行。
- 上一版只画“backend、wrapper、compile/load”三类代码，却有 Q41–Q44 四个子目录；虚拟 ops 到后端 handler 的交接被藏入 backend。第二轮改为 Q41/Q42/Q43/Q44 四项直接职责。
- 旧记录直接宣称 Q41–Q44 完成和“全系列审计通过”，但当时第 2/3 层仍有待回修/待讲项；该结论撤销。
- 实际读者反馈：尚无；作者迁移题与运行证据分别记录。

## 第二轮事实与理解验收

- 事实核对：实验 12 的 pointwise LoopBody 依次重放 load/add/relu/constant/mul/store，reduction 使用 `128*x0+x1`、accumulator 与 store_reduction；正文将统一语义和后端表达分开。
- 事实核对：实验 13 的 CPU 19 元素路径实际执行并与 eager 一致；Triton 19/8 的三组索引只作当前公式推演，明确记录未运行 CUDA/Triton。
- 事实核对：实验 14 的 wrapper 确有两次 allocation、两次 storage reuse、相应 del 与 tuple return；第二 mm 时输入 buf1 和新输出 buf2 并存。
- 事实核对：实验 15 两独立进程为 miss=1/hit=0 与 miss=0/hit=1，结果和相同；正文未用耗时推断缓存。
- 作者只凭正文可答：19 改 16 时 LoopBody 数学不变，CPU/Triton 尾部处理、wrapper size guard 和缓存候选合同会改变。
- 作者只凭正文可答：relu 结果另作输出时，kernel 数学不变，但 wrapper live range 阻止覆盖复用；源码/合同变化又要求 Q44 恢复相应新 callable。
- 本篇只完成 Q40 协作，Q41–Q44 内部仍待第二轮回修；实际读者反馈为空。

## 第三轮严格复读进度

- K02-R pointwise：正文可由 x0=0 推出 tmp0=-2、tmp1=1、tmp2=-1、tmp3=0、tmp4=2、tmp5=0，并由 x0=1 推出输出 10。
- K02-R reduction：x0=0 的前三个当前项为 1、2、3，累计 1、3、6，最终 8256；x0=1 读取 128..255、加 1 后累计 129..256，最终 24640。
- 实验 12 观察到 reduction 同一 LoopBody 有一次 scalar、两次 vector replay；正文明确这是编译期不同 handler 重放，不是运行时重复三次规约。
- 迁移：把 sum 改成 max 时 load/index/store 位置不变，reduction handler 的组合类型和 accumulator 更新规则改变；把 `x+1` 删除时 constant/add 两项消失，其余规约骨架保留。
- K02-R 完成。当前知识点转到 K03-R：用 19 个元素逐位置核对 CPU vector tail，并将 Triton 19/8 严格限定为源码公式推演。Q40 尚未完成。

### K03-R 事实与理解验收

- CPU 生成代码实际含 `x0=0; x0<16; x0+=16` 的 16 项主循环，以及 `x0=16; x0<19; x0+=3`、load/store count=3 的尾循环；位置 16、17、18 得到 28、30、32。
- Triton 只独立求值当前 `pid*XBLOCK+arange` 与 `<xnumel` 公式：19/8 形成三组，最后 mask 为 `[1,1,1,0,0,0,0,0]`。环境没有 CUDA/Triton，未报告实机结果。
- 迁移：17/8 仍为三组但最后只有位置 16 有效；16/8 为两组且没有第三组。CPU 的具体循环结构需重新生成，不能由 Triton program 数直接代替。
- K03-R 完成。当前知识点转到 K04-R：沿 `mm→relu→mm→sin` 推演 wrapper guards、allocation、reuse、last usage、extern/kernel 调用和 tuple return。Q40 尚未完成。

### K04-R 事实与理解验收

- 实验 14 的 wrapper 只有两次 `empty_strided_cpu`：buf0/buf1 指向 S0，buf2/buf3 指向 S1；两次 `# reuse` 都是逻辑名接管同一 storage，不是新分配。
- 第二 mm 期间 S0、S1 必须并存，因为 extern mm 不能把仍在读取的输入和输出当成同一空间；第二 mm 后 buf1 才到最后使用。
- `del` 删除 wrapper 局部引用，不等价于立刻释放设备物理内存；它把生存期事实交给运行时/内存计划落实。
- 迁移：额外返回 activated 时 S0 必须活到 tuple return，但 relu 仍可接管 first 的 S0；额外返回 relu 前的 first 时，relu 不得原地覆盖 S0，必须保留两个可观察值。
- K04-R 完成。当前知识点转到 K05-R：区分 source 内容缓存、PyCodeCache 加载、FX Graph Cache hit/miss/bypass 与无计算快速出口。Q40 尚未完成。

### K05-R 事实与理解验收

- 实验 15 使用同一独占 cache 的两个独立进程得到一次 miss、一次 hit，两次 `result_sum=272`；新增 cond 进程得到 `fxgraph_cache_bypass=1` 和结果 `[3,5]`。
- 完整编译内部：GraphLowering 生成 wrapper source，PyCodeCache 按内容写入并加载模块；FX Graph Cache 位于其外层，hit 时从保存的 source/cache key 恢复 callable，不重新运行本图的 GraphLowering codegen。
- guard 不通过的已有候选不算 hit；所有候选均不合法后走 miss。候选 artifact 恢复失败也回退到重新编译。
- 配置关闭、force_disable 或 AOT mode 未进入 FxGraphCache.load，不应写成 bypass；无 call 且非 AOT 的图在缓存选择前直接返回 boxed gm.forward，也不属于三种 cache counter。
- 迁移：仅改变不参与生成合同的运行数值可以复用同一合法候选；改变 shape/stride/symbol guard 后必须重新评估候选，不能由 FX 节点文本相同直接推出 hit。
- K05-R 完成。当前知识点转到 K06-R：贯通 kernel ops、backend、wrapper、cache/load，完成 Q40 与整个第 1 层审计。

### K06-R 与 Q40 完成审计

- 19 改 16：Q41 的 `relu(x+1)*2` ops 关系不变；Q42 的 CPU 生成不再需要 3 项尾循环，Triton 公式只需两个 block；Q43 的 guard/输出分配为 16；Q44 必须用本次 graph details 与 guards 重新判断候选，不能盲用 19 元素合同。
- 在 `mm→relu→mm→sin` 中额外返回 activated：Q41/Q42 的 relu 数学不变；Q43 让 S0 活到 tuple return，不得在后续覆盖；完整 wrapper source/合同变化后，Q44 恢复的 callable 必须与新输出结构一致。
- 额外返回 relu 前 first：Q43 连 relu 的原地 S0 复用也必须取消；这与只返回 activated 的变化不同，正文可从可观察值身份推出。
- cache guard 不满足：Q44 尝试同 hash 下其他候选，仍无合法项才 miss；它不能通过改变 Q41 的数学或跳过 Q43 的运行时 wrapper 来“适配”错误候选。
- 覆盖审计：pointwise/reduction replay、CPU tail、Triton mask、extern/template 边界、wrapper guard/allocation/reuse/return、source module load、FX cache hit/miss/bypass 与无计算快速出口均有入口、状态变化和本层出口。
- 证据分层：CPU 与 hit/miss/bypass 来自实际运行；Triton 仅源码公式；template dispatch 仅当前源码，因为本机尝试实际选择 extern。实际读者反馈仍为空。
- Q40 达到第 1 层完成条件；Q41–Q44 的内部算法仍待第二层回修。至此 Q10、Q20、Q30、Q40 的第 1 层协作均完成，可以按 BFS 进入 Q11。

## 第五轮重新复核

- 范围：系统图、四个子目录与根职责树一致，分别是 Q41 虚拟 ops 重放、Q42 设备后端、Q43 wrapper/内存、Q44 缓存/加载；最终出口始终是 boxed callable，不把孤立 kernel 当作完成。
- 过程：pointwise/reduction replay、CPU vector tail、Triton mask、extern/template 边界、wrapper guard/allocation/reuse/return、首次编译加载、FX cache hit/miss/bypass 和无计算快速出口都有具体入口、状态与出口。
- 综合：`mm→relu→mm→sin` 用 S0/S1 区分四个逻辑 buf 与两次真实分配，贯通 extern/kernel、last usage、复用、返回和 callable；额外返回不同中间值时能推出不同复用限制。
- 事实：CPU、wrapper 与 cache 三路径来自实际运行；Triton 只作当前源码公式推演，template 只按当前源码陈述，没有越过证据边界。
- 理解：只凭正文可判断 19→16、返回 activated、返回 relu 前 first、cache guard 失败分别改变哪项职责及原因。实际读者反馈仍为空。
