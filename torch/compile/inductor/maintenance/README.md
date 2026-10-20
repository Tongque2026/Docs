# 维护记录

状态依据为 [state.json](state.json)，正文开头树、四模块图与队列使用相同模块身份。当前已完成根篇、四个同层流程篇及属性与布局描述篇；正式篇目和待写队列见阅读索引，不建立空正文链接。

## 版本与边界

- 源码 `/work/pytorch`，HEAD `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 初始已有修改：`aten/src/ATen/native/dlc/SpectralOps.cpp`、`aten/src/ATen/native/native_functions.yaml`；已有未跟踪目录 `docs/inductor-guide/`、`docs/torchinductor-source-guide/`。没有阅读这些 doc 或修改任何 PyTorch 文件。
- 运行 Python 3，导入 `/work/pytorch/torch/__init__.py`，`torch.__version__=2.5.0a0+gitc77adf6`。Python 版本字符串中的构建哈希与 checkout HEAD 不同，因此运行结果是此工作树 Python 配合现有本地二进制的实验，不声称整套二进制重编译自 HEAD。
- 范围包含 `compile_fx` 边界、单图 JIT 主线、缓存旁路、AOT 出口。DLC 专用后端未展开，也未阅读 DLC kernel；GPU、训练、动态尺寸和 AOT 未做运行验证。

## 源码映射与覆盖

| 文件 | 本层阅读范围 | 本层结论 | 覆盖程度 |
| --- | --- | --- | --- |
| `torch/_inductor/__init__.py` | compile、aot_compile 入口 | 外部接口与出口区别 | 接口级 |
| `torch/_inductor/compile_fx.py` | 459–930、1242–1660 | 单图入口、缓存、图准备、Lowering 交接、前反向回调 | 流程级，未逐行完成 |
| `torch/_inductor/graph.py` | 初始化状态、778、922–1145、1268–1330、1775–1914 | 输入/输出/操作状态，算子分派，codegen 与加载 | 流程级，未逐行完成 |
| `torch/_inductor/scheduler.py` | 1726–1860、3381–3460 | 依赖、融合、最后使用与后端调用 | 流程级，未逐行完成 |
| `torch/_inductor/fx_passes/post_grad.py` | 72–150 | 图重写、属性更新、合法性维护 | 流程级，未逐行完成 |
| `torch/_inductor/codecache.py` | 1140–1195、1408–1480 | post_compile、CompiledFxGraph 状态与调用 | 接口级，未逐行完成 |

lowering.py、ir.py、各设备 codegen、wrapper、select_algorithm、async_compile 的内部属于后续模块研究范围。本篇未把依赖名字当作已覆盖实现。

## 案例选择与覆盖

| 路径或状态 | 证据 | 尚未覆盖 |
| --- | --- | --- |
| 单图编译主线 | pointwise CPU 真运行，FX、Lowering、源码、输出记录 | 大模型、reduction、模板 |
| Lowering 已组合表达式 | 一个 ComputedBuffer，Pointwise 内含三个计算 | 调度阶段融合的独立案例 |
| 无计算 kernel 但有 wrapper | view CPU 真运行，别名指针一致 | 非连续 view、复杂别名 |
| 输入列表生命周期 | 两例调用后列表为空，输入 Tensor 值保持不变 | 输入原地写回 |
| FX 图缓存命中 | 阅读入口与 post_compile；本实验禁用缓存 | 缓存命中运行、guards 变化 |
| 无调用节点快捷返回 | 阅读 count_calls 条件 | 未单独运行 |
| 缺少 lowering | 阅读 fallback 条件及异常 | 未注入失败 |
| JIT / AOT 不同出口 | 阅读 compile_to_fn 分支 | AOT 产物执行 |

## 复现与采集

```bash
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/capture.py
python3 /work/learning/gpt6-1sol/torch/compile/inductor/assets/draw.py
```

实验使用 make_fx 构造明确的 ATen 图，再直接调用 compile_fx_inner；不以其替代完整 torch.compile 上游。关闭 FX 图缓存和 CUDA Graph，编译线程为 1，缓存位于 experiments/cache。脚本对 GraphLowering.run 和 save_output_code 作进程内采集，finally 恢复；不改源文件。

`*_fx.txt` 是进入前的原始 FX；`*_lowered.txt` 是 run 完成后的图和 operations repr；`*_output.py` 是完整生成程序；`results.json` 保存数值、布局、共享存储、输入变化及生成 kernel 数。正文 IR 索引段是简化推演，C++ 与 wrapper 片段摘自生成程序。生成文件里的头文件绝对路径依赖本次环境；复现应重跑脚本，不把这些文件当作跨机器可移植程序。

两个案例均通过 torch.testing.assert_close，并记录布局和别名结果。输入不变、指针关系及列表清空为显式观察值，不单凭数值相等推断。未测性能，不报告加速比。

图使用 SVG 原生元素生成 PNG，字体 Noto Sans CJK SC；已以实际渲染图检查文字、箭头与旁路，调整过图注与主线重叠。保留可编辑生成脚本和 SVG。用户图片仅作为视觉风格参考，没有从其标签推定当前源码事实。

## 下一篇

按广度优先队列写 `01-graph-preparation/README.md`，解释图准备内部整体流程与具体改写交接。完成同层四篇前不正式交付更深层篇章。拆分需以完整职责为单位，任一层最多四个直接子模块；共享缓存的主要归属保持为代码生成与运行封装。

## 历史进度：2026-10-20 19:30

本轮已完成四个同层内部流程篇；当前以 [state.json](state.json) 和 [阅读索引](../INDEX.md) 为准。四个模块各有 3–4 个直接子模块，当时尚无“叶子·逐行完成”；下层不能据此视为已覆盖。

补充阅读了 `fx_utils.py:55–202`、`post_grad.py:636–800、974–982`、`lowering.py:458–545`、`ir.py:626–695、1135–1187、6330–6445`、`scheduler.py:1898–2015、2266–2340、2880–2998、3223–3275、3450–3530`、`codegen/common.py:196–250`、`codegen/wrapper.py:855–1025`、`kernel/mm.py:155–250`、`async_compile.py:175–283`、`codecache.py:2845–2950`。这是本层接口和流程覆盖，不是完整逐行覆盖。

新增 `experiments/stages.py` 真实调用单图编译器，进程内采集 post-grad 前后、Lowering 出口、融合前后、last_usage 和完整生成源码，finally 恢复。所有缓存写入 `experiments/cache-stages`，不修改产品源码。复现：

```bash
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/stages.py
python3 /work/learning/gpt6-1sol/torch/compile/inductor/assets/modules.py
```

七个案例：内部 clone、输出 clone、view、宽度 3 归约、宽度 64 归约、mm→relu、多输出。全部通过 assert_close，检查输入内容未变。采集结果确认内部 clone 删除而输出 clone 保留；view target 先变 reshape；小归约为展开 Pointwise，大归约保留 Reduction；多输出存在真实 Scheduler 横向融合；mm→relu 不融合但生成程序复用内部存储。

生成 kernel 数依次为 1、1、0、1、1、1、1；外部 mm 不计入自生成 kernel。原始结果保存在 `experiments/stages/results.json`，每个阶段文件用 case 名称前缀区分；正文精简表示不冒充完整 repr。大归约是为验证另一条真实路径新增的条件，不是调整计时或人工模拟候选。未进行性能计时。

四幅新图均已渲染并在实际图片中检查；Lowering 图改为普通/专用算子分支，避免将条件路径画成必经阶段。图准备明确收尾模块两次参与，调度明确分析与生成驱动同属一个职责。

下一篇更新为 `graph-preparation/properties`。之后按 state.json 中同深度队列逐项交付，不提前将子篇下的更深节点插到队首。仍需展开 pattern matcher、reinplace、符号与别名、专用算子、完整融合算法、设备生成、编译缓存和运行机制，不能把本次四个流程篇视为整个系列完成。

## 历史进度：属性维护篇（2026-10-20 19:30）

新增属性与布局描述篇，划为“初次属性建立”和“增量属性维护”两个子模块；现已完成本层，尚未逐行完成。受控实验 properties.py 将 relu 节点 target 改为 sum(dim=1)，保留节点身份，再调用真实 FakeTensorUpdater。采集证实编辑后但更新前 relu/mul 属性仍为 [2,3]；更新后两者都为 [2]。重新生成 FX forward 后真实计算为 [-6,12]，与 reference 一致，输入不变。该实验不是编译、kernel 调优或性能结果。

复现：`python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/properties.py`。证据为 experiments/properties.json；新增阅读 fx_utils.py:200–224 的参数取值辅助函数。

当时下一篇为“图重写”，之后继续完成同深度其他模块，最后才处理更深层的初次/增量属性叶子。具体以 state.json 的去重广度优先队列为准。

## 历史进度：图改写到节点解释（2026-10-20 19:40）

已完成图重写、写入语义恢复、图收尾、解释图与登记输入四个模块的本层篇。它们分别有 4、3、2、3 个直接子模块，尚未将未展开的子模块称为叶子。状态和目录以 state.json 为唯一依据，sync_state.py 现仅据其渲染各篇位置树、子篇链接和索引，重算广度优先队列，不在脚本另存一份模块划分。

新增源码研究覆盖：

| 文件 | 范围 | 结论和当前覆盖 |
| --- | --- | --- |
| fx_passes/reinplace.py | 1–371、371–689 | scatter 规范化、旧值/别名观察、copy epilogue、clone 清单；流程级 |
| fx_passes/post_grad.py | 800–896、1005–1050、1088–1319 | 包装分解、addmm 条件、构造迁移；流程级 |
| pattern_matcher.py | 970–1040、1598–1760、1893–1933 | entry 分派、mutation region、模式应用、稳定排序；流程级 |
| fx_passes/group_batch_fusion.py | 1–155、1284–1318 | 配置生成与批量规则入口；接口级 |
| fx_passes/ddp_fusion.py | 581–600 | 按 pass 配置和签名调用；接口级 |
| fx_passes/micro_pipeline_tp.py | 834–860 | 通信候选与简单重叠过滤；接口级 |
| graph.py | 261–449、803–847、1156–1185、1268–1621 | 图状态、输入尺寸、节点派发、符号核对；流程级 |
| virtualized.py | 95–140 | 线程局部状态与作用域恢复；接口级 |

新增可复现证据：

```bash
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/rewrites_mutation.py
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/finalization.py
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/interpretation.py
```

rewrites_mutation 直接运行真实 post-grad 或 reinplace pass，再执行 FX 图；共六个案例。Tensor bias mm+add 变 addmm，标量 bias 规则拒绝；内部 index_put 可原地化，图输入无回写时拒绝，有 copy 且无旧值观察时删除 copy，有中间旧值观察时拒绝。比较了结果、执行后输入和输出到输入的存储别名，全部通过。没有把这些解释成完整 Inductor kernel 运行或计时结果。

finalization 排序案例实际运行并确认幂等，数据环实际触发 AssertionError。构造迁移四个案例使用人工构造的真实 FX 图和 FakeTensor 设备属性，运行真实 pass；确认唯一目标设备且允许使用链时改成 cuda:0，输出设备约束、多设备、bool 索引 CPU 标量三个分支保留 CPU。它们不是实际 GPU 运行证据。

interpretation 只运行真实 GraphLowering，不生成后端程序：非连续 x 的 stride 为 (1,2)，二维 k 登记 ConstantBuffer，零维 s 内联 Constant；add 返回时 operations=0，mul 因 output 使用在 run_node 期间物化，operations=1；output 的 Python 返回为 None，但 graph_outputs 有一个结果。记录保存在 interpretation.json。IR 打印也放在相同 V.graph 上下文，避免把动态依赖的常量名称当作脱离图仍可随时读取的纯字符串。

四张图已渲染检查。图重写明确添加 pattern_matcher=False 旁路；节点解释改为按节点类别分派并沿独立右侧通道循环，避免误画成全部常量都先于全部计算登记。共享属性和相邻写入恢复标明归属。未验证通信、真实 GPU 迁移、自动功能化高阶算子运行或新动态符号断言。

当时下一篇是 Lowering 的“表达计算与布局”；顺序、完整树和更深层待写单元见 INDEX。上一目标轮已产生正式文档、实验和持久状态，分类为 progress；本轮同样有上述新增产物和证据。整体目标仍 active。

## 历史进度：表达式、物化、专用算子与依赖（2026-10-20 19:51）

本轮完成 Lowering 的后三个模块及调度依赖模块的本层篇，各自拆为三个职责完整的子模块；没有改变整个 Inductor 讲解范围，也没有将这些流程篇冒充完整逐行叶子。上一目标轮有文档、运行证据和持久状态变更，分类为 progress；本轮同样新增了以下权威证据。

源码研究新增/深入范围：lowering.py 的 transform_args/_register_lowering/make_pointwise、expand/view/permute、_make_reduction_inner/make_reduction/sum_、mutate_to/copy_、fallback_handler/make_fallback、cond/while_loop；ir.py 的 BaseView/ExpandView/PermuteView/View/ReinterpretView、FixedLayout.make_indexer、StorageBox、MutationLayout、ExternKernel.realize_input、FallbackKernel.create、Conditional/WhileLoop；dependencies.py 的 MemoryDep/StarDep/WeakDep/ReadWrites 与 extract_read_writes；scheduler.py 的节点读写设置、compute_dependencies、DCE、拓扑与祖先。当前仍为流程和接口覆盖，各函数完整实现待按队列逐行展开。

```bash
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/expressions_storage.py
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/special_ops.py
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/entry_boundary.py
```

expressions_storage 在独立缓存中运行完整单图编译与 CPU 程序。八个案例：broadcast、permute、expand、非连续 reshape、复用 exp、返回旧值并 copy 输入、只修改输入但不返回、删除死旧读。全部比较数值、调用后所有输入、输出对输入的存储别名。记录节点事件、Lowering repr、依赖建立后、DCE 后/融合前、融合后以及生成源码。精确结果保存 experiments/expressions-storage/results.json。

重要观察：

- 广播读取 y 的索引没有第一维；permute 输出 stride 为 (1,3)，expand 为 (0,1) 且共享输入，非连续 reshape 输出不共享原输入。
- reshape 的 raw make_fx 输入已含 clone/_unsafe_view，不将 clone 的出现误归因于后续 View.create。
- CPU 多用户 exp 在 Lowering 被实现为 buf0，后面两个输出为 buf1/buf2；调度融合后 wrapper 只分配 buf1/buf2，buf0 只为循环内值。
- copy_old 记录 WeakDep(buf0,mutating_buf=buf1)，先旧读后写；DCE 对只有 weak user 的旧读会删除并清除弱边；mutate_only 不返回 x 也保留其修改。修改输入索引均为 [0]。
- copy_old 的 generated_kernel_count 为 2，但 wrapper 只调用一个 cpp_fused_0，其内部有两个按顺序执行的循环。正文没有将这个统计当作外部函数调用数。

special_ops 在进程内通过 torch.library 注册 pair 双输出教学算子及 CPU/Fake 实现；不改 PyTorch。禁止 implicit fallback 时实际抛 MissingOperatorWithoutDecomp，允许后生成一次 FallbackKernel、两个 MultiOutput 和 relu 计算，真实输出与输入别名检查通过。生成源码依赖该进程注册；复现应运行实验脚本，不能将输出文件当作未注册算子的独立可执行程序。

同脚本构造 cond/while_loop 高阶算子图并真实运行 CPU：cond 的 true/false 分支均验证；while 的三次迭代与零次迭代均验证，零次的两个结果分别共享原 carried 输入存储。采集到父图 never_reuse_buffers 保护这类可能别名输入的输出。证据为 experiments/special-ops/results.json 与子图/父图 Lowering 和生成程序。没有运行 GPU 子图、CUDA Graph 或多候选模板调优。

entry_boundary 单独复现 raw make_fx 的 aten.t.default 直接交给单图入口时的 fallback/decomposition 冲突断言；这是已有入口要求的证据，不修改注册规则、不称作本轮修复。主实验用已有 lowering 的 permute 表达相同转置。错误结果与精确原图保存在 experiments/entry_boundary.json。

Conditional.create 当前代码的输出别名遍历体复用 true_outputs，而非按变量 outputs 检查每一分支；正文只陈述实际检查范围，不宣称 false 的所有别名已被验证。此处保留为版本实现行为，后续逐行篇完整解释；不修改产品源码。

新增四张图已实际渲染检查，普通表达式、物化需求与专用算子各用三分支图，避免把条件路径画成必经串行阶段。依赖图按实际主线连接读写、额外约束与活跃性处理。下一篇为“融合与分组”，剩余同深度流程篇完成后，按广度优先进入现有更深队列。


## 最新进度：2026-10-20 20:05

本轮新增剩余六个同层模块篇：融合、生命周期、kernel、wrapper、编译加载、缓存运行适配。至此根篇、四个一级篇及其十五个直接模块篇均已完成本层；之后才正式交付第一篇深一层叶子“初次属性建立”。这一顺序遵守现有广度优先队列，没有把简单叶子插到未完成同层流程之前。当前源状态为 20 个本层篇、1 个叶子篇；整体系列未完成，下一篇为增量属性维护。

上一目标轮新增正式文档、源码依据与实验，分类 progress。本轮同样完成文档和下面的新证据；整体目标仍 active。

新增源码研究：scheduler.py 的 get_possible_fusions/fuse_nodes_once、score_fusion、can_fuse/纵向检查、分组维护、last_usage、释放与 flush；codegen/cpp.py 的 CppScheduling/CppKernelProxy 接口、load/store、KernelGroup 定义与 flush；codegen/common.py 的 KernelArgs/CSE；triton.py 的 kernel/load 与分块接口；wrapper.py 的前缀、Allocate/Free/Reuse 行、规划、分配限制、子图/条件/循环；async_compile.py 的线程/进程任务、cpp_pybinding/triton、wait；codecache.py 的 Cpp/Python 代码缓存、FX 键/序列化/恢复/post_compile。这里除登记的叶子范围外仍属于流程或接口级研究。

```bash
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/fusion_lifetime.py
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/cache_lifecycle.py
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/async_loading.py
python3 /work/learning/gpt6-1sol/torch/compile/inductor/experiments/initial_properties.py
```

fusion_lifetime 四个 CPU 对照：融合大小 64/1，以及 mm→relu 允许/禁用两项存储复用。真实记录融合前后节点、last_usage、codegen 完成集合和 wrapper 登记。两种融合结果分别是一个融合节点/两个普通节点；两种复用的节点与 last_usage 相同，wrapper 却分别复用/独立分配。结果、输入保持与输出不别名输入均校验。没有测时。

cache_lifecycle 在各不相同的五个 worker 进程里共享本轮私有缓存。FakeTensor 示例配 ShapeEnv/tracing context，分别观察首次 miss（Lowering 1）、新数值 hit（0）、合法动态 (3,4) hit（0）、float64 miss（1）、无 context bypass（1）。五条路径 post_compile 均为 1，真实新数据输出校验通过。主脚本显式断言这些关系；结果和 hash 组成保留在 cache-lifecycle。首次用真实 Tensor 示例的对照另存 real_examples_results.json，证实 real Tensor reducer 会把值纳入键；不把它当作 FakeTensor 键行为。

动态 FakeTensor 示例生成 s0/s1 参数，合法 shape 变化不会自动变 key。这里没有把 shape 不同武断当作 miss，也没有用 metrics 累计计数证明 Lowering 是否发生，因为缓存恢复会应用保存的统计增量。关键证据来自直接包装 GraphLowering.run/post_compile。

async_loading 实际以 compile_threads=2 编译本地 CPU float* 函数，记录 LambdaFuture 经 wait 替换为 builtin_function_or_method；之后执行将 Tensor [5] 改成 [12]。它是编译类接口的受控实验，不是整个模型性能或 GPU 并行结果。

第一篇叶子完整保留 view_to_reshape（post_grad.py:974–981）和 fake_tensor_prop（compile_fx.py:392–417）原始实现，逐行解释签名、检测、分支、参数转换、临时 flag 和返回；没有省略逻辑。相关源码 hash、范围与正式篇目保存在 leaf-coverage.json，验证器检查正文代码块与完整本地源码逐字一致。这种结构检查不代替逐行语义阅读；本篇还人工沿同一图核对两个产物（mode 返回、metadata 副作用）。

initial_properties 六个真实传播路径覆盖：无 mode 新建、已有 mode 复用、混合输入拒绝、force 临时放宽并恢复、ambient context 检测、不同 mode 冲突拒绝；记录图节点身份和 recompile 前旧 forward 未更新。输入数据保持不变，没有执行后端 kernel。依赖 detect_fake_mode 和 FakeTensorProp 的相关契约据源码核对，不向整个上游系统扩张。

七张新增图已渲染检查：融合/codegen 用返回通道，JIT/AOT 有独立出口，缓存 hit 与编译/保存分路汇合到 post_compile，叶子图有 mode 分支和冲突异常出口。各篇树、子篇链接和 INDEX 从 state.json 同步。PyTorch 产品源码未改；GPU、CUDA Graph、AOT 和更深实现仍在待覆盖队列，不能把本轮完成视为整体目标完成。


## 最新进度：2026-10-20 20:22

本轮按广度优先新增三个完整叶子：增量属性维护、死节点清理与局部排序、空操作与存储保护。当前 20 个本层篇、4 个逐行叶子；剩余队列 40 项，下一篇为模式匹配。整体目标仍 active，本轮属于 progress。

完整源码范围新增 fx_utils.py:55–185、188–219；post_grad.py:79–84、160–189、636–736、739–797；pattern_matcher.py:1609–1657。加上已有两段，共九段原文范围，leaf-coverage.json 保存逐字核对依据。外部接口边界和共享辅助函数主要归属登记在 dependency-frontier.json。

新增真实受控实验 incremental_properties.py 五条路径，覆盖 shape 变化、属性等价、输入 metadata 缺失、dtype 单独改变、alias 改 clone；cleanup_noops.py 十六个案例覆盖排序、mutation region、DCE、内部/输出 clone 与 cat、pow、pad、dtype 转换及两种 output 结构。记录图、数值、输入修改、对象身份与存储关系。没有修改产品源码或运行 GPU。

需要读者知道的本版本行为已经放入对应正文：FakeTensorUpdater 等价比较没有显式比较 dtype，受控改 target 后可保留旧 metadata；相等分支没有保存新签名。remove_noop_ops 的 node in output_node.args 对通常 tuple 包装输出不能命中直接 Node，因此 alias tuple 输出的 Python 对象身份可改变；单 Node 输出对照则保留 alias。这些是局部 pass 实验，不宣称整个 torch.compile 的最终结果必然如此。

三张新增图均实际渲染并查看，包含灰度对照；返回线独立通道避免跨文字。verify.py 通过：367 个本地链接、九段完整源码以及所有已登记实验结果；该检查只验证已写范围。下一篇按队列展开 pattern matching，不将待写节点误标为叶子。


## 最新进度：2026-10-20 20:31

上一目标轮完成三个源码叶子并登记实验证据，属于 progress。本轮按广度优先完成模式匹配与批量通信两个本层篇，分别采用三个职责子模块，保持每层不超过四项。当前状态为 22 个本层篇、4 个逐行叶子，队列 44 项；下一篇为 View/scatter 表示与分解。新加入的六项职责仍明确标记待展开，本轮不把接口级研究当作逐行覆盖。

模式篇核对 pattern_matcher.py 的 Match/上下文、算子与参数匹配、变长/列表/多输出接口、三种 entry、登记/生成/追踪、apply 扫描，以及 graph.py 的 lowering handler 派发。实验 pattern_protocol.py 共十二项证据：五个 Graph entry 成功/拒绝路径、同名捕获两项、同根多 entry、候选快照、函数式 replacement、IR 图重传播失败边界、完整 CPU lowering handler 编译执行。注册使用独立 PatternMatcherPass，不改产品规则。完整编译通过 custom post pass 在初次属性传播后插入 handler，实际收到 TensorBox；运行生成程序不再调用该 Python handler。

模式实验特别区分：内部多用户和跨 mutation region 的拒绝发生在 extra_check 之前；根多用户允许；Graph handler 不改图仍可能被计入应用次数；新根不追加到当前快照。函数式 replacement 的实际事件为登记搜索追踪，应用时当前属性搜索追踪、用户检查、替代追踪。已插入纯 IR handler 的图直接重新 fake_tensor_prop 会收到 FakeTensor 并抛 StopIteration，这作为独立边界实验保存，不冒充完整编译失败。

批量篇核对 group_batch_fusion.py 的配置分流、候选 BFS、独立子集、post-grad relu 构造，以及 micro_pipeline_tp.py/DDP 的入口和条件。batch_communication.py 九项证据：五个真实 CPU relu 改写/执行对照，真实独立子集，真实 wait 图重排，DDP 配置 dispatcher，TP all-gather finder。当前 has_fbgemm=False，require_fbgemm 规则未加入，普通规则仍执行。没有创建 process group、启动 collective、运行 GPU fused op 或声称通信性能通过。

五个同形状 relu 的直接 FX 结果共享 batch 存储，元素 offset 为 0/6/12/18/24；原始结果是独立存储。数值、输入保持与输出不别名输入均通过，同时保存输出之间的存储矩阵并在正文明确这种变化。没有把节点减少当作性能收益。max/min=2 的五项对照产生两个批次和一项剩余；依赖候选 relu→relu_1 与独立 relu_2 只交出 [relu,relu_2]。

新增两张图已渲染查看彩色与灰度：模式图用配对连接点收回三个 entry，避免回线穿过框内文字，并明确延迟 IR 调用；批量图保留普通模式插入位置及独立 TP/DDP 关闭旁路。state.json、各篇阅读树、INDEX、dependency-frontier 与验证器同步。检查通过：414 个本地引用，九段既有完整源码，所有登记实验；源码逐行覆盖范围本轮未扩大。PyTorch 工作区差异与先前一致，未读取其 docs 目录、未改产品源码。


## 最新进度：2026-10-20 20:37

上一目标轮完成两篇本层讲解、两张图和二十一项实验证据，分类 progress。本轮按既定 BFS 交付 View/scatter 逐行叶子，未增加目录层级。当前 22 个本层篇、5 个逐行叶子，剩余 43 项；下一篇为存储可写性与原地化。整体目标仍 active。

完整范围：reinplace.py 36–88、91–160、163–222、225–325，覆盖四类 scatter/view 映射、带 FakeTensor 属性的节点构建、ViewOp、两种广义 scatter、递归功能分解、写入分解、策略判断、分解驱动及规范化/嵌套合并。完整源码逐字登记新增四段，累计十三段；原文、逐行表与具体递归状态对照，不修改产品实现。核心原地化仍归兄弟模块，不把这次实验调用误当作该核心已逐行覆盖。

实验 view_scatter.py 成功完成十五项：八组图（路径对接、返回内层、路径不对接、不同 base、copy 回写、内部 base、diagonal、as_strided）、两种直接 helper、广播、copy 错误、view 错误、空路径分解及中间 recompile 边界。八组均记录规范化/核心/分解图和中间策略，在中间使用真实 FX Interpreter、最终 recompile 执行，检查数值、输入修改、输出对输入存储；另保留输出间存储矩阵。

主要证据：输入 placeholder 与返回结果的 realized 预测均为 True 时，should_reinplace=True，仍因无 copy 回写意图保持功能 target；有 copy 时改成原地 target，最终没有 clone。内部 add base 的 realized 预测 False，策略 False。diagonal/as_strided 功能 target 即使不能原地化，也以 clone+view+copy 分解，不修改原输入。返回内层结果的合并将其表达为完整结果的 view，导致两个原独立输出共享存储、内层元素偏移为 4，已明确写入正文，未宣称完整别名保持。

规范化图携带 ViewOp dataclass，不是最终可生成 Python forward 的表示；实际 gm.recompile 抛 SyntaxError。实验将此保存为受控边界，并用 Interpreter 验证该阶段；正常分解后 recompile 成功。直接 helper 的空路径可写整块，但人工空路径功能节点的递归分解在 view_ops[0] 抛 IndexError，规范化器不创建这种节点。copy_ 错误重包装保留 cause，view 越界在 try 外抛出。这些都已与正常模型运行区分。

图已实际渲染查看彩色和灰度，主线明确规范化→兄弟核心→两种分解，不把策略当写入许可；无回线跨字。新增 is_node_realized 与外部 _is_view_op 的主要归属。验证通过：428 个本地引用、十三段原始源码与已有证据核对。没有 GPU、kernel 编译、符号 shape、重叠 as_strided 写入或性能测量；未查看 PyTorch docs、未改产品源码。


## 最新进度：2026-10-20 20:49

上一目标轮完成 View/scatter 叶子和十五项证据，分类 progress。本轮按 BFS 完成存储可写性/原地化及自动功能化包装分解两个逐行叶子；写入恢复的三个直接子篇现在都已完成。当前 22 个本层篇、7 个逐行叶子，队列 41 项，下一篇为顺序与入口一致性。整体目标保持 active。

原地化完整范围新增 reinplace.py 的 29–33、328–368、371–420、422–496、498–581、583–682、685–688，四段合起来保留完整核心函数的全部有语义代码；包括未使用的 mutated_inputs 集合、Triton JIT/Autotuner 分类、foreach、替换链收尾，未因无设备而省略原文。包装分解完整保留 post_grad.py:800–892 的三个 handler、decomp 闭包、apply 和三个残留断言。新增八段 hash，累计二十一段逐字核对范围。

reinplace_storage.py 实际完成二十二项执行/边界对照，并单独探测到本机缺少 Triton。普通 index_put 覆盖内部量、输入无回写、有回写、copy 前 view 旧读、copy 后新读、输入 view、缺少存储 metadata；直接构造的真实 metadata-only view→sym_size 图允许复用。foreach 对照覆盖完整回写、部分回写、重复存储，另有一次正常完整 CPU 编译。

foreach 核心出口仍有死 getitem，eager 原地 foreach 返回 None，直接 FX 执行实际 TypeError；实验显式 DCE 后再执行可通过。正常完整编译另行从原始功能图运行成功，生成 wrapper 直接 cpp_fused_0(x,y,x,y)，mutated_input_idxs 为 [0,1]，数值、输入修改和输出别名正确。源码 register_foreach_inplace 返回 mutation IR 列表，明确区分 eager 与 lowering 的返回契约，不将局部 FX 边界误报为完整编译失败。

八组 v1/v2 包装实际决策为 []、['y']、['x']、['x','y']、[]、[1]、[]、[0]；实际分解 clone 数为 0/1/1/2/0/1/0/1，全部对照数值、调用后输入、输出对输入别名。两个真实 mutable Tensor 列表包装独立存储可全部复用，重复存储保留整个列表并 clone 两项。FX immutable_list 显式实现 hash(tuple(self))，已读实际容器源码核对，不依普通 Python list 行为作推断。所有注册只在实验进程内部，不改产品规则。

functional_wrapper_boundaries.py 新增八项独立证据：v1/v2 跳过前置核心时缺省空名单造成无 clone，返回值虽一致但输入修改不同；执行核心后分别保护参数名/base 编号，均有一次 clone 且输入不变。真实 disable_cpp_codegen CPU 过滤使 v1/v2 残留并触发精确 AssertionError。Triton 残留测试只构造目标节点与 CPU Fake metadata，验证末尾断言，不执行 kernel lookup、dense 或 GPU。普通无 wrapper 图保持不变并返回 None。未把复用的十个正常包装分解案例再统计为新证据。

源码也核实 run_functional_passes=False 跳过 fwd_only 中替代图的 remove_noop_ops 与 DCE，并非关闭所有功能化或 decomposition；父篇、叶子及图示一致使用这个具体接口含义。dense 的参数名/base 列表、布局重建与字典返回按源解释，高阶设施作为边界外接口，不展开上游系统。

两张图均实际渲染检查彩色与灰度；原地化图明确反向建表、正向逐节点与扫描后删除，循环线位于文字外；包装图明确三种名单来源、flatten/unflatten、追踪与残留失败出口。状态、阅读树、INDEX、依赖归属和验证器同步。当前检查通过：462 个本地引用、二十一段完整源码、全部已登记证据。PyTorch commit 与既有工作区改动均未变化，未读取 docs、未改产品源码。无 GPU、Autotuner、collective、AOT get_attr 或性能测量证据，限制已在正文列明。


## 最新进度：2026-10-20 20:59

上一目标轮完成两个逐行叶子并留下三十项运行/边界证据，分类 progress。本轮完成图收尾两个逐行叶子。当前 22 个本层篇、9 个逐行叶子，待写 39 项。完整覆盖新增 pattern_matcher.py:1893–1932；post_grad.py:133–148、1088–1114、1117–1311、1314–1318，累计二十六段逐字核对范围。133–148 明确为调用职责块，不冒充整个 post_grad_passes 或其通信 helper 的实现覆盖。

用户在 20:54 明确要求“写完这个之后，优先讲解和支持新设备有关系的部分模块”。按此新指令先完成本轮图收尾，然后改变交付优先顺序；这是用户对原 BFS 默认顺序的明确覆盖，不缩减整体范围。state.json 保留唯一模块树和标准 breadth_first_queue，新增 priority_focus 与派生 reading_queue。sync/verify 对设备相关选定子树按深度及指定先后优先，其他待写节点保持原 BFS，相同节点不重复。当前 next 为 codegen-runtime/kernels/backends-loops。INDEX 显示同一优先队列，未来续写继续遵守该用户偏好。

已定位优先设备接口的实际代码：codegen/common.py 的 DeviceCodegen、register_backend_for_device、get_scheduling_for_device、get_wrapper_codegen_for_device、init_backend_registration 中的 PrivateUse1 支持，以及 DeviceOpOverrides 注册；scheduler.py:3333 消费 Scheduling，graph.py:1658 消费设备操作与 wrapper。这里只记录下一篇源码入口，尚未将待写节点标完成。

order_entry.py 九项证据：两种多输入乱序及真实 sorter locals 跟踪、无依赖对的相对顺序变化、嵌套/重复参数收集、旧 forward 与 Interpreter/recompile 对照、数据环、外图依赖、lint ownership、未编码写入顺序边界。sorted 再运行不变，原 Node 与附加 metadata 身份保留。末参数先 ready 的候选两次挂起；stable 不承诺保持所有无效原图的独立节点相对顺序。手工改变 copy/read 次序后两个链表都满足显式拓扑，排序与 lint 不修复语义差异，作为受控反例清楚标注。

constructor_devices.py 二十二项真实谓词与 Fake metadata 改图对照：唯一设备成功、输出默认阻断/显式放开、无目标、多设备、缺省/字符串 device、外部 CPU 输入、两构造关联、关联输出阻断、重复引用保守出口、不支持的 callable、同一 GPU index_put 边界两个构造，以及七项特殊索引条件、目标类型错误、tuple metadata。重复输入 add(c,c) 的 cpu_indeg=2，而 users 只遍历一次，实际阻断；构造 kwargs.device 改写不改变原 val 对象或 Node 身份。全部 CUDA 属性来自 meta 存储 FakeTensor，没有 GPU 分配、kernel 或通信。

特殊 index_put 实际 dtype 集合为 bool/int8，不含 uint8；按源码和七项对照解释，不用注释中的 byte 一词推断运行分支。CPU 单元素判断用 numel()==1，包括 shape(1,)；未据该谓词声称 int8 等索引实际 ATen 执行合法。cannot_be_moved 和 find_movable_constructors 原 docstring 的方向文字与条件不一致处，正文明确按实际返回值/target 解释。

两张图已渲染检查彩色与灰度，排序图分清挂起/放好/唤醒、后续跨模块与最终 recompile/lint；设备图分清计数、关联集合、阻断与最后只改 kwargs。源码原文、各篇阅读树、父篇、INDEX、优先队列、依赖主要归属和验证器一致。当前验证通过：485 个本地引用、二十六段完整源码及已登记证据。未读取 PyTorch docs、未改产品源码。整体目标仍 active，下一篇优先讲设备后端注册与调度/代码生成接口。


## 最新进度：2026-10-20 21:06

上一目标轮完成图收尾两篇、三十一项排序/metadata 证据并落实设备接入优先队列，属于 progress。本轮按新队列交付“后端与循环组织”本层篇，形成三个职责子模块：后端注册与能力查询、调度实例与后端委托、循环展开与 kernel 分组。它们仍待完整逐行展开；当前文档状态为本层完成，不把注册接口研究当作整个 CPU/Triton 实现覆盖。当前 23 个本层篇、9 个逐行叶子，待写 41 项；next 为 wrapper 入口与设备调用。新增后代继承设备接入优先级，非设备剩余范围未删除。

源码核对 codegen/common.py 的 DeviceCodegen、DeviceOpOverrides、两个注册表、BackendFeature、默认与 PrivateUse1 初始化；Scheduler.create_backend/get_backend、BaseScheduling；CUDACombinedScheduling 的普通/模板委托；CppScheduling/CppKernelProxy 的范围、类型与循环入口、KernelGroup/flush；SIMDScheduling 的 node schedule；GraphLowering 初始化 wrapper 与正式 codegen 的先后。还核对实际设备模块注册与 PrivateUse1 自定义 hook 查询。此轮源码覆盖 hash 不增加，既有二十六段完整逐行范围保持。

backend_registration.py 十个隔离 worker 成功完成：PrivateUse1 默认名、改名无模块、缺少 hook、Cpp hook 为 None、完整 hook、先显式登记、初始化后才加载模块，以及 lookup/覆盖/能力、device ops、完整 CPU 委托编译。PrivateUse1 改名只在各 worker 内进行，原始名均实测为 privateuseone。没有注册或运行 PrivateUse1 allocator、Tensor kernel 或 runtime。

完整 hook 的 worker 实际调用 GraphLowering.init_wrapper_code 与 Scheduler.get_backend/create_backend，图状态由最小 fixture 提供，说明真实选择器如何消费登记；同类型不同 index 得到不同 backend 实例，同一个完整 device 复用实例。这些 stub 没有 kernel 方法，明确不当作新硬件完整编译。

CPU worker 的 LessonScheduling/Wrapper 继承并委托已有 C++ 实现，先登记后进入正常完整单图编译，初始化没有覆盖它们。实际事件为 wrapper、Scheduling(None) 能力查询、Scheduling(本图)、group([6],[])、flush、codegen_node、flush、kernel_call(cpp_fused_0)。两次 flush 只有一次调用登记；真实执行 relu(x+y)*2 得到 [[0,0,0],[2,4,6]]，输入不变，输出不别名输入。精确生成代码与事件保留 backend-registration。该结果只证明注册 API 被完整 CPU 路径消费，不声称新物理设备支持。

重要接口条件已进入正文：自动发现要求 Scheduling/Python wrapper/C++ wrapper 三项为真；显式登记省略 C++ 项时保存 NoneType 类，两路径不能混同。init_backend_registration 有 lru_cache，后加载模块直接重试不生效，受控 cache_clear 后才被发现。任意 registry 字符串不自动成为 torch.device 名；能力查询用 Scheduling(None)。DeviceOpOverrides 独立注册且 getter 仅在整个字典空时导入默认模块，先登记其他项再查 cuda 的隔离路径返回 None，主动导入后可查到 CUDADeviceOpOverrides。生成 guard/stream 字符串未执行设备操作。is_gpu 当前仅 cuda/xpu，不能从 PrivateUse1 登记成功推断全部分类与 guard 行为已有支持。

新图已实际渲染检查彩色与灰度，三个子模块与目录名称一致，明确 wrapper 先建立、Scheduling 后构造、表达式为兄弟模块、flush 汇总再交 wrapper。父篇、INDEX、state/reading_queue、依赖归属与验证器同步。当前检查通过：510 个本地引用、既有二十六段原文与全部登记证据。未查看 PyTorch docs、未改产品源码，未运行 GPU/Triton、PrivateUse1 硬件或 benchmark。整体目标保持 active，按用户要求继续优先讲 wrapper 设备上下文与调用。


## 最新进度：2026-10-20 21:15

本轮完成“入口与设备调用”本层篇，按用户设备接入优先顺序继续推进。拆成入口参数与程序装配、设备上下文与 stream、kernel 调用与启动参数三个职责，完整逐行覆盖仍待子篇。状态合计 24 个本层篇、9 个逐行叶子、43 个待写节点；下一篇为图上下文与符号状态。既有二十六段完整源码 hash 未增加。

源码核对 wrapper 构造与缓存、header/prefix、符号绑定、设备行对象、stream 导入/赋值、generate/_generate、调用参数和三种启动分支、compile-time autotune，以及 Scheduler 设备切换、GraphLowering wrapper 选择、CUDA/XPU overrides 和 C++ wrapper 调用入口。重点区分生成时追加文本/行对象、最终规划与发射、模块加载等待、call(args) 运行四个时点。共享内存计划和输出只说明接口，归原兄弟节点；C++ 调用 ABI 实现登记后续覆盖，不冒充当前完整讲透。

wrapper_devices.py 十六项记录全部成功。真实 CPU 完整编译执行 x+1，shape[2,3]/stride[3,1]/float32，输出 [[1,2,3],[4,5,6]]；输入列表清空、另持 Tensor 内容不变，错误 shape 与 stride 都在入口 AssertionError。没有把列表引用变化当作 Tensor mutation，也未据数值结果额外断言输出别名。

设备协议案例调用真实 WrapperCodeGen 和延迟行对象，使用最小图 fixture；生成体以显式 Python stub guard/stream/kernel 执行，事件为 enter,set_device,stream,a,b,exit。两次运行 getter 查询两次，源码只有一条获取赋值；换 graph 后两条赋值、import 仍一次。GPU/CUDA C++/Triton 启动、编译期 autotune 和 C++ guard 都只验证源码发射；无 Triton 导入、GPU 分配、原生 GPU 编译或性能结果。同名 tuned 在主调用体两次、样例片段一次，arg_types 数量错误拒绝；AOT 不同 index 拒绝，JIT guard 首次/切换与 ABI 分支文本符合源码。

已渲染并检查新图彩色与灰度版本，三职责与共享模块明确、运行阶段单独成组。入口等待位于模块级，stream getter 位于调用体，避免把生成语句去重误写成跨调用缓存 runtime 句柄。新设备注意点涵盖 Python hooks、cuda/xpu 硬编码分类、launcher 协议及 C++ guard 硬编码。未查看 PyTorch docs，未改产品源码。整体目标继续 active。

本轮最终验证通过：538 个本地引用、76 个节点、三子模块限制、既有二十六段源码原文和十六项新增证据。补充核对 wrapper.py:926–955，compile-time autotune 在 _generate 期间 exec 独立片段，正文明确其例外时点；实验不执行该 GPU 片段。

## 最新进度：2026-10-20 21:22

上一目标轮完成 wrapper 入口与设备调用本层篇，并提供十六项生成协议/CPU 执行证据，分类为 progress。本轮按设备接入优先队列完成“图上下文与符号状态”职责叶子，不新增子目录。当前 24 个本层篇、10 个逐行叶子、42 个待写节点，整体目标继续 active；下一篇为“合法性与后端判断”。

本轮增加十一段完整源码原文及逐行说明：graph.py:261–306、308–465、631–660、717–726、1157–1164；virtualized.py:82–135、138–151、154–167、293–361；utils.py:295–304；ir.py:309–314。累计三十七段源码 hash 与原文核对通过。GraphLowering 构造器内所有字段、模式、共享引用、后端能力缓存和 helper 调用均按原顺序解释；未把调用 SizeVarAllocator、布局启发式、外部 serializer 等视为这些 helper 全部覆盖。已登记各自主要归属，ShapeEnv 整体算法仍是 Inductor 输入边界外设施。

graph_context.py 十三项成功：独立新环境、复用 ShapeEnv 与 assert 等待表浅复制、已有 FakeTensor 表达式转换不新增符号、新环境尺寸 duck-shaping/静态非连续 stride、设备 type/index/首节点登记、混合加速设备 wrapper 拒绝、const_module 容器共享、普通子图有限模式/名称/环境继承、嵌套及异常恢复、setter 即时安装、三线程各有当前图、每图能力缓存、完整 CPU IR Lowering。预期新增 deferred assert 冻结失败由真实 ShapeEnv API 触发并捕获，recording 的错误日志不是未处理失败；sympy.true 静态判断仍可通过。

设备元信息案例仅创建 torch.device 描述；cuda:0、cuda:1、xpu:0 来源映射分别记录，indices 集合 [0,1] 不能与 types [cpu,cuda,xpu] 按位置配对。没有 CUDA/XPU 分配、硬件调用或 GPU kernel。CPU 案例两个 placeholder 均登记输入，未使用 y 不引入设备记录，首个 CPU 来源是 x_1，留下一个 ComputedBuffer；只执行真实 Lowering，不生成或执行 Inductor kernel。make_fx/Fake propagation 是实验建图设施，不把 eager 运算当后端编译结果。

图按参考样式绘制，已渲染并检查彩色与灰度。明确调用者先安装 fake_mode、构造后安装 graph，避免把这两个作用域画成同一安装时点；从构造、随节点更新、后续消费到 finally 恢复可连贯阅读。当前验证通过：561 个本地引用、76 个节点、每层最多四个子模块、三十七段完整源码与所有已登记证据；INDEX、父篇、源码主要归属和设备优先队列同步。源码 HEAD 仍为 8332c5622aabd9535969e9a8d0ed60e4a90e4402；未查看 PyTorch docs、未改产品源码。

## 最新进度：2026-10-20 21:29

上一目标轮完成图上下文与符号状态的完整职责叶子、十三项证据及十一段源码原文，分类为 progress。本轮完成“合法性与后端判断”本层篇，新增三个待展开职责：通用约束与读写匹配、设备融合判断、环检测与融合收益。拆分依据为三套独立状态/算法：依赖合法性及可选循环重排，CPU/SIMD/模板的生成兼容性，沿当前组映射的 DFS 和后端收益选择；不按每个谓词或日志条件拆目录。当前 25 个本层篇、10 个逐行叶子、44 个待写节点；本层不是完整源码叶子，既有三十七段完整原文 hash 不增加。

源码核对 Scheduler.can_fuse、can_fuse_vertical、读写/弱依赖 matcher、共享数据重排、峰值距离策略、环 DFS、收益分支与 fuse_nodes_once 外层调用；CPU 的范围/outer-loop 条件，SIMD 的 numel/rnumel/tiling 条件，CUDACombined 委托与 CUDA C++/ROCm 钩子。明确 can_fuse 返回 True 后仍有 cycle/speedup 两门，且实际合组属于 merge-groups。候选过滤和消费时可能重复调用 hook，判断次数不是运行次数。循环重排可能修改候选节点，即使随后其他门拒绝；共享 score 和诊断实现主要归候选模块，未声称本篇完整覆盖这些 helper。

fusion_legality.py 四项完整 CPU 编译执行均成功：x float32/CPU/shape[2,3]/stride[3,1]，横向 (x+1,x*2)，纵向 a=exp(x)、返回(a,a+1)。每种图分别委托原 CppScheduling hook 或注入明确 False 策略，实际组数为 1/2，generated_kernel_count 为 1/2，数值相同、输入保持、输出不别名输入。允许时 hook 调用两次，否决时一次。实验不创建新设备注册，也不把人为否决当作默认后端原有缺陷。完整生成源码分别保留 *_output.py，未报告速度变化或峰值改善。

此外采集十八个实际通用谓词协议、九个真实依赖 helper 对照、五个 SIMD 条件、环检测两种路径，以及基类 hook/CUDA C++ 委托/CPU benchmark 跳过边界。协议节点和共享 score、后端 allow 是显式 fixture；SIMD 只替换 select_tiling 为固定 provider，其余真实谓词处理 group；环使用实际 FusedSchedulerNode 子类的显式元信息调用真实 DFS。没有 GPU/Triton kernel、真实 GPU tiling 或收益 benchmark，也没有虚构时间。CPU benchmark_fusion=True 的受控探针确认提前返回且未调用 benchmark 方法。

重要源码边界已进入正文：MemoryDep 相同非空 mode 在名称/索引比较之前提前 True，StarDep 对应分支还比较重命名后名称；不是任意跨 buffer atomic 融合正确性的证明。split-scan 的一处 WhyNoFuse 后没有立即 False，同组归约 fixture 最终返回 True；日志不等于控制流拒绝。CUDACombined 的横向 CUDA C++ 模板委托在当前版本落到未实现基类方法，直接元信息接口实验得到 NotImplementedError，与旁边“always False”注释不同；未声称完整模板候选链一定到达此路径，未改产品代码。

图已实际渲染、检查彩色与灰度，并修正 can_fuse 边界标题与入口箭头的重叠；共享模块统一灰框，三项职责与目录一致，拒绝/未实现异常/成功主线均明确。源码主要归属、父篇、INDEX、优先队列与验证器同步。当前检查通过：584 个本地引用、79 个节点、每层最多四子模块、既有三十七段原文及已登记证据。PyTorch 工作区仍只含原有两项修改和两个未跟踪 docs 目录，未查看其内容、未改产品源码。整体目标 active，下一篇按设备优先顺序讲元素表达式与目标代码。

## 最新进度：2026-10-20 21:39

上一目标轮完成融合合法性与后端判断本层篇、四个完整 CPU 对照和接口证据，分类为 progress。本轮按设备接入优先队列完成“元素表达式与目标代码”本层篇，直接拆成 ops 解释与类型语义、参数索引与访存、临时值复用与生成区三个职责。强调三者逐操作交错调用，不把它们画成整个 LoopBody 顺次完成的三个阶段。本轮是本层篇，完整逐行仍由子篇承担，三十七段既有源码覆盖 hash 不增加。当前 26 个本层篇、10 个逐行叶子、46 个待写节点；整体目标 active，下一篇为 Lowering 外部调用与 fallback 协议。

源码核对 LoopBody 捕获和回放、InterpreterShim、OpsValue/OpsWrapper、DataTypePropagation、OpOverrides/CppOverrides/TritonOverrides、目标 CSEVariable 注解、KernelArgs 和索引改名、CPU/Triton load/store、CSE.generate/invalidate/clone、Kernel CSEProxy、swap_buffers/ScopedDict，以及 CPU 低精度 legalization、Triton codegen_body 与后端循环拼接接口。共享的索引代数、局部 buffer 删除、循环组织、wrapper/ABI 装配均登记已有主要归属，不据本篇接口说明声称 helper 全部覆盖。

element_codegen.py 实际完成四个 CPU 标量编译执行案例并捕获 LoopBody、生成区、CSE 与参数状态：共享 exp 的两个输出，非连续 x 加广播 b，float32→float16→float32 往返，以及直接输出 float16。使用 cpp.simdlen=1 便于阅读，不是性能配置。建图使用本地 select_decomp_table 准备入口，未把 aten._to_copy 未分解图当合法入口。记录 dtype_at_generate_return 是 CSE.generate 返回时的类型，可能尚未执行 CSEProxy/目标后端后续 update_on_args；不据该阶段 None 推断最终临时值类型缺失。

共享 exp 案例实际有三个 LoopBody，在同一个 CppKernel 中：exp 写 buf0，两个消费者分别读 buf0。真实采集仅一次 std::exp(tmp0) 表达式生成，后续靠 store_cache 转发 tmp1，loads 区不增加 buf0 内存加载；不能把它写成第二次 exp 字符串 cache hit。中途 stores 区有 out_ptr0，最终 kernel/参数与 wrapper 不再包含内部 buf0，仅 out_ptr1/out_ptr2 两个输出。数值、输入保持和输出存储检查通过。普通表达式的重复缓存由独立真实 CSE 协议实验提供证据。

非连续 x shape[2,3]/stride[1,2]，b shape[3]/stride[1]，实际循环 x0 为列、x1 为行，x/输出偏移 x1+2*x0，b 偏移 x0；结果 [[10,22,34],[11,23,35]]，输出 stride[1,2]，数值和输入/别名检查通过。

类型往返案例保留明确差异，未放宽 assert_close 掩盖：输入 float32 的 [[0.1,0.2,0.3],[1.001,2.002,3.003]] 经当前 CPU legalization 后，生成 float 加载、convert<float> 和 float 输出，实际等于原 float32 输入；eager 的中间 half 舍入首值 0.0999755859375、末值 3.00390625，最大绝对差 0.0009062290191650391。该案例验证 observed mismatch，报告字段 eager_match=False；没有称 eager 等价性通过。直接输出 half 则生成 convert<half> 并与 eager 通过。四例均输入保持、输出不别名输入；三例 eager 数值比较通过，一例已记录差异。配置 emulate_precision_casts 的声明及相关元数据源码已查到，但未验证能修复本 CPU 路径，不作该承诺。此观察属于被讲源码已有行为，不修改产品源码。

七组公共协议记录：CSE 缓存/范围收紧/失效/clone、KernelArgs 原地指针与尺寸、workspace 字节累计/CPU 拒绝、removed 参数拒绝、真实 CSEProxy 加显式 TextKernel、Triton dtype 表达式与 FP8 条件、Triton mask 元信息传播。TextKernel 只吐教学 LOAD/ADD/STORE，不登记新设备或执行数值；它证明 removed store 可保留缓存记录而跳过写出，以及失效后 load 导致 must_keep。workspace 16+32 字节、偏移 0/16、共48字节；不混淆元素下标。Triton 只生成字符串/传播元信息，没有硬件 Tensor 分配、Triton 程序编译、masked 访存或性能测试。

主图已渲染并检查彩色与灰度，区分普通运算、load、store 分派及三个职责协作；修正合流箭头重叠。源代码归属、父篇、INDEX、state 设备优先队列与验证器同步，验证器明确列出低精度 eager mismatch，不用结构检查通过冒充全部数值等价。当前本地引用与证据检查通过，节点最多四个直接子模块，源码 HEAD 仍 8332c5622aabd9535969e9a8d0ed60e4a90e4402，产品工作区仅原有差异；未查看 PyTorch doc/docs。

## 最新进度：2026-10-20 21:48

上一目标轮完成元素表达式与目标代码本层篇、四个 CPU 编译对照（含明确舍入差异）和七组生成协议，分类为 progress。本轮按设备接入优先顺序完成外部调用与 fallback 协议本层篇，直接子模块为注册/派发/布局约束、参数物化与输出描述、调用生成与副作用，共三个。它们分别涉及注册与约束策略、结构化参数和输出 IR、schema 与 ABI/运行调用，不按每个 helper 或每一步骤另开目录。当前 27 个本层篇、10 个逐行叶子、48 个待写节点；本层未冒充完整逐行，原有三十七段 hash 范围不增加。整体目标 active，下一篇为模板与复合算子接入。

源码核对 lowering.make_fallback/fallback_handler、needs_realized/layout constraints；GraphLowering.call_function/run_node/propagate_mutation；ExternKernel 参数/布局、process_kernel、Alloc/Out；FallbackKernel 构造、create、find_device、codegen_args、schema mutation/alias、符号绑定、runtime dispatch/export；MultiOutput 与 wrapper 对应调用；外部节点 serializer 及 ATen namespace/auto-functionalizable schema 边界。图中构造与生成两个时点都标③，参数与结果两次标②，不把三个职责误画成互不回访的固定阶段。普通输入物化/布局算法、GraphLowering 派发、mutation 传播、wrapper 启动方法仍登记其主要归属；源码声明 C++/AOT 接口不当作本机已运行该 runtime 的证明。

external_fallback.py 五个 CPU 单图均真实编译、各运行两次，十次对照全部通过。数值、原输入写入、输入/输出 stride、存储别名和调用次数均与 eager 比较。先显式 make_fallback，再以 implicit_fallbacks=False 编译，证明该配置不禁用已登记的 handler。五图编译期各两次 Fake、自定义 CPU 实现零次；每次正式运行一个外部算子调用。次数是该实验事实，不承诺任意编译流程总是恰好两次 Fake。

pack 的 Tensor[]/dim/alpha/reverse 展平为两个已物化 ComputedBuffer 和 [0,3.0,True]，unflatten 保留列表/位置/关键字结构。一个 FallbackKernel 对应三项 MultiOutput 路径 tuple0、tuple1/list0、tuple1/list1；wrapper 一次真实 pack 加三条取项，再执行 relu。返回 [[5,7,10],[8,6,9]]、全2数组和 x+1 数组，输入不变、输出不别名输入。纯布局约束案例原 stride[1,2]、外部算子实际收到连续[3,1]，输出与 eager 对照。

mutable None 返回案例不依赖 mutation_names 非空：alias_names 有输入、mutation_outputs 有 MutationOutput、side_effects=True，mutated_buffers 登记输入。无约束 bump 直接写 x；带连续约束的 bump_copy 先复制 x 至 buf0，调用写 buf0，再由生成 kernel 写回原 x 并计算输出。原 x 值变 [[3,5,7],[4,6,8]] 且 stride[1,2] 保留，输出 [[6,10,14],[8,12,16]]；生成 cpp_fused_0(x,buf0)、调用、cpp_fused_1(buf0,x,out) 证明写回交接。自定义 view 则 alias_names 有输入、无 mutation/side effects，输出确实共享输入存储且保持非连续布局。

五组边界记录：自定义 Tensor(a!)→Tensor(a!) 被当前 FallbackKernel schema 规则拒绝并包装 LoweringException；CPU/Meta 已登记、CUDA 未登记却能产生 CUDA FakeTensor 输出；find_device 在受控元信息输入中优先第一个输入设备，输出 dict 无输入返回 None、无 Tensor 序列 IndexError；普通 callable 登记 RuntimeError；aten.t decomposition 冲突 AssertionError。预期 unsupported mutable 编译失败会保留专用缓存并打印 warning，脚本捕获验证后成功结束，不算未处理失败。CUDA 只有 Fake/注册表与 helper 元信息，没有 GPU 分配或运行，不由 Meta 成功推断设备数值实现。

正文区分 Alloc/Out/Fallback、NoneLayout 与返回 Tensor、别名与写入、元信息 device 选择与实际 dispatcher 实现；保留 process_kernel 重做 Fake stride 推导、创建时特殊非 Fake 例外与输出类型/设备推断限制。未验证 C++ wrapper、AOT proxy executor、数据相关动态输出和所有 schema；待子篇逐行继续，不省掉这些源码范围。图已实际渲染检查彩色与灰度，采用参考的纵向圆角浅色分组，运行触发单独标注。父篇、INDEX、设备优先队列、主要归属和验证器同步；当前 639 个本地引用、85 个节点、三十七段原文及全部登记证据检查通过。源码 HEAD 8332c5622aabd9535969e9a8d0ed60e4a90e4402，工作区仍为原有两项修改/两个 docs 未跟踪目录，未查看 PyTorch doc/docs、未改产品源码。

## 最新进度：2026-10-20 22:00

上一目标轮完成外部调用与 fallback 本层篇、五个完整 CPU 图/十次运行与五组边界，分类为 progress。本轮按设备接入优先队列完成模板与复合算子接入本层篇，拆成矩阵与卷积的候选接入、列表与复合计算、用户 kernel 与专用调用三个职责。它们对应不同的接入协议，不把所有路径排成必须经过 autotune 的阶段；选择器、模板渲染和设备生成维持已有主要归属。当前 28 个本层篇、10 个逐行叶子、50 个待写节点；原有三十七段完整原文 hash 不增加，本篇没有声称完整逐行。整体目标 active，下一篇为代码生成侧的模板候选与方案选择。

源码核对 mm/mm_args、addmm/int_mm、bmm 的形状与空候选分支、conv 的形状/布局/1x1 转 mm/候选构造、utils 模板设备/dtype/库/常量/架构 gate、foreach 构造/物化与列表登记、用户 Triton lowering 和 IR 参数/constexpr/MutationOutput，以及 EffectfulKernel/collective/wait 的接入与依赖接口。模板/ChoiceCaller/输出节点桥接只解释数据交接，完整 selector、打包/渲染、benchmark、编译路径仍归 template-selection；卷积/其他 GEMM 家族、effect 与通信尚无本轮运行证据，已留入后续子篇职责而未删去覆盖范围。

template_integration.py 七个真实 CPU 图全部编译执行，数值、输入保持、输出不与输入存储别名检查通过，并预先计算 eager 参考、核对原捕获 W 数值未变。记录选择器候选/输入layout/输出IR以及 GraphLowering operation、list、constant 状态。案例为 mm+relu 外部实现，常量权重 CPP 模板，mm 空集直接 ATen 回退，bmm 空后端策略，bmm m=1 分解，普通 bmm 外部实现，foreach 普通展开。七图的 AlgorithmSelectorCache.make_benchmark_fn 次数都为零，没有伪造计时、比较或加速结论。

真实 CPP 模板：X[8,16]、W[16,16]、CPU float32/连续；max_autotune_gemm=True、后端仅 CPP、autotune_fallback_to_aten=False、cpp.threads=1。候选为 DataProcessorChoiceCallerWrapper 包 CppTemplateCaller，输出 CppTemplateBuffer，实际生成并执行 C++ GEMM；当前 micro GEMM 将权重打包成 [1,16,48]/stride[768,48,1]，逻辑输出仍[8,16]，原 W 不变。只有一个非 CUDA 模板候选直接 output_node，开启 max_autotune 不构成实际测时证据。CPU return_multi_template 请求可能为 True，但 selector 对 CPU 明确关闭该路径，不能从请求字段推断返回 MultiTemplateBuffer。

空候选策略按实际算子区分：CPU只配置TRITON时，mm 禁用回退会以 NoValidChoicesError 经 LoweringException 拒绝；启用回退则 lowering 在 selector 之前直接返回外部节点，selector 记录为空。bmm 即使 autotune_fallback_to_aten=False，当前实现也自行补 ATen 再进 selector，真实执行通过。bmm m=1 的 CPU 图则不进选择器，广播乘法+归约形成 ComputedBuffer；一般形状保留 ExternKernelOut。没有把各家族视为同一 fallback 开关。

foreach 普通 CPU 图得到两个 ComputedBuffer，lists为空，数值通过。单独 Lowering-only 实验将 FOREACH 能力查询固定为True，实际登记 list_op0_op1 后停止，未交 CPU Scheduler/codegen，未称CPU新增原生 foreach 支持。Triton 模板 gate 用实际函数，只替换硬件大小与 BackendFeature provider 为True；cuda为True，cpu/xpu/privateuseone为False、int64为False，没有GPU运行时查询或分配，说明已有类型 gate 仍独立存在。

user_kernel_protocol.py 调用真实 triton_kernel_wrap_ 和 UserDefinedTritonKernel 构造/codegen，但 kernel 描述、mutation分析与 wrapper 是明确fixture，metadata comment跳过，输入为CPU IR。常量表 scale=2 覆盖kwargs scale=1；原签名 inp,out,BLOCK,scale、constexpr[2,3]，BLOCK由配置提供未进入raw_args，实际参数inp,out,scale重算constexpr为[2]。lowering交回原inp/out两个TensorBox，MutationOutput只对应out，图登记y_1被写；无Tensor参数触发AssertionError。没有验证真实Triton mutation分析、编译、launch或数值，未把替身协议当新设备完成。

主图已渲染检查彩色和灰度，三分支、bmm转普通分解、共享selector与后续设备生成清楚分离，使用参考的圆角浅色纵向结构。父篇、INDEX、source ownership和设备优先队列同步；验证通过：664个本地引用、88个节点、每层最多四子模块、既有三十七段原文与已登记证据。产品工作区仍仅原有两项修改和两个未跟踪docs目录，未读取PyTorch doc/docs，未改产品源码。

## 最新进度：2026-10-20 22:10

上一目标轮完成模板与复合算子接入本层篇、七个完整 CPU 图（包含真实 CPP 模板）及协议证据，分类为 progress。本轮完成代码生成侧“模板候选与方案选择”本层篇，直接子模块为候选对象与模板生成、预编译/校验/计时、计时缓存与结果交付三个。三者在 selector 中交错调用，图未把缓存、Future、延迟对象冒充固定顺次只执行一次的步骤。当前 29 个本层篇、10 个逐行叶子、52 个待写节点，既有三十七段完整源码 hash 未增加；本轮为本层流程交付，完整模板/selector/cache/设备实现继续待子篇。整体目标 active，下一篇按设备优先队列讲编译任务与等待。

源码核对 ChoiceCaller、ExternKernelChoice/Caller、Triton/Cpp/CUDA Caller/BenchmarkRequest、DataProcessor 包装、KernelTemplate.maybe_append_choice、Triton/Cpp generate 与 output_node、AlgorithmSelectorCache 全调用控制、PersistentCache 查询与系统/输入/精度键、make_benchmark_fn 的真实输入与 offset、错误/VERIFY 条件、CPU/nonCPU benchmark 分派，以及 MultiTemplateBuffer 首次求值/调度 finalize 消费接口。通用编译任务与动态库/二进制加载仍保留兄弟节点归属；模板候选前置算子条件归专用 lowering，不复制实现。非 CPU benchmark 默认走GPU实现和 CacheBase 系统信息的既有设备假设明确指出，没有凭注册类推断新设备可测时或键已完整区分新硬件。

algorithm_selection_cpu.py 在每次复现时新建私有目录，冷/暖两个独立子进程共享该计时缓存，关闭 FX 图缓存、开启真实 ATEN/CPP 双候选，max_autotune_gemm=True、cpp.threads=1、torchCPU线程=1。X[16,64]、常量W[64,64]、CPU float32连续；候选与hash都由真实lowering生成，selector及其benchmark未用人工时间替代。显式开启模块VERIFY(atol/rtol=1e-3)后，冷路径make_benchmark_fn创建一次、CPP precompile一次，实际比较得到mm=0.008941045962274075ms、cpp_packed_gemm_0=0.0014139804989099503ms，输出CppTemplateBuffer。只把该数据作为本机单次候选样例，不报告固定加速比或其他输入/整图性能收益。

暖子进程把输入X值整体加1.5，候选hash与计时文件相同，precompile为空、benchmark函数创建零次，返回完全相同的已缓存两项计时，仍选择CPP。实际模型输出冷首行前四项约[1.3022461,1.3027267,1.3032074,1.3036880]，暖约[48.5522461,48.5761642,48.6000824,48.6240044]，分别与预先计算eager对照通过；输入与原捕获W保持，输出不别名输入。暖命中跳过候选VERIFY，但正式模型数值比较仍执行。该缓存复用的是算法计时，不是模型答案。cache_path与候选hash及两组真实数值都保留real-cpu.json、两份完整生成代码和子进程stdout/stderr。

algorithm_selection_protocol.py 十项真实selector控制流协议全部通过：None过滤单候选快路、空集拒绝、空表返回首项、仅次项表仍返回首项、完整表取最小、全inf拒绝、单CUDATemplateCaller例外、真实MultiTemplateBuffer首次属性求值缓存、CPU禁延迟、非None input_gen_fns禁延迟。Caller/output/lookup为明确fixture，有限数值1/2只作人工排序槽位，不以毫秒表述，不与真实CPU数据混用；benchmark函数设置为禁止调用。CUDA类型子类只验证isinstance路径，没有真实workspace分配或kernel编译。延迟例子返回前lookup零次，两次choice_timings读取后仅一次，get_min_choice不再查；预编译超时设0，不能推断一般延迟对象返回前也不会提交任务。

正文与图明确：预编译可在延迟对象返回前提交，真正消费计时时先调用等待函数，再查计时；缓存命中仅绕过实际测试，不跳过源码规定的等待顺序。默认VERIFY为空，不能以benchmark存在推断总做正确性比较；普通单候选不测时，但CUDATemplateCaller有workspace例外；空/部分计时表与全非有限失败各有实际分支。GPU/Triton/ROCm模板、GPU子进程benchmark、epilogue联合选择与实际新设备数值执行未验证，保留源码/后续子篇边界。

新图已渲染并检查彩色与灰度，三个职责与目录一致，快路/拒绝/右侧缓存绕行和延迟消费时点可读。父篇、INDEX、设备优先队列、主要归属和验证器同步；检查通过：687个本地引用、91个节点、每层最多四直接子模块、既有三十七段源码与已登记证据。产品工作区仍为原有两项修改和两个未跟踪docs目录，未读取PyTorch doc/docs、未改产品源码。


## 2026-10-20 22:17：编译任务与等待本层完成

设备接入优先队列继续交付 tasks-wait。拆为三个待展开职责：backend-tasks、pools-workers、future-scope；本篇讲清源码/参数到立即对象或Future，再到scope替换的交接，不将约三百行入口和额外缓存/进程实现冒充叶子完整覆盖。节点94，本层完成30，逐行叶子10，待展开54；下一篇 source-binary-cache，设备优先及原范围均保留。

直接核对 async_compile.py 全文、runtime/compile_tasks.py、compile_worker 的 subproc_pool/__main__/watchdog，以及 codecache.py CppCodeCache.load_async、load_pybinding_async、三个Future包装的原始实现。共享缓存/装载不重复归属于任务树；进程序列化、父进程监视、任务入口和解析已分配唯一主要职责。记录重要边界：submit线程池不等于process_pool；就绪只查done；线程数1的wait不解析手工Future；只查顶层；异常部分更新；LambdaFuture本身不缓存；TritonFuture在主进程precompile之前清future，因此初始化异常不能靠同对象result重试；device_str本版本未使用；新缓存环境只显式转送两个变量。

compile_tasks.py实际完成两个CPU编译：线程数1直接绑定、提交次数0；线程数2新源码键和私有cache确实提交一个C++编译任务，返回LambdaFuture，经真实wait成为绑定。两条原地kernel都将CPUfloat32[5]改成[12]。独立submit身份对照确认同步同pid/thread、异步同pid不同thread。不测速度。真实subprocess管理进程和fork worker执行普通可pickle任务，pid区别主进程，scope由Future替换为dict；任务ValueError经SubprocException回传，随后正常shutdown。没有杀进程或触碰设备。

六项明确协议对照：顶层解析与嵌套保留、ValueError部分更新、BrokenProcessPool诊断包装、线程数1保留注入Future、Triton未就绪立即precompile、Triton就绪worker warm_cache_only再主进程precompile。后两项仅kernel/cache/pool描述fixture，运行真实dispatcher/worker helper/Future；没有Triton包和GPU执行。worker helper在本进程受控调用且恢复环境，不能当作真实GPU子进程编译。普通实际进程任务独立登记，不混充设备证据。

图 compile-tasks.svg/png 与彩色灰度review已渲染检查，①②③与三个职责一致，同步右侧绕行与提交后继续的虚线可读。正文逐步使用具体scope与[5]→[12]交接，失败项身份和后项未访问明确。验证器核对709个引用、模块树/优先队列、37段既有完整原始源码、两条真实CPU编译与进程/协议结果。GPU/ROCm/Halide编译、实际worker崩溃恢复未运行，子篇仍保留。没有查看PyTorch doc/docs，没有修改产品源码。


## 2026-10-20 22:22：源码和二进制缓存本层完成

设备接入优先队列完成 source-binary-cache。本层拆为三个职责：keys-files、conditions-locks、backend-artifacts，保持每层少于五个直接子模块。97节点、31本层完成、10逐行叶子、56待展开；下一篇jit-aot-loading。原始逐行覆盖仍37段，未将本层读源码和运行验证计作叶子完成。

核对codecache.py通用hash/path/write/atomic、CppCodeCache/CppPythonBindingsCodeCache闭包和装载、CUDA/ROCm write/compile/load、DLLWrapper、TritonCodeCache/PyCodeCache调用边界、Halide配置hash/生成目录/glue/runtime，以及cpp_builder选项/命令/build、runtime_utils目录、ROCm命令和cpu_vec_isa检测构建。Python模块attrs/行映射/重载与JIT/AOT出口保留相邻篇主归属；Future和池实现链接任务篇，不重复归属。

source_binary_cache.py五个独立原生CPU子进程：同源码C ABI整数函数kernel(5)，默认增量7得12；冷进程目标build1，暖进程目标build0，extra_flags=-DLEARNING_INCREMENT=9产生新键和新目标build1并真实得14。各进程重复load_async都得到同一个getter、getter返回同一lib，身份断言通过。另一全新目录两进程共同起点并发请求，目标build次数0+1，两者返回12、key一致。cold/race存在ISA能力探测的额外原生build；报告同时保留all_build_calls_including_isa_probes和全部命令，目标次数按最终binary_path匹配，避免混淆探测与目标构建。未报告加速倍数或整图Tensor结果。

六项真实源键接口对照：strip算键但第一份原文保留、extension只变路径、普通extra变键、指定目录只变路径、spv分支忽略extra；每项结果明确登记。CUDA/ROCm分别执行真实compile控制流，但命令生成和subprocess为fixture，写NOT A BINARY标记文件且从不装载；同source/ext的不同extra_args没有进入write键，两次内存复用/清cache后磁盘复用都只出现一次假编译调用，改目标ext则键改变。此源码边界没有擅自改产品实现，也没有把fixture误写为nvcc/hipcc数值结果。

正文覆盖源码/条件→key/path→闭包→锁/二进制→当前进程lib→真实函数值，区分内存对象身份与跨进程文件复用；命中已建立但未消费的闭包不保证无需构建。图按①键路径②构建③后端适配三个职责展示，右侧绕行仅标已装载/完成记录，彩色与灰度已渲染复查。已同步父篇子链接、INDEX、state队列、主要归属与验证器，检查731引用、所有树约束和已有源码/证据通过。源码HEAD仍8332c562，产品工作区保持既有两修改/两个未跟踪docs目录，未查看PyTorch doc/docs，未改产品源码。GPU/Halide实际构建装载、非本机文件系统锁和跨机器兼容性未运行，待后续职责保持覆盖。


## 2026-10-20 22:31：JIT 与 AOT 装载本层完成

设备优先队列完成jit-aot-loading，三子职责jit-module/aot-artifacts/aot-container仍待完整逐行展开。本层32篇、逐行叶子10、100节点、58待展开；下一篇cache-runtime/call-adaptation，仍优先设备相关子树，原范围保留。37段既有原始源码覆盖未因本层实验增加。

核对GraphLowering compile_to_module/_compile_to_module/compile_to_fn，PyCodeCache模块/attrs/linemap/stack_frames，runtime/compile_tasks的Python重载，AotCodeCompiler常量字节/平台对象/链接/包/mmap分支，package装载，aot_load入口；C++ runner/CPU/CUDA/pybind和aoti_runtime interface/model/device_utils及生成容器接口。共享CppBuilder条件与锁归缓存条件，Triton专用重载和Future归任务，生成wrapper主体不在运行容器复制归属。上游export仅用于实验准备和解释调用结构边界，未展开捕获全流程。

jit_aot_loading.py真实CPU图relu(x+w)，x[2,3]float32连续、w[3]，always_keep_tensor_constants=True/cpp.simdlen=1；记录load_request带_tensor_constant0和一条linemap、模块exec后该名仍None、load_return后真实attrs身份安装且call可用。实际生成C++入口得到[[10,21,32],[13,24,35]]，原x/w保持、输出不别名输入、boxed参数列表清空。完整jit_output.py保留，避免把常量名字已存在误记为Tensor已安装。

五组局部协议：同key重复attrs不会覆盖首次常量/行映射、实际frame查回learning_model.py:7；无attrs/linemap回调清本类cache后在同进程重载为新模块；语法错误RuntimeError与exec ValueError区分；aot_load Runner fixture分别cpu/cuda:1得5、privateuseone:0拒绝。fixture不启动CUDA，纯Python模块不冒充生成kernel，重载回调测试未冒充跨进程。

真实CPU AOT使用exported module准备输入边界，然后实际Inductor AOT，abi_compatible=True、always_keep_tensor_constants=True、cpp.simdlen=1/packageFalse，计算(x+w,)。实际构建model.so、保留weight常量、原生AOTIModelContainerRunnerCpu装载与run通过；常量映射为weight→weight。aot_output.cpp是实际传给AotCodeCompiler的完整生成源码快照，aot.json保存路径、结果、原生runner call_spec。另一个新进程aot_loading_runtime.py只装载已有so，通过真实aot_load两次调用，输入加1.5输出相应变化、tuple结构正确、输入保持、输出不别名，结果aot-runtime.json。这不是仅路径存在，也不是只有fixture通过。

构建研究中确认默认非abi模式在本机DLC扩展ATen头依赖dlc_runtime_api.h搜索路径而未完成，交付验证明确限定abi_compatible=True，不将失败模式声称验证通过；未改产品头/配置/库，不触碰设备。最终AOT实验证据不包含包、大权重mmap、常量更新/折叠、proxy执行或GPU事件。C++parse_device_str和Pythonaot_load只认cpu/cuda已据源码解释，新设备还需runtimestream/完成/常量/句柄协议。

图jit-aot-loading的三个职责与目录一致，JITcall/AOTpath和后续装载时点明确；彩色/灰度实渲染，分支标签与后续虚线走独立通道。同步父篇、INDEX、状态/主要归属/验证器，756引用、每层最多4子模块、既有源码/本轮原生JIT/AOT和协议证据检查通过。没有阅读PyTorch doc/docs，没有修改产品源码。整体目标仍未完成。


## 2026-10-20 22:38：调用适配与 CUDA Graph 本层完成

完成最后一个原设备优先根call-adaptation，本层拆policy-state/alignment-call/capture-trees三职责。103节点、33本层完成、10逐行叶子、60待展开，37段完整原始源码覆盖保持；下一篇设备后端注册registration，继续设备优先子树而不是遗忘其余范围。本篇不将三千余行Trees运行实现视作叶子已覆盖。

阅读codecache.py cudagraph_post_compile/maybe_realign_inputs/post_compile/CompiledFxGraph，compile_fx.py条件元信息/输入检查索引/延迟cudagraphify/legacy捕获，utils.py对齐假设/clone/替换/过滤/TracingContextstride，cudagraph_utils设备和mutation条件、Trees延迟整型键、manager._run状态和invariant/重录/普通执行分支。源代码条件区分门槛拒绝与延迟捕获异常；BoxedBool共享状态、cache后适配不持久化、AOT字符串旁路、backward禁捕获但仍通知manager，均有明确接口说明。

call_adaptation.py两个真实CPU图，x[2,3]float32/stride[3,1]，编译和运行的视图storage_offset=1，实际地址mod16=4。请求CUDA Graph，冷编译CPU设备原因使BoxedBoolFalse；post_compile仍一次、inputs_to_check空、current_callable同对象。pure x*2+1得[[3,5,7],[9,11,13]]、原x保持且无输出别名；mutation add_(3)原x变[[4,5,6],[7,8,9]]、输出与x共享存储。boxed列表两次都清空，不与Tensor修改混淆。完整生成源码两份保留，cpp.simdlen=1仅为可读性，不测性能。

真实CPU helper显式指定idx0，以[4,3]/stride[1,4]/mod16=4输入验证替换列表项、保持stride/数值、调用者原Tensor保持；已对齐项与空检查callable身份保持、static候选过滤[0,1,2]→[0]。另显式helper内add_(5)仅修改替换Tensor、原Tensor不变，证实无通用写回，不把强制CPU helper当真实编译器选择或GPU正确性证据。FakeTensorCUDA描述只验证编译期索引，默认nonstatic[0]/static[]、assume_aligned[0,1]；is_gpu实际仅cuda/xpu。没有GPU分配或捕获。

六组policy section：真实TracingContext回填[(3,1),None]；已禁用、fail_reasons、允许分支分别不调/不调/调用一次factory；backward manager fixture通知先于原model；早期mutation条件trees静态豁免、动态拒绝、legacy静态仍拒绝。四组deferred section：legacy/Trees外层构造不调factory、两调用仅初始化一次；首次factory异常ValueError传播、下次才重试；Trees真实延迟整数键7/7/8只对7/8调用替代manager，输出8/8/9。13个section含3alignment+6policy+4deferred，不冒充13个实卡案例；真实编译GPU从未运行。

新图三个职责对应树，展示安装与后续真实调用两时点，拒绝跳过③安装、普通入口绕过③运行，彩色灰度渲染复查。正文与父篇/INDEX/状态/主要归属/验证器同步，779引用、每层最多4直接子模块、既有精确源码与本轮数值/修改/alias/身份/异常断言通过。产品源码不变，不查看PyTorchdoc/docs；GPU重放/allocatorcheckpoint/真实设备事件、输出存储跨步生命期仍待后续覆盖。

## 2026-10-20 22:48：后端注册与能力查询叶子逐行完成

完成设备优先registration叶子，不再继续拆目录。完整原始common.py五范围87–91、96–113、116–246、256–268，合计219行含原注释/空行，逐段按原顺序展示；每一语义行、十个枚举成员、初始化导入/工厂/分支、例外捕获和隐式返回均解释，并给出具体状态交接。KernelArgType、index_prevent_reordering明确不属注册，具体后端/设备操作实现与边界外PrivateUse1设施保持主要归属，不虚报整个common.py已覆盖。源码hash与正文原始代码块逐字核对新增5段，总42段；叶子11，本层33，节点103，待展开59，下一篇dispatch。

复用已有backend_registration.py十worker证据和其中完整CPU编译/运行消费，不重复跑无变化的整图测试。新backend_registry_leaf.py六独立进程实际执行：裸getter不初始化；同一默认CPU factory在cpp/halide配置下分别构造真实类、能力键集合不同、非法配置调用时KeyError；三个能力查询各新构造None探测、返回原能力字典；torch.device(cpu:0)按cpu查表，字符串cpu:0按精确key查表缺失TypeError，显式添加cpu:0后两者能选不同实现；错误Enum参数AssertionError、合法meta缺Scheduling TypeError、非法torch.device RuntimeError。

默认设备操作模块真实导入一次填cuda/xpu表，清对象表后模块缓存不重跑顶层登记，lookup仍None；显式再次登记按对象身份覆盖，基类四方法均NotImplementedError。新PrivateUse1异常进程仅替换_get_custom_mod_func为ValueError fixture，真实init传播、不缓存返回、先写的默认项保留，恢复helper重试成功。另一进程真实helper采集CppWrapperCodeGen=False正常返回后truth拒绝，None在helper抛RuntimeError并被init吞掉；父篇对应None行修正为实际失败位置。原样槽位/None Scheduling默认替换/省略NoneType与显式None分别验证。六进程没有新设备Tensor分配、kernel或runtime执行，没有把真实类能力查询当Halide编译运行。

图保持两张表和垂直主线，补wrapper类直接交出的支路，避免误画所有wrapper查询都进入能力探测；彩色灰度已实渲染复查。同步父篇、状态、INDEX、frontier与验证器。引用检查跳过fenced/inline代码，避免原始下标后调用 `mapping[key](*args)` 被误判Markdown链接；完整代码仍由独立hash与正文逐字检查，未改动源片段以绕过校验。796个实际引用、每层最多4子模块、42段原始源码和所有登记实验证据检查通过。没有阅读PyTorchdoc/docs、没有修改产品源码；新设备硬件支持不由登记成功推断。

## 2026-10-20 23:01：调度实例与后端委托叶子逐行完成；扩大设备优先范围

完成dispatch叶子，完整原始源码五范围：utils.py decode_device222–230/is_gpu1799–1801，scheduler.py create_backend/get_backend3333–3361/BaseScheduling3592–3696，cuda_combined_scheduling.py全部1–100，共246行。原始实现和原注释按顺序保留，各语义行/接口/委托/默认值逐项解释；具体合并、循环和模板算法仍归原主要职责。新增5段精确覆盖，总47段；叶子12、本层33、节点103、待展开58。下一篇loops。

backend_dispatch.py新真实CPU图relu(x+1)*2，x[2,3]float32连续[-2,-1,0;1,2,3]；能力探测None一次，本图create一次/后端实例一个、之后两次实际get_backend hit同对象。真实CPU编译和执行得[0,0,2;4,6,8]、输入保持、输出不别名。登记派生CppScheduling只加事件，实际生成算法未替代，源码完整保留cpu_output.py。

设备元数据协议用真实Scheduler方法，receiver/V.graph设备记录、has_triton与CUDA属性为明确fixture：同完整device缓存、不同index区分、登记变更不更新已有实例但新receiver更新；cuda无index断言在设备记录前；缺meta实现、旧CUDAmajor6、CUDAmajor8无Triton、XPU无Triton均不写实例缓存；CPU/PrivateUse1未触发GPU条件并不证明设备可运行。构造ValueError两次不缓存，设备记录已两次保留。decode_device实际helper使用Worker编号2fixture，cuda/xpu/privateuseone补2、cpu/meta/明确index不查询接口，None本机真实默认CPU标量。

BaseScheduling九未实现方法逐项NotImplementedError，默认features空/readyFalse/priority0，BaseScheduling(None)实际TypeError；foreach/普通fuse只用合并fixture核对转交，不虚报真实合并算法。真实CUDACombinedScheduling配三个记录delegate，验证所有模板分支/双标记优先级/非空epilogue拒绝/九方法固定Triton/纵横融合分别规则/继承默认值。benchmark返回标签而非人工毫秒。真实CUDA C++水平谓词局部路径仍NotImplementedError，不以alwaysFalse注释代替实现；本地choose_node_backend仅找到定义，没有虚构统一调用链。没有GPU编译/测速/运行。

新图按设备cache hit绕开创建，构造成功才写缓存，彩色和灰度渲染已查看。父篇、INDEX、state/coverage/frontier/验证器同步，810引用与47段原始源码逐字/hash核对及已采集实验证据通过。将组合类转交方法主要归属统一到dispatch，具体设备融合规则仍归合法性子模块，避免后续重复覆盖。

用户22:56的新补充将图重写中的模式注册、匹配、子图替换加入设备接入优先范围，理由是新设备可提供自定义融合算子。state.priority_focus新增graph-preparation/rewrites/patterns，放在后端与循环根之后，三个待展开子职责均进入优先队列；原范围与其余广度顺序保持。用户并要求当前模块写完后提交/work/learning/gpt6-1sol/torch到Tongque2026/Docs；已核实远端为空，本地原先无Git仓库，将纳入正文、图、源码快照/脚本/实验证据，排除编译缓存和二进制，不创建Issue或PR。


用户23:08进一步明确现有表达/布局、物化/副作用、依赖/合法顺序和生命周期也应纳入设备接入优先，尤其自定义融合算子的布局/alias/mutation与异步buffer释放。本轮只把既有lowering/expressions、lowering/materialization、scheduling/dependencies、scheduling/lifetime加入priority_focus，不创建任何新模块或目录；相同深度按用户优先根顺序，其余原范围继续保留。实际kernel/schema/Fake-Meta/布局约束需一致，异步发射不等于完成，单靠FX依赖或last-use不构成释放安全证明，后续案例按相应所有权与stream/事件协议核对。
