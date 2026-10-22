# Q44 作者记录：缓存、编译、加载与运行入口

> 第五轮状态：本层完成。第五轮补齐 local/remote 候选顺序、共同保存和失败降级边界后，重新验收全篇。

## 本篇位置与教学目标

- 全系列编号：Q44；层次 2；父项 Q40；前置 Q43；完成后转深度 3 的 Q111。
- 读者已知：Codegen 已产生 wrapper source；不知道 source 怎样变成 callable，也容易把 Dynamo、FX Graph Cache、PyCodeCache 和 kernel cache 混成一个“torch.compile 缓存”。
- 入口：`GraphLowering.codegen()` 的 source/linemap、图/输入/配置/guard 信息。
- 出口：`CompiledFxGraph`，其中 `current_callable` 接收 boxed input list；运行 wrapper 并返回 tuple。AOT 文件部署不在本篇展开。
- 学完标准：能走完首次 source→文件→module.call、跨进程 FX miss/save→hit/load、HigherOrderOperator bypass 三条路径；能指出各缓存 key/guard、省掉的工作和仍需执行的工作。
- 依据：当前提交 `8332c5622aabd9535969e9a8d0ed60e4a90e4402`、当前源码/测试与独立双进程 CPU 实验。

## 本层职责与边界

1. source artifact：PyCodeCache 按去除首尾空白后的 source 内容哈希定位 `.py`，原子写并加载模块。
2. compiled graph artifact：`CompiledFxGraph` 保存 wrapper key/source、图合同、metrics、guards 等；boxed callable 只在内存对象中存在。
3. FX Graph Cache：按图详情 key 找候选，再按 backed SymInt guards 选版本；miss 编译并保存磁盘副本，hit 恢复 PyCodeCache module.call。
4. 运行入口：`CompiledFxGraph.__call__` 把 list 交给 current callable；wrapper 完成 Q43 的合同。

本篇只解释各层交接，不逐句展开 hash pickler、远端传输实现、C++ 编译器实现或 AOT 打包。

## 处理情况覆盖

| 情况 | 状态变化与出口 | 证据 | 深度/状态 |
| --- | --- | --- | --- |
| 首次 JIT 编译 | source→hash/path→module→module.call | GraphLowering/PyCodeCache 源码、前篇产物 | 机制已讲解 |
| 空计算图 | 直接 boxed `gm.forward`，无需设备 kernel | `_compile_fx_inner` 源码 | 边界已讲解 |
| FX cache miss/save | lookup None→完整编译→磁盘副本 current_callable=None | 源码+独立 run1 miss=1 | 机制已讲解 |
| 跨进程 hit/load | 反序列化候选→guard→恢复 wrapper `.call` | 源码+run2 hit=1 | 完整案例已复读 |
| local/remote 组合 | 先遍历本地候选，再取远端候选；miss 后可同时保存 | `_lookup_graph/_save_graph` 当前源码；无远端实测 | 机制与证据边界已讲 |
| guard 候选 | 同 key 下多个磁盘叶子，逐个以 hints 验证 | 当前源码/测试，无独立动态例 | 条件已讲 |
| bypass | HOO 等不缓存，但仍完整编译运行 | `torch.cond` 实际 bypass=1 | 对照案例已复读 |
| PyCodeCache 内存命中 | key→ModuleType 字典 | 当前源码 | 机制已介绍 |
| wrapper 文件缺失恢复 | cache hit 用保存 source 重写 artifact path | `_lookup_graph` 源码 | 边界已讲 |
| 子 kernel cache | 模块顶层登记/等待 C++/Triton 编译 | source 产物与源码 | 只讲职责边界 |
| AOT mode | 返回编译文件路径，不走普通 CompiledFxGraph callable | GraphLowering/compile_fx 源码 | 只介绍差异 |

## 篇内知识点队列

| 编号 | 完成条件 | 正文/案例 | 状态 |
| --- | --- | --- | --- |
| K01 | 区分四层缓存及各自省掉的工作 | 开篇与职责表 | 完成 |
| K02 | 从 wrapper source 走到 PyCodeCache module.call | 首次路径 | 完成 |
| K03 | 解释 CompiledFxGraph 字段与 boxed list 合同 | callable 小节 | 完成 |
| K04 | 沿 run1 走完 miss、compile、disk copy save | 双进程实验 | 完成 |
| K05 | 沿 run2 走完候选、guard、artifact/callable 恢复 | 双进程实验 | 完成 |
| K06 | 解释 cond bypass 不是 miss/失败 | bypass 对照 | 完成 |
| K07 | 区分动态候选 guards 与 Dynamo guards | 动态形状 | 完成 |
| K08 | 把全教程收束到最终调用 | 最后一跳 | 完成 |

## 独立实验与事实

- 脚本：`../../experiments/15_cache_load.py`；输出：`../../experiments/artifacts/cache_load.txt`。
- 使用独占 `TemporaryDirectory` 作为 `TORCHINDUCTOR_CACHE_DIR`，启用本地 FX graph cache，两个独立进程各编译并运行一次同一 `relu(x+1)*2`。
- run1：miss=1、hit=0、bypass=0；run2：miss=0、hit=1、bypass=0；结果和均为 272，并在 worker 内与 eager 核对。
- 第三个独立进程编译 `torch.cond`，实际 miss=0、hit=0、bypass=1，结果 `[3,5]`；当前 `_check_can_cache` 对 HigherOrderOperator 主动 bypass。
- 临时目录退出时只清理实验自身 cache，不触碰用户现有 cache。

两个进程用于排除“同一 compiled callable 直接复用”的混淆；计数器明确证明第二个进程进入 FX Graph Cache hit。实验没有单独构造动态 guard 多候选、远端 cache 或 artifact 文件丢失恢复。

## 当前源码依据与纠错

- `graph.py`：`codegen`、`_compile_to_module`、`compile_to_fn`。
- `compile_fx.py`：`_compile_fx_inner` 的 cache 门槛、boxed 标记与空图路径，`fx_codegen_and_compile`。
- `codecache.py`：通用 `write`、`PyCodeCache`、`FxGraphCache` lookup/save/load/bypass、`CompiledFxGraph`。
- `test/inductor/test_codecache.py`：miss/hit、动态 shapes、bypass 当前覆盖。

纠正两项措辞：

1. miss 保存时不是清空正在返回的原对象；`_save_graph` 先 `copy(compiled_graph)`，只把磁盘副本的 `current_callable` 设为 None，当前进程的原对象仍可运行。
2. cache hit 不是直接反序列化 callable；它用保存的 `cache_key/source/linemap/constants` 恢复 PyCodeCache 模块，再取得 `.call` 写回 `current_callable`。

## 作者复读：事实与理解分验

事实：miss、跨进程 hit、HOO bypass 都由当前实验计数器与数值结果直接支持；首次 source/module 及 guard/callable 恢复由当前源码支持。没有把同一 `torch.compile` 对象的第二次调用当作 FX cache 证据。

只依赖正文回答：

1. wrapper source 相同为何路径相同？`write` 对 strip 后内容哈希，得到同 key/path；文件存在则不重写。
2. FX hit 为什么仍需 PyCodeCache？磁盘 CompiledFxGraph 不保存 callable，必须加载 wrapper module 并取得 `.call`。
3. run2 为什么能证明磁盘 FX hit？它是新进程，没有 run1 的 callable/ModuleType 内存字典，且计数器 hit=1、miss=0。
4. cond bypass 是否编译失败？否；它跳过 cache lookup/save，随后调用 compile function，数值正常。
5. 同 key 动态候选怎样选择？逐个用当前 backed SymInt hints 求值 `guards_expr`；命中后再用真实 symbols 安装 guards。
6. dtype/stride/config 改变能否盲目复用？不能；会影响 graph hash 详情、输入信息、guard 或触发 bypass。
7. 本地候选 guard 拒绝是否立刻编译？不一定；若远端 cache 可用，还会取远端候选并验证。只有全部候选不命中才完整编译。
8. 远端 hit 是否能直接执行序列化 callable？不能；保存副本的 `current_callable=None`，仍要经 PyCodeCache 恢复 wrapper 模块的 `.call`。

作者复读通过；实际读者反馈尚无。

## 续写位置

## 第五轮全量复核

- **范围**：首次 source artifact、PyCodeCache 内存/磁盘、CompiledFxGraph、FX local/remote lookup/save、shape guards、miss/hit/bypass、空图、kernel 子缓存边界与 boxed runtime 都有明确位置。
- **过程**：首次请求从 source 写文件、执行模块顶层、等待 kernel 到 `module.call`；跨进程 run1 走 miss/save、run2 走候选/guard/artifact/callable 恢复；`torch.cond` 单独走 bypass 后完整编译；local/remote 分支走到共同的 guard 和 callable 恢复出口。
- **事实证据**：双进程本地实验实际得到 run1 miss=1、run2 hit=1；HOO 实验得到 bypass=1 且数值正确；远端顺序和失败降级只由当前源码支持，不声称服务实测。
- **理解迁移**：正文足以推出：新进程 hit 仍需 PyCodeCache；本地候选拒绝后可继续远端；两处都拒绝才编译；保存副本清空 callable 不影响当前进程原对象；dtype/stride/config 改变不能只凭相同节点文本复用。
- **层次**：本篇完成缓存与运行入口的本层协作；Q441 才逐句展开 FX Graph Cache 查找实现。

Q44 第五轮回修完成，深度 2 全部结束。根队列转深度 3 的首项 Q111；之后按 BFS 逐篇复核叶子的完整源码、代码内教学注释和具体状态推演。

## 第一次隔离冷读整改

- 报告：`../../inductor-fresh-reader-review/49_CACHE_LOAD_COLD_REVIEW_1.md`，评级 B。
- 读者能复述 miss、跨进程 hit、callable 恢复、bypass 和动态候选，但开篇对象定义、不同 key 的边界和 guard 术语顺序仍有断点。
- 本轮重排：先写本篇输入、三项动作与 `CompiledFxGraph` 出口；用表格分开 graph、wrapper、kernel 三类 key/artifact；在首次出现前定义 boxed call；把符号、hint、guard、ShapeEnv 的最小解释移到候选流程之前；把跨进程案例对齐三类 artifact；统一 PyCodeCache 一类职责、两种存放位置的口径；删除总结中的悬空 `LoopBody`。
- 下一步：用全新隔离读者只读整改后的 Q44，继续检查正文自身能否闭合。

## 第二次隔离冷读整改

- 评级仍为 B；首次、miss/save、跨进程 hit/load、bypass 已闭合，动态 guard 缺少编译时产生条件的动作。
- 本轮固定使用 graph key 与 wrapper source key 两个名称；补出 `fx_codegen_and_compile()` 构造交接对象、kernel 子缓存 hit/miss、guard 由代码生成依赖产生；把 32/30 例子前移到首次 guard 流程；删除空图岔路、悬空字段名和结尾新增的四组件解释；把“第三个进程”改为“另一个独立进程”。

## 第三次隔离冷读整改

- 评级 B；五条路径均已闭合，剩余问题来自首次路径被拆散和未定义的上层阶段名。
- 本轮把首次主线画到 `CompiledFxGraph` 构造、可调用原对象返回与无 callable 副本保存；统一使用 FX Graph candidate；明确 `attrs` 携带的就是恢复字段中的 constants；缓存边界改用实际动作描述；删除悬空 bypass 模式；结尾从本文已经建立的单图入口开始收束。

## 第四次隔离冷读整改

- 评级 B；只剩 bypass 收束、三类 key 的关系、首次路径裸 `key`、职责表重新使用未定义阶段名四项。
- 本轮说明两个传播 key 与独立 kernel 后端 key 的关系；首次路径显式写 wrapper source key；职责表改用正文已建立的动作；bypass 收束改为“资格检查后不查存 candidate”。

## 第五次隔离冷读验收

- 报告：`../../inductor-fresh-reader-review/53_CACHE_LOAD_COLD_REVIEW_5.md`，评级 A。
- 读者可仅凭正文区分三类 artifact 和三个 key，并完整复述首次编译、miss/save、跨进程 hit/load、bypass、动态 guard 五条路径。
- 必须修改项：无。
- 当前状态：Q44 通过隔离冷读验收，下一篇按广度优先顺序进入 Q111。

## 完成审计中的事实修正

- 完成审计重新沿 `compile_fx.py` 核对调用位置后，发现旧正文把 FX Graph Cache 命中概括为跳过“图优化、Lowering、Scheduler、Codegen”，范围过大。
- 当前调用顺序是：pre-grad 在进入 AOTAutograd 前运行；训练路径的 joint passes 在 partition 前运行；之后每张单图进入 `_compile_fx_inner`，才尝试 `FxGraphCache.load(codegen_and_compile=fx_codegen_and_compile, ...)`。
- 正文已改为：命中跳过当前单图的 `fx_codegen_and_compile`，即 post-grad、Lowering、Scheduler、Codegen 和子编译；pre-grad 与 joint 不在被跳过范围内。
