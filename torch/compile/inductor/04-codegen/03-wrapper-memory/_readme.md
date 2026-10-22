# Q43 作者记录：wrapper 与内存计划

> 当前状态：逐篇隔离冷读已重新打开本篇。第一轮评级 B，已完成首轮结构补强，等待新读者复验。

## 第六轮：隔离冷读后的首轮整改

- 报告：`inductor-fresh-reader-review/44_WRAPPER_MEMORY_COLD_REVIEW_1.md`，评级 B。
- 已通过：wrapper 主线、延迟 planning line、两次 mm 的两次分配/两次复用、`del` 不等于设备内存立即释放。
- 实质断点：逻辑 buffer、Python 引用和 storage 混写；mutated input 从模块 `(buf0,)` 到用户 `(old,x)` 的外层重组未展开；pool 实验 offset 全 0，不能解释 Temporal/Spatial split；reinterpret 和 alias/view 无完整案例；boxed/last_usage/BufferGroup 后置；无迁移题。
- 修改：开头固定三层对象与两层返回；修正 S0/S1 时间线；首次定义 boxed、last_usage、BufferGroup；增加 `(2,3)→(3,2)` reinterpret 推演；增加 A/B/C 三 group 的 112B pool 数值放置稿；展开外层保存原输入、模块原地写回、元数据重建 `(old,x)`；补 input view；增加含 reinterpret、同时活跃、最后使用再复用和输入 view 的新时间线题。
- 证据边界：A/B/C pool 是依据当前 planner 约束构造的教学放置稿，不冒充实际实验；实际 mm pool 仍只证明 group/live range/pool 首末使用且 offset=0。
- 当前状态：等待第二位隔离冷读者复验。

### 第二次隔离冷读

- 报告：`inductor-fresh-reader-review/45_WRAPPER_MEMORY_COLD_REVIEW_2.md`，评级 B。
- 已通过：A/B/C 112B 放置、reinterpret、S0/S1 时间线、模块 tuple 与用户 tuple 的基本分层、末尾 buf3 接管 S0。
- 核心断点：mutation contract、output-alias contract 和 fresh-output contract 混写；将保存 `original_x` 的语义模型写成未经限定的通用实现；开头“空间足够”与默认等容量 reuse key 冲突；迁移题把规划前 empty 和规划后 reuse 混在一起。
- 修改：固定三份独立合同并逐项解释 `return old,x`；限定该 tuple 形态为本次 inference 编译入口；将开头改成命中兼容 key；说明 last use 是静态复用候选而非运行时 del 后查池；展示迁移题规划后的 reinterpret 行；补 pool 基底引用与 allocator、同 stream 与 host/跨 stream 的边界。
- 当前状态：等待第三位隔离冷读者复验。

### 第三次隔离冷读

- 报告：`inductor-fresh-reader-review/46_WRAPPER_MEMORY_COLD_REVIEW_3.md`，评级 B。
- 已通过：三份输出合同与 ABI 限定、静态 free/reuse、默认等容量 key、pool 放置和模块/user tuple 分层。
- 剩余断点：迁移题把待规划 IR 写成可执行 `empty/del`；GPU 异步 kernel 未完成时，正文没有连接 host `del`、同 stream 复用和 stream-aware allocator。
- 修改：迁移题规划前改成 Allocate/Free/Call/Return 抽象行，规划后单列最终 reinterpret Python；说明同 stream 提交顺序保护先读后写，allocator 必须记录未完成 stream 使用，跨 stream 需要 event/wait 或等价记录；阅读清单改为 host 提交 last use，而非设备完成时刻。
- 当前状态：等待第四位隔离冷读者复验。

### 第四次隔离冷读

- 报告：`inductor-fresh-reader-review/47_WRAPPER_MEMORY_COLD_REVIEW_4.md`，评级 B。
- 剩余断点：跨 stream 直接 alias/reinterpret 与 allocator 回收后重分配混写；allocator tracking 不能替代直接跨流覆盖的执行依赖。迁移题预置 ReuseLine，但未解释其来源。
- 修改：拆分两条异步安全路径，直接接管跨流必须 event/wait，allocator tracking 只保护回收再分配；迁移题改用 `InplaceAliasLine` 教学标记，并说明它表示已批准原地覆盖，不是普通 free/allocate 配对结果。
- 当前状态：等待第五位隔离冷读者复验。

### 第五次隔离冷读

- 报告：`inductor-fresh-reader-review/48_WRAPPER_MEMORY_COLD_REVIEW_5.md`，评级 A。
- 结论：直接 alias/reinterpret 与 allocator 回收再分配已经分开；`InplaceAliasLine` 明确为教学标记；三份输出合同、普通规划、pool、guard/stream 与迁移题均可只凭正文闭环。
- 非阻断观察：迁移题保留了一个写入后无用户的 buf3，用于检验 free/reuse；它是 planner 抽象时间线，不是声称死结果一定会留在优化后的真实图中。
- 当前状态：Q43 通过隔离冷读验收。A 只证明指定冷读角色能从正文建立逻辑，不代替真实用户反馈；若用户仍有断点，继续按实际断点重开。

## 本篇位置与教学目标

- 全系列编号：Q43；层次 2；父项 Q40；前置 Q32、Q34、Q42；下一项 Q44。
- 读者已知：Scheduler 有 last_usage，backend 已生成 kernel；不知道逻辑 buffer 名、Python tensor 引用和底层 storage 的区别，也不知道释放记录怎样变成 reuse。
- 入口：按 Scheduler 顺序产生的 allocation/free/call/device 行、图输入输出合同及 buffer 布局。
- 出口：一次可调用 wrapper：输入 guards、实际分配/复用、kernel/extern 调用、引用释放、device guard 与返回 tuple。编译/加载留给 Q44。
- 学完标准：能从两次 mm 案例画出每个 storage 的生存期，解释为何只有两次新分配；能分别推演默认 reuse planner 与 inference pool planner；能判断输入、输出、alias、不同容量为何不能随意复用。
- 依据：当前提交 `8332c5622aabd9535969e9a8d0ed60e4a90e4402`、当前源码和 CPU 独立实验；无 GPU guard 实机运行。

## 本层职责与边界

1. 建立 wrapper 时间线：allocation/free 先保存成带 IR buffer 的 planning line，普通调用保持顺序边界。
2. 默认 reuse：顺序扫描 free/allocate，用 `(device,dtype,symbolic storage_size)` 栈匹配最近释放存储。
3. inference pool：把 inplace 名字并成 BufferGroup，计算 live range，分配到 device pool 的时间/空间树。
4. 生成调用合同：解包 boxed args、清容器、guard、分配/view、调用、del、guard/stream、返回引用。

本篇讲 wrapper 级对象与状态交接；allocator 内部、CUDA 异步执行和缓存加载分别不在本层展开。

## 处理情况覆盖

| 情况 | 状态变化与出口 | 证据 | 深度/状态 |
| --- | --- | --- | --- |
| 输入合同 | boxed args → 局部名，清 args，检查 size/stride | 8×8 实际 wrapper | 机制已讲解 |
| extern→pointwise→extern→pointwise | 调用顺序与 Scheduler 一致 | 两 mm pipeline | 完整案例已复读 |
| inplace reuse | `buf0→buf1`、`buf2→buf3` 接管同 storage | 默认与 pool 两产物 | 机制已讲解 |
| 同时活跃不可复用 | 第二 mm 时 buf1 是输入、buf2 是输出 | 实际时间线 | 因果已讲解 |
| 输入最后使用 | InputBuffer 直接生成 del，不进普通 reuse pool | 实际 wrapper+源码 | 边界已讲解 |
| 图输出保活 | buf3/storage 保留到 tuple 返回 | 两产物 | 机制已讲解 |
| 默认 free→reuse | `FreeIfNotReusedLine` 入 key 栈；后续 Allocate 变 ReuseLine | 源码+实际 `# reuse` | 机制已讲解 |
| 不同 shape/stride 同容量 | 允许接管后生成 reinterpret view | 源码；本例同布局 | 条件已讲，未声称实测 |
| inference pool | buffer groups/live ranges/pool 首末使用 | `memory_planning=True` 实际产物 | 机制已讲解 |
| pool 空间切分/offset | TemporalSplit/SpatialSplit 放置 | 当前源码；本例 offset=0 | 只按源码介绍 |
| alias/view/mutated output | 新 `old` storage 与调用者 x 原 storage 分开；局部 `del` 后修改仍可观察 | `mutation_output_code.py` 与指针断言 | mutated-input 完整案例；view 由同一输出合同介绍 |
| GPU guard/stream | device scope 包住相关生成调用 | Scheduler/wrapper 源码与测试 | 无硬件实测 |

## 篇内知识点队列

| 编号 | 完成条件 | 案例/正文 | 状态 |
| --- | --- | --- | --- |
| K01 | 区分逻辑名、tensor 引用和 storage | 开篇与案例 | 完成 |
| K02 | 解释 planning line 为何延迟到全局时间线后决定 | 流程与默认 planner | 完成 |
| K03 | 沿四次调用画出 storage A/B 的生存期 | 两 mm pipeline | 完成 |
| K04 | 从 last_usage 走到 input del、中间 free/reuse、输出保活 | 同案例 | 完成 |
| K05 | 代入真实 reuse key，说明布局不同如何 reinterpret | 默认 planner | 完成 |
| K06 | 用实际 pool 产物解释 BufferGroup/live range/首末使用 | pool pipeline | 完成 |
| K07 | 建立 guards、动态尺寸、输出 alias、device guard 边界 | 后半篇 | 完成 |
| K08 | 完成“first 也返回”迁移题 | 作者复读 | 完成 |

## 独立实验与实际事实

- 脚本：`../../experiments/14_wrapper_memory.py`。
- 默认产物：`wrapper_memory_output.py`；pool 产物：`wrapper_memory_pool_output.py`；摘要：`wrapper_memory.txt`。
- 请求：`mm(x,w0) → relu → mm(...,w1) → sin`，三个 8×8 contiguous float32 输入。
- 默认 planner：实际两次 `empty_strided_cpu`；`buf1=buf0; del buf0` 与 `buf3=buf2; del buf2`；第二次 mm 前不能复用 buf0 storage，因为 buf1 仍作输入。
- pool planner：实际 `pool1` 承载中间 `buf0/buf1`，第二 mm 后 `del pool1,buf0,buf1`；`pool0` 承载输出 `buf2/buf3` 并返回。两个结果均与 eager `assert_close`。
- 修改输入产物：`mutation_output_code.py` 中 kernel 同时把旧 x+1 写入新 `buf0`、把 y 写回 `arg0_1`；实验断言显式返回的 x 与调用者 x 保持相同 `data_ptr`。生成模块 tuple 只含新增 `buf0`，mutated input 由外层调用合同接回。
- 本例 pool 都是单个 8×8 allocation，未展示非零 offset 或多个不重叠对象共享同一更大 pool；相应算法只由源码支持。

## 当前源码依据与纠错

- `wrapper.py`：`buffer_reuse_key`、planning lines、`memory_plan_reuse`、allocation/free/reuse、输入/输出、device guard、kernel call。
- `memory_planning.py`：LiveRange、TemporalSplit、SpatialSplit、AllocationPool、BufferGroup、MemoryPlanner passes。
- `scheduler.py`：last_usage 到 pending free 与 flush 的交接。

纠正旧概括：默认 reuse key 不包含 shape/stride，只有 device、dtype 和简化后的 symbolic storage size。相同 key 且 `can_reuse()` 未命中特殊禁用条件即可选择旧 storage；shape/stride 相同生成直接别名，不同则用 `reinterpret_tensor` 建新 view。因此被保护的是设备、dtype、容量表达式和特殊身份约束，不是“布局必须相同”。

## 作者复读：事实与理解分验

事实：默认与 pool 两种配置均重跑成功并保存完整 wrapper；新分配数、reuse 行、del 位置和返回值由产物直接支持。GPU guard、动态 shape、空间切分只按当前源码解释。

只依赖正文回答：

1. `buf1=buf0` 是否复制 64 个值？否，只建立新 tensor 名对同一 storage 的引用；relu 的 in_out pointer 原地改该存储。
2. 第二 mm 为什么必须新分配 buf2？调用期间 buf1 仍是输入，输出不能覆盖尚在读取的数据。
3. 输入为什么有 del 却不能作为普通中间 reuse？`InputBuffer` 的局部引用可释放，但 `can_reuse` 排除 graph input，保护调用者可观察存储。
4. 同容量、不同 stride 能否复用？默认 key 可匹配；生成端用 reinterpret view 表达新 shape/stride，仍需满足 device/dtype/容量及非特殊 buffer 条件。
5. pool1 为什么在第二 mm 后删除？buf0/buf1 组的最后用户是第二 mm；pool0 含最终输出组，生存期延长到调用末尾。
6. 若 `first` 也作为图输出返回会怎样？其 storage 必须保留旧 mm 结果，relu 不能原地覆盖；live range 与 activated 重叠，需要另一存储或不同安排。

作者复读通过；实际读者反馈尚无。

## 续写位置

## 第五轮全量复核

- **范围**：输入解包/guard、规划行、默认 reuse、inference pool、last usage、kernel/extern 顺序、输入与输出保活、mutated input、device guard/stream 都有位置；allocator、异步执行和加载仍留在声明边界外。
- **过程**：两次 mm 案例从 boxed args 走到两次新分配、两次 pointwise 原地接管、最后返回；修改输入案例另走完“旧 x 读入新 buf0 + y 写回原 x storage + 外层返回修改后 x”的输出合同。
- **事实证据**：默认 planner 与 pool planner 产物均实际运行并与 eager 相等；修改输入的生成代码和 `data_ptr` 断言支持 storage 身份结论；GPU guard、动态 shape 和 pool 非零 offset 只按当前源码说明。
- **理解迁移**：正文足以推出：若 `first` 也返回，relu 不能覆盖其 storage；若同容量但不同 stride，默认 planner 可用 reinterpret view；若第二 mm 尚在读 `buf1`，`buf2` 必须另分配；局部 `del arg0_1` 不会使调用者的 x 消失。
- **层次**：本篇讲 wrapper 级时间线和对象交接；planning classes 与复用扫描的完整源码仍归 Q431。

Q43 第五轮回修完成。根队列转同层 Q44；Q431 仍按广度优先队列留到下一层。

## 第五次隔离冷读

- 报告：`../../inductor-fresh-reader-review/48_WRAPPER_MEMORY_COLD_REVIEW_5.md`，评级 A。
- 结论：正文已经把直接 alias/reinterpret 的流顺序要求与 allocator 回收后的异步安全分开；迁移题也明确区分了教学用 `InplaceAliasLine` 与 planner 实际产生的 `ReuseLine`。普通规划、pool 规划、三类返回合同和迁移题均能仅凭正文闭合。
- 非阻断观察：迁移稿里的 `buf3` 是用于推演 free/reuse 的抽象 planner 时间线，不表示真实优化图一定保留无消费者的结果；这不影响本篇要检验的内存规划关系。
- 当前状态：Q43 通过隔离冷读验收，停止继续润色，进入同层 Q44。
