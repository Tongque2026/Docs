# Scheduler 篇作者记录

## 所在层、输入与出口

- 队列项：Q30，第 1 层，父项 Q00；第五轮已按技能重新复核，保留“本层完成”。本状态只覆盖四项直接职责的协作，不替代 Q31–Q34 的内部回修。
- 读者已经掌握：Lowering 交出 operations、buffers、graph outputs 和 mutation/特殊调用状态；综合例实际为 `ComputedBuffer buf0 → ExternKernelOut buf1`。
- 本篇输入：已物化的 IR operations 及其读取、写入、layout、device、alias/mutation 和输出需求。
- 本篇出口：依设备与边界有序提交给后端的普通/融合/template/extern/nop 调度节点，以及 last usage、available/completed 等生成时状态。
- 四个直接子职责与目录一致：Q31 包装节点并建立依赖；Q32 拓扑、DCE、mutation 真名和 last usage；Q33 产生并筛选融合候选；Q34 按 device/backend/节点类型提交和 flush。
- 下一层打开四项内部；本篇不逐行讲依赖提取、融合主循环或 `_codegen` 分支。

## 篇内知识点队列

| 编号 | 问题、对象与完成条件 | 案例 | 状态 |
| --- | --- | --- | --- |
| K01-R | 四项职责怎样把 operations 变成后端提交序列 | 系统图与 operation→scheduler node 包装 | 完成，已复读 |
| K02-R | 精确读写、extern 全量依赖、mutation 顺序怎样成为图边 | reduction、mm→sin、old→copy_ | 完成，已作边构造、mutation 对照与迁移复读 |
| K03-R | 排序、DCE、ancestor 与 last usage 怎样改变计划而不改数学 | three-mm 分叉、controlled DCE、double mutation | 完成，已作拓扑分叉、DCE 与逐节点 last usage 复读 |
| K04-R | 垂直/水平融合为什么接受，shifted/extern/size 为什么拒绝 | 实验 10 五类对照 | 完成，已作候选分数、成员变化与拒绝原因复读 |
| K05-R | 后端分派怎样把计划变成 flush/extern/新 group 时间线 | pointwise→extern mm→pointwise；fallback | 完成，已作状态集合、flush 与证据边界复读 |
| K06-R | 综合迁移验收 | 增加图输出、旁路 user、跨 device 或 extern 边界 | 完成，已作四职责贯通与条件迁移复读 |

Q30 的六个知识点均已完成并在第五轮重新复核；根队列转同层 Q40，不提前进入 Q31。

## 处理情况与案例覆盖

| 情况 | 关键状态变化 | 本层出口 | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| 普通精确访问 | MemoryDep 记录 name/index/ranges，图输入已 available | SchedulerNode 及 producer/users | 实验 08 | 机制已讲并复读 |
| extern/fallback/template | StarDep 保守覆盖整 buffer，节点类型形成边界 | Extern/Template scheduler node | 实验 08/11、当前源码 | 机制已讲并复读；template 未实测 |
| mutation | WeakDep 保护旧 reader，renames 映射新旧逻辑版本 | 合法无环顺序与真实 storage 名 | 实验 08/09 | 机制已讲并复读 |
| 分叉与 DCE | topo 保持 producer 在前，删除无输出无副作用链，保留 effect | 排序后的 live nodes/ancestors | 实验 09 | 机制已讲并复读 |
| last usage | 每份真实 storage 的最后消费者变化 | wrapper 从哪个节点以后可以释放或复用它 | 实验 09/11 | 机制已讲并作逐节点状态复读 |
| 垂直/水平融合 | 候选评分、合法性、后端能力共同决定 | FusedSchedulerNode 或保留分开 | 实验 10 | 机制已讲并作接受/拒绝对照复读 |
| backend dispatch | device/node type 触发 create backend、flush、extern/template/combo | 有序后端动作与 available/completed | 实验 11、当前源码 | 机制已讲并复读；template 仅源码路径 |

## 依据、旧状态撤销与反馈

- 源码依据：`scheduler.py:Scheduler._init/create_scheduler_node/_codegen`、依赖与融合方法、各 `BaseScheduling` 后端接口。
- 实验 08–11 分别采集依赖、排序/生存期、融合候选和 backend dispatch。CPU 外行为只按当前源码和测试陈述，不伪装成本机实测。
- 上一版将“全局整理”和“生存期与后端分派”画成职责，和 Q31–Q34 四个子目录不一一对应；第二轮改为依赖、顺序/生存期、融合、后端分派四项。
- 旧记录直接宣称 K01–K04 与 Q31–Q34 完成，但缺完整覆盖表，融合只讲接受路径，DCE、fallback dispatch 和 available/completed 状态没有进入本层完整案例；旧完成结论撤销。
- 实际读者反馈：尚无；作者迁移题与实验分别记录，不能互相替代。

## 第二轮事实与理解验收

- 事实核对：实验 08 中普通 reduction 使用精确 MemoryDep；mm 使用 StarDep；input mutation 在 `compute_dependencies` 后增加 `StarDep(arg0_1)` 与 `WeakDep(buf0,mutating_buf=buf1)`，并建立新旧名字映射。
- 事实核对：实验 09 的 three-mm 中 op1/op2 都以 op0 为 ancestor，buf0 最后在 op2 使用；controlled DCE 只保留 effect/live，删除 dead leaf 及其 parent。
- 事实核对：实验 10 同时覆盖 vertical/horizontal 接受和 shifted-index、extern、max-fusion-size 拒绝；正文没有由相同 numel 直接推出融合。
- 事实核对：实验 11 的 mixed dispatch 是 `op0_op1 CPU group → flush → extern op2 → op3 CPU group → final flush`；available/completed 与 pending free 的变化已进入正文。
- 作者只凭正文可答：若 b 另作为图输出，Q31 增加 OUTPUT user，Q32 延长 storage 生存期，Q33 融合时仍须产出 b，Q34 的 extern 前 flush 顺序不变。
- 作者只凭正文可答：两个 8 项节点仍可能因 shifted index 没有 shared data 而拒绝；同 device 也可能因 extern node type 形成边界；跨 device 则同时改变融合合法性和 backend/guard 切换。
- 本篇只完成 Q30 协作，Q31–Q34 内部仍待第二轮回修；实际读者反馈为空。

## 第三轮严格复读进度

- K02-R 事实：实验 08 的 mm 写 buf0 用 StarDep，sin 读 buf0 用 MemoryDep，仍由共同 buffer 名形成 op0→op1；输入依赖因 graph inputs 已 available 而不是 unmet。
- mutation 对照：有旧 x reader 时 copy 节点新增 `WeakDep(buf0, mutating_buf=buf1)`；没有旧 reader、且新结果已等价改写为直接读 y 时没有 WeakDep。两例都保留对原 arg0_1 的 StarDep 和 mutation rename/real-name 映射。
- 迁移：若 mm 的输出名从 buf0 改成 buf7，只改 producer 而不改 sin 的 read 名会断开真实边；依赖类型相同并不能弥补名字不一致。若 copy 前增加一个新的旧 x reader，dependency completion 必须为相应旧结果增加 weak user。
- K02-R 完成。当前知识点转到 K03-R：用 three-mm、controlled DCE 和 mutation 真实名推演拓扑、删除、ancestor 与 last usage。Q30 尚未完成；实际读者反馈仍为空。

### K03-R 事实与理解验收

- three-mm 正常顺序为 op0、op1、op2，反转输入探针的另一合法顺序为 op0、op2、op1；两者都保留 op0 在先，说明兄弟消费者没有彼此祖先关系。
- 正常顺序的 last usage：op0 完成 arg0_1/arg1_1，op1 完成 arg2_1，op2 完成 arg3_1/buf0。把兄弟顺序交换后，buf0 必须跟随较后的消费者移动，而不是固定属于某个 operation。
- controlled DCE 只保留 effect 与 live operation；dead_leaf 无 user 后被删，dead_parent 随之成为无 user 并反向删除。effect operation 因副作用保留，但其无用输出 buffer 可移除。
- double mutation 到 Scheduler 已被整理为 before、直接读 y 的 middle、最终写 z 三项；只有最终 buf2 映射回 arg0_1。正文已删除“源码两次 copy 必然形成两个调度 mutation 版本”的隐含推断。
- 迁移：若 three-mm 新增第三个 buf0 消费者，buf0 的 last usage 必须移到最终顺序中三个消费者的最后一个；若 dead_leaf 变成 graph output，反向 DCE 不再删除它及其 producer。
- K03-R 完成。当前知识点转到 K04-R：用接受与拒绝对照解释垂直/水平融合决定。Q30 尚未完成。

### K04-R 事实与理解验收

- vertical 从 op0、op1 变为成员 `[op0,op1]` 的 fused node，实际分数 `(True,True,32,-1)`；正文逐项解释该排序分数，不把它误作合法性证明。
- horizontal 的候选同时包含两组垂直边和一组共享输入的水平边，最终四成员同组；branched 的共同 producer 与两路消费者最终五成员同组。
- 三个拒绝出口来自不同检查：shifted index 为 no shared data，extern 为节点类型边界，max_fusion_size 为规模限制。最终数值均与 eager 相同，拒绝只改变生成分组。
- 迁移：仅把 roll 输入长度从 8 改成 16 不会修复索引不匹配；把 max_fusion_size 从 1 恢复到允许两个成员可以重新开放 vertical 候选；把 mm 换成同位置 pointwise 才可能跨过节点类型关口，仍需重新检查共享数据。
- K04-R 完成。当前知识点转到 K05-R：把 fused/extern/fallback 节点变成 flush、提交、available/completed 和 pending-free 时间线。Q30 尚未完成。

### K05-R 事实与理解验收

- mixed 实验的实际动作是 `op0_op1 CPU group → flush → extern op2 → op3 CPU group → final flush`。正文列出每一步 available、completed、pending-free 的新增与清空。
- available/completed 是 Scheduler 的生成状态，不表示运行时 kernel 已执行；CPU group 要到 extern 前执行 flush 才完成代码发射。正文已明确区分“调度记录完成”与“代码已经写出”。
- fallback 的 FallbackKernel 与 MultiOutput 都按 extern node 分派；同为 CPU 不能让它们混进普通 pointwise group。
- 证据修正：名为 template_mm 的尝试实际选择 ExternKernelOut，没有观察到 codegen_template。本篇对 template 分派只采用 `_codegen` 当前源码，不再把该运行当成模板实测。
- 迁移：若删去 extern mm，两个 CPU group 之间少了强制边界，是否合并仍回到 K04 的融合检查；若 op1 的结果同时作为 graph output，flush 顺序不变，但 last usage 与 pending-free 不得在 mm 后终止该 storage。
- K05-R 完成。当前知识点转到 K06-R：只凭正文贯通依赖、顺序/生存期、融合和分派，并完成 Q30 本层审计。

### K06-R 与 Q30 完成审计

- 综合链 `op0→op1→extern mm→op2` 已从 Q31 依赖边走到 Q32 拓扑/last usage、Q33 融合分组和 Q34 flush/提交出口；每一步使用同一组逻辑 buffer，没有把 FX node、IR operation 和 scheduler node 混为一层。
- 增加 b 的 graph output：Q31 增加 OUTPUT user；Q32 将 b 保留到返回；Q33 即使融合 op0/op1也必须产出 b；Q34 的 extern 前 flush 仍存在。正文能指出改变与不变部分。
- 增加读取 a 的旁路 user：它既可能参与水平/分支融合，也会把 a 的 last usage 移到最终顺序中较后的消费者；不能仅凭源码相邻决定 op0/op1 融合。
- 把 extern 后的 op2 换到另一 device：数学 producer→consumer 边不变；融合因 device mismatch 拒绝；分派增加 backend/guard 切换和 flush。该结论限定为当前层交接，具体跨设备生成留给 Q34/Q42。
- 覆盖审计：精确/全量/weak 依赖、mutation、分叉/DCE、last usage、垂直/水平融合及三类拒绝、普通/extern/fallback/template 分派边界均有入口、状态变化和本层出口。template 分支只由当前源码支持，未标成实测。
- 事实与理解分开：实验 08–11 支持观测对象与状态；迁移题仅依赖正文。实际读者反馈仍为空。
- Q30 达到第 1 层完成条件；Q31–Q34 的内部算法仍待第二层回修，不能由本篇完成状态代替。

## 第五轮重新复核

- 范围：系统图、正文分工、四个子目录和根职责树一致，均为 Q31 依赖、Q32 顺序/生存期、Q33 融合、Q34 后端分派；last usage 没有混入分派职责，flush 也没有冒充融合算法。
- 过程：精确/全量/weak 依赖、mutation 版本、分叉/DCE、last usage、垂直/水平融合与三类拒绝、普通/extern/fallback/template 分派边界都有入口、具体状态和 Scheduler 出口。
- 综合：`op0→op1→extern mm→op2` 使用同一组 buffer 贯通四项职责，最终得到 `CPU group→flush→extern→CPU group→final flush`；available/completed 明确是生成状态，不是运行完成。
- 事实：实验 08–11 支持实际依赖、节点、候选和 dispatch 状态；template 尝试实际得到 ExternKernelOut，因此 template 分派只按当前源码陈述，没有伪装成实测。
- 理解：只凭正文可判断增加 b 输出、增加 a 的旁路 user、把 op2 换到另一 device 时，哪些依赖、生存期、融合与分派决定改变。实际读者反馈仍为空。
