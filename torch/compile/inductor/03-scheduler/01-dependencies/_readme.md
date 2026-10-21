# Scheduler 依赖篇作者记录

## 所在层、输入与出口

- 队列位置：Q31，第 2 层，父项 Q30；第五轮已按技能重新复核，保留“本层完成”。本状态不替代 Q311 叶子的源码回修。
- 读者前置：Q23 已理解 buffer/storage/alias/mutation，Q24 已知道普通、extern、template/control-flow operation 的接口。
- 本篇输入：GraphLowering 的 operations、每项 operation 的读写/alias/mutation 信息，以及 graph inputs/constants/outputs。
- 本篇出口：统一 SchedulerNode 包装、规范化 ReadWrites、每节点 unmet dependencies、每 buffer users、mutation version/real-name 映射；Q32 用它排序/DCE/生存期，Q33 用索引兼容性判断融合。
- 直接职责：operation→scheduler node、读写提取、producer/users 建边、alias/mutation 版本关系。Q311 下一层叶子逐行讲依赖构造核心源码；本篇只讲这些职责怎样共同形成图。
- 层次边界：本篇不决定最终拓扑序、不删除节点、不融合，也不把 buffer 名直接等同物理 storage。

## 篇内知识点队列

| 顺序·编号 | 目标与完成条件 | 案例/证据 | 状态 |
| --- | --- | --- | --- |
| 1·K01-R | 比较 Computed/Template/Extern/Nop operation 的 scheduler node 包装及读写来源 | reduction 与 mm | 完成，已实测并复读 |
| 2·K02-R | 从循环重放得到具体 MemoryDep index/vars/size，解释规范化意义 | `[2,3]` reduction | 完成，已实测并复读 |
| 3·K03-R | 区分 MemoryDep、StarDep、WeakDep 的信息强度和消费者 | mm→sin、copy_ 两种对照 | 完成，已实测并复读 |
| 4·K04-R | 走通 available inputs、producer/users、OUTPUT 虚拟 user 和 unmet pruning | mm→sin | 完成，已实测并复读 |
| 5·K05-R | 走通 mutation 前旧读、写回版本，以及 mutation 后表达被改写为新值来源的情况 | copy_ old/read-after 对照 | 完成，已按实际状态纠正预设 |
| 6·K06-R | 解释 alias names 共用 users，并完成多 reader→mutation 综合迁移 | 源码 + 多 reader 推演 | 完成，源码机制；未独立运行 alias 例 |

## 处理情况与案例覆盖

| 情况与条件 | 依赖变化 | 本层出口 | 证据 | 当前深度 |
| --- | --- | --- | --- | --- |
| ComputedBuffer loop | 符号重放 load/store | 精确 MemoryDep | 实验 08 reduction | 已实测并讲解 |
| ExternKernel | 无普通逐位置体 | 输入/输出 StarDep 或 operation 自报依赖 | 实验 08 mm | 已实测并讲解 |
| TemplateBuffer | 输入先按整块依赖建模 | 输入 StarDep，输出写依赖 | 当前 template 源码 | 已讲边界并复读 |
| graph input/constant read | 名字已在 available 集合 | 保留完整 reads，但从 unmet 中裁掉 | 实验 08 mm inputs | 已实测并讲解 |
| internal producer read | 名字由前节点写 | users 登记 consumer，read 留在 unmet | 实验 08 mm→sin | 已实测并讲解 |
| graph output | 无真实计算 consumer | users 增加 OUTPUT 虚拟项 | 实验 08 | 已实测并讲解 |
| mutation 前旧 reader | 覆盖必须等旧 reader | mutator 增加 WeakDep(old-reader output) | 实验 08 copy_ | 已实测并讲解 |
| mutation 后 Python 语义读取 x | functionalization 可将计算改写为直接读取新值来源 y | op0/op1 都读 y；无旧 reader WeakDep，rename 表仍记录写回版本 | 实验 08 read-after | 已实测并讲解 |
| 多次 mutation | 每次更新 rename 链与 real name | 后读依赖最新版本，生成仍写真实 target | 当前源码/实验 09 | 待后篇衔接 |
| alias names | 多个逻辑名共享 storage | name_to_users 键共享同一列表 | 当前源码 | 已介绍，未独立运行 alias 例 |

## 依据与边界

- 当前源码：`dependencies.py` 的 Dep/ReadWrites/extract，`scheduler.py` 的节点创建、set/prune、compute_dependencies 与 mutation maps。
- 实验 08 当前覆盖 reduction MemoryDep、mm extern StarDep、mm→sin producer/users、旧读 copy_ 和 mutation 后读取；需把最后一项写入正文。
- WeakDep 的可裁剪行为来自当前源码；实验证明插入与 weak user 标志，没有单独运行所有 prune/remove 分支。
- alias users 共享来自 compute_dependencies 源码；没有独立运行 NonOwningLayout/alias consumer 图，不能声称实测。
- CPU mm 只证明当前配置为 ExternKernelSchedulerNode；template 分派由源码支持。
- 实际读者反馈尚无。

## 作者复读与迁移题

1. mm 的输入 StarDep 为什么不留在 unmet？答：arg0_1/arg1_1 已在 available inputs；完整 reads 保留，但无需等待内部 producer。
2. sin 对 buf0 的 MemoryDep 为什么连接 Extern mm？答：依赖类型不同仍共享 name=buf0；buf0 producer 是 op0，sin user 是 op1。
3. 旧读案例中 op1 为什么同时有 StarDep(arg0_1) 和 WeakDep(buf0)？答：前者表达覆盖哪个旧版本且输入已可用；后者把 mutation 排在已产生 old 的 reader output buf0 之后。
4. read-after case 为什么没有 WeakDep，是否表示 copy_ 可被删除？答：没有旧 arg0_1 reader；输出算式已直接读 y。但 buf1 仍有 OUTPUT user，代表外部可见输入更新，不能因此删除。
5. mutation_renames 存在是否证明本例有 consumer 读取 buf1？答：不能。它记录最新逻辑版本；本例后继计算已改读 y，没有节点实际用原名检验 rename。
6. view_out/base 两个 alias 名的 readers 为什么 mutation 任一名字都能看见？答：compute_dependencies 让 alias keys 指向同一 name_to_users 列表。
7. 两个旧 readers 彼此是否自然有先后边？答：没有；它们各依赖同一 producer。后续 mutation 才必须排在两者之后，具体排序由 Q32 处理。

答案均指出 dep 名、users、available/unmet 或版本映射，不能仅背三种 Dep 名称。实际读者反馈尚无。

## 当前恢复点

- K01-R 至 K06-R 完成，Q31 达到第 2 层本篇完成条件。
- alias users 共享与 template/Nop 分派的未运行部分明确限定为当前源码证据。
- Q311 保留第 3 层待回修；根队列下一项为同层 Q32。

## 第五轮重新复核

- 范围：operation 包装、读写提取、producer/users 建边、alias/mutation 版本四项职责只建立依赖图；排序、DCE、融合和物理 storage 均留在后续职责。
- 过程：ComputedBuffer 的精确 MemoryDep、extern/template 的 StarDep、mutation 的 WeakDep、available/unmet、OUTPUT 虚拟 user、rename/real-name 和 alias users 共享都有明确对象与消费者。
- 对照：旧读 copy_ 与 read-after-mutation 使用实际 Scheduler 输入解释有无 WeakDep，避免从 Python 语句顺序猜测 functionalization 后的依赖。
- 事实：reduction、mm→sin、旧读和 read-after 来自实验 08；template、alias users 和部分 weak 裁剪只按当前源码陈述，没有冒充实测。
- 理解：只凭正文可判断改变 dep 名、增加两个旧 readers、view/base alias、删除一个弱 reader时，哪些 users/unmet/version 关系改变。实际读者反馈仍为空。
