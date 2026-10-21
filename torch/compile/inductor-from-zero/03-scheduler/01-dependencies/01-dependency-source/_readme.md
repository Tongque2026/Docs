# Q311 Scheduler 依赖构造核心源码备课记录

第五轮状态：补齐 `extract_read_writes` 各 `MemoryUsageType`、原子写 mode 与多个间接变量的机制解释后，叶子复核完成；根队列已转 Q321。旧“完成”记录只作历史依据，本状态来自本轮重新核对正文、产品源码、实验和迁移题。

## 读者、层次与出口

- 读者已经从 Q31 知道 `MemoryDep/StarDep/WeakDep` 的用途，能读懂 producer、users、unmet 和 mutation rename 的案例；尚不知道这些集合由哪些循环和状态更新建立。
- 本篇是第 3 层叶子。输入为 GraphLowering operations、每项 operation 的计算体或自报读写、graph inputs/constants/outputs；出口为 scheduler node、`ReadWrites`、`unmet_dependencies`、共享 users 表和 mutation 版本映射。
- 本篇不展开拓扑排序、DCE、last usage 和融合合法性；它们消费这里建立的关系，分别属于 Q321/Q331。
- 产品基线：`/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。

## 声明范围

| 产品顺序 | 声明 | 本篇职责 |
| --- | --- | --- |
| dependencies.py 1 | `index_vars_squeeze` | 由循环尺寸建立符号索引与范围 |
| dependencies.py 2 | `extract_read_writes` | 重放 LoopBody 或 tracing，汇总 ReadWrites |
| scheduler.py 1 | `BaseSchedulerNode.add_fake_dep/set_read_writes/prune_deps` | 安装读写并裁掉已可用输入依赖 |
| scheduler.py 2 | `SchedulerNode.__init__/_compute_attrs` | 简化循环体、确定 group、提取普通/template 读写 |
| scheduler.py 3 | `Scheduler.create_scheduler_node` | 按 operation 类型选择包装节点 |
| scheduler.py 4 | `Scheduler.compute_dependencies` | alias users、mutation 顺序、rename、输出虚拟用户 |

共 9 个方法/函数；最终正文去除教学注释后须 AST 9/9 等价，并保持各文件声明顺序。`MemoryDep/StarDep/WeakDep/ReadWrites` 的数据类实现不是本叶子的声明范围：Q31 已建立语义，本篇只在这些构造被实际创建和消费时解释字段。

## 篇内知识点队列

| 编号 | 完成条件 | 案例与证据 | 状态 |
| --- | --- | --- | --- |
| K01 | 从四类 operation 判断 scheduler wrapper，并追踪普通 ComputedBuffer 如何在初始化时得到 ReadWrites/unmet | 实验 08 reduction、mm；节点构造源码 | 完成，已复读 |
| K02 | 用 `[2,3]` 规约逐句推演 squeeze vars、LoopBody fast path、tracing slow path和最终 ReadWrites | 实验 08 + dependencies.py | 完成，已复读 |
| K03 | 逐句推演 `compute_dependencies` 的 DedupList、alias 共享、普通 producer/users 与输出保活 | mm→sin 实验；alias 三种合并分支源码推演 | 完成，已复读 |
| K04 | 逐句推演 mutation 的 prior writer、old readers、WeakDep、rename/real-name 和 input mutation 保活 | 实验 08 两种 mutation；两次 mutation 推演 | 完成，已复读 |

## 处理情况覆盖

| 条件 | 状态变化 | 出口/失败 | 归属 |
| --- | --- | --- | --- |
| no-op / computed / template / extern | 选择不同 wrapper 与读写来源 | 三类 scheduler node；未知 operation 报 NotImplementedError | K01 |
| graph input/constant read | 完整 reads 保留，unmet 被裁掉 | 可立即调度的依赖状态 | K01/K02 |
| LoopBody / 普通 callable | fast path 读 memory_usage / slow path ops tracing | ReadWrites | K02 |
| load seed / reduction store / index expr / bucketize / bounds check | 分别重放到记录器；bounds check 有意跳过 | reads、writes、index_exprs；不制造伪依赖 | K02 |
| 间接索引 / 原子 store mode | fast path 将间接变量依次对齐为 tmp0/tmp1；store mode 原样交给记录器 | 两条提取路径可比较；写种类不丢失 | K02 |
| alias 名已各自有 users / 只有一方有 / 新 alias | 合并或共享同一 DedupList | 任一名字登记均可见 | K03 |
| internal read / graph output / unbacked symbol | 登记真实或虚拟 user | producer-users 图 | K03 |
| mutation 有 prior readers / 无 prior readers | 增加 StarDep 与零到多条 WeakDep | 正确先后关系 | K04 |
| 多次 mutation | 递归 rename 到最新版本并保留 real name | 后读连接最新逻辑值 | K04 |

## 证据与恢复点

- 实验 08 已实际覆盖 ordinary ComputedBuffer、extern→pointwise、旧读 mutation 与 mutation 后值被改写为 y 的路径。
- template/no-op、alias 合并、unbacked symbol 的分支目前是源码证据；K03 前需要决定可否用最小独立对象实验直接命中 alias 分支。
- K01–K04 均完成，Q311 叶子完成。最终审计：9 项声明去除 `# 教学` 注释后 AST 9/9 等价；各声明内部及同一声明组保持产品顺序；3 个导航链接存在；实验 08 已重跑并通过 ordinary reduction、extern→pointwise、旧读 mutation 和无旧读 mutation 四组断言。事实验收与理解迁移题分开记录，alias/unbacked/multi-mutation 没有越界写成本机实测。下一项按根队列转 Q321，先建立排序、DCE、mutation 拓扑和 last-usage 叶子的声明范围与篇内计划。

## 第五轮叶子验收

- **声明完整性：** 声明表中的 9 个函数/方法均在正文按产品文件内顺序完整出现；去除固定 `# 教学` 行内注释后，AST 为 9/9 等价。正文没有用省略号或教学替代实现绕过分支。
- **分支覆盖：** wrapper 选择覆盖 no-op、computed/template、extern 与未知 operation 失败；读写提取覆盖 LoopBody fast path、callable slow path、normalize、各类 memory usage、间接变量和 store mode；依赖建图覆盖 available 裁剪、alias 三种合并、普通/输出 users、unbacked def/use、mutation 有/无旧 reader、多次 mutation 与输入副作用保活。
- **具体状态：** `[2,3]` 静态规约得到三条输入 `MemoryDep` 与一条输出 `MemoryDep`；mm→sin 展示 extern 输入依赖被裁剪而内部 buf0 留在 unmet；旧读 mutation 得到 `StarDep(arg0_1)+WeakDep(buf0)` 及版本表；无旧读 mutation 没有 WeakDep；两个旧 reader 得到两条相互独立的 WeakDep。
- **作者迁移复读：** 常量 read 保留在完整 reads 但 unmet 为空；第三个 alias 接入后与前两者共享同一 `DedupList` 对象；两个旧 reader 都约束 mutator，却不互相排序；两次 mutation 后后续 read 解析到最新版本，真实输入名仍可恢复。以上答案均可只凭正文推出。
- **证据边界：** reduction、mm→sin 和两种 mutation 来自实验 08；alias、unbacked、multi-mutation、MemoryUsageType 边支来自当前源码逐句推演，正文未把它们冒充本机运行结果。实际读者尚未提供作答，因此只登记作者复读完成。
