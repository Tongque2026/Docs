# Codegen 虚拟化 ops 篇作者记录

## 本篇位置与教学目标

- 全系列编号：Q41；层次 2；父项 Q40；前置 Q22、Q31、Q34；下一项 Q42。
- 读者已知：Lowering 留下逐位置规则，Scheduler 把节点交给 backend；不知道同一规则怎样既能做依赖分析又能生成 CPU/GPU 表达。
- 入口：`SchedulerNode` 中的 `LoopBody`、循环变量与范围，以及 backend 已建立的 kernel context。
- 出口：当前 handler 生成的标量/向量临时表达、load/store/reduction 语句；Q42 再打开 CPU 与 Triton 如何组织完整 kernel。
- 学完标准：能区分 callable、捕获后的 FX 小图、索引表、`ops` placeholder、当前 handler 与 `CSEVariable`；能沿 pointwise/reduction 写出具体重放状态，并判断依赖快速路径是否真的执行整张图。
- 依据：当前提交 `8332c5622aabd9535969e9a8d0ed60e4a90e4402`、当前源码与独立 CPU 实验；无 GPU 执行。
- 第五轮状态：已按技能重新复核，保留“本层完成”；Q411 尚未创建，仍按第 3 层队列待讲。

## 本层职责和边界

1. 操作协议：`OpsHandler[T]` 规定逐标量操作，不规定外层循环和目标语言。
2. 动态解释器：`Virtualized` 在线程局部动态作用域安装/恢复 handler；`OpsWrapper/OpsValue` 提供运算符写法。
3. 捕获：`LoopBodyBlock` 用 `CaptureIndexing(proxy_ops)` 执行 callable，得到 FX 小图、索引名和 memory usage；此时不读取真实 tensor。
4. 重放：`InterpreterShim` 把当前 handler 作为 `ops` placeholder 实参；kernel 的 `CSEProxy` 再包住 backend override。
5. 消费：codegen 可以重放；依赖提取对已有 LoopBody 走 memory-usage 快速路径，不重放完整算术图。

本篇不逐句解释所有 ops，也不展开 CPU/Triton 循环调度；对应 Q411 与 Q42。

## 处理情况覆盖

| 情况 | 对象变化与出口 | 案例/证据 | 深度/状态 |
| --- | --- | --- | --- |
| pointwise 捕获 | callable → `ops` placeholder FX 图 + `index0` | relu/add/mul 图 | 机制已讲解并复读 |
| pointwise CPU 重放 | load/add/relu/constant/mul/store → `tmp0..tmp5` | 实际 handler 轨迹、生成代码 | 机制已讲解并复读 |
| reduction 捕获与重放 | 增加 reduce var、reduction、store_reduction、累计对象 | 2×128 sum | 机制已讲解并复读 |
| 多种 CPU 候选 | 同一 LoopBody 以 scalar/vector handler 状态重放 | 实际三次 reduction 重放 | 边界已讲，不推广次数 |
| 依赖提取 | 已有 LoopBody 直接消费 `memory_usage/indexing_exprs` | `extract_read_writes` 源码 | 快速路径已讲解 |
| 普通 callable 依赖 | 安装 `RecordLoadStore` 并 define-by-run | 源码，无单独实验 | 慢路径条件已讲 |
| CSE/bounds/store cache | backend 表达经 `CSEProxy` 产生临时与公共状态 | pointwise tmp 轨迹、源码 | 本层机制已讲解 |
| masked/scan/sort/indirect | 捕获 subblock/shim 或后端特殊接口 | 当前源码 | 仅介绍协议位置，叶子再展开 |
| Triton/其他 handler | 复用协议，返回不同目标表达 | 当前源码；无 GPU 运行 | 只讲共同接口，不声称实测 |

## 篇内知识点队列

| 编号 | 完成条件 | 正文/案例 | 状态 |
| --- | --- | --- | --- |
| K01 | 用具体值解释为什么需要可替换 handler | 开篇与协议表 | 完成 |
| K02 | 区分 `_ops`、`V.ops`、模块级 `ops`、`OpsValue` | 动态作用域 | 完成 |
| K03 | 从 callable 走到 FX 图、索引表、memory usage | pointwise 捕获 | 完成 |
| K04 | 说明 placeholder 如何收到当前 handler 并重放 | InterpreterShim | 完成 |
| K05 | 沿 tmp0..tmp5 展示 backend override 与 CSEProxy | pointwise CPU | 完成 |
| K06 | 用 2×128 展示输出轴、规约轴和累计状态 | reduction | 完成 |
| K07 | 区分 codegen 重放与依赖快速路径 | Scheduler 分析 | 完成 |
| K08 | 只依赖正文完成新操作迁移题 | 末节 | 完成 |

## 独立实验与事实表

- 脚本：`../../experiments/12_codegen_virtualized_ops.py`；原始输出：`../../experiments/artifacts/codegen_virtualized_ops.txt`。
- 生成代码：`virtualized_ops_pointwise_output.py`、`virtualized_ops_reduction_output.py`。
- 方法：只在 `SchedulerNode.codegen` 动态范围内给当前 handler 套记录器，调用原 handler；不替换表达或数值行为。

| 案例 | 捕获对象 | 实际重放 | 结果 |
| --- | --- | --- | --- |
| pointwise | load x/y、add、relu、constant 2、mul、store | 一次 scalar；返回 `CppCSEVariable tmp0..tmp5` | `[0,10,4]` |
| reduction | load `128*x0+x1`、add 1、sum、store row | 三次：一次 scalar、两次 `is_vec=True`；最终宽度 16 向量规约 | `[8256,24640]` |

2×3 sum 会在当前实现展开，不能证明 reduction handler；所以选 2×128 并以实际 `ops.reduction` 为机制证据。三次重放是当前 CPU 候选生成行为，不是 OpsHandler 协议保证。

## 当前源码依据

- `ops_handler.py`：`OpsHandler`、`KernelFormatterHandler`、`WrapperHandler`、`OpCounterCSE`。
- `virtualized.py`：`Virtualized`、`OpsValue`、`OpsWrapper`、`_ops` 与 `V`。
- `loop_body.py`：`LoopBody`、`LoopBodyBlock`、`CaptureIndexing`、`InterpreterShim`。
- `scheduler.py`：`SchedulerNode.codegen` 的 SimplifyIndexing 与 current-node context。
- `codegen/common.py`：`Kernel.__enter__`、`CSEProxy`、bounds/load/store/reduction 处理。
- `dependencies.py`：`extract_read_writes` 的 LoopBody 快速路径与 callable 慢路径。

## 作者复读：事实与理解分验

事实：实验在当前提交重跑通过；原始 FX 图、每次 handler 调用、返回对象和数值均保留。GPU、masked/scan/sort/indirect 未运行，正文只讲接口位置。依赖快速路径由源码直接证明，不以 handler 实验冒充。

只依赖正文回答：

1. 捕获时 `ops.load` 是否读取输入值？否；`CaptureIndexing` 把索引登记并让 FX proxy 记录方法调用。
2. 为什么 FX 图有 `ops` 参数？它是 placeholder，重放时由 `LoopBodyBlock.__call__` 传入当前 handler。
3. `relu(x+y)*2` 的 tmp3 是什么？CPU handler/CSEProxy 对 relu(tmp2) 生成的 `CppCSEVariable`，对应 `max(tmp2,0)`。
4. reduction 为什么多一个 x1？x0 选输出行，x1 遍历该行 128 项；`ops.reduction` 把每项值交给累计状态。
5. 依赖提取一定重放 add/relu 吗？不一定；已有 LoopBody 直接遍历捕获期保存的 memory usage，只重建内存依赖。
6. 换成 Triton handler，什么不变？FX 图、索引和 ops 语义不变；handler 返回的目标表达和后端循环组织改变。

作者复读通过；实际读者反馈尚无。

## 续写位置

Q41 第五轮重新复核完成。根队列转同层 Q42；Q411 的完整源码仍按广度优先队列留到下一层。

## 第五轮重新复核

- 范围：ops 协议、动态 handler、LoopBody 捕获、InterpreterShim 重放、CSEProxy 和依赖提取消费形成完整对象链；CPU/Triton 外层循环留给 Q42。
- 过程：pointwise 从 callable→FX 小图→tmp0…tmp5→store，reduction 从 x0/x1→累计→store_reduction；同一 LoopBody 的 scalar/vector 多次重放明确是编译期候选，不是运行时重复计算。
- 快慢路径：codegen 重放完整 FX 小图，已有 LoopBody 的依赖提取直接消费 memory_usage/indexing_exprs，普通 callable 才安装 RecordLoadStore define-by-run；二者没有被笼统写成“都重放”。
- 事实：实验 12 支持当前 CPU handler 轨迹和数值；GPU、masked/scan/sort/indirect 只按协议位置陈述。
- 理解：只凭正文可判断换 handler 后哪些对象不变、CSE 作用域、规约为何增加 x1、依赖快速路径为何不需执行 add/relu。实际读者反馈仍为空。
