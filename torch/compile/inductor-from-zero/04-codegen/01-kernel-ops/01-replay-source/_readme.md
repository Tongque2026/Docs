# Q411 LoopBody 捕获与重放核心源码备课记录

## 读者、层次与出口

- 读者已从 Q41 理解 callable、FX 小图、索引表、当前 ops handler 与重放用途；尚不知道捕获时怎样截获每类内存/复合操作、重放时怎样代入索引并把当前 handler 传给 placeholder。
- 本篇是第 3 层叶子。输入为原始 loop callable、符号循环参数、`LoopBody` 保存的图/索引/submodule，以及当前动态 ops handler；出口为捕获后的 FX graph/memory usage，或重放产生的 backend 表达和副作用。
- 本篇把捕获纳入重放前提，因为 `masked/scan/indirect` 的 submodule 与 shim 决定重放时调用什么；不展开具体 CPU/Triton handler 的目标代码实现，那是 Q421。
- 产品基线 `/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`；仅用当前源码、测试和实验 12。

## 声明范围（按类内产品顺序）

| 类 | 声明 | 职责 |
| --- | --- | --- |
| `InterpreterShim` | `_dummy_gm`、`__init__`、`run_node`、`run` | 低开销 FX interpreter，保存当前节点并动态安装 interpreter handler |
| `LoopBody` | `indexing_from_args`、`__call__` | 以实际符号参数替换捕获索引，设置/清空临时 indexing 并调用 root block |
| `LoopBody` | `bind_set_indirect_shim`、`bind_scan_shim`、`bind_masked_shim` | 为复合操作建立可 clone 的运行时 submodule |
| `LoopBodyBlock` | `__init__`、`__call__` | 捕获完整 FX 小图/memory usage/subblocks；重放时传入当前 ops handler |

共 11 个方法。`LoopBodyBlock.__init__` 中局部 `CaptureIndexing` 类及其 load/load_seed/store/store_reduction/reduction/index_expr/check_bounds/bucketize/masked/scan/sort/frexp/indirect/output 分支全部属于该方法完整源码，不能省略。正文去除教学注释后须 AST 11/11 等价。

## 篇内知识点队列

| 编号 | 完成条件 | 案例 | 状态 |
| --- | --- | --- | --- |
| K01 | 从 `[x0,x1]` 代入解释 `indexing_from_args→LoopBody.__call__→LoopBodyBlock.__call__→InterpreterShim.run`，能追踪 indexing/current_node/handler 的建立与恢复 | pointwise | 完成，已复读 |
| K02 | 逐分支解释 CaptureIndexing 怎样把内存与索引动作写入 memory_usage/FX 图，区分整数 index_expr、Welford tuple、sort/frexp tuple | pointwise/reduction + 源码对照 | 完成，已复读 |
| K03 | 完整推演 masked subblock、scan shim 与 indirect variable 的捕获和重放，说明 clone 绑定如何换父 LoopBody | 三组最小源码推演 | 完成，已复读 |
| K04 | 对照实验 12 的 pointwise/reduction handler 轨迹，完成失败/异常清理边界、11/11 AST 和迁移题验收 | 实验 12 | 完成，已复读 |

## 处理覆盖

| 情况 | 状态变化 | 出口/失败 | 归属 |
| --- | --- | --- | --- |
| index 参数数正确/错误/仍含原 var | flatten、zip replacement | 新索引表或 assert | K01 |
| 普通 root 重放 | `self.indexing` 设表→Interpreter→清空 | handler 结果 | K01 |
| load/load_seed/store/store_reduction | add_index + FX call_method | memory usage + graph | K02 |
| reduction/welford | proxy result；welford 拆三项 tuple | 标量或三元引用 | K02 |
| index_expr int/symbolic | int 变 constant；符号登记 INDEX_EXPR | FX value | K02 |
| bounds/bucketize | 两项 bounds 索引或 offsets size 被登记 | graph + usage | K02 |
| masked | 新 subblock + bind_masked_shim + call_module | 嵌套图 | K03 |
| scan | bind_scan_shim + call_module + tuple 拆分 | 多结果 tuple | K03 |
| sort/frexp | proxy 按已知 arity显式拆 tuple | tuple | K02/K03 |
| indirect | 分配 INDIRECT 符号、set shim、call_module，重放时 replace indexing | 后续索引使用实际值 | K03 |
| output | 显式创建 FX output | root/subblock 出口 | K02 |
| 捕获配置 | SimplifyIndexing，按配置可再包 IndexPropagation | FX graph | K02 |

## 证据与边界

- 实验 12：`../../../experiments/12_codegen_virtualized_ops.py` 及 artifacts；本轮需重跑 pointwise/reduction 数值与 handler 轨迹。
- masked/scan/sort/frexp/indirect 没有本机 GPU 行为实验；本篇只推演 loop_body.py 中共同捕获/重放协议，不声称某 backend 已成功生成。
- `_dummy_gm` 的 lru cache 与 `garbage_collect_values=False` 是 interpreter 构造事实；性能收益不作未测量的数量主张。
- 实际读者反馈尚无；作者迁移题与事实验证分别记录。

## 当前恢复点

K01–K04 均完成，Q411 叶子完成。根队列转 Q421。

## 第五轮叶子验收

- **声明完整性：** 11 个目标方法全部完整出现；脚本遍历 Markdown Python AST（含 `InterpreterShim` class 内方法和 `LoopBodyBlock.__init__` 的局部类）并与当前类字典源码比较，11/11 等价。
- **分支覆盖：** 参数数与原符号 assert、普通重放和异常残留边界；load/load_seed/store/store_reduction、reduction/welford、整数/符号 index_expr、bounds、bucketize、masked、scan、sort、frexp、indirect 与 output 均有条件和出口。
- **具体状态：** `128*x0+x1` 代为 `128*c0+c1` 后经 get_index 交当前 handler；masked 保存 root call_module 与独立 subblock；indirect0 经 set shim 替换当前 indexing；scan 保留 combine_fn 并显式拆 tuple。
- **作者迁移复读：** 少传循环维在重放前 assert；换 Triton handler 不改索引表/FX 顺序；重排克隆时 shim 必须换父 body；handler 异常恢复动态上下文却不保证 `self.indexing` 清空。答案均可仅凭正文推出。
- **事实边界：** 实验 12 本轮通过，支持 CPU pointwise 一次重放和 reduction 一次 scalar/两次 vector 重放及数值；复合操作分支来自当前源码推演，未冒充 GPU 实测。实际读者尚未作答。

下一项 Q421：在 `04-codegen/02-backends/01-backend-source/` 建立 CPU/Triton 索引与 kernel 生成叶子计划。
