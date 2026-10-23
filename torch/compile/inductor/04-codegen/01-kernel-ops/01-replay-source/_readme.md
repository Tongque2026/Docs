# Q411 LoopBody 捕获与重放核心源码备课记录

## 第六轮：隔离读者按首次执行顺序回修

- 第一名冷读者只读正文，报告 `/work/learning/1021/56sol/inductor-fresh-reader-review/121_LOOPBODY_REPLAY_COLD_REVIEW_1.md`，评级 B。它指出字段状态未在长源码前建立、局部 staticmethod 中 `self.body` 的闭包来源未解释、scan callable 的身份不清、实验结果无法从给定输入推算。
- 正文增加 LoopBody 字段与生命周期表；区分外层 block self、CaptureIndexing 实例 self 和 staticmethod 捕获的外层 self；明确 combine_fn 保存在普通 Python shim 闭包中；补齐 pointwise/reduction 输入、算式和结果；分别说明动态上下文恢复与捕获/重放对象状态不回滚的边界。
- 第二名冷读者报告 `/work/learning/1021/56sol/inductor-fresh-reader-review/124_LOOPBODY_REPLAY_COLD_REVIEW_2.md`，评级 B。最早断点是完整源码先定义闭包、后创建 tracer，读者首次遇到 `tracer.create_proxy` 时尚未建立实际执行时序。
- 正文保持产品源码原顺序，在代码前新增定义期与执行期：空 graph、ops placeholder、`CaptureIndexing(proxy_ops)` 的 `_inner`、SimplifyIndexing 包装、执行 callable 与 output 的先后关系。
- 第三名冷读者报告 `/work/learning/1021/56sol/inductor-fresh-reader-review/125_LOOPBODY_REPLAY_COLD_REVIEW_3.md`，评级 B。它已完成两道迁移题，剩余断点是 SimplifyIndexing/IndexPropagation 契约解释迟到，以及只有 shim `.clone` 接口而没有复制侧调用。
- 正文将两个包装器的输入、动作、输出移到首次出现之前，并加入 `_init_with_copy` 遍历 root/subblocks/submodules、以新 LoopBody 调用 clone 的实际代码。
- 第四名全新读者报告 `/work/learning/1021/56sol/inductor-fresh-reader-review/126_LOOPBODY_REPLAY_COLD_REVIEW_4.md`，评级 A，无首次理解断点；能够从捕获走到重放，推演 `x0 → 2*x0+1`，并从复制侧追踪 masked shim 重新绑定。
- 2026-10-22 重跑 `python3 experiments/12_codegen_virtualized_ops.py` 通过：pointwise `[0.0,10.0,4.0]`，reduction `[8256.0,24640.0]`；产物更新为 `experiments/artifacts/codegen_virtualized_ops.txt`。运行证据保留在作者记录，不加入读者主线。

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

K01–K04 均完成，Q411 已通过第六轮隔离冷读 A。根队列转 Q421。

## 第五轮叶子验收

- **声明完整性：** 11 个目标方法全部完整出现；脚本遍历 Markdown Python AST（含 `InterpreterShim` class 内方法和 `LoopBodyBlock.__init__` 的局部类）并与当前类字典源码比较，11/11 等价。
- **分支覆盖：** 参数数与原符号 assert、普通重放和异常残留边界；load/load_seed/store/store_reduction、reduction/welford、整数/符号 index_expr、bounds、bucketize、masked、scan、sort、frexp、indirect 与 output 均有条件和出口。
- **具体状态：** `128*x0+x1` 代为 `128*c0+c1` 后经 get_index 交当前 handler；masked 保存 root call_module 与独立 subblock；indirect0 经 set shim 替换当前 indexing；scan 保留 combine_fn 并显式拆 tuple。
- **作者迁移复读：** 少传循环维在重放前 assert；换 Triton handler 不改索引表/FX 顺序；重排克隆时 shim 必须换父 body；handler 异常恢复动态上下文却不保证 `self.indexing` 清空。答案均可仅凭正文推出。
- **事实边界：** 实验 12 本轮通过，支持 CPU pointwise 一次重放和 reduction 一次 scalar/两次 vector 重放及数值；复合操作分支来自当前源码推演，未冒充 GPU 实测。实际读者尚未作答。

下一项 Q421：在 `04-codegen/02-backends/01-backend-source/` 建立 CPU/Triton 索引与 kernel 生成叶子计划。
