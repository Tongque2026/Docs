# Q132 作者记录：`FakeTensorUpdater`

> 第五轮状态：叶子完成。重新以完整源码、传播状态、alias 对照、跳过/失败边界和当前 dtype 缺口验收。

## 读者、层次与完成目标

- 所在层：深度 3，Q13 的第二个实现叶子。
- 读者前提：已从 Q131 理解拓扑顺序保证 producer 先于 user；知道 FX `node.meta['val']` 保存 FakeTensor 抽象结果。
- 本叶子职责：用结构 hash 找可能变化的 ATen/getitem 节点，重新 fake 执行；只有比较结果发生变化时才写回并点名 users。
- 入口：已经稳定拓扑排序、节点原 meta 大体可用、部分 target/args/kwargs 被 pass 修改的 FX Graph，以及 `V.fake_mode`。
- 正常出口：受支持且 fake 参数完整的变化节点得到更新；必要的下游 users 在同一线性遍历中继续更新。
- 跳过/失败边界：不支持 target、缺 fake 参数、比较判相同会停止；未知非 Tensor 类型触发 assert；符号重绑可能依赖 shape env。
- 承诺深度：完整类和四个直接 helper 源码，hash、类型递归、tensor 属性、storage alias、传播、缓存与当前缺口的具体状态。

## 声明范围与资料

- 产品文件：`/work/pytorch/torch/_inductor/fx_utils.py:55-191`。
- 完整范围：整个 `FakeTensorUpdater` 类，以及紧随其后的 `get_storage`、`get_node_storage`、`get_fake`、`get_fake_args_kwargs`。
- 源码基线：commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 调用顺序证据：当前 `fx_passes/post_grad.py` 先 stable sort，再 `fake_tensor_updater.incremental_update()`，最后 recompile/lint。
- 独立实验：`../../../experiments/03_graph_invariants.py` 与 `graph_invariants.txt`。

## 篇内知识点队列

| 顺序·编号 | 问题、对象与完成条件 | 案例与正文 | 状态 |
| --- | --- | --- | --- |
| 1·K01 | hash 基线怎样发现 target/args/kwargs 容器变化 | add 输入换 c/d、add→sub | 完成并复读 |
| 2·K02 | fake 参数怎样取得，哪些 node 可以直接执行 | helper 源码、missing val、Python target | 完成并复读 |
| 3·K03 | list/tuple/None/符号/Tensor 怎样判相同 | `is_fake_tensor_same` 完整源码与分支解释 | 完成并复读 |
| 4·K04 | shape 变化怎样沿拓扑 users 传播并缓存新 hash | `[2,3]→[4,3]` add→relu | 完成并实测 |
| 5·K05 | 独占 storage 与共享 alias 改变怎样分岔 | add→sub、view→clone | 完成并实测 |
| 6·K06 | 相同 meta、dtype 缺项和跳过路径的后续调用行为 | 两次 sub、add→eq、missing meta | 完成并实测 |
| 7·K07 | unbacked symbol 重绑与 bindings 写回位于何处 | 源码分支与出口边界 | 完成并复读；未单列动态实验 |

当前点：K07 完成；篇内无剩余。根队列下一项 Q211。

## 处理情况覆盖

| 情况 | 关键状态/出口 | 正文位置 | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| 结构 hash 未变、未被点名 | 首条 continue | 完整源码/传播表 | 第二次 shape update 无 op | 叶子机制完成 |
| target/args 改变 | hash miss，进入候选处理 | 传播表与边界 | add 输入、add→sub/eq | 叶子机制完成 |
| 非 ATen/getitem target | should_process false | 边界节 | operator.add 实验 | 叶子机制完成 |
| 输入缺 val | fake args 无效，跳过 | 边界节 | 删除 placeholder val | 叶子机制完成 |
| tuple/list/None | 递归比较类型、长度与成员 | 完整源码 | 源码推演 | 叶子机制完成 |
| SymInt/Bool/Float | shape env 静态判表达式相等 | 完整源码 | 源码推演，未单列动态实验 | 当前源码完成，实验边界已标 |
| tensor shape/layout | 不同则写回传播 | 三步表 | `[2,3]→[4,3]` | 叶子机制完成 |
| stride/offset/device | 不同则写回传播 | storage/迁移题 | 当前源码推演 | 叶子机制完成 |
| 同 storage | alias 不变 | storage 节 | 当前源码 | 叶子机制完成 |
| 两份独占 fresh storage | 视为相同，停止传播 | add→sub | 只执行 sub | 叶子机制完成 |
| 共享 view→fresh clone | alias 变化，点名 user | storage 节 | clone、relu 实测 | 叶子机制完成 |
| fake 判相同 | 不写 hash，后续重试 | 边界节 | sub 连续执行两次 | 叶子机制完成 |
| fake 判不同 | 写 meta/hash，点名 users | 三步表 | 第二次无 op | 叶子机制完成 |
| dtype 改变 | 当前比较漏项，错误停止 | 边界节 | eq 后 meta 仍 float32 | 当前行为已实测并明确 |
| unbacked symbol | rebind 后刷新 path bindings | 完整源码 | 源码证据 | 源码机制完成，未单列动态实验 |

## 独立实验事实

```text
shape change:
  before add/relu = [2,3]/[2,3]
  dispatch = add, relu
  after = [4,3]/[4,3]
  second update dispatch = []

equal abstract contract:
  add target -> sub
  first dispatch = sub
  second dispatch = sub

dtype boundary:
  add target -> eq
  dispatch = eq
  old meta dtype = float32
  updater after dtype = float32  # 新 bool fake 被当前比较判“相同”

missing val:
  target -> sub, delete input val
  dispatch = []

unsupported Python target:
  target -> operator.add
  skipped; old float32 meta remains

alias:
  view -> clone
  dispatch = clone, relu
  old_shared=True, new_shared=False
```

RecordOps 是 `TorchDispatchMode`，记录 updater 真正调用的 ATen overload；shape/dtype/storage 均直接读取更新前后的 FakeTensor。实验不验证 unbacked symbol binding 分支，也未将 dtype 现象解释成期望行为；它是当前源码的可复现限制。

## 源码完整性审计

- 类 docstring、构造器基线循环、hash_node 与 incremental_update 全部保留。
- incremental_update 的未使用 `processed`、storage 计数、三个内部 helper、to_process 主循环与每个 continue/writeback 完整。
- 类型、序列、None、符号值、tensor shape/layout/stride/offset/device/storage 的分支完整。
- `rebind_unbacked`、bindings、storage 新计数、users、processed_hashes 写回完整。
- 四个文件级 helper 的所有语句、docstring 与嵌套 pytree 检查完整。
- 教学注释统一 `# 教学：`；未用省略号替代产品逻辑。

正文代码块已与产品类和四个函数作 AST 对照；每次修改后需要重跑。源码等价不替代分支覆盖和理解题。

## 只凭正文的作者理解验收

1. **为什么 add 输入 shape 改变会让 hash 变化？** FX 为 `node.args` 设置新不可变 tuple，`id(node.args)` 改变；hash miss 后才重新 fake 执行。
2. **为什么 relu 自身 hash 未变仍会执行？** add meta 不同后把 `id(relu)` 放入 to_process，首条 continue 要求 hash 已知且 id 不在 to_process 才跳过。
3. **为什么 add→sub 第二次仍执行？** 第一次新 fake 被判相同，在 continue 前没有把新结构 hash 加入 processed_hashes。
4. **为什么 view→clone 会传播而 add→sub 不会？** view 的旧 storage 与输入共享，计数大于 1；clone 破坏 alias，不能用独占 fresh storage 豁免。add/sub 的新旧输出都是无共享的独占结果。
5. **eq 已产生 bool fake，为什么 meta 仍 float？** 当前 tensor 比较没有 dtype 字段；其余字段与独占 storage 条件通过，误判相同并 continue。
6. **缺 input val 为什么不尝试执行 target？** get_fake 保留该 FX Node；pytree leaves 检出残留 Node，is_valid=false，在 fake mode 调用前退出。
7. **普通 Python operator.add 为什么跳过？** should_process 只接受 call_function 且 target 为 OpOverload 或 operator.getitem。
8. **为什么 incremental_update 必须在 stable sort 后？** 主循环不回头；上游写回后只点名尚未遍历的 users，拓扑顺序保证 user 位于后方。
9. **shape 变化写回后为什么第二次无 dispatch？** 新 hash 已在成功写回末尾加入 processed_hashes，且没有 user 再点名。

答案均由正文源码、状态表和实验推出。实际读者反馈尚无，不能声称读者已经理解。

## 本轮整改与续写

原稿的源码范围较完整，但作者记录错误地称没有实际运行 shape/alias 路径；正文以未采集的 `[2,1,3]` 情景代替真实实验，并未揭示相同 meta 重试或 dtype 漏项。本轮扩展实验，重写案例事实、缓存行为、alias 对照和当前限制，同时保留动态 symbol 分支为源码证据边界。

## 第五轮叶子验收

- **完整源码**：整个类和四个相邻 helper 连续保留；hash、比较、重绑、bindings、storage 计数、users 与 processed_hashes 写回均未省略。
- **分支与出口**：结构未变、target/args 改变、unsupported、missing val、序列/None/符号/Tensor、shape/layout/stride/offset/device/storage、unbacked symbol 均有条件和出口。
- **具体状态**：`[2,3]→[4,3]` 的 add→relu 三步表实际 dispatch 两个 op；第二次无 dispatch；add→sub 展示相同抽象合同重试；view→clone 展示 alias 变化传播。
- **理解迁移**：正文能判断 transpose→连续 x 即使 shape 相同也因 stride 改变而传播；独占 fresh storage 可停止，共享 storage 破坏则继续。
- **当前限制与证据边界**：add→eq 实测暴露 dtype 未比较，旧 meta 错留 float32；动态 unbacked binding 只由源码支持。教程按当前实现记录，不把缺口改写成期望行为。

Q132 第五轮叶子复核完成。全系列下一项 Q211：GraphLowering 输入登记完整源码。
