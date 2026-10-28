# 排序、删除与生存期：从依赖图得到最终计划

[上一篇：节点与依赖](../01-dependencies/README.md) · [下一篇：融合判断与代价](../03-fusion/README.md) · [返回 Scheduler](../README.md) · [返回总览](../../README.md)

上一篇交出的每个 scheduler node 都带着 `ReadWrites`、`users` 和
`unmet_dependencies`，Scheduler 还持有 mutation 的逻辑版本表。本篇会在两个阶段使用
这些状态：

```text
融合前
  1. 拓扑排序：让 producer 和其他前驱排在使用者之前
  2. 反向删除：移除不影响输出、也没有副作用的工作
  3. 祖先与原始位置：把整条前驱链整理成融合判断可查询的状态
        │
        └────────→ 交给 Q33：活节点 + ancestors + min/max_order
                          │
                          ▼
                 融合、分组和最终节点整理
                          │
                          ▼
融合后
  4. 生存期：在最终节点序列上标出各项数据最后被触及的位置
        │
        └────────→ 交给生成循环：final nodes + last_usage
```

所以本篇有两个出口。前三项完成后，融合篇会改变节点的身份和组合；它完成后，Scheduler
再次回到本篇的生存期职责。下面先解释贯穿两次处理的一条名字规则，再按四项职责展开。

## mutation 名字怎样贯穿四项工作

写回前后的物理 storage 仍可能是调用者的同一个 x，依赖图却需要两个逻辑名字，否则“读取
旧 x”和“读取新 x”会混在一起。上一篇已经建立两张表：

```text
mutation_renames   = {arg0_1: buf1}
mutation_real_name = {buf1: arg0_1}
```

- `mutation_renames` 已经把后续依赖改到最新逻辑版本 `buf1`。排序、删除和祖先计算直接消费
  改名后的 `unmet_dependencies` 和 `users`。
- `mutation_real_name` 在生存期阶段把逻辑版本 `buf1` 映回真实 storage 名 `arg0_1`，避免把
  同一块输入 storage 当成两份可独立释放的数据。

Q32 不再新建 mutation 版本；它使用 Q31 已经建立的版本关系。

## 第一项：拓扑排序只兑现必要的先后

用三个 extern mm 观察一张分叉依赖图：

```python
shared = torch.mm(x, first_weight)       # op0 -> buf0
left   = torch.mm(shared, left_weight)   # op1 -> buf1
right  = torch.mm(shared, right_weight)  # op2 -> buf2
return left, right
```

```text
             ┌─→ op1：读 buf0，写 buf1
op0：写 buf0 ┤
             └─→ op2：读 buf0，写 buf2
```

`topological_sort_schedule()` 先沿当前节点的 unmet 关系访问图内 producer，再放入当前节点。
候选顺序为 `[op0,op1,op2]` 时结果不变；将候选反转后，实验得到：

```text
[op0, op2, op1]
```

op0 必须先于两名消费者；op1、op2 之间没有边，所以两者次序不唯一。拓扑排序只保证依赖
成立，并不给所有互不依赖的节点定义唯一名次。

旧读与覆盖则有强制关系：

```python
old = x + 1   # op0 读取旧 x
x.copy_(y)    # op1 覆盖 x
```

上一篇给 op1 增加了 `WeakDep(buf0)`，所以反转候选也会恢复 `[op0,op1]`。

### 坏依赖和环在哪里失败

若一个 unmet 名字既不是已可用输入，也找不到当前节点集合中的 producer，首次 DFS 排序会
跳过这个名字，并不在这里把它猜成外部输入。若含有这项坏依赖的节点没有有效输出或副作用，
它可能先被第二项 DCE 删除；若该节点在 DCE 后仍然存活，第三项建立祖先时还要查询这个
名字的 `defining_op`，届时查找失败，编译不会进入融合。

首次 DFS 也不是一套独立的完整环检测器。依赖环通常会在随后展开祖先时遇到尚未建立的
祖先状态而失败；Scheduler 的其他重排路径还使用一套按入度取节点的拓扑排序，剩余节点
无法取完时会明确报 `Topological sort failed!`。因此合法计划的前提仍是依赖图无环且每条
内部等待都能找到 producer。

## 第二项：从叶子向生产者删除无用工作

拓扑序已经保证消费者在生产者之后。`dead_node_elimination()` 反向扫描时，先判断叶子，
再判断它的生产者，因而能让一整条死链逐步退出计划：

```text
dead_parent → dead_leaf → 无有效使用者
effect                   → 无有效使用者，但 has_side_effects=True
live                     → OUTPUT
```

反向结果是：

| 节点 | 当前输出使用者 | 处理结果 |
| --- | --- | --- |
| `live` | `OUTPUT` | 保留节点和输出 |
| `effect` | 无 | 保留节点；无人使用的 `effect_out` 可标成 removed buffer |
| `dead_leaf` | 无 | 删除节点和输出 `tmp1` |
| `dead_parent` | 唯一 user 已被删除 | 删除节点和输出 `tmp0` |

最终活节点是 `[effect, live]`，删除集合是：

```text
removed_operations = [dead_leaf, dead_parent]
removed_buffers     = [effect_out, tmp0, tmp1]
```

这里的 effect 应理解为真实报告 `has_side_effects=True` 的 operation，例如
`EffectfulKernel`、某些检查或 collective。实验 09 的这个四节点例子直接调用当前 DCE 方法，
但节点本身是最小假对象；它证明方法如何区分副作用节点和无用输出，不证明某个具体 IR 类
一定报告副作用。

普通输入 mutation 通常不靠 `has_side_effects=True` 存活。Q31 已经为外部可见写回的最新
版本登记虚拟 `OUTPUT` user，它因此不会作为死工作删除。这与一般副作用的保留原因不同。

WeakDep 也不会独自救活一个已经无用的 reader。weak user 不算有效使用者；DCE 删除旧
reader 后，剩余节点再裁掉指向该 removed operation 的 WeakDep。它只为仍然活着的旧读取
维持覆盖顺序。

若所有节点都因无使用者且无副作用而删除，活节点列表变成空。祖先计算、融合和最终
last-usage 扫描都接收空列表，不会为已经删除的工作重新建立节点；生成循环也没有节点工作
要发出。

三个 mm 都通向 `OUTPUT`，所以 DCE 后的第一次出口快照仍是：

```text
live nodes = [op0, op1, op2]
removed_operations = []
removed_buffers = []
```

## 第三项：祖先集合保存整条前驱链

DCE 结束后，`compute_ancestors()` 按活节点的拓扑序前进。当前节点的祖先是每个直接
producer，再加上这些 producer 已经拥有的祖先。

三个 mm 得到：

| 节点 | 直接前驱 | `ancestors` | `min_order/max_order` |
| --- | --- | --- | --- |
| op0 | 无 | 空 | `[0,0]` |
| op1 | op0 | `{op0}` | `[1,1]` |
| op2 | op0 | `{op0}` | `[2,2]` |

若 op3 再读取 op1，op3 的祖先就是 `{op1,op0}`。融合判断可以据此发现两名候选之间是否
存在不能跨越的前驱关系。这里的 order 是首次拓扑排序和 DCE 之后、正式融合之前的节点
位置，不是 Lowering operation 的原始发现次序。融合组随后用成员的最小/最大值保存它覆盖
的这段位置范围。

至此形成本篇第一次出口：

```text
交给 Q33：
  live nodes       = [op0, op1, op2]
  ancestors        = {op0:{}, op1:{op0}, op2:{op0}}
  min/max_order    = op0:[0,0], op1:[1,1], op2:[2,2]
  ReadWrites/unmet = 继续随节点存在
```

## 融合和最终整理之后，节点已经可能改变

Q33 接住第一次出口后，Scheduler 还可能建立 foreach 分组、融合普通节点、合并循环、定型
`MultiTemplateBuffer`、重排计算与通信、建立 combo 节点。临时的 grouped node 会在
生存期计算前拆回其包含的节点。因此第四项面对的不是 Lowering 最初的 operation 列表，
而是这些步骤完成后的最终节点序列。

当前 three-mm 案例的三项工作都是 extern mm，没有普通 pointwise 融合；实验中的最终列表
仍是：

```text
final nodes = [op0, op1, op2]
```

这张快照把第一次出口与下面的生存期输入接在了一起。其他图若已经形成 fused、foreach 或
combo node，第四项必须按这些组合后的节点计算。

## 第四项：在最终节点上计算 last usage

`last_usage` 回答的是：走到哪个最终节点以后，一项数据不会再被后面的节点触及。本节点
“触及”的名字来自它的 reads、writes，以及 IR 已声明的 alias 闭包；mutation 逻辑版本还会
通过 `mutation_real_name` 映回真实 storage 名。

Scheduler 从最终节点末尾向前扫描，维护 `future_used_buffers`：

1. 初始放入图输出名，因为它们离开编译函数后仍要存活。
2. 收集当前节点读、写和 alias 触及的名字，映回 mutation 的真实名字。
3. 减去未来节点或图输出仍会触及的名字，剩下的就是当前节点的 `last_usage`。
4. 把这些名字加入 future 集合，继续处理更早的节点。

三个 mm 的输出是 buf1、buf2。反向代入：

| 反向处理 | 当前节点触及、且不是受保护输出的名字 | 后面是否还触及 | `last_usage` |
| --- | --- | --- | --- |
| op2 | `buf0`、`arg3_1` | 否 | `buf0`、`arg3_1` |
| op1 | `buf0`、`arg2_1` | buf0 已由更后的 op2 使用 | `arg2_1` |
| op0 | `arg0_1`、`arg1_1` | 否 | `arg0_1`、`arg1_1` |

buf0 不能在 op1 后结束生存期，因为 op2 仍读取它；它最终出现在 op2 的 `last_usage`。
buf1、buf2 作为图输出已在 future 初始集合中，所以不会出现在内部节点的 last_usage。

input-mutation 案例得到：

```text
op0.last_usage = []
op1.last_usage = [arg0_1, arg1_1]
```

`buf1` 是写回后的逻辑版本，生存期计算先把它映回真实输入 storage `arg0_1`，不会把二者
当成两份独立空间。

### 组合节点同时需要组外和组内生存期

`FusedSchedulerNode` 先把整个融合组当作一个最终节点，计算哪些名字在这个 kernel 之后不再
被后续 kernel 使用；随后又以空的内部 future 集合反向处理各成员，为 kernel 内部优化保留
成员级 last usage。`ForeachKernelSchedulerNode` 使用同一套组合节点逻辑，combo node 也建立
在 foreach 机制之上。临时 grouped nodes 已在这一步之前拆开，不直接接收最终 last_usage。

因此“一个最终节点有 last_usage”不代表只有一层信息：组级记录服务于跨 kernel 生存期，
成员级记录还会被 SIMD/Triton 生成逻辑用于 kernel 内最后使用判断。

## 第二次出口怎样进入生成循环

本篇最终交出：

```text
final nodes + 每个最终节点的 last_usage
```

Scheduler 的生成循环在准备生成一个节点时，把该节点的 `last_usage` 加入待释放名字集合。
真正发出 wrapper free 语句发生在后端 flush、extern 调用完成后的释放点或最终 flush；
`free_buffers()` 还会排除已经 removed、已经 free 或当前 `can_free` 不允许释放的名字。

所以 last_usage 是“从这个节点起可以考虑结束生存期”的编译状态，不等于运行时此刻已经
释放或复用了地址。wrapper 后续的 memory planning 还有自己的 first/last-use 分析；物理
地址复用在 wrapper 篇讲解，不能由 Scheduler 的 last_usage 单独推出。

## 用一张新图推算

给定活路径 `A → B → D`，另有支路 `A → C`，只有 D 是图输出：

- 拓扑序必须让 A 在 B、C 前，B 在 D 前；B、C 的相对次序不固定。
- C 无有效使用者且无副作用，DCE 删除 C；A 仍因 B 使用而保留。
- DCE 后 D 的 ancestors 是 `{B,A}`，不包含 C。
- Q33 若没有改变这条活路径，最终序列上的 B 是 A 输出的最后使用者。

若 C 是 `has_side_effects=True` 的节点，它会保留；只有它仍触及 A 的输出时，A 的
last_usage 才需考虑 C。节点因副作用存活，并不自动表示它读取所有上游数据。

排序、删除、祖先和生存期使用同一张依赖图，却分别回答四个问题：哪些次序合法、哪些工作
需要执行、哪些前驱不可跨越、何时可以考虑结束一项数据的生存期。

完整排序、删除和 last-usage 源码及逐句状态见
[排序与 last usage 核心源码](01-order-source/README.md)。
