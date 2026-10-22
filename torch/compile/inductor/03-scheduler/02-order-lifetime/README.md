# 排序、删除与生存期：让依赖图变成可执行顺序

[上一项：节点与依赖](../01-dependencies/README.md) · [返回 Scheduler](../README.md) · [返回总览](../../README.md) · 下一项：融合判断与代价

本篇的输入是 scheduler nodes、未满足依赖和使用者。Scheduler 先按依赖排序，删掉不会影响结果的节点，再计算每个节点的祖先。融合阶段使用这些结果形成最终节点；随后 Scheduler 才在最终节点上找出每个 buffer 的最后一次使用。

这里“使用”具体指某个后续节点的读记录引用了该 buffer。例如 op1 和 op2 都读取 buf0，而最终顺序是 `op0,op1,op2`，那么 op2 是 buf0 的最后一个读取者。Scheduler 把这个事实记在 op2 的 `last_usage` 中，交给 Codegen 使用。

```text
节点 + 依赖
    │
    ├─ 拓扑排序：生产者必须在消费者之前
    ├─ 反向删除：无有效使用者且无副作用的节点退出计划
    └─ 祖先集合：记住所有直接、间接前驱
    │
    ▼
融合及节点整理（由融合阶段完成）
    │
    ▼
反向扫描最终节点：记录每份数据的最后使用位置
    │
    ▼
最终节点序列 + 每个节点的 last_usage
```

mutation 的逻辑版本名贯穿融合前的排序、删除和祖先计算，保证“同一块输入存储”不会在依赖图中形成自环。融合完成后，`mutation_real_name` 又让最后使用记录对应到真实存储名。

## 拓扑排序只规定必要的先后

用三个不能作为普通点式循环融合的 mm 观察排序：

```python
shared = torch.mm(x, first_weight)       # op0 -> buf0
left   = torch.mm(shared, left_weight)   # op1 -> buf1
right  = torch.mm(shared, right_weight)  # op2 -> buf2
return left, right
```

依赖关系是：

```text
             ┌─> op1：读 buf0，写 buf1
op0：写 buf0 ┤
             └─> op2：读 buf0，写 buf2
```

`topological_sort_schedule()` 对每个节点递归访问它的 `unmet_dependencies` 所指向的生产者，然后才把当前节点加入结果。输入 `[op0, op1, op2]` 时输出不变；输入次序改为 `[op2, op1, op0]` 时，结果为：

```text
[op0, op2, op1]
```

这说明两件事：op0 一定先于两个消费者；op1 与 op2 互不依赖，所以两者谁先谁后不是依赖图规定的，稳定顺序可以随候选输入改变。拓扑序不是给所有节点排出唯一名次，只兑现依赖要求。

mutation 案例则有真实的强制顺序：

```python
old = x + 1   # op0 读取旧 x
x.copy_(y)    # op1 覆盖 x
```

op1 的 `WeakDep(buf0)` 指向 op0，因此即使把候选列表反转，排序仍恢复 `[op0, op1]`。

## mutation 用新名字表示新逻辑版本

写回前后物理上仍是调用者的 x，但依赖图必须区分“旧 x”和“写回后的 x”。否则 op1 一边依赖 x、一边产生 x，会像自环，后续读取也无法判断应看旧值还是新值。

本例在依赖构造后得到：

```text
op0 读取 arg0_1，产生 buf0（old）
op1 读取 y，产生 buf1；buf1 声明 mutations=['arg0_1']

mutation_renames  = {'arg0_1': 'buf1'}
mutation_real_name = {'buf1': 'arg0_1'}
```

两张表方向不同：

- `mutation_renames` 回答“依赖图里，原名现在对应哪个最新版本”。op1 以后再读 `arg0_1`，依赖会改成读 `buf1`。
- `mutation_real_name` 回答“这个逻辑版本实际属于哪块原始存储”。`buf1` 最终仍写回 `arg0_1`，last usage 也应按真实存储归并。

Python 中连续写两次 `copy_` 的案例进入 Scheduler 后，中间计算直接读取第一次写入的源 `y`，只有最后一次写回保留成 mutation operation。因此 Scheduler 处理的是优化后的 IR，不是按 Python 语句数机械建立 mutation 版本。

具体请求是：

```python
before = x + 1
x.copy_(y)
middle = x * 2
x.copy_(z)
return before, middle, x
```

当前上游把 middle 改写为直接读取 y，并消除第一次独立写回 operation，留下：

```text
op0：读旧 arg0_1，写 before=buf0
op1：读 arg1_1(y)，写 middle=buf1
op2：读 arg2_1(z)，写 buf2，mutations=[arg0_1]
```

因此版本表只有 `arg0_1→buf2`，而不是 `arg0_1→第一次写回→buf2`。op2 的 ancestors 只有 `{op0}`：它必须晚于旧 x reader，却不依赖 middle 的 buf1。op1 与 op2 在依赖图上是兄弟，所以合法顺序既可以是 `op0,op1,op2`，也可以是 `op0,op2,op1`。这组状态说明拓扑关系由优化后的 IR 决定，而不是逐字复刻 Python 语句顺序。

## 死节点删除为什么从后往前

一个节点的结果没有有效使用者，而且节点本身没有副作用，它就不会影响程序输出。`dead_node_elimination()` 按拓扑序的反方向扫描，这样检查生产者时，它的消费者是否已被删除是已知的。

看下面四个节点：

```text
dead_parent -> dead_leaf -> 无使用者
effect      -> 无使用者，但 has_side_effects=True
live        -> OUTPUT
```

这里的 effect 具体表示一次会修改外部可见状态的写入，例如把新值 `copy_` 到调用者输入。即使没有节点读取它的返回值，删掉它也会让调用后的输入保持旧值，因此必须保留；它附带但无人使用的临时输出仍可不分配。

反向过程如下：

| 检查节点 | 输出使用者 | 决定 |
| --- | --- | --- |
| `live` | `OUTPUT` | 保留 |
| `effect` | 无 | 输出 buffer 标为无用，但节点因副作用保留 |
| `dead_leaf` | 无 | 删除 operation 和输出 `tmp1` |
| `dead_parent` | 使用者 `dead_leaf` 已删除 | 连带删除 operation 和 `tmp0` |

最终只剩 `effect`、`live`，删除集合为：

```text
removed_operations = [dead_leaf, dead_parent]
removed_buffers = [effect_out, tmp0, tmp1]
```

“保留有副作用节点”和“保留它的无用输出”是两回事：`effect` 必须执行，但 `effect_out` 可以不分配。前面的三个 mm 都通向输出，所以不会被删除。

弱依赖不会单独挽救一个死节点。一个 user 若是 weak，删除判断把它视为可消除；删除结束后，剩余节点再调用 `prune_weak_deps()`，去掉指向已删 operation 的 `WeakDep`。WeakDep 只保护仍然存在的旧值读取，不强制已经无用的读取继续物化。

## 祖先集合把整条前驱链压成查询表

拓扑排序后，`compute_ancestors()` 从前往后处理节点。当前节点的祖先等于每个直接依赖生产者，加上那些生产者已经算出的祖先。

三个 mm 的结果是：

| 节点 | 直接依赖 | `ancestors` | 原始顺序范围 |
| --- | --- | --- | --- |
| op0 | 无 | 空 | `[0,0]` |
| op1 | op0 | `{op0}` | `[1,1]` |
| op2 | op0 | `{op0}` | `[2,2]` |

若再有 op3 读取 op1，op3 会取得 `{op1, op0}`。融合判断由此能快速发现一对候选之间是否夹着必须先执行的节点；`min_order/max_order` 则保留节点或融合组在原始计划中的位置范围。[融合篇](../03-fusion/README.md)继续使用这些信息。

## `last_usage` 是怎样反向算出的

生存期要回答的不是“谁第一次产生 buffer”，而是“走到哪个节点以后，后面再也不会使用它”。`compute_last_usage()` 从最终节点反向扫描，维护 `future_used_buffers`：

1. 初始放入图输出，因为它们离开编译函数后仍要存活。
2. 当前节点用到的 buffer 减去未来已经会用的 buffer，得到当前节点的 `last_usage`。
3. 把这批最后使用过的名字加入未来集合，继续看更早的节点。

三个 mm 的图输出为 `buf1`、`buf2`。反向代入：

| 反向检查 | 本节点使用的非输出 buffer | 后面是否还会用 | `last_usage` |
| --- | --- | --- | --- |
| op2 | `buf0`、`arg3_1` | 否 | `buf0`、`arg3_1` |
| op1 | `buf0`、`arg2_1` | `buf0` 已在 op2 使用 | `arg2_1` |
| op0 | `arg0_1`、`arg1_1` | 否 | `arg0_1`、`arg1_1` |

因此共享中间结果 `buf0` 不能在 op1 后释放：op2 还需要它。它被标在 op2 的 `last_usage`，表示 op2 完成后才进入可释放集合。两个输出没有出现在任一节点的 `last_usage`，因为初始未来集合保护了它们。

mutation 案例还会先用 `mutation_real_name` 把版本名映回真实存储。op0 的 `last_usage=[]`，op1 的为 `arg0_1,arg1_1`：旧 x 的读取与写回属于同一真实输入存储，不能把中间逻辑版本当成独立存储。`compute_last_usage()` 到这里完成本篇出口：它把记录附在最终 scheduler nodes 上，再交给 Codegen。

## 用新图推算

给定一条活路径 `A → B → D`，另有一条不通向输出的支路 `A → C`，并且只有 D 是输出：

- 拓扑排序必须让 A 在 B、C 之前，并让 B 在 D 之前；B、C 的相对次序不固定。
- C 没有有效使用者且没有副作用，因此被删除；A 仍因 B 使用而保留。
- 删除以后，D 的祖先是 `{B,A}`，不包含已经退出计划的 C。
- 融合和节点整理完成后，A 的输出由活路径上的 B 最后使用。

若 C 是必须执行的副作用节点，它会保留；只要它仍读取 A，A 的最后使用位置就要同时考虑 B 和 C。若 C 改为不再读取 A，它虽然因副作用保留，却不影响 A 的 last usage。生存期看实际读取的 buffer 或 alias，不看节点是否仅因副作用存活。

排序、删除和生存期都从同一组依赖出发，但各自回答不同问题：能否执行、是否需要执行、何时不再需要数据。

## 继续阅读实现

[排序与 last usage 核心源码：完整方法与状态推演](01-order-source/README.md)
