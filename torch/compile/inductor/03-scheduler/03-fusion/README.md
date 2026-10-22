# 融合：先找共享数据，再证明可以放进同一个生成单元

[上一项：排序与生存期](../02-order-lifetime/README.md) · [返回 Scheduler](../README.md) · [返回总览](../../README.md) · 下一项：后端分派与生成顺序

先看本篇的纵向例子。op0 对每个位置计算 `buf0[i]=x[i]+1`，op1 再算 `out[i]=buf0[i]*2`。分开生成需要第一个循环写完整 buf0、第二个循环再读它；融合后一个循环在同一 i 上先算临时值 `t=x[i]+1`，紧接着写 `out[i]=t*2`。如果 op1 改成读取 `buf0[(i-1) mod 8]`，它需要的就不是 op0 当前 i 的值，不能用这次直接传递替代完整 buf0。

融合的输入是已经按依赖排序、经过死节点删除、并算好 `ancestors` 与 `min_order/max_order` 的活 scheduler nodes。融合不是把相邻 Python 语句直接拼起来，也不是再改一次 FX 图。融合后的 `FusedSchedulerNode` 仍保存原成员，只是告诉下一步：“这些成员应当作为一个生成单元处理。”它要依次回答：哪两项值得尝试；放在一起是否仍保持正确读写和必要先后；目标后端能否生成；当前策略是否值得合并；多个候选冲突时先处理哪一对。

```text
pre-fusion scheduler nodes
       │
       ├─ 按共同使用的 buffer 找候选
       ├─ 正确性与后端能力：方向、设备、依赖、索引和循环能否正确生成
       ├─ 策略判断：共享访问收益、成员规模和峰值内存是否值得合并
       ├─ 候选优先级：backend priority + fusion score
       ├─ 执行前重查：映射到当前融合组、排除成环
       └─ 可选实测：融合 kernel 是否更快
       │
       ▼
SchedulerNode / FusedSchedulerNode 列表
```

判断时使用进入本篇的活节点所携带的四类信息：

| 信息 | 这里具体回答什么 |
| --- | --- |
| `ancestors` | B 是否经依赖路径使用 A，从而区分纵向与横向 |
| `read_writes` | 两边是否访问同一个名字、同一个精确位置和范围 |
| `group` | 目标后端看到的普通循环范围与规约范围能否组合 |
| `min_order/max_order` | 两节点原来相隔多远；融合是否可能拉长生存期 |

“共同使用一个 buffer 名”只负责缩小候选搜索范围；真正的“共享数据”要求依赖对象也相同。例如同样经过 `producer`，按位置 `c0` 读取和按 `(c0-1) mod 8` 读取并不是可以在同一位置省掉的一次访问。

## 纵向融合：消费者直接使用生产者当前位置

输入是两项已经物化的 scheduler 工作：op0 写 `buf0[c0]=x[c0]+1`，op1 读 `buf0[c0]` 并写 `out[c0]=buf0[c0]*2`。

这里假定中间结果已经物化。CPU 融合前有两个同为 8 元素逐点循环的节点：

```text
op0：x + 1，写中间 buffer
op1：读中间 buffer，再乘 2
```

候选记录为：

```text
(op0, op1)
score = (True, True, 32, -1)
backend priority = 0
```

这四个分数从左到右表示：是否符合 template/epilogue 偏好、两边是否同为逐点或同类规约、预计少读写多少字节、两节点在原顺序中相隔多远。这里两个节点都是逐点计算，融合省掉 8 个 float32 中间元素的共同内存访问，记为 32；它们相邻，所以最后一项是 `-1`。排序时元组较大的候选先尝试。`backend priority=0` 是另一层分组：数值越小，越先于 outer-loop 等特殊融合处理。

op1 的祖先包含 op0，因此这是纵向融合。`can_fuse_vertical()` 将 op1 对中间 buffer 的 `MemoryDep` 与 op0 的写依赖比较；两者都是位置 `c0`、范围 8，读写匹配。CPU backend 又确认循环分组兼容，最终得到：

```text
op0_op1: FusedSchedulerNode(members=[op0, op1])
```

生成单元在同一位置先算 `x[c0]+1`，随即乘 2；完整中间张量不需要跨 kernel 保存。输入 `x=[0,1,2,3,4,5,6,7]` 时，结果为 `[2,4,6,8,10,12,14,16]`。

## 横向融合：没有依赖，也可以共享一次循环

横向案例产生两个独立输出：

```python
left = x + 1
right = x * 2
return left, right
```

为了单独观察 Scheduler 融合，假定两个分支的中间结果已经物化，因此 pre-fusion 图有四个活节点：

| 节点 | 读取 | 写入 | `ancestors` |
| --- | --- | --- | --- |
| op0 | `x[c0]` | `left_tmp[c0]=x[c0]+1` | 空 |
| op1 | `left_tmp[c0]` | `left[c0]` | `{op0}` |
| op2 | `x[c0]` | `right_tmp[c0]=x[c0]*2` | 空 |
| op3 | `right_tmp[c0]` | `right[c0]` | `{op2}` |

op0→op1 和 op2→op3 是两条纵向生产—消费边。op0 与 op2 互不为祖先，却都读取 `x[c0]`，因此又产生一对横向候选。

第一轮候选顺序为：

| 候选 | memory score | proximity | 含义 |
| --- | ---: | ---: | --- |
| op0 + op1 | 32 | -1 | 同一分支的纵向相邻项 |
| op2 + op3 | 32 | -1 | 另一分支的纵向相邻项 |
| op0 + op2 | 32 | -2 | 两个分支共同读取 x |

三对都通过。逐次合并的状态变化是：

| 处理候选 | 执行前映射到的当前组 | 合并后 |
| --- | --- | --- |
| `(op0, op1)` | `op0`、`op1` | `op0_op1` |
| `(op2, op3)` | `op2`、`op3` | `op2_op3` |
| `(op0, op2)` | `op0_op1`、`op2_op3` | `op0_op1_op2_op3` |

候选表保存原节点对，但执行每对前，`fuse_nodes_once()` 会用成员的首个名字查询 `name_to_fused_node`。前两次合并已把四个原成员名分别指向两个新组，因此第三对实际合并两个二成员组。最终四个成员进入一个 `FusedSchedulerNode`。这既包含各分支内部的纵向融合，也包含两个分支间的横向融合。

横向融合还要检查它会不会把相距很远的节点绑在一起、延长中间张量生存期。当前启发式在原始顺序距离超过 64 时拒绝，以避免提高峰值内存；CPU backend 还要求循环范围或规约形态兼容，并限制 `cpp.max_horizontal_fusion_size`。

## 候选不是全体节点的笛卡尔积

`get_possible_fusions()` 默认先按 `used_buffer_names()` 分组：只有共同使用某个 buffer 的节点才进入两两检查。这样能覆盖生产者与消费者，也能覆盖共同读取一个输入的横向候选。

启用 `aggressive_fusion` 时，还会把 `group` 相同的节点放在一起检查，即使它们没有共享数据；普通配置下，“没有共享数据”会被当作没有收益的启发式拒绝。规约节点即使开启 aggressive 模式仍不能靠这个条件放宽。

每对只登记一次。模板与 foreach 的融合有方向要求；正向失败时，候选枚举会在适用情况下尝试反向，把 template 放在 producer 位置以形成 epilogue。

## 正确性检查与策略判断解决不同问题

源码中的 `Scheduler.can_fuse()` 同时承载两组判断。第一组决定组合后能否保持正确结果并由后端生成：

1. 不能与自己融合；grouped node、普通 extern、nop 有明确边界。
2. 方向必须符合依赖；template 只能接 epilogue，不能把 template 当消费者随意前移。
3. 显式 `no_fuse_buffer_names`、设备不一致直接拒绝。
4. 纵向融合必须让消费者读取与生产者写入匹配，且不能跨过依赖它们的中间节点。
5. device backend 必须支持相应循环范围、规约和模板组合。

第二组即使在数学上可以生成，也可能基于收益或资源成本拒绝：

1. 普通配置要求存在能省掉的共同精确内存访问；GPU 路径必要时可先尝试循环重排再比较。
2. 成员数不能超过 `max_fusion_size`，避免融合体无限增长。
3. 横向融合不能超过峰值内存距离限制。
4. 候选的 priority、score 以及可选 benchmark 决定冲突顺序和最终收益判断。

候选真正执行前，在第二次 `can_fuse()` 之后另行调用 `will_fusion_create_cycle()`。它不是 `can_fuse()` 内的一道门。早先的合并可能已经把若干节点的祖先集合组合到一个新组里；如果再融合会让一个组通过间接路径依赖自己，就必须拒绝。

## 相同大小不代表索引可融合

索引不匹配案例仍由 op0 写 `buf0[c0]=x[c0]+1`，但 op1 改为按滚动索引读取 `buf0[(c0-1) mod 8]`。

两个节点都是 8 个元素，但消费者输出位置 0 要读 producer 位置 7，输出位置 1 读 producer 位置 0。消费者的读取索引与生产者的同位置写入不相等。

这个 CPU 案例的判定是：

```text
reject op0+op1: no shared data
final_nodes = [op0, op1]
```

这里“no shared data”是融合收益判断中的术语：两个节点在名称层面确实通过同一中间 buffer 相连，但精确 `MemoryDep` 不相等，`score_fusion_memory()` 找不到能在同一位置省掉的共同访问。保留两节点可先完整生成 producer，再按滚动索引读取，数值为 `[8,1,2,3,4,5,6,7]`。

这就是为什么只比较 tensor 大小或节点相邻关系不够。

## extern 和规模限制形成明确边界

`sin(mm(x, weight))` 在当前 CPU 配置中是：

```text
op0 = ExternKernelSchedulerNode(aten.mm)
op1 = SchedulerNode(aten.sin)
```

判定返回 `node1 is extern or nop`，最终仍是两个节点。普通 extern 调用没有可内联的循环体；模板是另一类表示，只有满足模板 epilogue 条件时才允许融合。

若纵向逐点读写完全相同，但把 `max_fusion_size` 设为 1，两个单节点合并后将有两个成员，判定返回 `exceeds max fusion`。输出数值仍正确，但生成计划保留两节点。该限制是编译规模和生成复杂度的策略，不是数学正确性要求。

## 多个候选怎样排序

先由 backend 的 `get_fusion_pair_priority()` 分组，只保留最小整数的最高优先级组。普通融合把两个节点的完整循环体合在一起；CPU 的 outer-loop fusion 则只合并外层循环，例如两个循环都遍历第 0、1 行、但每行内部处理方式不同，它们仍可共享“遍历两行”这一层。后者返回 1，普通融合返回 0，因此同一轮先尝试能直接合并完整循环体的普通融合。

组内再按 `score_fusion()` 从大到小排序。当前四元组依次表达：

```text
(
  template/epilogue 配置偏好且确实节省内存访问,
  两边规约类型一致且确实节省内存访问,
  估计省下的共同内存访问字节数,
  原始顺序接近程度的负数
)
```

对 8 个 float32 元素，共同依赖估计为 `8×4=32` 字节。相邻候选距离 1，proximity 为 -1；跨一个节点距离 2，得到 -2。因此分支案例先尝试 `(32,-1)` 的候选，再尝试 `(32,-2)` 等距离更远的候选。

这个分数用于解决候选冲突，并不是运行时间预测。它优先节省更多内存往返，再偏向原计划中更接近的节点。

## `can_fuse` 通过后还可能不融合

`fuse_nodes_once()` 对排序后的候选再次取得其当前融合组，重新检查合法性与环，再调用 `speedup_by_fusion()`。

- `benchmark_fusion=False` 时，非 multi-template 候选直接通过收益检查。
- 当前实现对 CPU 即使启用 benchmark 也直接允许，不做 kernel 实测。
- 支持的 GPU/Triton 路径可分别编译两项和融合项；只有融合耗时小于两项之和才接受，同时拒绝寄存器溢出等失败。
- multi-template epilogue 会在候选实现之间比较融合耗时，并据此固定 template choice。

是否测量融合前后的 GPU kernel 由 `benchmark_fusion` 控制。没有测量时，接受表示候选通过了正确性、静态收益策略和后端能力检查。

把整个状态机连起来，顺序是：

```text
按 buffer（可选再按 group）枚举原节点对
  → 第一次 can_fuse，留下静态合法候选
  → backend priority 只保留最优先的一组
  → score 从高到低排序
  → 对每一对映射到当前融合组
  → 第二次 can_fuse
  → will_fusion_create_cycle
  → speedup_by_fusion
  → backend.fuse，并更新每个成员名指向的新组
  → 一轮末按 min_order 排列、拓扑排序、删冗余依赖
  → 节点数仍减少时开始下一轮，最多十轮
```

候选表只是待尝试列表。真正的出口是多轮结束后的调度节点列表；它已经纳入执行时的当前分组、成环检查和收益判断。

## 用新案例判断

看到两个节点 A、B 时，可以按以下顺序判断：

1. 它们共同读写哪个 buffer，精确索引是否匹配？
2. B 是否依赖 A；若是，消费者的每项未满足依赖能否在融合体内得到？
3. 若互不依赖，横向融合会不会延长生存期或循环范围不兼容？
4. 是否碰到 extern、template 方向、设备、mutation 或显式禁融等正确性边界？
5. backend 是否支持该循环组合？成员数、峰值内存和共享访问收益是否符合当前策略？
6. 多个候选中 priority 与 score 排在哪里？前面已完成的融合会不会让本次组合成环？收益策略是否最终接受？

回答完这些问题，才能从“看起来可以一起算”走到当前 Scheduler 实际会不会生成一个融合节点。

## 继续阅读实现

[融合主循环源码：候选、合法性、代价与循环状态](01-fusion-source/README.md)
