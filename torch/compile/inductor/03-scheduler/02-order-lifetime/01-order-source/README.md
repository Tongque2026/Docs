# 排序、删除与最后使用源码：依赖图怎样变成可执行顺序

[返回排序与生存期篇](../README.md) · [返回 Scheduler](../../README.md) · [返回总览](../../../README.md)

Q311 建成的依赖边还不是最终节点列表。Scheduler 必须先保证每个 producer 位于 consumer 之前；随后才能从尾部删除无用链、递推祖先并判断各 buffer 的最后使用位置。

本文沿用四类状态：

| 状态 | 含义 | 本文怎样使用 |
| --- | --- | --- |
| `node.unmet_dependencies` | 当前节点仍需等待的 producer 名；这是从 consumer 指向 producer 的正向依赖 | DFS 排序、祖先闭包 |
| `buffer.users` | 哪些节点消费这个结果；这是从 producer 结果指向 consumers 的反向关系 | DCE 判断结果是否仍活 |
| `removed_operations/removed_buffers` | DCE 已判定不再执行或不再保留的名字 | 裁 WeakDep；后续 codegen 和内存规划跳过删除项 |
| `node.last_usage` | 按最终调度顺序看，本节点是哪些名字的最晚接触者 | 后续累计可释放候选；真正 free/reuse 仍由内存规划决定 |

`get_buffer_names()` 返回一个 scheduler node 产生的 buffer 名，用于把依赖名找到 producer；`get_outputs()` 返回带 users 的 `SchedulerBuffer` 对象。`NodeUser` 是一条反向使用记录，`OutputNode` 是代表“结果要离开编译函数”的虚拟 user。`WeakDep` 只安排 mutation 晚于旧 reader，不传递数值；MemoryDep/StarDep 则代表真实数据或整项结果需要。

`BaseSchedulerNode` 表示一项待调度 operation；它的 `read_writes.reads/writes` 是完整访问集合。`FusedSchedulerNode` 把多个原节点按 `snodes` 顺序装成一个调度节点。`name_to_buf[name].defining_op` 从内部 buffer 名回到生产它的 operation。`mutation_real_name[logical_version]=original_name` 则把 mutation 产生的逻辑版本名还原为最初被写回的名字。

这些方法在构造 Scheduler 时不是任意排列。当前实现先执行：

```text
compute_dependencies
    -> 第一次 topological_sort_schedule
    -> dead_node_elimination（末尾同时 prune_weak_deps）
    -> compute_ancestors
    -> 建立 foreach 等组合节点
    -> 第二次 topological_sort_schedule
    -> 可选的 pre-fusion custom pass
    -> 融合、合并循环和其他分组处理（其中还会按需再次排序）
    -> compute_last_usage
```

第一次排序为反向 DCE 提供“consumer 一定在 producer 后面”的前提；DCE 删除旧 reader 后立刻裁掉相应 WeakDep，祖先集合才不会包含已经消失的节点。组合节点改变节点集合后再次排序，随后才进入融合。`compute_last_usage` 放在融合和分组处理之后，所以它既能处理普通节点，也能处理 `FusedSchedulerNode` 的外层与组内生命周期。

| 方法 | 前置状态 | 写入/返回 | 紧接着的消费者 |
| --- | --- | --- | --- |
| `topological_sort_schedule` | 无环依赖图、每个集合内 buffer 名有唯一 producer | producer 在 consumer 前的节点列表 | 反向 DCE；组合后第二次排序服务融合 |
| `dead_node_elimination` | 已拓扑排序的 nodes 与完整 users | 幸存 nodes、removed sets，并裁失效 WeakDep | ancestors、codegen/内存规划的删除过滤 |
| `compute_ancestors` | DCE 后的拓扑序 | ancestors 与初始 min/max order | 后续融合合法性和顺序跨度判断 |
| `compute_last_usage` | 融合完成后的最终节点序 | 外层及 fused 内部 last_usage | Scheduler 的可释放候选累计；Q431 展开最终 free/reuse |

这些方法不是事务。拓扑排序只构造局部 `name_to_node/seen/result`，非法环会产生错误顺序但不会改写 scheduler node；后面的 DCE、ancestors 和 last usage 会逐项改写图或节点字段，源码没有为异常保存快照或回滚。当前调用链也没有捕获这些异常后重建同一个 Scheduler：异常会向上传播；而 `Scheduler.__init__` 早期已经令 `V.graph.scheduler` 指向自身，因此调试时甚至可能看到部分初始化对象。失败后的这些半成品只能用于定位，不能当作一个已完成阶段继续交给下一消费者。

## K01：拓扑排序只固定必要先后，不强造兄弟关系

当前方法使用深度优先搜索，不使用入度或 ready queue。它接收一份候选节点列表，先建立“每个输出 buffer 名→生产节点”，再按候选顺序深度优先访问；访问 consumer 时，递归放入其尚未满足依赖的 producer。下面跟踪的是递归调用栈、`seen` 和 `result`。

以输入 `nodes=[op2,op1,op0]` 为例：`op0` 写 buf0，op1 和 op2 都读 buf0。类型标注 `List[BaseSchedulerNode]` 就是这三项组成的 Python 列表。`name_to_node={buf0:op0,...}`；`seen` 和 `result` 起初为空。访问 op2 时先递归访问 unmet 指向的 op0，于是 result 先加入 op0，再加入 op2；之后 op1 加入，最终为 `[op0,op2,op1]`。

```python
def topological_sort_schedule(
    self, nodes: List[BaseSchedulerNode]
) -> List[BaseSchedulerNode]:
    """
        Ensure nodes is in topologically sorted order
        """
    seen: OrderedSet[BaseSchedulerNode] = OrderedSet()
    name_to_node: Dict[str, BaseSchedulerNode] = dict()
    result: List[BaseSchedulerNode] = []

    def visit(n: BaseSchedulerNode) -> None:
        if n not in seen:
            # 教学：先标 seen，防止共享 producer 被重复加入。
            seen.add(n)
            for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                # We only care about doing toposort within `nodes`
                if dep.name not in name_to_node:
                    # 教学：graph input 或集合外节点不由本次局部排序负责。
                    continue
                visit(name_to_node[dep.name])
            # 教学：所有集合内 producer 已放入 result，最后才追加当前节点。
            result.append(n)

    for node in nodes:
        for name in node.get_buffer_names():
            name_to_node[name] = node
    for node in nodes:
        visit(node)
    return result
```

### 分叉图逐步执行

实验 09 的 three-mm 图是：

```text
op0 写 buf0
├─ op1 读 buf0，写 buf1（图输出）
└─ op2 读 buf0，写 buf2（图输出）
```

候选顺序 `[op2,op1,op0]` 时：

| 调用 | seen 变化 | result 变化 |
| --- | --- | --- |
| `visit(op2)` | 加 op2 | 发现 buf0，递归 op0 |
| `visit(op0)` | 加 op0 | 无集合内 producer，追加 op0 |
| 返回 op2 | 不变 | 追加 op2 |
| `visit(op1)` | 加 op1 | op0 已 seen，不重复追加；追加 op1 |
| `visit(op0)` | 已 seen | 不变 |

结果为 `[op0,op2,op1]`。op0 必须先于两名 consumer；op1 与 op2 之间没有边，输入候选的相对顺序因 OrderedSet/顺序遍历而稳定保留。

正常候选 `[op0,op1,op2]` 得到 `[op0,op1,op2]`。实验 09 两种结果都已实际调用当前方法验证。这说明拓扑序不唯一；正确性要求满足依赖，不要求为无边兄弟发明统一顺序。

### 依赖排序与节点候选排序是两层决定

一个节点有多条 unmet dep 时，方法按 `dep.name` 排序后递归，使同一节点内部 producer 访问顺序稳定。但顶层兄弟仍按传入 `nodes` 顺序访问。稳定性来自两处明确顺序，不等于图上新增了依赖边。

### 这个方法要求输入已经是无环依赖图

`seen` 在递归 producer 之前写入，它的用途是去重，不是检测递归栈中的环。若错误输入为 A 等 B、B 又等 A：

```text
visit(A)：seen={A}，递归 B
visit(B)：seen={A,B}，再次看到 A 时直接返回
append(B)，再 append(A)
```

方法会返回 `[B,A]`，却不会报告 A↔B 的矛盾，也不可能同时满足两条边。因此这里的前置条件是 Q311 建出的依赖关系为 DAG；环诊断不在本方法中完成。类似地，`name_to_node[name]=node` 假定集合内每个 buffer 名只有一个 producer；若两个节点报告同名输出，后者会覆盖前者，当前方法也不验证这种非法输入。

这两种非法输入不会留下“改了一半的 scheduler node”。本方法只改局部变量 `name_to_node`、`seen` 和 `result`：环会得到不满足全部依赖的返回顺序，重复 producer 会让后写入的节点覆盖前一个映射。方法既不抛出专门的校验错误，也不修改 `self.nodes`；当前调用链中也没有紧接着统一检查“结果仍是 DAG、每个 buffer 只有一个 producer”的步骤。

### mutation 的 WeakDep 在这里真正变成先后顺序

沿用 Q311 的旧读案例：

```text
op0：读旧 arg0_1，写 buf0          # old = x + 1
op1：写回 x，对 buf0 有 WeakDep     # x.copy_(y)
```

即使候选列表是 `[op1,op0]`，`name_to_node` 仍先建立 `buf0 -> op0`。随后访问 op1：

```text
seen={op1}
op1.unmet 中发现 WeakDep(buf0)
    -> visit(op0)
    -> result=[op0]
返回 op1
    -> result=[op0,op1]
```

因此 old 对旧 x 的读取排在覆盖之前。WeakDep 只约束这两个节点，不要求其他无关 sibling 采用固定顺序。若 op0 后来因没有真实消费者而被 DCE 删除，K02 的裁剪会从 op1 移除这条 WeakDep；组合节点建立后，第二次拓扑排序使用的就是裁剪后的依赖集合。

### K01 理解验收

若 op3 同时读取 buf1、buf2，传入 `[op3,op2,op1,op0]`，访问 op3 时会按依赖名递归两条支路；无论先访问哪支，op0 都只追加一次，op1/op2 都在 op3 前。在这个“op3 首次触发两支递归”的案例中，op1 与 op2 的相对次序由依赖名排序决定；若某支已被更早的顶层 `visit` 访问，顶层候选顺序也会影响最终兄弟顺序。两种顺序都不表示兄弟之间新增了依赖。

若 dep 指向 graph input `arg0_1`，它不在 `name_to_node`，方法直接 continue；这不是忽略真实读取，而是本次排序集合中没有需要放到前面的 producer node。

K01 到此完成。下一知识点 K02 从拓扑序尾部反向检查 users，解释 dead chain、side-effect output 和 weak-only user 分别怎样处理。

## K02：反向删除让“consumer 已死”逐级传播到 producer

拓扑序保证 producer 在前、consumer 在后。因此 DCE 反向遍历时，检查某个 producer 之前，它的所有 consumer 已经决定保留或删除；被删除 consumer 的名字已进入 `removed_operations`。

DCE 并不直接从图输出重新做一次可达性搜索。Q311 已经把每个图输出编码成 `OutputNode` user，也把普通消费者写进 `buffer.users`。这里逐个输出判断：

```text
有效强 user：保活这个 buffer
weak user：只为 mutation 排序，不单独保活
已删除 operation 对应的 user：不再保活
users=[]：all([]) 为 True，因此该 buffer 没有活消费者
```

只要一项输出有有效强 user，`active_buffers=True`，无副作用 operation 就仍要执行；所有输出都死且 operation 无副作用时才能删除。副作用是另一类活性根：即使没有活输出，operation 仍保留。

### WeakDep 的裁剪方法

幸存节点最后会调用下面的完整方法：

```python
def prune_weak_deps(self) -> None:
    # Prune weak dependencies on operations that have been removed
    def should_prune(dep: Dep) -> bool:
        if not isinstance(dep, WeakDep):
            return False
        # 教学：WeakDep.name 是旧 reader 的输出 buffer；先找其 defining operation。
        op = self.scheduler.name_to_buf[dep.name].defining_op
        return op.get_name() in V.graph.removed_operations

    to_remove = OrderedSet(
        dep for dep in self.read_writes.reads if should_prune(dep)
    )
    # 教学：重新安装删边后的 ReadWrites，同时重新计算 unmet。
    self.set_read_writes(self.read_writes.remove_reads(to_remove))
```

只有 `WeakDep` 会因 defining operation 已删除而裁掉；普通 MemoryDep/StarDep 表示真实数据需要，不能用这个规则删除。

### dead_node_elimination 完整实现

先固定四个节点的状态：`op0→op1`，op1 的输出无人使用且两者都无副作用；op2 是 `copy_` 一类会修改外部可见数据的节点，虽无输出使用者但 `has_side_effects=True`；op3 的输出是图结果。`self.nodes` 已按 `[op0,op1,op2,op3]` 排好，removed 集合起初为空。反向遍历先保留 op3、op2，再删除 op1；处理 op0 时它唯一的 user op1 已进入 removed 集合，因此 op0 也删除。

```python
def dead_node_elimination(self) -> None:
    """
        Remove any nodes without users
        """
    # self.nodes is in topological order, so by iterating in reverse order
    # we have visited (and potentially removed) all users before visiting a
    # given node.
    updated_nodes = []
    for node in reversed(self.nodes):

        def can_eliminate_user(user: NodeUser) -> bool:
            # 教学：weak user 不单独保活；已删除 operation 也不再是有效 consumer。
            return user.is_weak or user.get_name() in V.graph.removed_operations

        active_buffers = False
        for buf in node.get_outputs():
            can_eliminate = all(can_eliminate_user(u) for u in buf.users)
            if can_eliminate:
                log.debug("removed dead buffer: %s", buf.get_name())
                V.graph.removed_buffers.add(buf.get_name())
            else:
                active_buffers = True

        # 教学：节点同时满足“无副作用、没有活输出”才可删除。
        can_eliminate = not node.has_side_effects() and not active_buffers

        if not can_eliminate:
            updated_nodes.append(node)
        else:
            # dead code
            log.debug("removed dead operation: %s", node.get_name())
            V.graph.removed_operations.add(node.get_name())

    self.nodes = list(reversed(updated_nodes))

    # Prune any WeakDeps no longer needed
    for node in self.nodes:
        node.prune_weak_deps()
```

这个方法的修改不是最后一次性提交。反向扫描每经过一个 dead buffer 或 operation，就立即把名字写入 `V.graph.removed_buffers` 或 `removed_operations`。若扫描中途报错，这两个集合已经含有前面处理过的结果，而 `self.nodes` 仍是调用前的列表，因为替换列表的赋值还没执行。

反向扫描全部成功后，方法先用幸存节点替换 `self.nodes`，才逐个调用 `prune_weak_deps()`。因此裁弱边时出错，`self.nodes` 已经是删除后的列表，removed 集合也已写完，但排在出错节点之后的幸存节点还没有裁边。`prune_weak_deps()` 自身先在局部变量 `to_remove` 中查找待删弱边；若缺少 `name_to_buf[dep.name]`，当前节点尚未改变。只有这一步全部成功，`set_read_writes(...)` 才会把新访问集合和重新计算的 unmet dependencies 写回节点。

源码没有保存旧列表、旧 removed 集合或旧 `read_writes`，调用方也没有捕获异常后恢复同一个 Scheduler。这里若失败，异常向上传播；这些部分更新的字段只能用于定位错误，不能继续当作 DCE 已经完成。

buffer 与 operation 是两层决定。一个 side-effect node 即使所有输出都死，operation 仍需执行；其输出 buffer 却可以进入 `removed_buffers`，避免为无人读取的返回值保留存储。

### controlled DCE 的四个节点

实验 09 直接调用当前 `Scheduler.dead_node_elimination`，构造：

```text
dead_parent(tmp0) -> dead_leaf(tmp1) -> 无使用者
effect(effect_out)                    -> 无使用者，但 has_side_effects=True
live(result)                          -> OUTPUT 使用者
```

反向状态为：

| 访问节点 | 输出 users | buffer 决定 | operation 决定 |
| --- | --- | --- | --- |
| live | OUTPUT 非 weak、未删除 | result 活 | 保留 live |
| effect | 空，`all([])=True` | 删除 effect_out | 有副作用，保留 effect |
| dead_leaf | 空 | 删除 tmp1 | 无副作用，删除 dead_leaf |
| dead_parent | 唯一 user 为已删除 dead_leaf | 删除 tmp0 | 无副作用，删除 dead_parent |

实际出口：

```text
remaining = [effect, live]
removed_operations = [dead_leaf, dead_parent]
removed_buffers = [effect_out, tmp0, tmp1]
```

保留 effect 与删除 effect_out 不矛盾：前者保留调用产生的副作用，后者取消无人使用的结果存储。

### weak-only user 为什么不保活 producer

若 `buf0.users=[NodeUser(mutator,is_weak=True)]`，`can_eliminate_user` 直接返回 True。只要 producer 无副作用且没有其他活输出，它可以删除。随后幸存 mutator 的 `prune_weak_deps` 发现 WeakDep(buf0) 的 defining operation 已在 removed 集合中，就从 reads/unmet 移除这条人为顺序边。

这正是 WeakDep 与真实 MemoryDep 的差异：旧 reader 消失以后，mutation 无需为了一个不存在的计算继续等待。

### 多输出节点按“至少一个活输出”保留 operation

循环逐个判断输出。两个输出中一个只有 weak users、另一个被 OUTPUT 使用时，第一个 buffer 可标记 removed，`active_buffers` 仍因第二个输出变成 True，整个 operation 保留。DCE 不要求一个保留的 operation 的所有输出都分配。

### K02 理解验收

问题：一个无副作用节点有两个输出，A 无 users，B 的唯一 user 已被删除。结果怎样？

A 的空 users 与 B 的已删除 user 都满足 `all(can_eliminate_user)`，两项 buffer 均进入 removed；`active_buffers` 保持 False，因此 operation 也删除。

问题：side-effect node 的输出被另一个活节点读取，会发生什么？

输出存在非 weak 活 user，所以 buffer 活；operation 本来也因副作用不可删。两项判断都保留，不会因为“副作用已经足够保活”而误删仍被读取的输出。

K02 到此完成。下一知识点 K03 用拓扑序递推直接与间接 ancestors，并解释 min/max order 此时为何都初始化为当前位置。

## K03：拓扑序让祖先集合可以从前向后一次算完

这里处理的是全体幸存 scheduler nodes，不是 K01 接收任意局部子集的通用排序接口。`set_read_writes` 已从 unmet 中裁掉 graph inputs/constants；DCE 又裁掉指向已删除旧 reader 的 WeakDep。剩余每条 unmet 都应能在 `self.name_to_buf` 找到内部 buffer，其 defining operation 也应位于当前拓扑序更早位置。`compute_ancestors` 不再检查这个契约：若调用方留下集合外 producer 或缺失名字，会在 `name_to_buf` 或 `name_to_ancestors` 查找时失败。

进入源码前先用分叉图代入：`self.nodes=[op0,op1,op2]`，op0 无未满足依赖，op1/op2 的 `unmet_dependencies` 都只有 `buf0`，而 `name_to_buf[buf0].defining_op=op0`。因此处理 op0 后保存 `{}`；处理 op1 或 op2 时，先加入直接祖先 op0，再并入 op0 已保存的空集合，结果都是 `{op0}`。

```python
def compute_ancestors(self) -> None:
    """
        Populate each node.ancestors
        """
    # note self.nodes is topologically sorted
    name_to_ancestors: Dict[str, OrderedSet[str]] = {}
    for node in self.nodes:
        ancestors: OrderedSet[str] = OrderedSet()
        for dep in node.unmet_dependencies:
            # 教学：dep buffer 的 defining operation 是当前节点的直接祖先。
            dep_node_name = self.name_to_buf[dep.name].defining_op.get_name()
            ancestors.add(dep_node_name)
            # 教学：直接祖先已经处理过，可合并它的全部间接祖先。
            ancestors |= name_to_ancestors[dep_node_name]
        name_to_ancestors[node.get_name()] = ancestors
        node.ancestors = ancestors

    for order, node in enumerate(self.nodes):
        # 教学：融合前每个节点的位置范围退化为单点。
        node.min_order = order
        node.max_order = order
```

这里也按节点立即写回。第一个循环每完成一个节点，就已经设置该节点的 `ancestors`；若后面某个节点查不到 `dep.name`，或者查到的 producer 尚未出现在 `name_to_ancestors`，当前节点还没写入，前面节点的结果却会保留。只有第一个循环全部成功，第二个循环才开始依次写 `min_order` 和 `max_order`。源码同样没有在异常时清除已写字段。

### 从分叉扩展到三层有向无环图

先看实验 09 的分叉：op0 写 buf0，op1/op2 都读 buf0。按 `[op0,op1,op2]`：

```text
op0: unmet={}       -> ancestors={}
op1: unmet={buf0}   -> direct op0 + ancestors(op0)={} -> {op0}
op2: unmet={buf0}   -> direct op0 + ancestors(op0)={} -> {op0}
```

实验实际记录与此一致。现在增加 `op3`，让它同时读取 op1 写的 buf1 和 op2 写的 buf2：

```text
处理 buf1：加 op1，再合并 ancestors(op1)={op0}
处理 buf2：加 op2，再合并 ancestors(op2)={op0}
结果：op3.ancestors={op1,op0,op2}
```

OrderedSet 去掉重复 op0，同时保持首次插入顺序。集合表达“所有必须在我之前的 operation”，不是一条唯一执行序列；真实顺序仍来自拓扑节点列表。

### 为什么依赖的祖先一定已经存在

方法明确假设 `self.nodes` 已拓扑排序。当前节点的每条集合内 unmet dep 都指向更早 producer，所以 `name_to_ancestors[dep_node_name]` 已填写。若未先排序，访问晚出现的 producer 会 KeyError；这不是本方法负责自动修复的输入。

### min/max order 先初始化成同一个位置

对 `[op0,op1,op2,op3]`，初值分别为 `[0,0]`、`[1,1]`、`[2,2]`、`[3,3]`。此时每个 scheduler node 尚未表示跨多个原节点的融合组，所以最早与最晚原始顺序相同。

若后续把 op1 与 op3 组成一个 group，`init_group_node` 取组内最小和最大值，得到 `[1,3]`。融合启发式据此计算两个候选区间的距离，例如当前实现的 peak-memory 检查使用：

```python
proximity_score = max(
    abs(node1.min_order - node2.max_order),
    abs(node2.min_order - node1.max_order),
)
return proximity_score > 64
```

它用原顺序跨度阻止距离过远、可能拉长 live interval 的融合。融合节点建立后也按 `min_order` 排回候选列表，再做一次拓扑排序。本文解释初值、group 的 `[min,max]` 合并以及这个距离消费者；其他融合合法性在 Q331 展开。

### K03 理解验收

若 op4 只依赖 op3，它的 ancestors 不只是 `{op3}`，而是 `{op3} ∪ ancestors(op3)`，包含 op0、op1、op2。新增一层不需要重新遍历整张旧图，因为 op3 的闭包已经保存。

若两个依赖名字由同一个 fused defining operation 产生，`ancestors.add` 只保留一个 operation 名；依赖条目数不等于祖先节点数。

K03 到此完成。下一知识点 K04 从图输出建立 future set，反向计算普通与 fused 节点的 last_usage，并用 mutation_real_name 映回真实 storage。

## K04：反向 future 集合把末次使用归给最晚的实际 consumer

### 普通节点先把逻辑版本映回真实存储

`used_or_aliased_buffer_names()` 从本节点全部 reads 和 writes 的名字开始，再沿每个 buffer 的 `get_inputs_that_alias_output()` 递归加入被别名输出引用的输入名：

```python
def used_or_aliased_buffer_names(self) -> OrderedSet[str]:
    used_names: OrderedSet[str] = OrderedSet()
    deps = [
        dep.name
        for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
    ]
    while len(deps) > 0:
        dep = deps.pop()
        used_names.add(dep)
        if V.graph.name_to_buffer.get(dep):
            for alias in V.graph.name_to_buffer[dep].get_inputs_that_alias_output():
                if alias not in used_names:
                    deps.append(alias)
    return used_names
```

因此“使用”在这里包括读、写以及通过 view/alias 间接接触的输入。例如节点读取 view_v，而 `view_v.get_inputs_that_alias_output()` 返回 base，集合会同时包含 view_v 与 base。普通 alias 名不会在这里强行折成一个名字；算法把别名链涉及的名字都纳入生命周期判断。`mutation_real_name` 做的是另一件事：把 `buf1、buf2` 这类连续写回产生的逻辑版本映回原输入名。

先代入 mutation 例的 op1：`used_or_aliased_buffer_names()` 得到 `{buf1,arg1_1}`，`mutation_real_name={buf1:arg0_1}`，更晚节点已经使用的 `future_used_buffers` 为空。映射后当前使用集合变为 `{arg0_1,arg1_1}`，与 future 作差仍是这两项，所以它们都进入 op1.last_usage。

```python
def set_last_usage(
    self, future_used_buffers: OrderedSet[str], mutation_real_name: Dict[str, str]
) -> None:
    # 教学：收集本节点读写或 alias 涉及的全部 buffer 名。
    used_buffers = self.used_or_aliased_buffer_names()
    # 教学：mutation 产生的逻辑版本映回原输入/真实 storage 名。
    used_buffers = OrderedSet([mutation_real_name.get(k, k) for k in used_buffers])
    # 教学：此前反向遍历已见过的 future 集合中不存在，说明本节点是最晚使用者。
    self.last_usage = used_buffers - future_used_buffers
```

对一个普通节点而言，收集名字、做 mutation 映射和差集计算都先发生在局部变量中；只有最后一行才改 `self.last_usage`。所以在最后赋值之前失败，当前节点的旧值不变。不过外层 Scheduler 是反向逐节点调用它的：若较早节点失败，排在它后面的节点已经写好了新 `last_usage`，不会自动恢复。

`last_usage` 不是“本节点用到的所有 buffer”，而是其中尚未在更晚节点出现的部分。若 `mutation_real_name={buf1:arg0_1,buf2:arg0_1}`，节点使用 buf2 时先改写为 arg0_1；两次 mutation 的逻辑版本因此参与同一原输入名的 future/last-usage 判断。普通 alias 则通过上面的递归把相关名字都加入集合，而不是依靠这张 mutation 表。

### Scheduler 从图输出开始反向推进

three-mm 案例的入口是 `self.nodes=[op0,op1,op2]`，图输出 `{buf1,buf2}`，所以反向扫描开始时 `future_used_buffers={buf1,buf2}`。op2 使用 `{buf0,arg3_1,buf2}`，其中 buf2 已受输出保护，差集留下 `{buf0,arg3_1}`；下面循环随后把这两项加入 future，再处理 op1 和 op0。

```python
def compute_last_usage(self) -> None:
    """
        Populate node.last_usage recursively (also for the nodes within a FusedSchedulerNode)
        """

    # 教学：图输出离开当前函数后仍需存活，不能归给内部节点释放。
    future_used_buffers: OrderedSet[str] = OrderedSet(V.graph.get_output_names())

    for node in reversed(self.nodes):
        node.set_last_usage(future_used_buffers, self.mutation_real_name)
        # 教学：把刚确定由该节点结束生存期的名字加入更早节点的 future 集合。
        future_used_buffers.update(node.last_usage)
```

`future_used_buffers` 是本次调用的局部集合，但各节点的 `last_usage` 是对象字段。循环中途失败时，局部集合随异常退出而丢弃，已经处理过的较晚节点仍保留新字段，尚未访问的较早节点仍保留旧字段。因此不能把失败后的节点列表理解成一次完整的生命周期分析结果。

反向时只把 `node.last_usage` 加入 future，而不是无条件加入所有 used buffers。两者在这套递推中达到同样的“已被更晚节点占用”效果：已在 future 的名字无需重复加入；新名字正是 last_usage。

### three-mm：共享 buf0 归给更后的 op2

节点顺序 `[op0,op1,op2]`，图输出为 buf1、buf2：

| 反向步骤 | 进入时 future | 当前使用 | last_usage | 更新后 future |
| --- | --- | --- | --- | --- |
| op2 | `{buf1,buf2}` | `{buf0,arg3_1,buf2}` | `{buf0,arg3_1}` | 加入 buf0、arg3_1 |
| op1 | `{buf1,buf2,buf0,arg3_1}` | `{buf0,arg2_1,buf1}` | `{arg2_1}` | 加入 arg2_1 |
| op0 | 再含 arg2_1 | `{arg0_1,arg1_1,buf0}` | `{arg0_1,arg1_1}` | 完成 |

因此实验实际记录：

```text
op0.last_usage = [arg0_1,arg1_1]
op1.last_usage = [arg2_1]
op2.last_usage = [arg3_1,buf0]
```

buf0 被 op1、op2 共同读取，反向先遇到更后的 op2，所以末次使用归 op2；buf1/buf2 是图输出，从初始 future 起就被保护，不进入任何内部 last_usage。

### mutation：buf1 映回 arg0_1

旧读案例中 `mutation_real_name={buf1:arg0_1}`。op1 的使用集合包含逻辑输出 buf1，映射后变成 arg0_1；最终实验记录 `op1.last_usage=[arg0_1,arg1_1]`，不是 `[buf1,arg1_1]`。这防止后续阶段把代表调用者输入更新的 buf1 当作普通临时 storage 处理。

若随后 buf2 再次写回同一输入，Q311 建表时会继续保存 `mutation_real_name[buf2]=arg0_1`。反向扫描看到较后的 buf2 和较早的 buf1 时，两者规范化后都是 arg0_1：较后节点先把 arg0_1 加入 future，较早节点便不会再次把它列入 last_usage。

### fused 节点要计算两种尺度

```python
def set_last_usage(
    self, future_used_buffers: OrderedSet[str], mutation_real_name: Dict[str, str]
) -> None:
    # Set self.last_usage using the global information
    # This will be used for inter-kernel optimisations
    # 教学：先把整个 fused group 当成一个 kernel，计算跨 kernel 的末次使用。
    super().set_last_usage(future_used_buffers, mutation_real_name)
    # Set self.last_usage on the snodes
    # This will be used for optimisations within the kernel
    # 教学：组内重新从空 future 开始，找融合 kernel 内各原节点的局部末次使用。
    future_used_buffers: OrderedSet[str] = OrderedSet()
    for node in reversed(self.snodes):
        node.set_last_usage(future_used_buffers, mutation_real_name)
        future_used_buffers.update(node.last_usage)
```

外层 `self.last_usage` 处理融合 kernel 与其他 kernel 之间的边界，使用全局 future。各 `snode.last_usage` 只回答“在这个融合 kernel 内，哪一个原节点最后接触该名字”，所以从空集合按组内顺序重新计算；若把外部 future 带进来，一个稍后 kernel 仍使用的名字会从所有内部 snode 的局部结果中消失，组内优化便无法知道本 kernel 内最后接触它的位置。外层结果已经负责保证它不能在 kernel 之间过早释放。

代入一个没有 alias 和 mutation 的融合组 `F.snodes=[A,B]`。A 读取输入 x、写临时量 t；B 读取 t、写图输出 y。融合组之后还有另一个 kernel 使用 x。y 因为是图输出、x 因为被更晚 kernel 使用，都已经在全局 future 中，因此传给 F 的集合是 `{x,y}`：

```text
F 整体使用 {x,t,y}，减去全局 future {x,y}
    -> F.last_usage={t}          # kernel 之间的结果

组内从空 future 重新反向：
B 使用 {t,y}
    -> B.last_usage={t,y}，组内 future 变成 {t,y}
A 使用 {x,t}
    -> A.last_usage={x}          # t 已被更后的 B 使用
```

x 没有进入外层 `F.last_usage`，因为后面的 kernel 仍会使用它；但 A 是融合 kernel 内最后接触 x 的原节点，所以 x 进入 `A.last_usage`。y 也因图输出而不进入外层结果，却仍进入组内 `B.last_usage`。这就是同一个名字在“kernel 之间”和“kernel 内部”得到不同答案的原因。

融合节点的写入顺序也影响失败状态：它先调用父类写好外层 `F.last_usage`，再反向写各个 snode。若处理中间某个 snode 报错，外层结果和已经访问的较后 snode 已更新，尚未访问的较早 snode 未更新；源码没有把这几处字段一起回滚。

这仍不证明 wrapper 已释放某个地址。Scheduler 后续还要累计可释放名字、处理 alias/mutation/must-keep，并由 wrapper 内存规划生成实际 free/reuse 代码；该实现归 Q431。

### K04 理解验收

若把 three-mm 的节点顺序合法地换成 `[op0,op2,op1]`，buf0 的 last_usage 会转到 op1，因为它现在是更后的 consumer；数学结果不变，但生命周期边界随合法调度顺序改变。

若某个 buffer 既是 op2 的输入又是图输出，它从初始 future 中已经存在，op2 不会把它列入 last_usage。图边界外仍有人持有它，当前函数内部没有释放权。

下一篇继续看这些依赖与生存期结论怎样限制节点融合。

## 按产品声明顺序连读七个方法

前面的知识点按“排序→删除→祖先→生存期”建立理解。本节改按 `scheduler.py` 的物理声明顺序排列完整源码，作为叶子范围核验基准；教学注释不改变产品语句。

### BaseSchedulerNode：set_last_usage → prune_weak_deps

```python
def set_last_usage(
    self, future_used_buffers: OrderedSet[str], mutation_real_name: Dict[str, str]
) -> None:
    # 教学：逻辑 mutation 版本先映回真实 storage，再减去未来仍使用的集合。
    used_buffers = self.used_or_aliased_buffer_names()
    used_buffers = OrderedSet([mutation_real_name.get(k, k) for k in used_buffers])
    self.last_usage = used_buffers - future_used_buffers

def prune_weak_deps(self) -> None:
    # Prune weak dependencies on operations that have been removed
    def should_prune(dep: Dep) -> bool:
        if not isinstance(dep, WeakDep):
            return False
        op = self.scheduler.name_to_buf[dep.name].defining_op
        return op.get_name() in V.graph.removed_operations

    to_remove = OrderedSet(
        dep for dep in self.read_writes.reads if should_prune(dep)
    )
    self.set_read_writes(self.read_writes.remove_reads(to_remove))
```

### FusedSchedulerNode：set_last_usage

```python
def set_last_usage(
    self, future_used_buffers: OrderedSet[str], mutation_real_name: Dict[str, str]
) -> None:
    # Set self.last_usage using the global information
    # This will be used for inter-kernel optimisations
    # 教学：先计算融合组在其他 kernel 之间的末次使用。
    super().set_last_usage(future_used_buffers, mutation_real_name)
    # Set self.last_usage on the snodes
    # This will be used for optimisations within the kernel
    # 教学：再从空 future 反向计算组内原节点的局部末次使用。
    future_used_buffers: OrderedSet[str] = OrderedSet()
    for node in reversed(self.snodes):
        node.set_last_usage(future_used_buffers, mutation_real_name)
        future_used_buffers.update(node.last_usage)
```

### Scheduler：DCE → 拓扑排序 → ancestors → last usage

```python
def dead_node_elimination(self) -> None:
    """
        Remove any nodes without users
        """
    # self.nodes is in topological order, so by iterating in reverse order
    # we have visited (and potentially removed) all users before visiting a
    # given node.
    updated_nodes = []
    for node in reversed(self.nodes):

        def can_eliminate_user(user: NodeUser) -> bool:
            return user.is_weak or user.get_name() in V.graph.removed_operations

        active_buffers = False
        for buf in node.get_outputs():
            can_eliminate = all(can_eliminate_user(u) for u in buf.users)
            if can_eliminate:
                log.debug("removed dead buffer: %s", buf.get_name())
                V.graph.removed_buffers.add(buf.get_name())
            else:
                active_buffers = True

        can_eliminate = not node.has_side_effects() and not active_buffers

        if not can_eliminate:
            updated_nodes.append(node)
        else:
            # dead code
            log.debug("removed dead operation: %s", node.get_name())
            V.graph.removed_operations.add(node.get_name())

    self.nodes = list(reversed(updated_nodes))

    # Prune any WeakDeps no longer needed
    for node in self.nodes:
        node.prune_weak_deps()

def topological_sort_schedule(
    self, nodes: List[BaseSchedulerNode]
) -> List[BaseSchedulerNode]:
    """
        Ensure nodes is in topologically sorted order
        """
    seen: OrderedSet[BaseSchedulerNode] = OrderedSet()
    name_to_node: Dict[str, BaseSchedulerNode] = dict()
    result: List[BaseSchedulerNode] = []

    def visit(n: BaseSchedulerNode) -> None:
        if n not in seen:
            seen.add(n)
            for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                # We only care about doing toposort within `nodes`
                if dep.name not in name_to_node:
                    continue
                visit(name_to_node[dep.name])
            result.append(n)

    for node in nodes:
        for name in node.get_buffer_names():
            name_to_node[name] = node
    for node in nodes:
        visit(node)
    return result

def compute_ancestors(self) -> None:
    """
        Populate each node.ancestors
        """
    # note self.nodes is topologically sorted
    name_to_ancestors: Dict[str, OrderedSet[str]] = {}
    for node in self.nodes:
        ancestors: OrderedSet[str] = OrderedSet()
        for dep in node.unmet_dependencies:
            dep_node_name = self.name_to_buf[dep.name].defining_op.get_name()
            ancestors.add(dep_node_name)
            ancestors |= name_to_ancestors[dep_node_name]
        name_to_ancestors[node.get_name()] = ancestors
        node.ancestors = ancestors

    for order, node in enumerate(self.nodes):
        node.min_order = order
        node.max_order = order

def compute_last_usage(self) -> None:
    """
        Populate node.last_usage recursively (also for the nodes within a FusedSchedulerNode)
        """

    future_used_buffers: OrderedSet[str] = OrderedSet(V.graph.get_output_names())

    for node in reversed(self.nodes):
        node.set_last_usage(future_used_buffers, self.mutation_real_name)
        future_used_buffers.update(node.last_usage)
```

这一区域按类和产品声明位置核对七个方法；实际调用顺序以篇首流程图为准，两种顺序用途不同。
