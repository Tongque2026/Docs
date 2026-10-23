# 融合主循环源码：候选怎样逐轮变成当前融合组

[返回融合篇](../README.md) · [返回 Scheduler](../../README.md) · [返回总览](../../../README.md)

Q33 已经给出融合条件，本篇打开控制循环。融合并非“一次扫描所有 pair 后结束”：一轮中的 merge 会改变节点组，下一轮可能出现只有新组才能满足的候选。

源码会同时谈“原 operation”和“当前组”。二者先固定如下：

| 本篇名称 | 具体含义 |
| --- | --- |
| 原 operation | Lowering 交出的一个计算节点，例如 `op0`；融合后名字仍用于追踪成员 |
| 当前组 | 此刻作为一个整体参加下一次判断的调度节点；开始时通常只有一个 operation，合并后可成为 `FusedSchedulerNode(op0,op1)` |
| buffer | operation 读写的有名结果，例如 `buf0`；operation 名和 buffer 名是两套名字 |
| `ancestors` | 当前 operation 必须排在其后的全部上游 operation 名；若 `op1` 读取 `op0` 的结果，则 `op1.ancestors` 含 `op0` |
| `unmet_dependencies` | 当前组仍需由组外生产者满足的读取或顺序要求；内部生产者合并进来后，相应要求可被满足 |
| 旧候选端点 | 一轮开始时保存的两个节点；处理到它时，二者可能已经分别属于更大的当前组 |

以 `op0` 写 `buf0[c0]`、`op1` 读同一位置为例：

```text
原 operation：op0 ──buf0[c0]──> op1
祖先关系：op0 ∈ op1.ancestors
当前组开始为：{op0}、{op1}
融合后为：{op0,op1}
```

`SchedulerNode` 表示一个普通当前组，`FusedSchedulerNode` 表示已经含多个 operation 的当前组。`ExternKernelSchedulerNode` 是由外部 kernel 完成的组；template、foreach 和 `GroupedSchedulerNode` 各有专门接口约束，遇到对应门时再说明。

## K01：一轮节点数不再减少时，融合达到固定点

首段输入 `nodes: List[BaseSchedulerNode]` 是当前组列表。以 `[op0,op1]` 为例，op0 写 buf0、op1 同位置读取 buf0；`old_len=2`。`fuse_nodes_once` 处理完这一轮候选后返回含一个 `FusedSchedulerNode(op0,op1)` 的列表，`new_len=1`；控制回到外层，外层随即停止并返回这一项。若没有候选被接受，new_len 仍为 2，外层也停止。

```python
def fuse_nodes(self, nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
    """
    Combine eligible nodes into FusedSchedulerNodes.
    """
    for i in range(10):
        old_len = len(nodes)
        fusion_log.debug(
            "===== attempting fusion (%d/10): %d nodes =====",
            i + 1,
            old_len,
        )
        # 教学：一轮根据当前 nodes 生成候选并执行若干 merge。
        nodes = self.fuse_nodes_once(nodes)
        new_len = len(nodes)
        fusion_log.debug(
            "completed fusion round (%d/10): fused %d nodes into %d nodes\n",
            i + 1,
            old_len,
            new_len,
        )
        # 教学：没有减少说明本轮无 merge；只剩一项也无需再找 pair。
        if new_len == old_len or new_len == 1:
            fusion_log.debug("===== fusion complete (%d iterations) =====", i + 1)
            break
    return nodes
```

`range(10)` 限制重新建立候选表和运行启发式的轮数，控制编译开销；它不是“必须执行十轮”。常见结束有三种：

| 一轮前后 | 判断 | 下一步 |
| --- | --- | --- |
| 2→1 | `new_len==1` | 立即结束 |
| 4→1 | `new_len==1` | 立即结束 |
| 5→3 | 两项都不成立 | 用三个当前组开始下一轮 |
| 3→3 | `new_len==old_len` | 达到固定点，结束 |

实验 10 的 vertical、horizontal、branched 都在一次 `fuse_nodes_once` 中分别完成 2→1、4→1、5→1。这证明这些具体候选无需第二轮；不能据此推断外层循环多余。一次候选表基于本轮开始时的 nodes 建立，若 merge 后形成的新组与未入表节点产生新共享关系，只有下一轮会重新调用 `get_possible_fusions`。

### 为什么比较节点数足以判断本轮是否有融合

本方法的 `fuse_nodes_once` 只接受“两项替换为一项”的 merge，不会拆分或添加独立工作。因此一次成功使总数减少 1；长度不变就表示没有任何 pair 被接受。组合内部成员数可能增长，但外层只需观察列表项数。

### 十轮结束不等于理论上无候选

如果连续十轮每轮都减少节点但仍多于一项，for 循环自然结束并返回当前结果。代码没有在第十轮后额外证明固定点。因此“返回列表”只说明达到停止条件或轮数上限，不能一律表述为所有可能融合已经穷尽。

### K01 理解验收

若节点数依次 `8→6→5→5`，执行三轮：前两轮继续，第三轮因长度不变停止。若依次 `8→7→...` 连续十轮都减少但最终仍有多项，则因上限返回，不是因固定点。

若一轮内部把 A+B 成组，又用旧候选 `(B,C)` 映射成 `(AB,C)` 并成功，节点数可在同一轮减少两次；这属于 K04 的当前组映射机制，不要求外层每次只 merge 一对。

K01 到此完成。下一知识点 K02 逐句展开 `get_possible_fusions`：共享 buffer 分组、aggressive group、方向尝试、priority 过滤和 score 排序。

## K02：先用共享关系缩小 pair，再按 backend priority 和 score 排序

### 候选发现只比较可能有关的节点

先代入 branched 案例的入口状态。实验捕获到的五个节点如下；`c0` 的范围都是 8：

| 节点 | 精确读取 | 精确写入 | `ancestors` | `used_buffer_names()` |
| --- | --- | --- | --- | --- |
| op0 | `arg0_1[c0]` | `buf0[c0]` | `{}` | `{arg0_1,buf0}` |
| op1 | `buf0[c0]` | `buf1[c0]` | `{op0}` | `{buf0,buf1}` |
| op2 | `buf1[c0]` | `buf2[c0]` | `{op0,op1}` | `{buf1,buf2}` |
| op3 | `buf0[c0]` | `buf3[c0]` | `{op0}` | `{buf0,buf3}` |
| op4 | `buf3[c0]` | `buf4[c0]` | `{op0,op3}` | `{buf3,buf4}` |

`used_buffer_names()` 是读名与写名的并集，所以 buf0 组是 op0/op1/op3，buf1 组是 op1/op2，buf3 组是 op3/op4。组内两两比较会产生纵向 pair，也会产生 op1/op3 这类共同读取 buf0 的横向 pair。`seen` 与 `possible_fusions` 起初都为空；同一 pair 若因多个共享名字再次出现就被 `seen` 跳过。

```python
def get_possible_fusions(
    self, nodes: List[BaseSchedulerNode]
) -> List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]:
    """
    Helper to find all legal fusion opportunities, sorted by self.score_fusion()
    """
    possible_fusions = []
    seen: OrderedSet[Tuple[BaseSchedulerNode, BaseSchedulerNode]] = OrderedSet()

    def check_all_pairs(nodes: List[BaseSchedulerNode]) -> None:
        for node1_index, node1 in enumerate(nodes):
            for node2 in nodes[node1_index + 1 :]:
                key = (node1, node2)
                # 教学：同一 pair 可能因共享多个 buffer 多次进入分组，只检查一次。
                if key in seen:
                    continue
                seen.add(key)

                if self.can_fuse(node1, node2):
                    possible_fusions.append(key)
                elif (node2.is_template() or node2.is_foreach()) and self.can_fuse(
                    node2, node1
                ):
                    # foreach fusions and epilogue fusions are order dependent
                    # 教学：只有 template/foreach 的第二项才尝试反向，并保存反向 pair。
                    possible_fusions.append((node2, node1))

    buffer_names_grouping = collections.defaultdict(list)
    for node in nodes:
        for buf in node.used_buffer_names():
            buffer_names_grouping[buf].append(node)
    for node_grouping in buffer_names_grouping.values():
        check_all_pairs(node_grouping)

    if config.aggressive_fusion:
        group_grouping = collections.defaultdict(list)
        for node in nodes:
            group = getattr(node, "group", None)
            if group:
                group_grouping[group].append(node)
        for node_grouping in group_grouping.values():
            check_all_pairs(node_grouping)

    # 教学：backend priority 先丢掉非最高优先级组，再在组内按 score 降序。
    possible_fusions = self.get_possible_fusions_with_highest_priority(
        possible_fusions
    )
    possible_fusions.sort(key=self.score_fusion_key, reverse=True)
    fusion_log.debug("found %d possible fusions", len(possible_fusions))
    return possible_fusions
```

默认路径按 `used_buffer_names()` 分组。它包含当前组读取和写入的全部 buffer 名；`get_buffer_names()` 只取本组产生的 buffer，`read_writes.buffer_names()` 则从精确读写 Dep 中取名。两个节点若没有任何共同 used 名字，默认路径不会进入 `check_all_pairs`。`aggressive_fusion=True` 时，同一 loop `group` 的节点也会两两比较，让没有共享 buffer 的普通逐点节点继续接受 K03 的横向检查。

若 A、B 同时共享 x、y，它们会在两个 grouping 中出现两次；`seen` 使第二次直接跳过。注意 key 有方向：`(A,B)` 与 `(B,A)` 不相等，但这里的组合循环只按列表先后产生正向 key；反向仅在第二项为 template/foreach 且正向失败时显式尝试。

### backend priority 数字越小越优先

本次输入先固定为 `possible_fusions=[(A,B),(C,D),(A,C)]`，三对在同一 CPU device。backend 分别返回 priority 0、0、1，因此循环结束时字典为 `{0:[(A,B),(C,D)], 1:[(A,C)]}`，`min(...)` 只交回 priority 0 的两对。

```python
def get_possible_fusions_with_highest_priority(
    self, possible_fusions: List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]
) -> List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]:
    # Group the possible fusions based on their priority from the backend.
    # Only return the group of possible fusions with highest priority.
    if len(possible_fusions) == 0:
        return possible_fusions
    possible_fusions_group_by_priority: Dict[
        int, List[Tuple[BaseSchedulerNode, BaseSchedulerNode]]
    ] = {}

    for node1, node2 in possible_fusions:
        assert node1.get_device() == node2.get_device()
        device = node1.get_device()
        fusion_pair_priority = int(
            self.get_backend(device).get_fusion_pair_priority(node1, node2)
        )
        if fusion_pair_priority not in possible_fusions_group_by_priority:
            possible_fusions_group_by_priority[fusion_pair_priority] = [
                (node1, node2),
            ]
        else:
            possible_fusions_group_by_priority[fusion_pair_priority].append(
                (node1, node2)
            )
    # return the possible fusions with highest priority
    possible_fusions_with_highest_priority = min(
        possible_fusions_group_by_priority.items(), key=operator.itemgetter(0)
    )[1]
    assert len(possible_fusions_with_highest_priority) > 0
    return possible_fusions_with_highest_priority
```

例如 priorities 为 `{0:[A+B,C+D], 1:[A+C]}`，方法只返回 priority 0 的两项；priority 1 不是排在后面等待本轮执行，而是从本轮候选表移除。下一轮节点结构改变后会重新生成候选和 priority。

### score 是四项字典序键

先计算相邻逐点节点 op0/op1：两者都不是 template，规约类型相同；共同的 8 个 float32 中间元素给出 `memory_score=32`；`op0.max_order=0`、`op1.min_order=1`，所以距离项是 `-1`。下面函数应返回 `(True,True,32,-1)`；排序时 Python 从第一项开始逐项比较，较大的元组在前。

```python
def score_fusion(
    self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> Tuple[bool, bool, int, int]:
    """
    Assign a score (higher comes first) to the fusion of node1
    and node2.  When different fusions conflict with each other,
    this is the way we decide what order to run them in.

    Our current score is based on:
    - Estimate of the saved memory operations
    - Fusions closer together in original order
    """
    memory_score = self.score_fusion_memory(node1, node2)
    proximity_score = -max(
        abs(node1.min_order - node2.max_order),
        abs(node2.min_order - node1.max_order),
    )
    return (
        node1.is_template() == config.epilogue_fusion_first and memory_score > 0,
        node1.is_reduction() == node2.is_reduction() and memory_score > 0,
        memory_score,
        proximity_score,
    )
```

列表以 `reverse=True` 排序，所以 tuple 从左到右越大越先。前两项只有在 `memory_score>0` 时才能为 true。第一项的实际结果如下：

| `epilogue_fusion_first` | node1 是 template | 第一项 |
| --- | --- | --- |
| true | 是 | true |
| true | 否 | false |
| false | 是 | false |
| false | 否 | true |

因此它表达“node1 的 template 身份是否与当前配置偏好一致”，并非 template 永远优先。第二项在两节点 reduction 身份相同时为 true。随后才比较可省字节和原顺序距离。

### 共同精确依赖怎样变成字节数

```python
def dep_size_hint(self, dep: Dep) -> int:
    res = 0
    if dep not in self.__dep_size_hint_cache:
        try:
            if not dep.has_unbacked_symbols():
                res = dep.numbytes_hint()
        except KeyError:
            # In at least one test (test/inductor/test_torchbind.py) we
            # create a StarDep that doesn't exist in the graph and calling
            # `has_unbacked_symbols()` throws an error.
            pass
        self.__dep_size_hint_cache[dep] = res
    else:
        res = self.__dep_size_hint_cache[dep]
    return res

def score_fusion_memory(
    self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> int:
    """
    The first term in our fusion score that estimates number of saved
    memory operations.
    """
    node1_dep_len = len(node1.read_writes.reads) + len(node1.read_writes.writes)
    node2_dep_len = len(node1.read_writes.reads) + len(node2.read_writes.writes)

    # optimization: iter over smaller set
    if max(node1_dep_len, node2_dep_len) * 4 > min(node1_dep_len, node2_dep_len):
        if node1_dep_len > node2_dep_len:
            tmp = node1
            node1 = node2
            node2 = tmp

        deps = []
        for dep in node1.read_writes.reads | node1.read_writes.writes:
            if dep in node2.read_writes.reads or dep in node2.read_writes.writes:
                deps.append(dep)

        return sum(self.dep_size_hint(dep) for dep in deps)

    common_memory_deps = (node1.read_writes.reads | node1.read_writes.writes) & (
        node2.read_writes.reads | node2.read_writes.writes
    )
    return sum(self.dep_size_hint(dep) for dep in common_memory_deps)
```

方法比较的是完整 Dep 相等性，不只是 buffer 名。实验的 8 个 float32 同位置依赖给出 `8×4=32` 字节，因此 vertical/horizontal 候选 score 第三项为 32。shifted-index 两节点虽涉及同名 buffer，MemoryDep 索引不同，不形成共同精确依赖，memory score 不会凭名字得到 32。

`dep_size_hint` 对可估大小且不含 unbacked symbol 的 Dep 返回 `numbytes_hint()`；无法取得符号信息的特殊 Dep 返回 0。结果写入 cache，后续相同 Dep 直接复用。当前源码中 `node2_dep_len` 的 reads 部分写的是 `len(node1.read_writes.reads)`；这个数只影响选择“小集合遍历”还是“集合交集”策略，两个分支最终都只累计两边相等的 Dep。

```python
def score_fusion_key(
    self, nodes: Tuple[BaseSchedulerNode, BaseSchedulerNode]
) -> Tuple[bool, bool, int, int]:
    """
    Shim for list.sort(key=...)
    """
    node1, node2 = nodes
    return self.score_fusion(node1, node2)
```

### branched 案例的候选顺序

实验 10 实际候选依次为：

```text
(op0,op1) score=(True,True,32,-1)
(op1,op2) score=(True,True,32,-1)
(op3,op4) score=(True,True,32,-1)
(op1,op3) score=(True,True,32,-2)
(op0,op3) score=(True,True,32,-3)
```

前三个 pair 的四项分数完全相同，距离 -1 又优先于 -2、-3。`buffer_names_grouping` 按遍历节点时首次遇到 buffer 的顺序保存 key：先遇到 arg0_1、buf0，随后才遇到 buf1、buf2、buf3、buf4；只有 buf0、buf1、buf3 产生 pair。buf0 组先发现 `(op0,op1)`，buf1 组再发现 `(op1,op2)`，buf3 组随后发现 `(op3,op4)`。Python 稳定排序保持这个同分顺序。priority 在本 CPU 案例均为 0，没有提前淘汰任何一项。

### K02 理解验收

若 pair P score 为 `(False,True,64,-5)`，Q 为 `(True,False,4,-1)`，Q 排在前面：tuple 第一项先决定，不能用 64 字节越过 template 优先条件。

若两个普通节点没有共享 buffer、aggressive 关闭，它们不会进入 `can_fuse`；aggressive 打开且 group 相同，只是获得检查机会，仍须通过 K03 的静态门和 K04 的 cycle/收益门。

K02 到此完成。下一知识点 K03 逐句展开 `can_fuse/can_fuse_vertical`，把 vertical、shifted-index、extern 和 max-size 分别定位到具体拒绝门。

## K03：静态门与 backend 门依次决定一对当前组能否合并

第一次阅读下面长函数只走 vertical 正例：node1=op0，写 `buf0[c0]`，范围 8；node2=op1，读 `buf0[c0]` 并写输出，范围 8。两者同为 CPU 普通 SchedulerNode，成员总数 2 未超过限制；op0 是 op1 的祖先，读写索引完全一致，backend 允许相同循环范围融合。沿这组值，类型、设备、大小、共有数据和纵向依赖检查都会通过，函数返回 true。后面的 shifted、extern、max-size 三例再分别改动一个条件观察早退位置。

```python
def can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
    """
    Determine if it is possible to combine node1 and node2 into a
    single fused node.
    """

    if node1 is node2:
        return False

    why = WhyNoFuse(node1, node2)

    if isinstance(node1, GroupedSchedulerNode) or isinstance(
        node2, GroupedSchedulerNode
    ):
        why("grouped node must not be fused with other nodes")
        return False
    if (
        isinstance(node1, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
        and not node1.is_template()
    ):
        why("node1 is extern or nop")
        return False
    if (
        isinstance(node2, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
        and not node2.is_template()
    ):
        why("node2 is extern or nop")
        return False

    if node2.get_operation_names() & node1.ancestors:
        # 教学：反向融合会违反既有祖先关系。
        why("node1 must go before node2")
        return False

    if node2.is_template():
        why("templates can only fuse epilogues")
        return False
    if node1.is_template() and (
        node2.has_aliasing_or_mutation()
        or node2.is_reduction()
        or not config.epilogue_fusion
    ):
        why("template epilogue not satisfied")
        return False

    if (
        node1.get_buffer_names() | node2.get_buffer_names()
    ) & V.graph.no_fuse_buffer_names:
        why("fusion for buffer explicit disabled")
        return False

    device = node1.get_device()
    device2 = node2.get_device()
    if device != device2:
        why("device mismatch (%s vs %s)", device, device2)
        return False
    del device2

    no_shared_data = self.score_fusion_memory(node1, node2) == 0
    if no_shared_data:
        # 教学：精确 Dep 暂无交集时，允许 loop reorder helper 再找一次等价共享。
        no_shared_data = not self.has_shared_data_after_reordering_loop(
            node1, node2
        )

    loop_ordering_log.debug(
        "%s and %s has%s shared data",
        node1.get_name(),
        node2.get_name(),
        " no" if no_shared_data else "",
    )
    if no_shared_data and (
        not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
    ):
        if is_metric_table_enabled("fusion_failure_due_to_indexing_mismatch"):
            common_buf_names = (
                node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
            )
            if len(common_buf_names) > 0:
                get_metric_table("fusion_failure_due_to_indexing_mismatch").add_row(
                    lambda: {
                        "pre_grad_graph_id": V.graph.graph_id,
                        "post_grad_graph_id": V.graph.post_grad_graph_id,
                        "node1_name": node1.get_name(),
                        "node2_name": node2.get_name(),
                        "node1_debug_str": write_text(node1.debug_str()),
                        "node2_debug_str": write_text(node2.debug_str()),
                        "common_buffer_names": list(common_buf_names),
                        "failure_reason": self.decide_fusion_fail_reason(
                            node1, node2, common_buf_names
                        ),
                    }
                )

                why("no shared data due to indexing mismatch")
                return False
        why("no shared data")
        return False  # heuristic not needed for correctness

    if (
        not node1.is_foreach()
        and not node2.is_foreach()
        and len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
    ):
        why("exceeds max fusion")
        return False  # heuristic not needed for correctness

    if node1.get_operation_names() & node2.ancestors:
        # node2 depends on node1 outputs
        if not self.can_fuse_vertical(node1, node2):
            return False
        return self.get_backend(device).can_fuse_vertical(node1, node2)
    else:  # nodes don't depend on each other, but may have common reads
        if self.can_fusion_increase_peak_memory(node1, node2):
            why("will increase peak memory")
            return False
        return self.get_backend(device).can_fuse_horizontal(node1, node2)
```

门次决定“最先看到的拒绝原因”。实验的 `mm→sin` 在 extern 类型门就返回，尚未计算共享数据；`max_fusion_size=1` 的 pointwise pair 先通过类型、设备和共享检查，才在成员总数门拒绝。

shifted-index 两节点有共同 buffer 名，但精确 Dep 不相等。metric 未启用时 WhyNoFuse 记录 `no shared data`；启用对应 metric 且 common names 非空时，它先记录细分原因，再记录 `no shared data due to indexing mismatch`。

上面的类型、方向、device、共享、成员数和依赖判断都只读当前组状态。任一早退都不会修改 `fused_nodes` 或 `name_to_fused_node`；真正的组替换只发生在 K04 的 `backend.fuse` 之后。

template 只能位于 pair 的第一项，后面接它的 epilogue。若列表顺序得到 `(pointwise, template)`，正向检查会因 `node2.is_template()` 拒绝；K02 随后尝试反向 `(template, pointwise)`。反向 pair 仍需保证 pointwise 没有 alias/mutation、不是 reduction，并且启用了 epilogue fusion。

`has_shared_data_after_reordering_loop` 只在精确 Dep 交集为零时提供一次补充判断：它检查两组的访问在允许调整逐点循环顺序后能否变成相同的精确访问。返回 true 只让 pair 越过 `no shared data` 门；纵向 pair 随后仍要由 `can_fuse_vertical` 对 producer write 与 consumer unmet dependency 逐项匹配。

在 `aggressive_fusion=True` 下，两个同 loop group、无共享 buffer、都不是 reduction 的普通节点可以越过共享门。若它们没有祖先关系，就继续检查合并是否提高峰值内存，再交给 backend 的 horizontal 门。aggressive 只扩展候选与准入机会，不直接执行 merge。

### 纵向融合要求 consumer 的相关读取由 producer 写入满足

```python
def can_fuse_vertical(
    self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> bool:
    """
    Check if it is legal to fuse a consumer (node2) into a producer (node1).

    We can fuse them if all the reads of node2 either match
    corresponding writes in node1, or are written by nodes that can
    be scheduled before the fusion of node1 and node2.
    """
    node1_buf_names = node1.get_buffer_names()
    node1_op_names = node1.get_operation_names()
    computed_deps: OrderedSet[Dep] = OrderedSet()
    why = WhyNoFuse(node1, node2)

    for cd in node1.read_writes.writes:
        if not isinstance(cd, MemoryDep):
            continue
        for rd in node2.unmet_dependencies:
            if self.fusable_read_and_write(rd, cd):
                computed_deps.add(rd)

    for dep in node2.unmet_dependencies:
        if isinstance(dep, WeakDep) and self.fusable_weak_dep(dep, node1, node2):
            computed_deps.add(dep)

    remaining_deps = OrderedSet(
        dep.name for dep in node2.unmet_dependencies - computed_deps
    )
    if remaining_deps & node1_buf_names:
        # MemoryDeps didn't match and read different locations of the same buffer.
        # Examples here include:
        #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
        #   - MemoryDep("foo", x) != StarDep("foo")
        why("memory deps did not match")
        return False
    for name in remaining_deps:
        op_name = self.name_to_buf[name].defining_op.get_name()
        if node1_op_names & self.name_to_fused_node[op_name].ancestors:
            why("intermediate nodes between node1 & node2")
            return False

    return True
```

`computed_deps` 是融合后能在同一 kernel 内直接满足的 consumer 依赖。若 consumer 仍读取 producer 的 buffer，但索引/依赖类型没匹配，`remaining_deps & node1_buf_names` 立即拒绝；若剩余依赖来自别的 producer，则只在那个 producer 必须经过 node1 的祖先链时拒绝，避免把中间节点夹进不合法顺序。

### 普通读取与 weak 顺序要求怎样匹配写入

普通 `MemoryDep` 表示“consumer 必须从这个 buffer 的这些位置取值”；`WeakDep` 只登记 mutation 前后的顺序要求，不携带一条可直接与 producer write 比较的普通读取。因此 `can_fuse_vertical` 先用 `fusable_read_and_write` 匹配普通读取，再单独处理 weak dependency。

weak 正例中，node1 是旧值读取者，同时产生 `weak_dep.name="buf_old"`；node2 是后续 mutator，写入逻辑 mutation buffer `buf_mut`。具体状态为：

```text
mutation_real_name["buf_mut"] = "arg0"
node1.reads  = { MemoryDep("arg0", c0, size=[8]) }
node1 outputs 包含 "buf_old"
node2.writes = { MemoryDep("buf_mut", c0, size=[8]) }
weak_dep     = WeakDep(name="buf_old", mutating_buf="buf_mut")
```

第一道检查确认 weak dependency 等待的是 node1 产生的 `buf_old`。随后源码找到 node2 对 `buf_mut` 的唯一写入，通过 `mutation_real_name` 找回被修改前的真实名字 `arg0`，再从 **node1 的 reads** 中筛出对 `arg0` 的读取。正例只有一项 relevant read；它的位置、范围与 node2 写回相同，所以返回 true。把这项读取改成 `arg0[c0+1]` 就会失败。

完整函数还有一个必须按当前实现保留的空集合出口：若 node1 没有任何名为 `arg0` 的 read，`relevant_reads=[]`，Python 的 `all([])` 返回 true。源码没有在此处额外要求“至少找到一项读取”。因此本函数的准确规则是：找到的每一项 relevant read 都必须与写入位置和范围相同；零项时该条件也通过。正文不把正例中的“一项旧值读取”推广成源码未检查的必要条件。

```python
def fusable_weak_dep(
    self, weak_dep: WeakDep, node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> bool:
    if weak_dep.name not in node1.get_buffer_names():
        return False

    # A weak dep can be fused if and only if the fused operation acts inplace
    # on the buffer being mutated. i.e. the same index is being read then mutated
    mutating_writes = [
        write
        for write in node2.read_writes.writes
        if write.name == weak_dep.mutating_buf
    ]
    if len(mutating_writes) != 1:
        return False
    write = mutating_writes[0]
    assert isinstance(write, MemoryDep)

    if free_symbol_is_type(write.index, SymT.TMP):
        return False

    real_name = self.mutation_real_name[weak_dep.mutating_buf]
    relevant_reads = [
        read for read in node1.read_writes.reads if read.name == real_name
    ]
    return all(
        isinstance(read, MemoryDep)
        and not free_symbol_is_type(read.index, SymT.TMP)
        and read.index == write.index
        and read.size == write.size
        for read in relevant_reads
    )

# StarDep doesn't match MemoryDep, different indices don't match
# However, broadcasting sometimes strips dimensions, and if that's the case
# we still can match unmet dep
# if there's indirect indexing, don't match it
def fusable_read_and_write(self, read: Dep, write: MemoryDep) -> bool:
    if isinstance(read, MemoryDep):
        if read.mode == write.mode and write.mode is not None:
            return True
        read_name = self.mutation_renames.get(read.name, read.name)

        if (
            read_name != write.name
            or free_symbol_is_type(read.index, SymT.TMP)
            or free_symbol_is_type(write.index, SymT.TMP)
        ):
            return False

        if config.loop_ordering_after_fusion and read.num_vars != write.num_vars:
            # Need merge loops if we do loop ordering after fusion since
            # we have not merged the loops yet when creating the scheduler
            # nodes.
            read = read.normalize()
            write = write.normalize()

        return (
            read.index == write.index
            and len(read.size) >= len(write.size)
            and read.size[: len(write.size)] == write.size
        )
    elif isinstance(read, StarDep):
        read_name = self.mutation_renames.get(read.name, read.name)
        write_name = self.mutation_renames.get(write.name, write.name)
        if (
            read.mode == write.mode
            and write.mode is not None
            and read_name == write_name
        ):
            return True
    return False
```

普通 MemoryDep 需要版本名相同、无间接 tmp、索引相同，并允许 consumer 因广播多出尾部范围；若开启 fusion 后 loop ordering 且变量数不同，先 normalize 再比较。带非 None mode 的相同 mode 是特殊快速匹配。StarDep 只有在双方 mode 相同且非 None、版本名相同才匹配普通 write。

### 四个案例分别停在哪

| 案例 | 首个决定性门 | 结果 |
| --- | --- | --- |
| vertical `a=realize(x+1); b=a*2` | 祖先方向成立，consumer read 与 producer write 精确匹配，backend vertical 通过 | 可融合 |
| shifted `b=roll(a,1)` | 精确共享为 0，默认非 aggressive | `no shared data`，保留两节点 |
| extern `mm→sin` | node1 是普通 ExternKernelSchedulerNode | `node1 is extern or nop` |
| max size 1 | 两个成员总数 2 > 1 | `exceeds max fusion` |

### K03 理解验收

若 consumer 读取同名 buf0 的 `c0+1`，producer 写 `c0`，名字相同也不能进入 computed_deps；remaining 仍含 buf0，因此纵向检查报告 `memory deps did not match`。

若两节点无祖先关系但共读 x，它们走横向分支：先检查 peak-memory helper，再由 backend `can_fuse_horizontal` 决定。本方法不会误用 `can_fuse_vertical`。

K03 到此完成。下一知识点 K04 展开一轮执行：旧候选映射到当前融合组、cycle/收益复查、backend merge、重新排序与裁边。

## K04：旧候选先映射到当前组，再重新通过合法性、cycle 和收益门

`get_possible_fusions(nodes)` 只在一轮开始时生成候选。循环处理前一个 pair 后，候选里的原节点可能已经属于新的融合组，所以 `fuse_nodes_once` 不能直接拿旧对象继续合并。

### 收益门的完整实现

先走最常见的 CPU 普通节点路径：node1/node2 都不是 template 或 foreach，`benchmark_fusion=False`。于是 `is_multi_template=False`，函数在第一个收益判断直接返回 true，不会构造计时表。

普通 GPU 分支分别测得 `ms1`、`ms2` 和 `ms_fused`。例如 `ms1=1.4 ms`、`ms2=1.0 ms`、`ms_fused=1.1 ms`，比较的是 `1.1 < 1.4+1.0`。multi-template 分支的数据不同：`choice_timings` 保存每个 template choice **未融合时** 的耗时；函数切换到某个 Triton choice 后，另行调用 `benchmark_fused_nodes` 测出该 choice 的融合耗时，再选融合耗时最小且快于 `ms1+ms2` 的 choice。

```python
def speedup_by_fusion(
    self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> bool:
    """
    If config.benchmark_fusion is False, always return True.
    Otherwise, return True if fusion can brings speedup.
    """

    is_multi_template = node1.is_template() and isinstance(
        node1.get_template_node(), ir.MultiTemplateBuffer
    )
    if not config.benchmark_fusion and not is_multi_template:
        # 教学：普通节点在关闭 benchmark 时直接允许；multi-template 仍需选实现。
        return True

    if (
        node1.is_template()
        and not isinstance(node1.get_template_node(), ir.TritonTemplateBuffer)
        or node1.is_foreach()
        or node2.is_foreach()
    ):
        # TODO support benchmarking epilogue fusion
        return True

    node_list_1 = node1.get_nodes()
    device = node_list_1[0].get_device()

    # don't support benchmark fusion for CPU right now.
    if device.type == "cpu":
        return True

    node_list_2 = node2.get_nodes()
    node_list_fused = list(itertools.chain(node_list_1, node_list_2))

    # We can not accurately benchmark kernel using atomic_add
    # due to how we generate random integer inputs.
    # Skip benchmarking them by allowing fusion.
    if self._any_atomic_add(node_list_fused):
        return True

    from triton.compiler.errors import CompilationError

    why = WhyNoFuse(node1, node2)

    def log_fusion(ms_fused: float, ms1: float, ms2: float) -> None:
        if fusion_log.isEnabledFor(logging.DEBUG):
            if ms_fused < ms1 + ms2:
                fusion_log.debug(
                    "can fuse (benchmark): fusing %s with %s cause %sx speedup",
                    node1.get_buffer_names(),
                    node2.get_buffer_names(),
                    green_text(f"{(ms1 + ms2) / ms_fused:.3f}"),
                )
            else:
                fusion_log.debug(
                    "cannot fuse (benchmark): fusing %s with %s cause %sx slowdown",
                    node1.get_buffer_names(),
                    node2.get_buffer_names(),
                    red_text(f"{ms_fused / (ms1 + ms2):.3f}"),
                )

    if isinstance(node1, SchedulerNode) and isinstance(
        node1.node, ir.MultiTemplateBuffer
    ):
        multi_node = node1.node
        choice_timings = multi_node.choice_timings

        _, ms1 = multi_node.get_min_choice()
        ms2, path2 = self.benchmark_fused_nodes(node_list_2)

        min_ms_fused = float("inf")
        ms_fused_choice = None

        triton_choices = 0

        for choice, unfused_time in sorted(
            choice_timings.items(), key=lambda x: x[1]
        ):
            if not isinstance(choice, torch._inductor.ir.TritonTemplateCallerBase):
                continue

            if unfused_time >= ms1 + ms2:
                break

            triton_choices += 1
            if triton_choices > config.max_epilogue_benchmarked_choices:
                break

            # TODO - parallel compile triton templates
            # TODO - should prune/skip choices that are not within certain % of best choice
            with node1.node.swap_as_triton_caller(choice):
                ms_fused, _ = self.benchmark_fused_nodes(node_list_fused)

                if ms_fused < min_ms_fused:
                    min_ms_fused = ms_fused
                    ms_fused_choice = choice

        log_fusion(min_ms_fused, ms1, ms2)

        # after we do a fusion, we finalize a triton template.
        # TODO - could preserve multi template and choices for subsequent fusions
        if min_ms_fused < (ms1 + ms2) and ms_fused_choice is not None:
            node1.node.finalize_as_triton_caller(ms_fused_choice)
            return True
        else:
            return False
    else:
        try:
            ms1, path1 = self.benchmark_fused_nodes(node_list_1)
            if math.isinf(ms1):
                why("register spilling of the first kernel")
                return False
            ms2, path2 = self.benchmark_fused_nodes(node_list_2)
            if math.isinf(ms2):
                why("register spilling of the second kernel")
                return False
            ms_fused, path_fused = self.benchmark_fused_nodes(node_list_fused)
            if math.isinf(ms_fused):
                why("register spilling of the fused kernel")
                return False
        except CompilationError as e:
            # workaround triton issue: https://github.com/openai/triton/issues/2151
            if "Loop-carried variable" in str(e):
                return True  # allow fusion
            else:
                raise

    log_fusion(ms_fused, ms1, ms2)
    if (
        is_metric_table_enabled("slow_fusion")
        and ms_fused >= ms1 + ms2
        and (path1, path2) not in self.logged_slow_fusion
    ):
        self.logged_slow_fusion.add((path1, path2))
        get_metric_table("slow_fusion").add_row(
            lambda: {
                "kernel1_path": path1,
                "kernel1_latency": ms1,
                "kernel2_path": path2,
                "kernel2_latency": ms2,
                "fused_kernel_path": path_fused,
                "fused_kernel_latency": ms_fused,
                "slow_down_ratio": ms_fused / (ms1 + ms2),
            }
        )
    return ms_fused < ms1 + ms2
```

这段门有三类出口。关闭 benchmark 的普通节点、CPU、foreach、当前不支持测量的 template，以及含 atomic-add 的组直接允许；multi-template 会按已有 timing 过滤 Triton choice，再实际测融合版本并固定最快 choice；普通 GPU 路径分别测两项和融合项，只有 `ms_fused < ms1+ms2` 才接受。任一测量为无穷大表示寄存器 spilling，直接拒绝。特定 `Loop-carried variable` 编译错误被当前实现作为 workaround 放行，其他编译错误继续抛出。

实验 10 在 CPU 且 `benchmark_fusion=False`，所以它证明静态融合与组更新，不证明 GPU benchmark 更快。

### 一轮合并的完整实现

```python
def fuse_nodes_once(
    self, nodes: List[BaseSchedulerNode]
) -> List[BaseSchedulerNode]:
    """
    Combine eligible nodes into FusedSchedulerNodes.

    This relies on two key functions to control the logic:
        - self.can_fuse(): checks if a fusion is legal
        - self.score_fusion(): assigns priority to a given fusion
    """
    fused_nodes = OrderedSet(nodes)
    if fusion_log.isEnabledFor(logging.DEBUG):
        fusion_log.debug("fuse_nodes_once, candidates:")
        for node in fused_nodes:
            fusion_log.debug("  " + node.debug_str_short())  # noqa: G003
    for node1, node2 in self.get_possible_fusions(nodes):
        # 教学：候选保存旧节点；通过第一位原节点名找到它当前所属的融合组。
        node1 = self.name_to_fused_node[node1.get_first_name()]
        node2 = self.name_to_fused_node[node2.get_first_name()]
        if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
            node1, node2
        ):
            if not self.speedup_by_fusion(node1, node2):
                continue
            fusion_log.debug(
                "fusing %s with %s", node1.get_name(), node2.get_name()
            )

            # above can_fuse asserts that node2 has the same device
            device = node1.get_device()
            node3 = self.get_backend(device).fuse(node1, node2)
            # 教学：两项当前组替换为一项新组。
            fused_nodes.remove(node1)
            fused_nodes.remove(node2)
            fused_nodes.add(node3)
            # 教学：组内每个原节点名都指向 node3，供后续旧候选再次解析。
            self.name_to_fused_node.update(
                {n.get_name(): node3 for n in node3.get_nodes()}
            )
    # 教学：先按融合组覆盖的最早原位置恢复稳定候选序，再重新满足依赖拓扑。
    nodes = sorted(fused_nodes, key=lambda x: x.min_order)
    nodes = self.topological_sort_schedule(nodes)
    self.prune_redundant_deps(nodes)
    return nodes
```

映射后必须再次调用 `can_fuse`。旧候选 `(B,C)` 生成时 B 尚未与 A 合并；执行到它时实际检查的是 `(AB,C)`，成员数、祖先、设备、共享访问和 backend 条件都可能改变。若两个旧端点已映到同一组，`can_fuse` 的 `node1 is node2` 会拒绝，不会把一个组从 `fused_nodes` 删除两次。

### cycle 检查的完整实现

这项检查处理的是“旧候选映射到融合组后，两边与第三方融合组之间会不会互相依赖”。先看一个可以手算的状态：

```text
原有边：X ─> A       B ─> Y
当前组：{A}、{B}、{X,Y}
本次候选：合并 A 与 B
```

`X→A` 表示 A 依赖 X，所以候选 A+B 的祖先集合含 X。第三方当前组 `{X,Y}` 又包含 Y，而 Y 依赖候选中的 B。若把 A、B 合成 `{A,B}`，组间关系会变成：

```text
{X,Y} ──X→A──> {A,B}
{A,B} ──B→Y──> {X,Y}
```

这就是环。代入函数：`combined_names={A,B}`，`combined_ancestors={X}`；从 X 通过 `name_to_fused_node` 找到第三方融合组 `{X,Y}`。它的 operation 集合并非 `{X}` 的子集，而且它的 `ancestors` 含 B，于是 `combined_names & node.ancestors={B}`，检查返回 true。

源码中的 `found_path` 只展开已经融合的第三方组。若该组全部 operation 本来就都属于候选祖先，合并没有引入新的跨组成员，可以停止追踪；否则继续检查它是否依赖候选成员，或递归展开它带来的其他祖先组。

```python
def will_fusion_create_cycle(
    self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> bool:
    """
    Finds whether there's a path from node1 to node2 (or vice-versa)
    caused indirectly by other fusions.
    """
    # since we are just returning boolean here, use slightly faster, unordered set
    visited: Set[FusedSchedulerNode] = set()

    def found_path(node: BaseSchedulerNode) -> bool:
        # only fused nodes can introduce new ancestors.
        if isinstance(node, FusedSchedulerNode) and node not in visited:
            visited.add(node)
            if node.get_operation_names().issubset(combined_ancestors):
                # All fusion outputs are in ancestors of node1 and node2, thus
                # cannot introduce new path:
                #
                # 1. if output is neither descendent of node1 or node2, the
                #        output cannot introduce a path
                # 2. due to [can_fuse]: if WLOG output is descendent of node1, it cannot be
                #        on path(node1->node2), hence it cannot be ancestor of node2
                # 3. due to [acyclic]: if WLOG output is descendent of node1, it cannot be
                #        ancestor of node1
                return False
            else:
                # continue DFS of new ancestors introduced by the fusion
                return bool(combined_names & node.ancestors) or any(
                    found_path(self.name_to_fused_node[n])
                    for n in node.ancestors - combined_ancestors
                )
        return False

    # as above - use slightly faster, unordered set
    combined_names = (
        node1.get_operation_names()._dict.keys()
        | node2.get_operation_names()._dict.keys()
    )
    combined_ancestors = (
        node1.ancestors._dict.keys() | node2.ancestors._dict.keys()
    ) - combined_names
    cycle = any(found_path(self.name_to_fused_node[n]) for n in combined_ancestors)
    if cycle:
        WhyNoFuse(node1, node2)("will create cycle")
    return cycle
```

单个未融合节点沿用 Q321 已算好的 ancestors；能把原本分开的 X、Y 带进同一个检查对象的是已经合并过的第三方组。`combined_names` 是本次两组内全部原 operation，`combined_ancestors` 是两组祖先并集去掉组内成员。`visited` 防止在融合组图中重复递归。

### branched 案例的一轮四次 merge

实验 10 的一轮开始有五个节点，候选顺序为 `(op0,op1)`、`(op1,op2)`、`(op3,op4)`、`(op1,op3)`、`(op0,op3)`。每次处理时的实际对象如下：

| 旧候选 | 映射后的当前组 | 通过门后新组 | `fused_nodes` 中的组数 |
| --- | --- | --- | --- |
| `(op0,op1)` | `op0`、`op1` | `op0_op1` | 4 |
| `(op1,op2)` | `op0_op1`、`op2` | `op0_op1_op2` | 3 |
| `(op3,op4)` | `op3`、`op4` | `op3_op4` | 2 |
| `(op1,op3)` | `op0_op1_op2`、`op3_op4` | 五成员融合组 | 1 |
| `(op0,op3)` | 两端都映到同一五成员组 | `can_fuse` 因对象相同返回 False | 1 |

这里第四次不是把原始 op1 与 op3 合并，而是把当时的两个大组合并。每次成功后，组内所有原名都更新到新组，所以最后一个旧候选安全地变成“同组对自身”，不会重复 merge。

backend 返回的新组不只是成员名单。实验捕获到 `op0+op1→op0_op1` 后，新组同时带有：成员 `{op0,op1}`、buffers `{buf0,buf1}`、精确 reads/writes、对组外仍未满足的 dependencies、合并后的 ancestors、device、loop group、`min_order/max_order`，以及 template/foreach/reduction、alias/mutation 等组级查询结果。下一条旧候选 `(op1,op2)` 映射成 `(op0_op1,op2)` 后，`can_fuse` 正是读取这些聚合状态重新判断，而不是沿用原来 op1 的结论。这个具体新组的对外读取为 `arg0_1[c0]`，unmet dependencies 已为空。

本例关闭 benchmark 且运行在 CPU，收益门直接允许；实际记录没有命中 cycle。若 cycle 检查或收益门拒绝，循环只是 `continue` 或跳过 if 块，`fused_nodes` 与 name mapping 都不改变，后面的候选仍可继续检查。

### 一轮出口为何还要排序与裁边

`fused_nodes` 的删除和追加顺序不等于合法执行顺序。方法先按 `min_order` 恢复稳定基线，再调用 Q321 的拓扑排序满足组间依赖。

裁边可以用 vertical 例子直接观察。融合前 op1 的 unmet dependency 是“读取 `buf0[c0]`”，它由组外 op0 满足；融合后 op0 与 op1 已在同组，这项要求已由组内成员满足，不应继续作为组间入边。实验捕获的新组因此为：

```text
融合前：op1.unmet = {read buf0[c0]}
融合后：op0_op1.unmet = {}
对外读取仍为：{read arg0_1[c0]}
```

`prune_redundant_deps(nodes)` 清理的就是这类已经内部满足或可由当前融合映射证明冗余的组间要求。出口因此是一份重新拓扑排序且只保留对外依赖的当前组列表。

branched 从 5 变 1，外层 K01 立即因 `new_len==1` 停止。若本轮只变成 3 项，下一轮会以这三项重新建立共享分组、priority 和 score；旧候选表不会跨轮复用。

### K04 理解验收

若首个 merge 后旧候选 `(A,C)` 的 A 已属于 AB，系统会复查 `(AB,C)`；若 AB+C 超过 `max_fusion_size`，它在第二次 `can_fuse` 被拒绝，不能沿用 A+C 旧结论。

若 benchmark 测得 `ms1=2.0`、`ms2=3.0`、`ms_fused=5.0`，严格小于条件不成立，融合拒绝；相等并不算收益。若普通 CPU 案例配置 benchmark=True，当前实现仍直接允许，因为 CPU benchmark fusion 尚不支持。

至此，一对节点从候选生成、合法性检查、环检查和收益判断走到 backend 合并；合并后的节点列表再进入下一轮，直到某轮节点数不再减少、只剩一个组，或达到十轮上限。
