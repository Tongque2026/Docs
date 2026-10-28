# `PatternMatcherPass.apply`：从注册表筛出锚点，再逐条尝试规则

[返回 Pattern Matcher](../README.md) · [返回图优化](../../README.md) · [返回总览](../../../README.md)

Q12 已经建立了完整对象关系：规则以 `PatternEntry` 保存，注册进某个
`PatternMatcherPass.patterns` 表；阶段代码对同一个对象调用 `apply`，图被原地修改，
返回值是应用次数。本叶子继续打开 `apply` 的内部循环：一张图可能有很多节点、同一
末端也可能注册多条规则，它怎样只取可能的锚点，按什么顺序尝试，以及在哪些门前退出？

本篇完整讲解当前 `/work/pytorch/torch/_inductor/pattern_matcher.py` 中 `PatternMatcherPass` 从 `__init__` 到 `apply` 返回的连续范围。规则内部怎样匹配由 Q12 建立；replacement graph 怎样嵌入原图留给下一叶子 Q122。正常返回时，本篇为每个通过全部门禁且 `entry.apply(...)` 正常结束的 entry 返回一次计数；异常边界在后文单独说明。

## 先建立注册表与锚点

每个 `PatternEntry` 都有一棵 pattern。pattern 的最外层操作就是锚点。例如：

```text
pattern: mul(add(x, y), 2)
锚点键: ("call_function", aten.mul.Tensor)
```

注册表把这个键映射到一个 entry 列表。FX `Graph` 自己维护一张按 `op/target` 查询节点的
表：插入或删除节点时这张表也随之更新，`find_nodes(op=..., target=..., sort=False)` 从中
返回符合条件的一组节点。应用 pass 时无须拿每条规则与所有 FX 节点比较；先取得所有 mul
候选，再由 pattern 从锚点向输入追踪 add、x、y 和常数 2。这里的“索引”指这张查询表，
不是 tensor 下标，也不表示返回结果已经按图顺序排列。

entry 不是由 `apply` 创建的。注册时，`PatternEntry.register` 从 pattern 取得外层节点的
`op` 和可接受的 target，然后操作 `PatternMatcherPass.__getitem__` 返回的列表。这个列表
就是 `matcher.patterns[key]` 中保存的同一对象，不是副本。主例可简化为：

```python
key = (entry.pattern.op, aten.mul.Tensor)
matcher[key].append(entry)
```

`matcher.patterns` 初始是 `defaultdict(list)`：第一次读取尚不存在的 key 时，它创建空列表并保存；`append` 后状态从 `{}` 变为：

```text
{
  ("call_function", aten.mul.Tensor): [entry]
}
```

同一 key 后注册的 entry 继续追加到该列表末尾；设置 `prepend=True` 的注册入口则插到开头。于是 `apply` 看到的 entry 顺序已经由注册过程确定。

主例的根 `CallFunction` 只接受 `aten.mul.Tensor`，所以只登记一个 key。如果一个根 pattern
声明了多个可接受 target，`register` 会依次把同一个 entry 登记到每个 target 对应的 key；
这会增加候选 bucket，但不会复制 entry 对象。

## 声明范围的完整源码

先给类型标注和主案例具体值：`Optional[str]` 表示 pass_name 可以是字符串或 `None`，本例取
`"tutorial"`；注册表的键是 `(op,target)`，值是 `List[PatternEntry]`，本例只有键
`("call_function", aten.mul.Tensor)`，值为一条规则；`gm` 是含 x、y、add、mul、output 的
`GraphModule`。后文局部变量 `nodes` 则是“多个候选集合组成的列表”，不是注册表，也不是
已经摊平的 Node 列表。`apply` 用键找到 mul 节点，再把该节点交给列表中的 entry；正常
完成时 `count` 从 0 变 1，最终返回 1。中间的 `GraphTransformObserver` 只是观察这个 pass
前后哪些节点被增加或删除，并把记录交给图变换日志；它不替 matcher 改图，也不改变
`count`。

下面保留 `__init__`、`__getitem__`、`apply` 的全部产品语句和原顺序。仅 `# 教学：` 注释由本教程加入；产品注释原样保留。

```python
class PatternMatcherPass:
    def __init__(
        self,
        pass_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        # 教学：键是锚点的 FX op/target，值是按注册顺序保存的候选 entry。
        self.patterns: DefaultDict[
            Tuple[str, torch.fx.node.Target], List[PatternEntry]
        ] = defaultdict(list)
        self.pass_name = pass_name

    def __getitem__(self, item: Tuple[str, torch.fx.node.Target]) -> List[PatternEntry]:
        # 教学：注册逻辑可用 matcher[key] 取得并修改同一份 entry 列表。
        return self.patterns[item]

    def apply(self, gm: torch.fx.GraphModule) -> int:
        # 教学：空注册表在输入类型检查之前返回；此时连图都不会读取。
        if not self.patterns:
            return 0
        # 教学：GraphModule 交出其 graph；Graph 则反取 owning_module 供观察器使用。
        if isinstance(gm, torch.fx.GraphModule):
            graph = gm.graph
        elif isinstance(gm, torch.fx.Graph):
            graph = gm
            gm = graph.owning_module
        else:
            raise RuntimeError(
                f"The input to PatternMatcherPass must be a GraphModule or a Graph, but got {type(gm)}"
            )
        # 教学：尚无 region id 时，按图顺序给每个节点标上 mutation 区域。
        if should_compute_mutation_region_ids(graph):  # type: ignore[arg-type]
            compute_mutation_region_ids(graph)  # type: ignore[arg-type]
        get_mutation_region_id_partial = functools.partial(
            get_mutation_region_id, graph
        )
        count = 0
        nodes = []
        has_call_module = False
        # 教学：普通键按精确 op/target 查索引；call_module 先统一收集，稍后按模块类过滤。
        for op, target in self.patterns:
            if op == "call_module":
                has_call_module = True
            else:
                nodes.append(graph.find_nodes(op=op, target=target, sort=False))
        if has_call_module:
            nodes.append(graph.find_nodes(op="call_module", sort=False))
        pass_name = self.pass_name if self.pass_name is not None else "pattern_matcher"
        with GraphTransformObserver(
            gm, pass_name, trace_config.log_url_for_graph_xform
        ):
            # 教学：把各候选集合摊平，再按 FX 节点位置从后向前处理。
            for node in sorted(itertools.chain.from_iterable(nodes), reverse=True):
                target = extract_target(node)
                if node.op == "call_module":
                    # 教学：call_module 的原 target 是属性名；extract_target 得到模块类。
                    if (node.op, target) not in self.patterns:
                        continue

                # conservatively not applying pattern for cpu input,
                # since some of the patterns induce codegen and split nodes.
                # Note: we will only skip cpu compute if disable_cpp_codegen=True
                # 教学：当前 helper 判断本节点输入/输出是否含不能 codegen 的 tensor。
                if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                    continue

                # 教学：同一锚点键下，entry 按注册表中的顺序逐个尝试。
                for entry in self.patterns[(node.op, target)]:
                    # 教学：前一个 entry 已擦除锚点时，后续 entry 不再触碰它。
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    # pattern match crosses mutation barrier - discard
                    if (
                        is_match(m)
                        and len(set(map(get_mutation_region_id_partial, m.nodes))) != 1  # type: ignore[possibly-undefined]
                    ):
                        continue
                    # 教学：环境变量只为指定 node 输出匹配诊断，不改变是否应用。
                    if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
                        log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
                    # 教学：结构匹配与额外条件都通过，才计数并执行 entry。
                    if is_match(m) and entry.extra_check(m):
                        count += 1
                        entry.apply(m, graph, node)  # type: ignore[arg-type]
                        counters["inductor"]["pattern_matcher_count"] += 1
                        counters["inductor"]["pattern_matcher_nodes"] += len(m.nodes)
        return count
```

## 入口分支：空表、GraphModule、Graph 和错误类型

这四条路径的先后顺序很重要：

| 输入与注册表 | 经过的分支 | 结果 |
| --- | --- | --- |
| `object()`，注册表为空 | 第一条 `if not self.patterns` | 返回 0，不检查类型 |
| `GraphModule`，有规则 | 读取 `.graph` | 继续扫描 |
| 该 GraphModule 所属的 `Graph`，有规则 | 使用 Graph 本身，并取 `.owning_module` | 同一 `(x+y)*2` 实测匹配 1 次 |
| `object()`，有规则 | 落入 `else` | 抛出包含实际类型的 `RuntimeError` |

因此“apply 只接受 GraphModule 或 Graph”要带上条件：非空 matcher 才会走到类型检查。空 matcher 是更早的快速出口。

Graph 输入路径还依赖 `graph.owning_module`：本例的 Graph 来自一个 GraphModule，所以可以反取所属模块。一个没有 owning module 的裸 Graph 没有在这里被显式拒绝，但后面的观察器，以及 call_module 的类提取，都可能需要所属模块；正文不把裸 Graph 推广为已验证入口。

## mutation region 怎样写入节点

`should_compute_mutation_region_ids(graph)` 检查图中第一个节点的 `node.meta` 是否已有 `"mutation_region_id"`。没有时，`compute_mutation_region_ids` 按图顺序从 0 开始扫描：遇到 `copy_` 等 mutation 节点先把编号加一，再把当前编号写入每个节点的 `node.meta["mutation_region_id"]`。

例如：

```text
add.meta["mutation_region_id"]   = 0
copy_.meta["mutation_region_id"] = 1
mul.meta["mutation_region_id"]   = 1
```

以后匹配到 add 与 mul 时，`get_mutation_region_id(graph, node)` 从这些 meta 读取 0 和 1，集合得到 `{0,1}`。若某个后来新建的节点还没有 id，该 helper 会沿图向前找到最近的已编号位置，再按途中 mutation 补算并写回；因此门禁查询的不只是一个临时局部表。

## 候选收集：普通 target 与 call_module 不同

对 `call_function/aten.mul.Tensor`，`graph.find_nodes` 可以按 op 和 target 精确取节点。`call_module` 节点的 FX target 却是模块属性名，例如 `"relu"`、`"sigmoid"`；注册表使用的是 `nn.ReLU` 这样的模块类。实现因而先收集所有 `call_module`，循环里再用 `extract_target(node)` 从 owning module 取实例并得到类。

实验图为：

```text
x → call_module("relu") → call_module("sigmoid") → output
```

注册表只有 `(call_module, nn.ReLU)`。逆序扫描先看到 sigmoid，提取出 `nn.Sigmoid`，注册表没有该键，执行 `continue`；随后 ReLU 命中并应用一次。实际事件只有 `['relu']`。

`extract_target(node)` 对普通 `call_function`、`call_method` 等直接返回 `node.target`；只有 `call_module` 才从 `node.graph.owning_module` 取属性实例并返回它的类。因此普通 mul 仍以 `aten.mul.Tensor` 查表，而模块调用以 `nn.ReLU` 查表。

`nodes` 不是一个节点列表，而是“候选集合的列表”。若注册表有 mul 与 neg 两个普通 key，图索引分别交回：

```text
nodes = [
  {mul_0, mul_1},
  {neg_0},
]
```

这里花括号只表示 `find_nodes(..., sort=False)` 返回的索引集合，不承诺集合内部已有图顺序。
`chain.from_iterable(nodes)` 摊平成 `mul_0, mul_1, neg_0` 这样的节点流；最后一次
`sorted(..., reverse=True)` 比较 FX Node 的 `_sort_key`。节点被插入双向链表中的某个位置
时，FX 同步计算这个位置的排序键，因此统一排序得到从图尾到图头的顺序。不同普通 key
精确限制了不同的 `(op,target)`，而所有 call_module key 只追加一份“全部 call_module”
集合，所以本构造不会因为多个注册 key 重复加入同一个候选节点。

例如整张图的确定顺序是：

```text
... → mul_0 → neg_0 → mul_1 → output
```

即使外层注册表先枚举 mul key，flatten 暂时得到 `[mul_0, mul_1, neg_0]`，最后的全图排序仍把执行顺序定为：

```text
mul_1 → neg_0 → mul_0
```

因此最终顺序不由 key 注册先后或索引集合的迭代次序决定。

## 一次成功匹配经过哪些门

主案例仍是 float32 的 `(x+y)*2`，规则锚点是末端 mul：

| 次序 | 当前对象与判断 | 本例状态 |
| ---: | --- | --- |
| 1 | 候选索引 | 只取到 `mul` |
| 2 | unsupported type | 默认允许当前 CPU float32 输出，继续 |
| 3 | `node._erased` | false |
| 4 | `entry.pattern.match(mul)` | 得到含 `add,mul` 的 Match，捕获 x、y |
| 5 | mutation region 集合 | `{0}`，大小为 1 |
| 6 | `extra_check` | x 的 fake dtype 为 float32，返回 true |
| 7 | 应用与计数 | entry 原地改图并返回 `None`，外层不读取其返回值；`count=1` |

把所有短路放到一条线上：

```text
候选 node
  ├─ call_module 类未注册 → 下一个 node
  ├─ unsupported          → 下一个 node
  └─ 对当前 key 的每个 entry
       ├─ node 已 erased   → 结束该 node 的 entry 循环
       ├─ match 失败       → 下一个 entry
       ├─ 跨 region        → 下一个 entry
       ├─ debug 命中       → 只记日志，继续
       ├─ extra_check 失败 → 下一个 entry
       └─ local count +1 → entry.apply → 两个 global counter → 下一个 entry
```

实际 counters 是：

```text
pattern_matcher_count = 1
pattern_matcher_nodes = 2
```

第一个数按正常完成的 entry 次数加一；第二个数按该 Match 中节点数增加，本例是 add 与 mul 两个。replacement 产生的新三个节点不计入这次 `pattern_matcher_nodes`。

`entry.apply` 正常返回后，主例的可观察图已经从左边变成右边：

```text
应用前：x,y → add(old_add) → mul(old_mul, 2) → output

应用后：x → mul_tensor(x,2) ─────┐
                                  ├→ add_tensor → output
        y → mul_tensor_1(y,2) ────┘
```

`output` 原来读取 `old_mul`，现在读取 `add_tensor`；旧 `old_mul` 和已无其他使用者的 `old_add` 被擦除。本篇只把这项前后变化当作 `entry.apply` 的接口结果，用它闭合“匹配成功后图发生了什么”；三个新节点怎样复制进原图、怎样重接 users 并删除旧节点，由下一叶子 Q122 展开。

## 六种不应用路径分别在哪里退出

### 1. 根 target 根本不存在

图只有 add→relu，没有 mul。候选集合为空，根本不进入 node 循环，返回 0。

### 2. 当前类型要求 fallback

当前 helper 的实际实现会检查输入中的不支持类型，并检查候选节点输出。CPU 情况下，只有 `disable_cpp_codegen=True` 才把 CPU 输出判为 unsupported。实验保持同一个 float32 图与规则，只打开该配置，候选 mul 在调用 pattern 之前被跳过，返回 0。

因此这个配置组合跳过的是会产生 CPU 输出的候选；输入仍按 helper 的 dtype 支持条件检查。

### 3. 结构或常数不匹配

图是 `(x+y)*3`。锚点仍是 mul，所以进入 entry；但 pattern 要求常数 2，`m` 是失败对象。mutation 检查因 `is_match(m)` 为 false 而不进入，最后成功条件也为 false，返回 0。

### 4. 匹配跨 mutation region

图是：

```text
add(x,y) [region 0] → copy_(x,y) → mul(add,2) [region 1]
```

结构匹配成功，但 `m.nodes` 的 region 集合为 `{0,1}`，大小不是 1，于是在 mutation 屏障处 `continue`。`extra_check` 和 entry 都不执行。

### 5. extra_check 拒绝

int64 的 `(x+y)*2` 结构与 region 均通过；`extra_check` 读取 x 的 fake dtype，返回 false。结果保持原图，local count 与两个全局 matcher counter 都为 0。

### 6. call_module 类不在注册表

前面的 Sigmoid 节点属于候选集合，却在提取模块类后发现键不存在，跳过而不索引一个不存在的 entry 列表。

调试环境变量分支不是拒绝门：名字相等时记录当前 node、参数、匹配结果和 pattern，之后仍照常检查成功条件。

## 从图尾向前的实际顺序，以及 `_erased` 保护哪一步

两项互不相关的请求：

```text
neg = -x
neg_1 = -y
return neg, neg_1
```

候选集合按图中顺序含 `neg, neg_1`；外层 `sorted(..., reverse=True)` 的实际应用顺序是：

```text
neg_1 → neg
```

两项均匹配时，返回 `count=2`，两个 counter 也都是 2。这里能够直接确认的是实现采用图尾到图头的顺序；不能仅由这个案例推出任意 replacement 都不会影响尚未扫描的旧候选。`sorted` 已经建立了一份节点句柄列表，entry 仍必须正确维护图。

另一个实验在同一个 neg 锚点下注册两条 entry。第一条把 neg 的所有使用改接到 x，再擦除 neg：

```text
entry 1：执行，node._erased 变为 true，count 从 0 到 1
entry 2：循环顶部看到 _erased，break，不做 match
```

实际事件是 `['first']`，返回 1。若第一条只是结构失败或 `extra_check` 返回 false，它没有擦除节点，第二条仍会获得尝试机会。

这个 `_erased` 判断证明的是“当前锚点被前一条 entry 擦除”时，后续 entry 不再 match 它。若某条 replacement 擦除了另一个尚未轮到的候选，旧列表仍可能保留那个节点句柄；当前函数在 `_erased` 判断前还会执行 `extract_target` 和 unsupported-type 检查，因此本篇不把当前案例扩张成对任意跨候选擦除都安全的保证。

## 多消费者与重复参数不是 apply 自己决定的

主规则把内部 add 声明为 `_users=MULTIPLE`，所以 add 同时供 mul 与 relu 使用时，规则仍能替换 mul 路径，并保留 relu 所需的原 add。相反，重复参数 pattern `add(same,same)` 只匹配 `x+x`，不匹配 `x+y`。

这些条件由 `entry.pattern.match(node)` 产生的 Match 或失败对象决定。`apply` 的职责是选锚点、组织门禁和调用 entry；它不会自行推断多消费者或重复参数等价性。

## 改一个条件，推算计数

若图有两个互不相干的 float32 `(a+b)*2` 与 `(c+d)*2`：逆序先处理后一个，再处理前一个。两次都成功时三个数依次为 local `count=2`、全局 `pattern_matcher_count=2`、全局 `pattern_matcher_nodes=4`；最后一项来自两个 Match 各含 add、mul 两个旧节点。

若后一个是 int64、前一个是 float32，后一个通过结构与 region 后被 `extra_check` 拒绝；前一个成功。因此三个最终增量分别是 1、1、2。这个结果来自“每个成功 entry 加一次”和“只累加成功 Match 的节点数”，而不是来自候选节点总数。

最后看异常边界。源码顺序是：

```text
count += 1
entry.apply(...)
global pattern_matcher_count += 1
global pattern_matcher_nodes += len(m.nodes)
```

若 `entry.apply` 抛异常，本函数直接异常退出，不会正常返回 local count；那次 local 加一已经发生，但两个 global counter 尚未增加。entry 在抛出前是否已部分改图取决于 entry 实现，本篇不能保证回滚。因此“返回成功应用次数”只描述函数正常返回时的结果，不表示这四行具有事务原子性。

至此，声明范围内的快速出口、输入类型、region 初始化、两类候选收集、逆序循环、所有拒绝门、擦除保护、调试分支、entry 调用和 counters 都已落到具体状态。下一叶子将继续进入 replacement graph 的真正嵌入过程。
