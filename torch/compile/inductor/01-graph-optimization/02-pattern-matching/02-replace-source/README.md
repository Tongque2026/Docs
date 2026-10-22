# `replace_with_graph`：在原图中重建 replacement，再迁移旧输出的使用者

[返回 Pattern Matcher](../README.md) · [上一篇：候选扫描源码](../01-apply-source/README.md) · [返回总览](../../../README.md)

Q121 停在 `entry.apply(m, graph, node)`。对 replacement entry 来说，下一步不是调用 replacement 函数计算一次数值，而是把 replacement 的 FX 图解释一遍，在原图中创建等价的新节点，然后把旧输出的所有消费者改接过去。

本叶子完整讲解当前 `/work/pytorch/torch/_inductor/pattern_matcher.py` 中 `ReplacementPatternEntry` 的声明、`replace_with_graph` 与 `apply`。读完后，你应能推演单输出、多输出、packed tuple、可选 `None` 输出、meta 与重算标签各怎样迁移，以及失败时图是否已经发生部分修改。

## 进入源码前先看两个图

主案例的旧图和 replacement 图是：

```text
旧图：x,y → add_old → mul_old → output

replacement：x → mul_x ─┐
                         ├→ add_new → replacement output
             y → mul_y ─┘
```

`x`、`y` 已经是原图节点，不应复制 placeholder。`mul_x`、`mul_y`、`add_new` 要在原图创建。最后让原来读取 `mul_old` 的 output 改读 `add_new`，再删除失去用户的旧节点。

## 声明范围的完整源码

进入长代码块前固定几项 Python 与 FX 对象：`@dataclasses.dataclass` 会为字段生成初始化等样板方法；这里的实例保存 `normalize_args` 等 replacement 配置。`match` 保存本次捕获的 x、y 和旧输出节点，`graph` 是正在修改的原图，`replacement_graph` 是含 placeholder、mul、add、output 的模板图，`args=[x,y]` 把两个模板 placeholder 绑定到原图节点。内部 `Replacer` 逐个解释模板节点：placeholder 直接取 args，普通调用在原 graph 新建节点，output 收集新 add；随后旧 mul 的 users 改读新 add。

下面保留类装饰器、字段、静态方法、内部类与函数、全部分支、产品注释和 `apply` 的原顺序。仅 `# 教学：` 开头的注释由教程加入。

```python
@dataclasses.dataclass
class ReplacementPatternEntry(PatternEntry):
    normalize_args: Callable[..., List[Any]]

    @staticmethod
    def replace_with_graph(
        match: Match,
        graph: torch.fx.Graph,
        replacement_graph: Union[torch.fx.Graph, torch.fx.GraphModule],
        args: Sequence[torch.fx.Node],
    ) -> None:
        # 教学：解释 replacement；placeholder/output 沿用 Interpreter，普通调用则写进原 graph。
        class Replacer(torch.fx.Interpreter):
            call_method = None  # type: ignore[assignment]
            call_module = None  # type: ignore[assignment]
            get_attr = None  # type: ignore[assignment]

            def run_node(self, node: torch.fx.Node) -> Any:
                if node.op in ("placeholder", "output"):
                    return super().run_node(node)
                if node.op == "call_function":
                    target = node.target
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    result = graph.call_function(target, args, kwargs)  # type: ignore[arg-type]
                    # 教学：replacement trace 有抽象值时，把 val/tensor_meta 带到原图的新节点。
                    if "val" in node.meta and "val" not in result.meta:
                        result.meta["val"] = node.meta["val"]
                        if isinstance(node.meta["val"], torch.Tensor):
                            assert "tensor_meta" in node.meta
                            result.meta["tensor_meta"] = node.meta["tensor_meta"]
                    return result
                raise NotImplementedError(f"unhandled {node}")

        output_nodes = match.output_nodes()

        # 教学：所有 replacement 节点都插到最早的旧输出前，先定义再供消费者使用。
        if len(output_nodes) == 1:
            last_node = output_nodes[0]
        else:
            assert output_nodes[0]
            nodes = list(output_nodes[0].graph.nodes)
            indices = [
                (nodes.index(n), n)
                for n in output_nodes
                if isinstance(n, torch.fx.Node)
            ]
            last_node = min(indices, key=operator.itemgetter(0))[1]

        def percolate_tags(
            node: torch.fx.Node,
            tag_name: str,
            tag_value: str,
            input_stops: Set[torch.fx.Node],
        ) -> None:
            queue = [node]
            visited = set()

            # 教学：从新输出反向遍历新内部节点，遇到捕获的原图输入便停止。
            while queue:
                arg = queue.pop()
                if (
                    arg not in visited
                    and arg not in input_stops
                    and hasattr(arg, "meta")
                ):
                    visited.add(arg)
                    arg.meta[tag_name] = tag_value
                    queue.extend(arg.all_input_nodes)

        with graph.inserting_before(last_node):
            # 教学：replacement placeholder 依次绑定 args；返回值是原图中的新 Node/序列/None。
            replacement = Replacer(replacement_graph).run(*args)  # type: ignore[arg-type]
            if isinstance(replacement, torch.fx.Node):
                replacement = [replacement]

            def maybe_getitem(node: torch.fx.Node) -> Any:
                if node.op != "call_function":
                    return None
                if node.target != operator.getitem:
                    return None
                assert len(node.args) == 2
                return node.args[1]

            def replace(
                old: Union[torch.fx.Node, None],
                new: Union[torch.fx.Node, Sequence[torch.fx.Node], None],
            ) -> None:
                # 教学：pattern 的可选旧输出与 replacement 的可选新输出必须同时为空。
                if old is None:
                    assert new is None
                    return
                assert isinstance(old, torch.fx.Node)
                # 教学：旧输出变成 None 时，消费者改读 None，随后删除旧节点。
                if new is None:
                    old.replace_all_uses_with(None)  # type: ignore[arg-type]
                    graph.erase_node(old)
                    return
                if isinstance(new, torch.fx.Node):
                    # 教学：新输出没有 trace val 时，以对应旧输出的整份 meta 作为后备。
                    if "val" not in new.meta:
                        new.meta.update(old.meta)

                    # Preserve the recompute tags in the replacement graph. We
                    # look at the recompute tags of the original output node to
                    # propagate the tag from the output all the way to the input
                    # args (named as args in the replace_with_graph).
                    # Note that this is best effort. Since patterns are from
                    # many to many, there is no easy way to correctly map the
                    # recomputable tags. It is possible in some scenarios that we
                    # incorrectly tag some nodes as recomputables.
                    for tag_name in ["recompute", "ac_graph_id"]:
                        if tag_name in old.meta:
                            percolate_tags(new, tag_name, old.meta[tag_name], set(args))

                    # 教学：先把全部消费者从 old 改接 new，old 无用户后才能擦除。
                    old.replace_all_uses_with(new)
                    graph.erase_node(old)
                    return

                # `new` is not a node: it's a list of nodes.
                #
                # This happens when we want to replace a node that has a single
                # packed return with multiple unpacked returns. We need to do
                # some graph surgery here.
                #
                # Example:
                #   def original_graph(x):
                #      a = op(x)
                #      b = a[0]
                #      c = a[1]
                #      ...
                #
                # Assume that we want to replace op(x) with the graph
                #   def new_op(x):
                #      w = x + 1
                #      z = x + 2
                #      return (w, z)
                #
                # We need to replace `op` with the contents of `new_op`,
                # and then rewrite a[0] to be w and a[1] to be z, as so:
                #   def new_graph(x):
                #     w = x + 1
                #     z = x + 2
                #     b = w
                #     c = z
                #     ...
                old_uses = list(old.users.keys())
                for user in old_uses:
                    idx = maybe_getitem(user)
                    if idx is None:
                        raise AssertionError("can't handle")
                    replace(user, new[idx])  # type: ignore[index]
                graph.erase_node(old)

            # 教学：数量相等时逐位置映射；否则只支持一个 packed 旧输出。
            if len(output_nodes) == len(replacement):
                for old, new in zip(output_nodes, replacement):
                    replace(old, new)
            else:
                assert len(output_nodes) == 1
                replace(output_nodes[0], replacement)

        # 教学：输出迁移成功后，再删除 match 内剩余且已无用户的旧节点。
        match.erase_nodes()

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        assert match.replacement_graph is not None
        self.replace_with_graph(
            match,
            graph,
            match.replacement_graph,
            self.normalize_args(*match.args, **match.kwargs),
        )
```

## Interpreter 怎样把模板节点变成原图节点

`Replacer(...).run(*args)` 按 replacement 图的节点顺序执行。`env` 是“模板节点 → 本次解释结果”的表；父类 `Interpreter` 在每个 `run_node` 返回后把结果写入 env。处理一个普通调用前，`fetch_args_kwargs_from_env` 递归查看该模板节点的参数：若参数是模板 Node，就换成 env 中已经保存的结果。

主例的逐步状态是：

| 当前模板节点 | 本步读取 | `run_node` 返回 | 本步后的关键 env |
| --- | --- | --- | --- |
| placeholder `rx` | `args[0]=x_old` | 原图节点 `x_old` | `rx→x_old` |
| placeholder `ry` | `args[1]=y_old` | 原图节点 `y_old` | `rx→x_old, ry→y_old` |
| `r_mul_x(rx,2)` | fetch 把 `rx` 换成 `x_old` | 原图新节点 `mul_x(x_old,2)` | `r_mul_x→mul_x` |
| `r_mul_y(ry,2)` | `ry→y_old` | 原图新节点 `mul_y(y_old,2)` | `r_mul_y→mul_y` |
| `r_add(r_mul_x,r_mul_y)` | 两个模板输入换成 `mul_x,mul_y` | 原图新节点 `add_new(mul_x,mul_y)` | `r_add→add_new` |
| replacement output `r_add` | 从 env 取 `add_new` | `add_new` | `run()` 最终返回 `add_new` |

因此模板图只提供“应创建什么关系”；env 把模板里的引用逐步换成原图现有节点或刚创建的新节点。placeholder 不复制，普通 `call_function` 也不执行 tensor 数值计算。

## 第一条路径：一个旧输出换成一个新输出

对 `(x+y)*2 → x*2+y*2`：

| 时刻 | 关键状态 | 原图变化 |
| --- | --- | --- |
| 取旧输出 | `output_nodes=[mul_old]` | 插入点为 `mul_old` |
| 解释 placeholder | replacement 的 x、y 绑定 `args=[x_old,y_old]` | 不创建新 placeholder |
| 解释三个 call_function | env 依次得到 `mul_x,mul_y,add_new` | 三个节点插在 `mul_old` 前 |
| 解释 output | 返回 `add_new`，单 Node 被包成列表 | 准备与旧输出逐位对应 |
| `replace(mul_old,add_new)` | output user 改读 `add_new` | 擦除 `mul_old` |
| `match.erase_nodes()` | 检查 match 中旧 add | 无其他用户则删，有 relu 用户则保留 |

users 的状态也可以逐步写出：

```text
开始： add_old.users = {mul_old}
      mul_old.users = {output}

三个新节点建立后：旧 users 不变；add_new.users 仍为空

old_mul.replace_all_uses_with(add_new) 后：
      mul_old.users = {}
      add_new.users = {output}
      output 原参数 mul_old → add_new

erase mul_old 后：
      add_old.users = {}

match.erase_nodes()：
      若 add_old 没有 match 外的 user，就擦除它；
      若 relu 仍读取 add_old，就保留 add_old。
```

实际 float32 输入 `[1,2,3]` 与 `[10,20,30]` 得到 `[22,44,66]`。数值相等只验证 replacement 语义；节点迁移机制由前后图和 users 变化证明。

## Replacer 只重建 call_function

replacement 的 placeholder 与 output 由 `Interpreter` 处理。中间 `call_function` 不真正运行，而是调用原图的 `graph.call_function`；从 env 取得的参数已经是原图输入或刚建立的新 Node。

`call_method`、`call_module`、`get_attr` 字段被设为 `None`，而 `run_node` 遇到这些 op 也会落入 `NotImplementedError("unhandled ...")`。所以当前 replacement trace 需要由 placeholder、call_function、output 构成；这不是任意 FX Graph 的通用拷贝器。

参数注解允许 `torch.fx.Graph | GraphModule`，但当前 `Replacer` 继承的 `torch.fx.Interpreter` 构造器会调用 `replacement_graph.named_modules()`。独立实验直接传裸 `Graph`，实际得到：

```text
AttributeError: 'Graph' object has no attribute 'named_modules'
```

当前正常调用路径传入 GraphModule；若只传裸 Graph，这一实现会在读取 `owning_module.graph` 时失败。

## 两个旧输出怎样选择插入点并逐位替换

实验旧图有两个输出 `neg(x)` 与 `relu(x)`；replacement 为：

```python
first = x + 1
return first * 2, x + 3
```

`output_nodes` 长度为 2。实现列出原图节点，为每个非 None 旧输出求位置，取更早的 `neg` 作为 `last_node`。三个新节点因此全部定义在最早旧输出之前，再执行位置映射：

```text
old neg  → new mul(first,2)
old relu → new add(x,3)
```

输入 `[-1,2]` 的实际结果为 `([0,6],[2,5])`；新 call 顺序是 `add_tensor, mul_tensor, add_tensor_1`。即使第二个旧输出和消费者更靠后，也不会把它对应的新值定义在消费者之后。

多输出路径要求 `output_nodes[0]` 是真实 Node；其余位置可以是 None。若所有真实旧输出都不在第 0 位，当前断言不会接受这种排列。

## packed 旧输出为什么要改写 getitem

另一种接口不把两个旧结果登记成两个 output node，而是一个 packed 节点：

```text
packed = op(x)
first  = getitem(packed,0)
second = getitem(packed,1)
return first,second
```

replacement 直接返回 `[x+1,x+2]`。此时 `len(output_nodes)=1`、`len(replacement)=2`，进入 packed 分支：

1. 复制 `packed.users`，避免循环时删除 user 改变正在遍历的字典。
2. 对第一个 getitem，`maybe_getitem` 得到 0，递归执行 `replace(getitem0,new[0])`。
3. 对第二个 getitem，索引为 1，改接 `new[1]`。
4. 两个 getitem 均擦除后，再擦除 packed。
5. 最后调用 `match.erase_nodes()`。

输入 `[4]` 的实测结果为 `([5],[6])`，最终 call target 只剩两个 `aten.add.Tensor`，没有 packed 函数和 `operator.getitem`。

若 packed 还有一个直接 user，例如 output 同时返回整个 packed，`maybe_getitem(output)` 返回 None，抛 `AssertionError("can't handle")`。这里没有事务或回滚：排在该 user 之前的 getitem 可能已经被替换。因此失败只说明函数异常退出，不能假设原图保持原样。

本文失败实验按创建顺序给 packed 登记三个 users：`getitem0`、`getitem1`、最后是 `output`。`old_uses=list(...)` 保持这一次快照顺序，所以异常发生前已经做了：

```text
1. 新建 new0=x+1、new1=x+2
2. output 的第一项从 getitem0 改为 new0；擦除 getitem0
3. output 的第二项从 getitem1 改为 new1；擦除 getitem1
4. 遇到 output 仍直接读取 packed，抛 "can't handle"
```

异常后的残留图含 `new0`、`new1` 和仍未擦除的 `packed`；output 读取 `(new0,new1,packed)`；两个 getitem 已消失；函数没有走到 `graph.erase_node(packed)` 或末尾 `match.erase_nodes()`。换一种 user 插入顺序，失败前完成的子集也可能不同，这正是非事务修改的含义。

## `None` 输出的两条路径

Match 的 output pattern 可以包含 None。源码明确区分：

| old | new | 动作 |
| --- | --- | --- |
| None | None | 不改图，直接返回 |
| None | Node | 断言失败 |
| Node | None | 所有 users 改接 None，擦除 old |
| Node | Node | 正常迁移 meta、tag 和 users |

实验一将 `[old,None]` 对应 `[x+1,None]`，输入 `[2]` 得到 `([3],None)`；实验二将两个旧输出对应 `[None,x+3]`，得到 `(None,[5])`。它们分别走过 None/None 与 Node/None 分支。

## meta 有两种来源

### replacement trace 已有 val

`make_fx` 产生的 replacement 节点带 `node.meta['val']` 和 tensor meta。Replacer 在原图新建节点后复制 `val`；若 val 是 Tensor，还断言并复制 `tensor_meta`。双输出实验的三个新 call 均实际带这两项。

### replacement 新输出没有 val

手工 replacement GraphModule 的 relu 节点没有 trace meta。它成为对应旧输出时，`replace` 发现新节点仍无 val，于是执行 `new.meta.update(old.meta)`。实验旧输出的 `old_only=17` 与 `recompute='r2'` 均出现在新 relu 上。

若 replacement 已有 val，`replace` 不会把旧输出的所有其他 meta 整体覆盖过去；只有下面两种重算标签另走显式传播路径。

例如旧输出 meta 为 `{val: old_val, old_only: 17, recompute: r1}`，而 traced replacement 新输出已经有 `{val: new_val, tensor_meta: tm}`，最终关键状态是：

```text
val=new_val、tensor_meta=tm     来自 replacement trace
old_only 不存在                不做 old.meta 整体覆盖
recompute=r1                   由下面的显式标签传播写入
```

## recompute 与 ac_graph_id 怎样反向传播

这两个字段来自 activation checkpoint。`recompute` 告诉 AOTAutograd 的 partitioner 某个前向节点倾向在反向时重新计算还是必须保存；`ac_graph_id` 标出该节点属于哪一段 checkpoint 图，partitioner 用相邻节点的 id 判断 checkpoint 边界。替换若丢掉标签，新的等价节点可能被错误地保存或重算，因此 Replacer 要把旧输出标签反向传给构成新输出的内部节点。

双输出实验在第一个旧输出设置：

```text
recompute = r1
ac_graph_id = g1
```

对应的新输出是 `mul(add(x,1),2)`。`percolate_tags` 从新 mul 开始，用栈反向访问其 add，再看到捕获输入 x。x 位于 `input_stops=set(args)`，所以不写标签，也不继续越过它。

实际状态为：

```text
new add: recompute=r1, ac_graph_id=g1
new mul: recompute=r1, ac_graph_id=g1
captured x: 无这两项
第二个独立新输出 x+3: 无这两项
```

以第一项新输出 `new_mul(new_add(x,1),2)` 为例，一种栈状态如下：

| 轮次 | pop 前 queue | 弹出 | 动作 | 本轮后 visited / queue |
| ---: | --- | --- | --- | --- |
| 1 | `[new_mul]` | `new_mul` | 写标签，加入输入节点 `new_add` | `{new_mul}` / `[new_add]` |
| 2 | `[new_add]` | `new_add` | 写标签，加入输入节点 `x` | `{new_mul,new_add}` / `[x]` |
| 3 | `[x]` | `x` | `x in input_stops`，不写、不展开 | visited 不变 / `[]` |

常数 1、2 不是 FX Node，不会出现在 `all_input_nodes`。若两条新路径共享一个内部节点，第二次弹到它时 `visited` 会阻止重复展开。

`visited` 防止共享子图被反复处理。产品注释明确称这种 many-to-many 标签映射为 best effort；它可能把某些新节点标得过宽，不是精确的一一来源证明。

## 输出数相等与 packed 分支的决定

replacement 返回单 Node 时先包装成一元素列表。之后只有两个形状：

- `len(output_nodes)==len(replacement)`：用 `zip` 按位置逐项 replace，包括 None。
- 数量不等：断言旧输出恰好一个，把整份 replacement 序列交给 packed surgery。

若两个旧输出却只有一个新输出，会在数量不等分支因 `len(output_nodes)!=1` 断言失败；它不会猜测哪一个旧输出对应新值。packed 分支中的 getitem 索引也直接用于 `new[idx]`，越界或不适用索引会按 Python/序列行为失败。

## `apply` 最后补上的一层参数整理

`ReplacementPatternEntry.apply` 先断言注册时已经准备 `match.replacement_graph`，再执行：

```text
match.args/kwargs
    → normalize_args(...)
    → 按 replacement placeholder 顺序排列的原图 Node 序列
    → replace_with_graph(...)
```

传入的 `node` 不在这个方法里直接使用；真正的旧输出来自 `match.output_nodes()`。这使 many-to-many pattern 不必把“当前锚点”误当成唯一输出。

`normalize_args` 解决的是“pattern 捕获次序不一定等于 replacement placeholder 次序”。例如 Match 保存为：

```text
match.args = []
match.kwargs = {"right": y_old, "left": x_old}
replacement placeholder 顺序 = [left, right]
```

对应的规范化函数按 placeholder 名读取捕获值，返回 `[x_old,y_old]`。随后 `Replacer.run(*args)` 才能让第一个 placeholder 绑定 x、第二个绑定 y；若直接依赖 kwargs 的出现顺序，模板输入可能接反。

## 用一个变化检查理解

假设旧输出 O0、O1 对应新输出 N0、N1，O1 在图中更早。插入点不是参数顺序中的 O0，而是节点顺序更早的 O1；之后映射仍按列表位置执行 O0→N0、O1→N1。插入位置负责拓扑合法性，列表位置负责接口对应，两者不能混为一种顺序。

若 O0 带 recompute 标签、N0 依赖一条两节点链，而 N1 独立，标签只从 N0 沿其输入反向传播到捕获 args 前，不会因为 N1 与它同属一次 replacement 就自动传播到 N1。

至此，类字段、replacement 解释、插入点、四种 old/new 组合、逐位多输出、packed surgery、meta、tags、旧 users 迁移、残余旧节点删除和 `apply` 参数整理均已落到具体状态。
