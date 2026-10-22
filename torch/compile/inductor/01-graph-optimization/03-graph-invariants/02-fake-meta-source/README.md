# FakeTensorUpdater：只沿真正变化的节点刷新 FakeTensor

[返回图不变量](../README.md) · [上一篇：稳定拓扑排序](../01-stable-sort-source/README.md) · [返回图优化](../../README.md)

图改写会改变节点的输入，但 `node.meta["val"]` 不会随 `args` 自动变化。这里要解决的问题是：怎样找到可能过期的节点，重新执行它的 fake 运算，并且只在结果属性真的变化时继续检查下游。

`node.meta["val"]` 保存的 FakeTensor 没有真实元素数据，但带有 shape、stride、device 和 storage 关系等信息。例如 add 节点的真实输入可能位于 GPU，meta 中只需要两份 `[2,3]` 的 FakeTensor，便能推导 add 的结果也是 `[2,3]`。Lowering 和后续图规则读取这些信息，而不需要在编译时计算六个真实元素。

本叶子完整保留当前版本 `torch/_inductor/fx_utils.py` 中 `FakeTensorUpdater` 及其直接依赖的四个本地 helper：`get_storage`、`get_node_storage`、`get_fake`、`get_fake_args_kwargs`。源码顺序不变；新增说明统一以 `# 教学：` 开头。

## 先看实际实验中的一次变化怎样传播

原图是：

```text
x:[2,3] ─┐
          add(a):[2,3] ── relu(b):[2,3] ── return
y:[2,3] ─┘

c:[4,3]、d:[4,3] 起初未被 add 使用
```

构造 updater 后，实验把 `a.args` 从 `(x,y)` 改成 `(c,d)`。`id(a.args)` 改变，更新器用 c、d 的 fake 参数重新执行 `aten.add`：

```text
旧 a.fake：shape=[2,3]
新 a.fake：shape=[4,3]
```

shape 不同，所以更新器写回 `a.meta["val"]`，把 relu 的 id 放入 `to_process`；遍历到 relu 时，即使它自己的 hash 未变，也会用新的 a.fake 重算并写回 `[4,3]`。实际 dispatch 顺序是 `aten.add.Tensor, aten.relu.default`。紧接着第二次调用 updater，没有任何结构或上游 meta 再变化，实际执行列表为空。

这里传播的是 fake 属性变化，不是实际 tensor 数值变化。对照实验只把 target 从 add 改成 sub，shape、stride、device 与独占 storage 关系均保持等价：第一次只执行 sub，不执行 relu，传播在当前节点停止。由于没有写回，sub 的新 hash 也没有缓存；下一次调用 updater 时会再次执行 sub。

## fake 执行怎样得到新结果

`V.fake_mode` 是当前这次编译已经安装的 `FakeTensorMode`。它是一种 `TorchDispatchMode`：进入 `with V.fake_mode` 时，mode 被放到当前线程的 dispatch mode 栈上。随后调用 `aten.add`，dispatcher 在选择 CPU/CUDA backend kernel 之前先把这次调用交给 mode。FakeTensorMode 根据 FakeTensor 参数调用对应的 fake/meta 实现，只计算输出的 shape、stride、device 等属性，再包装成新的 FakeTensor。这个路径没有真实元素可读，也不会进入 add 的 CPU/CUDA 数据计算 kernel。

以 a 的新输入 c、d 为例：

```text
c.meta["val"] = FakeTensor(shape=[4,3], device=cuda)
d.meta["val"] = FakeTensor(shape=[4,3], device=cuda)

with V.fake_mode:
    aten.add(c.fake, d.fake)
        └─ dispatcher 由 FakeTensorMode 接管
             └─ 返回 FakeTensor(shape=[4,3], device=cuda)
```

updater 随后比较这份新 FakeTensor 与 a 原来的 `meta["val"]`。这里的“相同”只表示：当前比较器关心的属性足以让传播停止。它不是 FakeTensor 全部字段的完整相等关系；当前比较器没有检查 dtype，后文会用 add→eq 展示这个实现缺口。

## hash 只负责找出候选节点

构造 updater 时，每个节点记录四项：节点身份、target、`id(node.args)`、`id(node.kwargs)`。通常的 FX 改写会重新赋值：

```python
a.args = (c, d)       # 新 tuple，id(a.args) 改变
a.target = aten.sub   # target 改变
```

这两种修改都让 a 成为候选。hash 不递归检查容器内容。例如下面这种原地修改若保持外层 tuple 和内部 list 的身份不变，当前 hash 看不到列表元素已经替换：

```python
inputs = [x, y]
node.args = (inputs,)
# updater 已记录 id(node.args)
inputs[0] = c         # 外层 tuple 的 id 没变
```

FX 节点改写通常通过重新设置规范化后的 `args/kwargs` 容器完成；当前源码也直接标注这个 hash “Not a great hash function”。因此它是面向现有改写习惯的快速候选检查，不是任意嵌套 Python 对象的深度变化检测。

## 嵌套参数怎样换成 FakeTensor

一个 target 的参数可以同时含 tuple、list、dict、常量和 Node 引用。假设节点记录：

```text
args   = (x, [y, 2])
kwargs = {"other": {"z": z}}
```

`tree_map(get_fake, ...)` 递归进入这些容器，把有 meta 的 Node 换成对应 FakeTensor，常量 2 保持不变：

```text
args   = (x.fake, [y.fake, 2])
kwargs = {"other": {"z": z.fake}}
```

接着 `arg_tree_leaves` 把整棵参数树摊成叶子 `[x.fake, y.fake, 2, z.fake]`。若 y 没有 `meta["val"]`，`get_fake(y)` 会留下 FX Node y；叶子检查发现它后返回 `is_valid=False`，target 本轮不会执行。

## 完整源码

进入源码时，`graph` 是上面含 x、y、a、b、output 的 `torch.fx.Graph`；`a.args` 已从 `(x,y)` 改成 `(c,d)`。构造器只保存各节点修改前的结构 hash。producer 与 user 的关系早已记录在 FX 节点的 `node.users` 中。

调用 updater 后，`to_process` 从空集开始。遍历到 a 时，新 `args` tuple 造成 hash miss，所以 a 即使不在 `to_process` 也会重算；a 的 fake 结果从 `[2,3]` 变成 `[4,3]` 后，代码才把 b 的 id 加入 `to_process`。遍历到 b 时，它依靠这次点名重算。

源码中的符号比较可用两组值代入。旧 shape `[s0,3]`、新 shape `[s0,3]` 时，`sym_eq` 产生可静态证明为真的相等式，`is_intlist_same` 返回 true；若新 shape 是 `[s0+1,3]`，当前 ShapeEnv 不能证明 `s0==s0+1`，便判为变化并继续传播。这里的 s0 来自输入 shape，运行时能从输入的第 0 维取得，因此属于有来源的符号。

数据相关运算还可能产生 unbacked symbol：它的值不是某个输入维度，而要到运行时从该节点的结果中读取。假设节点旧结果是一份 shape `[u0,3]` 的 Tensor，`node.meta["unbacked_bindings"]` 保存的映射可以具体写成：

先看两次处理在写回点两侧的位置：

```text
新 fake 已判定不同
    ├─ 写回前：rebind_unbacked
    │            沿旧路径找到新符号 u1，把它重命名为旧 u0
    ├─ node.meta["val"] = new_fake
    └─ 写回后：compute_unbacked_bindings
                 在新结果树中记录 u0 的新取值路径
```

```text
{
    u0: (CallMethodKey("size"), SequenceKey(0))
}

沿路径读取：result.size()[0]
```

重新 fake 执行会新建符号 u1，但它仍出现在新结果的 `size()[0]`。`rebind_unbacked` 从旧 node meta 取出上面的映射，用 `key_get(new_result, path)` 得到 u1，再让 ShapeEnv 把 u1 重命名为旧符号 u0。随后 `compute_unbacked_bindings` 遍历新结果的 size、stride 和 storage_offset；返回非空映射时，updater 才覆盖 `node.meta["unbacked_bindings"]`。

若结果是 tuple，路径会先进入 tuple 项。例如新符号位于 `result[1].size()[0]`，路径就是 `(SequenceKey(1), CallMethodKey("size"), SequenceKey(0))`。这样后续运行时断言仍知道从结果树的哪里取得 u0。`_maybe_evaluate_static` 只在 ShapeEnv 已有约束足以算出表达式时返回具体结论，否则保留符号而不猜值。

```python
class FakeTensorUpdater:
    """
    The main idea here is that it's difficult to maintain accurate fake
    tensors (our primary form of metadata) for each node in our graph as we
    transform it.

    The most reliable way to obtain this information is by rerunning
    faketensor propagation. However, in general, faketensor propagation is
    fairly expensive. So, instead we'd like to only rerun faketensor
    propagation on nodes that have changed.

    In order to detect which nodes have changed, we first hash its node,
    target, and argument lists (which are immutable in FX).

    Then, whenever we call incremental_update, we check which FX nodes have a
    new hash, and recompute the faketensor metadata for that node. Then, we
    continue to recursively compute the faketensors for all users until the
    fake tensors stop changing.
    """

    def __init__(self, graph: torch.fx.Graph) -> None:
        # 教学：保存已经见过的结构指纹；集合里不是 tensor 数值。
        self.processed_hashes = set()
        self.graph = graph

        # 教学：在 pass 修改图以前，为每个现有节点建立基线。
        for node in self.graph.nodes:
            self.processed_hashes.add(self.hash_node(node))

    def hash_node(self, node: torch.fx.Node):
        # todo(chilli): Not a great hash function
        # 教学：target 改变或 args/kwargs 被新的不可变容器替换，指纹就改变。
        # 教学：参数中的实际 tensor 数值不是这里的检查对象。
        return (node, node.target, id(node.args), id(node.kwargs))

    def incremental_update(self):
        # 教学：当前源码保留了 processed 局部集合，但后续没有读取它。
        processed = set()
        # 教学：先统计旧 meta 中每个 storage 被多少节点引用，用于区分
        # 教学：无关的新 storage 与必须保持的 alias 关系。
        existing_storages: DefaultDict[Optional[int], int] = defaultdict(int)
        for node in self.graph.nodes:
            existing_storages[get_node_storage(node)] += 1

        def is_intlist_same(new, old):
            # 教学：shape/stride 可以含 SymInt；只有静态可证明相等才算相同。
            return statically_known_true(sym_eq(new, old))

        def is_fake_tensor_same(new, old):
            # 教学：先保持结果外层类型；tensor、tuple 和符号标量不可混用。
            if type(new) != type(old):
                return False
            if isinstance(new, (list, tuple)):
                if len(new) != len(old):
                    return False
                return all(
                    is_fake_tensor_same(new_i, old_i) for new_i, old_i in zip(new, old)
                )
            if new is None:
                return old is None
            if not isinstance(new, torch.Tensor):
                # 教学：非 tensor 结果只接受三种符号标量，再比较符号表达式。
                assert isinstance(
                    new, (torch.SymInt, torch.SymBool, torch.SymFloat)
                ), f"Unknown type {type(new)} in {self.graph}"
                return (
                    new.node.shape_env._maybe_evaluate_static(
                        sympy.Eq(new.node.expr, old.node.expr)
                    )
                    == sympy.true
                )
            # 教学：tensor 先比较 shape 和 layout；任一变化都影响消费者。
            if not is_intlist_same(new.shape, old.shape) or new.layout != old.layout:
                return False
            # 教学：strided tensor 还要比较 stride 与 storage_offset。
            if new.layout == torch.strided and (
                not is_intlist_same(new.stride(), old.stride())
                or not statically_known_true(
                    new.storage_offset() == old.storage_offset()
                )
            ):
                return False

            if new.device != old.device:
                return False

            # 教学：同一底层 storage 表示 alias 身份未变。
            if get_storage(new) == get_storage(old):
                return True

            # This is the case where it returns a completely fresh storage that's used nowhere else.
            # 教学：旧 storage 只被一个节点持有，且新 storage 从未出现时，
            # 教学：两者虽编号不同，但都只是无共享关系的独占结果，可视为相同。
            if (
                existing_storages[get_storage(old)] == 1
                and get_storage(new) not in existing_storages
            ):
                return True
            return False

        def should_process_node(node):
            # node.target for nodes returning true from this function
            # are called under fake mode and does not work for inductor
            # lowerings. We check if the node.target is an aten operator
            # or operator.getitem which is used when returning multiple
            # tensors from an op.
            # 教学：placeholder、output、call_module 和普通 Python call_function
            # 教学：不在本 updater 的直接 fake 执行范围内。
            return node.op == "call_function" and (
                isinstance(node.target, torch._ops.OpOverload)
                or node.target == operator.getitem
            )

        # 教学：集合中存 node 的 id；某节点 meta 变化后，其 users 的 id 加入这里。
        to_process = set()
        for node in self.graph.nodes:
            # 教学：结构指纹没变且未被变化的上游点名，就直接跳过。
            if (
                self.hash_node(node) in self.processed_hashes
                and id(node) not in to_process
            ):
                continue

            if not should_process_node(node):
                continue

            # 教学：把 Node 引用替换成其 meta["val"]；缺少任何 fake 值就不能执行。
            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            # 教学：运行的是 fake 运算，只产生抽象结果，不读取真实 tensor 数据。
            with V.fake_mode:
                new_fake_tensor = node.target(*args, **kwargs)
            # 教学：若受比较的属性及 alias 关系都相同，传播在此停止。
            if "val" in node.meta and is_fake_tensor_same(
                new_fake_tensor, node.meta["val"]
            ):
                continue

            # 教学：新结果可能含新的 unbacked symbol，先把旧节点关系重新绑定。
            rebind_unbacked(V.fake_mode.shape_env, node, new_fake_tensor)

            node.meta["val"] = new_fake_tensor
            if (shape_env := V.fake_mode.shape_env) and (
                symbol_to_path := compute_unbacked_bindings(shape_env, new_fake_tensor)
            ):
                # Refresh the bindings to the new symbols
                # 教学：保存“符号由结果树中哪条路径产生”的新映射。
                node.meta["unbacked_bindings"] = symbol_to_path

            # 教学：更新 storage 计数，使同一次遍历后面的 alias 比较看到新结果。
            existing_storages[get_node_storage(node)] += 1

            # 教学：只有 meta 真变化才点名所有直接消费者；图的线性拓扑顺序
            # 教学：保证这些 user 稍后才会在同一轮遍历中出现。
            to_process.update([id(user) for user in node.users])

            # 教学：记录成功写回后的新结构指纹，供以后 incremental_update 跳过。
            self.processed_hashes.add(self.hash_node(node))


def get_storage(t: torch.Tensor) -> int:
    # 教学：_cdata 是底层 untyped storage 的身份标识，用于比较共享关系。
    return t.untyped_storage()._cdata


def get_node_storage(node: torch.fx.Node) -> Optional[int]:
    # 教学：没有 val、val 不是 tensor、或 tensor 没有 storage 时统一交回 None。
    if "val" not in node.meta:
        return None
    if not isinstance(node.meta["val"], torch.Tensor):
        return None
    if not torch._C._has_storage(node.meta["val"]):
        return None
    return get_storage(node.meta["val"])


def get_fake(x):
    # 教学：树中的 Node 若已有 val 就换成 fake 值；缺失时保留 Node，
    # 教学：让调用方识别“参数尚未准备完整”。常量原样返回。
    if isinstance(x, torch.fx.Node):
        if "val" not in x.meta:
            return x
        return x.meta["val"]
    return x


def get_fake_args_kwargs(x: torch.fx.Node) -> Tuple[bool, Tuple[Any], Dict[str, Any]]:
    """
    First value returns a boolean if any of the input nodes don't have a faketensor.
    """
    # 教学：tree_map 同时进入嵌套 tuple/list/dict，替换所有可用 Node 引用。
    args, kwargs = tree_map(get_fake, (x.args, x.kwargs))
    # 教学：替换后仍残留 Node，说明至少一个输入没有 meta["val"]。
    if any(
        isinstance(a, torch.fx.Node) for a in pytree.arg_tree_leaves(*args, **kwargs)
    ):
        return False, args, kwargs
    return True, args, kwargs
```

## 用三步状态表走完传播

设初始节点均已有正确 meta，构造 updater 后把 `a.args` 从 `(x,y)` 换成 `(c,d)`，其中 x、y 是 `[2,3]`，c、d 是 `[4,3]`：

| 遍历位置 | hash / 待处理状态 | fake 执行与比较 | 写回后的动作 |
| --- | --- | --- | --- |
| x、y、c、d | 原节点 hash 已记录；不在 `to_process` | 不执行 | 跳过 |
| a | 新 `args` tuple 令 `id(a.args)` 改变 | `aten.add` 得 `[4,3]`，不同于旧 `[2,3]` | 写回 a；把 `id(b)` 加入 `to_process` |
| b | 自身 hash 没变，但 id 在 `to_process` | relu 用新 a.fake 得 `[4,3]` | 写回 b；点名其 users |

若 b 后面还有一个 sum user，它会继续按新的 b.fake 推导输出。若某一层重新执行后比较结果不变，那一层不会点名自己的 users，传播在该处结束。

这个循环本身没有工作队列回跳。它依赖图已经按拓扑顺序排列：上游变化时，user 尚未被遍历。稳定拓扑排序与本 updater 因此是前后相接的两项职责。

## storage 比较为什么不能只看编号

fake 运算每执行一次，普通独占输出可能得到一个全新的 storage 编号。若只比较编号，`add` 即使 shape、stride、device 全都没变，也会被误判为变化并传播到所有下游。

先把 storage 与节点关系画具体。x 的 FakeTensor 拥有 storage S；`view(x)` 的结果 v 仍指向 S，因此构造 updater 时计数为 `S:2`：

```text
x.meta["val"] ─┐
                 ├─ storage S
v.meta["val"] ─┘
```

若 v 的 target 改成 clone，重新 fake 执行得到独立 storage C：

```text
x.meta["val"] ── storage S
new_v.fake    ── storage C
```

旧关系是共享，新的关系是独占，所以消费者看到的 alias 合同已经变化。相反，普通 add 的旧输出独占 storage A，重算后的新输出独占 storage B；A、B 编号不同，但二者都没有与其他节点共享，传播所需的 alias 关系没有变化。

当前实现把以下两种情况看作 storage 关系相同：

1. 新旧结果确实指向同一 storage；
2. 旧 storage 只出现一次，新 storage 在旧图中从未出现——二者都是无人共享的独占结果。

若旧结果与另一个节点共享 storage，第一项计数大于 1；重新执行后换成新 storage 就不能使用第二条豁免，因为 alias 关系已经改变。消费者可能依赖视图和修改语义，必须继续更新。

`None` 也会进入 `existing_storages` 计数：缺 val、非 Tensor val、无 storage Tensor，以及 tuple/list val 的节点都会由 `get_node_storage` 记在 None 名下。这个初始计数 helper 有 `_has_storage` 防护；`is_fake_tensor_same` 的 Tensor 分支则直接调用 `get_storage(new/old)`，当前实现假定走到这里的新旧 FakeTensor 都具有 storage。

tuple/list 输出会逐项递归比较，但初始计数不会递归统计其中 Tensor 的 storage。假设一个节点旧 val 是 `(t0, t1)`，这个节点只给 `existing_storages[None]` 加一；t0、t1 的 storage 计数通常仍为 0。重算产生两份新的独占 storage 后，成员比较无法满足“旧 storage 计数等于 1”，于是保守地判为变化并传播。

独立实验还把 `view(x,[2,3])` 改成 `clone(x)`。旧 view 与输入 x 共享 storage，因此旧 storage 计数大于 1；clone 产生新的独占 storage，不能使用“旧 storage 只出现一次”的豁免。更新器实际执行 `aten.clone.default`，判定 alias 关系变化，再执行下游 `aten.relu.default`。记录从 `old_shared=True` 变成 `new_shared=False`。

## 四个容易混淆的边界

- hash 检查结构身份：节点、target、`args/kwargs` 容器；它不比较真实数据。
- fake 执行检查抽象结果：Python 外层类型、shape、layout、stride、offset、device、符号表达式和 storage 关系。当前比较没有 dtype；实验把 add 改成 eq，fake dispatch 确实产生 bool 结果，但 updater 将其判为相同，节点 meta 仍是旧 `float32`，下游也未被点名。这是当前实现边界，不是期望语义。
- `should_process_node` 限定可直接重新执行的 target；不支持的节点与 fake 参数不完整的节点会跳过。
- `processed_hashes` 只在构造时和“meta 发生变化并写回”后加入。add→sub 的第一次调用执行 sub 后判 meta 相同；新 hash 未加入集合，第二次调用又实际执行一次 sub。shape 改变并写回的案例会缓存新 hash，第二次执行列表为空。

输入参数树与输出比较器的支持范围不同。`tree_map` 可以从输入 tuple/list/dict 中取 fake 参数；`is_fake_tensor_same` 对输出只递归 list/tuple。普通 Python `int`、`bool`、`float` 结果也不在可比较集合中：除 `None` 外，非 Tensor 只接受 `SymInt`、`SymBool`、`SymFloat`，其他类型会触发 assert。输出 dict 同样会进入这条断言路径。

“输入缺 val”和“当前节点旧 val 缺失”有不同出口：输入 Node 缺 val 时 target 无法执行，本轮跳过；输入都完整而当前节点自己没有旧 `meta["val"]` 时，target 可以执行，新 fake 直接进入写回分支并点名 users。

unbacked binding 也有一个空结果出口：若旧 meta 含 `unbacked_bindings`，而新 fake 不再产生任何 unbacked symbol，`compute_unbacked_bindings` 返回空值，条件赋值不会执行。当前代码不会在这里删除旧键；正文只把非空返回称为“刷新 bindings”。

此外，实验删除一个输入的 `meta['val']` 后，`get_fake_args_kwargs` 留下 FX Node 并返回 false，目标 sub 完全没有执行；把 target 改成普通 Python `operator.add` 时，`should_process_node` 也直接跳过。这两条路径同样不会记录新 hash，条件修复后后续调用仍有机会重试。

device 比较位于 storage 豁免之前。设旧 a.fake 为 CPU `[4,3]`，重新 fake 执行得到 CUDA `[4,3]`；即使两者 shape、layout、stride、offset 都相同，新旧 device 仍不同。比较器立即返回 false，updater 写回 CUDA fake，并点名 a 的 users。独占新 storage 的豁免只处理“编号变了但共享关系未变”，不会掩盖 device 变化。

同一轮遍历写回新 fake 后，`existing_storages` 只增加新 storage 的计数，没有扣减旧 storage。这会保守地保留旧共享记录：后面的节点可能因此无法使用“旧 storage 只出现一次”的豁免，从而多传播一次；它不会把已经共享的旧 storage误判为独占。比较函数本身对 Tensor 直接调用 `get_storage`，所以当前实现还假定进入 Tensor 比较分支的 fake 结果具有 storage；只有建立初始计数的 `get_node_storage` 会为无 storage Tensor 返回 None。

## 用新情况检验理解

假设 `a` 从 `transpose(x,0,1)` 改为直接使用 x，且二者 shape 恰好都是 `[3,3]`。shape 相同还不足以停止：转置结果通常 stride 为 `[1,3]`，连续 x 为 `[3,1]`，`is_fake_tensor_same` 会判不同，写回 a 并传播 users。

再假设只把 add 的一个独占 fake 输出换成另一份独占 storage，而 shape、layout、stride、offset、device 均相同。旧 storage 计数为 1、新 storage 未出现，比较会判相同，传播停止。若旧 storage 同时被 view 节点共享，计数不再为 1，结果会判不同并继续传播。

到这里，Q13 的两个实现叶子已经接起来：稳定排序保证“先上游后 user”，FakeTensorUpdater 在这条顺序上只刷新受影响的抽象属性。下一项按广度优先队列进入 GraphLowering 的节点解释入口。
