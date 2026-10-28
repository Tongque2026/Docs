# LoopBody 捕获与重放源码：符号索引怎样交给当前 handler

[返回虚拟化 ops 篇](../README.md) · [返回 Codegen](../../README.md) · [返回总览](../../../README.md)

Q41 已经说明“同一张 FX 小图换 handler 就得到不同目标表达”。本篇把这句话落实成源码状态：捕获阶段怎样建立图、索引表和 submodule；重放阶段怎样把新循环符号代入索引，再把当前动态 handler 填进图的 `ops` placeholder。

真实生命周期是先捕获、后重放：调用方用原 callable、捕获参数和符号范围构造 `LoopBody`；构造函数先建立各个容器，再创建 `root_block`，而 `LoopBodyBlock(...)` 的构造过程会立即执行 callable 并捕获 FX 图。以后 backend 调用同一个 `LoopBody`，才进入 K01 的重放链。

```python
def __init__(self, fn, args, var_ranges, iter_vars, reduce_vars):
    super().__init__()

    _flat_sizes = tuple(var_ranges.values())
    self.sizes = (
        _flat_sizes[: len(iter_vars)],
        _flat_sizes[len(iter_vars) :],
    )

    self.iter_vars = iter_vars
    self.reduce_vars = reduce_vars
    self.var_ranges = var_ranges

    if isinstance(fn, LoopBody):
        self._init_with_copy(fn, args)
    else:
        self._init_with_tracing(fn, args)

    self.indexing = None

def _init_with_tracing(self, fn, args):
    """Do an FX trace of an arbitrary callable to construct self"""
    self.indexing_exprs = {}
    self.indexing_exprs_name = {}
    self.submodules = {"get_index": self.get_index}
    self.subblocks = {}
    self.indirect_vars = []
    self.indirect_var_ranges: Dict[sympy.Symbol, sympy.Expr] = {}
    self.memory_usage = {t: [] for t in MemoryUsageType}
    self.root_block = LoopBodyBlock(self, fn, args)  # traces
    del self.indexing_exprs_name  # not used after _init_with_tracing

def _init_with_copy(self, other: LoopBody, args):
    """
    _init_with_tracing() is slow, so this is a fast path in the case
    where we are just reordering/merging/splitting the args of an
    existing LoopBody.
    """
    indexing_exprs = other.indexing_from_args(args)
    self.indexing_exprs = {
        name: V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges)
        for name, expr in indexing_exprs.items()
    }
    self.subblocks = {k: v.clone(self) for k, v in other.subblocks.items()}
    self.indirect_vars = other.indirect_vars
    self.indirect_var_ranges = other.indirect_var_ranges
    self.memory_usage = other.memory_usage
    self.root_block = other.root_block.clone(self)

    submodules = {**other.submodules}
    submodules.pop("get_index")
    self.submodules = {
        "get_index": self.get_index,
        **{k: v.clone(self) for k, v in submodules.items()},  # type: ignore[attr-defined]
    }
```

普通 callable 走 trace 路径：容器先置空，`root_block = LoopBodyBlock(...)` 随即触发捕获，成功返回后才删除只在捕获期使用的反向名字表，并最终把 `indexing` 设为 `None`。若构造中途失败，`LoopBody` 可能只写到一部分字段，甚至尚未执行最后的 `indexing=None`；源码没有回滚。

已有 `LoopBody` 走 copy 路径：root/subblock/shim 会重新绑定到新父对象，但 `indirect_vars`、`indirect_var_ranges` 和 `memory_usage` 直接复用旧对象的引用，并非整份状态深拷贝。后文解释 clone 时只对重新绑定的部分作承诺。

`var_ranges` 还是一张有顺序的映射。最终用于代码生成的常规路径通过 `index_vars_no_squeeze(iter_sizes, reduce_sizes)` 建立它，插入顺序固定为全部逐点循环变量在前、全部规约变量在后；backend 重放也必须按同一位置顺序传入两组 indices。`LoopBody` 自己只检查扁平参数总数是否等于 `len(var_ranges)`，不会验证“前几项确实是逐点变量、后几项确实是规约变量”；自定义调用者若传入同样数量但顺序颠倒的符号，会得到错误替换而不是专门报错。

初始的 squeeze 路径还有一个临时边界：大小为 1 的维可能在 `iter_vars/reduce_vars` 中表现为常量 0，却不占 `var_ranges` 的符号项，所以此时仅按 `len(iter_vars)` 切 `sizes` 不能作为最终分类依据。调用方随后按实际符号重新分类，并用 no-squeeze helper 重建最终 body。本文后续重放案例都使用这份重建后的有序 `var_ranges`。

现在固定捕获成功后的 `LoopBody` 状态。捕获把原 callable 的操作保存到这些字段；重放读取同一批字段，只临时改动 `indexing`：

| 字段 | 捕获后保存的内容 | 重放时怎样使用 |
| --- | --- | --- |
| `var_ranges` | 捕获符号及其范围，例如 `{x0:2, x1:128}` | 确定新循环参数应有几维，并建立符号替换表 |
| `indexing_exprs` | 有名字的索引式，例如 `index0 = 128*x0+x1` | 把 x0/x1 换成本次 backend 传入的 c0/c1 |
| `memory_usage` | 哪个 buffer 以哪个索引被 load、store 或规约写 | 依赖分析可直接读取，不必执行算术小图 |
| `root_block` | 普通 load、算术、store 组成的 FX 小图 | 用当前 ops handler 重放 |
| `submodules`、`subblocks` | masked、scan、间接索引所需的 shim 和嵌套小图 | FX `call_module` 节点调用它们，把复合操作交给当前 handler |
| `indirect_var_ranges` | 从 Tensor 值产生的间接索引符号及范围 | 重放到设置节点时，用当前值替换占位符 |
| `indexing` | 捕获结束时为 `None` | 一次重放期间保存已经代换的索引表；正常返回后恢复为 `None` |

例如逐点计算 `relu(x+y)*2` 会留下 root FX 图、输入/输出的 memory usage 和 `index0=x0`；backend 传入自己的循环变量 c0 后，临时表变为 `index0=c0`，同一张图中的 load 和 store 都读取它。

## K01：重放链先代换索引，再运行 FX interpreter

### InterpreterShim 保留当前 FX node，并安装动态 interpreter

`torch.fx.Interpreter` 是按 FX 节点顺序执行处理方法的父类；这里的 `InterpreterShim` 不运行用户 Tensor 计算，而是执行 LoopBody 捕获的小图。构造时，`graph` 是这张小图，`submodules` 是其中 `get_attr` 名字到对象的字典。`_dummy_gm()` 只提供父类要求的临时 GraphModule，随后 `self.graph=graph` 换成真正要重放的小图。`current_node` 起初为 `None`，每执行一项前更新为该 FX node。

```python
class InterpreterShim(torch.fx.Interpreter):
    @staticmethod
    @functools.lru_cache(None)
    def _dummy_gm():
        return torch.fx.symbolic_trace(identity)

    def __init__(self, graph, submodules):
        # call super() with a placeholder to avoid constructing a
        # GraphModule which is very expensive (it does codegen).
        super().__init__(self._dummy_gm(), garbage_collect_values=False)
        self.module = self  # type: ignore[assignment]
        self.graph = graph
        self.submodules = submodules
        self.extra_traceback = False
        self.fetch_attr = submodules.__getitem__  # type: ignore[method-assign]
        self.current_node = None

    def run_node(self, n: torch.fx.Node) -> Any:
        # 教学：先公布当前 FX node，handler 处理失败时可把错误归因到这里。
        self.current_node = n
        return super().run_node(n)

    def run(self, *args, **kwargs):
        # 教学：重放期间 V.interpreter 指向本解释器；退出或异常时由上下文恢复。
        with V.set_interpreter_handler(self):
            return super().run(*args, **kwargs)
```

`_dummy_gm` 只为满足父类构造接口，并由无参数 LRU cache 复用；真正执行的 `graph` 随后直接替换进去，`fetch_attr` 也改为从捕获期保存的 `submodules` 取对象。关闭 FX interpreter 的 value garbage collection，使重放中的引用按这份小图所需保留。

每执行一个 FX node，`current_node` 先更新；整个 `run` 动态安装自身为 interpreter handler。若内部抛异常，上下文仍恢复进入前的 handler，但 `current_node` 会停在失败节点，供错误定位使用。

### LoopBody 把新循环位置代入捕获索引

本次 LoopBody 在捕获时保存 `var_ranges={x0:2,x1:128}`、`indexing_exprs={"index0":128*x0+x1}`。CPU backend 重放时传入 `indices=[[c0,c1]]`；摊平后长度仍为 2，替换表是 `{x0:c0,x1:c1}`，返回 `{"index0":128*c0+c1}`。

```python
def indexing_from_args(self, indices):
    index = [*itertools.chain.from_iterable(indices)]
    assert len(index) == len(self.var_ranges), (index, self.var_ranges)
    assert all(
        v not in self.var_ranges for v in index
    ), f"{self.var_ranges=}, {indices=}"
    replacements = dict(zip(self.var_ranges.keys(), index))
    return {
        name: sympy_subs(expr, replacements)
        for name, expr in self.indexing_exprs.items()
    }

def __call__(self, *indices):
    self.indexing = self.indexing_from_args(indices)
    result = self.root_block()
    self.indexing = None
    return result
```

假设捕获时：

```text
var_ranges = {x0:2, x1:128}
indexing_exprs = {index0:128*x0+x1, index1:x0}
```

backend 用新符号 `c0,c1` 重放并调用 `body([c0],[c1])`。flatten 后是 `[c0,c1]`，replacement 为 `{x0:c0,x1:c1}`，得到：

```text
self.indexing = {index0:128*c0+c1, index1:c0}
```

FX 图执行 `get_index("index0")` 时，`LoopBody.get_index` 就从这张临时表返回 `128*c0+c1`；load handler 因而看到本轮 backend 的符号，而非捕获期的 x0/x1。root 返回后表被设回 None，避免下一次重放误用旧参数。

第一个 assert 防止少传/多传循环维；第二个 assert 防止把原 `var_ranges` 中的同一符号原样当作新参数，避免替换关系自指。当前实现没有 `try/finally` 包住 `root_block()`：若重放中抛异常，`self.indexing=None` 这句不会执行。因此正常重放能恢复，异常后的同一对象可能保留临时表；这是源码边界，不能泛称所有状态都会自动清理。`InterpreterShim.run` 在重放期间安装的 `V.interpreter` 会由 context manager 恢复；这段重放代码没有安装新的 `V.ops`，只是把调用方当前 handler 作为 placeholder 实参，并让 shim 读取调用方已有的 `V.ops`。若外层调用方另用 ops context，它的恢复也由外层负责。

### LoopBodyBlock 把当前 ops handler 作为 placeholder 参数

```python
def __call__(self):
    graph = self.graph
    submodules = self.body.submodules

    return InterpreterShim(graph, submodules).run(V.get_ops_handler())
```

捕获图的第一个 placeholder 名为 `ops`。重放时唯一显式参数是当前 `V.get_ops_handler()`：CPU handler、Triton handler或分析 handler 均从这里进入同一张图。图内的 `call_module("get_index",...)` 则经 `fetch_attr` 找到 `LoopBody.get_index`，读取上一步设置的 indexing。

完整调用链是：

```text
LoopBody.__call__([c0],[c1])
    -> indexing_from_args: index0=128*c0+c1
    -> root_block.__call__
       -> InterpreterShim.run(current ops handler)
          -> placeholder ops = 当前 handler
          -> call_module get_index("index0") = 128*c0+c1
          -> call_method load/add/reduction/store ...
    -> self.indexing=None
```

### K01 理解验收

若原索引是 `4*x0+x1`，范围两维，但只传 `[c0]`，在执行 FX 图前就因长度 1≠2 assert。若传 `[c0,c1]`，CPU 与 Triton 重放得到同一符号索引 `4*c0+c1`，差别从 placeholder 接到的 handler 开始。

K01 到此完成。K02 打开捕获期 `LoopBodyBlock.__init__` 的全部操作分支。

## K02：CaptureIndexing 同时建立 FX 图和 memory usage

下面是 `LoopBodyBlock.__init__` 的完整实现。此次 `fn([x0])` 依次发出 `load("x",x0)`、加 1、`store("buf0",x0,...)`；`args=[x0]`，memory_usage 与 FX graph 初始为空，配置关闭额外的 IndexPropagation。局部 handler 包住 FX proxy handler：第一次 `add_index(x0,LOAD,name="x")` 把 x0 登记成 `index0`，再让内层 proxy 把 get_index、load、add、store 写进图；出口同时留下 LOAD/STORE 两条 memory usage。

产品源码先写局部函数和局部类，后创建它们运行时需要的 tracer。阅读时要区分“定义”与“执行”：定义 `add_index` 和 `CaptureIndexing` 只建立函数/方法对象，方法体中的 `tracer.create_proxy` 此时尚未运行。

真正执行 callable 前还会套两层索引处理器。`SimplifyIndexing(inner,var_ranges)` 接收原 ops 调用和循环变量范围；遇到 load、store、store_reduction、index_expr 或 check_bounds 时，它先用范围化简 SymPy 索引，再把同一操作交给 `inner`，这里的 inner 就是 CaptureIndexing。可选的 `IndexPropagation(inner,...)` 放在最外层：它尝试让常量和符号 `index_expr` 穿过中间算术；能保持为符号式的结果在交回 inner 时变成 constant 或 index_expr，其余操作解包后直接转发。本例关闭 IndexPropagation，所以实际路径是 `fn → SimplifyIndexing → CaptureIndexing → proxy_ops`。

完整的捕获顺序是：

```text
1. tracer = torch.fx.Tracer()，并给它一张空 FX graph
2. proxy_ops = tracer.create_proxy("placeholder", "ops", ...)
   graph 此时已有第一个节点：placeholder ops
3. CaptureIndexing(proxy_ops)
   WrapperHandler.__init__ 把 proxy_ops 保存为 CaptureIndexing._inner
4. SimplifyIndexing(CaptureIndexing(...), var_ranges)
   本例的动态 ops handler 指向这层；索引化简后才进入 CaptureIndexing
5. 执行 fn(*args)
   fn 调用 load/store 时，才进入前面定义的方法并使用已经存在的 tracer
6. ops.output(...) 建立 output 节点，self.graph = tracer.graph
```

所以第一次执行 `add_index` 时，它写入的正是第 1 步创建的 graph；`self._inner.load(...)` 则在 `proxy_ops` 上记录 `call_method("load",...)`，而不是读取真实 Tensor。

这段源码里还有两个不同的 `self` 来源。`LoopBodyBlock.__init__` 的参数 `self` 是正在创建的 block；类体中的 `self.name = "CaptureIndexing"` 也读取这个外层 block，并不创建 handler 字段。局部类的普通实例方法另有自己的 `self`，它指向 `CaptureIndexing` handler，并通过 `self._inner` 调用第 3 步保存的 proxy handler。三个 `@staticmethod` 没有 handler 实例参数，所以方法体里的 `self` 仍是从外层 `LoopBodyBlock.__init__` 闭包捕获的 block；`self.body` 因而指向 block 所属的 `LoopBody`。`staticmethod` 保持 masked、scan、indirect_indexing 的 ops 调用签名不被额外插入 handler 参数。

```python
def __init__(self, body: LoopBody, fn: Callable[..., Any], args: List[Any]):
    self.body = body

    def add_index(expr: sympy.Expr, mtype: MemoryUsageType, **kwargs):
        return tracer.create_proxy(
            "call_module",
            "get_index",
            (body.add_index_expr(expr, mtype, **kwargs),),
            {},
        )

    class CaptureIndexing(V.WrapperHandler):  # type: ignore[name-defined]
        self.name = "CaptureIndexing"

        def load(self, name: str, index: sympy.Expr):
            index = add_index(index, MemoryUsageType.LOAD, buffer_name=name)
            return self._inner.load(name, index)

        def load_seed(self, name: str, index: int):
            assert isinstance(index, int)
            body.add_index_expr(
                sympy.Integer(index), MemoryUsageType.LOAD_SEED, buffer_name=name
            )
            return self._inner.load_seed(name, index)

        def store(self, name, index, value, mode=None):
            index = add_index(
                index, MemoryUsageType.STORE, buffer_name=name, mode=mode
            )
            return self._inner.store(name, index, value, mode)

        def store_reduction(self, name, index, value):
            index = add_index(
                index, MemoryUsageType.STORE_REDUCTION, buffer_name=name
            )
            return self._inner.store_reduction(name, index, value)

        def reduction(self, dtype, src_dtype, reduction_type, value):
            result = self._inner.reduction(dtype, src_dtype, reduction_type, value)
            if "welford" in reduction_type:
                return tuple(result[i] for i in range(3))
            return result

        def index_expr(self, index, dtype):
            if isinstance(index, (int, sympy.Integer)):
                return self._inner.constant(int(index), dtype)
            index = add_index(index, MemoryUsageType.INDEX_EXPR)
            return self._inner.index_expr(index, dtype)

        def check_bounds(self, index, size, lower, upper):
            index = add_index(index, MemoryUsageType.CHECK_BOUNDS)
            size = add_index(size, MemoryUsageType.CHECK_BOUNDS)
            return self._inner.check_bounds(index, size, lower, upper)

        def bucketize(
            self,
            values,
            offsets_name: str,
            offsets_size: sympy.Expr,
            indexing_dtype: torch.dtype,
            right: bool,
        ):
            offsets_size = add_index(
                offsets_size, MemoryUsageType.BUCKETIZE, buffer_name=offsets_name
            )
            return self._inner.bucketize(
                values, offsets_name, offsets_size, indexing_dtype, right
            )

        @staticmethod
        def masked(mask_proxy, masked_body: Callable[..., Any], other_proxy):
            """
            Recursively capture the masked out body in another LoopBodyBlock
            """
            name = self.body.add_submodule(None, "masked_subblock")
            self.body.submodules[name] = self.body.bind_masked_shim(name)
            self.body.subblocks[name] = LoopBodyBlock(self.body, masked_body, [])
            return tracer.create_proxy(
                "call_module", name, (mask_proxy, other_proxy), {}
            )

        @staticmethod
        def scan(
            dtype_proxy,
            combine_fn: Callable[
                [Tuple[Any, ...], Tuple[Any, ...]], Tuple[Any, ...]
            ],
            value_proxy,
        ):
            shim = self.body.bind_scan_shim(combine_fn)
            name = self.body.add_submodule(shim, "scan")
            result = tracer.create_proxy(
                "call_module",
                name,
                (dtype_proxy, value_proxy),
                {},
            )
            # Proxies are iterable, but some methods expect tuples/lists
            return tuple(result[i] for i in range(len(value_proxy)))

        def sort(self, dtypes, values, stable, descending):
            result = self._inner.sort(dtypes, values, stable, descending)
            # Proxies are iterable, but some methods expect tuples/lists
            return tuple(result[i] for i in range(len(values)))

        def frexp(self, value_proxy):
            result = self._inner.frexp(value_proxy)
            # Proxies are iterable, but some methods expect tuples/lists
            return (result[0], result[1])

        @staticmethod
        def indirect_indexing(index_proxy, size, check=True, wrap_neg=True):
            """
            Flow data from tensors into indexing formulas.
            Introduce a call_module to update the indexing.
            """

            var = self.body.add_indirect(size)
            set_indirect = self.body.bind_set_indirect_shim(
                var, size, check, wrap_neg
            )
            tracer.create_proxy(
                "call_module",
                self.body.add_submodule(set_indirect, f"set_{var}"),
                (index_proxy,),
                {},
            )
            return var

        @staticmethod
        def output(result):
            tracer.create_proxy("output", "output", (result,), {})

    tracer = torch.fx.Tracer()
    tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
    proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})

    from .index_propagation import IndexPropagation
    from .sizevars import SimplifyIndexing

    handler: Any = SimplifyIndexing(
        CaptureIndexing(proxy_ops), self.body.var_ranges
    )
    if config.constant_and_index_propagation:
        handler = IndexPropagation(
            handler, self.body.var_ranges, self.body.indirect_var_ranges
        )

    with V.set_ops_handler(handler):
        # This indirection is just a cute way to get IndexPropagation to
        # unwrap the return value.
        ops.output(fn(*args))
    self.graph = tracer.graph
```

### 普通内存动作同时留下两份信息

以 `load("x",128*x0+x1)` 为例，`add_index` 先调用 `body.add_index_expr`：索引式得到稳定名字，例如 `index0`，并在 `memory_usage[LOAD]` 加 `(index0,"x",None)`。随后 tracer 在 FX 图创建 `call_module get_index("index0")`，这个 proxy 再作为 `_inner.load` 的索引参数。

所以捕获出口同时有：

```text
indexing_exprs[index0] = 128*x0+x1
memory_usage[LOAD] += MemoryEntry(index0,"x",None)
FX graph += get_index(index0) -> ops.load("x",proxy_index)
```

store 同样保存 mode；原子等写模式不会因捕获而丢失。`load_seed` 要求 index 已是 Python int，保存前转成 SymPy Integer，但传给 FX proxy 的仍是原 int。`store_reduction` 使用独立 usage 类型，使依赖快速路径能恢复规约写。

### index、规约和多结果为何分支处理

- `index_expr(3,dtype)` 不需要符号代换，直接捕获成 `constant(3,dtype)`；符号式才登记 `INDEX_EXPR` 并经 get_index。
- `check_bounds` 的 index 与 size 都登记成 `CHECK_BOUNDS`；它们是两个可代换表达式。
- `bucketize` 把 offsets_size 登记为 BUCKETIZE，并把 offsets buffer 名一起保存；values 仍按普通 proxy 传入。
- 普通 reduction 返回一个 proxy；名称含 `welford` 时语义上有 mean/m2/weight 三项，显式取 `result[0:3]` 形成 Python tuple。
- FX Proxy 可迭代，但下游有些接口要求真实 tuple/list，所以 scan 按输入 value 数量拆项，sort 按 values 数量拆项，frexp 固定拆成两项。

捕获尾部先创建 `ops` placeholder，再组合 `CaptureIndexing→SimplifyIndexing`。`SimplifyIndexing` 在 load、store、store_reduction、index_expr 和 bounds 进入 CaptureIndexing 前，利用 `var_ranges` 化简其中的 SymPy 索引；例如范围已经证明某个取模多余时，登记的是化简后的式子。配置开启时，最外层再包 `IndexPropagation`：它尝试把常量和 `index_expr` 的符号值继续穿过算术操作，能够保持为符号式的结果在交回内层时物化成 constant 或 index_expr，其他操作则解包后转发。本例关闭这项配置，因此 `fn` 的 load/add/store 直接经过 SimplifyIndexing 进入 CaptureIndexing，节点轨迹就是前面给出的形态。

在动态 ops 上下文中执行原 callable，最后通过 `ops.output` 显式建立 graph output。这里执行的是符号 proxy 过程，不读取真实 tensor。

动态 ops 上下文在正常返回和异常时都会恢复进入前的 handler，但这段构造不是事务：执行 `fn(*args)` 之前，`body.indexing_exprs`、`memory_usage`、`submodules` 或 `subblocks` 可能已经追加内容。若后续捕获抛异常，构造函数不会返回一个可用的 `LoopBodyBlock`，源码也没有把这些已经写入 `body` 的记录逐项回滚。调用方不应把这个失败对象当作一次完整捕获继续使用。

### K02 理解验收

若 `index_expr` 输入从整数 4 改成 `x0+4`，前者图中是 constant，没有 INDEX_EXPR memory entry；后者先登记索引名，重放时随新 x0 替换。若 reduction type 从 sum 改为 welford，捕获图的结果接口从单 proxy 变为三项 tuple，不能只取第 0 项。

K02 到此完成。K03 打开三个可 clone shim 和 masked/scan/indirect 的状态交接。

## K03：submodule shim 把嵌套操作留给重放时的 handler

先代入 masked 例：捕获期名字为 `masked_subblock0`，mask 是 `x0<8`，other=0；`self.subblocks` 起初为空，递归捕获的主体执行 `load("x",x0)`。shim 保存的静态值只有子块名字；重放到另一份 LoopBody 时，`clone` 用同名子块重新绑定新的 `self`，函数体再从当时的 `V.ops` 取得 CPU 或 Triton handler。indirect 例统一使用 `var=indirect0,size=8,check=True,wrap_neg=True`。

```python
def bind_set_indirect_shim(self, var, size, check, wrap_neg):
    def set_indirect(new_var):
        self.replace_indirect(
            var, V.ops.indirect_indexing(new_var, size, check, wrap_neg)
        )

    set_indirect.clone = functools.partial(  # type: ignore[attr-defined]
        LoopBody.bind_set_indirect_shim,
        var=var,
        size=size,
        check=check,
        wrap_neg=wrap_neg,
    )
    return set_indirect

def bind_scan_shim(self, combine_fn):
    def shim(dtypes, values):
        return V.ops.scan(dtypes, combine_fn, values)

    shim.clone = functools.partial(LoopBody.bind_scan_shim, combine_fn=combine_fn)  # type: ignore[attr-defined]
    return shim

def bind_masked_shim(self, name):
    def shim(mask, other):
        return V.ops.masked(mask, self.subblocks[name], other)

    shim.clone = functools.partial(LoopBody.bind_masked_shim, name=name)  # type: ignore[attr-defined]
    return shim
```

这些对象被放入 `LoopBody.submodules`，所以 InterpreterShim 执行 `call_module` 时会调用它们。shim 不固定捕获期 handler：函数体在重放时重新访问 `V.ops`，因此 scan/masked/indirect 仍由当前 CPU、Triton 或分析 handler 解释。

每个 shim 额外挂一个 `clone=functools.partial(...)`。LoopBody 因循环合并/重排复制到新父对象时，submodule 可以用同一静态参数重新绑定 `self`，不会继续修改旧 LoopBody 的 indexing/subblocks。

复制一份 `LoopBody` 时，篇首 `_init_with_copy` 会实际遍历旧对象的 subblock 和 submodule，并把新 body 传给每个 clone 接口。那段代码里的 `self` 是新 LoopBody：`LoopBodyBlock.clone(self)` 让复制后的 root/subblock 指向新父对象；每个 shim 的 `v.clone(self)` 则以同一个新父对象重新调用 bind 方法。复制完成后，新 root、subblocks、`get_index` 和 shim 都从同一份 LoopBody 读取临时 indexing 和 subblocks；但 memory usage 与 indirect 列表/范围仍是新旧对象共享的引用。

### masked：主体单独捕获成 subblock

捕获 `ops.masked(mask, masked_body, other)` 时：

1. 先为名字 `masked_subblock...` 在 submodules 占位；
2. 用 `bind_masked_shim(name)` 替换占位；
3. 对 `masked_body` 递归构造新的 `LoopBodyBlock`，存入 `subblocks[name]`；
4. root 图只保存 `call_module(name,mask_proxy,other_proxy)`。

重放 root 到该 call_module 时，shim 调用当前 `V.ops.masked(mask,self.subblocks[name],other)`。具体 handler 决定怎样执行 mask，并在需要时调用 subblock；subblock 自己再次用 InterpreterShim 传入当时的 handler。这样被 mask 的 load/算术仍保留独立小图，而不是在 root 捕获时无条件执行。

为了手算操作语义，先假定当前是一个按具体布尔值惰性执行 body 的解释型 handler，并取 `x=[10,11,...,25]`。位置 `x0=3` 时 mask 为真，handler 调用 subblock，`load("x",3)` 得 13，所以 masked 结果为 13；位置 `x0=9` 时 mask 为假，handler 返回 `other=0`，无需读取 `x[9]`。代码生成 handler 不一定在 Python 重放时就按这个布尔值跳过 subblock：例如它可以先调用 subblock 生成一段 lambda/表达式，再让生成代码在运行时选择 13 或 0。LoopBody 只保证把 mask、subblock 和 other 一起交给当前 handler。

### scan：combine_fn 固定，值和 handler 延迟

捕获 scan 时，`combine_fn` 被闭包保存进 shim，FX root 图保存 dtype/value proxies 的 call_module。`combine_fn` 本身是一个普通 Python callable，不会在这里再捕获成 `LoopBodyBlock`。例如两项状态的组合函数可以接收左侧 `(sum_l, max_l)` 和右侧 `(sum_r, max_r)`，返回：

```text
(sum_l + sum_r, maximum(max_l, max_r))
```

若 `value_proxy=(sum_value,max_value)`，捕获图中的 scan call_module 只有 dtypes 和这两项 value proxy；shim 的闭包另外保存上述 callable。重放时 shim 把三者交给当前 `V.ops.scan(dtypes, combine_fn, values)`，由当前 handler 决定怎样调用组合函数和生成目标表达。scan 返回两项时，捕获代码显式建立 `result[0]`、`result[1]`，组成真正的 Python tuple，后续消费者不必把一个可迭代 FX Proxy 当作 tuple。

代入一次具体 combine：左状态为 `(sum_l,max_l)=(2,5)`，右状态为 `(sum_r,max_r)=(3,4)`，闭包中的 combine_fn 得到 `(2+3, maximum(5,4))=(5,5)`。捕获图保存的不是这个常量答案，而是“调用 scan shim，并从返回 proxy 取第 0、1 项”的结构；实际值或符号表达由重放时的 handler 产生。

### indirect：先用符号占位，重放时再改索引表

假设 tensor load 产生值 p，后续把它当索引：

```text
捕获：add_indirect(size=8) -> indirect0
      root graph 插入 call_module set_indirect0(p)
      后续索引式可写 indirect0 + x0
```

重放到 set shim 时，当前 handler 先执行 `indirect_indexing(p,8,check,wrap_neg)`，得到适合该 backend 的新索引对象；`replace_indirect` 再把当前 `self.indexing` 所有表达式中的 indirect0 替换为它。因此必须先执行 set call_module，后续 `get_index` 才能看到实际间接值。

同样先用能返回具体整数的解释型 handler 手算：取 `p=5`、当前直接循环位置 `x0=2`，并令输入 `x=[100,101,...,107]`。捕获索引式是 `indirect0+x0`；重放 set shim 后临时表中的 indirect0 被 5 替换，后续 `get_index` 得到 `5+2=7`，所以 load 读取 `x[7]=107`。CPU/Triton 代码生成 handler 也可能把 p 变成 CSE 或符号表达，此时临时索引是“该表达式 + 2”，而不是 Python 整数 7；生成代码运行到 p=5 时才得到同一地址。若把 set 节点移到 get_index 后面，第一次取索引时仍含未替换的 indirect0，无法建立这条数据流。

`check` 与 `wrap_neg` 捕获进闭包，控制是否检查范围、是否把负索引环绕；clone 时也原样保留。两个 indirect 调用分别得到 indirect0、indirect1，不共享占位符。

### K03 理解验收

若 LoopBody 重排后克隆，masked shim 必须查询新 body 的 `subblocks[name]`；若仍闭包旧 self，root 图与 subblock 会来自不同对象。clone partial 正是重新绑定父对象的接口。

若间接索引 set 节点被移到后续 get_index 之后，get_index 会读到仍含 indirect0 的旧表达式；当前捕获顺序用显式 call_module 保存“先由值设置索引、再访存”的数据流。

下一节把这些捕获和重放步骤代入普通逐点计算与规约计算。

## K04：实验轨迹覆盖当前 CPU 重放，其他 handler 分支按源码逐项核对

实验 12 的逐点函数是：

```python
def pointwise(x, y):
    return torch.relu(x + y) * 2
```

输入 `x=[-2,1,3]`、`y=[1,4,-1]` 时，三个位置依次计算 `relu(-1)*2=0`、`relu(5)*2=10`、`relu(2)*2=4`，所以结果是 `[0,10,4]`。捕获表包含同一 `index0=x0` 的两项 LOAD 和一项 STORE；root 图按顺序保存 load x、load y、add、relu、constant 2、mul、store。重放 `body([c0])` 后，临时 indexing 把 index0 换成 c0，CPU handler 按这张图产生对应表达。

规约函数是 `(x+1).sum(dim=1)`，输入为 `torch.arange(256).reshape(2,128)`。第一行从 0 到 127，加 1 后求和为 `1+...+128=8256`；第二行从 128 到 255，加 1 后求和为 `129+...+256=(129+256)*128/2=24640`。捕获时，读取输入的索引是 `128*x0+x1`，写输出的索引是 `x0`；x0 选择哪一行，x1 遍历该行的 128 项。重放时它们分别代成 backend 的输出变量和规约变量，handler 轨迹包含 `ops.reduction(sum,...)` 与 `store_reduction`。当前 CPU 候选产生一次 scalar、两次 vector 重放；重放次数来自这个候选的生成策略，不是 LoopBody 接口规定的常数。

masked、scan、sort、frexp 和 indirect 也通过同一套记录再重放机制传递操作；某个后端是否支持其中一种组合，仍由该后端的 handler 决定。

### K04 理解验收

换成 Triton handler 时，`indexing_from_args`、FX node 顺序、memory usage 和 submodule 名不变；placeholder 实参、各 ops 返回对象及目标代码改变。依赖快速路径甚至可以不执行这张算术图，只消费 memory usage；这不改变 codegen 重放链的实现。

若 root 重放中 handler 抛异常，`InterpreterShim.run` 安装的 `V.interpreter` 会恢复，但当前 `LoopBody.__call__` 没有 finally，`self.indexing` 可能仍保留。产品重放路径本身没有替换 `V.ops`；实验 12 为记录轨迹而设置的 ops context 属于实验调用方。再次使用前不能依据 interpreter 或外层 ops context 已恢复，就推断 LoopBody 临时状态也已清空。

现在可以用一个变化检验整条链：若逐点例的访问从 `x[x0]` 改为 `x[2*x0]`，捕获表应保存 `index0=2*x0`；backend 传入 c0 后临时表变为 `index0=2*c0`；FX 小图结构和 memory usage 类型不变，当前 handler 收到的新 load/store 索引随之变成 `2*c0`。下一篇 Q421 继续追踪 CPU/Triton backend 怎样把这些 ops 变成各自的目标表达。
