# LoopBody 捕获与重放源码：符号索引怎样交给当前 handler

[返回虚拟化 ops 篇](../README.md) · [返回 Codegen](../../README.md) · [返回总览](../../../README.md)

Q41 已经说明“同一张 FX 小图换 handler 就得到不同目标表达”。本篇把这句话落实成源码状态：捕获阶段怎样建立图、索引表和 submodule；重放阶段怎样把新循环符号代入索引，再把当前动态 handler 填进图的 `ops` placeholder。

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

第一个 assert 防止少传/多传循环维；第二个 assert 防止把原 `var_ranges` 中的同一符号原样当作新参数，避免替换关系自指。当前实现没有 `try/finally` 包住 `root_block()`：若重放中抛异常，`self.indexing=None` 这句不会执行。因此正常重放能恢复，异常后的同一对象可能保留临时表；这是源码边界，不能泛称所有状态都会自动清理。动态 interpreter/ops 上下文则由各自 context manager 恢复。

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

捕获尾部先创建 `ops` placeholder，再组合 `CaptureIndexing→SimplifyIndexing`；配置开启时外层再包 `IndexPropagation`。在动态 ops 上下文中执行原 callable，最后通过 `ops.output` 显式建立 graph output。这里执行的是符号 proxy 过程，不读取真实 tensor。

### K02 理解验收

若 `index_expr` 输入从整数 4 改成 `x0+4`，前者图中是 constant，没有 INDEX_EXPR memory entry；后者先登记索引名，重放时随新 x0 替换。若 reduction type 从 sum 改为 welford，捕获图的结果接口从单 proxy 变为三项 tuple，不能只取第 0 项。

K02 到此完成。K03 打开三个可 clone shim 和 masked/scan/indirect 的状态交接。

## K03：submodule shim 把嵌套操作留给重放时的 handler

先代入 masked 例：捕获期名字为 `masked_subblock0`，mask 是 `x0<8`，other=0；`self.subblocks` 起初为空，递归捕获的主体执行 `load("x",x0)`。shim 保存的静态值只有子块名字；重放到另一份 LoopBody 时，`clone` 用同名子块重新绑定新的 `self`，函数体再从当时的 `V.ops` 取得 CPU 或 Triton handler。indirect 例可同样代入 `var=u0,size=16,check=True,wrap_neg=True`。

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

### masked：主体单独捕获成 subblock

捕获 `ops.masked(mask, masked_body, other)` 时：

1. 先为名字 `masked_subblock...` 在 submodules 占位；
2. 用 `bind_masked_shim(name)` 替换占位；
3. 对 `masked_body` 递归构造新的 `LoopBodyBlock`，存入 `subblocks[name]`；
4. root 图只保存 `call_module(name,mask_proxy,other_proxy)`。

重放 root 到该 call_module 时，shim 调用当前 `V.ops.masked(mask,self.subblocks[name],other)`。具体 handler 决定怎样执行 mask，并在需要时调用 subblock；subblock 自己再次用 InterpreterShim 传入当时的 handler。这样被 mask 的 load/算术仍保留独立小图，而不是在 root 捕获时无条件执行。

### scan：combine_fn 固定，值和 handler 延迟

捕获 scan 时，`combine_fn` 被闭包保存进 shim，FX root 图保存 dtype/value proxies 的 call_module。重放时 shim 调用当前 `V.ops.scan(dtypes,combine_fn,values)`。返回 proxy 在捕获时按 `len(value_proxy)` 拆成确定数量，后续消费者能把它当真实 tuple 使用。

### indirect：先用符号占位，重放时再改索引表

假设 tensor load 产生值 p，后续把它当索引：

```text
捕获：add_indirect(size=8) -> indirect0
      root graph 插入 call_module set_indirect0(p)
      后续索引式可写 indirect0 + x0
```

重放到 set shim 时，当前 handler 先执行 `indirect_indexing(p,8,check,wrap_neg)`，得到适合该 backend 的新索引对象；`replace_indirect` 再把当前 `self.indexing` 所有表达式中的 indirect0 替换为它。因此必须先执行 set call_module，后续 `get_index` 才能看到实际间接值。

`check` 与 `wrap_neg` 捕获进闭包，控制是否检查范围、是否把负索引环绕；clone 时也原样保留。两个 indirect 调用分别得到 indirect0、indirect1，不共享占位符。

### K03 理解验收

若 LoopBody 重排后克隆，masked shim 必须查询新 body 的 `subblocks[name]`；若仍闭包旧 self，root 图与 subblock 会来自不同对象。clone partial 正是重新绑定父对象的接口。

若间接索引 set 节点被移到后续 get_index 之后，get_index 会读到仍含 indirect0 的旧表达式；当前捕获顺序用显式 call_module 保存“先由值设置索引、再访存”的数据流。

下一节把这些捕获和重放步骤代入普通逐点计算与规约计算。

## K04：实验轨迹覆盖当前 CPU 重放，其他 handler 分支按源码逐项核对

实验 12 的 pointwise 捕获表包含同一 `index0=x0` 的两项 LOAD 和一项 STORE。重放 `body([c0])` 后，临时 indexing 把 index0 换成 c0，InterpreterShim 依次让 CPU handler 产生 load x、load y、add、relu、constant、mul、store，结果为 `[0,10,4]`。

2×128 reduction 的捕获索引是 `128*x0+x1` 与 `x0`。重放时分别代成 backend 的输出变量和规约变量，handler 轨迹包含 `ops.reduction(sum,...)` 与 `store_reduction`，最终得到 `[8256,24640]`。当前 CPU 候选产生一次 scalar、两次 vector 重放；这不是 LoopBody 接口保证的固定次数。

masked、scan、sort、frexp 和 indirect 也通过同一套记录再重放机制传递操作；某个后端是否支持其中一种组合，仍由该后端的 handler 决定。

### K04 理解验收

换成 Triton handler 时，`indexing_from_args`、FX node 顺序、memory usage 和 submodule 名不变；placeholder 实参、各 ops 返回对象及目标代码改变。依赖快速路径甚至可以不执行这张算术图，只消费 memory usage；这不改变 codegen 重放链的实现。

若 root 重放中 handler 抛异常，V.interpreter/V.ops 的上下文会恢复，但当前 `LoopBody.__call__` 没有 finally，`self.indexing` 可能仍保留。再次使用前不能仅依据“上下文已恢复”推断 LoopBody 临时状态也已清空。

K04 到此完成。Q411 的 11 个声明、主要分支和状态推演已经齐全；下一篇 Q421 打开具体 CPU/Triton backend 如何消费这些 ops。
