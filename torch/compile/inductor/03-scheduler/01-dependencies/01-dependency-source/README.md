# Scheduler 依赖构造核心源码：先包装 operation，再记录它读写什么

[返回依赖篇](../README.md) · [返回 Scheduler](../../README.md) · [返回总览](../../../README.md)

Q31 已经展示最终依赖图，本篇从实现重新建立它。输入是 Lowering 交出的 operation 列表；出口是每个 scheduler node 的读写清单、尚未满足的依赖、每个结果的 users，以及 mutation 的版本关系。拓扑排序、死代码删除和最后使用位置随后读取这些信息。

贯穿全文的几个对象先固定用途：

- `ReadWrites` 保存一项工作的完整访问事实：`reads`、`writes`、不直接访问 buffer 的 `index_exprs`，以及解释索引符号所需的 `range_vars/var_ranges`。
- `MemoryDep(name,index,...)` 表示按索引访问某个 buffer；`StarDep(name)` 只要求这个名字代表的整项结果已经完成；`WeakDep(reader_output,mutating_buf=...)` 只安排先后，不读取 `reader_output` 的数值。
- `unmet_dependencies` 从完整 reads 开始，再删掉 graph input 和常量等调度开始前已经可用的名字。它回答的是“当前节点还要等哪些内部结果”，不是完整访问清单。
- scheduler wrapper 把不同种类的 operation 接到统一调度接口。wrapper 不改变计算含义，只统一提供名字、输出、读写和 users 等信息。

## K01：operation 类型决定包装节点和读写来源

进入方法时，`node: ir.Operation` 是 Lowering 列表中的一项工作。类型标注里的 `Union[A,B]` 表示值可以是 A 或 B；本例普通逐元素结果是 `ComputedBuffer buf0`，所以 `is_no_op()` 为假、命中 `ComputedBuffer` 分支，返回 `SchedulerNode(self,node)`。若换成 mm 的 `ExternKernel`，则返回 `ExternKernelSchedulerNode`。`origins` 是产生这项工作的 FX 节点集合，缺失时断言失败。

`Scheduler.create_scheduler_node` 完整实现如下：

```python
def create_scheduler_node(self, node: ir.Operation) -> BaseSchedulerNode:
    assert (
        node.get_origins() is not None
    ), "All nodes passed to scheduling must have an origin"
    if node.is_no_op():
        # 教学：no-op 仍可能携带依赖，用专门 wrapper 保留而不生成普通 kernel。
        return NopKernelSchedulerNode(self, node)
    elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
        # 教学：二者都有可供 Scheduler 分析的循环或模板读写接口。
        return SchedulerNode(self, node)
    elif isinstance(node, ir.ExternKernel):
        # 教学：外部调用自报读写，不通过普通循环重放。
        return ExternKernelSchedulerNode(self, node)
    else:
        raise NotImplementedError(node)
```

判断顺序很重要：一个 operation 若 `is_no_op()` 为真，先进入 Nop 分支，不再按其 Python 类型进入后两支。所有可调度 operation 还必须已有 origins；缺失时在任何 wrapper 构造前断言失败。

普通 `ComputedBuffer` 与 `TemplateBuffer` 共用 `SchedulerNode`，但初始化内部再分开读写来源：

下面构造器收到的 `scheduler` 是当前全图 Scheduler，`node` 是 buf0 对应的 ComputedBuffer。`_init_from_node` 保存名字、输出和来源；随后从 node 取得 loop body，并生成本节点的 `ReadWrites`。本例结束时读集合含输入 x，写集合含 buf0。

```python
def __init__(
    self,
    scheduler: Scheduler,
    node: Union[ir.ComputedBuffer, ir.TemplateBuffer],
) -> None:
    super().__init__(scheduler)
    # 教学：先保存 IR node、输出 buffer 等共同字段。
    self._init_from_node(node)
    # 教学：随后建立循环尺寸、group 和读写清单。
    self._compute_attrs()

def _compute_attrs(
    self,
    extra_indexing_constraints: Optional[Tuple[Dict[Any, Any], List[Any]]] = None,
    recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
) -> None:
    assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer))
    # 教学：取得已经简化、必要时重排过的迭代尺寸和循环体。
    self._sizes, self._body = self.node.simplify_and_reorder(
        extra_indexing_constraints=extra_indexing_constraints,
        recompute_sizes_body_func=recompute_sizes_body_func,
    )

    group_fn = self.scheduler.get_backend(self.node.get_device()).group_fn
    # 教学：后端把尺寸转换为调度分组键，设备与分组尺寸共同组成 group。
    self.group = (self.node.get_device(), group_fn(self._sizes))

    # Don't normalize since normalization will merge loops which
    # makes it hard to decide new loop orders.
    should_normalize = (
        not config.loop_ordering_after_fusion
        or self.node.get_device().type != "cuda"
    )

    if isinstance(self.node, ir.TemplateBuffer):
        # 教学：模板知道自己的特殊输入关系，由模板接口提取。
        self.set_read_writes(
            self.node.extract_read_writes(normalize=should_normalize)
        )
    else:
        # 教学：普通 ComputedBuffer 重放循环体，让 handler 记录 load/store。
        self.set_read_writes(
            dependencies.extract_read_writes(
                self._body, *self._sizes, normalize=should_normalize
            )
        )
```

no-op 与 extern wrapper 的构造路径更短：二者都直接采用 operation 自报的 `get_read_writes()`，区别是 no-op 后续不生成普通 kernel，extern 后续生成已有函数调用。

```python
class ExternKernelSchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler, node):
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())

class NopKernelSchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler, node):
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())
```

读写安装不是简单赋字段。`BaseSchedulerNode` 立即生成一份初始未满足集合，再裁掉已经存在的图输入和常量：

```python
def add_fake_dep(self, dep: Dep) -> None:
    # 教学：StarDep/WeakDep 等人工关系也统一并入 ReadWrites，再重新裁剪 unmet。
    self.set_read_writes(self.read_writes.with_read(dep))

def set_read_writes(self, rw: dependencies.ReadWrites) -> None:
    self.read_writes = rw
    # 教学：先假定全部读取都需要等待。
    self.unmet_dependencies = self.read_writes.reads
    self.prune_deps()

def prune_deps(self) -> None:
    self.unmet_dependencies = OrderedSet(
        dep
        for dep in self.unmet_dependencies
        # 教学：graph inputs/constants 已在调度开始前可用，不需要内部 producer。
        if dep.name not in self.scheduler.available_buffer_names
    )
```

### 两个实际节点的初始化状态

实验 08 的规约只有一个 `ComputedBuffer(buf0)`。重放得到三条读取 `arg0_1` 的 MemoryDep 和一条写 `buf0`；因为 `arg0_1` 属于 `available_buffer_names`，`set_read_writes` 后：

```text
read_writes.reads = {arg0_1[3*c0], arg0_1[3*c0+1], arg0_1[3*c0+2]}
read_writes.writes = {buf0[c0]}
unmet_dependencies = {}
```

`mm→sin` 的 mm 是 `ExternKernelOut`，所以得到 `ExternKernelSchedulerNode`；它自报两个输入 `StarDep`，两者同样因输入已可用而不进入 unmet。后继 sin 是普通 `SchedulerNode`，读取内部 `buf0`；此时 `buf0` 尚不在 available 集合，于是初始 unmet 保留 `MemoryDep(buf0,c0)`。

| operation | wrapper | 完整 reads | 初始 unmet |
| --- | --- | --- | --- |
| reduction ComputedBuffer | SchedulerNode | 三条 arg0_1 MemoryDep | 空 |
| mm ExternKernelOut | ExternKernelSchedulerNode | 两条输入 StarDep | 空 |
| sin ComputedBuffer | SchedulerNode | 一条 buf0 MemoryDep | buf0 MemoryDep |

另外两条路径可用同一规则判断。假设 no-op 自报 `reads={StarDep(buf0)}`、不写新 buffer，它的 wrapper 仍保留这项 read；若 buf0 是内部结果，unmet 也保留 buf0。假设 template 自报读取 graph input x 和内部临时量 buf2，`SchedulerNode` 通过 template 的 `extract_read_writes()` 得到两项 read，裁剪后只剩 buf2。决定 unmet 的是名字是否已经 available，而不是 wrapper 是否生成普通 kernel。

### K01 理解验收

若新增一个 ExternKernel 只读取 graph constant `weight`，其完整 reads 仍保留 `StarDep(weight)`，但 `weight` 已在 available 集合，所以 unmet 为空。裁剪改变的是“还要等谁”，不是删除 operation 的访问事实。

若未知的 `ir.Operation` 有 origins、不是 no-op，也不属于 computed/template/extern，创建会在最后一支抛 `NotImplementedError`；不会退化成一个没有依赖的普通节点。

K01 到此完成。下一知识点 K02 打开 `index_vars_squeeze` 和 `extract_read_writes`，解释 `[2,3]` 规约的 `c0`、索引式和范围怎样真实进入 ReadWrites。

## K02：用符号索引重放循环，load/store 就变成 MemoryDep

真正执行 kernel 会读取 tensor；依赖提取只需要知道“会读写哪里”。因此它先用符号变量代表循环位置，再让记录 handler 接管 `load/store`。同一计算体在这个阶段产生依赖对象，不产生数值。

### 尺寸 1 的维度先被挤掉

`index_vars_squeeze` 的完整源码如下：

本次调用是 `index_vars_squeeze((2,1,3), prefix="d")`。初始 `var_ranges={}`；`SqueezeView.squeezer` 删除中间的 size-1 维，交回 `new_size=[2,3]` 和“把 0 插回中间”的 reindex。`add_var` 随后创建 d0、d1 并登记 `{d0:2,d1:3}`，reindex 的最终结果是 `[d0,0,d1]`。

```python
def index_vars_squeeze(*argsizes: Tuple[sympy.Expr, ...], prefix: str = "d"):
    from .ir import SqueezeView

    # 教学：var_builder 负责按 prefix 创建符号，并保存“符号→范围”。
    var_ranges, add_var = var_builder(prefix)
    args: List[List[sympy.Expr]] = []
    new_sizes: List[List[sympy.Expr]] = []
    for size in argsizes:
        # 教学：删除 size=1 的循环维，同时得到把精简索引还原到原维数的函数。
        new_size, reindex = SqueezeView.squeezer(size)
        new_sizes.append(new_size)
        # 教学：只为剩余维创建变量，再把 0 插回被 squeeze 的维度。
        args.append(reindex(list(map(add_var, new_size))))
    return args, var_ranges
```

例如输入一组循环尺寸 `(2,1,3)`，中间状态是：

```text
new_size = [2,3]
新符号   = [d0,d1]
reindex  = [d0,0,d1]
var_ranges = {d0:2, d1:3}
```

尺寸为 1 的维度永远只有位置 0，不需要独立循环变量。保留原三维索引形状又让计算体仍可按原维数读取 `index[0:3]`。

### extract_read_writes 的两条路径

先走普通 callable：`fn(i)` 发出 `load("x",3*i)` 和 `store("buf0",i,...)`，`argsizes=([2],)`，`hidden_args=()`，`normalize=False`，prefix 为 c。入口先建立 `args=[[c0]]`、`var_ranges={c0:2}`；记录 handler 初始 reads/writes 都为空，调用 fn 后分别得到 `MemoryDep(x,3*c0,size=[2])` 与 `MemoryDep(buf0,c0,size=[2])`。若 fn 已是 LoopBody，则不再调用它，而是把其中保存的同两条 memory_usage 代入当前 c0，出口仍是相同 ReadWrites。

```python
def extract_read_writes(
    fn: Callable[..., Any],
    *argsizes: Tuple[sympy.Expr, ...],
    normalize: bool = False,
    prefix: str = "d",
    hidden_args=(),
):
    # 教学：每组迭代尺寸得到一组符号参数，共享一份符号范围表。
    args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)

    from .loop_body import LoopBody, MemoryUsageType

    if isinstance(fn, LoopBody):
        # Fast path to avoid tracing when we already have a LoopBody
        # 教学：LoopBody 已记录内存操作，只需把它保存的索引名代入当前符号参数。
        inner = _RecordLoadStoreInner(var_ranges=var_ranges, normalize=normalize)
        name_to_index = fn.indexing_from_args([*args, *hidden_args])
        if fn.indirect_vars:
            # mimic the `tmpX` naming tracing gives us
            # 教学：间接索引变量改名成 tracing 路径使用的 tmp0/tmp1，保持两路结果可比。
            repl = {v: sympy.Symbol(f"tmp{i}") for i, v in enumerate(fn.indirect_vars)}
            name_to_index = {k: sympy_subs(v, repl) for k, v in name_to_index.items()}
        for entry in fn.memory_usage[MemoryUsageType.LOAD]:
            inner.load(entry.buffer_name, name_to_index[entry.index_name])
        for entry in fn.memory_usage[MemoryUsageType.LOAD_SEED]:
            inner.load_seed(entry.buffer_name, int(name_to_index[entry.index_name]))
        for entry in fn.memory_usage[MemoryUsageType.STORE]:
            inner.store(
                entry.buffer_name, name_to_index[entry.index_name], None, entry.mode
            )
        for entry in fn.memory_usage[MemoryUsageType.STORE_REDUCTION]:
            inner.store_reduction(
                entry.buffer_name, name_to_index[entry.index_name], None
            )
        for entry in fn.memory_usage[MemoryUsageType.INDEX_EXPR]:
            inner.index_expr(name_to_index[entry.index_name], None)
        for entry in fn.memory_usage[MemoryUsageType.BUCKETIZE]:
            inner.bucketize(
                None, entry.buffer_name, name_to_index[entry.index_name], None, None
            )
        # fn.memory_usage[MemoryUsageType.CHECK_BOUNDS] intentionally skipped
    else:
        # Slow path tracing the function
        # 教学：普通 callable 需要实际重放；RecordLoadStore 接管 V.ops 并记录访问。
        rw = RecordLoadStore(var_ranges, normalize=normalize)
        with V.set_ops_handler(rw):
            fn(*args, *hidden_args)
        inner = rw.parent_handler

    if normalize:
        # 教学：normalize 可能合并循环，原始变量数量不再可靠。
        range_vars = []  # Number of vars could differ due to normalization
    else:
        # 教学：不规范化时保留调用计算体所用的原符号参数顺序。
        range_vars = [*itertools.chain.from_iterable(args)]

    return ReadWrites(
        OrderedSet(inner._reads),
        OrderedSet(inner._writes),
        inner._index_exprs,
        range_vars,
        var_ranges,
    )
```

两条路径的出口完全相同：`inner._reads/_writes` 被去重并放入 `ReadWrites`。区别是信息从哪里来：

| 输入 `fn` | 怎样发现访问 | 为什么需要这条路径 |
| --- | --- | --- |
| `LoopBody` | 遍历已保存的 `memory_usage`，把索引名代入当前符号 | 避免再次 tracing，保持间接变量命名一致 |
| 普通 callable | 在 `V.set_ops_handler(rw)` 内调用函数，拦截它发出的 ops | 尚无 LoopBody 记录时现场采集 |

slow path 依赖当前 `V.graph.sizevars` 来规范化索引。它必须在合法 GraphLowering 上下文运行；把函数孤立拿到模块外调用并不是产品路径。

`LoopBody.memory_usage` 是 tracing LoopBody 时保存的访问事件分类表。每项事件保存 buffer 名、索引表达式的名字，以及 store 的 mode；fast path 不重新执行计算，只逐类重放这张表。它不只处理普通 load/store：

| 已保存的使用类型 | 重放动作 | 对 `ReadWrites` 的影响 |
| --- | --- | --- |
| `LOAD` | `inner.load(name,index)` | 增加一项读依赖 |
| `LOAD_SEED` | `inner.load_seed(name,seed)` | 把规约种子来源登记为读；这里把保存的索引值转成 Python `int` |
| `STORE` | `inner.store(name,index,None,mode)` | 增加写依赖，并原样传递写模式；例如原子写的 mode 不会在本函数中丢失 |
| `STORE_REDUCTION` | `inner.store_reduction(name,index,None)` | 登记规约写，而不是伪装成普通 store |
| `INDEX_EXPR` | `inner.index_expr(index,None)` | 收集纯索引表达式到 `_index_exprs`，不凭空产生 buffer 读写 |
| `BUCKETIZE` | `inner.bucketize(None,name,index,None,None)` | 让记录器登记 bucketize 对边界 buffer 的访问 |
| `CHECK_BOUNDS` | 有意跳过 | 边界检查本身不成为调度 buffer 依赖 |

其中 `None` 是因为依赖提取只需要“访问哪个名字、哪个位置、采用哪类写法”，不需要保存当时计算出的数值。若一条 `STORE` 的 `mode` 是原子更新，调度看到的仍是一项写；原子操作怎样生成指令属于后端代码生成，不在这里展开。

这些动作怎样变成最终依赖，由同一个记录器完成：

```python
def load(self, name, index):
    self._reads.add(MemoryDep(name, *self.canonicalize(index)))

def load_seed(self, name, index):
    assert isinstance(index, int)
    return self.load(name, sympy.Integer(index))

def store(self, name, index, value, mode=None):
    self._writes.add(
        MemoryDep(name, *self.canonicalize(index), mode=mode)
    )

def bucketize(self, values, offsets_name, offsets_size, indexing_dtype, right):
    self._reads.add(StarDep(offsets_name))
```

所以 `STORE(out,tmp1,mode="atomic_add")` 最终产生的 write 是 `MemoryDep(out,tmp1,...,mode="atomic_add")`；mode 不是只在调用途中出现。`LOAD_SEED(seed,index=4)` 先要求索引能转成 Python `int`，再产生常量位置的 `MemoryDep(seed,4,var_names=(),size=())`；若保存的表达式不能转成整数，fast path 在 `int(...)` 处失败。`BUCKETIZE(offsets,...)` 不建立某个具体位置的 MemoryDep，而是建立 `StarDep(offsets)`，要求整个边界数组已经可用。

间接索引还多一步名称对齐。假设 LoopBody 保存的访问式为 `indirect0 + d0`，`indirect_vars=[indirect0]`；fast path 先建立 `indirect0 -> tmp0`，再得到 `tmp0+d0`。slow path tracing 临时值时也采用 `tmp0` 命名，因此两条路径不会仅因内部符号名字不同而得到不可比较的依赖。若有两个间接变量，就按保存顺序成为 `tmp0、tmp1`，而不是都映到同一个符号。

例如事件表含 `LOAD(x,index0)`、`STORE(out,index1,mode="atomic_add")`，并且：

```text
index0 = indirect0 + d0
index1 = indirect1
indirect_vars = [indirect0, indirect1]
```

代入当前循环符号后，记录器收到 `load(x,tmp0+d0)` 和 `store(out,tmp1,mode="atomic_add")`。出口中的读依赖使用索引 `tmp0+d0`，写依赖使用 `tmp1` 并保留 `atomic_add` mode。名称替换改变的是内部临时符号，不能把两个间接位置合并；mode 传递保留写入种类，不能把原子更新误记成普通覆盖。

### `[2,3]` 规约为何最后只有变量 `c0`

前面的 `(2,1,3)` 只是说明 size-1 维怎样 squeeze。现在回到另一项实际案例：Python 请求对 2×3 输入按行求和。实验 08 中，Scheduler 从 `ComputedBuffer.simplify_and_reorder()` 得到 `argsizes=([2],[])`：第一组 `[2]` 是输出位置，第二组为空，说明当前 LoopBody 已没有单独的 reduction 循环变量。长度 3 的规约在更早建立 LoopBody 时已展开成三次读取。

进入 fast path 前，实际保存状态是：

```text
memory_usage[LOAD] = [
  (arg0_1, index0),
  (arg0_1, index1),
  (arg0_1, index2),
]
memory_usage[STORE] = [(buf0, index3, mode=None)]
indexing_exprs = {
  index0: 3*z0,
  index1: 3*z0 + 1,
  index2: 3*z0 + 2,
  index3: z0,
}
```

`index_vars_squeeze([2],[])` 建立当前符号 `c0`，`indexing_from_args` 把旧表达式中的 z0 换成 c0，得到：

```text
index0 -> 3*c0
index1 -> 3*c0 + 1
index2 -> 3*c0 + 2
index3 -> c0
```

随后三次 `inner.load(...)` 让 `_reads` 从空集合依次变成一、二、三项；一次 `inner.store(...)` 把 `buf0[c0]` 加入 `_writes`。本例的输出事件是普通 `STORE`，不是 `STORE_REDUCTION`，因为静态三项规约已经展开在三个 load 中。

LoopBody 中的三个 load 索引名代入后分别成为：

```text
3*c0
3*c0 + 1
3*c0 + 2
```

store 索引成为 `c0`。记录器据此建立：

```text
reads = {
  MemoryDep(arg0_1, 3*c0,     var_names=(c0,), size=(2,)),
  MemoryDep(arg0_1, 3*c0 + 1, var_names=(c0,), size=(2,)),
  MemoryDep(arg0_1, 3*c0 + 2, var_names=(c0,), size=(2,)),
}
writes = {MemoryDep(buf0, c0, var_names=(c0,), size=(2,))}
range_vars = [c0]
var_ranges = {c0: 2}
```

代入 `c0=0` 得到输入位置 0、1、2 和输出位置 0；代入 `c0=1` 得到输入位置 3、4、5 和输出位置 1。依赖记录因此同时表达“读哪个 buffer”和“本循环怎样覆盖它的位置”。

若没有现成 LoopBody，同一访问可写成普通 callable：

```python
def row_sum(c):
    a = ops.load("arg0_1", 3 * c)
    b = ops.load("arg0_1", 3 * c + 1)
    d = ops.load("arg0_1", 3 * c + 2)
    ops.store("buf0", c, a + b + d)
```

slow path 在记录 handler 下执行它，四次 ops 调用形成同样的三项 reads 和一项 write。两条路径共享出口，但 fast path 的输入是事件表，slow path 的输入是会发出 ops 的 callable。

注意，Python 请求里有 reduction 轴，不代表这里一定出现第二个符号变量。当前实际 LoopBody 已把三个固定项展开；实验输出中的三条 MemoryDep 是目标机制的直接证据，不能凭 `sum(dim=1)` 名称杜撰一个运行时 reduction 变量。

### normalize 改变表示，不改变访问集合的含义

当 `normalize=False`，返回原始 `range_vars`，便于后面比较和调整循环顺序。当 `normalize=True`，记录器可能合并循环，函数把 `range_vars` 置空，依赖自身仍携带规范化后的 `var_names/size`。这不是删除访问，只是选择一份更容易比较的索引表示。

记录器的 `canonicalize` 在 normalize 打开时调用 `V.graph.sizevars._simplify_loops(...)`，取得新的循环尺寸和 reindex，再用规范前缀 `c0、c1...` 改写 index；关闭时只简化原范围并删除 index 没有使用的尾部变量。也就是说，`V.graph.sizevars` 的参与发生在依赖构造器内部，而不是 `extract_read_writes` 外层。

```text
normalize=False：外层 ReadWrites.range_vars 保存原符号，如 [d0,d1]。
normalize=True ：外层 ReadWrites.range_vars 固定为空；每项 MemoryDep 自己仍保存
                 规范化后的 name、index、var_names 和 size。
```

因此消费者不能在 `normalize=True` 时从外层字段恢复原循环数，但仍可从每项依赖判断它访问哪个名字和哪些位置。

### K02 理解验收

问题：尺寸 `(4,1)`、读取索引 `4*d0+d1` 经过 squeeze 后会保留几个循环变量？

一个。第二维只可能为 0，reindex 给计算体的参数是 `[d0,0]`，所以索引化为 `4*d0`，范围表只有 `{d0:4}`。

问题：fast path 与 slow path 都得到相同的 load/store 时，能否据此认定内部过程也相同？

不能。fast path 遍历 LoopBody 已保存的 `memory_usage`；slow path 切换虚拟 ops handler 并重新调用 callable。它们刻意交出相同 ReadWrites 接口，但触发条件和状态来源不同。

K02 到此完成。下一知识点 K03 打开 `compute_dependencies` 的前半段：DedupList、alias 列表共享、普通 read 的 user 登记以及 OUTPUT 虚拟用户。

## K03/K04 共用源码：一次遍历同时建立 users、mutation 顺序和版本表

`compute_dependencies` 是一个连续函数。进入源码前先固定普通两节点案例：`self.nodes=[op0,op1]`；op0 读 arg0_1 并写 buf0，op1 读 buf0 并写图输出 buf1；没有 alias、mutation 或 unbacked symbol。`name_to_users` 起初为每个输入/输出名准备空 DedupList，版本表为空。遍历 op0 时 arg0_1 增加 user op0；遍历 op1 时 buf0 增加 user op1；末尾 buf1 再增加虚拟 OUTPUT user。随后再把 alias 名设为共享同一列表，并用 K04 的 mutation 案例观察版本表与 weak user 怎样变化。

```python
def compute_dependencies(self) -> None:
    """
        Create dependency edges between nodes, handling aliasing and
        mutation properly.
        """

    T = TypeVar("T")

    class DedupList(Generic[T]):
        """
            This data structure behaves like a list except it makes sure the
            elements remain unique.
            Normally one could use a OrderedSet/dict for this purpose however
            the list in question gets elements appended as it is being
            iterated over which means that we need to keep the list
            semantics.
            """

        def __init__(
            self,
            items: Optional[List[T]] = None,
            membership: Optional[OrderedSet[T]] = None,
        ) -> None:
            self.items = items or []
            self.membership = membership or OrderedSet()

        def append(self, node_user: T) -> None:
            # 教学：membership 判重，items 保持可在遍历期间继续追加的列表语义。
            if node_user in self.membership:
                return
            self.items.append(node_user)
            self.membership.add(node_user)

        def __add__(self, other: DedupList[T]) -> DedupList[T]:
            new_membership = OrderedSet.union(self.membership, other.membership)
            new_items = self.items + [
                x for x in other.items if x not in self.membership
            ]
            return DedupList(new_items, new_membership)

    name_to_users: DefaultDict[str, DedupList[NodeUser]] = collections.defaultdict(
        DedupList
    )

    # handle aliasing by using python aliasing in name_to_users
    # if foo aliases bar then we will make name_to_users["foo"] point
    # to the same python list as name_to_users["bar"]
    for node in self.nodes:
        for buf1 in node.get_outputs():
            buf1_name = buf1.get_name()
            for buf2_name in buf1.get_aliases():
                if buf1_name in name_to_users and buf2_name in name_to_users:
                    # merge the two
                    list1 = name_to_users[buf1_name]
                    list2 = name_to_users[buf2_name]
                    combined = list1 + list2
                    for key in name_to_users.keys():
                        if (
                            name_to_users[key] is list1
                            or name_to_users[key] is list2
                        ):
                            name_to_users[key] = combined
                elif buf1_name in name_to_users:
                    name_to_users[buf2_name] = name_to_users[buf1_name]
                else:
                    name_to_users[buf1_name] = name_to_users[buf2_name]

    def rename(n: str) -> str:
        # 教学：沿 mutation 链递归取得当前最新逻辑版本。
        if n in self.mutation_renames:
            return rename(self.mutation_renames[n])
        return n

    def add_user(
        used_by_name: str,
        user_node: Union[BaseSchedulerNode, OutputNode],
        can_inplace: bool = False,
        is_weak: bool = False,
    ) -> None:
        name_to_users[rename(used_by_name)].append(
            NodeUser(user_node, can_inplace, is_weak)
        )

    unbacked_symbol_to_origin_node: Dict[sympy.Symbol, Optional[str]] = {}

    # NB: None means that the dependency is on an input.  Don't actually
    # generate a dependency because if we do, Inductor will start trying
    # to free the unbacked int but that's pointless
    for name, val in V.graph.graph_inputs.items():
        if isinstance(val, sympy.Expr):
            for fs in val.free_symbols:
                unbacked_symbol_to_origin_node[fs] = None

    for node in self.nodes:
        log.debug("scheduling %s", node.node)

        # unbacked symbols don't follow ordinary buffer dependencies, so
        # we track their def/uses separately
        assert node.node is not None
        unbacked_symbol_defs = sorted(
            node.node.get_unbacked_symbol_defs(), key=lambda x: x.name
        )
        for s in unbacked_symbol_defs:
            assert isinstance(s, sympy.Symbol)
            # Pick the first definer as canonical.  There may be multiple
            # because if a MultiOutputLayout buffer propagates an unbacked
            # symint to multiple outputs, they will all claim to def it.
            if s not in unbacked_symbol_to_origin_node:
                unbacked_symbol_to_origin_node[s] = node.get_name()

        unbacked_symbol_uses = sorted(
            node.node.get_unbacked_symbol_uses(), key=lambda x: x.name
        )
        # if a kernel takes unbacked symints, register dependencies
        for s in unbacked_symbol_uses:
            assert (
                s in unbacked_symbol_to_origin_node
            ), f"{s} not in {unbacked_symbol_to_origin_node}"
            if (r := unbacked_symbol_to_origin_node[s]) is not None:
                for buf in self.name_to_node[r].get_outputs():
                    node.add_fake_dep(StarDep(buf.get_name()))

        if (
            len(node.read_writes.writes) == 1
            and (dep := next(iter(node.read_writes.writes)))
            and isinstance(dep, MemoryDep)
        ):
            node_mode = dep.mode
        else:
            node_mode = None

        # Handle output mutations
        for buf in node.get_outputs():
            # a node will mutate either 0 or 1 buffers
            assert len(buf.get_mutations()) <= 1
            for alt_name in buf.get_mutations():
                alt_name = rename(alt_name)
                # this node must run after the prior writer
                add_user(alt_name, node)
                node.add_fake_dep(StarDep(alt_name, mode=node_mode))
                for user in name_to_users[alt_name].items:
                    if user.get_name() == node.get_name():
                        continue

                    assert isinstance(user.node, BaseSchedulerNode)
                    for other_name in user.node.get_buffer_names():
                        # this node must run after all prior readers
                        other_name = rename(other_name)
                        node.add_fake_dep(
                            WeakDep(other_name, mutating_buf=buf.get_name())
                        )
                        add_user(other_name, node, is_weak=True)

        # add normal non-mutation dependencies
        for read in node.read_writes.reads:
            if not isinstance(read, WeakDep):
                add_user(read.name, node, node.can_inplace(read))

        # 教学：当前节点自身的依赖名改写到此前已知的最新版本。
        node.update_mutated_names(self.mutation_renames)

        # update our renaming scheme for the next iteration
        for buf in node.get_outputs():
            for alt_name in buf.get_mutations():
                self.mutation_renames[rename(alt_name)] = buf.get_name()
                self.mutation_renames[alt_name] = buf.get_name()
                self.mutation_real_name[
                    buf.get_name()
                ] = self.mutation_real_name.get(alt_name, alt_name)

    # make sure outputs aren't dead-code-eliminated
    for buf_name in V.graph.get_output_names():
        log.debug("scheduling output %s", buf_name)
        add_user(buf_name, OutputNode(StarDep(buf_name)))

    # make sure unbacked symints aren't dead-code-eliminated
    for out in V.graph.graph_outputs:
        for s in out.get_unbacked_symbol_uses():
            assert (
                s in unbacked_symbol_to_origin_node
            ), f"{s} not in {unbacked_symbol_to_origin_node.keys()}"
            if r := unbacked_symbol_to_origin_node[s]:
                for buf_name in self.name_to_node[r].get_buffer_names():
                    log.debug(
                        "scheduling output %s for unbacked symint %s", buf_name, s
                    )
                    add_user(buf_name, OutputNode(StarDep(buf_name)))

    # make sure input mutation isn't dead-code-eliminated
    for name in self.mutation_renames:
        if name in V.graph.graph_inputs:
            add_user(name, OutputNode(StarDep(name)))
            V.graph.mutated_inputs.add(name)
        elif name in V.graph.constants:
            # In AOTI, module parameters and buffers are not lifted as graph inputs
            add_user(name, OutputNode(StarDep(name)))

    inp_names = {
        name: index for index, name in enumerate(V.graph.graph_inputs.keys())
    }
    V.graph.mutated_input_idxs = [
        inp_names[name] for name in V.graph.mutated_inputs
    ]

    # copy users information onto the nodes
    for node in self.nodes:
        for buf in node.get_outputs():
            buf.set_users(name_to_users[buf.get_name()].items)
```

## K03：同一 storage 的名字必须看到同一份 users

每个内部结果已有自己的 producer，例如 buf0 是 op0 的输出；本节不重新寻找 producer，而是反向记录“哪些节点会使用这个结果”。`NodeUser(node,can_inplace,is_weak)` 保存消费者节点、它是否可能原地复用该结果，以及这条关系是否只用于排序。`name_to_users[buffer_name]` 就是按结果名字查消费者的临时表，最后复制到对应 `SchedulerBuffer.users`。

同一块 storage 可能同时叫 `base` 和 `view`。从任一名字登记的 reader 都会影响这块 storage 的生命周期和 mutation 顺序，因此这些名字不能各自维护互不相干的 users 表。`name_to_users` 的值使用 `DedupList`：它保留插入顺序、拒绝相同 `NodeUser`，并允许 mutation 循环在遍历旧 users 时追加 weak user。alias 预处理进一步让不同名字指向同一个 `DedupList` 对象。

下面先人为给两张待合并的表各放一个 reader，只为说明 `DedupList.__add__` 怎样保留双方条目；在 `compute_dependencies` 的真实顺序中，alias 预处理早于后面的普通 reader 登记。假设合并时 `base` 的表含 A、`view` 的表含 B：

```text
name_to_users[base] -> list1=[A]
name_to_users[view] -> list2=[B]
combined = list1 + list2 = [A,B]
base ─┐
      ├─> 同一个 combined 对象
view ─┘
```

代码不只改这两个 key；它遍历全部 key，把此前指向 list1 或 list2 的其他 alias 名也重接到 combined。若只有一方已存在，另一方直接指向现有对象；若两方都未显式存在，`defaultdict` 在 else 中创建并共享一份空列表。

### mm→sin 的普通 user 路径

对 op0 mm 和 op1 sin：

1. op0 的输入 reads 是 graph inputs，`add_user(arg0_1,op0)` 与 `add_user(arg1_1,op0)` 登记输入使用者；它们不对应内部 producer。
2. op1 的 `MemoryDep(buf0,c0)` 调用 `add_user(buf0,op1,can_inplace=True)`，于是 buf0 的 users 得到 op1。
3. 循环结束后，图输出 buf1 追加虚拟 `OutputNode(StarDep(buf1))`。
4. 最后 `buf.set_users(...)` 把临时表复制到实际 SchedulerBuffer。

实验 08 实际结果是：

```text
users[buf0] = [op1(weak=False,inplace=True)]
users[buf1] = [OUTPUT(weak=False,inplace=False)]
```

`OUTPUT` 不是计算节点；它使 buf1 在后续 DCE 中具有消费者。`can_inplace=True` 也不是已经决定复用存储，只是把当前依赖关系可供后续内存规划判断的信息一并保留。

### unbacked symbol 走单独的来源表

unbacked symbol 是运行时数据决定的整数，例如“过滤后实际留下多少项”形成的长度 u0。它不是一次对 buf0 某个位置的读取，无法用 `MemoryDep(buf0,index)` 描述；使用者需要的是“先让产生 u0 的工作整体完成”。

```text
op0 运行后得到数据结果 buf0，并定义长度 u0
                         |
                         | 来源表 u0 -> op0
                         v
op1 的输出 shape 使用 u0 → 为 op1 增加 StarDep(buf0)
```

若 u0 直接来自 graph input，表中记录 `u0→None`，因为输入在调度开始前已经可用，不需要建立可释放的内部 buffer 依赖。多个 MultiOutput 都携带 u0 时，源码选择第一次登记的 operation 作为统一来源；这就是这里的 canonical origin，而不是任选一个新定义。

### K03 理解验收

若 `alias_a`、`alias_b` 已共用列表，随后第三个名字 `alias_c` 声明 alias_a，三者最终指向同一对象；从 alias_c 登记 reader 后，按任一名字查 users 都能看到它。复制三份相等列表不够，因为后续 append 只会修改其中一份。

若内部 buf0 没有真实 reader，但它是 graph output，末尾仍为它登记 OUTPUT；因此“没有普通 SchedulerNode user”不能推出它是死值。

K03 到此完成。K04 继续使用上面的同一函数，专门推演 mutation 分支怎样在遍历中加入 WeakDep 并更新版本链。

## K04：mutation 先等待旧 readers，再成为原名的最新版本

案例：

```text
op0：读 arg0_1，写 buf0          # old=x+1
op1：读 arg1_1，写 buf1
     buf1.mutations=[arg0_1]     # x.copy_(y)
```

处理 op0 时，普通 read 把 op0 登记到 `name_to_users[arg0_1]`。处理 op1 的 mutation 输出时：

1. `alt_name=rename(arg0_1)`，此时仍为 `arg0_1`；
2. `add_user(arg0_1,op1)` 让 mutator 成为旧版本的 user；
3. `add_fake_dep(StarDep(arg0_1))` 明确它写的是旧版本之后的新版本；
4. 遍历 `name_to_users[arg0_1].items`，跳过刚加入的 op1，自 op0 取得其输出名 `buf0`；
5. 为 op1 加 `WeakDep(buf0,mutating_buf=buf1)`，并把 op1 作为 weak user 加入 buf0；
6. 当前节点依赖重命名完成后，写入 `mutation_renames[arg0_1]=buf1` 与 `mutation_real_name[buf1]=arg0_1`。

所以最终：

```text
op1.reads = MemoryDep(arg1_1) + StarDep(arg0_1) + WeakDep(buf0, mutating_buf=buf1)
op1.unmet = WeakDep(buf0, mutating_buf=buf1)
mutation_renames = {arg0_1: buf1}
mutation_real_name = {buf1: arg0_1}
```

StarDep 指向已可用 graph input，`set_read_writes` 会从 unmet 裁掉它；WeakDep 指向内部 buf0，因而留下并迫使 op1 等 old reader 完成。

### 为什么遍历时追加不会遗漏或重复

mutation 先把 op1 追加到 alt_name 的 users，再遍历同一 `items`。列表语义允许遍历看见追加后的内容，代码以 `user.get_name()==node.get_name()` 跳过自身。对旧 readers 追加的 weak user 又可能落到别的 buffer 列表；DedupList 的 membership 避免同一关系重复加入。

### 没有旧 reader时就没有 WeakDep

`x.copy_(y); return x*2` 在当前上游表示里，乘法已直接读取 y。处理 mutation 时 `name_to_users[arg0_1]` 中没有旧 reader，因此循环除自身外没有对象，op1 只增加 `StarDep(arg0_1)`。实验 08 实际看到 unmet 为空，但版本表仍更新为 arg0_1→buf1，并给 buf1 添加 OUTPUT，保证调用者可见的输入更新不被 DCE。

### 两次 mutation 的版本链

在第一次 mutation 与第二次 mutation 之间加入一名 reader，才能同时看到 prior writer 和 old readers 怎样变化：

```text
op1：buf1 mutation arg0_1          # 第一次写 x
op2：读 arg0_1，写 mid             # 逻辑上读取第一次写后的 x
op3：buf2 mutation arg0_1          # 第二次写 x
```

第一次 mutation 完成后：

```text
mutation_renames = {arg0_1: buf1}
mutation_real_name = {buf1: arg0_1}
```

处理 op2 的普通 read 时，`add_user` 内部先执行 `rename(arg0_1)`，所以 op2 登记到 `name_to_users[buf1]`；随后 `update_mutated_names` 又把 op2 自己的 read 从 arg0_1 改成 buf1。它等待的是第一次 mutation 产生的新版本，而不是最初输入。

处理 op3 的 mutation 输出时，状态依次变化：

```text
1. alt_name = rename(arg0_1) = buf1
2. add_user(buf1, op3)
3. op3 增加 StarDep(buf1)              # 等第一次 writer op1
4. 遍历 name_to_users[buf1]
   - 跳过 op3 自己
   - 发现旧 reader op2，其输出为 mid
   - op3 增加 WeakDep(mid, mutating_buf=buf2)
5. prune 后：StarDep(buf1) 与 WeakDep(mid,...) 都指向内部结果，均留在 unmet
```

此时 op3 还没有宣布 buf2 是最新版本。它先用上述依赖保证 op1、op2 都已完成，然后才更新版本表：

```text
mutation_renames[arg0_1] = buf2
mutation_renames[buf1]   = buf2
mutation_real_name[buf2] = mutation_real_name.get(buf1,buf1) = arg0_1
```

第二次 mutation 后，新的普通 reader 若仍写着 arg0_1，`rename` 和 `update_mutated_names` 都把它接到 buf2。本函数只负责保存 `mutation_real_name[buf2]=arg0_1`；后续 `process_weak_deps` 用它把 weak dependency 的 mutating buffer 还原成真实名字，代码生成的 buffer-name 映射也用它把最新逻辑版本对应回调用者输入 x。

若两次 mutation 之间没有 op2，第二次 mutator仍有 `StarDep(buf1)` 等第一次 writer，但没有针对中间 reader 的 `WeakDep`。这正是 prior writer 与 old readers 两类约束的区别。

### K04 理解验收

两个旧 readers A、B 都读 arg0_1，A 的 `get_buffer_names()` 为 `{a0}`，B 的为 `{b0}` 时，mutator 最终得到：

```text
reads = 原有数值读取
      + StarDep(arg0_1)
      + WeakDep(a0, mutating_buf=mut)
      + WeakDep(b0, mutating_buf=mut)
unmet = WeakDep(a0, mutating_buf=mut)
      + WeakDep(b0, mutating_buf=mut)
users[a0] 追加 mutator(is_weak=True)
users[b0] 追加 mutator(is_weak=True)
```

`StarDep(arg0_1)` 因 graph input 已可用而从 unmet 裁掉；a0、b0 是内部结果，所以两条 WeakDep 保留。A、B 彼此没有先后关系，但 mutator 必须晚于两者。源码对每名旧 reader 遍历 `get_buffer_names()`；若一名 reader 对应多个 buffer 名，就逐名建立 WeakDep，而不是假设每名 reader 永远只有一个输出。

若 input mutation 的结果不在 Python 返回 tuple 中，末尾仍遍历 `mutation_renames`，为 graph input 原名加入 OUTPUT 并记录 `mutated_input_idxs`。外部可见副作用本身就是保活理由。

下一篇打开这些依赖关系的消费者：拓扑排序、死代码删除、mutation 顺序和最后使用位置。
