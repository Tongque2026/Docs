# 计算 IR 构造：从输入 loader 到 Pointwise、Reduction 与 View

[返回计算 IR](../README.md) · [返回 Lowering](../../README.md) · [返回总览](../../../README.md)

父篇已经用位置表说明三类 IR 的含义。本叶子进一步对应当前实现：lowering 怎样把 TensorBox、Python 常量和符号常量整理成 loader；怎样建立普通输出范围与规约范围；view 怎样在固定存储布局和延迟 reindex 之间选择。

完整源码较长，先把本篇始终追踪的对象固定下来。lowering 传递张量结果时，外层是 `TensorBox`；它里面的 `StorageBox` 保存“这份结果当前由哪种 IR 表示”，因此后续物化可以更新底层表示，而持有同一结果的代码仍能找到更新后的 IR。`Pointwise.create(...)` 和 `Reduction.create(...)` 的正常出口都是这种包装：

```text
TensorBox                         lowering 传递的张量结果
└─ StorageBox                    可更新的结果引用
   └─ Pointwise / Reduction      当前的计算规则
```

这不是示意性的猜测；包装入口本身只有一行：

```python
class TensorBox(MutableBox):
    @staticmethod
    def create(data):
        return TensorBox(StorageBox(data))
```

所以由 `Pointwise.create` / `Reduction.create` 走 `TensorBox.create` 得到的 `result.data` 是 StorageBox，`result.data.data` 才是当前 Pointwise/Reduction。后文用双层 `.data` 判断这类结果，正是在拆两层包装。`TensorBox` 也可以在其他位置被直接构造成 `TensorBox(ReinterpretView(...))` 或 `TensorBox(ComputedBuffer(...))`；“双层 data”不是所有 TensorBox 的永久结构。

TensorBox 和 StorageBox 都继承 `MutableBox`。TensorBox 上调用 `realize()` 时，第一层方法把调用转给 `self.data.realize()`；未显式定义的 `make_loader()` 等可调用属性由 `__getattr__` 同样转发。因此 `result.realize()` 的实际链条是 `TensorBox.realize → StorageBox.realize`，`intermediate.make_loader()` 则转发到当前 StorageBox/ComputedBuffer 的 loader：

```python
class MutableBox(IRNode):
    data: IRNode

    def __getattr__(self, name):
        fn = getattr(self.data, name)
        if callable(fn):
            return fn
        raise AttributeError(f"{type(self.data).__name__}.{name} not callable")

    def realize(self):
        return self.data.realize()
```

正文按实际阅读顺序完成四件事：构造 Pointwise；准备 Reduction 的两组坐标；读懂 Reduction 对象、拆分决策和 `create` 的所有出口；构造 View。Reduction 部分较长，下面这张账本给出各节接力的对象：

| 顺序 | 本节回答的问题 | 接给下一节的状态 |
| --- | --- | --- |
| K03-1 | 一项值怎样被组合成一个输出 | `ranges`、`reduction_ranges`、`inner_fn`、dtype、type、hint |
| K03-2 | 大规约是否拆分，若拆分采用哪种方向 | `(split, hint)` |
| K03-3 | 0、1、小静态、单层和多层分别返回什么 | Pointwise、Reduction 或两层构造请求 |
| K03-4 | 第一层怎样把长规约切成带遮罩的小块 | 新 ranges、block size、mask、局部结果规则 |
| K03-5 | 局部结果怎样物化并被第二层读取 | 中间 buffer 与最终 Reduction |

这里的“物化”只表示：把延迟计算规则登记成一个有名字、可被别的 IR 读取的中间结果；布局怎样选由下一篇讲。本篇负责构造 IR，不负责列出节点退出时所有可能触发物化的消费者条件。K03-5 会完整展示本篇自身主动调用 `realize()` 的一处：两层规约必须先保存第一层结果，第二层才能读取。

对 StorageBox 调用 `realize()` 时，当前实现完成下面的原地替换：

```text
调用前：同一个 StorageBox → Pointwise / Reduction
                             |
                             └─ device、dtype、size、计算规则

realize：用这些字段建立 ComputedBuffer(FlexibleLayout, data=旧规则)
         向 graph 注册 buffer 和 operation，并取得名字 bufN

调用后：同一个 StorageBox → ComputedBuffer(name=bufN)
                             └─ data → 原 Pointwise / Reduction
```

StorageBox 对象没有换掉，所以已有 TensorBox 仍指向它；变化的是 `StorageBox.data`。此后 `make_loader()` 生成的是读取具名 ComputedBuffer 的规则。若 data 原本已经是 ComputedBuffer、输入 buffer 或 ReinterpretView，`realize()` 直接返回现有名字，不再包一层。

对应方法的关键顺序完整列在下面：先判断是否已经有存储，再保存旧规则的来源信息，随后替换 `self.data`，最后注册并返回名字。

```python
def realize(self):
    if isinstance(
        self.data,
        (ComputedBuffer, InputsKernel, InputBuffer, ReinterpretView, TemplateBuffer),
    ):
        return self.data.get_name()
    assert isinstance(self.data, (Pointwise, Reduction, Scan, Sort)), type(self.data)
    origin_node = self.data.get_origin_node()
    traceback = self.data.get_traceback()
    self.data = ComputedBuffer(
        name=None,
        layout=FlexibleLayout(
            device=self.data.get_device(),
            dtype=self.data.get_dtype(),
            size=self.data.get_size(),
        ),
        data=self.data,
    )
    self.data.name = V.graph.register_buffer(self.data)
    V.graph.register_operation(self.data)
    self.data.origins = self.origins
    self.data.origin_node = origin_node
    self.data.traceback = traceback
    return self.data.name
```

## K01：Pointwise 捕获同一输出位置的全部输入

对 `z=x+2`，常数 2 没有 shape 和 loader。`promote_constants` 先以 x 的 dtype、device、size 将它变成可广播常量 view；`make_pointwise` 再取得 x 与常量各自的 loader。构造结束时没有遍历任何元素，只保存：

```text
ranges = x.size
inner_fn(index) = add(x_loader(index), constant_loader(index))
```

若输入全是 Python 或 SymPy 常量，`promote_constants` 会把它们变成标量 `Constant`/`IndexingConstant`。二者继承的 `BaseConstant.get_size()` 返回空 tuple，因此继续进入 `make_pointwise` 时可取到 `ranges=()`，构造零维标量 Pointwise；它们不会凭空得到非空 tensor shape。下文的 `x+2` 则展示至少一个 tensor 提供 `[2,3]` ranges 时，常量怎样被广播到同一位置空间。

### 完整源码：常量提升

主案例调用时 `inputs=[x_ir,2]`：x_ir 是 float32 CPU、size `[2,3]` 的 TensorBox，2 是 Python int。`override_return_dtype=None`，`type_promotion_kind=None`，所以先采用默认类型提升；循环找到 x_ir 作为设备和 shape 参照，把常数 2 变成同 device、可广播到 `[2,3]` 的常量 IR。第一项判断中的 `and/or` 都产生 Python bool，用来选择默认配置或返回分支，并不是张量逐元素运算。

```python
def promote_constants(inputs, override_return_dtype=None, type_promotion_kind=None):
    # 教学：调用方只能直接指定返回 dtype，或指定类型提升种类，不能同时指定。
    assert (
        override_return_dtype is None or type_promotion_kind is None
    ), "only one of override_return_dtype or type_promotion_kind may be given"

    if override_return_dtype is None and type_promotion_kind is None:
        type_promotion_kind = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT

    # 教学：没有 Python/SymPy 常量时，原输入序列无需改写。
    if not any(isinstance(x, (sympy.Basic, int, float)) for x in inputs):
        return inputs
    if all(isinstance(x, (int, float, sympy.Basic)) for x in inputs):
        # 教学：全是常量时先计算共同 dtype，再分别建立标量 IR。
        dtype = override_return_dtype or get_promoted_dtype(
            *inputs, type_promotion_kind=type_promotion_kind
        )

        def const_func(x):
            if isinstance(x, sympy.Basic):
                return ir.IndexingConstant(x, dtype, decode_device(None))
            else:
                return ir.Constant(x, dtype, decode_device(None))

        return [const_func(x) for x in inputs]
    # 教学：混合输入选择第一份 tensor/view/IR Constant 作为 shape、dtype、device 参照。
    ex = next(x for x in inputs if isinstance(x, (TensorBox, ExpandView, ir.Constant)))
    out = []
    for x in inputs:
        if isinstance(x, (int, float)):
            # 教学：Python 标量先变 Constant，再广播到参照 tensor 的 size。
            out.append(
                ExpandView.create(
                    ir.Constant(x, ex.get_dtype(), ex.get_device()), list(ex.get_size())
                )
            )
        elif isinstance(x, sympy.Basic):
            # 教学：符号标量使用 IndexingConstant，随后同样广播。
            out.append(
                ExpandView.create(
                    IndexingConstant(x, ex.get_dtype(), ex.get_device()),
                    list(ex.get_size()),
                )
            )
        else:
            out.append(x)

    return out
```

### 完整源码：make_pointwise

第一次阅读只走普通 add：输入是两个 CPU float32 TensorBox，shape 都为 `[2,3]`；`override_return_dtype=None`、`override_device=None`、两个替代函数均为 None、`allow_alpha=False`、`triton_fallback=None`，调用时也没有 alpha。工厂把输入变成两个 loader，以第一项的 device/shape 为出口，dtype promotion 已由调用它的 lowering 完成；最终建立 `Pointwise(device=cpu,dtype=float32,ranges=[2,3],inner_fn=...)`。后面的 bool、GPU float64、alpha 与 fallback 分支各自只在对应选项非空时改变这条主路径。

```python
def make_pointwise(
    fn,
    override_return_dtype=None,
    override_device=None,
    override_fn_when_input_bool=None,
    override_fn_when_gpu_float64=None,
    allow_alpha=False,
    triton_fallback=None,
):
    def inner(*inputs: List[TensorBox], alpha=None):
        # 教学：若任一输入来自 Triton 且注册了专用 fallback，直接走该实现。
        if triton_fallback is not None and any(map(is_triton, inputs)):
            assert not allow_alpha  # not implemented
            return triton_fallback(*inputs)

        inputs = promote_constants(inputs, override_return_dtype)
        if allow_alpha:
            # 教学：alpha≠1 时先把最后一个输入乘 alpha，再进入统一 loader 路径。
            if alpha is not None and alpha != 1:
                inputs = list(inputs)
                inputs[-1] = mul(inputs[-1], alpha)
        else:
            assert alpha is None
        # 教学：每份输入交出“给定逻辑 index 怎样读取”的 callable。
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_return_dtype or inputs[0].get_dtype()
        is_gpu_device = is_gpu(decode_device(inputs[0].get_device()).type)

        # 教学：非标量常量的其他输入必须与第一项维数一致；广播应在此前变成 view。
        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(
                other.get_size()
            ), f"ndim mismatch {fn} {ranges} {other.get_size()}"

        # in tracing, we will annotate pointwise nodes that correspond to the output of
        # a pointwise node that would have been run in eager. intermediary pointwise nodes
        # during decompositions are not annotated.
        emulate_precision_casts = (
            V.graph is not None
            and getattr(V.graph, "current_node", None) is not None
            and V.graph.current_node.meta is not None
            and V.graph.current_node.meta.get("low_precision_pointwise_barrier", False)
            and dtype in (torch.bfloat16, torch.float16)
        )

        def inner_fn(index):
            assert len(index) == len(ranges), f"wrong ndim {index} {ranges}"
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                # 教学：bool 可使用算子注册时提供的替代语义。
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            elif (
                override_fn_when_gpu_float64
                and is_gpu_device
                and dtype == torch.float64
            ):
                # 教学：GPU float64 也可选择专门替代函数。
                return override_fn_when_gpu_float64(*[load(index) for load in loaders])
            else:
                inputs_loaded = []
                for load in loaders:
                    out = load(index)
                    if emulate_precision_casts:
                        # 教学：先下转再回计算 dtype，模拟 eager 中间低精度截断。
                        downcast = ops.to_dtype(out, dtype, use_compute_types=False)
                        out = ops.to_dtype(downcast, dtype)
                    inputs_loaded.append(out)

                out = fn(*inputs_loaded)
                if emulate_precision_casts:
                    # fp16/bf16 kernels are computed in fp32. Casting down to fp16/bf16 here,
                    # then upcasting again, to emulate casts that eager would do.
                    downcast = ops.to_dtype(out, dtype, use_compute_types=False)
                    return ops.to_dtype(downcast, dtype)
                return out

        if not override_device:
            # 教学：混合输入中优先选择首个 GPU device，否则使用第一项 device。
            device = None
            for i in inputs:
                if is_gpu(i.get_device().type):
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()

        device = override_device or device

        # 教学：Loops.create 最终把这四项包装成 TensorBox。
        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
        )

    return inner
```

### 完整源码：Loops 的构造与索引

`Pointwise.create` 继承下面两个 `Loops` 方法：

```python
@classmethod
def create(cls, *args, **kwargs):
    # 教学：允许调用方显式提供来源与 traceback，不把它们传给 dataclass 构造器。
    origin_node = kwargs.pop("origin_node", None)
    tb = kwargs.pop("traceback", None)
    r = cls(*args, **kwargs)
    r.origin_node = origin_node
    r.traceback = (
        tb or traceback.format_stack() if config.debug_ir_traceback else None
    )
    # 教学：裸 Pointwise/Reduction 经 StorageBox 包装后成为 lowering 可传递的 TensorBox。
    return TensorBox.create(r)

@staticmethod
def _index(ranges, prefix=SymT.INDEX):
    # 教学：size=1 的维度索引恒为 0；其余维度得到 i0/i1... 或指定前缀符号。
    return [
        sympy.Integer(0) if s == 1 else sympy_index_symbol_with_prefix(prefix, n)
        for n, s in enumerate(ranges)
    ]
```

### 完整源码：Pointwise 的 loader 与 store

```python
def nop_loader_fn(idx: Union[Expr, Sequence[Expr]], *, dtype: torch.dtype) -> OpsValue:
    # 教学：零元素循环不会真正读取；浮点使用 nan，其他类型使用 0 作为占位表达式。
    if dtype.is_floating_point:
        return ops.constant(float("nan"), dtype)
    else:
        return ops.constant(0, dtype)


class Pointwise(Loops):
    def make_loader(self):
        # Make zero-element loops into a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.dtype)

        return self.inner_fn

    def get_reduction_size(self):
        return []

    def get_reduction_type(self):
        return None

    def store_output(self, output_name, indexer, vars):
        # 教学：codegen 给出循环 vars 时，同一 vars 驱动读取规则并写到输出 indexer。
        loader = self.make_loader()
        return ops.store(output_name, indexer(vars), loader(vars))

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(device, self.dtype, loader, self.ranges)
```

## 具体推演：2×3 tensor 加常数 2

输入 x 的 size `[2,3]`、dtype float32、device cpu。`promote_constants((x,2))` 选择 x 为参照，将 2 变成 float32 CPU Constant，再经 ExpandView 广播为 `[2,3]`。

```text
loaders[0] = x.make_loader()
loaders[1] = expanded_2.make_loader()
ranges     = [2,3]
dtype      = float32
device     = cpu
```

调用 `Pointwise._index([2,3])` 得到符号 `[i0,i1]`。当 codegen 分析位置 `[1,2]`：

```text
x loader([1,2]) → load x[1,2] → 5
2 loader([1,2]) → 广播维映回常量 → 2
fn(5,2) → add → 7
store indexer([1,2]) ← 7
```

`inner_fn` 的构造发生在 Lowering；上面的具体 5、2、7 是按规则推演。真正值由生成程序运行时读取与计算。

## 从简单路径到条件路径

| 条件变化 | 哪一步改变 | 结果表示 |
| --- | --- | --- |
| `x+y`，二者已广播成同维 | 两份 loader 同收 index | 普通 Pointwise |
| `add(x,y,alpha=2)` 且 `allow_alpha` | y 先变为 `mul(y,2)` | outer add 捕获新的 y loader |
| bool dtype 且有 override | inner_fn 直接调用 bool 替代函数 | ranges/device 不变 |
| GPU float64 且有 override | 调用 GPU float64 替代函数 | 不走普通 fn |
| fp16/bf16 barrier | 每个输入及输出插入 downcast/upcast | 模拟 eager 精度截断 |
| 任一 Triton 输入且有 fallback | 在常量提升前直接返回 fallback | 不构造本 Pointwise |

这些分支改变“当前位置怎样算”，但普通路径的构造出口仍是 `TensorBox(StorageBox(Pointwise))`。此时 `StorageBox.data` 仍指向 Pointwise 规则，还没有被本函数改成具名 buffer；以后哪个消费者要求保存它，属于这项构造完成后的处理。

`make_pointwise` 在这里显式检查的是输入 rank，不逐维比较 shape。逐维广播关系应已经由各输入的 view/loader 表达；因此两个同 rank、不同 shape 的输入能否安全组合，不能只凭这段 rank 断言判断，要看 lowering 在调用前交入了什么 view。

## size=1 与 size=0 的边界

shape `[2,1,3]` 的 `_index` 得到 `[i0,0,i2]`。中间维永远只有合法位置 0，直接使用常数能让索引式更简单，并与广播 view 的清零语义一致。

若 ranges 含 0，`Pointwise.make_loader()` 返回 `nop_loader_fn`。循环没有合法输出位置，因此占位 nan/0 不会作为用户 tensor 元素被写出；它让 IR 分析仍有一个可调用 loader，而不去执行原 inner_fn 的潜在非法读取。

## Pointwise 边界对照

1. 输入 `(3,4)` 全是 Python 整数：`promote_constants` 计算共同 dtype，返回两个标量 Constant；没有 tensor shape 可广播。
2. x 为 CPU、y 为 GPU，未 override device：扫描输入后选择首个 GPU device。
3. x、y 的 rank 分别为 2 和 1，且 y 尚未经过广播 view：维数断言失败；`make_pointwise` 不负责在这里发明广播映射。
4. ranges `[1,1]`：索引是 `[0,0]`，但输出仍有一个元素，不走 zero-element loader。
5. `allow_alpha=False` 却传 alpha=2：在建立 loaders 前断言失败。

## K02：把一组输入坐标拆成输出坐标和组内坐标

规约 lowering 的第一项工作不是相加，而是决定哪些维度留下、哪些维度被组合。对 shape `[2,3]` 的 `sum(dim=1)`：维度 0 留下，成为 `ranges=[2]`；维度 1 被规约，成为 `reduction_ranges=[3]`。loader 收到两组 index 后再恢复原输入坐标 `[row,column]`。

### 完整源码：规范化 reduction axis

```python
def _validate_reduction_axis(x, axis):
    size = x.get_size()
    # 教学：单个整数统一包装成列表；None 或其他 falsy axis 表示所有维度。
    if isinstance(axis, int):
        axis = [axis]
    elif not axis:
        axis = range(len(size))
    if len(size) == 0:
        # 教学：标量只接受空、0 或 -1，统一交回“没有实际维度”的空列表。
        assert tuple(axis) in [(), (0,), (-1,)], f"invalid axis: {axis}"
        return []
    axis = list(axis)
    for i in range(len(axis)):
        if axis[i] < 0:
            # 教学：负轴加 rank，-1 因而变成最后一维。
            axis[i] += len(size) if len(size) else 1
        assert 0 <= axis[i] < len(size) or (len(size) == 0 and axis[i] == 0)
    # 教学：同一维不能重复规约。
    assert len(set(axis)) == len(axis), "reduction axis not unique"
    return axis
```

注意当前实现的 `elif not axis`：`axis=None` 和空列表 `axis=[]` 都会变成“规约所有维度”，而不是“一个维度也不规约”。这是当前 lowering 的接口行为。

### 完整源码：建立两组范围和坐标重组 loader

```python
def _make_reduction_inner(x, *, axis, keepdims, dtype, override_return_dtype):
    # 教学：显式 dtype 先转换输入，后面的 src_dtype 与 loader 都看到转换后 tensor。
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = set(_validate_reduction_axis(x, axis))

    kept_sizes = []
    kept_idx = []
    reduced_sizes = []
    reduced_idx = []
    # 教学：按原维度顺序分组；set 只用于成员判断，不改变输出维的排列。
    for i in range(len(size)):
        if i in axis:
            reduced_idx.append(i)
            reduced_sizes.append(size[i])
        else:
            kept_idx.append(i)
            kept_sizes.append(size[i])

    def loader(index, reduction_index):
        assert len(reduction_index) == len(reduced_idx)
        if keepdims:
            # 教学：keepdim 输出 index 仍含 size-1 规约维，先只取 kept 位置。
            assert len(index) == len(size)
            index = [index[i] for i in kept_idx]
        assert len(index) == len(kept_idx)
        # 教学：建立与原输入 rank 相同的空槽，再按原维号放回两组坐标。
        new_index = [None] * (len(index) + len(reduction_index))
        for idx, var in itertools.chain(
            zip(kept_idx, index), zip(reduced_idx, reduction_index)
        ):
            new_index[idx] = var
        return inner_loader(new_index)

    if keepdims:
        # 教学：输出保留原 rank，但每个 reduced dimension 的 size 改为 1。
        new_size = list(size)
        for i in reduced_idx:
            new_size[i] = sympy.Integer(1)
    else:
        new_size = kept_sizes

    # 教学：捕获转换后 x 的 loader；真正按 index 读取发生在后续 IR 分析/生成。
    inner_loader = x.make_loader()
    return dict(
        device=x.get_device(),
        dst_dtype=override_return_dtype or x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=loader,
        ranges=new_size,
        reduction_ranges=reduced_sizes,
    )
```

### 完整源码：构造 Reduction 并决定是否立即实现

```python
def make_reduction(reduction_type: str, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
        )
        # 教学：K03 的 create 可能返回真正 Reduction，也可能退化成 Pointwise。
        result = Reduction.create(reduction_type=reduction_type, input_node=x, **kwargs)
        if isinstance(
            result.data.data, Reduction
        ):  # Only realize if reduction isn't unrolled
            # 教学：仅真正的 Reduction 立即形成调度边界；展开后的 Pointwise 保持延迟。
            result.realize()
        return result

    return inner
```

### 完整源码：sum 怎样确定输入与累计值 dtype

```python
@register_lowering([aten.sum, prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    # 教学：整数和 bool 在未显式指定 dtype 时默认以 int64 规约并输出。
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    fn = make_reduction("sum", override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)
```

## 2×3 输入的完整坐标状态

设 x：

```text
shape=[2,3]
x=[[0,1,2],
   [3,4,5]]
axis=1
```

规范化后 `axis={1}`。按原维度扫描得到：

```text
kept_idx=[0]       kept_sizes=[2]
reduced_idx=[1]    reduced_sizes=[3]
ranges=[2]         reduction_ranges=[3]
```

生成输出 row=1 时，外层 index 与组内 reduction_index 的变化是：

| index | reduction_index | new_index 槽位回填 | input loader | sum 状态 |
| --- | --- | --- | ---: | ---: |
| `[1]` | `[0]` | `[1,0]` | 3 | 3 |
| `[1]` | `[1]` | `[1,1]` | 4 | 7 |
| `[1]` | `[2]` | `[1,2]` | 5 | 12 |

`new_index` 的长度为 `1+1=2`，等于原输入 rank。kept 坐标放回槽 0，reduction 坐标放回槽 1，所以 loader 始终收到合法原输入坐标。

## keepdims 只改变外层 index 接口

同一请求设 `keepdims=True`：

```text
ranges=[2,1]
reduction_ranges=[3]
```

输出 row=1 的 index 是 `[1,0]`。loader 先用 `kept_idx=[0]` 取出 `[1]`，再和 column 组合回 `[1,column]`。因此每一行仍读取同样三项；变化只是输出 shape 从 `[2]` 变成 `[2,1]`。

若 axis 写成 -1，验证函数加 rank 2 后同样得到 1，后续状态完全一致。若 axis 为 `[1,1]`，重复轴断言在分组前失败。

## dtype 状态：int32 sum 为什么变成 int64

输入 x 为 int32，调用 `sum_(x,dim=1,dtype=None)`：

```text
sum_：dtype ← int64
make_reduction：override_return_dtype=int64
_make_reduction_inner：先 to_dtype(x,int64)
src_dtype = 转换后 x.dtype = int64
dst_dtype = override = int64
```

因此 loader 读取的是转换后的 int64 值，规约累加与输出都是 int64。浮点输入且 dtype=None 时不触发这项默认提升，src/dst 沿用输入 dtype。

## 构造后为何检查实际返回类型

`Reduction.create` 并不保证永远返回 Reduction。规约范围为 0、1 或满足小静态展开条件时，它可能返回 Pointwise。`make_reduction` 因此检查 `result.data.data` 的实际类型：

- 真正 Reduction：立即 `realize()`，形成明确调度边界；
- Pointwise：不在这里 realize，仍可与后续逐元素规则组合。

这解释了为何父篇用 2×128 实验确认 Reduction IR，而 2×3 只作为坐标和数值教学例。K03 将完整展开 create 的实际分支条件。

## Reduction 参数边界对照

1. shape `[2,3,4]`，axis `[0,2]`：`kept_idx=[1]`、`ranges=[3]`、`reduction_ranges=[2,4]`；loader 将 index 放槽 1，将两个 reduction index 放槽 0、2。
2. axis=None：falsy 分支选择所有维，`ranges=[]`，结果是标量接口。
3. axis=[]：当前源码同样选择所有维，不是 identity reduction。
4. 标量输入 axis=-1：验证后交回空 axis；输入和输出坐标均没有实际维。
5. keepdims=True 规约 `[0,2]`：输出 ranges 为 `[1,3,1]`，loader 只从外层 index 取维 1。

这里列的是 lowering 内部 helper 收到 `axis` 后的行为。用户层 API 是否接受空列表、是否在进入该 helper 前改写参数，由更早的 schema 和图处理决定；本文据此判断的是已经到达 `_validate_reduction_axis` 的值。

## K03-1：Reduction 保存两套范围，并把单项值交给组合操作

进入 `Reduction.create` 分支前，先看真正 Reduction 对象怎样参与生成。它在 Pointwise 的 `ranges` 之外增加 `reduction_ranges`、`reduction_type`、输入值 dtype 和 hint。codegen 提供两组循环变量，`inner_fn(vars,reduction_vars)` 产生本次组内位置的单项值，`ops.reduction` 才负责把许多单项组合起来。

### 完整源码：组合规则选择

```python
REDUCTION_COMBINE_FN: Dict[str, Callable[..., OpsValue]] = {
    "any": ops_wrapper("logical_or"),
    "max": ops_wrapper("maximum"),
    "min": ops_wrapper("minimum"),
    "prod": ops_wrapper("mul"),
    "sum": ops_wrapper("add"),
    "xor_sum": ops_wrapper("bitwise_xor"),
}


def get_reduction_combine_fn(
    reduction_type: str, dtype: torch.dtype, arg_break_ties_left: bool = True
) -> Callable[..., object]:
    # 教学：简单二元组合直接从注册表取得 ops 包装函数。
    if reduction_type in REDUCTION_COMBINE_FN:
        return REDUCTION_COMBINE_FN[reduction_type]

    elif reduction_type in ("argmax", "argmin"):

        def argmax_combine_fn(
            a: Tuple[object, object], b: Tuple[object, object]
        ) -> Tuple[OpsValue, OpsValue]:
            # 教学：arg reduction 的状态是 (当前最佳值, 当前索引)。
            a_value, a_index = a
            b_value, b_index = b

            if reduction_type == "argmin":
                mask = ops.lt(a_value, b_value)
            else:
                mask = ops.gt(a_value, b_value)

            equal = ops.eq(a_value, b_value)
            if is_float_dtype(dtype):
                # 教学：NaN 通过 value!=value 识别，并参与最佳值及相等判断。
                a_isnan = ops.ne(a_value, a_value)
                b_isnan = ops.ne(b_value, b_value)
                mask = ops.logical_or(mask, ops.gt(a_isnan, b_isnan))
                equal = ops.logical_or(equal, ops.logical_and(a_isnan, b_isnan))

            # 教学：值相等时按配置选择较左或较右索引。
            tie = (
                ops.lt(a_index, b_index)
                if arg_break_ties_left
                else ops.gt(a_index, b_index)
            )
            mask = ops.logical_or(mask, ops.logical_and(equal, tie))
            return (
                ops.where(mask, a_value, b_value),
                ops.where(mask, a_index, b_index),
            )

        return argmax_combine_fn

    elif reduction_type == "welford_combine":

        def welford_combine_fn(
            a: Tuple[OpsValue, OpsValue, OpsValue],
            b: Tuple[OpsValue, OpsValue, OpsValue],
        ) -> Tuple[OpsValue, OpsValue, OpsValue]:
            # 教学：每部分状态保存 mean、平方离差和 m2、weight。
            a_mean, a_m2, a_weight = a
            b_mean, b_m2, b_weight = b

            delta = b_mean - a_mean
            new_weight = a_weight + b_weight
            w2_over_w = b_weight / new_weight
            return (
                a_mean + delta * w2_over_w,
                a_m2 + b_m2 + delta * delta * a_weight * w2_over_w,
                new_weight,
            )

        return welford_combine_fn

    else:
        # 教学：未登记的 reduction type 在构造组合函数时明确失败。
        raise NotImplementedError(f"unknown reduction_type={reduction_type}")
```

### 完整源码：Reduction 对象的执行接口

下面完整保留 `Reduction` 类从字段声明到 `constant_to_device` 的连续前缀。先代入 `[2,128]` 沿第 1 维求和：device=CPU、dst/src dtype 都是 float32、`ranges=[2]`、`reduction_ranges=[128]`、type=sum、hint=DEFAULT；inner_fn 接收输出坐标 `[i0]` 与组内坐标 `[r0]`，读取 `x[128*i0+r0]`。`store_reduction` 只用 i0 作为输出索引，把 128 项合并后的值写到两个输出位置。后续再按源码顺序展开 split 与多层 helper。

```python
@dataclasses.dataclass
class Reduction(Loops):
    reduction_ranges: List[Expr]
    reduction_type: str
    # self.dtype represents the dst dtype
    src_dtype: torch.dtype
    reduction_hint: ReductionHint

    def __str__(self) -> str:  # type: ignore[override]
        return Loops.__str__(  # type: ignore[call-arg]
            self, names=("ranges", "reduction_ranges", "reduction_type")
        )

    def __repr__(self) -> str:  # type: ignore[override]
        return self.__str__()

    def get_unbacked_symbol_uses(self) -> OrderedSet[sympy.Symbol]:
        # 教学：除普通 ranges/inner_fn 外，还收集 reduction_ranges 中的数据依赖符号。
        return super().get_unbacked_symbol_uses() | OrderedSet().union(
            *(free_unbacked_symbols(e) for e in self.reduction_ranges)
        )

    def get_reduction_size(self):
        return self.reduction_ranges

    def get_reduction_type(self):
        return self.reduction_type

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        # 教学：inner_fn 先用外层与组内变量得到一项，ops.reduction 描述组合。
        value = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        # 教学：最终存储位置只由外层 vars 决定，不含 reduction_vars。
        return ops.store_reduction(output_name, indexer(vars), value)

    def index_length(self):
        return len(self.ranges) + len(self.reduction_ranges)

    def inner_fn_args(self):
        # 教学：外层使用 i 前缀；组内使用 r 前缀，size=1 维仍由 _index 变 0。
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.RINDEX)
        return (index, rindex)

    def inner_fn_free_unbacked_symbols(self):
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.RINDEX)
        return extract_free_unbacked_symbols(self.inner_fn, index, rindex)

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device,
            self.dtype,
            loader,
            self.ranges,
            self.reduction_ranges,
            self.reduction_type,
            self.src_dtype,
            ReductionHint.DEFAULT,
        )
```

## sum 的两组循环状态

沿用 K02 的 `x.sum(dim=1)`：

```text
ranges=[2]              → 外层 index=[i0]
reduction_ranges=[3]    → 组内 rindex=[r0]
reduction_type="sum"
src_dtype=dst_dtype=float32
```

为输出 row=1 生成时，`vars=[1]` 固定输出行；`reduction_vars` 依次覆盖 0、1、2。每次 inner_fn 返回 x[1,r0] 的一项值，`ops.reduction(...,"sum",value)` 表达加法组合，`ops.store_reduction(output,indexer([1]),value)` 只写输出位置 1。

如果错误地把 r0 传入 output indexer，会产生三个存储位置；当前接口把两组变量分开，正是为了保证一组输入最后对应一项输出。

## argmax 相等值怎样选索引

给定两份候选状态：

```text
a=(value=7,index=2)
b=(value=7,index=5)
```

argmax 的大于比较为 false，equal 为 true。默认 `arg_break_ties_left=True`，tie 检查 `2<5` 为 true，所以 mask 为 true，两个 `ops.where` 选择 a，结果仍是 `(7,2)`。若配置为 false，tie 改查 `2>5`，结果选择 b。

浮点 NaN 另有显式处理：`v!=v` 标记 NaN；两边都是 NaN 时也视为 equal，再由索引 tie 决定。正文只说明当前组合表达，不把它声称为某次硬件运行观测。

## Welford 两部分怎样合并

若 a 表示 2 项样本，mean=3、m2=2、weight=2；b 表示 1 项样本，mean=6、m2=0、weight=1：

```text
delta=3
new_weight=3
w2_over_w=1/3
new_mean=3+3×1/3=4
new_m2=2+0+3²×2×1/3=8
```

返回 `(4,8,3)`。组合状态不是单个 scalar，因此不能放进简单的 REDUCTION_COMBINE_FN 二元算术表。

## dtype 与 hint 的位置

`self.dtype` 是目标输出 dtype。`src_dtype` 是传给 `ops.reduction` 的“源值语义 dtype”：后端用它选择单位元、累加器和组合规则。普通单层规约中，它通常也等于 `inner_fn` 产生的值 dtype；分层规约会出现例外——第二层读取 fp32 中间 buffer，但仍携带原始低精度 `src_dtype`，让后端沿用原规约的语义设置。`reduction_hint` 不改变数学定义，它给后端提供 inner/outer 等实现倾向；K03-2 接着解释 hint 怎样从读索引和设备规模产生。

`constant_to_device` 只允许所有读取均来自常量的前提下改写 ConstantBuffer device，并将 hint 重置为 DEFAULT；新的设备需要重新选择合适策略。

## Reduction 对象边界对照

1. ranges `[2,1]`、reduction_ranges `[3,1]`：`inner_fn_args` 为 `[i0,0]` 与 `[r0,0]`，`index_length=4`。
2. reduction_type 未出现在简单表、argmin/max 或 welford_combine：`get_reduction_combine_fn` 抛 NotImplementedError。
3. argmin 候选值相等且默认 left tie：同样选择较小索引。
4. reduction_ranges 含 unbacked s0：`get_unbacked_symbol_uses` 会在父类结果上并入 s0。
5. store_reduction 的 output index 只由 vars 构成，reduction_vars 只进入 inner_fn。

## K03-2：num_splits 先决定是否拆，再判断 inner 或 outer

`num_splits` 的返回是 `(ReductionHint, split)`。它先用后端能力、规约类型、配置和静态性决定是否允许拆分；再用设备规模估算 split 数；若不是单输出，还从实际读索引判断 reduction 维在内层还是外层。

### 完整源码：num_splits

先走一个确定分支：CUDA 风格设备有 80 个 SM；SM 是 GPU 上可同时驻留并执行多个 thread block 的流式多处理器，因此数量会影响需要多少并行块才能占满设备。当前 `ranges=[1]`、`reduction_ranges=[16384]`，输出与输入 dtype 都是 float32，type=sum，配置开启 split，范围都可静态求值，后端没有“高效直接规约成单元素”的能力。下面函数先通过 should_split，再按 `num_sm=80`、warp/thread 估算得到 inner hint 和 split=2。把 type 改成 argmax 或把 16384 换成无法静态求值的 u0，会在规模公式前直接返回 `(DEFAULT,1)`。

```python
@staticmethod
def num_splits(
    device,
    dst_dtype,
    src_dtype,
    inner_fn,
    ranges,
    reduction_ranges,
    reduction_type,
    reduction_numel,
    input_node: Optional[IRNode] = None,
):
    def _is_static(x):
        return isinstance(x, (int, sympy.Integer))

    # 教学：symbolic_hint 尝试取得启发式可用的规模值。
    reduction_numel_hint = V.graph.sizevars.symbolic_hint(reduction_numel)
    numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(ranges))

    should_split = (
        # 教学：后端若原生擅长规约到单元素，就不由这里拆层。
        not V.graph.has_feature(device, BackendFeature.REDUCE_TO_SINGLE_ELEMENT)
        and reduction_type
        not in (
            "argmax",
            "argmin",
        )
        and config.split_reductions
        # We don't support unbacked symints
        and _is_static(reduction_numel_hint)
        and _is_static(numel_hint)
    )
    if not should_split:
        return ReductionHint.DEFAULT, 1

    # 教学：取得当前目标设备属性；XPU 与默认 CUDA 路径使用不同字段表示计算单元数。
    device_interface = get_interface_for_device(get_device_type(device))  # type: ignore[arg-type] # next PR
    device_properties = device_interface.Worker.get_device_properties(device)
    if get_device_type(device) == "xpu":
        num_sm = device_properties.gpu_subslice_count
    else:
        # default is cuda behavior
        num_sm = device_properties.multi_processor_count

    min_elements_per_thread = 32
    max_elements_per_thread = 512
    threads_per_sm = 2048
    min_elements_per_device = min_elements_per_thread * num_sm * threads_per_sm
    max_elements_per_device = max_elements_per_thread * num_sm * threads_per_sm

    def inner_reduction_splits(reduction_numel_hint, numel_hint):
        # do heuristics that's close to eager mode for split inner reduction
        # we leak reduction autotune configs here, and will need to refactor to avoid this later
        num_warps = 8
        num_threads = 32 * num_warps
        if numel_hint >= 2 * num_sm:  # don't split if there are enough outputs
            return 1
        if reduction_numel_hint <= 8192:
            return 1
        if reduction_numel_hint * numel_hint <= min_elements_per_device:
            split_size = min_elements_per_thread
        elif reduction_numel_hint * numel_hint < max_elements_per_device:
            target_blocks = num_sm * threads_per_sm // (2 * num_threads)
            blocks_per_output = (target_blocks + numel_hint - 1) // numel_hint
            tmp_split_size = (
                reduction_numel_hint + num_threads * blocks_per_output - 1
            ) // (num_threads * blocks_per_output)
            divisors = sympy.divisors(reduction_numel_hint)
            closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
            if abs(closest - tmp_split_size) < 30:
                # prefer even splits, but never smalle than min_elements_per_thread
                split_size = max(closest, min_elements_per_thread)
            else:
                split_size = tmp_split_size
        else:
            divisors = sympy.divisors(reduction_numel_hint)
            closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
            if abs(closest - max_elements_per_thread) < 50:
                # prefer even splits
                split_size = closest
            else:
                split_size = max_elements_per_thread
        # 教学：每份 split 由 split_size×256 个线程容量承接，向上取整得到份数。
        return (reduction_numel_hint + split_size * num_threads - 1) // (
            split_size * num_threads
        )

    def outer_reduction_splits(reduction_numel_hint, numel_hint):
        # TODO the best heuristic currently has XBLOCK (corresponding to numel_hint) 128
        # extend to even smaller number of outputs
        num_warps = 8
        num_threads = num_warps * 32
        rvals_per_thread = 4  # comes from heuristics, refactor to not leak here
        xvals_per_block = 128
        xblocks = (numel_hint + xvals_per_block - 1) // xvals_per_block
        if reduction_numel_hint * numel_hint < min_elements_per_device:
            split_size = min_elements_per_thread
        elif reduction_numel_hint * numel_hint < max_elements_per_device:
            target_blocks = num_sm * threads_per_sm // (num_threads)
            target_blocks = (target_blocks + xblocks - 1) // xblocks
            tmp_split_size = (
                reduction_numel_hint + rvals_per_thread * target_blocks - 1
            ) // (rvals_per_thread * target_blocks)
            divisors = sympy.divisors(reduction_numel_hint)
            closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
            if abs(tmp_split_size - closest) < 20:
                split_size = max(closest, min_elements_per_thread)
            else:
                split_size = tmp_split_size
        else:
            divisors = sympy.divisors(reduction_numel_hint)
            closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
            if abs(closest - max_elements_per_thread) < 50:
                # prefer even splits
                split_size = closest
            else:
                split_size = max_elements_per_thread

        # 教学：outer 路径按每线程 4 个规约值和 split_size 向上取整。
        return (reduction_numel_hint + rvals_per_thread * split_size - 1) // (
            rvals_per_thread * split_size
        )

    # easy cases
    if numel_hint == 1:
        # 教学：单输出天然按 inner heuristic 处理。
        split = inner_reduction_splits(reduction_numel_hint, numel_hint)
        if split == 1:
            # No need to split.
            return ReductionHint.INNER, split
        if input_node is not None and isinstance(input_node, TensorBox):
            # 教学：若输入链已有匹配总项数的 Reduction 结构，优先复用其 ranges。
            new_ranges, new_reduction_ranges = extract_input_node_reduction_ranges(
                input_node
            )
            if new_ranges is not None and new_reduction_ranges is not None:
                extracted_numel_hint = V.graph.sizevars.symbolic_hint(
                    sympy_product(new_ranges + new_reduction_ranges)
                )
                if reduction_numel_hint == extracted_numel_hint:
                    log.debug(
                        "Use previous IRNode's range and reduction_ranges instead of split. "
                        "current ranges: %s, current reduction ranges: %s, current split: %d, "
                        "new ranges: %s, new reduction ranges: %s",
                        ranges,
                        reduction_ranges,
                        split,
                        new_ranges,
                        new_reduction_ranges,
                    )
                    # If the input_node or its dependent nodes are also Reduction nodes,
                    # use reduction_sizes of this node or its dependent nodes directly.
                    return ReductionHint.INNER, -1
        return ReductionHint.INNER, split
    if (
        # 教学：规约很短，或外层输出足够多可占满设备时，无需拆层。
        reduction_numel_hint <= min_elements_per_thread
        or numel_hint >= num_sm * 2 * 32
    ):
        return ReductionHint.DEFAULT, 1

    # 教学：构造临时 Reduction/ComputedBuffer，只为分析实际读取索引。
    r = Reduction(
        device,
        dst_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        src_dtype,
        ReductionHint.DEFAULT,
    )

    def get_read_indices(r):
        cb = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=r.get_device(),
                dtype=r.get_dtype(),
                size=r.get_size(),
            ),
            data=r,
        )
        read_writes = cb.get_read_writes()
        # try finding the full size producer
        # TODO this will fail for something like ((1, N) * (N, 1)).sum()
        # this would also possibly be wrong for producers with the different contiguity but we hope those cases are rare
        range_vars = [
            r
            for r in read_writes.range_vars
            if isinstance(r, sympy.Expr) and not isinstance(r, sympy.Number)
        ]
        indices = []
        changed = False
        for md in sorted(read_writes.reads, key=lambda x: x.name):
            # 教学：只采用包含全部循环变量的读取，过滤广播等不完整访问。
            if all(r in md.index.free_symbols for r in range_vars):
                indices.append(md.index)
                if md.name in V.graph.name_to_buffer:
                    buf = V.graph.name_to_buffer[md.name]
                    original_stride = buf.layout.stride
                    buf.decide_layout()
                    if buf.layout.stride != original_stride:
                        changed = True
        return indices, changed

    indices, changed = get_read_indices(r)
    if changed:
        # 教学：布局决定改变索引式时，重新采集一次读索引。
        indices, _ = get_read_indices(r)

    if len(indices) == 0:
        # TODO determine splits when all inputs are broadcast
        return ReductionHint.DEFAULT, 1

    # 教学：产生符号化外层/组内 vars 及范围，用于 stride hint 分析。
    (_, reduction_vars), ranges = dependencies.index_vars_squeeze(
        r.get_size(), r.get_reduction_size()
    )
    num_outer = 0
    num_inner = 0
    for i in indices:
        i = V.graph.sizevars.simplify_with_ranges(i, ranges)
        strides = V.graph.sizevars.stride_hints(i, reduction_vars, ranges.keys())
        # 教学：所有 reduction-var stride 都大于 1 记 outer，否则记 inner。
        outer = all(s > 1 for s in strides)
        if outer:
            num_outer += 1
        else:
            num_inner += 1
    if num_inner > num_outer:
        return ReductionHint.INNER, inner_reduction_splits(
            reduction_numel_hint, numel_hint
        )
    else:
        # 教学：平票也归 outer。
        return ReductionHint.OUTER, outer_reduction_splits(
            reduction_numel_hint, numel_hint
        )
```

## 第一关：哪些情况绝不 split

`should_split` 必须同时满足五类条件：后端没有 `REDUCE_TO_SINGLE_ELEMENT` 能力；不是 argmax/argmin；配置开启；规约项数与输出项数都能取得静态整数 hint。任何一项失败都返回 `(DEFAULT,1)`。

这意味着“规约很大”本身不足以推出多层：后端能力、类型、动态性和配置都在规模计算之前。

## 单输出 inner 示例：16384 项怎样得到 split=2

以下只代入源码公式，假定 CUDA 风格设备 `num_sm=80`，不是本机测量：

```text
numel_hint=1
reduction_numel_hint=16384
min_elements_per_device=32×80×2048=5,242,880
num_threads=8×32=256
```

输出数 1 小于 `2×80`，规约项数大于 8192；总工作 16384 小于最小设备阈值，所以 `split_size=32`：

```text
split=ceil(16384 / (32×256))=ceil(2)=2
```

若 input_node 中能提取一组 ranges/reduction_ranges，且总项数同为 16384，函数返回 `(INNER,-1)`，让 K03-3 走“复用已有 ranges”的 multilayer 路径；否则返回 `(INNER,2)`。

这里的“能提取”由 `extract_input_node_reduction_ranges(input_node)` 按下面的对象链判断。`num_splits` 的调用点要求 `input_node` 是 TensorBox，但它的 data 不一定是 StorageBox：由 `TensorBox.create` 产生的 Loops 通常有 StorageBox 包装，其他路径也可能直接构造 `TensorBox(ComputedBuffer)`。首分支检查后一种直接包装；普通 Loops 结果则走 `input_node.data.data`：

```text
input_node.data 直接是 ComputedBuffer
  ├─ 它带非空 reduction_size → 直接返回它的 size 与 reduction_size
  └─ reduction_size 为空      → 提取失败

input_node 仍包着 Loops
  └─ 从 input_node.get_reads() 沿每个 MemoryDep 查 defining op
       ├─ 遇到带 reduction_size 的 ComputedBuffer
       │    ├─ 首个：记下 size 与 reduction_size
       │    ├─ 后续与它相同：继续
       │    └─ 后续不同：提取失败
       └─ 遇到其他 defining op：继续沿它的 reads 向前找
```

搜索按 frontier 一层层向前推进。某一轮第一次发现 Reduction 后，会继续扫描这一轮剩余 reads，要求同一轮直接发现的 Reduction 形状一致；循环条件 `while reduction_size is None` 随后使搜索停止，不再深入本轮收集到的更早 `new_reads`。因此它检查的是“最近一层被发现的 Reduction”，不是整个依赖闭包。

`num_splits` 随后比较提取范围的元素总数与当前 `reduction_numel_hint`，只有相等才返回 `split=-1`。例如同一轮发现一个 `ComputedBuffer(size=[2], reduction_size=[8192])`，总项数为 16384，可复用；若同一轮另一条 read 直接发现 `size=[1], reduction_size=[16384]`，两组形状不同，提取失败。若冲突 Reduction 藏在还需继续展开的更深支路，而本轮另一支路已经找到了 Reduction，当前函数会在本轮结束后停止，并不会继续检查那个更深冲突。

完整搜索循环如下。`seen` 只在当前一层读取中去重；`reads == new_reads` 只处理“下一层没有变化”的停滞情况。这里没有跨层 visited 集合。编译图的 defining-op 依赖按已经产生的结果指向更早的结果，因此正常输入是一张无环图；若脱离这个前提人为构造 A→B→A，这段函数自身不能阻止两组 reads 交替。

```python
def extract_input_node_reduction_ranges(input_node):
    if isinstance(input_node.data, ComputedBuffer):
        size = input_node.get_size()
        reduction_size = input_node.get_reduction_size()
        if len(reduction_size) > 0:
            return (size, reduction_size)
        return (None, None)

    if not isinstance(input_node.data.data, Loops):
        return (None, None)

    reads = input_node.get_reads()
    reduction_size = None
    size = None
    while reduction_size is None and len(reads) > 0:
        seen = OrderedSet()
        new_reads = []
        for read in reads:
            if not isinstance(read, MemoryDep):
                continue
            if read.name in seen:
                continue
            seen.add(read.name)
            buffer = V.graph.try_get_buffer(read.name)
            if buffer is None:
                continue
            op = buffer.get_defining_op()
            if op is None:
                continue

            if isinstance(op, ComputedBuffer) and len(op.get_reduction_size()) > 0:
                if reduction_size is None:
                    reduction_size = op.get_reduction_size()
                    size = op.get_size()
                elif reduction_size != op.get_reduction_size() or size != op.get_size():
                    return (None, None)
            else:
                new_reads.extend(op.get_reads())
        if reads == new_reads:
            return (size, reduction_size)
        reads = new_reads
    return (size, reduction_size)
```

若 reduction_numel 只有 8192，inner helper 的第二个 easy condition直接返回 1。

## 多输出先判断是否已经足够并行

假定仍为 80 SM：`num_sm×2×32=5120`。外层输出数达到 5120 时，主函数返回 `(DEFAULT,1)`，不再分析读索引；已有足够多输出任务可占用设备。

规约项数不超过 32 也直接不拆。只有“外层输出还不够多且每项规约不短”才值得构造临时 Reduction 分析内外方向。

## 各个读取索引怎样共同决定 inner/outer

临时 ComputedBuffer 让 `get_read_writes()` 产生真实读取索引。只保留含全部 range variables 的读取；若决定输入 buffer layout 后 stride 改变，再采集一次。

对某读取简化后，`stride_hints(index,reduction_vars,...)` 给出组内变量改变 1 时物理索引怎样变化：

- 有任一组内 stride ≤1：记为 inner；
- 所有组内 stride >1：记为 outer；
- `num_inner > num_outer` 才选 INNER；相等也选 OUTER。

例如二维连续 `[M,N]` 沿 N 规约，r 每加 1 物理位置通常加 1，投 inner。转置排列中同一逻辑规约维可能跨较大 stride，投 outer。这里最终采用实际索引表达式，而不是仅凭 dim 编号猜测。

代入 `out[row] = x[row,r] + y[row,0]`，`get_read_writes()` 在符号执行 inner_fn 时记录两项 MemoryDep：连续 x 的 index 可写成 `row*N+r`，广播 y 的 index 是 `row`。本次循环变量为 `{row,r}`；x 的表达式含两者，进入投票，且 r 的 stride 是 1，所以投 INNER。y 的表达式缺 r，被“必须包含全部 range_vars”的条件过滤。若另一项读取是转置 storage 上的 `r*M+row`，r stride 为 M；它投 OUTER。一个 inner、一个 outer 时平票，最终选择 OUTER。

若所有输入都是广播访问，没有一项读取包含全部循环变量，indices 为空，当前实现返回 `(DEFAULT,1)`；产品 TODO 明确这类 split 尚未决定。

## outer 公式示例

仍假定 `num_sm=80`，输出数 64、规约项数 4096，且读索引投票 OUTER。总元素 262144 小于最小设备阈值，所以 outer helper 取 `split_size=32`，每线程处理 4 项：

```text
split=ceil(4096 / (4×32))=32
```

create 随后把 split=32 交给 K03-4、K03-5 展开的 `create_multilayer`。这里的 32 是把上述假定代入公式得到的结果；换设备属性、layout 投票或配置，split 会随之改变。

## 中等与超大工作量怎样偏向整除

inner/outer 两个 helper 在中等规模区间都先算临时 split_size，再查看 `sympy.divisors(reduction_numel_hint)`：足够接近时选实际因数，减少尾部；否则保留临时值。超大规模则寻找接近 `max_elements_per_thread=512` 的因数，在距离小于 50 时偏好均匀切分。

这解释了 split_size 不总是固定 32 或 512；它受总工作量、设备规模、输出数量和规约项数的因数共同影响。

## split 决策边界对照

1. argmax 有 1,000,000 项：`should_split` 因类型失败，仍返回 DEFAULT,1。
2. reduction_numel 是 unbacked SymInt：静态检查失败，不 split。
3. numel_hint=200、num_sm=80：已达到 `2×num_sm`，inner helper直接返回 1。
4. 有两项读取，一项 inner、一项 outer：平票选择 OUTER。
5. 决定输入 layout 后索引改变：必须重新调用 get_read_indices，再投票。
6. 所有读取均广播、indices 为空：返回 DEFAULT,1，不调用 inner/outer 公式。

## K03-3：Reduction.create 先消除退化规约，再选择单层或多层

`Reduction.create` 先计算 `reduction_numel=product(reduction_ranges)`。0 项规约有数学单位元；1 项规约无需组合；很小的静态规约可把所有组内位置展开进 Pointwise。只有这些条件都不成立，才调用刚读过的 K03-2 `num_splits` 决定单层或多层 Reduction。

### 完整源码：把小静态规约展开成 Pointwise

```python
@staticmethod
def _unroll_reduction_fn(inner_fn, reduction_ranges, reduction_type, src_dtype):
    """Convert inner_fn from a reduction to an pointwise"""
    # 教学：这条路径只接受可静态求值的规约范围，先变成 Python 整数。
    reduction_ranges = [
        V.graph.sizevars.evaluate_static_shape(x) for x in reduction_ranges
    ]

    combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

    def fn(index):
        # 教学：笛卡尔积枚举每一组 rindex，再用 K03-1 的组合函数折叠。
        return functools.reduce(
            combine_fn,
            (
                value_fn(index, rindex)
                for rindex in itertools.product(
                    *[range(x) for x in reduction_ranges]
                )
            ),
        )

    if reduction_type in ("argmin", "argmax"):
        # 教学：arg reduction 除值外还要把多维 rindex 变成连续扁平索引。
        flatten_index = FixedLayout(
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            reduction_ranges,
            FlexibleLayout.contiguous_strides(reduction_ranges),
        ).make_indexer()

        def value_fn(index, rindex):
            rindex = [sympy.expand(i) for i in rindex]
            return (
                inner_fn(index, rindex),
                ops.index_expr(flatten_index(rindex), torch.int64),
            )

        # 教学：组合内部携带 (value,index)，最终 Pointwise 只返回获胜 index。
        return lambda index: fn(index)[1]
    else:
        value_fn = inner_fn
        return fn
```

### 完整源码：Reduction.create 可能返回哪些对象

```python
@classmethod
def create(  # type: ignore[override]
    cls,
    device: torch.device,
    dst_dtype: torch.dtype,
    src_dtype: torch.dtype,
    inner_fn: Callable[..., Any],
    ranges: List[Expr],
    reduction_ranges: List[Expr],
    reduction_type: str,
    reduction_hint: ReductionHint = ReductionHint.DEFAULT,
    input_node: Optional[IRNode] = None,
):
    # 教学：先把多维规约范围乘成每个输出要组合的总项数。
    reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

    if reduction_numel == 0:
        # N.B. This is a hack to generate the literal of the given type
        # Ideally, we should be fixing `def constant` in triton.py
        # but it breaks due to hardcoded dtypes in other places
        def py_cnst(val):
            # 教学：单位元先转换为目标 dtype 对应的 Python bool/float/int。
            return (
                bool(val)
                if dst_dtype == torch.bool
                else float(val)
                if dst_dtype.is_floating_point
                else int(val)
            )

        rtypes_to_inits = {
            "sum": py_cnst(0),
            "xor_sum": py_cnst(0),
            "prod": py_cnst(1),
            "any": py_cnst(0),
            # "all" is desugared to `!any(!val)`
        }

        # 教学：max/min/arg 等没有在此定义空集合结果，会明确失败。
        assert (
            reduction_type in rtypes_to_inits.keys()
        ), f"{reduction_type} not supported for zero-dimension tensors!"

        def const_fn(index):
            return ops.constant(rtypes_to_inits[reduction_type], dst_dtype)

        # 教学：每个外层输出位置写同一单位元，因此退化成 Pointwise。
        return Pointwise.create(
            device=device,
            dtype=src_dtype,
            inner_fn=const_fn,
            ranges=list(ranges),
        )

    if reduction_numel == 1:
        # this reduction is actually a pointwise op
        if reduction_type in ("argmin", "argmax"):

            def fn(index):
                # 教学：唯一候选的扁平索引必为 0，不需读取值来选择。
                return ops.constant(0, dst_dtype)

        else:

            def fn(index):
                # 教学：每个规约维唯一合法位置都是 0，直接调用原 inner_fn。
                reduction_index = [sympy.Integer(0) for _ in reduction_ranges]
                return inner_fn(index, reduction_index)

        return Pointwise.create(device, dst_dtype, fn, ranges)

    if (
        # 教学：只有具体静态项数、小于阈值、且输出不止一个元素时才展开。
        isinstance(reduction_numel, sympy.Integer)
        and V.graph.sizevars.size_hint(reduction_numel)
        < config.unroll_reductions_threshold
        and sympy_product(ranges) != 1
    ):
        return Pointwise.create(
            device,
            dst_dtype,
            cls._unroll_reduction_fn(
                inner_fn, reduction_ranges, reduction_type, src_dtype
            ),
            ranges,
        )

    # triton doesn't support reduce to single element well, so break it up
        # 教学：其余情况由 num_splits 根据设备、范围与读索引选择 hint 和 split。
    hint, split = cls.num_splits(
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_numel,
        input_node,
    )
    # intermediate reduction in split can contain complex indexing,
    # and num_splits will fail to correctly set the hint
    # reuse the passed hint if available
    # 教学：调用方已有非 DEFAULT hint 时保留它，否则采用启发式结果。
    if reduction_hint == ReductionHint.DEFAULT:
        reduction_hint = hint
    if split == -1:
        # 教学：-1 表示复用输入链中已有 Reduction ranges 组织多层规约。
        assert input_node is not None
        new_ranges, new_reduction_ranges = extract_input_node_reduction_ranges(
            input_node  # type: ignore[arg-type]
        )
        assert new_ranges is not None
        assert new_reduction_ranges is not None
        return cls.create_multilayer_existing_ranges(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            new_ranges,
            new_reduction_ranges,
            reduction_type,
            reduction_hint,
        )
    elif split > 1:
        # triton doesn't support reduce to single element well, so break it up
        # 教学：先产生 split 份中间结果，再由第二层 Reduction 合并。
        return cls.create_multilayer(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            split,
            reduction_hint,
        )

    # 教学：split 为 1 的普通路径保留真正 Reduction，并包装成 TensorBox。
    return TensorBox.create(
        Reduction(
            device,
            dst_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            src_dtype,
            reduction_hint,
        )
    )
```

## 四个具体 reduction_numel

### 0 项：返回单位元 Pointwise

设输入 shape `[2,0]`、沿 dim=1 求和：`ranges=[2]`、`reduction_ranges=[0]`、numel=0。每个 row 没有输入值可加，因此 `const_fn` 在两个外层位置都返回 0，结果逻辑值 `[0,0]`。

prod 的单位元为 1，any 为 false，xor_sum 为 0。若这里请求 max，字典没有条目，断言失败；当前代码不在此分支自行定义空 max 的结果。

这里有一个容易忽略的当前源码细节：`const_fn` 用 `dst_dtype` 生成常量，但包装这个 Pointwise 时传入的字段是 `dtype=src_dtype`。因此本节只能准确地说“常量表达式按目标 dtype 产生，而 Pointwise 的 dtype 字段取输入值 dtype”；不能把两者概括成同一个“目标 dtype”。当 `src_dtype == dst_dtype` 时二者没有可见差异，二者不同时要按这两个字段分别追踪。

当前 lowering 支持走到这个空规约分支的 sum/prod/any 等调用，会在构造前先完成所需 dtype 转换：例如 int32 的 sum 默认先把输入转成 int64，于是传入这里的 `src_dtype` 与 `dst_dtype` 都是 int64；any 的两者都是 bool。因此这些可达路径中，Pointwise 字段、常量表达式和最终 buffer dtype 一致。上面的两种参数写法仍是 `Reduction.create` 泛化接口的源码事实，但不能据此虚构一个当前 lowering 会产生的 dtype 冲突案例。

### 1 项：把唯一 rindex 固定为 0

输入 shape `[2,1]`、dim=1 sum：numel=1。普通 sum 的 Pointwise fn 对 row=1 调用 `inner_fn([1],[0])`，直接读 x[1,0]；没有 combine 步骤。

argmax/argmin 更简单：唯一候选的索引恒为 0，fn 直接产生 int64/目标 dtype 的 0，不读取候选值来比较。

### 3 项：满足阈值时静态展开

输入 shape `[2,3]`、dim=1 sum，若当前 `unroll_reductions_threshold>3`：输出 numel 为 2，不等于 1，所以进入展开。Pointwise fn 在 row=1 内部枚举：

```text
rindex=(0) → inner_fn([1],[0]) = 3
rindex=(1) → inner_fn([1],[1]) = 4
rindex=(2) → inner_fn([1],[2]) = 5
functools.reduce(add,[3,4,5]) = 12
```

这就是“函数名是 sum，但实际 IR 是 Pointwise”的具体来源。若 ranges 的乘积等于 1，即最终只产生一个标量输出，这个小静态展开条件故意不进入，继续让 num_splits/Reduction 路径决定。

argmax 对多维 reduction_ranges `[2,3]` 会用连续 stride `[3,1]` 把 rindex `[1,2]` 扁平为 `1×3+2=5`，组合结束只交回获胜 index。

### 128 项：进入 split 决策

输入 shape `[2,128]`、dim=1 sum 通常不满足小静态展开条件，于是调用 `num_splits`。它可能返回：

| split | create 出口 |
| ---: | --- |
| `-1` | 从 input_node 提取已有 ranges，走 `create_multilayer_existing_ranges` |
| `>1` | 走 `create_multilayer`，建立中间规约与第二层规约 |
| `1` | 直接 `TensorBox(Reduction(...))` |

在当前 CPU 实验中，2×128 最终是单层 Reduction。换到另一后端或配置后，仍需把 K03-2 中的后端能力、静态规模、输出数量和读取 stride 代入，才能判断它是单层还是多层；“128 项”本身没有唯一出口。

## hint 的覆盖规则

`num_splits` 同时返回 hint。只有传入 hint 为 DEFAULT 时才采用计算值；调用方显式提供 INNER/OUTER 等值时保留。原因写在产品注释：中间 split reduction 可能有复杂 indexing，使再次计算的 hint 不可靠。

hint 与 split 是不同决定：hint 描述后端倾向怎样组织规约；split 决定是否创建多层 IR。

## create 出口边界对照

1. reduction_numel=0、type=prod、ranges=[3]：返回三个单位元 1 的 Pointwise。
2. reduction_numel=1、type=argmin：每个外层输出均为索引 0。
3. reduction_numel=3 但 ranges product=1：不走静态展开，进入 num_splits。
4. reduction_numel 是无法静态确定的 SymPy 表达式：不满足 `isinstance(...,sympy.Integer)`，进入 num_splits。
5. num_splits 返回 -1 但 input_node=None：在提取范围前断言失败。
6. 调用方传 OUTER、num_splits 建议 INNER：create 保留 OUTER。

## K03-4：第一层分块，并在尾块越界时加入遮罩

`num_splits` 只交回拆分方向和 `split` 数。真正建立第一层坐标前还要解决三个问题：空位置用什么单位元；第二层是否改用更小的 outer hint；原 loader 怎样从“块编号 + 块内编号”还原成原规约坐标。

### 完整源码：单位元与第二层 hint

```python
@staticmethod
def default_accumulator(reduction_type, dtype):
    # 教学：max/argmax 的空位置从该 dtype 的最小值开始。
    if reduction_type in ("max", "argmax"):
        if is_float_dtype(dtype):
            return float("-inf")
        elif is_boolean_dtype(dtype):
            return 0
        else:
            return torch.iinfo(dtype).min
    # 教学：min/argmin 相反，从该 dtype 的最大值开始。
    if reduction_type in ("min", "argmin"):
        if is_float_dtype(dtype):
            return float("inf")
        elif is_boolean_dtype(dtype):
            return 1
        else:
            return torch.iinfo(dtype).max

    return {
        "sum": 0,
        "prod": 1,
        "xor_sum": 0,
        "any": 0,
        "welford_reduce": (0, 0, 0),
        "welford_combine": (0, 0, 0),
    }[reduction_type]

@staticmethod
def default_value(reduction_type, dtype):
    # 教学：welford_reduce 的单项 loader 返回标量；组合状态的三元单位元在 accumulator 中。
    if reduction_type == "welford_reduce":
        return 0
    return Reduction.default_accumulator(reduction_type, dtype)

@staticmethod
def _multilayer_second_step_hint(
    split: int, numel_hint: int, reduction_hint: ReductionHint
) -> ReductionHint:
    # 教学：existing-ranges 路径用 -1 表示不按普通 split 改 hint。
    if split == -1:
        return reduction_hint
    # 教学：outer 的第二层规模较小时，标成 OUTER_TINY 供后端采用对应策略。
    if split <= 512 and numel_hint <= 512 and reduction_hint == ReductionHint.OUTER:
        return ReductionHint.OUTER_TINY
    if (
        split <= 1024
        and numel_hint <= 256
        and reduction_hint == ReductionHint.OUTER
    ):
        return ReductionHint.OUTER_TINY

    return reduction_hint
```

这些默认值必须是对应组合的单位元：sum 的尾部补 0 不改变和，prod 补 1 不改变积，max 补负无穷不会胜过任何有限输入。它们不是最终输出的初始化猜测，而是第一层最后一个不满块中“越过原规约范围”的虚拟输入。

### 完整源码：从块坐标还原原规约坐标

```python
@classmethod
def _multilayer_wrap_loader(
    cls,
    loader,
    reduction_ranges,
    reduction_numel,
    split,
    block_size,
    default,
):
    # 教学：先把原多维规约范围视为一条长度 reduction_numel 的线性范围。
    reindex = View.dynamic_reshape_indexer(reduction_ranges, [reduction_numel])
    need_mask = not V.graph.sizevars.is_expr_static_and_true(
        sympy.Eq(reduction_numel % split, 0)  # type: ignore[arg-type]
    )

    def wrapper_fn(index, reduction_index):
        # 教学：第一层只有一个块内规约坐标。
        (reduction_index,) = reduction_index
        # 教学：index 尾项是块编号；前缀仍是原输出坐标。
        *new_index, reduction_block = index
        indices = block_size * reduction_block + reduction_index

        def body():
            # 教学：线性 indices 再映射回原 reduction_ranges 的多维坐标。
            return loader(new_index, reindex([indices]))

        if need_mask:
            # 教学：尾块越界位置不调用原 loader，而交回该 reduction 的单位元。
            mask = ops.lt(
                ops.index_expr(indices, torch.int32),
                ops.index_expr(reduction_numel, torch.int32),
            )
            return ops.masked(mask, body, default)
        else:
            return body()

    return wrapper_fn
```

### 1000 项、split=32 的坐标推演

普通 multilayer 路径会计算：

```text
reduction_numel = 1000
split           = 32
block_size      = ceil(1000/32) = 32
第一层输出范围   = [*ranges, 32]     # 32 个块
第一层规约范围   = [32]              # 每块最多 32 项
```

若原输出坐标是 `[row]`，第一层 `wrapper_fn` 收到 `index=[row,reduction_block]` 和 `reduction_index=[r]`。例如块 2、块内位置 5：

```text
indices = 32*2+5 = 69
loader([row], reindex([69]))
```

若原规约 shape 是 `[10,100]`，线性 69 会由 `dynamic_reshape_indexer` 还原为 `[0,69]`；线性 237 则还原为 `[2,37]`。因此拆层改变循环的分组，不改变原 loader 看见的逻辑元素。

最后一块编号 31 对应线性位置 `992..1023`。其中 `992..999` 的 mask 为真，读取原输入；`1000..1023` 为假，sum 返回默认值 0。第二层随后组合 32 份第一层结果，虚拟的 0 不改变最终和。若 reduction_numel 能被 split 整除，`need_mask=False`，每一项都直接调用 body。

`_multilayer_second_step_hint(32, 128, OUTER)` 返回 `OUTER_TINY`；同样规模但 hint 为 INNER 时原样返回 INNER。这个 helper 改的是第二层实现提示，不改变上述坐标、mask 或数学组合。

### 第一层分块边界对照

1. 1001 项、split=32 时 block_size 仍为 32；最后一块只有线性 992..1000 共 9 项有效，其余 23 项取单位元。
2. reduction_type 为 prod 时，越界位置必须取 1；若错误取 0，任何不满尾块都会把局部积变成 0。
3. 原规约范围 `[4,5]` 的线性 13 会还原为 `[2,3]`，所以原 loader 不需要知道 split。
4. split=-1 时第二层 hint 不改写，这是 existing-ranges 路径的约定，不表示创建负数个块。

下一子点 K03-5 展开 `create_multilayer_helper`、`create_multilayer` 和 existing-ranges 变体，把本节 wrapper 产生的第一层 Reduction 物化，再连接成第二层 Reduction。

## K03-5：物化第一层结果，再让第二层规约读取它

K03-4 已把原来的长规约改写成“split 个块，每块内部规约 block_size 项”。这还只是第一层 loader。接下来必须把第一层 Reduction 登记成中间 buffer，并让它的 loader 读取该 buffer；第二层才能把这 split 个局部结果当作输入，再组合成原输出。

### 完整源码：已有 ranges 的坐标包装

```python
@classmethod
def _multilayer_wrap_loader_existing_ranges(
    cls,
    loader,
    original_ranges,
    original_reduction_ranges,
    new_ranges,
    new_reduction_ranges,
    default,
):
    # 教学：这条路径只为原输出 numel 为 1 的规约启用。
    assert all(
        r == 1 for r in original_ranges
    ), f"Only enabled for numel_hint == 1, found {original_ranges=}"
    # 教学：把新外层与新规约坐标合起来，还原成原规约坐标。
    reindex = View.dynamic_reshape_indexer(
        original_reduction_ranges, tuple(new_ranges) + tuple(new_reduction_ranges)
    )

    def wrapper_fn(merged_index, new_reduction_index):
        # 教学：merged_index 的前缀保留原输出坐标，后缀是新外层坐标。
        original_idx = merged_index[: len(original_ranges)]
        new_index = merged_index[len(original_ranges) :]
        return loader(
            original_idx,
            reindex(tuple(new_index) + tuple(new_reduction_index)),
        )

    return wrapper_fn
```

这里没有普通 split 路径的尾块 mask。`new_ranges × new_reduction_ranges` 是 `num_splits` 已选出的完整重排，坐标总数与原 `original_reduction_ranges` 一致。参数 `default` 为两条 wrapper 保持统一接口，但当前函数体并不使用它。

### 完整源码：共同的两层连接器

```python
@classmethod
def create_multilayer_helper(
    cls,
    device: torch.device,
    dst_dtype: torch.dtype,
    src_dtype: torch.dtype,
    wrapper_fn: Callable[..., Any],
    original_ranges: List[Expr],
    original_reduction_ranges: List[Expr],
    new_ranges: List[Expr],
    new_reduction_ranges: List[Expr],
    reduction_type: str,
    split: int,
    reduction_hint: ReductionHint,
):
    """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
    # triton will automatically compute reductions in fp32 if reducing over fp16/bf16
    # within the kernel. keep the intermediate in fp32 so as to keep the whole reduction
    # in fp32 and not reduce precision by breaking up the kernel into multiple layers
    # 教学：半精度输入的局部结果用 fp32 保存，避免两层之间先舍入回半精度。
    intermediate_dtype = (
        dst_dtype
        if dst_dtype not in (torch.float16, torch.bfloat16)
        else torch.float
    )
    # 教学：第一层的外层范围是 new_ranges，块内规约范围是 new_reduction_ranges。
    intermediate = Reduction.create(
        device,
        intermediate_dtype,
        src_dtype,
        wrapper_fn,
        new_ranges,
        new_reduction_ranges,
        reduction_type,
        reduction_hint,
    )
    # 教学：第二层需要稳定读取局部结果，所以第一层在这里物化并取得 loader。
    intermediate.realize()
    intermediate_loader = intermediate.make_loader()

    def intermediate_fn(index, reduction_index):
        # 教学：第二层外层坐标与 reduction 坐标拼成第一层 buffer 的完整坐标。
        return intermediate_loader([*index, *reduction_index])

    numel_hint = V.graph.sizevars.size_hint(sympy_product(original_ranges))
    reduction_hint = cls._multilayer_second_step_hint(
        split, numel_hint, reduction_hint
    )

    # 教学：第一层范围必须仍以前缀形式保留原输出坐标。
    assert original_ranges == new_ranges[: len(original_ranges)]
    return TensorBox.create(
        Reduction(
            device,
            dst_dtype,
            intermediate_fn,
            original_ranges,
            new_ranges[len(original_ranges) :],
            reduction_type,
            src_dtype,
            reduction_hint,
        )
    )
```

第一层调用 `Reduction.create` 而不是直接构造 `Reduction`，因此它仍可应用 0/1/小静态等已讲过的退化规则；随后 `realize()` 保证第二层读取的是具名中间 buffer。第二层则直接构造 `Reduction`，外层恢复为 `original_ranges`，规约范围取第一层 `new_ranges` 去掉原输出前缀后的部分。

### 完整源码：普通 split 与已有 ranges 分别传入什么

```python
@classmethod
def create_multilayer(
    cls,
    device: torch.device,
    dst_dtype: torch.dtype,
    src_dtype: torch.dtype,
    inner_fn: Callable[..., Any],
    ranges: List[Expr],
    reduction_ranges: List[Expr],
    reduction_type: str,
    split: int,
    reduction_hint: ReductionHint,
):
    """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
    # TODO(jansel): realize the reduction so we can do dynamic indexing
    reduction_numel = sympy_product(reduction_ranges)
    # 教学：向上取整，让 split 个块足以覆盖全部规约位置。
    block_size = FloorDiv(reduction_numel + (split - 1), split)
    default = cls.default_value(reduction_type, dst_dtype)
    wrapper_fn = cls._multilayer_wrap_loader(
        inner_fn, reduction_ranges, reduction_numel, split, block_size, default
    )

    return cls.create_multilayer_helper(
        device,
        dst_dtype,
        src_dtype,
        wrapper_fn,
        ranges,
        reduction_ranges,
        [*ranges, split],  # type: ignore[list-item]
        [block_size],
        reduction_type,
        split,
        reduction_hint,
    )

@classmethod
def create_multilayer_existing_ranges(
    cls,
    device: torch.device,
    dst_dtype: torch.dtype,
    src_dtype: torch.dtype,
    inner_fn: Callable[..., Any],
    original_ranges: List[Expr],
    original_reduction_ranges: List[Expr],
    new_ranges: List[Expr],
    new_reduction_ranges: List[Expr],
    reduction_type: str,
    reduction_hint: ReductionHint,
):
    """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
    default = cls.default_value(reduction_type, dst_dtype)
    wrapper_fn = cls._multilayer_wrap_loader_existing_ranges(
        inner_fn,
        original_ranges,
        original_reduction_ranges,
        new_ranges,
        new_reduction_ranges,
        default,
    )
    return cls.create_multilayer_helper(
        device,
        dst_dtype,
        src_dtype,
        wrapper_fn,
        original_ranges,
        original_reduction_ranges,
        [*original_ranges, *new_ranges],
        new_reduction_ranges,
        reduction_type,
        -1,
        reduction_hint,
    )
```

### 1000 项案例怎样穿过两层

沿用 K03-4 的 `ranges=[4]`、`reduction_ranges=[1000]`、`split=32`：

| 阶段 | 外层 ranges | reduction ranges | 一个位置表达什么 |
| --- | --- | --- | --- |
| 原规约 | `[4]` | `[1000]` | 每个 row 组合 1000 项 |
| 第一层 | `[4,32]` | `[32]` | 每个 `(row,block)` 组合最多 32 项 |
| 中间 buffer | `[4,32]` | 无 | 保存 4×32 个局部结果 |
| 第二层 | `[4]` | `[32]` | 每个 row 组合 32 个局部结果 |
| 最终 buffer | `[4]` | 无 | K03-5 返回最终 Reduction 后，K02 的 `make_reduction` 识别它并调用 `result.realize()` |

以 row=2 为例，第一层产生 `intermediate[2,0]` 到 `intermediate[2,31]`。第二层的 `index=[2]`、`reduction_index=[7]` 被 `intermediate_fn` 拼成 `[2,7]`，读取第 7 个局部结果。组合完 32 项后，第二层只输出 row=2 的最终值。

若 `dst_dtype=float16`，三个 dtype 在源码对象中的位置如下：

| 对象 | `dtype` 字段 | `src_dtype` 字段或 loader 实际值 |
| --- | --- | --- |
| 第一层 Reduction | `intermediate_dtype=float32` | 原 `src_dtype` |
| 中间 buffer loader | buffer 为 float32 | 读出第一层的 float32 局部结果 |
| 第二层 Reduction | 最终 `dst_dtype=float16` | 构造器仍保存原 `src_dtype` |

也就是说，第一层局部累计被保存在 float32 buffer 中，第二层 loader 读取的是该 buffer 的值；最终 Reduction 的目标字段回到 float16。源码传给第二层 `Reduction` 的 `src_dtype` 参数仍是原值，不能把这个字段改称为 `intermediate_dtype`。这套安排避免在两层之间先把局部结果截回低精度。

以 Triton 后端为例，reduction codegen 用 `src_dtype` 调 `triton_acc_type(src_dtype)` 选择累加器。当前函数对有符号整数选 int32/int64，其余类型转给 `triton_compute_type`；后者只在 `config.triton.codegen_upcast_to_fp32` 开启时把 float16/bfloat16 映射为 float32：

```python
def triton_acc_type(dtype):
    if is_integer_dtype(dtype) and dtype.is_signed:
        nbits = 64 if dtype == torch.int64 else 32
        return f"tl.int{nbits}"
    return triton_compute_type(dtype)

# triton_compute_type 的相关分支
elif (
    triton_type_name in ("float16", "bfloat16")
    and config.triton.codegen_upcast_to_fp32
):
    triton_type_name = "float32"
```

所以在该配置开启的 Triton 路径，第二层的旧低精度 `src_dtype` 会选择 fp32 累加器，与 fp32 中间值相配；不能把这个结论无条件推广到所有配置和后端。本篇构造出口能确定的是：`ops.reduction` 收到最终 dtype、原 `src_dtype`、type，以及来自 fp32 中间 loader 的 value；具体后端按自己的 handler 解释这组参数。

existing-ranges 路径假设 `original_ranges` 各项均为 1。例如原规约 `[2,3,4]` 共 24 项，启发式已经选出 `new_ranges=[2]`、`new_reduction_ranges=[3,4]`：第一层外层为 `[1,2]`，规约 `[3,4]`；第二层恢复外层 `[1]`，再规约新增的 `[2]`。

代入一个位置：第一层收到 `merged_index=[0,1]` 与 `new_reduction_index=[2,3]`。wrapper 先切出原输出坐标 `[0]` 和新增外层坐标 `[1]`，再把 `(1,2,3)` 交给 reshape indexer。因为新旧规约 shape 都是 `[2,3,4]`，它还原出的原规约坐标仍是 `[1,2,3]`，最终调用 `loader([0],[1,2,3])`。于是 block=1 这项局部结果覆盖原规约第一维坐标 1。这里 `split=-1` 只让第二层保留原 hint，不表示负数个块。

### 两层连接边界对照

1. `ranges=[2,3]`、split=8 时，第一层 `new_ranges=[2,3,8]`；第二层必须恢复外层 `[2,3]`，只规约尾部 `[8]`。
2. 若误删 `intermediate.realize()`，第二层 loader 就没有稳定的中间存储边界，违背“两层 reduction”需要先保存局部结果的合同。
3. float32 输出不会提升 intermediate dtype，因为它已经不是 float16/bfloat16；float16 输出会提升到 float32。
4. existing-ranges 的 `original_ranges=[2]` 会立即触发断言，因为这条优化只为原输出 numel 为 1 的情形设计。

K03 至此把 split 决策、第一层坐标/mask 和两层连接完整闭合。下一知识点 K04 转向 View：先走固定 layout 可直接重解释的快速路径，再走需要 reindex 的延迟路径。

## K04a：先选固定 layout 出口或延迟坐标出口

View 对象本身不执行元素搬运；它回答“新逻辑位置应读取旧 storage 的哪里”。最直接的情况是 permute：只需按维度顺序重排 size 与 stride。reshape 先处理 `-1` 和元素总数：已有连续 storage 时直接建立新连续 layout；含 unbacked size 而输入尚未连续时，先通过 View 之外的物化/必要复制路径取得可冻结的 storage，再建立 ReinterpretView；其余静态延迟对象保存坐标函数，留给 K04b。

本节会出现两种出口。`ReinterpretView` 持有 storage 和新的固定 layout，新位置可直接用新 stride 算地址。普通 `View`（它继承 `GenericView`）则持有旧数据、new size 和一条 reindex 函数；它的 loader 收到新坐标后，先算出旧坐标，再调用旧对象的 loader。两者都共享原结果，不复制元素。

这条“先换坐标，再读旧对象”的连接在 `BaseView.make_loader` 中完成；`GenericView.make_reindexer` 只需交回构造时保存的函数：

```python
def make_loader(self):
    inner = self.data.make_loader()
    reindex = self.make_reindexer()

    def loader(idx):
        return inner(reindex(idx))

    return loader


def make_reindexer(self):
    return self.reindex
```

例如 `View(data=x, size=[3,2], reindex=f)` 的 loader 收到新坐标 `[2,1]`，先算 `f([2,1])=[1,2]`，再执行 `x.make_loader()([1,2])`。K04b 的全部公式最终都接到这里，而不是孤立地保存一条无人调用的算式。

### 完整源码：PermuteView 的 layout 与坐标两条路径

```python
@dataclasses.dataclass
class PermuteView(BaseView):
    dims: List[Expr]

    @classmethod
    def create(cls, x, dims):
        # 教学：先把 -1 等负维编号映射为非负编号，并确认它是完整排列。
        dims = cls._map_neg_dims(dims)
        assert OrderedSet(dims) == OrderedSet(range(len(dims)))

        if is_storage_and_layout(x):
            # 教学：已有 storage/layout 时无需保留 PermuteView 对象；重排 size/stride 即可。
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                [old_layout.size[i] for i in dims],
                [old_layout.stride[i] for i in dims],
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        # 教学：延迟计算还没有固定 layout 时，保存 dims，读取时再做坐标逆置换。
        return PermuteView(x, dims)

    @classmethod
    def _map_neg_dims(cls, dims):
        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self):
        assert OrderedSet(self._map_neg_dims(self.dims)) == OrderedSet(
            range(len(self.dims))
        )
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def make_reindexer(self):
        # 教学：dims 描述新维来自哪个旧维；读取旧对象时需要它的逆排列。
        inv = {j: i for i, j in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]  # type: ignore[index]
        assert OrderedSet(inv) == OrderedSet(range(len(self.dims)))

        def reindex(index):
            return [index[i] for i in inv]

        return reindex
```

对旧 shape `[2,3]`、stride `[3,1]` 做 `dims=[1,0]`：新 size 为 `[3,2]`，新 stride 为 `[1,3]`。新位置 `[2,1]` 的 storage offset 是 `2×1+1×3=5`，正是旧位置 `[1,2]` 的 `1×3+2×1=5`。实验 05 的完整编译实际得到 `ReinterpretView(size=[3,2], stride=[1,3])`，底层仍是同一个 `InputBuffer`，没有新增 buffer/operation。

若 x 还是延迟 Pointwise，`create` 返回 `PermuteView(x,[1,0])`。其逆排列仍为 `[1,0]`：新 loader 收到 `[new0,new1]` 后，用 `[new1,new0]` 读取旧 Pointwise。两条路径表达相同位置关系，只是前者已有固定 stride，后者只有计算规则。

### 完整源码：reshape 先规范化大小，再选择返回对象

```python
@dataclasses.dataclass
class View(GenericView):
    @staticmethod
    def handle_negative_index(idx, size):
        idx = sympy.expand(idx)
        size = sympy.expand(size)
        evaluate_expr = V.graph.sizevars.shape_env.evaluate_expr
        if evaluate_expr(sympy.Lt(idx, 0)):
            idx = idx + size
        return idx

    @classmethod
    def create(cls, x, new_size):
        assert isinstance(new_size, (tuple, list))
        old_size, new_size = cls.resolve_negative_size(x.get_size(), new_size)

        # Skip pointless views
        # 教学：形状静态相同，直接复用原对象，连 view 包装都不增加。
        if V.graph.sizevars.statically_known_list_equals(old_size, new_size):
            return x

        unbacked_symbols_in_sizes = False
        if (
            len(free_unbacked_symbols(old_size)) > 0
            or len(free_unbacked_symbols(new_size)) > 0
        ):
            unbacked_symbols_in_sizes = True

        if 0 in new_size:

            def fake_reindex(index):
                return tuple([0] * len(old_size))

            # 教学：空结果没有合法元素会调用 loader；占位 reindex 只满足接口。
            return cls(x, list(new_size), fake_reindex)
        # TODO: a new class for FixedTransferLayout that output layout is constrained by input layout
        elif is_contiguous_storage_and_layout(x) or unbacked_symbols_in_sizes:
            if unbacked_symbols_in_sizes and (not is_contiguous_storage_and_layout(x)):
                # realize x; otherwise, the dynamic_reshape_indexer below will fail
                # due to the size_hint's inability to process unbacked SymInts
                # 教学：动态且尚无连续 storage 时先物化，避开依赖 size_hint 的坐标算法。
                x = ExternKernel.realize_input(x)

            storage, old_layout = as_contiguous_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                FlexibleLayout.contiguous_strides(new_size),
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        # 教学：非连续且没有 unbacked size 时，建立从新坐标到旧坐标的延迟函数。
        reindex = cls.dynamic_reshape_indexer(old_size, new_size)
        return cls(x, list(new_size), reindex)

    @staticmethod
    def resolve_negative_size(old_size, new_size):
        new_size = [V.graph.sizevars.simplify(x) for x in new_size]
        old_size = [V.graph.sizevars.simplify(x) for x in old_size]

        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.Integer(1)
                # 教学：缺失维 = 旧总元素数 / 其余新维元素数；只处理第一个 -1。
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break

        # 教学：无论是否有 -1，reshape 前后总元素数必须建立相等 guard。
        V.graph.sizevars.guard_equals(sympy_product(old_size), sympy_product(new_size))
        return old_size, new_size
```

对连续 `[2,3]` reshape 为 `[3,2]`，元素总数都为 6，新连续 stride 是 `[2,1]`；实验 05 实际观察到 `ReinterpretView` 继续引用原 `InputBuffer`，buffer/operation 仍为 0。

对延迟 `(x+1)` 做同一 reshape，调用 `View.create` 时输入仍是 `StorageBox→Pointwise`，所以本函数返回 `View(size=[3,2])→Pointwise`。它的 loader 用上面的 `inner(reindex(idx))` 继续读取 Pointwise。若之后有消费者要求把这项延迟结果保存下来，物化会发生在 `View.create` 已经返回之后；因此本节的构造出口仍是 View，而不是 ReinterpretView。

`new_size=[3,-1]` 时，先临时把 -1 换成 1，得到其余维乘积 3，再计算 `6/3=2`，最终 `[3,2]`。若新 size 含 0，返回的 Generic View 没有合法输出位置；fake reindex 不代表所有位置都读取旧 `[0,...]`，因为没有任何位置会执行读取。

循环只替换遇到的第一个 `-1`。这个 helper 本身没有“第二个 `-1`”计数检查；若非法输入已经走到这里，第二个 `-1` 仍留在 `new_size` 中，元素数乘积甚至可能在两个负号相乘后通过相等 guard。合法性应由更早的算子校验保证，不能把这里的 `break` 解释成它会自行拒绝多个 `-1`。

动态尺寸走的是另一条具体路径。设旧 size 是 `[u0,4]`、新 size 是 `[2,u1]`，其中 `u0/u1` 是运行时才有数值的 unbacked SymInt。源码先发现 size 中有 unbacked symbol；如果 x 还只是延迟 Pointwise，就先把它物化成具有连续 layout 的中间结果。随后新建连续 stride `[u1,1]` 的 `ReinterpretView`，并保留元素数相等的约束 `u0×4 = 2×u1`。它不会把 `u0` 或 `u1` 交给后面的静态 `size_hint` reshape 算法。

这里“物化”与“连续”是相邻两步，不是同一保证。`ExternKernel.realize_input(x)` 先让 StorageBox 指向带 `FlexibleLayout` 的 ComputedBuffer；紧接着的 `as_contiguous_storage_and_layout(x)` 以 `want_contiguous=True` 冻结这份 FlexibleLayout，使旧 buffer 采用连续 stride，然后才返回 `(storage, old_layout)`。最后 `View.create` 为相同 storage 建立新 shape 的连续 stride。因而调用顺序是：

```text
延迟规则 → ComputedBuffer(FlexibleLayout)
         → 冻结旧 layout 为 contiguous
         → ReinterpretView(storage, 新连续 layout)
```

两个 helper 的相关分支如下。`realize_input` 会递归剥 TensorBox；ReinterpretView 保留自己的 layout 并处理底层 data；普通 BaseView 先物化，能换成 ReinterpretView 时换，不能时最终落入 `copy_input`；StorageBox 原地 realize。随后第二个 helper 把 `want_contiguous=True` 传入通用 layout helper。通用 helper 读到 `StorageBox(ComputedBuffer)` 后调用 `freeze_layout()`，并断言冻结结果连续：

```python
# ExternKernel.realize_input
if isinstance(x, TensorBox):
    return cls.realize_input(x.data)
if isinstance(x, ReinterpretView):
    return ReinterpretView(cls.realize_input(x.data), x.get_layout())
if isinstance(x, BaseView):
    x.realize()
    if is_storage_and_layout(x.unwrap_view()):
        try:
            return cls.convert_to_reinterpret_view(x)
        except NotImplementedError:
            pass
if isinstance(x, StorageBox):
    x.realize()
    return x
return cls.copy_input(x)


as_contiguous_storage_and_layout = functools.partial(
    as_storage_and_layout, want_contiguous=True
)


# as_storage_and_layout 中的 StorageBox 分支
if isinstance(x, StorageBox) and isinstance(x.data, Buffer):
    if freeze:
        if want_contiguous:
            x.data.freeze_layout()
            assert x.data.layout.is_contiguous()
        else:
            x.data.decide_layout()
    return x, x.data.layout
if isinstance(x, ReinterpretView):
    # 只把连续要求传到底层 storage；view 自己保留原 layout。
    buffer, _ = as_storage_and_layout(x.data, freeze=freeze)
    return buffer, x.layout
```

因此动态案例要按输入种类区分：延迟 Loops/普通 BaseView 可以物化为 FlexibleLayout 后冻结连续；无法转成 storage/layout 的其他对象会先 `copy_input`。已有 ReinterpretView 则走一个特殊分支：通用 helper 明确不把 `want_contiguous` 继续传给底层，并直接交回该 view 的 layout；这段 helper 自身没有再次断言 view layout 连续。正文的 `[u0,4]` 延迟 Pointwise 案例走第一种，能够推出连续出口；不能把该结论无条件推广到一个任意的非连续 ReinterpretView。

### View 出口边界对照

1. `[2,3,4]` 以 dims `[-1,0,1]` permute：负维变成 2，新 size `[4,2,3]`、stride `[1,12,4]`。
2. 连续 `[2,3,4]` reshape `[6,-1]`：缺失维为 4，新连续 stride `[4,1]`。
3. reshape 前后 shape 都是 `[2,3]`：静态相等分支直接返回 x，不建立 View。
4. 非连续延迟结果 reshape 时，不能靠重排现有固定 stride；它进入 K04b 的 `dynamic_reshape_indexer`，保存新坐标到旧坐标的算式。

下一子点 K04b 完整展开 `dynamic_reshape_indexer/_dynamic_reshape_indexer`：怎样合并或拆开相邻维、怎样用 ModularIndexing 还原旧坐标，以及乐观算法失败后为什么先展平再重建。

## K04b：把新坐标变回旧坐标，而不复制数据

非连续延迟结果没有一份可以直接改 stride 的固定 layout。`View` 因此保存 reindex 函数：未来代码生成遍历新 shape 时，把每个新坐标换算成旧 shape 的坐标，再调用旧 loader。算法从最后一维向前配对，遇到边界不一致时先合并相邻新维或拆开相邻旧维。

### 完整源码：乐观配对失败后先展平

```python
@classmethod
def dynamic_reshape_indexer(cls, old_size, new_size):
    try:
        reindex = cls._dynamic_reshape_indexer(old_size, new_size)
    except (AssertionError, IndexError):
        # optimistic algorithm failed, lets do a fallback
        # 教学：把 old→new 拆成 old←flat 与 flat←new 两次坐标变换，再组合函数。
        flat = [sympy_product(old_size)]
        reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
        reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
        reindex = fuse_reindexing(reindex1, reindex2)
    return reindex
```

直接算法优先保留 old/new 中已经对齐的连续维组；例如 `[2,3,4]→[6,4]` 的尾维 4 可以直接对应。`[2,3]→[3,2]` 的右侧边界不对齐：新末维 2 小于旧末维 3，尝试再并入新维 3 后得到 6，又越过旧维 3，乐观配对失败。fallback 改成两步：`[2,3]←[6]` 与 `[6]←[3,2]`，两步都能以完整线性范围为边界，再由 `fuse_reindexing` 合成最终函数。

### 完整源码：从末维开始合并和拆分

```python
@staticmethod
def _dynamic_reshape_indexer(old_size, new_size):
    """
        Perform a reshape entirely by modifying indexing math
        """
    size_hint = V.graph.sizevars.size_hint
    # TODO: These symbols may not escape, if they don't assert so and
    # treat them as temporary
    # 教学：每个新维先分配一个符号，最后再用调用者传入的具体 index 替换。
    vars = [
        sympy_index_symbol_with_prefix(SymT.VIEW, i) for i in range(len(new_size))
    ]

    stack_new = list(zip(vars, new_size))
    stack_old = list(old_size)

    view_expr = []
    while stack_new and stack_old:
        size_old = stack_old.pop()
        var, size_new = stack_new.pop()
        if size_old == 1:
            # 教学：旧 size-1 维坐标只能为 0；新维留到下一轮继续匹配。
            view_expr.append(sympy.Integer(0))
            stack_new.append((var, size_new))  # re-add
        elif size_new == 1:
            # 教学：新 size-1 维不消耗旧维；旧维放回继续匹配。
            stack_old.append(size_old)  # re-add
        elif size_hint(size_new) == size_hint(size_old):
            # 教学：维长相等，新坐标可直接作为对应旧坐标。
            view_expr.append(var)
            V.graph.sizevars.guard_equals(size_new, size_old)
        elif size_hint(size_new) < size_hint(size_old):
            # 教学：一个旧维需要多个相邻新维共同覆盖，按行主序合并新坐标。
            while size_hint(size_new) < size_hint(size_old):
                var2, size_new2 = stack_new.pop()
                var = var2 * size_new + var
                size_new = size_new * size_new2
            view_expr.append(var)
            V.graph.sizevars.guard_equals(size_new, size_old)
        elif size_hint(size_new) > size_hint(size_old):
            # 教学：一个新维覆盖多个旧维，用除法与取模依次拆出旧坐标。
            divisor = sympy.Integer(1)
            modulus = size_old
            view_expr.append(ModularIndexing(var, divisor, modulus))
            divisor = divisor * modulus
            while size_hint(size_new) > size_hint(size_old):
                modulus = stack_old.pop()
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                size_old = size_old * modulus
            V.graph.sizevars.guard_equals(size_new, size_old)
        else:
            raise AssertionError

    # 教学：若旧维还有剩余，只允许它们都是 size 1，对应坐标补 0。
    while stack_old:
        size_old = stack_old.pop()
        V.graph.sizevars.guard_equals(size_old, 1)  # type: ignore[arg-type]
        view_expr.append(sympy.Integer(0))

    # 教学：若新维还有剩余，也只允许 size 1；它们不需要出现在旧坐标表达式中。
    while stack_new:
        var, size_new = stack_new.pop()
        V.graph.sizevars.guard_equals(size_new, 1)  # type: ignore[arg-type]

    # 教学：算法从末维产生表达式，返回前翻转为旧 shape 的正向维序。
    view_expr.reverse()
    assert len(view_expr) == len(old_size)

    def reindex(index):
        assert len(index) == len(vars), (len(index), len(vars))
        replacements = dict(zip(vars, index))
        return tuple(sympy_subs(x, replacements) for x in view_expr)  # type: ignore[arg-type]

    return reindex
```

### `[6]→[2,3]`：两个新维合成一个旧坐标

旧 shape 只有一维 6，新 shape 是 `[2,3]`。从末尾取旧 size 6、新 size 3，进入 `size_new < size_old`：

```text
初始：var=v1, size_new=3
再取：var2=v0, size_new2=2
合并：var=v0*3+v1, size_new=3*2=6
```

所以 `reindex([row,column])=(row*3+column,)`。新坐标 `[1,2]` 变成旧线性坐标 5。`guard_equals(6,6)` 记录两个维组确实覆盖同样数量的元素。

### `[2,3]→[6]`：一个新坐标拆成两个旧坐标

新坐标记作 v0。先遇到旧末维 3：

```text
旧 column = (v0 // 1) % 3
旧 row    = (v0 // 3) % 2
```

源码先按末维顺序 append，最后 reverse 成 `(row,column)`。代入 v0=5，得到 row=1、column=2；再代入旧连续地址 `1*3+2=5`，与新线性位置相同。

### `[2,3]→[3,2]`：fallback 怎样组合两次变换

第一步 `[2,3]←[6]` 把 flat f 变成 `(f//3, f%3)`；第二步 `[6]←[3,2]` 把新坐标 `(a,b)` 变成 `f=a*2+b`。组合后：

```text
old_row = (a*2+b) // 3
old_col = (a*2+b) % 3
```

代入新坐标 `[2,1]`，flat=5，旧坐标 `[1,2]`。这与连续 reshape 的逻辑顺序一致；区别在于这里把关系保存为符号算式，供旧 loader 读取，而不是声称底层已有可直接重解释的连续 layout。

size-1 分支也维持同一原则。旧 `[2,1,3]` reshape 为 `[2,3]` 时，旧中间维只能取 0，最终 reindex 为 `(new0,0,new1)`。反过来新增一个 size-1 新维时，该新坐标不会消耗旧维。

### reindex 边界对照

1. `[12]→[2,2,3]` 的旧线性坐标是 `i0*6+i1*3+i2`。
2. `[2,2,3]→[12]` 对新线性 11 拆出旧坐标 `[1,1,2]`。
3. `[2,1,3]→[6]` 对 flat=4 得到旧 `[1,0,1]`；size-1 维始终为 0。
4. 若 old/new 总元素数不同，前面的 `resolve_negative_size` 已通过 `guard_equals` 拒绝；indexer 不负责偷偷截断或补元素。

下一叶子继续解释这些延迟规则在什么消费者要求下物化，以及 layout 怎样最终决定。
