# 从延迟规则到具名存储：realize 与布局决定

[返回存储与布局](../README.md) · [返回 Lowering](../../README.md)

Q221 构造出的 Pointwise 或 Reduction 可以只保存“给定位置怎样计算”的规则。只要后续算子愿意继续内联，这份结果不必立即拥有独立名字和存储。遇到 graph output、外部调用、旧值保护或内联复杂度限制时，编译器才需要把它登记为一个可被稳定引用的结果 buffer。

本叶子沿当前源码顺序解释两项连续决定：StorageBox 怎样把延迟 Loops 换成 ComputedBuffer；ComputedBuffer 的 FlexibleLayout 随后怎样根据实际读取方式或已有函数的 size/stride 要求选定具体 stride。运行时空间何时分配由 wrapper/codegen 继续处理。

## K01：StorageBox 原地切换底层表示

同一个 `StorageBox` 可先指向 Pointwise，后来原地改为 ComputedBuffer：

```text
realize 前：TensorBox → StorageBox → Pointwise
realize 后：TensorBox → 同一 StorageBox → ComputedBuffer(buf0) → 原 Pointwise
```

持有 StorageBox 的消费者不需要全部换引用；它们随后都会看到 `data` 已经是 buf0。

### 完整源码

进入代码时，`StorageBox` 继承可原地替换 `data` 字段的 `MutableBox`。本例 `self.data` 起初是 size `[2,3]` 的 Pointwise；它只读取 x 的 `3*i0+i1` 一次，所以 `nontrivial_read_count=1`。这里每个首次出现、索引不是常数的 load 计一次；同一读取值被公共子表达式复用不会重复计数，`realize_hint()` 只在计数大于 1 时因这项条件立即物化。`is_input_buffer()` 和 `is_module_buffer()` 都返回 False。当前 FX 节点名 add，stack trace 指向 `model.py:12`；realize 把这份 origin/traceback 复制到新 buffer，Scheduler/Codegen 的调试映射随后可把 buf0 指回该 add。`realize()` 建立 ComputedBuffer，把它登记成 buf0/op0，再令 `self.data` 指向这个新对象并返回名字 `buf0`。已经持有同一 StorageBox 的 TensorBox 不需要更换。

```python
class StorageBox(MutableBox):
    def is_input_buffer(self):
        # 教学：InputBuffer 或指向输入的 ReinterpretView 都保留图输入身份。
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
        return False

    def is_module_buffer(self):
        # 教学：只有登记在 graph.constants 中的 ConstantBuffer 才是模块常量身份。
        return (
            isinstance(self.data, (ConstantBuffer))
            and self.data.get_name() in V.graph.constants
        )

    def realize(self):
        # 教学：这些对象已经有 storage/调用结果身份，重复 realize 只返回现有名字。
        if isinstance(
            self.data,
            (
                ComputedBuffer,
                InputsKernel,
                InputBuffer,
                ReinterpretView,
                TemplateBuffer,
            ),
        ):
            return self.data.get_name()
        # 教学：其余只允许四类延迟循环；未知 IR 在这里失败而非静默包装。
        assert isinstance(self.data, (Pointwise, Reduction, Scan, Sort)), type(
            self.data
        )
        # 教学：换包装前保存诊断来源，稍后转交给新 ComputedBuffer。
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
        # 教学：登记 buffer 名和 producing operation；这仍是编译期 IR 状态。
        self.data.name = V.graph.register_buffer(self.data)
        V.graph.register_operation(self.data)
        self.data.origins = self.origins
        self.data.origin_node = origin_node
        self.data.traceback = traceback
        return self.data.name

    def realize_hint(self):
        """
        Called on buffers we expect to be forced to realize later.
        """
        # 教学：hint 只在非平凡读取超过 1 时立即执行；否则保留延迟规则。
        if (
            isinstance(self.data, (Pointwise, Reduction))
            and self.data.inner_fn_opcount().nontrivial_read_count > 1
        ):
            self.realize()

    def has_exceeded_max_reads(self):
        # 教学：累计读取数或表达式体积超过阈值，只对 Pointwise 生效。
        return isinstance(self.data, Pointwise) and (
            self.num_reads() > config.realize_acc_reads_threshold
            or self.has_large_inner_fn()
        )

    def should_realize_on_reuse(self, users):
        """
        A heuristic to decide if we should realize a tensor
        that is used multiple times.
        """
        if users > 1 and isinstance(self.data, (Pointwise, Reduction)):
            if is_cpu(self.data):
                # Heuristic for realizing reused result of heavy ops on cpu
                opcount = self.data.inner_fn_opcount()
                heavy_ops = ["exp", "sigmoid"]  # a list of heavy ops
                # 教学：CPU 的重操作被复用时，即使读数不多也倾向保存结果。
                if any(x in opcount.used_ops for x in heavy_ops):
                    return True
            return (
                self.num_reads() > config.realize_reads_threshold
                or self.has_large_inner_fn()
            )
        return False

    def mark_reuse(self, users):
        if self.should_realize_on_reuse(users):
            self.realize()

    def num_reads(self):
        return self.data.num_reads()
```

`inner_fn_opcount()` 会用 `OpCounterCSE` 符号执行一次 inner_fn。计数器的 load 分支说明了前文“非平凡读取”和去重的精确定义：底层 mock handler 为相同读取产生同一 value；只有该 value 尚未出现在 `var_names` 时才登记读取，而 index 不是 Python int 或 SymPy Integer 时才增加 `nontrivial_read_count`。

```python
@cache_on_self
def inner_fn_opcount(self) -> OpCountResult:
    opcounter = OpCounterCSE(V.MockHandler())
    with V.set_ops_handler(opcounter), patch.object(
        FlexibleLayout, "allow_indexing", True
    ):
        self.inner_fn(*self.inner_fn_args())
        return opcounter.getvalue()


def load(self, name: str, index: sympy.Expr) -> str:
    val = self.parent_handler.load(name, index)
    if val not in self.var_names:
        self._used_ops.add("load")
        self._read_names.append(name)
        if not isinstance(index, (sympy.Integer, int)):
            self._nontrivial_read_count += 1
    return self._update_count(val)
```

### realize 对已有存储与延迟计算分别返回什么

输入 x 已经是 `StorageBox(InputBuffer("arg0_1"))`。调用 `realize()` 直接返回 `arg0_1`，不会登记新 operation。转置输入若是 `ReinterpretView`，`BaseView.get_name()` 会继续调用底层 data 的 `get_name()`，因此返回的仍是底层 storage 名；view 自己的 FixedLayout 另外保存 size、stride 和 offset。

对延迟 `x+1`，StorageBox.data 是 Pointwise。`realize()` 保存它的 origin/traceback，建立 `ComputedBuffer(name=None, FlexibleLayout(cpu,float32,[2,3]), pointwise)`，再由 GraphLowering 登记为例如 buf0/op0。原 Pointwise 没有消失，而是成为 ComputedBuffer.data，负责将来生成每个位置的 store 值。

因此 “realize 之后有 buf0” 是编译期承诺：后续调度和代码生成能引用 buf0。它不表示此刻执行了 x+1，也不表示 Python 调用期间已经分配设备内存。

### 三个启发式分别问什么

| 接口 | 问题 | 不满足时 |
| --- | --- | --- |
| `realize_hint()` | Pointwise/Reduction 内部非平凡读取是否超过 1 | 保留延迟规则 |
| `has_exceeded_max_reads()` | Pointwise 累计读取或 inner_fn 大小是否超过阈值 | 返回 false |
| `mark_reuse(users)` | users>1 且重算代价是否高；CPU exp/sigmoid 特判 | 不改变 StorageBox.data |

例如 `a=exp(x); return a+1,a*2` 中 a 有两个 users。CPU 分支在 opcount 中看到 exp，`should_realize_on_reuse(2)` 返回真，mark_reuse 调用 realize。若相同 a 只有一个 user，最外层 users>1 不成立，重操作特判也不会进入。

这些接口是启发式，不承担 graph output、extern input 或 mutation 的强制合同；调用方在需要稳定 storage 时会直接 `realize()`。

### 用新情况检验理解

1. 对已经是 ComputedBuffer 的值连续调用两次 realize：两次都返回同一名字，不新增 operation。
2. Reduction 的非平凡读取数为 1，调用 realize_hint：仍保持 Reduction；hint 不是强制命令。
3. Pointwise 有三个 users，但读取数低、表达式小、CPU opcount 不含 exp/sigmoid：`should_realize_on_reuse` 可返回 false。
4. data 是未知 IR 类型：realize 在允许类型断言处失败，不会把任何对象都包装成 ComputedBuffer。

下一知识点 K02 打开 Buffer 的五种 layout 冻结接口，比较默认连续、stride order、fill order、same order 和 exact strides 各自固定什么。

## K02：五类要求怎样为同一 shape 选定 stride

FlexibleLayout 已有一份暂定 stride，但仍允许改变。冻结分两层：`FlexibleLayout.as_*` 负责根据规则构造 FixedLayout；`Buffer.freeze_layout*` 负责检查当前 layout 是否允许改变，再把 buffer.layout 替换掉。

### 完整源码：FlexibleLayout 计算目标 stride

第一次阅读用 shape `[2,3]`、CPU float32、offset 0。默认连续 layout 的 fill order 是 `[1,0]`，得到 stride `[3,1]`；stride order `[1,0]` 转成 fill order 后同样得到 `[3,1]`，stride order `[0,1]` 则得到 `[1,2]`；exact strides `[10,2]` 会原样冻结成 `[10,2]`。

当前 `FlexibleLayout` 不保存一个以后才解释的 `_stride_order`。构造时若第四个参数非空，代码直接把它当 fill order 计算 stride；否则立即计算默认连续 stride。后面的 `as_stride_order` 才接收 stride order，并先调用 `stride_order2fill_order`：

```python
def stride_order2fill_order(
    order: Sequence[Union[int, Integer]]
) -> Sequence[Union[int, Integer]]:
    lookup = {pos: idx for idx, pos in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


class FlexibleLayout(Layout):
    def __init__(self, device, dtype, size, stride_order=None):
        if stride_order:
            strides = FlexibleLayout.fill_ordered(size, stride_order)
        else:
            strides = FlexibleLayout.contiguous_strides(size)
        super().__init__(device, dtype, size, strides)
```

构造器参数名虽然叫 `stride_order`，实际传给 `fill_ordered`；本文为了避免混淆，把这个构造参数称为“初始 fill order”。

padding 只在 `comprehensive_padding=True` 且调用路径允许时发生。以默认阈值 1024、128-byte 对齐和 float32 为例，shape `[3,1025]` 的连续候选 stride 原为 `[1025,1]`；外层 stride 1025 超过阈值且不是 32 个 float32 元素的倍数，会向上补到 1056，得到 `[1056,1]`。动态尺寸、禁用配置或 `allow_padding=False` 都保留原 stride。

对齐元素数和总开关来自这两个直接 helper；阈值与 byte 对齐的具体默认值是当前配置值，案例把 1024 和 128 bytes 代入，而算法本身读取 config：

```python
def get_align_for_dtype(dtype: torch.dtype) -> int:
    return config.padding_alignment_bytes // dtype.itemsize

def should_pad_strides(self):
    return config.comprehensive_padding and isinstance(self, FlexibleLayout)
```

完整 padding 规则保持原 stride order，从最快维重新累乘：最快维 stride 固定为 1；每生成下一维 stride，如果它大于阈值且不是 `align=128 bytes / dtype.itemsize` 的倍数，就向上取整到 align。channels-last 在配置禁止 padding 时跳过；当前 FX 节点带 `dislike_padding` 时跳过；任一 size/stride 不是静态整数时也跳过。只有确实发生过补齐才采用重算结果，避免 size=1 维把等价 stride 无谓改掉。

```python
@staticmethod
def _pad_strides(in_strides, size, dtype):
    align = get_align_for_dtype(dtype)
    if len(in_strides) == 0:
        return in_strides
    if not config.pad_channels_last and Layout.is_channels_last_contiguous(
        size, in_strides
    ):
        return in_strides
    current_fx_node = V.get_current_node()
    if hasattr(current_fx_node, "meta") and current_fx_node.meta.get(
        "dislike_padding", False
    ):
        return in_strides
    if not all(
        isinstance(s, (int, sympy.Integer))
        for s in itertools.chain(in_strides, size)
    ):
        return in_strides

    stride_order = get_stride_order(in_strides)
    fill_order = stride_order2fill_order(stride_order)
    new_strides = [0 for _ in range(len(in_strides))]
    new_strides[fill_order[0]] = 1
    padded = False
    for rank, idx in enumerate(fill_order[1:], start=1):
        prev_idx = fill_order[rank - 1]
        stride = new_strides[prev_idx] * size[prev_idx]
        if stride > config.padding_stride_threshold and stride % align != 0:
            stride = ceildiv(stride, align) * align
            padded = True
        new_strides[idx] = stride
    if not padded:
        return in_strides
    metrics.num_comprehensive_padding += 1
    return new_strides
```

```python
class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    allow_indexing = False

    # WARNING!  This doesn't handle zero size tensors correctly
    @staticmethod
    def contiguous_strides(sizes):
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.Integer(1)]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def fill_ordered(sizes, order):
        """
        Create a stride based on the order the dimensions should be filled in.

        In this format, channels last would be:
            [1, 3, 2, 0]
        """
        assert OrderedSet(range(len(sizes))) == OrderedSet(order), (sizes, order)
        next_stride = sympy.Integer(1)
        strides = [None] * len(order)

        # 教学：order 从最快变化维开始；每放一维，下一 stride 乘该维 size。
        for i in order:
            strides[i] = next_stride
            next_stride = next_stride * sizes[i]
        return strides

    @staticmethod
    def stride_ordered(sizes, order):
        """
        Create a stride based on the sorted order of a permuted range.

        In this format, channels last would be:
            [3, 0, 2, 1]
        """
        assert OrderedSet(range(len(sizes))) == OrderedSet(order)
        fill_order = stride_order2fill_order(order)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @staticmethod
    def stride_ordered_for_memory_format(sizes, memory_format):
        """
        Create a stride based on a memory format.

        Memory format is translasted into a stride order,
        so channels_last is the same as:
            FlexibleLayout.stride_ordered(sizes, [3, 0, 2, 1])

        This interface does not support memory_format `torch.preserve_format`
        which should be used to deduce a format from another source
        """
        if memory_format == torch.channels_last:
            return FlexibleLayout.stride_ordered(sizes, NHWC_STRIDE_ORDER)
        elif memory_format == torch.channels_last_3d:
            return FlexibleLayout.stride_ordered(sizes, NHWDC_STRIDE_ORDER)
        elif memory_format == torch.contiguous_format:
            return FlexibleLayout.contiguous_strides(sizes)
        else:
            log.debug(
                "stride_ordered_for_memory_format, unsuppored memory_format: %s",
                memory_format,
            )
            raise NotImplementedError

    @staticmethod
    def same_ordered(sizes, stride):
        """
        Create a stride that has the same stride order as given stride

        For example, if given stride is [1000, 1, 100, 10],
        the fill order should be [1, 3, 2, 0]
        """
        assert len(sizes) == len(stride)
        stride = [V.graph.sizevars.size_hint(x) for x in stride]
        fill_order = sorted(range(len(stride)), key=stride.__getitem__)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    def as_stride_order(self, order, allow_padding=False):
        new_stride = self.stride_ordered(self.size, order)
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def as_exact_strides(self, exact_strides, allow_padding=False):
        new_stride = exact_strides
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def as_fill_order(self, order):
        new_stride = self.fill_ordered(self.size, order)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )

    def as_same_order(self, stride):
        new_stride = self.same_ordered(self.size, stride)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
        )
```

### 完整源码：Buffer 将选择写回自身 layout

```python
def freeze_layout(self):
    # 教学：普通 flexible/fixed layout 可转 fixed；多输出和非拥有 layout 保留特殊合同。
    if not isinstance(self.layout, (MultiOutputLayout, NonOwningLayout)):
        self.layout = self.layout.as_fixed()

def freeze_layout_with_stride_order(self, order, allow_padding=False):
    assert isinstance(self.layout, FlexibleLayout)
    self.layout = self.layout.as_stride_order(order, allow_padding=allow_padding)

def freeze_layout_with_fill_order(self, order):
    assert isinstance(self.layout, FlexibleLayout)
    self.layout = self.layout.as_fill_order(order)

def freeze_layout_with_same_order(self, stride):
    assert isinstance(self.layout, FlexibleLayout)
    self.layout = self.layout.as_same_order(stride)

def freeze_layout_with_exact_strides(self, exact_strides, allow_padding=False):
    assert isinstance(self.layout, FlexibleLayout)
    self.layout = self.layout.as_exact_strides(
        exact_strides, allow_padding=allow_padding
    )
```

这五个方法在产品中属于 `Buffer`；代码块去掉类级缩进。除普通 `freeze_layout` 外，四个定向接口都断言当前仍是 FlexibleLayout，因此一旦先冻结为 FixedLayout，不能再调用另一种接口随意改写。

### `[2,3]` 的五种具体结果

忽略可选 padding，device/dtype/offset 均保持不变：

| 调用 | 中间规则 | 得到 stride |
| --- | --- | --- |
| `freeze_layout()` | 使用 FlexibleLayout 当前默认连续 stride | `[3,1]` |
| `freeze_layout_with_fill_order([1,0])` | dim1 最快：stride1=1；再乘 size1=3 | `[3,1]` |
| `freeze_layout_with_fill_order([0,1])` | dim0 最快：stride0=1；再乘 size0=2 | `[1,2]` |
| `freeze_layout_with_stride_order([1,0])` | 转为 fill order `[1,0]` | `[3,1]` |
| `freeze_layout_with_same_order([100,1])` | 参考 stride 从小到大为 dim1、dim0 | `[3,1]` |
| `freeze_layout_with_exact_strides([10,2])` | 不重新计算排列，直接采用指定值 | `[10,2]` |

fill order 直接列“从最快维到最慢维”；stride order 列每个维在 stride 排名中的位置，两者不能只看相同数字就混为一个概念。`same_order` 只保留参考 stride 的相对快慢，不保留 100 和 1 的具体间距；`exact` 才保留具体数值。

`same_ordered` 会先对每个符号 stride 调 `size_hint`，因此它要求当前 shape 环境能提供可用 hint；随后 Python 的稳定排序按 stride 数值升序排列，相等时保留原维号顺序。例如参考 `[1,1,6]` 的 fill order 是 `[0,1,2]`。`significant_strides_equal` 对符号 size/stride 同样依据 `size_hint`；没有可靠 hint 的动态情形不能仅凭本文静态数值案例推断一定复用。

`allow_padding` 只有对应 layout 的 `should_pad_strides()` 为真时才允许 `_pad_strides` 改写计算出的 stride。没有这个条件时，上表就是实际结果；不能看到参数名就断言所有张量都会 padding。

### 特殊 layout 为什么不使用普通的 stride 选择过程

NonOwningLayout 表示 producer 直接写入另一 storage 的 view；MultiOutputLayout 表示一个 operation 的多项输出接口。普通 `freeze_layout()` 对它们不调用 `as_fixed()`，因为把它们压成独立 FixedLayout 会丢失拥有关系或多输出合同。四个定向接口则直接要求 FlexibleLayout，传入这些特殊 layout 会在断言处失败。

### 用新情况检验理解

1. shape `[2,3,4]`、fill order `[2,1,0]`：stride 依次为 dim2=1、dim1=4、dim0=12，即 `[12,4,1]`。
2. 同 shape、参考 stride `[1,8,2]`：相对顺序 dim0、dim2、dim1，得到 `[1,8,2]`；这里恰好数值相同来自 shape，合同仍只保证顺序。
3. 已调用 exact strides 冻结后再调用 fill-order：断言失败，因为 layout 已是 FixedLayout。
4. NonOwningLayout 调普通 freeze：保持原对象；它不是待选择排列的独立 storage。

下一知识点 K03 进入 ComputedBuffer：从实际 MemoryDep 读取式提取 stride hints，经 `pick_loop_order` 得到 fill order，再调用本节接口冻结。

## K03：ComputedBuffer 用实际读取式选择 fill order

### 完整源码

本次对象是 `ComputedBuffer(name="buf0", layout=FlexibleLayout(size=[2,3]), data=Pointwise)`；Pointwise 读取 x 的索引是 `3*d0+d1`，没有 reduction 轴，用户尚未要求 exact stride。依赖提取得到这一条 MemoryDep，stride hint 表明 d1 变化最快、d0 次之，因此 `get_fill_order()` 返回 `[1,0]`；`decide_layout()` 再把 layout 冻结为 stride `[3,1]`。若只有 StarDep 或 layout 已固定，分别返回 None 或直接不改。

```python
def get_fill_order(self):
    """
        If our layout is still flexible, try to determine the stride order based on stride orders of reads.

        TODO(jansel): A better algorithm here would look at downstream consumers of this
                      value and try to do global graph-level layout optimization.
                      This is also something just begging to be autotuned.
        """
    if isinstance(self.layout, FlexibleLayout):
        (index_vars, reduction_vars), _ = dependencies.index_vars_squeeze(
            self.data.get_pointwise_size(), self.data.get_reduction_size()
        )
        reads = self.get_read_writes().reads
        # only consider reads to buffer of same size
        # ignore StarDeps because they don't contribute stride information
        assert all(
            isinstance(r, (dependencies.StarDep, dependencies.MemoryDep))
            for r in reads
        )
        # 教学：MemoryDep 保留 index；规约变量先置 0，避免规约循环左右输出布局。
        reads = [
            sympy_subs(
                r.index, {v: sympy.Integer(0) for v in reduction_vars if v != 0}
            )
            for r in reads
            if isinstance(r, dependencies.MemoryDep)
        ]

        if reads:
            if isinstance(self.data, (Scan, Sort)):
                indices = self.data.reindex(index_vars, reduction_vars)
            else:
                indices = index_vars
            # 教学：每个读取式分别求各输出维前进一步时，物理 index 增加多少。
            stride_lengths = [
                V.graph.sizevars.stride_hints(expr, indices) for expr in reads  # type: ignore[arg-type]
            ]
            from .scheduler import pick_loop_order

            return pick_loop_order(stride_lengths, self.get_size())

    return None

def decide_layout(self):
    if isinstance(self.layout, FlexibleLayout):
        order = self.get_fill_order()
        if order:
            # 教学：有读访问建议时，按 fill order 冻结。
            self.freeze_layout_with_fill_order(order)
        else:
            # 教学：无建议时保留 FlexibleLayout 当前 stride，并普通冻结。
            self.freeze_layout()
```

上面真正把多条 stride hints 合成一个 fill order 的是 `pick_loop_order`。它比较任意两个输出维 a、b：对每一项读取分别投票；a 的绝对 stride 更小，或 b 在该读取中是广播 stride 0，就给 a 一票。b 同理。票多的维放在前面，表示先填、stride 更小；平票时维号较大的维在前，保持默认连续顺序。size=1 的维不影响地址，被移到最后。

```python
def pick_loop_order(
    stride_lengths: List[List[int]],
    sizes: List[sympy.Expr],
    priority_idx: Tuple[int, ...] = (),
) -> List[int]:
    @functools.cmp_to_key
    def index_cmp(a: int, b: int) -> int:
        if sizes[a] == 1 or sizes[b] == 1:
            return cmp(sizes[a] == 1, sizes[b] == 1)

        stride_len_a = [abs(sl[a]) for sl in stride_lengths]
        stride_len_b = [abs(sl[b]) for sl in stride_lengths]
        a_first = sum(
            sl_b == 0 or sl_a < sl_b
            for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        b_first = sum(
            sl_a == 0 or sl_b < sl_a
            for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        if a_first > b_first:
            return -1
        if b_first > a_first:
            return 1
        return cmp(b, a)

    order = list(reversed(range(len(stride_lengths[0]))))
    if len(priority_idx) > 0:
        stride_lengths = [stride_lengths[pi] for pi in priority_idx]
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order
```

广播 stride 0 的处理容易读反。对单条 `[0,1]` 比较 dim0 与 dim1：dim0 本身为 0 时，`b_first` 的 `sl_a==0` 成立，票给 dim1；所以有效读取 stride 1 的 dim1 排在广播 dim0 前，得到 `[1,0]`。广播维不会因为数值 0 被误选成最快输出维。

再看两项冲突读取：第一项 hints `[1,4]` 偏好 dim0，第二项 `[4,1]` 偏好 dim1。两维各得一票，比较器走平票规则，让维号较大的 dim1 在前，fill order 为 `[1,0]`。若 `priority_idx=(0,)`，投票前只保留第一项读取，结果改为 `[0,1]`。如果关闭 `config.pick_loop_orders`，函数不排序，直接返回默认的倒序维号。

size=1 维在地址上没有多个位置可排序。例如输出 shape `[2,1,3]`，即使某项 hints 是 `[3,99,1]`，比较器遇到 dim1 时先走 size=1 分支，把 dim1 移到最后；dim2 与 dim0 再按 1 和 3 排序，得到 fill order `[2,0,1]`。对应紧凑 stride 是 `[3,3,1]`；中间维 stride 数值不影响任何合法地址。

两个方法属于 `ComputedBuffer`，代码块去掉类级缩进。若 layout 已经被外部合同固定，最外层 `isinstance(FlexibleLayout)` 为假，`decide_layout` 不再覆盖它。

### 四种 `[2,3]` 读取式

实验 21 用真实 Pointwise、ComputedBuffer 和产品依赖提取直接观测：

| inner loader 的物理 index | stride hints | pick_loop_order | 最终 stride |
| --- | --- | --- | --- |
| `3*d0+d1` | `[3,1]` | fill `[1,0]` | `[3,1]` |
| `d0+2*d1` | `[1,2]` | fill `[0,1]` | `[1,2]` |
| `d1` | `[0,1]` | fill `[1,0]` | `[3,1]` |
| 常量，无 MemoryDep | 无 | `None` | 默认 `[3,1]` |

第一行让 d1 连续变化，因为它使输入地址每次只加 1。第二行让 d0 最快，因为 d0 的地址增量为 1。第三行对 d0 是广播 stride 0；当前 `pick_loop_order` 仍选 d1 为最快有效访问维，输出冻结为行连续。最后一行没有读取可投票，走普通 freeze。

有多个 MemoryDep 时，`stride_lengths` 保存多行建议，`pick_loop_order` 按上面的逐对投票比较各维；它不是只看第一项。MemoryDep 是“读某个 buffer 的具体 index 表达式”，因此能计算 stride；StarDep 只表示对整个结果的依赖，没有单一 index，源码将它排除。

`index_vars_squeeze(pointwise_size,reduction_size)` 为普通输出维建立 `index_vars`，为规约维建立 `reduction_vars`；size=1 的维没有变化空间，可能直接用字面量 0 表示，所以替换表要过滤 `v != 0`，不能把常数 0 当成待替换符号。其余组内变量统一替换成 0，是因为这里决定输出位置的排列，而不是规约组内循环怎样执行。

### 用新情况检验理解

1. layout 已是 exact `[10,2]`：decide_layout 不动作，读取启发式不能覆盖外部合同。
2. 只有 StarDep：过滤后 reads 为空，get_fill_order 返回 None，普通冻结。
3. Pointwise 读取 `6*d0+2*d1`：相对快慢仍是 d1 再 d0；fill order `[1,0]`，输出获得紧凑 `[3,1]`，不会照抄输入间距 6 和 2。
4. Scan/Sort 使用自己的 reindex 后再算 stride hints；普通 Pointwise 直接使用 index_vars。

下一知识点 K04 解释 ExternKernel 的 stride constraint：何时只冻结现有 FlexibleLayout，何时对 view 建 ReinterpretView，何时必须复制出满足合同的新结果。

## K04：外部调用明确要求的 stride 优先于布局启发式

### 完整源码

先走普通 buffer：x 的完整包装是 `TensorBox → StorageBox → ComputedBuffer(size=[2,3])`，其中 ComputedBuffer 当前持有 FlexibleLayout stride `[3,1]`；调用给 `order=[1,0]`、`exact_strides=None`、`allow_padding=False`。helper 先剥 TensorBox，再在 StorageBox 分支找到内部 Buffer，把它的 layout 冻结成按该顺序计算的 FixedLayout，不复制数据。裸 ComputedBuffer 不是这个接口的直接输入。随后再对照四种变化：x 是 view 时尝试建立 ReinterpretView；已有 FixedLayout 不满足 exact `[1,2]` 时复制；零元素 tensor 直接返回。

`require_strides` 开头反复调用的 storage helper 先把包装拆开。TensorBox 递归到 `data`；StorageBox 只有在 `data` 已是 Buffer 时才交回 `(StorageBox, layout)`，并按参数冻结；ReinterpretView 递归取得底层 StorageBox，但保留 view 自己的 layout。其他对象抛 NotImplementedError，因此 `is_storage_and_layout` 返回 false。

```python
def is_storage_and_layout(x: IRNode) -> bool:
    try:
        as_storage_and_layout(x, freeze=False)
        return True
    except NotImplementedError:
        return False


def as_storage_and_layout(
    x: IRNode,
    freeze: bool = True,
    want_contiguous: bool = False,
    stride_order: Optional[Sequence[Union[int, Integer]]] = None,
    allow_padding: bool = False,
    exact_strides: Optional[Sequence[Union[int, Integer]]] = None,
) -> Tuple[StorageBox, Layout]:
    if isinstance(x, TensorBox):
        return as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
    if isinstance(x, StorageBox) and isinstance(x.data, Buffer):
        if freeze:
            if want_contiguous:
                x.data.freeze_layout()
                assert x.data.layout.is_contiguous()
            elif stride_order is not None:
                x.data.freeze_layout_with_stride_order(
                    stride_order, allow_padding=allow_padding
                )
            elif exact_strides is not None:
                x.data.freeze_layout_with_exact_strides(
                    exact_strides, allow_padding=allow_padding
                )
            else:
                x.data.decide_layout()
        return x, x.data.layout
    if isinstance(x, ReinterpretView):
        buffer, _ = as_storage_and_layout(x.data, freeze=freeze)
        return buffer, x.layout
    raise NotImplementedError
```

所以“x 是 StorageBox 时递归处理 data”更准确地说，是 helper 检查 `StorageBox.data` 是否已经是 Buffer；它不会把仍指向 Pointwise 的 StorageBox 当成现成 storage。

```python
@classmethod
def require_strides(
    cls,
    x,
    order: Optional[Sequence[int]] = None,
    exact_strides: Optional[Sequence[_IntLike]] = None,
    allow_padding=False,
):
    assert order is not None or exact_strides is not None
    if x.get_numel() == 0:  # Layout doesn't matter
        return x
    # require x to have the layout
    if is_storage_and_layout(x):
        # 教学：NonOwningLayout 沿 view 找到真正承担 storage 合同的对象。
        while isinstance(x.get_layout(), NonOwningLayout):
            x = x.get_layout().view
        if isinstance(x.get_layout(), FlexibleLayout):
            if order:
                # If the the FlexibleLayout already has the size and stride in the required order,
                # freeze it to a FixedLayout by using its current size and stride.
                # The behavior of using its current size and stride or the given order can be different
                # if the size and stride has ambiguilty, for example for a 4D input where the iC = 1:
                # size=[s0, 1, 28, 28], stride=[784, 784, 28, 1]. If the required order is [3, 0, 2, 1] (channels last),
                # the current size and stride already satisfies this order.
                # However by freezing it to the required order, the layout will be changed to:
                # size=[s0, 1, 28, 28], stride=[784, 1, 28, 1]), which is not actually necessary.

                # fix flexiblelayout to be FixedLayout with stride_order
                # 教学：当前 stride 已满足 order 时保留其实际排序，避免 size-1 歧义导致无谓改变。
                as_storage_and_layout(
                    x,
                    freeze=True,
                    want_contiguous=False,
                    stride_order=get_stride_order(
                        V.graph.sizevars.size_hints(x.get_layout().stride)
                    )
                    if is_stride_order_storage_and_layout(x, order)
                    else order,
                    allow_padding=allow_padding,
                )
                return x
            else:
                # If the exact_strides is given, freeze the FlexibleLayout to a FixedLayout with the exact_strides.
                as_storage_and_layout(
                    x,
                    freeze=True,
                    want_contiguous=False,
                    stride_order=None,
                    allow_padding=allow_padding,
                    exact_strides=exact_strides,
                )
                return x
        elif isinstance(x.get_layout(), FixedLayout) and (
            (order and x.get_layout().is_stride_ordered(order))
            or (
                exact_strides
                and significant_strides_equal(
                    exact_strides, x.get_layout().stride, x.get_size()
                )
            )
        ):
            # 教学：已经满足 order/exact（exact 比较忽略 size-1 维）时复用原对象。
            return x
        elif isinstance(x.get_layout(), MutationLayoutSHOULDREMOVE):
            if isinstance(x.get_layout().real_layout(), FlexibleLayout):
                raise AssertionError(
                    "the MutationLayoutSHOULDREMOVE's real layout shouldn't be FlexibleLayout"
                )
            elif isinstance(x.get_layout().real_layout(), FixedLayout) and (
                (order and x.get_layout().real_layout().is_stride_ordered(order))
                or (
                    exact_strides
                    and significant_strides_equal(
                        exact_strides,
                        x.get_layout().real_layout().stride,
                        x.get_size(),
                    )
                )
            ):
                return x

    # TODO - Storage to InputBuffer
    if isinstance(x, InputBuffer) and (
        (order and x.get_layout().is_stride_ordered(order))
        or (
            exact_strides
            and significant_strides_equal(
                exact_strides, x.get_layout().stride, x.get_size()
            )
        )
    ):
        return x
    if (
        isinstance(x, TensorBox)
        and isinstance(x.data, BaseView)
        and not isinstance(x.data, ReinterpretView)
        and is_storage_and_layout(x.unwrap_view())
        and not isinstance(x.unwrap_view().data, ExternKernelAlloc)
    ):
        try:
            # 教学：可转换的 Generic View 先改为同 storage 的 ReinterpretView，再递归检查合同。
            x.data = cls.convert_to_reinterpret_view(x.data)
            if order:
                return cls.require_stride_order(
                    x, order, allow_padding=allow_padding
                )
            elif exact_strides:
                return cls.require_exact_strides(
                    x, exact_strides, allow_padding=allow_padding
                )
        except NotImplementedError:
            pass
    # Although this is a clone, inductor is good about fusing clones into previous
    # operations if they weren't realized and their layouts were flexible.
    # 教学：前面都不能复用时，复制出新逻辑结果并冻结为目标布局。
    x = cls.copy_input(x)
    as_storage_and_layout(
        x,
        freeze=True,
        want_contiguous=False,
        stride_order=order,
        allow_padding=allow_padding,
        exact_strides=exact_strides,
    )
    if order:
        assert is_stride_order_storage_and_layout(x, order)
    return x

@classmethod
def require_exact_strides(cls, x, exact_strides, allow_padding=False):
    return cls.require_strides(
        x, exact_strides=exact_strides, allow_padding=allow_padding
    )

@classmethod
def require_stride_order(cls, x, order, allow_padding=False):
    return cls.require_strides(x, order=order, allow_padding=allow_padding)
```

三个方法属于 `ExternKernel`。两个短入口把 order 或 exact 合同传给共同分支；非空参数的主要出口是返回满足合同的 storage/view，或建立 copy 后返回新对象。后文单列 rank-0 空序列的 truthiness 差异，以及 view/NonOwning 路径可能产生的入口包装副作用和返回身份变化。

这里的“exact 相等”会忽略 size=1 维，而不是忽略 size=0 维或任意符号差异。helper 先保留 `size_hint(dim,fallback=2) != 1` 的维，再把两组对应 stride 求 hint 后逐项比较：

```python
def significant_strides_equal(
    strides1: Sequence[_IntLike], strides2: Sequence[_IntLike], size: Sequence[_IntLike]
) -> bool:
    non_1_indices = [
        i
        for i, dim in enumerate(size)
        if V.graph.sizevars.size_hint(dim, fallback=2) != 1
    ]
    strides1 = [V.graph.sizevars.size_hint(strides1[i]) for i in non_1_indices]
    strides2 = [V.graph.sizevars.size_hint(strides2[i]) for i in non_1_indices]
    return strides1 == strides2
```

前面不能复用时，`copy_input` 不是一个含糊的“以后也许复制”：它立即建立读取 x loader 的同 shape/dtype Pointwise，并调用 realize，使返回值拥有新的 ComputedBuffer 身份。新 Pointwise 只读旧 x，不继承 MutationLayout；因此外部调用拿到的是独立逻辑结果，原 mutation 对象仍保留自己的写入身份。

```python
@staticmethod
def copy_input(x):
    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=x.get_size(),
        origin_node=x.get_origin_node(),
        traceback=x.get_traceback(),
    )
    pw.realize()
    return pw
```

“可能融合”表示调度器以后可以把这个读取旧 x、写新 buffer 的 Pointwise 与 producer 合并；在当前 IR 出口，它已经是另一份具名 ComputedBuffer，而不是原对象的别名。

普通 BaseView 能否零复制转换，取决于它的最终索引是否为仿射式 `dot(range_vars,strides)+offset`。转换函数先冻结底层 buffer，再从 view 的 `make_indexer()` 得到 index，求每个逻辑维的 stride 和常数 offset；若重新拼出的 expected 不等于原 index，抛 NotImplementedError，调用方随后 copy。成功时才建立同一 data 上的 FixedLayout ReinterpretView：

```python
@classmethod
def convert_to_reinterpret_view(cls, x):
    assert isinstance(x, BaseView)
    if isinstance(x, ReinterpretView):
        return x

    x_unwrap_view = x.unwrap_view()
    buf = V.graph.get_buffer(x_unwrap_view.get_name())
    assert buf is not None
    x_unwrap_view_fx_node = buf.get_origin_node()
    if (
        x_unwrap_view_fx_node is not None
        and "val" in x_unwrap_view_fx_node.meta
        and isinstance(x_unwrap_view.layout, FlexibleLayout)
        and (
            x_unwrap_view_fx_node.meta["val"].is_contiguous(
                memory_format=torch.channels_last
            )
            or x_unwrap_view_fx_node.meta["val"].is_contiguous(
                memory_format=torch.channels_last_3d
            )
        )
    ):
        x_unwrap_view.freeze_layout_with_same_order(
            make_channels_last_strides_for(x_unwrap_view.get_size())
        )
    else:
        x_unwrap_view.freeze_layout()

    index_args, var_ranges = dependencies.index_vars_squeeze(
        x.get_size(), prefix="r"
    )
    range_vars = index_args[0]
    index = x.make_indexer()(range_vars)
    index = V.graph.sizevars.simplify_with_ranges(index, var_ranges)
    strides = V.graph.sizevars.stride_vars(index, range_vars)
    offset = V.graph.sizevars.offset_var(index, range_vars)
    expected = sympy_dot(range_vars, strides) + offset
    if index != expected:
        log.debug(
            "convert_to_reinterpret_view failed: stride=%s offset=%s index=%s",
            strides,
            offset,
            index,
        )
        raise NotImplementedError

    return ReinterpretView(
        data=x.data,
        layout=FixedLayout(
            device=x.get_device(),
            dtype=x.get_dtype(),
            size=x.get_size(),
            stride=strides,
            offset=offset,
        ),
    )
```

底层布局不是一律按默认连续冻结：若来源 FX 节点的 fake value 显示 channels-last，函数用同顺序冻结；否则普通冻结。`index_vars_squeeze` 为 view 的逻辑 shape 建变量，`simplify_with_ranges` 在合法范围内化简；`stride_vars` 和 `offset_var` 分别抽取各维一次项系数与常数项。只有重新组合后与完整 index 严格相等，才证明固定 stride 足以表达该 view。

```text
view index = 3*r0 + r1 + 5
stride_vars → [3,1]，offset → 5
expected = 3*r0 + r1 + 5，与原式相等
→ ReinterpretView(data=x.data, FixedLayout(size, [3,1], offset=5))

view index = r0*r1
无法写成各维线性 stride 加常数 offset
→ 转换失败 → copy_input
```

转换函数在检查仿射等式之前已经调用 `freeze_layout*` 固定底层 buffer。若随后 `index != expected` 并抛 NotImplementedError，调用方会 copy，但这次底层冻结不会回滚。只有 NotImplementedError 被外层 `try` 捕获；`assert` 失败、buffer 查找失败等错误会直接向上传播，不会自动降级为 copy。

转换成功后，赋值语句先把传入 TensorBox 的 `x.data` 原地替换为 ReinterpretView，再递归调用 `require_stride_order/exact_strides` 检查该 view 自己的 FixedLayout。若递归发现 layout 不满足并返回 copy，函数返回的是新 TensorBox，但调用者传入的旧 x 仍已经变成 ReinterpretView；这是编译期包装副作用，不应概括成“copy 路径完全不改输入”。

NonOwningLayout 分支执行 `x = x.get_layout().view`，改变的是函数内局部变量。若该 view 满足合同，返回值就是承担 storage 合同的 view，未必与最初传入对象具有同一 Python 身份。这样 view 转换只证明“能用固定 stride 表达”，不保证它已经满足外部要求，也不保证所有复用分支都原样返回入口对象。

rank-0 的 order/exact 是空序列。入口断言使用 `is not None`，所以空序列合法；后续部分分支却用 `if order` / `if exact_strides` 判断。Flexible scalar 传 `exact_strides=[]` 时走 exact 冻结；已经 Fixed 的 scalar 因空列表为假，不命中“已满足”快速路径，可能复制一次。空 `order=[]` 对 scalar 没有可排序维，Flexible 路径最终普通冻结。正文按当前 truthiness 行为描述，不把空列表等同于参数缺失。

### 五个实际探针

实验 21 的 direct IR probes 得到：

| 输入 | 要求 | 结果 |
| --- | --- | --- |
| FlexibleLayout `[2,3]` | order `[0,1]` | 同一 TensorBox，原地冻结为 stride `[1,2]` |
| FixedLayout stride `[3,1]` | order `[1,0]` | 已满足，返回同一对象 |
| FixedLayout stride `[3,1]` | order `[0,1]` | 不满足，返回新 TensorBox，stride `[1,2]` |
| Fixed `[2,1,3]`, stride `[3,3,1]` | exact `[3,99,1]` | size-1 差异无地址意义，返回同一对象 |
| size `[0,3]` | 任意 exact `[99,7]` | numel=0，layout 无可观察元素，返回同一对象 |

FlexibleLayout 路径为何能原地冻结：这个具名逻辑 buffer 尚未承诺具体排列，运行时物理空间也还没有在此处分配；改变 layout 不会换掉 buffer 名和引用身份。FixedLayout 不匹配时不能原地改写；copy_input 建立一份满足消费者合同的新结果。源码注释说明这份 clone 可能继续融合进 producer，因此“走 copy 分支”也不等于最终必有独立复制 kernel。

MutationLayout 比较的不是包装本身，而是 `real_layout()`；若真实 layout 仍 flexible，当前源码直接断言失败。如果它已经 Fixed 且满足合同，保留 mutation 对原 storage 的写入身份。

Generic BaseView 分支先尝试转换为 ReinterpretView。这只改变坐标解释，不复制数据；递归后若该固定布局仍不满足目标 order/exact，仍会落入 copy。转换不支持或底层是 ExternKernelAlloc 时也直接走 copy，因此不能把“view”统一说成零复制。

### 用新情况检验理解

1. FlexibleLayout 当前 stride 已在 size-1 维表现出歧义但满足 order：使用当前 stride 的实际 order 冻结，避免无谓改变。
2. Fixed `[2,3]` stride `[3,1]` 请求 exact `[3,2]`：有效 size=3 维差异显著，必须 copy。
3. Generic view 能转换为 ReinterpretView，但其自然 order 不满足消费者：递归检查后仍 copy。
4. 输入为零元素：在任何 storage/view 分支之前返回，合同不触发复制。

下一叶子继续解释 template、fallback 和控制流怎样建立不走普通 Pointwise/Reduction 路径的 IR。
