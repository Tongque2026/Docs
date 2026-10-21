# CPU/Triton backend 核心源码：逻辑位置怎样成为目标 kernel

[返回 CPU/Triton 后端篇](../README.md) · [返回 Codegen](../../README.md) · [返回总览](../../../README.md)

Q42 从生成结果比较了 CPU 16+3 与 Triton block/mask。本篇进入叶子实现，沿产品声明顺序保留核心源码，再用相同 N=19 pointwise 和 2×128 reduction 推演状态。先从 CPU 的共同标量 kernel 开始；向量候选选择在 K06 打开。

## K01：CppKernel 先建立循环变量，再把索引变成指针访问

产品文件中四个目标方法的声明顺序是 `load→store→store_reduction→set_ranges`。执行时 `CppKernelProxy` 会先调用后声明的 `set_ranges`，再重放 LoopBody 触发前三者；声明顺序与运行顺序不是同一件事。

首段代码使用 `sympy.Expr` 表示符号索引；在 N=19 的案例中，`name="arg0_1"`、`index=x0`，而 `x0` 的取值由稍后 `set_ranges([19],[])` 建立。`self.args.input(name)` 把逻辑名映射为 C++ 参数名 `in_ptr0`；`rename_indexing(x0)` 换成本 kernel 的循环变量；`load` 最终返回代表 `in_ptr0[x0]` 的临时表达式。`store` 以同样过程把 `name="buf0"` 映射为 `out_ptr0` 并写出赋值行。

```python
def load(self, name: str, index: sympy.Expr):
    # 教学：args 为逻辑输入名分配/复用 C++ 参数名。
    var = self.args.input(name)
    # 教学：把 LoopBody 符号换成本 kernel 的 itervar，并做当前 kernel 的索引改写。
    index = self.rename_indexing(index)
    line = f"{var}[{cexpr_index(index)}]"
    csevar = self.cse.generate(self.loads, line)
    csevar.update_on_args("load", (self, name, index), {})
    return csevar

def store(self, name, index, value, mode=None):
    assert "buf" in name
    var = self.args.output(name)
    index = self.rename_indexing(index)
    if mode is None:
        line = f"{var}[{cexpr_index(index)}] = {value};"
    elif mode == "atomic_add":
        if not config.cpp.dynamic_threads and self.num_threads == 1:
            line = f"{var}[{cexpr_index(index)}] += {value};"
        else:
            dtype = V.graph.get_dtype(name)
            # mirroring static_cast<float>(...) in load:
            value = f"static_cast<{DTYPE_TO_CPP[dtype]}>({value})"
            line = f"atomic_add(&{var}[{cexpr_index(index)}], {value});"
    else:
        raise NotImplementedError(f"store mode={mode}")
    self.stores.writeline(DeferredLine(name, line))

def store_reduction(self, name, index, value):
    index = self.rename_indexing(index)
    var = self.args.output(name)
    self.reduction_suffix.writeline(
        DeferredLine(name, f"{var}[{cexpr_index(index)}] = {value};")
    )

def set_ranges(self, lengths, reduction_lengths):
    if self.call_ranges:
        assert self.call_ranges == tuple(lengths) + tuple(
            reduction_lengths
        ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
        assert self.reduction_depth == len(lengths)
    else:
        self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
        self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
        self.itervars = [
            sympy_index_symbol_with_prefix(SymT.XBLOCK, n)
            for n in range(len(self.ranges))
        ]
        self.reduction_depth = len(lengths)
    return (
        self.itervars[: self.reduction_depth],
        self.itervars[self.reduction_depth :],
    )
```

### pointwise N=19：一项范围产生一个 x 变量

第一次 `set_ranges((19,),())`：

```text
call_ranges=(19,)
ranges=[19]
itervars=[x0]
reduction_depth=1
return ([x0],[])
```

LoopBody 的逻辑索引原来写作自己的符号，例如 z0。重放时 Q411 已把它代成 x0；`load("arg0_1",x0)` 再调用 `rename_indexing`，在这个简单例中仍是 x0。`args.input` 给逻辑名分配形如 `in_ptr0` 的参数，于是候选表达为 `in_ptr0[x0]`。CSE 产生临时变量并用 `update_on_args` 记住它来自哪个 buffer 和已改写索引，供后续分析/复用。

普通 `store("buf0",x0,tmp,None)` 通过 `args.output` 得到 `out_ptr0`，把 `out_ptr0[x0] = tmp;` 包在 `DeferredLine("buf0",...)` 中。DeferredLine 让后续若 buf0 被删除，输出阶段可以跳过该行；这里登记语句不等于立即写进最终字符串。

同一 kernel 再次用相同 ranges 调用 `set_ranges` 时不会创建新 x0，只断言 ranges 与 reduction 深度一致并返回原变量。融合成员因此共享同一循环坐标；若第二项意外要求不同 `(20,)`，在这里 assert，而不是悄悄套用 x0。

### atomic_add 有单线程简化与真正原子两条出口

普通 mode 为 None。mode 为 `atomic_add` 时：

- 静态线程配置且 `num_threads==1`，没有并发写冲突，生成普通 `+=`；
- 动态线程或多线程时，先按输出 dtype 把 value 显式 cast，再生成 `atomic_add(&ptr[index],value)`；
- 其他 mode 直接 `NotImplementedError`。

因此“mode 被 Q311 捕获并一路保存”在这里得到实际消费者。是否需要硬件原子不是凭操作名字决定，而取决于 mode 与当前线程条件。

### 2×128 reduction：范围被切成输出轴和规约轴

调用 `set_ranges((2,),(128,))` 得到：

```text
call_ranges=(2,128)
itervars=[x0,x1]
reduction_depth=1
return ([x0],[x1])
```

load 索引 `128*x0+x1` 生成输入访问。规约累计语句由相邻 `reduction` 方法放在规约循环区域；本声明范围的 `store_reduction("buf0",x0,tmp_acc0)` 不写入普通 stores，而写到 `reduction_suffix`。这保证只有 x1 的 128 项累计完成后，才执行一次 `out_ptr0[x0]=tmp_acc0`。

对 x0=1、x1=5，线性输入位置是 `128*1+5=133`；store 仍写输出位置 1。若误把 store_reduction 放在普通 store 区，它会在每个 x1 迭代中反复写尚未完成的累计值，破坏区域语义。

### K01 理解验收

若 pointwise shape 改为 2×3，`set_ranges((2,3),())` 产生 `[x0,x1]` 且 reduction_depth=2；连续布局索引可为 `3*x0+x1`。若随后融合成员调用 `(6,),()`，元素总数虽相同，tuple 不同，当前实现仍 assert：CppKernel 要求共享的 loop range 结构相同，而不是只比较乘积。

若 atomic_add 输出是 float32 且 num_threads=4，生成 `atomic_add(&out_ptr[index], static_cast<float>(value))` 形式；不能使用单线程 `+=` 分支。若 mode 为 `atomic_max`，此方法明确失败。

K01 到此完成。K02 将打开 Triton range tree，建立 pid、block、x/r mask。

## K02：range tree 把逻辑范围分成 program offset、block lane 与 mask

`TritonKernel.__init__` 建好 `range_trees` 后立即调用：

先走 N=19 的逐点例：只有一棵 x tree，`prefix="x"`、`is_loop=False`、`grid_dim=0`、`tensor_dim=0`、`numel=19`，当前 XBLOCK=8，kernel 不在规约中。下面循环因此立刻为 x tree 生成 `xoffset=program_id(0)*8`、`xindex=xoffset+arange(0,8)` 和 `xmask=xindex<19`；规约 rbase 分支不进入。

```python
def codegen_range_tree(self):
    for tree in self.range_trees:
        # reduction indexing goes inside a loop
        if not tree.is_loop:
            self.iteration_ranges_codegen_header(tree, self.body)
    if self.inside_reduction and self.range_trees[-1].is_loop:
        # workaround for this issue:
        # https://gist.github.com/jansel/6527126f781559095c5531f98a4235a7
        self.body.writeline(
            f"rbase = {self.iteration_ranges_ranges_code(self.range_trees[-1])}"
        )
```

非 loop tree 的 header 立即写入 body；分块 reduction 的 r tree 是 loop，索引 header 要在规约循环内部生成，因此这里跳过，只预先写 `rbase=arange(...)`。这就是 pointwise x 范围与 non-persistent r 范围首先发生的阶段差异。

真正生成 header 的六个 helper 在同类后部，按产品顺序如下：

```python
def iteration_ranges_ranges_code(self, entry):
    assert entry.tensor_dim is not None
    size = self.indexing_size_str(entry.tensor_dim)
    index_dtype = self.index_dtype
    convert = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
    return f"tl.arange(0, {entry.prefix.upper()}BLOCK){size}{convert}"

def iteration_ranges_scalar_code(self, entry, value):
    index_dtype = self.index_dtype
    ndim = self.triton_tensor_ndim()
    size = [1] * ndim
    return f"tl.full({size}, {value}, {index_dtype})"

def iteration_ranges_get_pid(self, entry):
    assert entry.grid_dim is not None
    key = f"tl.program_id({entry.grid_dim})"
    # y_grid has a limit, so express it in terms of y and z in case of overflow.
    # z grid is only exercised when max_tiles == 3 (off by default).
    if (
        entry.grid_dim == 1
        and not entry.has_zdim
        and not V.graph.sizevars.statically_known_leq(entry.numel, get_max_y_grid())
    ):
        # For ynumel larger than max_ygrid, we need to use zdim.
        # For each z dimension, there are tl.num_programs(1) yblocks which is passed by grad(x,y,z).
        # So, we need to add tl.program_id(z) * tl.num_programs(y) *YBLOCK to get the correct yoffset.
        key = f"({key} + tl.program_id({entry.grid_dim + 1}) * tl.num_programs({entry.grid_dim}))"
    pid = entry.pid_cache.get(key, key)
    if self.index_dtype != "tl.int32":
        return f"{pid}.to({self.index_dtype})"
    return pid

def _has_constant_mask(self, tree: IterationRangesRoot):
    if not self.optimize_mask:
        return False
    if V.graph.sizevars.statically_known_equals(tree.numel, 1):  # type: ignore[arg-type]
        return True
    # Masks are superfluous if numel is a multiple of BLOCK
    # (We use the fact that BLOCK is required by triton to be a power of 2)
    if tree.prefix == "r" and self.persistent_reduction:
        max_block = self._get_persistent_RBLOCK(tree.numel)
    elif tree.prefix == "x" and self.no_x_dim:
        max_block = 1
    else:
        if tree.prefix.upper() not in TRITON_MAX_BLOCK:
            return False
        max_block = TRITON_MAX_BLOCK[tree.prefix.upper()]

    # Optional optimization: if block divides numel exactly, we will
    # never need to do a masked load to handle stragglers at the end.
    # It's faster to avoid masking at all.  But it is sound to always
    # mask.
    return V.graph.sizevars.statically_known_multiple_of(tree.numel, max_block)

def filter_masks(self, mask_vars):
    for tree in self.range_trees:
        if self._has_constant_mask(tree):
            mask_vars.discard(f"{tree.prefix}mask")

def iteration_ranges_codegen_header(self, entry, code):
    x = entry.prefix
    if entry.is_loop:
        code.writeline(f"{entry.name} = {x}offset + {x}base")
    elif entry.grid_dim is None:
        # no need to "{x}offset = "
        code.writeline(f"{entry.name} = {self.iteration_ranges_ranges_code(entry)}")
        code.writeline(f"{x}offset = 0")
    else:
        if entry.tensor_dim is not None:
            line = f"{x}offset + {self.iteration_ranges_ranges_code(entry)}"
        else:
            line = self.iteration_ranges_scalar_code(entry, f"{x}offset")
        code.writelines(
            [
                f"{x}offset = {self.iteration_ranges_get_pid(entry)} * {x.upper()}BLOCK",
                f"{entry.name} = {line}",
            ]
        )

    if self._has_constant_mask(entry):
        sizes = self.dense_size_str()
        code.writeline(f"{x}mask = tl.full({sizes}, True, tl.int1)")
    else:
        code.writeline(f"{x}mask = {entry.name} < {x}numel")
```

### N=19、XBLOCK=8 的具体 header

pointwise x root 有 grid_dim=0、tensor_dim 非空。假定 int32 index，helper 依次给出：

```text
xoffset = tl.program_id(0) * XBLOCK
xindex  = xoffset + tl.arange(0,XBLOCK)[形状标注]
xmask   = xindex < xnumel
```

pid=2 时 xoffset=16，lanes 是 16..23；xnumel=19，mask 为 `[T,T,T,F,F,F,F,F]`。`iteration_ranges_ranges_code` 根据 tensor_dim 追加 broadcast shape，使 x block 能与其他维对齐；index_dtype 不是 tl.int32 时，arange 与 pid 都显式 `.to(index_dtype)`。

若 entry 没有 tensor_dim，header 不制造 arange，而用 `tl.full([1,...,1],xoffset,index_dtype)` 表达 scalar program 坐标。若 grid_dim=None，则无需 program offset：直接以 arange 作为 entry，并设 offset=0。

### reduction loop 的 rbase/rindex

对 non-persistent 2×128 reduction，x tree 非 loop，先生成 x header；r tree 是 loop，`codegen_range_tree` 只写：

```text
rbase = tl.arange(0,RBLOCK)[r 维形状]
```

后续规约循环每轮设置 roffset，再由 header 的 loop 分支形成 `rindex=roffset+rbase` 和 `rmask=rindex<rnumel`。persistent reduction 的 r tree 不走这个分块 loop形态，可在外层直接建立覆盖范围的 block。

### mask 何时可变成全 True

`optimize_mask=False` 时绝不声称常量 mask。开启后，numel 静态等于 1，或静态证明 numel 是所选最大 block 的倍数，才返回 True。r persistent 使用实际 persistent RBLOCK，no-x-dim 的 x 使用 1，其他前缀必须存在于 TRITON_MAX_BLOCK。

常量并不是“删除 xmask 变量”：header 仍写一个相应形状的 `tl.full(...,True,tl.int1)`；`filter_masks` 从某次访存需要合取的 mask 名集合中移除它，避免生成无意义的 `& xmask`。若 N=19，它不是最大 block 的倍数，仍保留比较 mask。

### y grid 超限怎样借 z 维展开

普通 entry 使用 `tl.program_id(grid_dim)`。当 grid_dim=1、没有现成 zdim 且 y numel 静态不能证明不超过最大 y grid 时，逻辑 pid 改为：

```text
pid_y + pid_z * num_programs_y
```

再乘 YBLOCK 得 yoffset。`pid_cache` 可让相同 key 复用已登记表达；index dtype 非 int32 时最终转换。这个分支改变 program 坐标，不改变每个 block 内的 arange 规则。

### K02 理解验收

N=20、XBLOCK=8 时 pid2 候选 16..23，mask 前四项真；N=16 虽能被解释用 XBLOCK=8 整除，但 `_has_constant_mask` 按产品的最大 block 规则判断，不能仅凭这个教学 XBLOCK 就断言 mask 一定优化掉，必须看 tree 前缀对应的 max block 与静态倍数证明。

若 rnumel=130、RBLOCK=64 的 non-persistent 规约，第三轮 roffset=128，rindex=128..191，只有 128、129 有效；rbase 自始终是 0..63，变化来自 roffset。

K02 到此完成。K03 将分段打开完整 `TritonKernel.indexing`，追踪 SymPy index、mask、dense/indirect 与 block pointer 出口。


## K03：indexing 把一个 SymPy 地址变成 offset、mask 或 block pointer

第一次阅读固定普通连续访问：`index=xindex`，active tree 只有上述 x tree；`dense_indexing=False`、`override_mask=None`、`block_ptr=False`，没有间接临时变量，`_load_mask=None`，CSE 为空。`prepare_indexing` 不改表达式，free symbol 只有 xindex；函数生成 offset xindex 和 mask xmask。后文再分别把 index 换成常数、带 FloorDiv/ModularIndexing 的表达式或间接 tmp，并一次只打开 dense/override/block-pointer 中一个条件。

```python
def indexing(
    self,
    index: sympy.Expr,
    *,
    copy_shape=None,
    dense_indexing=False,
    override_mask=None,
    block_ptr=False,
):
    """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
    index = self.prepare_indexing(index)
    index_vars = index.free_symbols
    has_rindex = False

    mask_vars: OrderedSet[str] = OrderedSet()
    for var in index_vars:
        assert isinstance(var, sympy.Symbol)
        has_rindex = has_rindex or symbol_is_type(var, SymT.RINDEX)
        # 教学：调用方显式 mask 覆盖符号推导结果；load mask 再额外合取。
        if override_mask:
            pass
        elif symbol_is_type(var, SymT.TMP):
            # indirect indexing
            cse_var = self.cse.varname_map[var.name]
            mask_vars.update(cse_var.mask_vars)
        elif symbol_is_type(
            var,
            (
                SymT.UNBACKED_INT,
                SymT.SIZE,
                SymT.PRECOMPUTED_SIZE,
                SymT.INDEX,
                SymT.FLOAT,
                SymT.UNBACKED_FLOAT,
            ),
        ):
            pass
        else:
            # var is one of xN, yN or rN
            assert symbol_is_type(
                var, (SymT.RINDEX, SymT.XBLOCK, SymT.YBLOCK)
            ), var.name
            mask_vars.add(f"{var.name[0]}mask")

    need_dense = (
        config.triton.dense_indexing
        or dense_indexing
        or self._load_mask is not None
    ) and index != 0

    have_dense = True
    have_loop_vars = False
    dense_mask_vars: OrderedSet[str] = OrderedSet()

    for tree in self.active_range_trees():
        if index_vars.intersection(tree.var_list):
            have_loop_vars = True
        else:
            have_dense = False
        dense_mask_vars.add(f"{tree.prefix}mask")

    if (
        block_ptr
        and self.allow_block_ptr
        and config.triton.use_block_ptr
        and not override_mask
        and not self._load_mask
        and len(mask_vars - dense_mask_vars) == 0
        and not self.is_indirect_indexing(index)
        and have_loop_vars
        # workaround https://github.com/openai/triton/issues/2821
        and self.index_dtype == "tl.int32"
    ):

        def match_strided_block(
            index: sympy.Expr, range_tree: IterationRangesEntry
        ) -> Optional[BlockParameters]:
            """
                Matches expressions of the form:
                    idx = s * xindex

                This implies stride (s,), and shape (XBLOCK,).
                """
            symbol = range_tree.symbol()
            stride = sympy.Wild("stride", exclude=[symbol])
            m = index.match(symbol * stride)
            if m is None:
                return None

            return BlockParameters(
                shape=[range_tree.numel],
                block_shape=[self._get_block_size(range_tree)],
                strides=[m[stride]],
                offsets=[self._get_block_offset(range_tree)],
            )

        def match_mod_div_block(
            index: sympy.Expr, range_tree: IterationRangesEntry
        ) -> Optional[BlockParameters]:
            """
                Matches higher-dimensional blocks coming from FloorDiv and ModularIndexing.

                Example expression to match:
                   sN * ((rindex//(d1 * ... * d(N-1))))
                       + s1 * ModularIndexing(rindex, 1, d1)
                       + ...
                       + s(N-1) * ModularIndexing(rindex, d1 * ... * d(N-2), d(N-1))

                This iterates over a block of shape (dN, ..., d1) and stride
                (sN, ..., s1). (d1,...,d(N-1)) and (s1,...,sN) are
                wildcards that we match.

                Note that dN does not appear in the expression, but we solve for it
                using range tree numels and the other dims.
                """
            # Bound the possible number of dims. We use the following heuristics:
            # - At least one dim for each range tree node.
            # - At least one dim for every FloorDiv or ModularIndexing op.
            # - At least 2 dims to pattern match.
            num_dims = max(
                2,
                len(self.range_tree_nodes),
                (index.count(FloorDiv) + index.count(ModularIndexing)),
            )

            # Pattern match to find the strides and offset.
            index_var = range_tree.symbol()
            wild = functools.partial(sympy.Wild, exclude=[index_var])
            dims: List[sympy.Expr] = [
                wild(f"dim_mod{idx}") for idx in range(num_dims)
            ]
            strides: List[sympy.Expr] = [
                wild(f"stride_mod{idx}") for idx in range(num_dims)
            ]

            def get_slice_numels(dims: List[Any]) -> List[Any]:
                """
                    Compute the cumulative size of each dimension's slice.
                    This proceeds from the last dim up to the second.
                    """
                numels = [sympy.Integer(1)]
                for dim in dims[:0:-1]:
                    numel = dim * numels[0]
                    numels.insert(0, numel)
                return numels

            # The first dimension's index is computed by division.
            # The remaining are computed by modulo.
            slice_numels = get_slice_numels(dims[:num_dims])
            block_index_exprs = [FloorDiv(index_var, slice_numels[0])] + [
                ModularIndexing(index_var, numel, dim)
                for dim, numel in zip(dims[1:], slice_numels[1:])
            ]

            # Calculate a linear index from block indices.
            match_expr = sympy_dot(strides, block_index_exprs)

            # Pattern match.
            match = index.match(match_expr)
            if match is None:
                return None

            # Provide default values for unmatched dims and strides.
            for dim in dims[1:]:
                if dim not in match:
                    match[dim] = sympy.Integer(1)
            for stride in strides[1:]:
                if stride not in match:
                    match[stride] = sympy.Integer(0)

            sizevars = V.graph.sizevars

            def get_match(expr: sympy.Expr) -> sympy.Expr:
                return sizevars.lookup_precomputed_size(match[expr])

            # Replace wildcards with matched expressions.
            dims = [dims[0]] + [get_match(dim) for dim in dims[1:]]
            strides = [get_match(stride) for stride in strides]
            slice_numels = get_slice_numels(dims)
            block_index_exprs = [
                sympy_subs(expr, match) for expr in block_index_exprs
            ]

            # The leading dimension is not directly matched in our expression.
            # We solve for it by dividing the range tree numel by the product of
            # all other dimensions. We quit if they are not known to be divisible.
            assert (
                dims[0] not in match
            ), "Expected not to match the leading dimension!"
            if not sizevars.statically_known_multiple_of(
                range_tree.numel, slice_numels[0]
            ):
                return None
            dims[0] = range_tree.numel / slice_numels[0]

            # Check for applicable iteration range sizes.
            # When mapping a 1D block into an ND one, we need to know that
            # the number of elements is not changed. This means the slice numels of
            # the ND iteration range must evenly divide the length of the 1D block.
            # There are two cases where we can guarantee this:
            #  1. Numels are powers of 2. If numel == 2 ** n, and we know XBLOCK == 2 ** m,
            #     with n and m integers, then either numel is a multiple of XBLOCK, or numel
            #     is less than XBLOCK. (If numel is less than XBLOCK, we round up to 1 below.)
            #  2. Numels are multiples of the maximum possible block size.
            max_block = self._max_block_size(range_tree)
            if any(
                not sizevars.statically_known_multiple_of(numel, max_block)
                and not sizevars.statically_known_power_of_2(numel)
                for numel in slice_numels
            ):
                return None

            def identity(expr: sympy.Expr) -> sympy.Expr:
                return expr

            # Compute the ND block shape from the linear block size.
            # Use CielDiv to round leading dimensions up to 1.
            # Non-leading dimensions are clamped to the size of the iteration range,
            # while the leading dimension can exceed this to accomodate a larger
            # block size.
            linear_block_size = self._get_block_size(range_tree)
            block_shape: List[sympy.Expr] = [
                CeilDiv(linear_block_size, slice_numels[0])
            ] + [
                sympy.Min(CeilDiv(linear_block_size, numel), dim)
                for numel, dim in zip(slice_numels[1:], dims[1:])
            ]

            # Compute block offsets from {xyzr}offset and the matched expressions.
            block_offsets: List[sympy.Expr] = [
                sympy_subs(expr, {index_var: self._get_block_offset(range_tree)})
                for expr in block_index_exprs
            ]

            return BlockParameters(
                shape=dims,
                block_shape=block_shape,
                strides=strides,
                offsets=block_offsets,
            )

        def match_block_pointer_subexpr(
            expr: sympy.Expr, range_tree: IterationRangesEntry
        ) -> Optional[BlockParameters]:
            """
                Match a block indexing subexpression involving a single range tree.
                """
            for match_func in (
                match_strided_block,
                match_mod_div_block,
            ):
                match = match_func(expr, range_tree)
                if match is not None:
                    return match

            return None

        def match_block_pointer() -> Optional[BlockPtrOptions]:
            index_relative_to_xyr_index = sympy_subs(
                index, {v: t.expr for v, t in self.range_tree_nodes.items()}
            )
            range_trees = self.active_range_trees(reorder=True)

            # Match each range tree separately.
            range_symbols = {tree.symbol() for tree in range_trees}
            index_terms = sympy.Add.make_args(index_relative_to_xyr_index)
            block_params = BlockParameters()
            for tree in range_trees:
                # Partition the index into subexpressions pertaining to each range tree.
                # For example xindex * 5 + rindex * 3 is partitioned to
                # (xindex * 5, rindex * 3).
                symbol = tree.symbol()
                subexpr = sympy.Integer(0) + sum(
                    expr for expr in index_terms if symbol in expr.free_symbols
                )

                # Reject mixed terms, e.g. xindex * rindex.
                # NB: the zero expression is allowed, for broadcasting.
                if len(range_symbols.intersection(subexpr.free_symbols)) > 1:
                    return None

                # Match the subexpression for this range tree.
                params = match_block_pointer_subexpr(subexpr, tree)
                if params is None:
                    return None
                block_params += params

            # Collect leftover terms as a constant offset.
            offset = sum(
                expr
                for expr in index_terms
                if not range_symbols.intersection(expr.free_symbols)
            )

            # Form the block pointer.
            self.filter_masks(mask_vars)
            return BlockPtrOptions.create(
                params=block_params,
                constant_offset=offset,
                range_trees=range_trees,
                mask_vars=mask_vars,
            )

        # Return a block pointer, if indexing matches the pattern.
        options = match_block_pointer()
        if options is not None:
            return options

    expand_str = None
    index_str = self.index_to_str(index)
    if isinstance(index, sympy.Integer):
        expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
        index_str = f"tl.full({expand_str}, {index_str}, tl.int32)"
        return IndexingOptions(
            index_str, OrderedSet(), "None", expand_str, has_rindex, index
        )

    if need_dense and not have_dense:
        expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
        index_str = f"tl.broadcast_to({index_str}, {expand_str})"
        mask_vars = dense_mask_vars
    elif not have_loop_vars and copy_shape:
        index_str = f"tl.broadcast_to({index_str}, {copy_shape}.shape)"
        mask_vars = dense_mask_vars

    if override_mask:
        mask_vars = OrderedSet([override_mask])

    if self._load_mask:
        mask_vars.add(self._load_mask)

    self.filter_masks(mask_vars)

    mask_str = " & ".join(sorted(map(str, mask_vars))) if mask_vars else "None"
    return IndexingOptions(index_str, mask_vars, mask_str, expand_str, has_rindex, index)  # type: ignore[arg-type]
```

这段长实现可以按返回对象的三种形态读：成功匹配时返回 `BlockPtrOptions`；普通整数和普通符号索引返回 `IndexingOptions`。三者都回答“地址怎样写、要合取哪些 mask”，但 block pointer 还携带多维 shape/stride/offset。

### 第一段：从 index 的符号决定基础 mask

`prepare_indexing` 先做当前 kernel 的预处理，随后取 `free_symbols`。以 `index=xindex` 为例，符号类型是 XBLOCK，加入 `xmask`；`128*xindex+rindex` 同时加入 xmask、rmask，并把 `has_rindex=True`。

符号分类不同：

| 符号 | 处理 |
| --- | --- |
| TMP 间接索引 | 找到产生该 tmp 的 CSEVariable，继承它已有的 `mask_vars` |
| unbacked/size/precomputed/index/float | 本身不对应 range-tree lane，不直接加 x/r mask |
| xN/yN/rN | 加相应前缀的 `xmask/ymask/rmask` |
| 其他类型 | assert，禁止静默生成未知坐标 |

若调用方给了 `override_mask`，遍历符号时不推导这些 mask，函数末尾直接用 override 替换集合。`_load_mask` 则在末尾额外加入；两者的语义不是相同操作。

### 第二段：need_dense 与 have_dense 比较“需要覆盖几棵范围树”

`need_dense` 在全局 dense 配置、局部 `dense_indexing=True` 或存在 load mask 时成立，但常数零索引例外。函数遍历 `active_range_trees()`：index 若含每棵 tree 的任一 var，`have_dense` 才保持 True；任何一棵缺失都会置 False，同时收集所有 tree 的 mask 名。

例如 active trees 为 x、r，index=`xindex`：

```text
index_vars={xindex}
have_loop_vars=True
have_dense=False       # 缺 r tree
dense_mask_vars={xmask,rmask}
```

若调用要求 dense，普通 offset `xindex` 会 broadcast 到完整 x×r shape，并把 mask 扩成两者。若不要求 dense，它可保留只沿 x 变化的广播语义。

### 第三段：block pointer 是一条严格受限的优化路径

只有以下条件全部成立才尝试：调用方要求 block_ptr、kernel/配置允许、没有 override/load mask、当前 mask 没有超出 active tree masks、不是间接索引、至少使用一项循环变量、index dtype 为 int32。任一不满足都直接进入普通 offset 路径。

最简单匹配是 `index=s*xindex`。例如 shape N=19、index=`3*xindex`：

```text
shape       = [19]
block_shape = [XBLOCK]
strides     = [3]
offsets     = [xoffset]
```

更复杂的 `FloorDiv + ModularIndexing` 把一个线性 range 映射成多维 block。函数先用表达式中 floor/mod 数量和 range-tree 数估计维数，建立 wildcard dims/strides；匹配后为缺省维填 1、缺省 stride 填 0，再通过 precomputed-size 表换回实际表达式。

首维不会直接出现在 modulo 表达式里，所以由 `range_tree.numel / 其余维乘积` 解出；不能静态证明整除就拒绝。各 slice numel 还必须是最大 block 的倍数或 2 的幂，才可安全把 1D block reshape 成 ND block。最终 block_shape 由线性 block size 对各 slice 做 CeilDiv/Min，offset 则把 index var 换成对应 tree offset。

多个 active tree 时，先把加法项按 x/r/y 符号分区。若一项同时混合 `xindex*rindex`，不能独立构造两棵 block 维，立即拒绝。每棵树匹配成功后合并 `BlockParameters`，不含 range 符号的剩余项作为 constant offset；最后过滤静态恒真的 mask，并交 `BlockPtrOptions.create`。

因此 `5*xindex+3*rindex+2` 可被拆成 x stride 5、r stride 3、常量偏移 2；`xindex*rindex` 不可走这条路径。匹配失败不是 codegen 失败，只是回退普通 pointer offset。

### 第四段：普通 offset 的 constant、broadcast 和 mask 出口

block pointer 未返回时先生成 `index_str`：

- SymPy Integer 用 `tl.full(shape,value,tl.int32)` 扩为 tensor，mask 为空、字符串为 None；shape 优先采用 `copy_shape.shape`，否则采用 kernel dense shape。
- `need_dense and not have_dense` 时，把普通 index broadcast 到完整 dense shape，并使用所有 active tree masks。
- 没有 loop var 但给了 `copy_shape` 时，也 broadcast 到参照值 shape。
- override mask 最终完全替换推导集合；`_load_mask` 再追加；`filter_masks` 移除静态恒真的范围 mask。

最终 `mask_str` 按名字排序并用 ` & ` 合取；空集合返回字符串 `"None"`。`IndexingOptions` 同时保留原规范化 SymPy index、是否含 rindex 和 expand shape，供 load/store 后续分支判断。

### 三组具体状态

1. Pointwise `index=xindex`、N=19：`mask_vars={xmask}`，普通出口是 offset `xindex`、mask `xmask`、has_rindex=False。
2. Reduction `index=128*xindex+rindex`：mask 为 `rmask & xmask`，has_rindex=True；若两棵 tree 都 active，则 have_dense=True。
3. 间接 `index=tmp0+xindex`：tmp0 从其 CSE producer 继承 mask；函数标记为 indirect，不能走 block pointer；再与 xmask 合并。

### K03 理解验收

若 active trees 为 x/r，index 是常数 0，即使 dense 配置开启，`need_dense` 因 `index==0` 为 False；整数出口仍用 `tl.full` 形成 dense-shaped 零 tensor，但不要求 x/r mask。若 index 改为仅 xindex 且 `_load_mask=mask0`，need_dense=True、缺 r tree，于是 broadcast 到 dense shape，mask 至少含 xmask、rmask、mask0，随后静态恒真者才会被 filter 删除。

若 `override_mask="m"`，x/r 符号遍历不加范围 mask，末尾集合变成 `{m}`；若同时 `_load_mask="q"`，最终是 `m & q`。不能把 override 理解成“在原有 xmask 上再加 m”。

K03 到此完成。K04 将打开 `load/store/store_reduction`，查看这些 options 怎样变成 tl.load/tl.store、block pointer 或 atomic 语句。


## K04：load/store 消费 indexing 结果并选择普通、block pointer 或 atomic 路径

先代入 2×128 max 规约中的一次输入读取：`name="arg0_1"`，`index=128*xindex+rindex`，dtype=float32，当前位于 reduction loop，rindex/rmask 都存在；它不是间接索引、不是 inplace buffer，也没有 broadcast 或额外 `_load_mask`。`last_usage` 尚未到该输入，所以不使用 `evict_last`；`indexing(...,block_ptr=True)` 在这组普通地址上返回 offset 与 xmask/rmask，随后生成带默认 other 值的 `tl.load`。

```python
def load(self, name: str, index: sympy.Expr):
    # 教学：先取得 pointer 参数，再让 indexing 决定 offset/mask/block-ptr。
    var = self.args.input(name)
    indirect_indexing = self.is_indirect_indexing(index)
    original_index = index
    indexing = self.indexing(index, block_ptr=True)
    has_rindex = indexing.has_rindex()
    has_tmpmask = indexing.has_tmpmask()

    # Keep the variable in cache if were going to reuse it. Equiv., if any of the following hold
    #  1) We are doing broadcasting
    #  2) It is a non-coalesced load. The intuition is that if it's
    #  non-coalesced, we will likely load each element multiple times in
    #  practice.
    #  3) It will be used later and it won't be CSE'd. Equiv., if all the following hold
    #   3.1) We are in a reduction loop
    #   3.2) Its not its last use
    #   3.3) This load will not be lifted to the body
    #
    is_coalesced = any(
        i == 1 for i in self.get_strides_of_load(original_index).values()
    )
    if self.is_broadcasted(original_index):
        ep = ", eviction_policy='evict_last'"
    elif not is_coalesced:
        ep = ", eviction_policy='evict_last'"
    elif self.inside_reduction and self.range_trees[-1].is_loop:
        if name in self.args.inplace_buffers:
            names: OrderedSet[str] = OrderedSet(
                self.args.inplace_buffers[name].other_names
            )
        else:
            names = OrderedSet([name])
        last_use = len(names & self.last_usage) > 0
        evict_last = not last_use and (has_rindex or indirect_indexing)
        if evict_last:
            ep = ", eviction_policy='evict_last'"
        else:
            ep = ", eviction_policy='evict_first'"
    else:
        ep = ""

    if (has_tmpmask or has_rindex) and indexing.has_mask():
        if self._load_other:
            other = f", other={constant_repr(self._load_other)}"
        else:
            other = ", other=0.0"
    else:
        other = ""

    advance_block_ptr = None
    append_broadcast = None
    if should_unwrap_unspec_arg(name):
        line = var
    else:
        if isinstance(indexing, BlockPtrOptions):
            block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                name, var, indexing, other
            )
            line = f"tl.load({block_ptr}{other}{ep})"
            # add needed size=1 dimensions
            block_shape = [str(dim) for dim in indexing.block_shape]
            line = triton_reshape(line, block_shape, indexing.reshape_suffix)
        elif isinstance(original_index, sympy.Integer):
            line = f"tl.load({var} + ({original_index}))"
            append_broadcast = indexing.expand_str
        else:
            line = f"tl.load({var} + ({indexing.index_str}), {indexing.mask_str}{ep}{other})"

        dtype = V.graph.get_dtype(name)
        if (
            dtype in (torch.float16, torch.bfloat16)
            and config.triton.codegen_upcast_to_fp32
        ):
            line += ".to(tl.float32)"
        if dtype == torch.bool and torch.version.hip is None:
            # Workaround for https://github.com/openai/triton/issues/2151
            # tl.load returns int8 when loading from pointer to int1
            # NOTE: Currently causes hangs on bool UTs for ROCm
            line += ".to(tl.int1)"

    load_buffer = self.get_load_buffer(indexing)
    result_var = self.cse.generate(load_buffer, line)
    assert isinstance(result_var, TritonCSEVariable)
    result_var.mask_vars = indexing.mask_vars  # type: ignore[assignment]

    if append_broadcast:
        line = f"tl.broadcast_to({result_var}, {append_broadcast})"
        result_var = self.cse.generate(load_buffer, line)

    if advance_block_ptr:
        load_buffer.writeline(advance_block_ptr)

    if not self.inside_reduction or (not indexing.has_rmask() and not has_rindex):
        self.outside_loop_vars.add(result_var)

    return result_var

def store(
    self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
) -> None:
    # 教学：store 强制 dense indexing；普通写才允许 block pointer。
    var = self.args.output(name)
    original_index = index
    indexing = self.indexing(index, dense_indexing=True, block_ptr=mode is None)

    # Guard against write-after-read corruption in triton.
    # See # https://github.com/openai/triton/issues/1615
    # This triton bug means that a load which is broadcasted over multiple
    # warps may see the result of a store that happens later in the triton
    # program. The workaround is to add a barrier before storing, which
    # enforces that all warps have already read the data.
    is_inplace = name in self.args.inplace_buffers
    is_broadcasted = self.is_broadcasted(original_index)
    if is_inplace and is_broadcasted:
        self.stores.writeline(DeferredLine(name, "tl.debug_barrier()"))

    advance_block_ptr = None
    if isinstance(indexing, BlockPtrOptions):
        block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
            name, var, indexing
        )
        # block_ptr stores don't do implicit casting
        line = self.codegen_block_ptr_store_line(
            name, indexing, block_ptr, value, other
        )
    elif mode is None:
        line = f"tl.store({var} + ({indexing.index_str}), {value}, {indexing.mask_str})"
    elif mode == "atomic_add":
        line = f"tl.atomic_add({var} + ({indexing.index_str}), {value}, {indexing.mask_str}, sem='relaxed')"
    else:
        raise NotImplementedError(f"store mode={mode}")
    self.stores.writeline(DeferredLine(name, line))
    if advance_block_ptr:
        self.stores.writeline(advance_block_ptr)

    if not self.inside_reduction:
        self.outside_loop_vars.add(value)

def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
    # 教学：临时退出 reduction 状态，为规约结果生成外层输出索引。
    assert self.inside_reduction
    self.inside_reduction = False
    indexing = self.indexing(index, block_ptr=True)
    self.inside_reduction = True
    var = self.args.output(name)

    if isinstance(indexing, BlockPtrOptions):
        self.suffix.writeline(
            DeferredLine(
                name,
                self.codegen_block_ptr_store_line(
                    name,
                    indexing,
                    indexing.format(var),
                    value,
                    f", boundary_check={indexing.boundary_check()!r}",
                ),
            )
        )
    else:
        assert isinstance(indexing, IndexingOptions)
        self.suffix.writeline(
            DeferredLine(
                name,
                f"tl.store({var} + ({indexing.index_str}), {value}, {indexing.mask_str})",
            )
        )

```

这三个方法接到的不是一个裸字符串，而是 K03 返回的 `IndexingOptions` 或
`BlockPtrOptions`。它们要完成最后一次选择：读写是普通 pointer offset、block
pointer，还是原子写；需要哪些 mask；生成的语句应进入循环体、store 区还是
规约后的 suffix。

### `load`：地址、缓存策略和填充值共同决定一条读取语句

`load` 先保留 `original_index`，因为后面的合并访问判断和广播判断需要看原始
SymPy 表达式；随后调用 `indexing(index, block_ptr=True)`，取得已经整理好的
offset、mask 或 block pointer 参数。间接索引也在这里单独记下，因为它会影响
cache eviction 策略。

eviction policy 的判断顺序有实际意义：

1. 广播 load 会在多个 lane 复用同一值，使用 `evict_last`；
2. stride 中没有 1 的非合并访问也可能重复取数，使用 `evict_last`；
3. 非 persistent 的规约循环再根据 buffer 的全部别名、`last_usage`、是否含
   `rindex` 或间接索引，在 `evict_last` 与 `evict_first` 之间选择；
4. 其他普通读取不附加 eviction policy。

当索引含临时 mask 或 `rindex` 且确实存在 mask 时，越界 lane 必须获得一个
明确值。普通读取用 `0.0`；规约生成过程可以通过 `_load_other` 把它改成适合
组合操作的单位元，例如 max 规约常用负无穷，避免越界 lane 错误地赢过有效值。

接下来有四个实际出口：unspecialized scalar 参数直接使用参数变量；block pointer
调用 `codegen_block_ptr` 并在 load 后按需要 reshape；整数常量地址省去 mask
参数、读取后再 broadcast；普通 offset 生成带 mask、eviction 和 other 的
`tl.load`。float16/bfloat16 可以按配置提升到 float32，bool 在非 HIP 路径转为
`tl.int1`。生成结果进入 CSE，并继承 `indexing.mask_vars`；block pointer 若需要
移动，还会在读取后写入 advance 语句。

最后的 `outside_loop_vars` 判断决定该值能否留到规约循环外复用。完全不在规约中
的值可以提升；处在规约中但索引既不含 `rmask` 也不含 `rindex` 的值也可以提升。
例如按行 max 时，读取 `x[row, column]` 依赖 rindex，必须留在规约循环内；读取
只依赖 row 的缩放系数则可以在进入内层规约前取得一次。

### `store`：普通写允许 block pointer，原子写强制普通 offset

`store` 请求 `dense_indexing=True`，因为写入不能只产生一个广播后的局部形状。
只有 `mode is None` 时才允许 block pointer；`atomic_add` 必须保留显式地址与 mask，
从而生成：

```text
tl.atomic_add(pointer + offset, value, mask, sem='relaxed')
```

这解释了一个迁移条件：把普通 store 改为 `atomic_add` 后，即使同一索引原本能匹配
block pointer，也会改走普通 `IndexingOptions`。未知 mode 不猜测语义，直接抛出
`NotImplementedError`。

原地写且原始索引发生广播时，方法先向 store buffer 写入 `tl.debug_barrier()`。
这是为了保证各 warp 的旧读取先完成，再执行会覆盖同一存储的写入。随后普通写、
原子写或 block-pointer 写均包装为 `DeferredLine(name, line)`；这样 buffer 被移除时，
对应行还能由后续阶段一并跳过。block pointer 的 advance 紧跟 store。若当前不在
规约循环中，被写的 `value` 也登记为可在外层使用。

### `store_reduction`：先退出内层坐标，再把最终值写到输出坐标

规约体内的 `value` 已经组合了 r 维，但输出地址不应继续带 rindex。因此方法先
断言自己确实位于规约中，临时把 `inside_reduction` 设为 False，再调用
`indexing(index, block_ptr=True)` 生成外层输出索引，随后恢复 True。生成的写语句
进入 `suffix`，即规约循环结束后的位置；block pointer 路径显式附上 boundary
check，普通路径使用 offset 与 mask。

以 2×128 的按行 max 为例：规约体里读取地址含 `128*xindex+rindex`，无效 r lane
由 `_load_other=-inf` 填充；累计结束后，`store_reduction` 为输出只生成 x 维地址。
如果不临时退出 reduction 状态，索引器可能把 r 维 mask 或 dense 形状带进最终
两元素输出，这正是状态切换要避免的问题。

### K04 理解验收

- 把普通写改成 `atomic_add`：block pointer 被禁用，输出改为普通 offset、mask 和
  relaxed atomic；这是 `store` 在调用 `indexing` 前由 mode 决定的。
- 把 sum 规约改成 max：地址范围没有因此改变，但越界 load 的 `other` 必须从加法
  单位元改成 max 的单位元；否则尾部 lane 会改变规约答案。
- 在规约中读取一个只依赖 x 的系数：它没有 rindex/rmask，可以加入
  `outside_loop_vars`；读取输入元素仍依赖 rindex，不能提升。
- 规约最终 store 只按输出位置写一次，因为 `inside_reduction=False` 时生成的是
  外层输出地址，而语句放在 `suffix`，不在每个 r lane 中执行。

K04 到此完成。K05 继续打开 `codegen_body` 与 `codegen_kernel`，把上述分散的
loads、compute、stores 和 reduction suffix 拼成完整 Triton 函数。


## K05：从分段代码缓冲区组装完整 Triton kernel

### 完整源码：`codegen_body`

对同一个 2×128 规约，进入函数前五个缓冲区分别含：`indexing_code` 的 x/r 索引，`loads` 的输入读取，`compute` 的 max 累计，`stores` 为空，`suffix` 含最终输出 store；`inside_reduction=True`，r tree 是 loop。函数把前三段放进 roffset 循环，每轮清除只在该块有效的 CSE，循环结束后才拼 suffix。逐点 N=19 例没有 r loop，五段直接依次拼接一次。

```python
def codegen_body(self):
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        # 教学：空 buffer 不产生 body；否则按 pointwise/loop reduction 两条路径拼接。
        if not (
            self.indexing_code
            or self.loads
            or self.stores
            or self.compute
            or self.suffix
        ):
            return

        # 教学：loop reduction 把 indexing/load/compute/store 放进 r 分块循环。
        if self.inside_reduction and self.range_trees[-1].is_loop:
            self.body.writeline("for roffset in range(0, rnumel, RBLOCK):")
            with self.body.indent():
                # last range tree is always reduction
                self.iteration_ranges_codegen_header(self.range_trees[-1], self.body)
                self.body.splice(self.indexing_code)
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)

            # invalidate any caches that came from inside the reduction loop
            self.cse.invalidate(self.outside_loop_vars)
            self.range_trees[-1].cache_clear()
        else:
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        # 教学：suffix 永远在规约循环之后，再清空本轮暂存区。
        self.body.splice(self.suffix)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.suffix.clear()
```

### 完整源码：`codegen_kernel`

先走 N=19 逐点例：`numels=[19]`，size hint 向上取为 32；签名有输入 pointer arg0_1 和输出 pointer buf0，没有 mutation、workspace 或等于 1 的维度，range tree 只有 x，body 已由上一方法组成，metadata 使用 pointwise heuristic。预期出口是带 `@triton_heuristics.pointwise(size_hints=[32])`、两个 pointer 形参和 x block body 的完整函数。

```python
def codegen_kernel(self, name=None):
        code = IndentedBuffer()

        # 教学：先从各维 numel 形成 heuristic 的 2 次幂提示。
        size_hints = []
        for numel in self.numels:
            numel_hint = V.graph.sizevars.symbolic_hint(numel)
            if not isinstance(numel_hint, (int, sympy.Integer)):
                # This default heuristic hint was picked carefully: it is
                # large, to ensure that we don't shrink the block size (since
                # if you don't have many elements, it'd be wasteful to pick a
                # large block size).  Since we don't know how many elements we
                # might have, we should be OK with some inefficiency to make
                # sure we handle the large case well.  8192 is the largest
                # block size we support, so we pick that.
                #
                # If we have a better hint for unbacked SymInts (e.g., because
                # a user told us, or we are tracking upper bounds) we could
                # use that here.
                size_hint = 8192
            else:
                size_hint = next_power_of_2(int(numel_hint))
            size_hints.append(size_hint)

        if not self.inside_reduction:
            size_hints.pop()

        heuristics = self._get_heuristic()

        if name is None:
            code.splice(gen_common_triton_imports())

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        # 教学：参数定义与 signature 是 decorator metadata 和函数形参的共同来源。
        argdefs, _, signature, _ = self.args.python_argdefs()
        # maps actual expression to SizeArg if it is in sizevars replacements
        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                # mypy is unhappy about the sympy.Expr
                # type for the key of the dict below
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                    )

        # 教学：收集所有可观察写入，供 inductor metadata 标注副作用。
        mutated_args: OrderedSet[str] = OrderedSet()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                mutation in self.args.inplace_buffers
                and mutation not in V.graph.removed_buffers
                and mutation not in self.removed_buffers
            ):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])

        # workspace arguments are mutated, but are not marked as mutations in self.mutations
        # because their buffers are added during codegen, and aren't tracked during
        # lowering/scheduling. So we add them as mutated_args explicitly below.
        #
        # In the logic below, we only mark the workspaces a mutated if they are marked with
        # zero_fill: that's because, if we don't expect the buffer to be pre-filled with
        # zeros, then, although we still mutate the data, we don't care about those
        # mutations because we don't make any assumptions about the contents of the
        # workspace buffer.
        for argname, arg in zip(argdefs, signature):
            if isinstance(arg, WorkspaceArg) and arg.zero_fill:
                mutated_args.add(argname)

        mutated_args = sorted(mutated_args)

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype
        )
        triton_meta = {
            "signature": triton_meta_signature,
            "device": DeviceProperties.create(
                V.graph.scheduler.get_current_device_or_throw()
            ),
            "constants": {},
        }

        inductor_meta = {
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "no_x_dim": self.no_x_dim,
            "num_load": self.num_load,
            "num_reduction": self.num_reduction,
            **self.inductor_meta_common(),
        }

        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        # 教学：把 active range 的 numel 加入运行时参数与 Triton signature。
        for tree in self.active_range_trees():
            sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
            signature.append(sizearg)
            triton_meta_signature[len(argdefs)] = signature_of(
                sizearg, size_dtype=self.index_dtype
            )
            argdefs.append(f"{tree.prefix}numel")
            # constexpr version causes issues, see
            # https://github.com/pytorch/torchdynamo/pull/1362
            # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
            #     tree.numel
            # )
            # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
        triton_meta["configs"] = [config_of(signature)]

        # Triton compiler includes equal_to_1 args into constants even
        # when they are not constexpr. otherwise there may be a segfault
        # during launching the Inductor-compiled Triton kernel.
        # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
        # https://github.com/openai/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
        for arg_num in triton_meta["configs"][0].equal_to_1:  # type: ignore[index]
            triton_meta["constants"][arg_num] = 1  # type: ignore[index]

        self.triton_meta = triton_meta

        for tree in self.range_trees:
            if tree.prefix == "r" and self.persistent_reduction:
                # RBLOCK for persistent_reduction is defined in codegen_static_numels
                continue
            if tree.tensor_dim is None:
                continue
            argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        # 教学：形参和 BLOCK constexpr 确定后，才冻结并拼接 kernel body。
        self.codegen_body()

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        # 教学：规约与 pointwise 选择不同 heuristic decorator。
        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if len(signature) == 4:  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        # 教学：decorator 后输出函数签名，再写静态 numel、alias 与 body。
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
        )
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        return code.getvalue()
```

### `codegen_body`：先固定语句次序，再处理规约循环边界

LoopBody 重放期间，语句没有直接写进最终函数，而是分别积存在
`indexing_code`、`loads`、`compute`、`stores` 和 `suffix`。这个划分保证地址准备先于
读取、读取先于计算、计算先于写出。五个缓冲区全空时直接返回，避免产生空 body。

pointwise 或 persistent reduction 不需要由这里生成 r 分块循环，五段按上述顺序
直接拼入 `body`。普通 loop reduction 则生成：

```text
for roffset in range(0, rnumel, RBLOCK):
    本 r 块的 rindex/rmask
    indexing_code
    loads
    compute
    stores
suffix                         # 最终 reduction store 在循环外
```

每个 r 块结束后，来自循环内部的 CSE 缓存不都能跨到下一个块。`invalidate` 保留
`outside_loop_vars` 中与 r 无关的值，清除其余缓存；range tree 自己的缓存也清空。
最后清空五个暂存区，使下一次重放不会重复旧语句。

以 `x` 为 2×128 的按行规约为例，`xindex` 对应输出行，`roffset` 依次取
`0,RBLOCK,...`，header 由当前 roffset 形成 rindex/rmask；load/compute 在每一块执行，
最终两项规约值的 store 位于 suffix，所以每个输出行只在规约完成后写出。

### `codegen_kernel`：从范围和参数生成可被 wrapper 调用的函数

函数按六个连续阶段工作：

1. **尺寸提示。** 每个 numel 取符号 hint；静态值提升到下一个 2 的幂，无法获得
   整数 hint 时用 8192。pointwise 没有真正的 reduction 维，因此移除末项占位。
2. **参数与副作用。** `python_argdefs()` 同时交出函数形参和 signature。预计算尺寸
   换回原表达式；input/inplace/output mutation 与需要 zero-fill 的 workspace 汇总为
   `mutated_arg_names`，供运行和缓存元数据使用。
3. **Triton 与 Inductor metadata。** 前者保存 signature、device、常量与 config；
   后者保存 heuristic 提示、kernel 名、mutation、`no_x_dim` 和 load/reduction 计数。
   bandwidth 估计只在 benchmark/profile 配置下加入。
4. **范围参数。** 每棵 active tree 增加 `xnumel/rnumel/...` 运行时参数及 signature；
   `equal_to_1` 参数同步登记为常量。非 persistent 且有 tensor_dim 的树再增加
   `XBLOCK/RBLOCK/... : tl.constexpr`。persistent reduction 的 RBLOCK 稍后由静态
   numel 直接定义，因此不重复成为形参。
5. **函数装饰与正文。** 先调用 `codegen_body` 冻结缓冲区，再输出 helper。规约使用
   reduction heuristic 和 `reduction_hint`；pointwise 使用 pointwise heuristic，二维
   情形还选择 tile hint。函数体依次写静态 numel、参数 alias、最终 body。
6. **可选 benchmark。** 配置开启时追加构造参数与计时代码；正常 wrapper 使用的
   kernel 定义不依赖这一段。

### N=19 pointwise 的完整状态

假定一棵 x range tree，`xnumel=19`，运行配置选择 `XBLOCK=8`。生成定义的关键形态是：

```text
@triton_heuristics.pointwise(size_hints=[32], ...)
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    ...
    tl.store(out_ptr0 + xindex, value, xmask)
```

`size_hints=[32]` 是 heuristic 输入，不表示要写 32 项。pid=0、1、2 分别产生
0..7、8..15、16..23；最后一组只有 16、17、18 的 xmask 为真。wrapper 传入实际
`xnumel=19`，heuristic 决定 launch/grid 与 block 配置，kernel 内的 mask 保证边界。

### 条件改变时哪一段随之变化

- N 从 19 改成动态符号且没有整数 hint：size hint 变成 8192，但实际 `xnumel` 仍是
  函数参数，正确性仍由 index 与 mask 决定；8192 只参与配置选择。
- pointwise 改成 loop reduction：保留 r 维 size hint 和 `RBLOCK` 形参，decorator 改为
  reduction heuristic，body 多出 roffset 循环，suffix 仍在循环外。
- 改成 persistent reduction：仍是 reduction decorator，但 RBLOCK 由
  `codegen_static_numels` 定义，不列入函数形参，也不由 roffset 循环逐块推进。
- 某个 inplace buffer 已被删除：它不进入 `mutated_arg_names`；仍存活的同名 mutation
  才作为可观察副作用交给后续运行与缓存逻辑。

### K05 理解验收

若 N=17 且 XBLOCK=8，size hint 仍为 32，pid 仍需 0、1、2 三组；最后一组只有位置
16 有效。若将同一计算变为 r 维分块规约，变化不在普通 load/store 的相对顺序，
而在 `codegen_body` 把四段包入 roffset 循环、清理块内 CSE，并把 suffix 留在外面。
因此 decorator、形参和 body 结构共同表达执行策略，任何一项都不能只靠函数名推断。

K05 到此完成。K06 回到 CPU，完整打开 `CppKernelProxy` 怎样从 scalar 底稿选择
向量主循环、尾部和最终 loop nest，从而接回 Q42 的 16+3 实测结果。


## K06：CppKernelProxy 从 scalar 语义底稿生成 vector 主循环与尾部

### 完整源码：`codegen_functions`

本次 CPU 输入是一个 LoopBody callable，`fn_list=[fn]`、`var_sizes_list=[((19,),())]`；机器与 float32 支持 16 lane，tiling 轴为 x、factor=16，尾部配置允许标量 fallback。`set_ranges` 建立长度 19 的一维循环；scalar kernel 先形成正确底稿，vector checker 通过后生成 0..15 主段，剩余 16..18 用 scalar 尾段。若 checker 因 dtype 或操作失败，则直接保留覆盖 0..18 的 scalar loop。

```python
def codegen_functions(self, fn_list, var_sizes_list):
        assert len(fn_list) == len(var_sizes_list)
        kernel_group = self.kernel_group
        group, reduction_group = max(var_sizes_list, key=lambda sizes: len(sizes[1]))

        self.set_ranges(group, reduction_group)

        def codegen_kernel(cls, *args):
            with kernel_group.new_kernel(cls, *args) as kernel:
                # Ugly hack to maintain the metrics kernel count since
                # we only count in CppKernelProxy, not those contained in it
                metrics.generated_kernel_count -= 1

                run(kernel)
                return kernel

        def run(kernel):
            vars, reduction_vars = kernel.set_ranges(group, reduction_group)
            in_suffix = False
            for fn, var_sizes in zip(fn_list, var_sizes_list):
                if var_sizes in [
                    (group, reduction_group),
                    (tuple(itertools.chain(group, reduction_group)), ()),
                ]:
                    assert not in_suffix
                    fn(vars, reduction_vars)
                else:
                    in_suffix = True
                    assert var_sizes == (
                        group,
                        (),
                    ), f"unexpected group: {var_sizes} != {group}, {reduction_group}"
                    # we can fuse in some extra pointwise into the suffix
                    with kernel.write_to_suffix():
                        fn(vars, ())

        scalar_kernel = codegen_kernel(CppKernel)
        V.graph.removed_buffers |= scalar_kernel.removed_buffers
        V.graph.inplaced_to_remove |= scalar_kernel.inplaced_to_remove
        self.loop_nest = LoopNestWithSplit.build(scalar_kernel)

        if not self.picked_vec_isa:
            return

        if not self.itervars:
            # not a loop
            return

        # Kernels share the same global contexts like V.graph.wrapper_code, V.kernel.args.
        # But the generated scalar kernel has updated these global contexts. Hence, the other kernels
        # should not do this again to avoid context conflict. By now, we only control the
        # config.inplace_buffers. In the future, we could maintain more contexts.
        with torch._inductor.config.patch(inplace_buffers=False):
            tiling_select = TilingSelect()
            tiling_factors, tiling_indices = tiling_select.select_tiling(
                fn_list, var_sizes_list
            )
            assert len(tiling_factors) == len(tiling_indices)
            # <TODO> This should be removed after full support for vectorization is implemented.
            could_masked_vec = True
            all_dtypes = _get_dtype_from_loopbodies(_get_loop_body(fn_list))
            if any(dtype not in MASKED_VECTORIZABLE_DTYPES for dtype in all_dtypes):
                # can be removed after masked vectorizable dtype are same with vectorizable dtype
                could_masked_vec = False

            if len(tiling_indices) == 1:
                vec_kernel = codegen_kernel(
                    CppVecKernel, tiling_factors[0], tiling_indices[0]
                )
                metrics.generated_cpp_vec_kernel_count += 1
                main_loop, tail_loop = self.loop_nest.split_with_tiling(
                    tiling_indices[0], factor=tiling_factors[0]
                )
                main_loop.set_kernel(vec_kernel)
                main_loop.simd_vec = True
                if config.cpp.enable_loop_tail_vec and could_masked_vec:
                    tail_loop.steps = tail_loop.size - tail_loop.offset
                    masked_vec_kernel = codegen_kernel(
                        CppVecKernel,
                        tiling_factors[0],
                        tiling_indices[0],
                        tail_loop.steps,
                    )
                    tail_loop.set_kernel(masked_vec_kernel)
                    tail_loop.simd_vec = True
                else:
                    tail_loop.set_kernel(scalar_kernel)
                    tail_loop.simd_omp = True
                # We chop the loop into two cubes by the nelements - main loop and tail loop.
                # Regarding the main loop, it is straightforward that it could be vectorized with
                # nelements. But for the tail loop, it still could be vectorized. For example,
                # if the nelements is 8(256bits), then the tail loop still could be vectorized
                # as 4(128bits).
                tail_loop.simd_nelements = tiling_factors[0] // 2
            elif len(tiling_indices) == 2:
                assert (
                    tiling_indices[1] == len(self.itervars) - 1
                    and tiling_factors[0] == tiling_factors[1]
                )

                metrics.generated_cpp_vec_kernel_count += 2
                outer_main_loop, outer_tail_loop = self.loop_nest.split_with_tiling(
                    tiling_indices[0], factor=tiling_factors[0]
                )
                (
                    inner_main_loop,
                    inner_tail_loop,
                ) = outer_main_loop.split_with_tiling(
                    tiling_indices[1] - tiling_indices[0], factor=tiling_factors[0]
                )
                tile2d_kernel = codegen_kernel(
                    CppTile2DKernel, tiling_factors[0], tiling_indices
                )
                inner_main_loop.set_kernel(tile2d_kernel)

                if config.cpp.enable_loop_tail_vec and could_masked_vec:
                    (
                        inner_main_loop_of_outer_tail_loop,
                        inner_tail_loop_of_outer_tail_loop,
                    ) = outer_tail_loop.split_with_tiling(
                        tiling_indices[1] - tiling_indices[0], factor=tiling_factors[0]
                    )

                    for tail_loop in (
                        inner_tail_loop,
                        outer_tail_loop,
                        inner_tail_loop_of_outer_tail_loop,
                    ):
                        tail_loop.steps = tail_loop.size - tail_loop.offset

                    for tail_loop, inner_tail_size, outer_tail_size in (
                        (inner_tail_loop, inner_tail_loop.steps, None),
                        (
                            inner_main_loop_of_outer_tail_loop,
                            None,
                            outer_tail_loop.steps,
                        ),
                        (
                            inner_tail_loop_of_outer_tail_loop,
                            inner_tail_loop_of_outer_tail_loop.steps,
                            outer_tail_loop.steps,
                        ),
                    ):
                        masked_tile2d_kernel = codegen_kernel(
                            CppTile2DKernel,
                            tiling_factors[0],
                            tiling_indices,
                            inner_tail_size,
                            outer_tail_size,
                        )
                        tail_loop.set_kernel(masked_tile2d_kernel)
                else:
                    vec_kernel = codegen_kernel(
                        CppVecKernel, tiling_factors[0], tiling_indices[0]
                    )
                    inner_tail_loop.set_kernel(vec_kernel)

                    outer_tail_loop.set_kernel(scalar_kernel)
```

### 完整源码：`codegen_nodes`

```python
def codegen_nodes(self, nodes: List[SchedulerNode]):
        # Legalize BF16 node by adding to_dtype explicitly
        self.legalize_lowp_fp_dtype(nodes)
        self.data_type_propagation(nodes)
        assert len(nodes) >= 1

        def fn(node, *index_vars):
            node.decide_inplace_update()
            node.mark_run()
            if isinstance(V.kernel, NullKernelHandler):
                return node._body(*index_vars)
            else:
                return node.codegen(index_vars)

        fn_list = [functools.partial(fn, node) for node in nodes]

        if (
            isinstance(V.local_buffer_context, LocalBufferContext)
            and V.local_buffer_context.local_buffers
        ):

            def wrap_fn(fn):
                wrapped_fn = V.local_buffer_context.localize_function(
                    fn,
                )
                wrapped_fn.original_fn = fn
                return wrapped_fn

            fn_list = [wrap_fn(fn) for fn in fn_list]

        var_sizes_list = [node.group[1] for node in nodes]
        self.codegen_functions(fn_list, var_sizes_list)
```

### 完整源码：`codegen_loops`

```python
def codegen_loops(self, code, worksharing):
        self.codegen_loops_impl(self.loop_nest, code, worksharing)
```

`CppKernelProxy` 的关键做法是先建立一份 scalar 语义底稿，再让向量实现重放同一组
node。这样向量化选择改变每次处理多少元素，却不重新发明计算语义。

### `codegen_nodes`：把 SchedulerNode 变成可重放函数

入口先显式补齐低精度转换并传播 dtype；这些步骤发生在选择向量宽度之前，否则
vector kernel 不知道每个值实际需要哪一种 C++ 向量类型。每个 node 被包装成 `fn`：
先决定原地更新、标记 node 已运行，再依据当前 kernel handler 选择直接执行 LoopBody
或调用 node codegen。若存在 local buffer，所有函数再经同一 context 本地化，并保留
`original_fn` 供后续分析。最后从 `node.group[1]` 取得每项迭代/规约范围，交给
`codegen_functions`。

### `codegen_functions`：scalar 先成立，vector 再覆盖可处理的循环块

方法先取拥有最多规约维的 group 作为共同范围。内部 `run` 按原节点顺序重放：范围
等于完整 group 的函数写入主体；范围只有外层 group 的额外 pointwise 函数，在遇到
规约后写入 suffix。任何其他范围组合触发 assert，不会把不兼容循环强行融合。

随后总会生成 `CppKernel` scalar kernel，并用它建立 `LoopNestWithSplit`。它既是无
SIMD、无循环等出口的最终实现，也是所有 tail 无法向量化时的正确性后备。scalar
生成期间发现的 removed/inplaced buffer 同步回 graph。

有可用 ISA 且确实存在循环时，才在关闭重复 inplace 更新的配置中选择 tiling：

- 一个 tiling index：把循环切成整向量宽度的 main loop 和不足一宽的 tail。main
  使用 `CppVecKernel`；允许 masked tail 且所有 dtype 可支持时，tail 使用带实际
  `steps` 的 masked vector kernel，否则回退 scalar。`simd_nelements` 还允许编译器
  对尾部尝试半宽 SIMD。
- 两个 tiling index：要求第二个是最内层且两个 factor 相同。完整 tile 使用
  `CppTile2DKernel`；masked tail 开启时分别为“内层余数”“外层余数”“内外都有余数”
  生成带对应 tail size 的 tile kernel。否则内层 tail 用一维 vector，外层 tail
  用 scalar。
- 零个可选 index：保留先前 scalar loop nest；源码不进入任一向量分支。

方法本身不打印 C++。它把选中的 kernel 绑定到 loop nest 各块；最后
`codegen_loops` 才把这棵已经决定好的结构交给 `codegen_loops_impl` 输出。

### N=19 的 16+3 状态推演

实验 13 当前 CPU 选择 16 元素 SIMD。单一 tiling index 的状态是：

| 阶段 | 起止位置 | kernel | 处理元素 |
| --- | --- | --- | --- |
| scalar 底稿 | 0..18 | `CppKernel` | 建立完整正确语义，暂作 fallback |
| split 后 main | 0..15 | `CppVecKernel(factor=16)` | 一次处理 16 项 |
| split 后 tail | 16..18 | masked vector 或 scalar | `steps=3`，只处理 3 项 |

因此“16+3”不是两个 SchedulerNode，也不是先运行一遍 scalar 再运行 vector。scalar
kernel 是生成期底稿和后备；最终 loop nest 的 main 位置绑定 vector kernel，tail
位置绑定实际选中的 masked-vector/scalar kernel。打印阶段只输出这棵最终 loop nest。

若 N 从 19 改为 32，split 的 tail 为空，两个完整宽度均由 main vector loop 处理；
若 dtype 不在 `MASKED_VECTORIZABLE_DTYPES`，main 仍可向量化，但 3 元素 tail 回退
scalar。若根本没有循环变量（单个 scalar 结果），在生成 scalar kernel 后立即返回，
不会因为机器支持 SIMD 而制造一层虚假循环。

### 2D tiling 的四块覆盖

假设两个可切维度都是 18，factor=16。第一次按外层切成 16+2；外层 main 再按内层
切成 16+2，形成完整 16×16 tile 和 16×2 内尾。开启 masked tail 时，外层 2 行也
按内层切开，得到 2×16 外尾和 2×2 双尾。三种 tail kernel 分别收到：

```text
16×2：inner_tail_size=2, outer_tail_size=None
2×16：inner_tail_size=None, outer_tail_size=2
2×2 ：inner_tail_size=2, outer_tail_size=2
```

四块正好覆盖 18×18，没有重复或遗漏。关闭 masked tail 时，16×2 用一维 vector，
包含外层余数的部分整体回退 scalar；这体现的是可用实现边界，不改变原 node 语义。

### K06 理解验收

若 N=35、factor=16，main 覆盖 0..31，tail 的 `steps=3` 覆盖 32..34；若 masked dtype
不受支持，仅 tail 的 kernel 选择改变，main 的两个完整 SIMD 块仍保留。若融合组在
规约后还有同外层范围的 pointwise，它由 `write_to_suffix()` 放到规约 suffix；范围
既不等于完整 group、也不等于纯外层 group 时触发断言，因为现有 loop nest 无法在
不改变坐标含义的情况下安放它。

至此，CPU 路径从 SchedulerNode 重放、scalar 语义底稿、向量/尾部选择走到最终
loop nest；Triton 路径从 range tree、indexing、访存缓冲区走到完整 decorated kernel。
两者都消费 Q411 的 LoopBody，但 CPU 把并行宽度固化进 C++ loop nest，Triton 把范围
参数与 block hint 交给 decorator 和运行 launch 配置。
