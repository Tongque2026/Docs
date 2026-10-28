# GraphLowering run_node：让单个节点满足整张图的约束

[返回 GraphLowering](../README.md) · [上一篇：输出合同](../03-output-source/README.md) · [返回 Lowering](../../README.md)

`call_function` 能选择一个 lowering，却没有处理这项结果与整张图之间的关系。例如某个外部调用只接受特定 layout，GraphLowering 可能先复制输入再调用；若该操作会原地写入，修改不能只停留在复制品。`run_node` 正是在每个 FX node 外围补齐这类图级约束。

本叶子声明范围为当前 `torch/_inductor/graph.py` 中完整的 `propagate_mutation` 和 `run_node`。源码较长，正文按原顺序组织为连续学习单元；所有单元合起来才构成完整叶子。

## 本实现的连续状态

```text
进入 node
  → 记录处理前的 buffer/operation 列表长度与 origin
  → 特殊分派：unsupported fallback / layout constraint / symbolic magic
  → 输出与 as_strided 的 stride 合同
  → 多用户、读次数和表达式大小触发物化
  → origin_node 与 buffer readers 登记
  → 找出本节点新定义的 unbacked symbols，生成可执行断言
  → 返回 result 给 FX Interpreter env
```

第一步要先解决 layout constraint 带来的 mutation 问题。

## K01：复制参数后，把写入传播回原参数

设某个可写操作收到原参数 `old_x`，但后端只接受特定排列。layout constraint 建立满足要求的 `new_x`，实际 lowering 对 `new_x` 原地写入。若到这里结束，调用者和后续节点仍读 `old_x`，看不到修改。

`propagate_mutation` 同时收到修改前后的 positional args 和 kwargs。它不根据 Python 函数名猜测谁可写，而是读取 ATen `OpOverload` 的 schema：只有 `alias_info.is_write` 的参数才需要把 `new_arg` 复制回 `old_arg`。

### 完整源码

进入这段代码时，`fx_node: torch.fx.Node` 是当前可能写入参数的 FX 节点；`old_args` 指向 layout 转换前的原 IR，例如 `old_x`；`new_args` 指向为外部调用准备的连续副本，例如 `new_x`。参数 schema 的 `alias_info.is_write=True` 表示这一个位置会被写入。循环找到该位置后，`mutated_arg.copy_(new_arg)` 把 new_x 的修改记录传播回 old_x；只读参数不会执行这一步。

```python
def propagate_mutation(
    self,
    fx_node: torch.fx.Node,
    old_args: Tuple[Any],
    old_kwargs: Dict[str, Any],
    new_args: Tuple[Any],
    new_kwargs: Dict[str, Any],
) -> None:
    """Propagate mutations on new_args/new_kwargs back to old_args/old_kwargs.

        Assumes we may have cloned old_args/old_kwargs into new_args/new_kwargs
        and then called fx_node(*new_args, **new_kwargs).

        If fx_node mutates any of new_args/new_kwargs, and they are different from
        old_args/old_kwargs, then we need to update the original tensor.
        """
    # 教学：schema 只存在于具体 ATen overload；参数数量必须保持一一对应。
    assert isinstance(fx_node.target, torch._ops.OpOverload)
    assert len(old_args) == len(new_args)
    assert len(old_kwargs) == len(new_kwargs)

    def maybe_propagate(
        schema_arg: torch._C.Argument, old_arg: ir.IRNode, new_arg: ir.IRNode
    ) -> None:
        # 教学：layout constraint 没替换这项对象时，写入已经落在原参数上。
        if old_arg is new_arg:
            return
        # 教学：对象不同还不够；只有 schema 声明为写参数才需要传播。
        if schema_arg.alias_info is not None and schema_arg.alias_info.is_write:
            # The lowering for copy_ is smart enough to "replace" old_arg with
            # new_arg in all future uses so a copy_ kernel never gets emitted.
            # 教学：调用 copy_ lowering，让后续对 old_arg 的引用切换到 new_arg 当前值。
            self.call_function(torch.ops.aten.copy_.default, (old_arg, new_arg), {})

    # 教学：位置参数按 schema.arguments 的同一索引配对。
    schema = fx_node.target._schema
    for idx, (old_arg, new_arg) in enumerate(zip(old_args, new_args)):
        schema_arg = schema.arguments[idx]
        maybe_propagate(schema_arg, old_arg, new_arg)

    # 教学：关键字参数先以 schema 名字建表，再按原 kwargs 的 key 配对。
    schema_kwargs = {arg.name: arg for arg in schema.arguments}

    for key in old_kwargs.keys():
        old_arg = old_kwargs[key]
        new_arg = new_kwargs[key]
        schema_arg = schema_kwargs[key]
        maybe_propagate(schema_arg, old_arg, new_arg)
```

源码副本去掉类级四空格缩进，方法内部完整保留。

### 三种参数逐项判断

假设教学 schema 有三个参数：

```text
dst: Tensor(a!)    # alias_info.is_write = True
src: Tensor        # 不可写
alpha: Scalar      # 无 alias_info
```

layout constraint 得到：

```text
old_args = (old_dst, old_src, 2)
new_args = (new_dst, new_src, 2)
```

| idx | 身份比较 | schema 可写 | 动作 |
| ---: | --- | --- | --- |
| 0 dst | `old_dst is not new_dst` | 是 | 调用 `copy_(old_dst,new_dst)` lowering |
| 1 src | `old_src is not new_src` | 否 | 不传播；它只是只读布局副本 |
| 2 alpha | 同一 Python 标量对象或值 | 无 | 不传播 |

关键字参数执行同一判断，只是用参数名取得 schema 项。若 `dst` 没被 constraint 替换，`old_dst is new_dst`，原操作已经直接写在旧对象上，第一条 return 防止重复 copy。

### “调用 copy_”不等于一定生成 copy kernel

源码注释指出，当前 `copy_` lowering 会聪明地把 `old_arg` 的未来使用替换为 `new_arg`，因此这里登记语义写回，不必据此断言最终程序一定出现独立 copy kernel。是否需要实际搬运还取决于别名、存储与后续生成安排。

这一步发生在 `run_node` 的 layout-constraint 分支中：先以新参数调用真实 target，得到 result；再调用本 helper 修复所有可写参数。result 的输出 stride、物化和 reader 登记仍由后续连续代码处理。

### 用新情况检验理解

1. constraint 只替换只读 src：不会调用 copy_，因为 schema 不可写。
2. dst 可写但 constraint 原样返回同一对象：不会调用 copy_，因为修改已作用于 old_dst。
3. dst 作为 kwarg 传入：通过 `schema_kwargs["dst"]` 得到同一写属性，处理结果不变。
4. 新旧 kwargs 数量不同：在读取 schema 前就触发断言，因为无法保证一一对应。

实验 20 直接把当前 `aten.copy_.default` schema 交给这个 helper：同时替换可写 `self` 与只读 `src` 时，只记录到一次 `copy_(old_dst,new_dst)`；可写 `self` 保持同一对象时，调用数为 0。这里用 mock 记录 `self.call_function` 的参数，因此它证明 helper 的 schema 分支和交接参数，不证明最终会生成 copy kernel。

## K02：建立本节点上下文并选择初始 result

`run_node` 先记录“进入本节点前已有多少 buffer 和 operation”。这两个数字不参与当前 target 的计算，而是在函数后半段切出“本节点刚刚新增的对象”，从中寻找新定义的 unbacked symbols。

call_function 节点还要先从 FX Interpreter env 取得 IR 参数，并把参数已有的 origins 与当前 FX node 合并。这样 `a=x+y; b=relu(a)` 处理 b 时，新 IR 的来源不仅有 b，也能带上 a/x/y 经已有 IR 汇集的来源。

分派期间同时安装三份“当前节点”状态，分别服务不同读取方式：

| 上下文 | 临时保存的内容 | 本篇中的直接消费者 |
| --- | --- | --- |
| `IRNode.current_origins(origins)` | 旧 origin 集合与本节点 `origins` 的并集 | 上下文内新建的每个 IRNode 在 `__post_init__` 中复制它，形成自己的 `origins` |
| `self.set_current_node(n)` | `GraphLowering.current_node = n` | placeholder 的 users 判断、layout/lowering 辅助代码以及本类中通过 `V.graph.current_node` 读取当前 FX 节点的路径；`V.graph` 此时就是本 GraphLowering |
| `V.set_current_node(n)` | 虚拟化的 `V.current_node = n` | 不持有 GraphLowering 引用的 lowering/IR helper，例如读取当前 node 的 users、meta 和 unbacked binding |

三者进入时都先保存旧值。若 lowering 内部再解释子区域，内层退出后恢复外层节点；本次 `run_node` 的 `with` 退出后再恢复进入本节点以前的值。它们保存的是同一个 FX node，但读取接口和消费者不同，因此源码同时设置三处。

### 完整源码：run_node 开头至初始 result

下面是 `run_node` 的第一连续段，下一段紧接产品源码中的 stride 注释。先代入普通 add 节点 n：`n.target=aten.add.Tensor`，env 已把两个 placeholder 映成 TensorBox，进入前 `len(buffers)=0`、`len(operations)=0`，所以两个 watermark 都是 0。`fetch_args_kwargs_from_env` 得到这两个 TensorBox；`super().run_node(n)` 调用 add lowering，返回尚未物化的 Pointwise TensorBox，此时全图列表仍为空。

```python
def run_node(self, n: torch.fx.Node) -> object:
    def debug(msg: str) -> None:
        # 教学：延迟格式化 FX node，仅在 debug 日志实际启用时付出格式化开销。
        log.debug("lowering %s %s", LazyString(n.format_node), msg)

    # 教学：保存进入节点前的长度；后面用切片只看本节点新增对象。
    buffer_watermark = len(self.buffers)
    operation_watermark = len(self.operations)

    # 教学：当前 FX node 本身一定是新 IR 的直接来源。
    origins = {n}
    is_call_function = n.op == "call_function"
    if is_call_function:
        # 教学：把 args/kwargs 中的 FX Node 引用替换为 env 里的 IR 结果。
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        # 教学：再收集输入 IR 已携带的来源，形成传递性的 origin 集合。
        origins |= gather_origins(args, kwargs)
    # 教学：三个上下文分别服务 IRNode 构造、GraphLowering 实例读取，
    # 教学：以及只通过虚拟化接口读取 V.current_node 的 helper；退出后恢复旧值。
    with ir.IRNode.current_origins(origins), self.set_current_node(  # type: ignore[arg-type]
        n
    ), V.set_current_node(
        n
    ):
        # 教学：优先处理 dtype/device/layout 等类型组合不受原生 lowering 支持的节点。
        if (
            n.op == "call_function"
            and n.target is not operator.getitem
            and fallback_node_due_to_unsupported_type(n)
        ):
            debug("fallback_handler")
            result = fallback_handler(n.target, add_to_fallback_set=False)(
                *args, **kwargs  # type: ignore[possibly-undefined]
            )
        # 教学：第二优先级是 target 注册的 layout constraint；它可替换输入。
        elif n.op == "call_function" and (
            layout_constraints := maybe_layout_constraints(n.target)  # type: ignore[arg-type]
        ):
            debug("layout_constraints")
            old_args = args  # type: ignore[possibly-undefined]
            old_kwargs = kwargs  # type: ignore[possibly-undefined]
            args, kwargs = layout_constraints(n, *args, **kwargs)  # type: ignore[index]
            # 教学：用满足 layout 要求的新参数进入 Q212 已讲的统一算子分派。
            result = self.call_function(n.target, args, kwargs)  # type: ignore[arg-type]
            # layout_constraints are allowed to make new copies of the inputs.
            # if they do, and if the target is mutable, then we need to
            # write the new values back into the original inputs.
            # 教学：K01 根据 schema 把新副本上的写入传播回旧参数。
            self.propagate_mutation(n, old_args, old_kwargs, args, kwargs)  # type: ignore[possibly-undefined]
        elif is_magic_method(n.target):
            # TODO: this is sus, it probably should be handled in the
            # lowerings themselves similarly to sym_size/sym-stride
            # https://github.com/pytorch/pytorch/issues/127789
            debug("is_magic_method")
            if isinstance(
                n.meta["val"], (torch.SymInt, torch.SymFloat, torch.SymBool)
            ):
                # 教学：符号标量的 magic method 直接交出底层 SymPy 表达式。
                result = n.meta["val"].node.expr
            else:
                # 教学：不是符号标量时仍让父 Interpreter 按节点 op 分派。
                result = super().run_node(n)
        else:
            debug("")
            # 教学：普通 placeholder/get_attr/call_function/output 都走父类分派，
            # 教学：父类再调用 Q211–Q213 已解释的对应覆写方法。
            result = super().run_node(n)
```

这段故意没有关闭 `with`：产品源码中的 stride、物化判断仍在同一上下文内，K03 会原顺序接上。直到那些处理结束，IR origin 集合、`self.current_node` 和虚拟化的 `V.current_node` 才分别恢复进入前的值。`V.graph.current_node` 只是经当前 GraphLowering 读取第二项，不是第四份独立状态。

### 四条分派怎样决定 result

| 节点条件 | 使用的参数 | 产生 result 的路径 | 后续仍执行 stride/物化吗 |
| --- | --- | --- | --- |
| unsupported type 的 call_function，且非 getitem | env 解引用后的 args | `fallback_handler(...)(*args,**kwargs)` | 是 |
| target 有 layout constraint | constraint 可能替换后的 args | `self.call_function`，再传播 mutation | 是 |
| target 是 magic method，meta 为符号标量 | 读取 `meta["val"]` | 直接取 `.node.expr` | 是，但 tensor 专属条件通常不命中 |
| 其他节点 | 由父 Interpreter 按 op 处理 | placeholder/get_attr/call_function/output | 是 |

分支有明确优先级。一个 call_function 若已经因 unsupported type 进入 fallback，不再执行其 layout constraint。magic method 也排在 layout constraint 之后。`operator.getitem` 被第一条显式排除：Python 容器取项或多输出取项应走相应正常分派，而不是仅因类型检查被这里强制 fallback。

### 具体状态：普通 add

假设进入 add 前：

```text
len(buffers)=0
len(operations)=0
env[x]=TensorBox(InputBuffer x)
env[y]=TensorBox(InputBuffer y)
```

执行开头后：

```text
buffer_watermark=0
operation_watermark=0
args=(env[x], env[y])
origins={add_node} ∪ origins(x_ir) ∪ origins(y_ir)
```

add 不需 unsupported fallback，没有专用 layout constraint，也不是 magic method，于是进入 `super().run_node(n)`；FX Interpreter 再调用 GraphLowering 的 `call_function`，得到延迟 Pointwise TensorBox。此时水位仍是 0。后半段若该 node 物化并新增 buf0/op0，切片 `buffers[0:]`、`operations[0:]` 就只包含 add 本次新增的对象。

### 具体状态：layout constrained 可写操作

假设 env 参数是 `old_dst`，constraint 为满足外部实现要求建立 `new_dst`：

```text
进入 constraint 前：args=(old_dst, src)
constraint 后：      args=(new_dst, src)
call_function：      target(new_dst,src) → result
propagate_mutation： schema 说明 dst 可写 → copy_ lowering(old_dst,new_dst)
```

注意 result 与参数写回是两项状态：result 继续代表该 FX node 的返回值；`propagate_mutation` 修复 old_dst 的后续语义。随后 K03 的 stride/物化逻辑处理 result，不会用 mutation helper 的返回值，因为它返回 None。

### 用新情况检验理解

1. 一个 getitem 节点触发 `fallback_node_due_to_unsupported_type`：第一分支仍不进入，因为 target 是 getitem。
2. 普通 placeholder 没有执行 `fetch_args_kwargs_from_env`；它由父 Interpreter 从运行参数顺序取得 example input。
3. magic method 的 meta 是普通 tensor 而非 SymInt：进入 magic 分支，但 result 来自父 Interpreter，而不是 `.node.expr`。
4. add 的 lowering 新增两个 buffer，而此前已有三个：水位为 3，后面只检查 `buffers[3:]`，不会把旧 buffer 的符号误算成本节点新定义。

## K03：先满足后继要求的 stride，再决定是否继续内联

这一段仍在 K02 建立的三个 current-node 上下文内。它先判断 result 是否将被直接返回、是否将成为 `as_strided/resize` 一类操作的输入，再依据 FX meta 强制 stride order 或 exact strides。随后才处理另一类问题：多消费者反复展开同一规则是否代价过高，或表达式本身是否已经太大。

### 完整源码：stride 与物化连续段

继续代入这个 add：结果 shape `[2,3]`，它既不是 graph output，也不被 as_strided/stride-sensitive op 使用，用户只有后继 relu；配置没有 `inductor_realize_to_strides`，result 仍是 Pointwise。对照例把 relu 结果同时交给 mm 和 output：`is_output=True`、`num_users=2`。mm 属于 `needs_realized_inputs`，先给出物化 hint 并固定 stride order；随后 output 分支发现底层仍是 Pointwise/Reduction 时才强制 `realize()`。因此这个案例最终形成 ComputedBuffer，不能只把原因概括成“mm 需要真实输入”。

`needs_realized_inputs` 是 lowering 模块导入时建立的 target 集合。`add_needs_realized_inputs(...)` 把 as_strided、mm、convolution 等要求接收存储形式输入的 overload（packet 会展开为各 overload）加入集合。`run_node` 只读取这张集合：在多用户情形遇到这些消费者时调用 `realize_hint()`；集合名称不代表此处无条件执行强制 `realize()`。

```python
        # require the same stride order for dense outputs,
        # 1. user-land view() will not throw because inductor
        # output different strides than eager
        # long term the solution is to make view() always succeed
        # with infallible strides.
        # 2: as_strided ops, we need make sure its input has same size/stride with
        # eager model to align with eager behavior.
        # 教学：这些操作会让 stride 成为用户可观察语义，先列出消费者集合。
        as_strided_ops = [
            torch.ops.aten.as_strided.default,
            torch.ops.aten.as_strided_.default,
            torch.ops.aten.as_strided_scatter.default,
            torch.ops.aten.resize.default,
            torch.ops.aten.resize_as.default,
        ]
        is_output = any(user.op == "output" for user in n.users)
        is_input_for_as_strided = any(
            user.target in as_strided_ops for user in n.users
        )

        # 教学：显式 meta 标记要求先物化，再尽量符合指定 stride order。
        if n.meta.get("inductor_realize_to_strides", False) and isinstance(
            result, TensorBox
        ):
            result.realize()
            strides = n.meta["val"].stride()
            sym_strides = torch._inductor.utils.any_is_symbolic(*strides)
            if (
                not hasattr(result, "get_stride")
                or result.get_stride() != strides
                and not sym_strides
            ):
                stride_order = ir.get_stride_order(strides)
                result = ir.ExternKernel.require_stride_order(result, stride_order)
        if (
            is_output
            and isinstance(result, TensorBox)
            and isinstance(result.data, ir.BaseView)
        ):
            # Realize so that outputs are correctly aliased
            # 教学：直接返回的 BaseView 先物化，使对外 alias 关系有实际 storage 承载。
            result.realize()

        if (is_output or is_input_for_as_strided) and isinstance(
            n.meta["val"], torch.Tensor
        ):
            strides = n.meta["val"].stride()
            if len(strides):
                # 教学：as_strided 输入绝不允许 padding；用户可见输出通常也不允许，
                # 教学：内部/非用户可见输出可由配置放宽。
                allow_padding = (
                    config.pad_outputs or n.name not in self.user_visible_outputs
                ) and not is_input_for_as_strided
                dense = torch._prims_common.is_non_overlapping_and_dense(
                    n.meta["val"]
                )
                unbacked_symbols_in_strides = (
                    len(free_unbacked_symbols(strides)) > 0
                )
                # 教学：满足全部条件的 4D 结果可把目标 stride 改为 channels-last。
                if (
                    not unbacked_symbols_in_strides
                    and dense
                    and len(result.get_size()) == 4
                    and n in self.nodes_prefer_channels_last
                    and n.name not in self.user_visible_outputs
                    and not is_input_for_as_strided
                ):
                    strides = ir.FlexibleLayout.stride_ordered_for_memory_format(
                        result.get_size(), torch.channels_last
                    )
                # 教学：含 unbacked symbol 的 stride 在这里不强制具体布局。
                if not unbacked_symbols_in_strides and len(strides):
                    # To avoid converting possible view ops to a copy kernel, we use the previous
                    # require_exact_strides to handle views. But ultimately it's better to require
                    # the right strides at the tensor definition.
                    if n.meta["val"]._is_view() or isinstance(
                        result.data, ir.BaseView
                    ):
                        # 教学：view 只要求维度快慢顺序，避免为 exact stride 强制复制。
                        result = ir.ExternKernel.require_stride_order(
                            result,
                            ir.get_stride_order(strides),
                            allow_padding=allow_padding,
                        )
                    else:
                        # 教学：非 view 输出把 SymInt 解包后要求精确 strides。
                        strides = [
                            s.node.expr if isinstance(s, torch.SymInt) else s
                            for s in strides
                        ]
                        result = ir.ExternKernel.require_exact_strides(
                            result, strides, allow_padding=allow_padding
                        )

        # Realize if (1) any user need inputs realized, or (2) there is
        # already too many reads and rematerializing can be bad.
        # 教学：去重后的消费者超过一个时，才进入多用户存储/重算权衡。
        num_users = len(OrderedSet(n.users))
        if num_users > 1 and isinstance(result, TensorBox):
            for user in n.users:
                if user.target in needs_realized_inputs:
                    # 教学：至少一个消费者需要真实输入，先给出 realize hint。
                    result.realize_hint()
                    # This inclusion is somewhat controversial (from
                    # discussion between Horace, Natalia, and Elias).
                    # Currently, it's not very clear why this is helpful.
                    # The general idea here is that even though a node may
                    # have FlexibleLayout, we still often *treat* it as if
                    # it was contiguous. This appears to sometimes result in
                    # suboptimal behavior.
                    #
                    # When we do a better job selecting layout, we should
                    # revisit this.
                    need_fixed_layout = [
                        torch.ops.aten.convolution_backward.default,
                        torch.ops.aten.mm.default,
                        torch.ops.aten._int_mm.default,
                    ]
                    need_fixed_channels_last_layout = []
                    if not self.layout_opt:
                        need_fixed_layout.append(torch.ops.aten.convolution.default)
                    if torch._C._has_mkldnn:
                        need_fixed_layout += [
                            torch.ops.mkldnn._linear_pointwise.default,
                            torch.ops.mkldnn._linear_pointwise.binary,
                            torch.ops.aten.mkldnn_rnn_layer.default,
                            torch.ops.onednn.qlinear_pointwise.default,
                            torch.ops.onednn.qlinear_pointwise.tensor,
                            torch.ops.onednn.qlinear_pointwise.binary,
                            torch.ops.onednn.qlinear_pointwise.binary_tensor,
                        ]
                        need_fixed_channels_last_layout += [
                            torch.ops.mkldnn._convolution_pointwise.default,
                            torch.ops.mkldnn._convolution_pointwise.binary,
                            torch.ops.mkldnn._convolution_pointwise_.binary,
                            torch.ops.mkldnn._convolution_transpose_pointwise.default,
                            torch.ops.onednn.qconv2d_pointwise.default,
                            torch.ops.onednn.qconv2d_pointwise.binary,
                        ]
                        if torch._C.has_mkl:
                            need_fixed_layout += [torch.ops.mkl._mkl_linear.default]
                    if user.target in need_fixed_layout:
                        # 教学：这组消费者要求按当前 FX meta 的 stride order 固定结果。
                        result = ir.ExternKernel.require_stride_order(
                            result,
                            ir.get_stride_order(n.meta["val"].stride()),
                            allow_padding=True,
                        )
                    if (
                        user.target in need_fixed_channels_last_layout
                        and n is user.args[0]
                    ):
                        # 教学：只有当前节点是这类算子的第一个参数时才强制 channels-last。
                        result = ir.ExternKernel.require_stride_order(
                            result,
                            ir.get_stride_order(
                                make_channels_last_strides_for(n.meta["val"].shape)
                            ),
                        )
                if user.op == "output":
                    if isinstance(result.data.data, (Pointwise, Reduction)):
                        # 教学：多用户结果还要独立返回时，不能只在其他用户内重算。
                        result.realize()

            # TODO(jansel): introduce a store vs inline choice
            # 教学：立即运行 StorageBox 的复用启发式；必要时会在这里直接物化。
            result.mark_reuse(len(n.users))

        # Realize if the IRNode already has accumulated lots of reads
        if isinstance(result, TensorBox) and result.has_exceeded_max_reads():
            # Prevent excessive accumulation in a computed buffer, when
            # there are multiple branches each with small number of memory
            # reads, but they converge to a user.
            # 教学：即使单个分支不大，汇合后的累计读取过多也给出物化提示。
            result.realize_hint()

        # Realize if a Pointwise has too much stuff to be inlined.
        # As this may cause RecursionError during Inductor's evaluation.
        if isinstance(result, TensorBox) and isinstance(result.data, StorageBox):
            curr = result.data.data
            if isinstance(curr, Pointwise):
                # Use inner fn as a rough proxy. Good enough.
                if curr.has_large_inner_fn():
                    # 教学：Pointwise 内部函数过大时直接物化，截断表达式递归。
                    result.realize()
```

最后一行之后退出 K02 建立的三个上下文。下一产品语句开始给最终 result 填写 `origin_node`。

### stride 要求的判断顺序

`inductor_realize_to_strides` 的内层条件按 Python 运算优先级等价于：

```text
没有 get_stride
或
(当前 stride 与 meta 不同 且 meta stride 不是 symbolic)
```

因此“meta 含 symbolic stride”只会阻止第二项 mismatch 强制布局；若 result 根本没有 `get_stride`，第一项仍成立。后面的 output/as_strided 合同以 `n.meta["val"]` 是 Tensor 为入口；对这类 FX 节点，Lowering result 必须提供 tensor 的 size/data 接口，否则访问 `result.get_size()` 或 `result.data` 会直接暴露图与 Lowering 结果不一致，而不会退回另一条分支。

用 shape `[2,3]` 的转置 view 举例，它的 FX meta stride 为 `[1,3]`，result 也是 BaseView，并直接返回：

1. `is_output=True`，先 `result.realize()`，确保返回 view 的 alias 有实际 storage；
2. meta 是 tensor 且 stride 非空，进入合同处理；
3. view 分支调用 `require_stride_order([1,3] 对应的次序)`，不要求每个 stride 数字完全相等；
4. `allow_padding` 对用户可见输出通常为 false；
5. 若该 result 还有其他 user，后面的多用户逻辑继续判断是否独立存储。

换成一个直接计算得到的非 view tensor，返回 meta stride `[3,1]`，则走 `require_exact_strides`。这正是 view 与普通结果的区别：前者尽量避免为了 exact stride 把共享观察变成复制，后者可以在定义处确定精确排列。

### as_strided 消费者为什么更严格

设 a 的下一个 user 是 `as_strided(a,...)`。`is_input_for_as_strided=True`，所以 `allow_padding` 无条件为 false。原因可以用位置说明：a 原 stride 若是 `[3,1]`，as_strided 的参数是在这块既定存储上解释位置；若偷偷 padding 成 `[4,1]`，逻辑位置 `[1,0]` 会从元素位置 3 变成 4，后续视图语义已经改变。

即使 a 不是函数最终输出，这类消费者也使其 stride 成为可观察合同。若 stride 中含 unbacked symbol，当前分支不强制具体 order/exact strides，而把动态关系留给后续符号机制。

### 多消费者：返回与 mm 同时使用

考虑：

```python
a = torch.relu(x)
b = torch.mm(a, w)
return a, b
```

a 有两个不同 users：mm 和 output。a 最初可为延迟 Pointwise。

| 遍历 user | 判断 | a 的状态动作 |
| --- | --- | --- |
| mm | target 在 `needs_realized_inputs`，也在 `need_fixed_layout` | `realize_hint()`，再按 a 的 meta stride order 固定布局 |
| output | result 底层仍为 Pointwise/Reduction 时 | `realize()`，因为 a 必须有独立返回存储 |
| 循环后 | 多用户 | `mark_reuse(len(n.users))` 立即评估复用启发式，必要时物化 |

如果只有 mm 一个 user，`num_users > 1` 不成立，这一整段不进入；mm 自身 lowering 或其他边界仍可能要求输入实现。这里的规则专门处理“一份结果将被多处使用”时的存储/重算权衡。

### 三种停止继续内联的原因

- 某个多用户消费者在 `needs_realized_inputs`：调用 `realize_hint`，必要时固定布局；
- 展开的 IR 已累计过多内存读取：`has_exceeded_max_reads()` 后调用 `realize_hint`，防止多分支汇合时重复展开；
- Pointwise 的 inner function 已过大：直接 `realize()`，避免后续求值递归过深。

hint 表达“倾向在这里形成存储边界”，而直接 `realize()` 当场把 StorageBox 底层转换为已登记 buffer。正文不把两者写成同一个动作。

`mark_reuse(users)` 也不是把 users 数保存起来留给以后。它立即调用 StorageBox 的 `should_realize_on_reuse(users)`：只有多用户且底层仍是 Pointwise/Reduction 时才考虑；CPU 重计算包含 exp/sigmoid 等重操作，或读取次数/inner function 超过阈值时，当前调用会直接 `realize()`。条件不满足便不改变 result。它与前面的 `realize_hint()`、output 强制实现是同一轮中相互独立的三个判断。

固定布局的基础列表始终包含 convolution backward、mm 和 int_mm。若 `layout_opt` 关闭，普通 convolution 也加入。只有当前 PyTorch 构建带 MKLDNN 时，源码才把 MKLDNN/oneDNN target 加入两个列表；没有 MKLDNN 时这些追加完全跳过，基础列表仍照常使用。MKL linear 又要求该构建同时具有 MKL。平台条件只改变“哪些 consumer 进入特殊列表”，不跳过多用户循环、output 物化或 reuse 标记。

### 用新情况检验理解

1. result 只有一个 output user：不会进入 `num_users > 1` 段，但前面的 `is_output` stride/alias 合同仍执行。
2. 四维 dense 内部输出偏好 channels-last，但它是 user-visible output：channels-last 改写条件失败，保留对外 meta 合同。
3. result 有两个相同 user 记录但 `OrderedSet` 去重后只有一个：`num_users` 为 1，不进入多用户段。
4. Pointwise 没有多用户，但 inner function 过大：最后的独立条件仍直接物化。
5. as_strided 输入同时允许 `config.pad_outputs=True`：`and not is_input_for_as_strided` 仍使 padding 为 false。

## K04：把最终结果接回来源、读者和本节点新增对象

K03 结束后，result 已完成本节点内的 layout 与物化决定，三个 current-node 上下文也已经退出。`run_node` 接下来补三份图级索引：把适用的底层 IR 指回当前 FX node；登记这份结果仍会读取哪些 buffer；从水位之后的 buffer/operation 收集本节点新定义的 unbacked symbols。

### 完整源码：origin、readers 与新符号扫描

本次 result 是刚物化的 `TensorBox→StorageBox→ComputedBuffer(buf0)`，当前 n 是 relu；进入节点时 watermark 都为 0，当前 buffers/operations 各新增 buf0/op0。于是来源写到 buf0 与其 Loops，`register_users_of(result)` 登记它仍读取的输入，水位切片只扫描 `[buf0]`、`[op0]`。源码注释里的 “YOLO tree_map” 是指“不检查对象种类便对任意嵌套结果强行递归赋值”；这里明确不这样做，因为 view 或某些 buffer 不能安全改写 origin。

```python
    # This is not complete, but it doesn't have to be: origin_node
    # tracking is best effort.  The logic here critically relies on direct
    # TensorBox -> StorageBox denoting a non-view; we don't bother trying
    # to get views to work.  Feel free to add any extra cases as needed.
    #
    # Note: we can't YOLO tree_map over this result, because if there are
    # buffers or a view involved, we might not be able to validly assign
    # the origin_node here.
    # 教学：只处理能安全认定为非 view 的直接 TensorBox→StorageBox 链。
    if isinstance(result, TensorBox) and isinstance(result.data, ir.StorageBox):
        if isinstance(result.data.data, ir.Loops):
            # 教学：未包进 Buffer 的 Loops 直接指向当前 FX node。
            result.data.data.origin_node = n
        elif isinstance(result.data.data, ir.Buffer):
            # 教学：已物化 Buffer 自身先记录来源。
            result.data.data.origin_node = n
            if isinstance(result.data.data, ir.ComputedBuffer) and isinstance(
                result.data.data.data, ir.Loops
            ):
                # 教学：ComputedBuffer 内部的计算 Loops 也记录同一来源。
                result.data.data.data.origin_node = n
            # Not really multi-output, can straightforwardly recurse in
            elif (
                isinstance(result.data.data, ir.MultiOutput)
                and not result.data.data.indices
            ):
                # 教学：无 indices 的 MultiOutput 实际是直通包装，可把来源传给首个输入 Buffer。
                if isinstance(result.data.data.inputs[0], ir.Buffer):
                    result.data.data.inputs[0].origin_node = n

    # 教学：读取 result 的 read names，把这份结果登记成相应 buffer 的 reader。
    self.register_users_of(result)

    # 教学：从 K02 保存的水位向后扫描，只合并本节点新增对象定义的符号。
    new_unbacked_defs: OrderedSet[sympy.Symbol] = OrderedSet()
    for buf in self.buffers[buffer_watermark:]:
        new_unbacked_defs |= buf.get_unbacked_symbol_defs()
    for op in self.operations[operation_watermark:]:
        new_unbacked_defs |= op.get_unbacked_symbol_defs()

    def format_new_defs() -> str:
        # 教学：若 K05 的 FX/Inductor 符号覆盖断言失败，逐项打印来源对象。
        r = []
        for buf in self.buffers[buffer_watermark:]:
            r.append(
                f"unbacked_symbol_defs={buf.get_unbacked_symbol_defs()} in:\n{buf}\n"
            )
        for op in self.operations[operation_watermark:]:
            r.append(
                f"unbacked_symbol_defs={op.get_unbacked_symbol_defs()} in:\n{op}\n"
            )
        return "***\n".join(r)
```

下一产品语句是 K05 的 `if n.op != "placeholder":`。`format_new_defs` 此时只是闭包，只有后面的覆盖断言失败并构造错误消息时才需要它的字符串。

### origin_node 记录哪一层

假设 add 的 Pointwise 已在 K03 物化为：

```text
TensorBox
  └─ StorageBox
      └─ ComputedBuffer buf3
          └─ Pointwise Loops
```

当前 n 是 add FX node。源码先令 `buf3.origin_node=add`，再令内部 `Pointwise.origin_node=add`。以后 Scheduler 或 codegen 报告 buf3/内部循环时，可以回指这项 FX 计算。

若 result 是直接 `TensorBox(BaseView(...))`，第二个 isinstance 条件不成立，本段不强行向 view 内部写 origin。产品注释把这项追踪明确称为 best effort；缺少这一字段不改变计算语义，不能据此声称 view 没有 FX 来源。

MultiOutput 的特例只在 `indices` 为空时向 `inputs[0]` 递归。这类对象不是真正从多项结果中按 indices 选取，因此首个输入 Buffer 可以直接承接当前来源；有 indices 时不采用这条捷径。

### reader 登记保护旧值

设当前 result 表达 `a=x+1`，它尚会读取 InputBuffer x。`register_users_of(result)` 取得 read name `x`，建立：

```text
name_to_users["x"].append(result)
```

若后面的节点要修改 x，`mark_buffer_mutated("x")` 能找到这份 result，并先使依赖旧 x 的读取实现。否则 a 的延迟规则可能在覆盖后才展开，错误读到新 x。

origin 与 reader 索引不要混淆：`origin_node` 回答“这段 IR 来自哪个 FX node”；`name_to_users[x]` 回答“哪些仍存活的 IR 结果读取 x”。前者服务追踪，后者参与 mutation 正确性。

### 用处理前的列表长度找出本节点新增项

进入当前 node 时假设：

```text
buffers = [buf0, buf1, buf2]       buffer_watermark=3
operations = [op0, op1]            operation_watermark=2
```

本节点新增 `buf3`，其 `get_unbacked_symbol_defs()` 为 `{s0}`；又新增 `op2`，定义 `{s1}`。扫描结果为：

```text
buffers[3:]   = [buf3]
operations[2:]= [op2]
new_unbacked_defs = OrderedSet([s0,s1])
```

buf0–buf2 即使也定义符号，也不会重新进入本节点集合。`OrderedSet` 同时去重并保留首次遇到顺序，使后续断言和诊断稳定。

若覆盖断言失败，`format_new_defs()` 才会形成类似：

```text
unbacked_symbol_defs={s0} in:
buf3...
***
unbacked_symbol_defs={s1} in:
op2...
```

这份文字是错误诊断，不是生成程序的一部分。

### 用新情况检验理解

1. result 是 BaseView：本段不写 `origin_node`，但仍执行 `register_users_of` 和新符号扫描。
2. result 没读取任何 buffer：reader 登记不增加条目，origin 仍可能写入。
3. 本节点只新增 operation、不新增 buffer：buffer 切片为空，仍从 operation 收集符号。
4. buf3 与 op2 都定义 s0：集合最终只有一个 s0；后续范围/断言不会重复处理同一符号。
5. 前一节点 buf1 定义 s_old：因为在 watermark 之前，本节点不会把它误判为新定义。

## K05：在符号绑定点生成可执行断言

unbacked symbol 表示值来自运行时数据，编译期没有可用于普通 guard 的具体整数。某个 buffer/operation 第一次定义这类符号时，运行程序才有位置检查它的值域和依赖它的 runtime assertions。

K04 已得到 `new_unbacked_defs`。最后一段跳过 placeholder，对每个新符号处理值域和等待断言，再确认 FX node 元数据声明的 bindings 都能在本节点新增 IR 中找到对应定义。

`ras_by_symbol` 不是 `run_node` 临时创建的空表。ShapeEnv 在更早的符号推导中遇到不能静态证明、又含 unbacked symbols 的条件时，建立 `RuntimeAssert(expr,msg,stack)`，并按表达式中编号最大的 unbacked symbol 放进 `shape_env.deferred_runtime_asserts[key]`。GraphLowering 构造时复制这张表为 `self.ras_by_symbol`。本段的职责是：当某个 key 首次在 Inductor IR 中被定义时取出该组记录；依赖已齐便生成 AssertScalar，依赖未齐便换到另一个缺失 symbol 的 key 下继续等待。

FX `node.meta["unbacked_bindings"]` 则来自 fake/meta 推导，内容是“符号 → 从该节点结果树取值的 key path”，例如 u0 对应 `result.size()[0]`。`resolve_unbacked_bindings` 只把已重命名的 symbol key 换成当前名字，path value 原样保留。当前覆盖检查只消费这些 key，确认 FX 声明的每个结果符号都由本节点新增 IR 定义；path 已在相应 lowering 建立 IR 定义时用于定位运行时取值，本段不再次沿 path 读结果。

### 完整源码：新符号处理至 return result

先区分两类断言。`var_to_range[u0]` 中的有限上下界在处理 u0 时可以立即形成 `AssertScalar`。`ras_by_symbol[u0]` 中的 deferred runtime assertion 则要重新计算它引用的全部 unbacked symbols；本节点的新定义直到整个循环结束后才并入 bound 集合，所以这类 assertion 可能在本轮重新排队，而不是立即生成。末尾的 binding 覆盖检查只确认 FX 声明的符号确实由本节点 IR 定义，不等于 deferred 队列已经清空。

```python
    if n.op != "placeholder":
        # Note [Backwards runtime asserts]
        # Backwards poses an interesting problem for deferred runtime
        # asserts.  In the easy case, we may solely close over data
        # dependent sized tensors, and there are no binding sites for
        # unbacked SymInts.  In this case, we can just drop all the
        # runtime asserts on the floor: no non-placeholder bindings, no
        # problem.
        #
        # However, it is *possible* for a fresh runtime assert to show up
        # between forwards and backwards.  Right now, the freezing process
        # that happens when we lower forwards means that we will freeze
        # runtime asserts, and then the moment the backwards lowering
        # process attempts to add a new deferred runtime assert, we will
        # fail.  Let's say you remove that assert.  Now when we get here,
        # we need to make sure we actually emit these asserts (because we
        # can't emit them in forwards, we already compiled it).  So we
        # have to do something here.  But we don't want to reemit ALL
        # deferred runtime asserts, we only want to emit the NEW ones.
        # Therefore needing some sort of stratification in the ShapeEnv.
        # This is all doable, it just hasn't been done yet.
        # 教学：这里只处理当前非 placeholder 节点新出现的 binding 和相关断言。
        shape_env = V.graph.sizevars.shape_env

        def make_assert(expr: Expr, msg: str) -> None:
            # 教学：AssertScalar 同时登记为 buffer 和 operation，进入后续调度/生成。
            assert_op = ir.AssertScalar(expr, msg)
            self.register_buffer(assert_op, set_name=True)
            self.register_operation(assert_op)

        for i0 in new_unbacked_defs:
            # 教学：取出此前按 i0 分组等待的 runtime assertions；没有则为空。
            ras = self.ras_by_symbol.pop(i0, [])
            # NB: size-like not needed, we won't retrace
            vr = shape_env.var_to_range[i0]
            # 教学：若实际值域比默认“不指定”范围更窄，尝试生成有限上下界断言。
            if not shape_env._default_unspecified_value_range().issubset(vr):

                def is_convertible(s: Expr) -> bool:
                    # 教学：正负无穷不能变成运行时整数界；有限可转整数表达式才可生成。
                    if s in (int_oo, -int_oo):
                        return False
                    try:
                        int(s)
                        return True
                    except TypeError:
                        return False

                if is_convertible(vr.lower):
                    make_assert(i0 >= vr.lower, f"{i0} >= {vr.lower}")
                if is_convertible(vr.upper):
                    make_assert(i0 <= vr.upper, f"{i0} <= {vr.upper}")

            for ra in ras:
                # 教学：runtime assertion 只有在其全部 unbacked symbols 已绑定后才能生成。
                fvs = free_unbacked_symbols(ra.expr)
                missing = fvs - self.bound_unbacked_symbols
                if missing:
                    # 教学：仍缺依赖时，稳定选择名字最小的一项作为下一等待 key。
                    i1 = min(missing, key=str)
                    self.ras_by_symbol.setdefault(i1, []).append(ra)
                else:
                    make_assert(ra.expr, f"{ra.expr}")

        # 教学：本轮所有新定义处理结束后，才整体加入已绑定集合。
        self.bound_unbacked_symbols |= new_unbacked_defs

        # 教学：解析 FX meta 记录的“符号来自结果树哪条路径”。
        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env, n.meta.get("unbacked_bindings", {})
        )
        # When we do lowering, it is possible we reallocate unbacked SymInts.
        # So we need to line up the unbacked SymInts when performing the test
        # here
        #
        # In principle, we could permit lowering to introduce MORE unbacked
        # SymInts: as long as all the old unbacked ones are accounted for,
        # it's fine for inductor to introduce extra calls to item()/unbacked()
        # whatever.  This actually happens in practice when an unbacked SymInt
        # gets memoized away; naively, when Inductor reprocesses a kernel, it
        # doesn't know that the memo still applies, and ends up allocating a
        # new symbol.  However, this is generally a bad thing: we may still
        # end up needing to test equalities on the symbols, and a fresh
        # symbol is likely to hit lots of GuardOnDataDependent errors that
        # we already know facts for.
        # 教学：按 FakeTensor ShapeEnv 的重命名表把 FX 符号对齐到当前名字。
        renamed_unbacked_bindings = OrderedSet(
            V.fake_mode.shape_env.unbacked_renamings.get(s, s)
            for s in unbacked_bindings.keys()
        )
        # 教学：Inductor 本节点新定义集合必须覆盖 FX 声明；允许前者有额外符号。
        assert new_unbacked_defs >= renamed_unbacked_bindings, (
            f"failed {new_unbacked_defs} >= {renamed_unbacked_bindings} (inductor >= fx)\n"
            f"fx node is: {n.format_node()}\n"
            f"new operations are:\n\n{format_new_defs()}"
        )

    # 教学：所有图级收尾完成，result 才交回 FX Interpreter 存入 env。
    return result
```

至此，`run_node` 从定义行到 `return result` 已按产品原顺序完整覆盖。K01 的 helper 与 K02–K05 的四段合起来构成本叶子声明的全部源码。

### 值域怎样变成 AssertScalar

假设本节点新定义 `s0`，ShapeEnv 记录范围 `[2,10]`。默认未指定范围并不包含于这段更窄范围，两个边界都能转成整数，因此建立：

```text
AssertScalar(s0 >= 2, "s0 >= 2")
AssertScalar(s0 <= 10, "s0 <= 10")
```

每项都同时进入 `buffers` 和 `operations`。若下界为 `-∞`、上界为 `+∞`，`is_convertible` 返回 false，不为对应方向生成无意义断言。若范围没有比默认范围更窄，外层条件也会跳过整个边界生成。

### 延迟断言的集合怎样推进

代入同一轮的完整状态：

```text
进入循环前：
bound_unbacked_symbols = {s_old}
new_unbacked_defs      = [s0, s1]
ras_by_symbol[s0]      = [s0 < s1]
ras_by_symbol[s1]      = []
```

处理 s0 时先 `pop(s0)`，取出 `s0 < s1`。表达式的 unbacked symbols 为 `{s0,s1}`；此刻新定义尚未并入 bound，所以：

```text
missing = {s0,s1} - {s_old} = {s0,s1}
min(missing,key=str) = s0
ras_by_symbol[s0] = [s0 < s1]   # 重新排队，没有生成 AssertScalar
```

处理 s1 时，它的队列为空。循环结束后才执行：

```text
bound_unbacked_symbols = {s_old,s0,s1}
```

刚重新放回 `ras_by_symbol[s0]` 的 assertion 不会在本次 `run_node` 中自动重访。当前方法没有在 union 后再扫一遍队列，也不在这里保证未来一定再次遇到同一个定义 key；离开本方法时能够确认的状态只是“记录仍保存在队列中，尚未生成 AssertScalar”。

后续某次 `run_node` 只有在它的 `new_unbacked_defs` 再包含该等待 key 时，才会 `pop` 并重试这条记录；若编译过程中不再出现这个 key，本方法没有额外的收尾扫描来保证生成。这里应把“留在队列”当作当前源码的明确出口，而不能把“最终一定生成”当作教程补出的保证。

对照一条已经就绪的记录：若 `ras_by_symbol[s0]` 中某项表达式只引用 `{s_old}`，那么 `missing={}`，处理 s0 时会立即建立 `AssertScalar`。所以判断依据不是“当前正在处理哪个 key”，而是 assertion 的 free unbacked symbols 是否已全部属于进入本轮前的 bound 集合。

范围断言走的是另一条路径。若 s0 的范围为 `[2,10]`，`s0>=2` 与 `s0<=10` 在处理 s0 时直接建立，不参与上述 `ras_by_symbol` 的 missing/requeue 过程。

### 为什么检查 `Inductor >= FX`

设 FX node meta 声明结果绑定 `{u0: path}`，重命名表把 u0 对齐为 s0。本节点新增 IR 实际定义：

```text
new_unbacked_defs = {s0,s_extra}
renamed_unbacked_bindings = {s0}
```

集合覆盖成立：FX 所承诺的 s0 确实有定义；Inductor 多产生 s_extra 在当前断言中允许。若新增 IR 只有 `{s_extra}`，覆盖失败，错误消息会列出 FX node 和 K04 的 `format_new_defs()`，帮助定位哪个新 buffer/operation 定义了哪些符号。

这项检查比较符号定义，不比较 tensor 数值，也不证明所有 runtime assertions 已经执行；断言对象仍需后续 Scheduler 与 codegen 生成。

### placeholder 为什么跳过

条件明确排除 `n.op == "placeholder"`。输入符号属于图接口已有绑定，本段针对非输入节点新产生的数据依赖符号及其 runtime assert。即使 K04 扫描得到空集合，普通非-placeholder 仍会解析空 bindings 并完成覆盖检查；placeholder 则整段不进入。

### 用新情况检验理解

1. s0 范围 `[0,+∞]`：若下界 0 可转整数而上界是 `+∞`，只生成 `s0>=0`。
2. FX bindings 为空，Inductor 新定义 `{s0}`：覆盖成立，因为空集是任何集合的子集。
3. FX 经重命名要求 `{s1}`，新定义只有 `{s0}`：覆盖失败，即使两个名字在文字上都像 unbacked symbol。
4. 一条等待断言仍缺 s2、s3：用 `min(missing,key=str)` 选一个 key 重新排队，不复制到两个队列。
5. result 是 SymPy 表达式而非 TensorBox：前面的 origin/reader 可能不动作，但本节点新 operation 定义的符号仍在这里处理，最后原 result 照常返回。

### 哪些不变量失败会停止本节点 Lowering

本方法只为预期的分派路径选择 fallback；进入具体路径后，不变量失败不会被 `run_node` 吞掉，也没有事务回滚：

| 位置 | 失败方式 | 失败前已经可能发生的变化 |
| --- | --- | --- |
| `propagate_mutation` 的 target/数量断言 | 显式断言失败 | target lowering 已产生 result，新副本上的 mutation 可能已经发生；尚未完成旧参数语义修复 |
| kwargs schema 配对 | 查表失败 | positional writable 参数和排在失败项以前的 kwargs 可能已经调用 `copy_` lowering |
| tensor stride/layout 合同 | 属性访问或下层 layout 要求失败 | result 与 layout mutation 已产生；排在失败点以前的 realize/hint/layout 包装可能已改变或登记对象；上下文管理器仍会恢复三份 current state |
| 新 symbol 的 `var_to_range` 查表 | KeyError | 分派、stride/物化、origin、readers 与新对象扫描已完成；更早 symbol 的 AssertScalar 或 requeue 可能已登记；当前 key 的 ras 已被 pop，bound 尚未 union |
| FX/IR binding 覆盖 | 带 FX node 和新对象定义信息的断言失败 | 所有新 symbol 已完成范围/requeue 处理，AssertScalar 可能已登记，`bound_unbacked_symbols` 已并入本节点定义 |
| `register_buffer/register_operation` 或相关 helper 自身异常 | 原异常向外 | 调用以前的所有图级修改保留；若 buffer 登记成功而 operation 登记失败，可能只留下前一项 |

这些是编译期失败，不是生成程序中的 runtime assertion。有限值域和已就绪 deferred expression 形成的 `AssertScalar` 才会作为 buffer/operation 进入后续调度与代码生成。

## 把完整 run_node 串成一次 relu→mm→output

对 `a=relu(x); b=mm(a,w); return a,b`：

1. K02 记录进入 relu 前的水位，取得 x 的 IR，合并 origins，通过普通分派得到 Pointwise result；
2. K03 看到 a 有 mm 与 output 两个 users：给出物化提示，按 mm 要求固定 stride，因 a 还要返回而实现为 buffer；
3. K04 将 a 的 ComputedBuffer/Loops 指回 relu FX node，登记它读取 x，并只扫描本节点新增 buffer/operation 的符号；
4. K05 为新数据依赖符号登记有限值域及已就绪断言，核对 FX bindings；result 返回 env；
5. 处理 mm 时重新记录新的水位与 origin，外部/模板 lowering 产生自己的 result；
6. output 节点走 Q213，将 a、b 放进 `graph_outputs`，最后决定所有 buffer layout。

这个案例把五个单元接成同一次节点解释过程。`run_node` 没有替代各算子 lowering；它围绕 lowering 结果维护整张图必须一致的 layout、存储、来源、读者和符号状态。

实验 20 对这一结构做了完整 CPU 编译。当前 post-grad 图中的 relu 节点有两个直接 users：`aten.mm.default` 和 `output`；`run_node` 返回时，relu 的结果已经是 `TensorBox→StorageBox→ComputedBuffer`，本节点新增 `buf0/op0`。随后 mm 形成 `ExternKernelOut buf1/op1`。实际输出为：

```text
activation = [[0,2], [3,0]]
product    = [[6,8], [3,6]]
```

这个结果展示了 K03 的关键分支：relu 同时被 mm 与 output 使用，因此先物化为 buf0；mm 随后读取 buf0 建立 buf1。若节点另外定义数据相关符号，则在同一次 `run_node` 的 K05 段登记相应运行时断言。

## 迁移验收

若把 a 改为只被 mm 使用，不再返回：K03 的 `num_users>1` 不成立，a 不会由这段“多用户且 output”规则立即实现；其他 mm 输入合同仍可在相应 lowering/边界要求它有真实存储。K04/K05 是否动作取决于最终 result 和本节点实际新增对象，而不是源码里是否出现 relu 名字。

若在 relu 后新增对 x 的原地覆盖，K04 已将 a 登记为 x 的 reader；mutation 处理可以先保护 a 的旧读。若 relu 产生 s0 且 FX meta 声明同一 binding，K05 的覆盖检查通过；若 IR 遗漏 s0 定义，编译在这里以带新对象诊断的断言失败。

Q214 至此达到叶子完成条件。下一项 Q221 打开 Pointwise、Reduction、View 的构造核心，解释 lowering 处理器怎样建立 `ranges`、indexing function 与 loader。

