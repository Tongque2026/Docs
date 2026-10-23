# 特殊调用分派核心源码：候选怎样变成 IR operation

[返回特殊调用篇](../README.md) · [返回 Lowering](../../README.md) · [返回总览](../../../README.md)

这一篇追踪四条不会变成普通 `Pointwise` 或 `Reduction` 的 Lowering 路径：算法候选、fallback、`cond` 和 `while_loop`。每条路径都从 Lowering handler 收到的对象开始，一直走到登记进图的 operation 和输出引用。

```text
ChoiceCaller：一份候选，保存输入、输出 layout 和“选中后怎样建 IR”
       |
       | output_node()
       v
TemplateBuffer / ExternKernel：已经进入 Lowering 图的一项 operation

MultiTemplateBuffer：也已经是 operation，但内部 render 仍可等 Scheduler
                     看见 epilogue 后再固定
```

## K01：算法选择怎样从候选表走到一项 IR operation

设矩阵乘有两个候选：T 是 Triton template，E 是 `at::mm_out`。二者都接收 A、B，输出 layout 都是 4×4 float32。`AlgorithmSelectorCache.__call__` 收到的是候选表 `[T,E]`，不是已经选好的候选。它有两个出口：

```text
立即选择：测 T、E → 取耗时最小者 → 调用该 caller.output_node() → 具体 operation

延迟选择：建立 MultiTemplateBuffer operation → Scheduler 比较基础候选以及可融合的 epilogue
                                            → 固定 Triton render，或用 extern operation 替换它
```

caller 只保存“如果选中我，怎样建立 IR”，不会出现在 `GraphLowering.operations`。operation 的构造器负责把自己登记到当前 `GraphLowering`，它才是后续 Scheduler 和 Codegen 处理的图节点。`TensorBox` 是 Lowering 传递 tensor 值的外层盒子；`TensorBox.create(buffer)` 会再套一层 `StorageBox`，由这层持有底层 buffer。

`render` 是 Triton template 的代码生成函数：layout 和输入已经确定后，它把模板填成内核代码。`do_autotuning(...)` 返回以 caller 为 key、运行时间为 value 的字典，例如 `{T: 2.4, E: 1.1}`，单位是毫秒。

### 立即选择：先计时，再调用胜者的 `output_node()`

当前 selector 的收束代码是：

```python
timings = do_autotuning(precompile_fn)
if timings == {} or choices[0] not in timings:
    return choices[0].output_node()

selected_key = builtins.min(timings, key=timings.__getitem__)
selected_choice = selected_key.output_node()
return selected_choice
```

若得到 `{T: 2.4 ms, E: 1.1 ms}`，`selected_key` 是 E，随后才调用 `E.output_node()`。若计时表为空或缺少第一项，代码不凭空补全结果，而是直接使用第一候选。无论哪条分支，selector 返回的都不再是 caller，而是 caller 建立的 `TensorBox`。

Triton caller 的转换只有一条路径：

调用前，`self` 是已选中的 `TritonTemplateCaller`：`self.layout` 是 4×4 float32 输出排布，`self.input_nodes=[A,B]`，`self.make_kernel_render` 是稍后填模板的函数，`self.mutated_inputs` 记录模板会改写的输入（本例为空）。`output_node()` 不运行矩阵乘；它把这些字段装进 `TritonTemplateBuffer`，再包成 TensorBox 交回 GraphLowering 登记。

```python
def output_node(self):
    # 教学：layout、输入和 render 函数都来自已经构造好的候选。
    return ir.TensorBox.create(
        ir.TritonTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            debug_extra=self.debug_extra,
            mutated_inputs=self.mutated_inputs,
        )
    )
```

返回值外层是 `TensorBox`，底层已经是 `TritonTemplateBuffer`。后续使用者引用的是这项 buffer，不再直接引用 caller。

extern caller 必须先决定采用哪一种外部节点：

```python
def output_node(self):
    if config.abi_compatible and self.choice.use_fallback_kernel:
        # 教学：ABI 模式要求借用 FallbackKernel 的参数与输出封装。
        assert (
            self.choice.op_overload is not None
        ), "Please provide an op_overload to use ir.FallbackKernel"
        inner = ir.FallbackKernel.create(
            self.choice.op_overload, *self.input_nodes, **self.kwargs
        )
    elif self.choice.kernel_creator is not None:
        # 教学：某些候选自己提供 IR 构造器，直接交给它建立 inner。
        inner = self.choice.kernel_creator(*self.input_nodes, **self.kwargs)
    else:
        # 教学：有 out variant 时调用方提供输出，否则由外部调用分配返回值。
        cls = ir.ExternKernelOut if self.has_out_variant else ir.ExternKernelAlloc
        inner = cls(
            layout=self.layout,
            inputs=self.input_nodes,
            python_kernel_name=self.choice.call_name(),
            cpp_kernel_name=self.choice.cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel=self.choice.ordered_kwargs_for_cpp_kernel,
            op_overload=self.choice.op_overload,
            kwargs=self.kwargs,
        )

    return ir.TensorBox.create(inner)
```

实验 07 的 CPU 4×4 `mm` 进入最后一支，得到：

```text
self.has_out_variant = True
cls = ExternKernelOut
inner = ExternKernelOut(layout=4×4, inputs=[A,B], ...)
return TensorBox(StorageBox(ExternKernelOut))
GraphLowering.operations = [ExternKernelOut]
```

这里的关键变化不是“算出了矩阵乘数值”，而是候选身份消失，图中留下具有输入、layout 和调用名的 operation。数值要到生成程序运行时才由 `at::mm_out` 计算。

### extern caller 的三个建图出口

若同一个 extern choice 把 `has_out_variant` 从 `True` 改为 `False`，最后一支只改变 `cls`：图中得到 `ExternKernelAlloc`，表示外部函数自己返回存储；输入和 choice 身份没有因此变成 template。

若 `abi_compatible=True` 但 choice 没有 `op_overload`，代码在建立 FallbackKernel 之前触发断言，所以图里不会留下半成品 operation。这说明失败出口发生在 `inner` 建立之前。

### 延迟选择：先把 MultiTemplateBuffer 放进图

普通 selector 的薄入口还会补一个影响出口的默认参数：

```python
def autotune_select_algorithm(*args, **kwargs):
    global _ALGORITHM_SELECTOR_CACHE
    # 教学：整个进程复用 selector cache；第一次调用才创建。
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()

    # 教学：调用者未显式决定时，由 epilogue benchmark 配置决定是否返回多模板。
    if "return_multi_template" not in kwargs:
        kwargs[
            "return_multi_template"
        ] = torch._inductor.config.benchmark_epilogue_fusion

    return _ALGORITHM_SELECTOR_CACHE(*args, **kwargs)
```

只有 `return_multi_template` 为真，同时 `max_autotune` 或 `max_autotune_gemm` 开启时，selector 才走延迟出口：

```python
if return_multi_template and (config.max_autotune or config.max_autotune_gemm):
    def get_timings():
        timings = do_autotuning(precompile_fn)
        min_extern_choice = float("inf")
        for choice, timing in timings.items():
            if isinstance(choice, ExternKernelCaller):
                min_extern_choice = min(min_extern_choice, timing)

        return {
            choice: time
            for choice, time in timings.items()
            if time <= min_extern_choice or not isinstance(choice, ExternKernelCaller)
        }

    return ir.TensorBox.create(ir.MultiTemplateBuffer(layout, input_nodes, get_timings))
```

这里还没有选定 T 或 E。`get_timings` 被保存成回调；`MultiTemplateBuffer` 已登记为 operation，但 `make_kernel_render=None`。extern 候选保留耗时等于最快 extern 的项，通常是一项，并列最快时可能有多项；Triton 候选全部保留，供 Scheduler 后面比较。

底层 operation 是下面这个完整类：

本次对象的 `layout` 是 4×4 float32，`inputs=[A,B]`。第一次读取 `choice_timings` 才执行计时回调并缓存结果；`get_min_choice()` 只返回基础耗时最小项，不改变图中 operation。

```python
class MultiTemplateBuffer(TritonTemplateBuffer):
    """
    Represents a Buffer with multiple backing implementation choices.

    Choices can be TritonTemplates or ExternKernels. During scheduling if there is a potential
    epilogue we will benchmark each of the choices with the epilogue to determine an implementation.
    Otherwise, the fastest base choice will be chosen.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: List[IRNode],
        choice_timings: Callable[[], Dict[ChoiceCaller, float]],
    ):
        # 教学：先按 template buffer 登记 operation，但 render 暂时没有固定。
        super().__init__(layout=layout, inputs=inputs, make_kernel_render=None)
        self._choice_timings_fn = choice_timings
        self._choice_timings: Optional[Dict[ChoiceCaller, float]] = None
        self.original_inputs = inputs

    @property
    def choice_timings(self) -> Dict[ChoiceCaller, float]:
        # 教学：第一次消费者读取才运行计时函数，以后复用同一份结果。
        if self._choice_timings is None:
            self._choice_timings = self._choice_timings_fn()
        return self._choice_timings

    @contextlib.contextmanager
    def swap_as_triton_caller(self, caller: TritonTemplateCallerBase):
        assert isinstance(caller, torch._inductor.select_algorithm.TritonTemplateCaller)
        assert self.layout == caller.layout

        # 教学：benchmark 某个 caller+epilogue 时只临时替换 render，退出后恢复。
        render = self.make_kernel_render
        self.make_kernel_render = caller.get_make_kernel_render()
        try:
            yield
        finally:
            self.make_kernel_render = render

    def finalize_as_triton_caller(self, caller: TritonTemplateCallerBase):
        assert isinstance(caller, torch._inductor.select_algorithm.TritonTemplateCaller)
        assert self.layout.size == caller.layout.size
        assert self.layout.stride == caller.layout.stride
        # 教学：最终选择后不再恢复，后续 codegen 使用这个 render。
        self.make_kernel_render = caller.get_make_kernel_render()

    def get_min_choice(self) -> Tuple[ChoiceCaller, float]:
        # 教学：没有 epilogue 需要比较时，直接取基础 timings 中耗时最小的候选。
        min_choice = min(self.choice_timings, key=self.choice_timings.get)  # type: ignore[arg-type]
        return (min_choice, self.choice_timings[min_choice])
```

用候选 `extern_mm=8.0 ms`、`triton_32=5.0 ms` 推进状态：

| 时刻 | `_choice_timings` | `make_kernel_render` | 图中对象 |
| --- | --- | --- | --- |
| 构造后 | `None` | `None` | 已是 MultiTemplateBuffer operation |
| 第一次 `get_min_choice()` | 两项 timings 被计算并缓存 | `None` | 最小项为 triton_32，但尚未安装 render |
| 临时 `swap_as_triton_caller` 内 | 同一缓存 | triton_32 render | 可测 template+epilogue |
| 离开 context | 同一缓存 | 恢复 `None` | 临时测试没有篡改最终状态 |
| `finalize_as_triton_caller` 后 | 同一缓存 | triton_32 render | 后续 codegen 使用最终 render |

### Scheduler 怎样把延迟选择收束成最终 operation

若没有值得保留的 epilogue 融合，`finalize_multi_template_buffers()` 读取基础最小项：

```python
min_node_unfused, _ = multi_node.get_min_choice()

if isinstance(min_node_unfused, TritonTemplateCallerBase):
    node.node.finalize_as_triton_caller(min_node_unfused)
    continue

out_tensorbox = min_node_unfused.output_node()
out_buffer = out_tensorbox.data.data
out_buffer.layout = multi_node.layout
replace_operation_buffer(multi_node, out_buffer)
```

两类胜者的处理不同：

- Triton 胜出时，原来的 `MultiTemplateBuffer` 保留在图中，只把胜者的 render 永久装入 `make_kernel_render`。
- extern 胜出时，先调用 E 的 `output_node()` 建立具体 `ExternKernel`，再用它替换图中的 `MultiTemplateBuffer`。替换函数把旧 buffer 和 operation 的名字移交给新节点，并同步 Scheduler 的节点与输出用户。

以旧节点名 `buf0`、新 extern 构造时暂得名 `buf1` 为例：

```text
替换前：GraphLowering.operations = [MultiTemplateBuffer(name=buf0)]
        下游 relu 读取 buf0

E.output_node()：新建 ExternKernel(name=buf1)，两个 operation 暂时都已登记

replace_operation_buffer：从索引表删掉 buf1
                          把 extern 的 buffer 名和 operation 名改成旧节点的名字
                          在 buffers/operations 原位置用 extern 替掉 MultiTemplateBuffer

替换后：GraphLowering.operations = [ExternKernel(name=buf0)]
        Scheduler 为 extern 重建节点，并把旧输出的 users 交给新输出
        下游仍通过 buf0 读取同一逻辑结果
```

上面的名字移交直接来自替换函数的这些语句：

```python
del V.graph.name_to_buffer[replaced_buf_name]
new_node.name = orig_buf_name
del V.graph.name_to_op[replaced_op_name]
new_node.operation_name = orig_op_name

orig = V.graph.buffers.index(orig_node)
V.graph.buffers.remove(new_node)
V.graph.buffers[orig] = new_node

orig = V.graph.operations.index(orig_node)
V.graph.operations.remove(new_node)
V.graph.operations[orig] = new_node
```

若 Scheduler 正在考虑把 `relu` 融入矩阵乘，它会临时把每个 Triton caller 的 render 装进 `MultiTemplateBuffer`，实测“T+relu”。临时测试退出后恢复旧 render。判定条件直接比较 `min_ms_fused` 与 `ms1 + ms2`：

```python
if min_ms_fused < (ms1 + ms2) and ms_fused_choice is not None:
    node1.node.finalize_as_triton_caller(ms_fused_choice)
    return True
else:
    return False
```

`ms1` 是最快基础候选，`ms2` 是单独 relu，`min_ms_fused` 是测得的最快融合结果。条件成立才永久固定该 Triton render 并允许融合；否则拒绝这次融合，之后仍由基础最小项完成收束。本节假定计时表至少有一个有效候选；空表传到 `get_min_choice()` 会使 `min(...)` 无法给出胜者。

独立实验 [`22_multi_template_state.py`](../../../experiments/22_multi_template_state.py) 直接调用当前产品类的 `choice_timings` 和 `get_min_choice`，验证计时函数只执行一次：第一次与第二次均选择 `triton_32, 5.0`，计数仍为 1。实验没有伪造 Triton caller，所以不把 `swap/finalize` 声称为本机运行；那两条状态由上面的当前源码顺序支持。

### K01 理解验收

问题：第一次 `get_min_choice()` 得到 triton 最快，能否立即断言图中已有最终 Triton render？

不能。此时 `choice_timings` 只从 `None` 变成缓存字典，`get_min_choice` 只返回 `(caller,time)`，没有写 `make_kernel_render`。只有 `finalize_as_triton_caller` 才永久安装 render；若 Scheduler 要比较 epilogue，它还可能先在 `swap_as_triton_caller` 的临时状态中测试。

问题：若显式传入 `return_multi_template=False`，入口会不会被全局配置改回 True？

不会。`autotune_select_algorithm` 只在 kwargs 缺少该键时填默认值。显式 False 保留下来，selector 按这次调用的要求处理。

因此 K01 的完整交接是：Lowering 把候选表交给 selector；selector 或立即让胜者建图，或先建立 `MultiTemplateBuffer`；Scheduler 再把延迟状态固定成 Triton render 或替换成 extern operation。下一节追踪没有专用 Lowering 的 operator 怎样通过 fallback 留在同一张图里。

## K02：fallback 保留一次调用，再把 packed 返回值拆成图内引用

### 为什么先注册，而不是遇到算子时临时猜测

`histc` 没有在本版本中转换成 Pointwise 或 Reduction。模块加载时执行 `make_fallback(aten.histc)`，把它明确接到 fallback handler。这样 GraphLowering 读到节点时，仍由统一的 lowering 表找到处理器；它没有离开当前编译图。

下面是 `make_fallback` 的完整产品实现：

本次传入 `op=aten.histc`，它不在 decomposition 表；`warn=True`，普通本地环境没有 CI 环境变量，`fallback_random` 与这个非随机算子无关。histc 是 overload packet，所以代码遍历其 overload，把每个具体 target 注册到 lowerings，并标记输入必须先物化。沿这组状态，decomposition 冲突断言、CI 警告和随机早退都不会改变登记结果。

```python
def make_fallback(op, layout_constraint=None, warn=True):
    # 教学：同一 operator 不能既声明 decomposition 又声明 fallback。
    assert op not in decompositions, f"both a fallback and a decomp for same op: {op}"
    if (
        warn
        and bool(os.getenv("CI"))
        and get_decompositions([op])
        # if fallback_random, we allow not decomposing random
        and not (
            config.fallback_random
            and op in torch._decomp.decompositions_for_rng.extra_random_decomps
        )
    ):
        # Note: 'warn' is holdover from when this was a warning, but for ops that previously
        # set warn=False we do not want a CI error.
        # Ignore the 'suppress errors' configs in CI, as this particular warning happens on startup anyway and is not
        # likely to be triggered preferentially on one CI config over another.
        if torch._dynamo.config.suppress_errors:
            torch._dynamo.config.suppress_errors = False
            log.warning(
                "A make_fallback error occurred in suppress_errors config,"
                " and suppress_errors is being disabled to surface it."
            )
        raise AssertionError(
            f"make_fallback({op}): a decomposition exists, we should switch to it."
            " To fix this error, either add a decomposition to core_aten_decompositions (preferred)"
            " or inductor_decompositions, and delete the corresponding `make_fallback` line."
            " Get help from the inductor team if unsure, don't pick arbitrarily to unblock yourself.",
        )

    def register_fallback(op_overload):
        # 教学：外部调用需要实体输入；可选 layout constraint 同时登记。
        add_needs_realized_inputs(op_overload)
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        # 教学：lowering 表最终保存的是调用 FallbackKernel.create 的 handler。
        return register_lowering(op_overload, type_promotion_kind=None)(
            fallback_handler(op_overload)
        )

    if isinstance(op, torch._ops.OpOverloadPacket):
        # 教学：传入 packet 时逐个 overload 注册，例如 histc 的全部 overload。
        for ol in op.overloads():
            op_overload = getattr(op, ol)
            register_fallback(op_overload)
    elif isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        register_fallback(op)
    else:
        # 教学：普通 Python callable 不具备这里所需的 operator schema 身份。
        raise RuntimeError(f"Unsupported fallback {op} with type {type(op)}")
```

`aten.histc` 是一组同名重载的 packet，`aten.histc.default` 是其中一个具体 overload。这里传入 packet，因此循环把每个具体 overload 分别登记。登记同时调用 `add_needs_realized_inputs`：外部 operator 运行时需要真实 tensor，不能接收一条尚未执行的 Pointwise 规则。

### create 先用 fake 值确定接口，再建立 packed operation

处理器命中以后进入 `FallbackKernel.create`。这里的 `packed` 是代表“一次完整 operator 调用及其返回对象”的 operation；返回对象既可能直接是 Tensor，也可能是 tuple/list/dict。`MultiOutput` 不会再次调用 operator；它保存 `input=packed` 和一条取值路径，Codegen 据此从 packed 返回值中取出一个 Tensor。它自己的构造器会登记 buffer 和 operation，但 `should_allocate()` 返回 False，因为实际存储来自 packed 调用的返回值。

`process_kernel` 完成三件前置工作：把 TensorBox 参数换成对应的 fake 值；用 fake 参数调用 kernel 一次，得到描述返回接口的 `example_output`；同时拆出运行时调用需要的 `tensor_args`、`non_tensor_args` 和重建原参数结构的 `unflatten_args`。它不计算本次真实输入的结果。

`FallbackKernel.create` 先建立 packed，再为每个 Tensor 叶子建立 `MultiOutput`。`fallback_handler` 最后用 `TensorBox.create` 包装这些 IRNode，才形成 Lowering 交回父图的 tensor 值：

```python
def handler(*args, **kwargs):
    def wrap_tensors(x):
        return TensorBox.create(x) if isinstance(x, ir.IRNode) else x

    return pytree.tree_map(
        wrap_tensors, FallbackKernel.create(kernel, *args, **kwargs)
    )
```

以下是 `FallbackKernel.create` 的完整方法；`process_kernel` 的内部参数扁平化属于这个类的共享前置机制，本篇关注它交回的五项数据怎样被 create 消费。

继续代入 `histc(x,bins=4,min=0,max=4)`：kernel 是 `aten.histc.default`，原始位置参数是 `(x,)`，kwargs 含三个标量；x 已物化为输入 TensorBox。`process_kernel` 用对应 FakeTensor 得到 shape `[4]`、float32 的 `example_output`，并交回运行时调用所需的 `tensor_args=[x]`、非 tensor 参数和参数重建信息。create 因而登记一个 FallbackKernel operation，再建立一个 `indices=[]` 的 MultiOutput 指向根返回值。

```python
@classmethod
def create(cls, kernel, *args, **kwargs):
    fake_incorrect_kernels = (aten._fused_moving_avg_obs_fq_helper_functional,)
    # 教学：通常在当前 fake mode 中运行；已知 fake 不正确的特例退出该 context。
    context: ContextManager[None] = (
        V.graph.fake_mode if kernel not in fake_incorrect_kernels else nullcontext()  # type: ignore[assignment]
    )
    with context:
        (
            example_output,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            unbacked_bindings,
        ) = cls.process_kernel(kernel, *args, **kwargs)

    # 教学：输入或 fake 输出提供 operation 所属 device。
    device = cls.find_device(tensor_args, example_output)
    if example_output is None:
        # 教学：没有返回值仍建立 operation，但它没有可布局的 Tensor 输出。
        packed = cls(
            NoneLayout(device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            unbacked_bindings=unbacked_bindings,
        )

    else:
        assert device, "Not sure where to find device info"
        # 教学：有返回值时，packed 节点只声明“多输出容器”；各 Tensor 稍后各建布局。
        packed = cls(
            MultiOutputLayout(device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            unbacked_bindings=unbacked_bindings,
        )

    def generate_output(output, indices):
        if isinstance(output, (list, tuple)):
            # 教学：保留容器类型，并把“容器类型+下标”追加到取值路径。
            return type(output)(
                generate_output(output[i], indices + [(type(output), i)])
                for i in range(len(output))
            )
        elif isinstance(output, dict):
            # 教学：字典用 key 继续递归，结果仍保持相同 key。
            return {
                key: generate_output(val, indices + [(type(output), key)])
                for key, val in output.items()
            }
        elif isinstance(output, torch.Tensor):
            # 教学：每个 Tensor leaf 得到独立布局，但共同引用同一个 packed 调用。
            return MultiOutput(
                cls.tensor_to_layout(output),
                packed,
                indices,
            )
        elif isinstance(output, int):
            # 教学：普通整数 fake 结果可直接成为编译期值。
            return output
        elif isinstance(output, torch.SymInt):
            # 教学：符号整数保留表达式，而非包装成 Tensor 输出。
            return output.node.expr
        else:
            assert (
                output is None
            ), f"FallbackKernel output type {type(output)} is not supported"
            return None

    outputs = generate_output(example_output, [])
    if isinstance(outputs, (list, tuple, dict)):
        packed.outputs = outputs  # type: ignore[assignment]
    else:
        packed.outputs = [outputs]
    return outputs
```

`example_output` 不是实际计算结果；它是 fake execution 得到的接口样本，提供 shape、stride、dtype、device 和返回容器结构。真正的 `histc` 数值仍在生成程序调用 operator 时产生。

### 单输出 histc：路径为空也要建立 MultiOutput

输入 `x=[0.5,1.5,1.8,3.2]`，参数为 `bins=4,min=0,max=4`。`process_kernel` 交回一项 shape `[4]` 的 fake Tensor。状态依次是：

| 步骤 | `example_output` / `indices` | 新对象 |
| --- | --- | --- |
| fake 调用后 | Tensor `[4]` / 尚无 | 尚未建 packed |
| 选择 layout | 非 None | `FallbackKernel(MultiOutputLayout)` |
| `generate_output` 根节点 | Tensor / `[]` | `MultiOutput(layout=[4], input=packed, indices=[])` |
| `create` 返回 | 单个 MultiOutput | 取根返回值的图内引用 |
| handler 包装 | 同一个 MultiOutput / `[]` | `TensorBox → StorageBox → MultiOutput` 交回父图 |

空 `indices=[]` 表示 operator 的根返回值本身就是 Tensor，不需要先取 tuple 下标。实验 07 实际观察到 operations 为 `[FallbackKernel, MultiOutput]`，运行数值为 `[1,2,0,1]`。

### tuple 输出 sort：一次调用，两条取值路径

独立实验 [`23_fallback_output_tree.py`](../../../experiments/23_fallback_output_tree.py) 增加 `torch.sort([3,1,2])`。当前实现实际得到：

```text
FallbackKernel(buf0)
├─ MultiOutput(buf1, indices=[(tuple,0)])  -> values  [1,2,3]
└─ MultiOutput(buf2, indices=[(tuple,1)])  -> indices [1,2,0]
```

递归时根节点是 tuple：第 0 项追加 `(tuple,0)`，第 1 项追加 `(tuple,1)`；两项都是 Tensor，于是各自建立 `MultiOutput`，但它们的 `input` 都是同一个 packed operation。packed 节点生成一次 sort 调用；两个 `MultiOutput.codegen()` 分别把同一返回值的 tuple 第 0、1 项绑定为自己的名字。

### K02 理解验收

问题：若 fake 返回 `{"value": tensor, "count": 3}`，图中会建立几个 MultiOutput？

一个。字典分支为 value 追加 `(dict,"value")`，Tensor leaf 建立 MultiOutput；`count` 进入 int 分支，直接保留 3。packed operation 的 `outputs` 保存整个字典结构。

问题：若 fake 返回一个当前不支持的字符串，失败前是否已经可能建立 packed operation？

按这个方法内部的语句顺序，代码先构造 `FallbackKernel(MultiOutputLayout)`，随后 `generate_output` 才因字符串不是 None 而断言失败。调用者不会得到 outputs；本文没有展开外围异常处理，因此不能继续断言该临时登记是否会留在一次失败编译的最终图状态中。

K02 到此完成。下一知识点 K03 打开 `_has_aliased_buffers` 与 `Conditional`，追踪两支子图从 Lowering 到统一父输出的全部状态。

## K03：cond 先完成两支 Lowering，再承诺一个统一父接口

父图后续只看见一组结果，因此设计上需要 true/false 两支交出相同的输出接口，并禁止分支输出互相 alias。当前实现完整比较两支输出的数量、size、stride、device、dtype 和 offset；alias 部分却只实际检查 true 分支，后面的源码会逐句看到这个边界。编译期并不执行 predicate 来删掉另一支：两支都先 Lower，运行时 wrapper 才按 predicate 选择其中一支。

`Subgraph` 是这段实现保存分支的容器：`graph_module` 是尚未 Lower 的 FX `GraphModule`，也就是可执行的 FX 图及其 Python 模块外壳；`graph` 初始为 None。第一次使用时，`make_subgraph` 为它建立独立的 `GraphLowering`，`run()` 把该分支的 FX 节点 Lower 成子图 operations，并把返回的 IRNode 列表留在 `graph_outputs`。`V.set_graph_handler(subgraph.graph)` 会在代码块内把“当前登记目标”切到这张子图，退出代码块后再恢复父图。

```python
@dataclasses.dataclass
class Subgraph(IRNode):
    name: str
    graph_module: torch.fx.GraphModule
    graph: Optional[GraphLowering] = None
```

### alias 辅助函数比较的是 IRNode 对象身份

在 `Conditional` 之前，产品先定义这个完整辅助函数：

```python
def _has_aliased_buffers(buffers: Sequence[IRNode]) -> bool:
    buffers = [
        # 教学：ReinterpretView 先还原到被观察的底层对象，避免两个 view 掩盖共享身份。
        buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
        for buffer in buffers
    ]
    # assuming the same buffer is represented by the same IRNode object
    # 教学：对象 id 去重后数量减少，说明至少两项输出引用同一 IRNode。
    return len(OrderedSet(id(buffer) for buffer in buffers)) < len(buffers)
```

例如输出 `[a,a]` 的 id 列表是 `[17,17]`，去重后只有一项，返回 True；`[a,b]` 是 `[17,29]`，数量不变，返回 False。若两项是包在 `ReinterpretView` 中的同一 a，`unwrap_view()` 后仍能识别共享。它依赖“同一 buffer 由同一 IRNode 对象表示”的假设，不是一项能识别所有运行时共享存储的通用 alias 分析。

### Conditional 完整实现

先固定条件例：predicate 是 bool 标量 p，operands=[x]，x 为 `[2]` float32；true 子图输出 `x+1`，false 子图输出 `x-1`，两边 Lowering 后各有一个 `[2]`、stride `[1]`、CPU float32 ComputedBuffer，父图 operations 起初不含 Conditional。`fake_operands` 是 x 的 FakeTensor 版本。下面构造器分别 Lower 两支并逐项比较输出接口，通过后登记一项 Conditional 和一个 MultiOutput；运行时只执行 p 选中的一支。

```python
@dataclasses.dataclass
class Conditional(ExternKernel):
    predicate: Optional[IRNode] = None
    operands: Optional[List[TensorBox]] = None
    true_subgraph: Optional[Subgraph] = None
    false_subgraph: Optional[Subgraph] = None
    outputs: Optional[List[MultiOutput]] = None

    def __init__(
        self,
        predicate: IRNode,
        operands: List[TensorBox],
        true_subgraph: Subgraph,
        false_subgraph: Subgraph,
        layout: MultiOutputLayout,
    ):
        # 教学：保存 wrapper codegen 需要的 predicate、操作数和两张已 Lower 子图。
        self.predicate = predicate
        self.operands = operands
        self.true_subgraph = true_subgraph
        self.false_subgraph = false_subgraph

        inputs = []
        if not isinstance(predicate, ShapeAsConstantBuffer):
            # 教学：Tensor predicate 是运行时依赖；shape 常量不作为 buffer 输入。
            inputs.append(predicate)
        inputs.extend(operands)

        super().__init__(
            name=None,
            layout=layout,  # type: ignore[arg-type]
            inputs=inputs,  # type: ignore[list-item]
        )

        # 教学：ExternKernel 的通用初始化只保存公共字段；这两句才把父节点登记进当前图。
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(
        cls,
        predicate: TensorBox,
        true_fn: Subgraph,
        false_fn: Subgraph,
        operands: List[TensorBox],
    ):
        # 教学：父控制节点与子图接口都需要真实存储，先物化 predicate 和 operands。
        predicate = cls.realize_input(predicate)
        operands = [cls.realize_input(x) for x in operands]

        fx_operands = V.graph.current_node.args[-1]
        fake_operands = [x.meta["val"] for x in fx_operands]  # type: ignore[union-attr]

        for subgraph in (true_fn, false_fn):
            if subgraph.graph is None:
                # create and lower subgraphs
                # 教学：每个 Subgraph 第一次使用时建立自己的 GraphLowering。
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=fake_operands,
                    subgraph_name=subgraph.name,
                )
                with V.set_graph_handler(subgraph.graph):
                    # 教学：切换 graph handler，使子图产生的 operation 归入子 GraphLowering。
                    subgraph.graph.run(*fake_operands)

        true_outputs = true_fn.graph.graph_outputs  # type: ignore[union-attr]
        false_outputs = false_fn.graph.graph_outputs  # type: ignore[union-attr]

        for name, outputs in (("true_fn", true_outputs), ("false_fn", false_outputs)):
            if _has_aliased_buffers(true_outputs):
                raise AssertionError(
                    "Output aliasing is currently not supported in compiled torch.cond. "
                    f"The outputs of the {name} subgraph of torch.cond are aliased: {outputs}"
                )

        # make sure true and false outputs are structurally equivalent
        assert len(true_outputs) == len(false_outputs), (true_outputs, false_outputs)
        for i, (to, fo) in enumerate(zip(true_outputs, false_outputs)):
            # 教学：逐输出比较父图建立统一 layout 所需的全部属性。
            assert to.get_size() == fo.get_size(), (i, to, fo)
            assert to.get_stride() == fo.get_stride(), (i, to, fo)
            assert to.get_device() == fo.get_device(), (i, to, fo)
            assert to.get_dtype() == fo.get_dtype(), (i, to, fo)
            assert to.get_layout().offset == fo.get_layout().offset, (i, to, fo)

        if not isinstance(predicate, ShapeAsConstantBuffer):
            # use predicate device for consistent codegen-ing
            device = predicate.get_device()
        else:
            # predicate is not a Tensor: use first operand's device
            assert (
                len(operands) > 0
            ), "When predicate is not a Tensor, there must be at least one operand in torch.cond."
            device = operands[0].get_device()

        # 教学：至此所有失败检查已通过，才登记父 Conditional operation。
        conditional = Conditional(
            predicate=predicate,
            operands=operands,
            true_subgraph=true_fn,
            false_subgraph=false_fn,
            layout=MultiOutputLayout(device),
        )

        outputs = [
            MultiOutput(
                FixedLayout(
                    device=output.get_device(),
                    dtype=output.get_dtype(),
                    size=output.get_size(),
                    stride=output.get_stride(),
                    offset=output.get_layout().offset,
                ),
                conditional,
                [(list, i)],
            )
            # as the true and false outputs are equivalent,
            # we can use either of them here as a "template"
            # 教学：合同已证明等价，所以用 true 输出逐项复制父输出 layout。
            for i, output in enumerate(true_outputs)
        ]

        conditional.outputs = outputs
        return outputs

    def codegen(self, wrapper):
        # 教学：运行时分支由 wrapper 表达，本节点自身不生成两支算术代码。
        wrapper.codegen_conditional(self)
```

Python wrapper 展开这项调用时，生成的关键控制结构是：

```python
predicate = conditional.predicate.codegen_reference()
if not isinstance(conditional.predicate, ShapeAsConstantBuffer):
    predicate = f"{predicate}.item()"

self.writeline(f"if {predicate}:")
self.codegen_subgraph(conditional.true_subgraph, outer_inputs, outer_outputs)
self.writeline("else:")
self.codegen_subgraph(conditional.false_subgraph, outer_inputs, outer_outputs)
```

因此两张子图都在编译期建立，但生成程序的一次运行只进入 predicate 对应的一个分支；该分支把结果写入同一组 `outer_outputs`。

### 用 add/sub 两支推进状态

实验 07 的请求是：predicate 为 true 时返回 `x+1`，false 时返回 `x-1`，`x=[2,4]`。

| 时刻 | true_fn | false_fn | 父图 operations |
| --- | --- | --- | --- |
| 进入 create | 只有 Subgraph/GraphModule | 只有 Subgraph/GraphModule | 尚无 Conditional |
| 第一轮 run 后 | `ComputedBuffer(Pointwise add)`，输出 `[2]` float32 contiguous | 尚未 Lower | 尚无 Conditional |
| 第二轮 run 后 | 保持 | `ComputedBuffer(Pointwise sub)`，同接口 | 尚无 Conditional |
| 合同检查后 | 数量和五项属性逐项相同 | 数量和五项属性逐项相同 | 仍尚无 Conditional |
| 调用构造器 | 作为 true_subgraph 保存 | 作为 false_subgraph 保存 | 新增 `Conditional(buf0)` |
| 建立输出 | 提供 layout 模板 | 运行时备选值 | 新增 `MultiOutput(buf1,[(list,0)])` |

编译一次后，以 true 和 false 分别运行，得到 `[3,5]` 与 `[1,3]`。编译期建立两张子图；生成的 wrapper 在运行时读取 predicate，只调用被选中的子图。

### 失败发生在哪个状态

若 false branch 返回同 shape、dtype、device 但 stride 不同的 view，代码在逐项 `get_stride()` 断言失败。此时两张子 GraphLowering 都已经存在，各自 operation 也已经建立；父 `Conditional` 尚未调用构造器，因此父图没有 Conditional/MultiOutput 半成品。

当前源码的 alias 循环有一个需要按实际语句理解的边界：循环变量 `outputs` 依次是 true 和 false 输出，但判断始终调用 `_has_aliased_buffers(true_outputs)`。因此 true 分支 alias 会在第一轮被拒绝；只有 false 分支 alias 时，这段代码不会拒绝它。第二轮也不会单独触发 alias 异常：true 有 alias 时第一轮已经抛出，true 无 alias 时第二轮仍然为假。接口属性检查则确实逐项比较 true 与 false。

父节点的建立边界不受这个细节影响：两张子图先完成 Lowering；当前实现执行完 alias 与接口检查后才构造 `Conditional`。在此之前失败，不会留下父 `Conditional` 或它的 `MultiOutput`。

### K03 理解验收

问题：两支 shape 都是 `[2,3]`，为何 false 返回 transpose view 仍可能失败？

shape 只决定合法逻辑位置。父 MultiOutput 还需要一份固定 stride 和 offset；若 true 为 `[3,1]`、false 为 `[1,2]`，同一个父 layout 无法同时代表两者，所以在 Conditional 登记之前失败。

问题：predicate 是 `ShapeAsConstantBuffer` 且 operands 为空，会在哪一步失败？

两支 Lowering 与合同检查先执行；选择父节点 device 时进入 shape 常量分支，因没有第一个 operand 而断言。失败仍早于 `Conditional(...)` 构造和父 operation 登记。

K03 到此完成。下一知识点 K04 使用同样方法展开 WhileLoop，并重点解释 carried/body 合同和零轮循环为何改变 buffer 复用状态。

## K04：while 的 body 输出必须能原样成为下一轮 carried 输入

`cond` 的两支输出只需要彼此等价；`while_loop` 的关键关系不同：body 第 i 项输出会成为下一轮第 i 项 carried input。因此编译器要验证一条循环不变量：每轮前后，每个 carried slot 的 size、stride、device、dtype 和 offset 都保持不变。

### WhileLoop 完整实现

本次循环的 carried inputs 是 `i=0` 与 `x=[1,2]`，没有 additional inputs；cond 子图返回 bool 标量 `i<2`，body 返回 `i+1` 与 `x+1`。`carried_inputs` 每轮都会被 body 输出按位置替换；`additional_inputs` 同样传给 cond 和 body，但每轮保持不变。两项 body 输出的 shape、stride、device、dtype 分别与下一轮的 i、x 输入一致。编译期先用 fake inputs Lower cond/body 并检查这组对应关系，再登记 WhileLoop 与两个 MultiOutput。若初始 i 已为 2，循环零次，最终 x 仍可能直接引用 graph input；因此对应输出的 buffer 名会加入 `never_reuse_buffers`。

```python
@dataclasses.dataclass
class WhileLoop(ExternKernel):
    carried_inputs: Optional[List[TensorBox]] = None
    additional_inputs: Optional[List[TensorBox]] = None
    cond_subgraph: Optional[Subgraph] = None
    body_subgraph: Optional[Subgraph] = None
    outputs: Optional[List[MultiOutput]] = None

    def __init__(
        self,
        carried_inputs: List[TensorBox],
        additional_inputs: List[TensorBox],
        cond_subgraph: Subgraph,
        body_subgraph: Subgraph,
        layout: MultiOutputLayout,
    ):
        # 教学：carried 每轮更新；additional 每轮只读，但二者都是父节点依赖。
        self.carried_inputs = carried_inputs
        self.additional_inputs = additional_inputs
        self.cond_subgraph = cond_subgraph
        self.body_subgraph = body_subgraph

        super().__init__(
            name=None,
            layout=layout,  # type: ignore[arg-type]
            inputs=carried_inputs + additional_inputs,  # type: ignore[list-item]
        )

        # 教学：所有合同通过后，父 while 才登记 buffer 和 operation。
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(
        cls,
        cond_fn: Subgraph,
        body_fn: Subgraph,
        carried_inputs: List[TensorBox],
        additional_inputs: List[TensorBox],
    ):
        # 教学：子图调用和父控制节点都需要真实接口，先物化全部输入。
        carried_inputs = [cls.realize_input(x) for x in carried_inputs]
        additional_inputs = [cls.realize_input(x) for x in additional_inputs]
        all_inputs = carried_inputs + additional_inputs

        fx_all_inputs = V.graph.current_node.args[-2] + V.graph.current_node.args[-1]  # type: ignore[operator]
        fake_all_inputs = [x.meta["val"] for x in fx_all_inputs]  # type: ignore[union-attr]

        for subgraph in (cond_fn, body_fn):
            if subgraph.graph is None:
                # create and lower subgraphs
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=fx_all_inputs,  # type: ignore[arg-type]
                    subgraph_name=subgraph.name,
                )
                with V.set_graph_handler(subgraph.graph):
                    # 教学：cond/body operation 各自登记到对应子 GraphLowering。
                    subgraph.graph.run(*fake_all_inputs)

        cond_outputs = cond_fn.graph.graph_outputs  # type: ignore[union-attr]
        body_outputs = body_fn.graph.graph_outputs  # type: ignore[union-attr]

        if _has_aliased_buffers(body_outputs):
            raise AssertionError(
                "Output aliasing is currently not supported in compiled torch.while_loop. "
                f"The outputs of the body_fn subgraph of torch.while_loop are aliased: {body_outputs}"
            )

        # make sure cond_fn returns a boolean scalar Tensor
        # 教学：cond 必须恰好交回一项、dtype bool、size 列表为空，即零维标量。
        assert len(cond_outputs) == 1, cond_outputs
        assert cond_outputs[0].get_dtype() == torch.bool, cond_outputs
        assert len(cond_outputs[0].get_size()) == 0, cond_outputs

        assert (
            len(all_inputs) > 0
        ), "torch.while_loop is assumed to have at least one operand."

        device = all_inputs[0].get_device()

        # make sure carried_inputs and body outputs are structurally equivalent
        assert len(carried_inputs) == len(body_outputs), (carried_inputs, body_outputs)
        for i, (op, bo) in enumerate(zip(carried_inputs, body_outputs)):
            # 教学：body 输出必须能直接回到同一 carried slot，逐项保持接口。
            assert op.get_size() == bo.get_size(), (i, op, bo)
            assert op.get_stride() == bo.get_stride(), (i, op, bo)
            # assume all carried_inputs and outputs are on the same device
            # as the MultiOutputLayout below requires single device
            assert op.get_device() == bo.get_device() == device, (i, op, bo, device)
            assert op.get_dtype() == bo.get_dtype(), (i, op, bo)
            assert op.get_layout().offset == bo.get_layout().offset, (i, op, bo)

        while_loop = WhileLoop(
            carried_inputs=carried_inputs,
            additional_inputs=additional_inputs,
            cond_subgraph=cond_fn,
            body_subgraph=body_fn,
            # asserted above that there is at least one operand
            layout=MultiOutputLayout(device),
        )

        outputs = [
            MultiOutput(
                FixedLayout(
                    device=output.get_device(),
                    dtype=output.get_dtype(),
                    size=output.get_size(),
                    stride=output.get_stride(),
                    offset=output.get_layout().offset,
                ),
                while_loop,
                [(list, i)],
            )
            # 教学：body 已通过循环不变量检查，可作为最终输出 layout 模板。
            for i, output in enumerate(body_outputs)
        ]

        for inp, out in zip(carried_inputs, outputs):
            if inp.get_name() in V.graph.graph_inputs:
                # if a carried input of the while_loop is a graph input,
                # it can be returned as is when the number of iterations
                # is zero. due to this, we can't (generally) reuse the
                # output buffers corresponding to the graph inputs, as
                # the inputs may end up being mutated.
                # 教学：零轮时 out 可能仍代表调用者输入，禁止将其输出名用于存储复用。
                V.graph.never_reuse_buffers.add(out.get_name())

        while_loop.outputs = outputs
        return outputs

    def codegen(self, wrapper):
        wrapper.codegen_while_loop(self)
```

这里有两份同位置的输入。`fx_all_inputs` 是父 FX 节点中的输入节点，当前 `WhileLoop.create` 原样把它传给 `make_subgraph(..., example_inputs=...)`；`fake_all_inputs` 是逐个读取 `node.meta["val"]` 得到的 FakeTensor，随后作为参数传给子图的 `run()`。cond 和 body 都按 `carried + additional` 的相同顺序接收这组参数。

这与上一节 `Conditional.create` 不完全相同：Conditional 把 `fake_operands` 同时传给 `make_subgraph` 和 `run`，WhileLoop 则把 FX 节点传给前者、FakeTensor 传给后者。这是当前两段源码的实际差异；本篇不从这两处调用推断差异的设计原因。两条路径真正执行子图 Lowering 时，`run(...)` 收到的都是 FakeTensor。

### 三轮与零轮使用同一份已编译循环

下面改用终止条件 `i<3` 展示三轮。独立实验 [`24_while_loop_state.py`](../../../experiments/24_while_loop_state.py) 编译下面的关系：

```text
carried = (i, x)
cond(i,x): i < 3
body(i,x): (i+1, x+2)
```

初始 `i=0,x=[1,2]` 时，状态为：

| cond 检查前 | cond | body 后 |
| --- | --- | --- |
| `(0,[1,2])` | true | `(1,[3,4])` |
| `(1,[3,4])` | true | `(2,[5,6])` |
| `(2,[5,6])` | true | `(3,[7,8])` |
| `(3,[7,8])` | false | 不再执行，返回当前 carried |

同一编译函数传入 `i=3,x=[1,2]`，第一次 cond 就为 false，结果保持 `(3,[1,2])`。编译轨迹实际包含 cond 子图、body 子图、父 `WhileLoop` 和两个 `MultiOutput`；两项 carried 都来自图输入，因此这两个输出名都进入 `never_reuse_buffers`。

实验的最小观测是：

```text
输入 (i=0, x=[1,2]) → 输出 (i=3, x=[7,8])
输入 (i=3, x=[1,2]) → 输出 (i=3, x=[1,2])
父图 operation：WhileLoop + 两个 MultiOutput
子图：一张 cond GraphLowering，一张 body GraphLowering
```

这里需要区分两个时间：编译期 body 用 fake 输入运行一次来建立 IR 和验证接口；运行期 body 可执行零次、一次或多次。`never_reuse_buffers` 保护的正是运行期零次这一可能性。

加入只读参数后，位置关系是：

```text
carried    = (i, x)
additional = (limit, step)

cond(i, x, limit, step): i < limit
body(i, x, limit, step): (i + 1, x + step)

本轮输入：(0, [1,2], 2, 3)
body 输出：(1, [4,5])
下一轮输入：(1, [4,5], 2, 3)
```

下一轮只替换 carried 槽位 i、x；limit、step 仍取父节点传入的原值。body 只返回两项 carried，因此父 `WhileLoop` 最终也只有 i、x 两个 `MultiOutput`，不会把 additional inputs 追加到输出。

例如零轮时输出 `buf_out` 实际仍指向调用者传入的 x。若存储规划又把 `buf_out` 的空间借给后续临时值并写入 0，写坏的就是调用者仍可见的 x。把这个输出名加入 `never_reuse_buffers`，就是禁止这种复用。

### 合同失败怎样截断构造

- cond 返回 bool 向量 `[True]`：数量和 dtype 通过，`len(size)==0` 失败；父 WhileLoop 尚未登记。
- body 把 carried `[2,3]` 转置成 `[3,2]`：在 size 比较失败；即使保持 shape 却改变 stride，也会在下一行失败。
- body 返回同一个 buffer 两次：先在 `_has_aliased_buffers(body_outputs)` 拒绝，尚未检查 cond 合同。
- carried 为空但 additional 非空：`all_inputs` 非空检查通过，但 body 输出数量必须为 0；这种接口是否由上游允许还受 HOP 合同约束。本方法本身并不把“carried 至少一项”写成断言。

### K04 理解验收

问题：为什么仅在 carried input 是 `graph_inputs` 成员时，把对应输出加入 `never_reuse_buffers`？

零轮循环返回原 carried。若它来自调用者，输出可能仍与调用者可见存储同一身份；复用该输出空间可能覆盖外部数据。对于不是 graph input 的内部 carried，这段零轮规则不会把对应输出加入 `never_reuse_buffers`；它是否受其他 alias 或生命周期规则限制，不由这里的条件分支决定。

问题：body 的输出 shape、stride、dtype 都相同，但 device 改了，是否可以等下一轮再搬回？

不可以。当前 create 要求 `op.device == bo.device == device`，没有插入搬运的分支；断言发生在父 WhileLoop 构造前。要支持搬回需要另一项明确实现，不能从这段代码推断存在。

K04 到此完成。候选和 fallback 通过具体 operation 或 packed return 留在父 Lowering 图中；控制流也在父图留下 `Conditional` 或 `WhileLoop` operation，而分支、cond 和 body 的 operation 分别保存在各自的子 `GraphLowering` 中，供父 operation 在生成代码时引用。
