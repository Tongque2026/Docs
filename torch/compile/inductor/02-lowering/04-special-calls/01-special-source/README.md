# 特殊调用分派核心源码：候选怎样变成 IR operation

[返回特殊调用篇](../README.md) · [返回 Lowering](../../README.md) · [返回总览](../../../README.md)

Q24 已经解释了三类结果，本篇打开它们的实现。先从算法选择路径开始，因为这里最容易把三个不同对象混成“模板”：

```text
ChoiceCaller：一份候选，保存输入、输出 layout 和“选中后怎样建 IR”
       |
       | output_node()
       v
TemplateBuffer / ExternKernel：已经进入 Lowering 图的一项 operation

MultiTemplateBuffer：也已经是 operation，但内部 render 仍可等 Scheduler
                     看见 epilogue 后再固定
```

## K01：候选被选中后，才建立对应的 IR 节点

设两个候选接收相同的 A、B，输出 layout 都是 4×4 float32。一个 caller 指向 Triton template，另一个指向 `at::mm_out`。caller 本身不会出现在 `GraphLowering.operations`；调用 `output_node()` 后，才产生可登记、调度和生成的 IR。

Triton caller 的转换只有一条路径。下面保留当前产品方法的完整语句，并用 `# 教学` 注释指出数据交接：

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

## 已建立的判断

若同一个 extern choice 把 `has_out_variant` 从 `True` 改为 `False`，最后一支只改变 `cls`：图中得到 `ExternKernelAlloc`，表示外部函数自己返回存储；输入和 choice 身份没有因此变成 template。

若 `abi_compatible=True` 但 choice 没有 `op_overload`，代码在建立 FallbackKernel 之前触发断言，所以图里不会留下半成品 operation。这说明失败出口发生在 `inner` 建立之前。

## MultiTemplateBuffer：operation 已存在，render 仍可延迟固定

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

当 selector 选择“先把多个候选保留进图”，底层 operation 是下面这个完整类：

本次对象的 `layout` 是 4×4 float32 输出，`inputs=[x,w]`，候选表有 Triton choice T 与 extern choice E；尚未附加 epilogue 时测得 `choice_timings={T:2.4ms,E:1.1ms}`，所以 `min_choice=E`。`finalize_as_triton_caller(T)` 会把 operation 固定为 T 并取得对应 render；`finalize_as_extern_caller(E)` 则把同一个图中 operation 改成 extern 调用。若 Scheduler 后来尝试融合 relu epilogue，它会重新比较“各候选 + relu”，而不是无条件沿用基础计时的 E。

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

独立实验 [`22_multi_template_state.py`](../../../experiments/22_multi_template_state.py) 直接调用当前产品类的 `choice_timings` 和 `get_min_choice`，验证计时函数只执行一次：第一次与第二次均选择 `triton_32, 5.0`，计数仍为 1。实验没有伪造 Triton caller，所以不把 `swap/finalize` 声称为本机运行；那两条状态由上面的当前源码顺序支持。

### K01 理解验收

问题：第一次 `get_min_choice()` 得到 triton 最快，能否立即断言图中已有最终 Triton render？

不能。此时 `choice_timings` 只从 `None` 变成缓存字典，`get_min_choice` 只返回 `(caller,time)`，没有写 `make_kernel_render`。只有 `finalize_as_triton_caller` 才永久安装 render；若 Scheduler 要比较 epilogue，它还可能先在 `swap_as_triton_caller` 的临时状态中测试。

问题：若显式传入 `return_multi_template=False`，入口会不会被全局配置改回 True？

不会。`autotune_select_algorithm` 只在 kwargs 缺少该键时填默认值。显式 False 保留下来，selector 按这次调用的要求处理。

K01 到此完成：caller、具体 operation 和“已入图但 render 未固定”的 operation 已经分开。下一知识点 K02 从 `make_fallback(histc)` 开始，追踪 packed operation 与嵌套输出树。

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

`histc` 传入的是 overload packet，因此循环把它的 overload 分别登记。登记同时调用 `add_needs_realized_inputs`：外部 operator 运行时需要真实 tensor，不能接收一条尚未执行的 Pointwise 规则。

### create 先用 fake 值确定接口，再建立 packed operation

处理器命中以后进入 `FallbackKernel.create`。以下是完整方法；`process_kernel` 的内部参数扁平化属于这个类的共享前置机制，本篇关注它交回的五项数据怎样被 create 消费。

继续代入 `histc(x,bins=4,min=0,max=4)`：kernel 是 `aten.histc.default`，`unflattened_args=(x,)`，kwargs 含三个标量；x 已物化为输入 TensorBox。`process_kernel` 用对应 FakeTensor 得到 shape `[4]`、float32 输出，`tensor_args=[x]`，没有 mutated args 或 alias names，output_view 描述“packed 返回的第 0 项就是唯一 tensor”。create 因而登记一个 FallbackKernel operation，再建立一个 MultiOutput 指向它的第 0 项。

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
| 返回 | 单个 MultiOutput | `TensorBox → StorageBox → MultiOutput` 供父图引用 |

空 `indices=[]` 表示 operator 的根返回值本身就是 Tensor，不需要先取 tuple 下标。实验 07 实际观察到 operations 为 `[FallbackKernel, MultiOutput]`，运行数值为 `[1,2,0,1]`。

### tuple 输出 sort：一次调用，两条取值路径

独立实验 [`23_fallback_output_tree.py`](../../../experiments/23_fallback_output_tree.py) 增加 `torch.sort([3,1,2])`。当前实现实际得到：

```text
FallbackKernel(buf0)
├─ MultiOutput(buf1, indices=[(tuple,0)])  -> values  [1,2,3]
└─ MultiOutput(buf2, indices=[(tuple,1)])  -> indices [1,2,0]
```

递归时根节点是 tuple：第 0 项追加 `(tuple,0)`，第 1 项追加 `(tuple,1)`；两项都是 Tensor，于是各自建立 `MultiOutput`，但它们的 `input` 都是同一个 packed operation。生成程序因此调用一次 sort，再按路径取得两项，而不是生成两次 sort 调用。

### K02 理解验收

问题：若 fake 返回 `{"value": tensor, "count": 3}`，图中会建立几个 MultiOutput？

一个。字典分支为 value 追加 `(dict,"value")`，Tensor leaf 建立 MultiOutput；`count` 进入 int 分支，直接保留 3。packed operation 的 `outputs` 保存整个字典结构。

问题：若 fake 返回一个当前不支持的字符串，失败前是否已经可能建立 packed operation？

是。代码先根据非 None 输出建立 `FallbackKernel(MultiOutputLayout)`，随后 `generate_output` 才走到最后一个分支并因字符串不是 None 而断言失败。调用者不会得到 outputs，但与 `make_fallback` 的非法输入类型不同，这个失败发生在 packed 构造之后。

K02 到此完成。下一知识点 K03 打开 `_has_aliased_buffers` 与 `Conditional`，追踪两支子图从 Lowering 到统一父输出的全部状态。

## K03：cond 先完成两支 Lowering，再承诺一个统一父接口

父图后续只看见一组结果，因此 true/false 两支必须交出相同数量、形状、步长、设备、类型和起点的值。编译期并不执行 predicate 来删掉另一支：两支都先 Lower，运行时 wrapper 才按 predicate 选择其中一支。

### alias 检查比较的是底层 IRNode 身份

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

例如输出 `[a,a]` 的 id 列表是 `[17,17]`，去重后只有一项，返回 True；`[a,b]` 是 `[17,29]`，数量不变，返回 False。若两项是包在 `ReinterpretView` 中的同一 a，`unwrap_view()` 后仍能识别共享。

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

        # 教学：只有合同检查完成并调用构造器后，才把父节点登记为 buffer/operation。
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
            # 教学：保留当前实现事实——两轮都把 true_outputs 传给辅助函数。
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

编译一次后，以 true 和 false 分别运行，得到 `[3,5]` 与 `[1,3]`。这证明运行时确实选择不同子图；编译轨迹只在第一次编译时出现两支 Lowering。

### 失败发生在哪个状态

若 false branch 返回同 shape、dtype、device 但 stride 不同的 view，代码在逐项 `get_stride()` 断言失败。此时两张子 GraphLowering 都已经存在，各自 operation 也已经建立；父 `Conditional` 尚未调用构造器，因此父图没有 Conditional/MultiOutput 半成品。

两支还必须满足源码实现的 alias 检查，避免父输出槽位无法唯一代表实际 storage。正常阅读路径只依赖已经明确的出口：接口检查全部通过才建立父 Conditional；任一检查失败，都不会留下父节点半成品。

### K03 理解验收

问题：两支 shape 都是 `[2,3]`，为何 false 返回 transpose view 仍可能失败？

shape 只决定合法逻辑位置。父 MultiOutput 还需要一份固定 stride 和 offset；若 true 为 `[3,1]`、false 为 `[1,2]`，同一个父 layout 无法同时代表两者，所以在 Conditional 登记之前失败。

问题：predicate 是 `ShapeAsConstantBuffer` 且 operands 为空，会在哪一步失败？

两支 Lowering 与合同检查先执行；选择父节点 device 时进入 shape 常量分支，因没有第一个 operand 而断言。失败仍早于 `Conditional(...)` 构造和父 operation 登记。

K03 到此完成。下一知识点 K04 使用同样方法展开 WhileLoop，并重点解释 carried/body 合同和零轮循环为何改变 buffer 复用状态。

## K04：while 的 body 输出必须能原样成为下一轮 carried 输入

`cond` 的两支输出只需要彼此等价；`while_loop` 的关键关系不同：body 第 i 项输出会成为下一轮第 i 项 carried input。因此编译器要验证一条循环不变量：每轮前后，每个 carried slot 的 size、stride、device、dtype 和 offset 都保持不变。

### WhileLoop 完整实现

本次循环的 carried inputs 是 `i=0` 与 `x=[1,2]`，没有 additional inputs；cond 子图返回 bool 标量 `i<2`，body 返回 `i+1` 与 `x+1`。两项 body 输出的 shape、stride、device、dtype 分别与下一轮的 i、x 输入一致。编译期先用 fake inputs Lower cond/body 并检查这组对应关系，再登记 WhileLoop 与两个 MultiOutput。若初始 i 已为 2，循环零次，最终 x 仍可能直接引用 graph input；因此对应输出 storage 会加入 never_reuse_buffers。

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
        # 教学：cond 必须恰好交回一项、dtype bool、size 维数为 0。
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

### 三轮与零轮使用同一份已编译循环

独立实验 [`24_while_loop_state.py`](../../../experiments/24_while_loop_state.py) 编译下面的关系：

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

这里需要区分两个时间：编译期 body 用 fake 输入运行一次来建立 IR 和验证接口；运行期 body 可执行零次、一次或多次。`never_reuse_buffers` 保护的正是运行期零次这一可能性。

### 合同失败怎样截断构造

- cond 返回 bool 向量 `[True]`：数量和 dtype 通过，`len(size)==0` 失败；父 WhileLoop 尚未登记。
- body 把 carried `[2,3]` 转置成 `[3,2]`：在 size 比较失败；即使保持 shape 却改变 stride，也会在下一行失败。
- body 返回同一个 buffer 两次：先在 `_has_aliased_buffers(body_outputs)` 拒绝，尚未检查 cond 合同。
- carried 为空但 additional 非空：`all_inputs` 非空检查通过，但 body 输出数量必须为 0；这种接口是否由上游允许还受 HOP 合同约束。本方法本身并不把“carried 至少一项”写成断言。

### K04 理解验收

问题：为什么仅在 carried input 是 `graph_inputs` 成员时，把对应输出加入 `never_reuse_buffers`？

零轮循环返回原 carried。若它来自调用者，输出可能仍与调用者可见存储同一身份；复用该输出空间可能覆盖外部数据。内部临时 carried 的存储不具有这项调用者可见身份，因而这段规则不一律禁止其复用。

问题：body 的输出 shape、stride、dtype 都相同，但 device 改了，是否可以等下一轮再搬回？

不可以。当前 create 要求 `op.device == bo.device == device`，没有插入搬运的分支；断言发生在父 WhileLoop 构造前。要支持搬回需要另一项明确实现，不能从这段代码推断存在。

K04 到此完成。四个知识点共同说明：候选、fallback 与控制流虽然都绕开普通 Loops，但它们分别通过具体 operation、packed return 路径或稳定子图接口留在同一 Lowering 图中。
