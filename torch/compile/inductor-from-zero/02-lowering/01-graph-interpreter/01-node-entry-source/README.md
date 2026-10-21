# GraphLowering 输入登记：从样例参数建立运行时接口

[返回 GraphLowering](../README.md) · [返回 Lowering](../../README.md) · [返回总览](../../../README.md)

FX 的 `placeholder` 只写着一个参数名字，例如 `arg0_1`。后续 lowering 却必须知道运行时从什么设备、什么 dtype、什么 shape 和 stride 的存储读取。`GraphLowering.placeholder` 把样例参数提供的这些属性转换成 Inductor 输入接口；它不会把样例 tensor 的数据值编进计算。

本叶子声明并完整保留这一职责直接使用的五个方法：`symbolic_sizes_strides`、`static_sizes_strides`、`qualify_name`、`add_device_info`、`placeholder`。它们在产品文件中并不相邻，以下仍按 `torch/_inductor/graph.py` 中的原始出现顺序排列；每个方法内部无省略，新增行内说明统一使用 `# 教学：`。实验直接运行当前 `GraphLowering`，而不只从一次完整 compile 的最终结果反推输入状态。

## 一份 2×3 输入最终登记了什么

设样例输入 `x` 为 float32 CPU tensor：

```text
逻辑值（只帮助识别样例）：[[0,1,2], [3,4,5]]
size  = [2,3]
stride= [3,1]
device= cpu
dtype = float32
```

`placeholder("arg0_1", ...)` 形成：

```text
FixedLayout(cpu, float32, [2,3], [3,1])
        ↓
InputBuffer("arg0_1", layout)
        ↓
StorageBox(InputBuffer) → TensorBox
```

然后同一份对象进入 `graph_inputs["arg0_1"]`；裸 `InputBuffer` 身份进入 `graph_inputs_original`，以便函数出口判断输入是否被修改。后续 add 节点从解释器 `env` 取得这个 TensorBox，生成的是“运行时读取 arg0_1”的规则，而不是读取上面列出的 0 到 5。

## 完整源码

先固定首个代码块会看到的类型和值。`ex: torch.Tensor` 是本次样例输入 x；它的 size=`[2,3]`、stride=`[3,1]`、dtype=float32、device=cpu。返回标注中的 `Tuple[List[...],List[...]]` 表示函数交回两张列表：第一张是各维 size，第二张是各维 stride；动态维可用符号表达式，静态维就是整数。`placeholder` 后续用这两张列表建立 layout，再建立名为 `arg0_1` 的 InputBuffer。

第一次逐行阅读先走静态主路径：`reuse_shape_env=True`，当前编译任务已有 `_shape_env`，x 没有符号尺寸，target 为 `arg0_1`，没有子图前缀，`current_node.users` 非空；CPU 输入不参与 GPU 对齐优化，所以 `aligned_inputs` 保持空。遇到另一分支时再换状态：动态 FakeTensor 的 size 为 `[u0,3]`，stride 为 `[3,1]`；若 `reuse_shape_env=False`，本 GraphLowering 会用来源名 `__inductor_unknown_tensor_0` 新建并保存代表第一维的符号，而不是把运行时长度写成某个固定整数。

```python
def symbolic_sizes_strides(
    self, ex: torch.Tensor
) -> Tuple[Union[List[int], List[Expr]], Union[List[int], List[Expr]]]:
    """
    Support dynamic shapes and dynamic strides by assigning variables
    to each dimension.  We duck-shape tensors, so if two tensors
    have the same size they get assigned the same symbolic variable.
    """
    # 教学：已有 ShapeEnv 时沿用其中的符号，保证相等维度复用关系。
    if self.reuse_shape_env:
        return convert_shape_to_inductor(ex.size()), convert_shape_to_inductor(
            ex.stride()
        )
    else:
        from torch._dynamo.source import ConstantSource

        # TODO: this should not be needed once #93059 lands
        # https://github.com/pytorch/pytorch/pull/94031#discussion_r1096044816
        # TODO: make a dedicated UnknownSource for this?
        # NB: This is using the legacy default behavior from
        # create_symbolic_sizes_strides_storage_offset but we hope we can
        # just delete this entirely
        # 教学：没有可复用环境时，为这份未知 tensor 建立符号来源。
        source = ConstantSource(
            f"__inductor_unknown_tensor_{len(self._shape_env.var_to_val)}"
        )
        (
            size,
            stride,
            _,
        ) = self._shape_env.create_symbolic_sizes_strides_storage_offset(
            ex,
            source,
        )

    # 教学：Inductor 后续使用 SymPy 表达式，而不是外层 torch.SymInt 包装。
    size = [i.node.expr if isinstance(i, torch.SymInt) else i for i in size]
    stride = [i.node.expr if isinstance(i, torch.SymInt) else i for i in stride]
    return size, stride

def static_sizes_strides(
    self, ex: torch.Tensor
) -> Tuple[List[sympy.Expr], List[sympy.Expr]]:
    """
    Primarily used to weights
    """
    # 教学：即使维度是具体整数，也统一变成 SymPy Integer 交给 IR。
    size = [sympy.Integer(i) for i in ex.size()]
    stride = [sympy.Integer(i) for i in ex.stride()]
    return size, stride

def qualify_name(self, name: str) -> str:
    """Prepend the given name with the graph name if any."""
    # 教学：子图用前缀避免自己的 arg/buf 名与父图冲突。
    if self.name is not None:
        return f"{self.name}_{name}"
    return name

def add_device_info(self, device: torch.device) -> None:
    # 教学：集合回答整张图涉及哪些设备类型和编号。
    self.device_types.add(device.type)
    if device.index is not None:
        self.device_idxs.add(device.index)
    # 教学：首次遇到某设备时，还记录是哪个当前 FX node 引入它。
    if V.graph.current_node and device not in self.device_node_mapping:
        self.device_node_mapping[device] = V.graph.current_node

def placeholder(
    self, target: str, args: Tuple[object], kwargs: Dict[str, object]  # type: ignore[override]
) -> Union[Expr, TensorBox, None]:
    # 教学：父类按 placeholder 顺序取出相应 example input。
    example = super().placeholder(target, args, kwargs)  # type: ignore[arg-type]
    # 教学：所有输入名先进入有序列表，包括后面返回 None 的特殊输入。
    self.graph_input_names.append(target)
    if isinstance(example, SymTypes):
        # 教学：符号标量直接交出底层 SymPy 表达式，不建立 InputBuffer。
        expr = example.node.expr
        self.graph_inputs[target] = expr
        return expr
    elif isinstance(example, (int, bool, float)):
        # 教学：普通 Python 标量也统一为 SymPy 对象。
        expr = sympy.sympify(example)
        self.graph_inputs[target] = expr
        return expr
    elif example is None:
        return None
    if isinstance(example, BackwardState):
        # Ignored arg, must be unused
        # Alternately we could filter this out in AotAutograd
        return None
    assert isinstance(example, torch.Tensor), example
    # todo(chilli): We can remove the last check once we turn buffers into
    # static shape tensors. That's a hack to workaround Inductor believing
    # the buffer should be static but us passing in a fake tensor with
    # symbolic shapes.
    if not example._has_symbolic_sizes_strides:
        # the first N inputs are weights
        # 教学：静态输入得到具体 SymPy Integer size/stride。
        sizes, strides = self.static_sizes_strides(example)
    else:
        # 教学：动态输入保留 ShapeEnv 中的符号关系。
        sizes, strides = self.symbolic_sizes_strides(example)  # type: ignore[assignment]
    # TODO(jansel): handle input aliasing
    # 教学：到这里才给 tensor 输入加可能的子图前缀。
    target = self.qualify_name(target)
    tensor = TensorBox.create(
        InputBuffer(
            target,
            FixedLayout(example.device, example.dtype, sizes, strides),
        )
    )
    # 教学：一个表供后续按名字找 TensorBox；另一个保留原 InputBuffer 身份。
    self.graph_inputs[target] = tensor
    self.graph_inputs_original[target] = tensor.data.data
    if self.current_node.users:  # cudagraphs should work with an unused CPU input
        # 教学：完全未使用的 CPU 输入不必因此阻止 cudagraphs。
        self.add_device_info(example.device)

    # Note: [Input Alignment handling in Inductor]
    # Alignment matters for generating efficient code. Some operations,
    # e.g. vectorized loads, can only be performed on aligned inputs.
    #
    # But if we codegen assuming aligned inputs and then get unaligned
    # inputs at runtime, then we are forced to clone - which is bad for
    # both perf and memory usage.
    #
    # One option would be to guard on storage_offset%ALIGNMENT, and then
    # codegen based on this. But storage_offset guards turned out to be
    # expensive and cause recompiles; Instead, we're generating code
    # based on the alignment of the example input without guarding.
    with maybe_get_suppress_shape_guards_ctx():
        # 教学：依据样例输入决定生成时是否可假定对齐，并登记输入名。
        if should_assume_input_aligned(example):
            self.aligned_inputs.add(target)
    return tensor
```

源码副本去掉了类级四个空格缩进，方法内部顺序和逻辑保持不变。

## 四类输入分别走到哪里

| example input | `graph_input_names` | `graph_inputs[target]` | 返回给 FX env | InputBuffer |
| --- | --- | --- | --- | --- |
| Tensor | 追加原 target | TensorBox | 同一 TensorBox | 建立 |
| SymInt/SymFloat/SymBool | 追加 | SymPy expr | expr | 不建立 |
| int/bool/float | 追加 | `sympy.sympify` 结果 | expr | 不建立 |
| None / BackwardState | 追加 | 不登记 | None | 不建立 |

这张表解释一个细节：`graph_input_names` 表示 FX 接口顺序，不等同于 `graph_inputs` 的键集合。`None` 与 BackwardState 仍占调用参数位置，但不会成为生成程序读取的 tensor 或标量表达式。

placeholder 接受某一类型，不等于后续 output 一定支持它。实验中 bool 在这里成功登记为 SymPy `BooleanTrue`，随后 output 处理报 `Unsupported inductor graph input type`。因此失败不能倒推成“bool placeholder 未登记”。任意 `object()` 则在追加 `graph_input_names=['scalar']` 后，既不进入特殊分支，也不是 Tensor，于 Tensor assert 处失败；`graph_inputs` 仍为空。

## 具体状态推演：静态 tensor 与标量一起进入

设图有 `x` 和 `alpha` 两个 placeholder：`x` 是上面的 `[2,3]` tensor，`alpha=2`。

| 处理后 | `graph_input_names` | `graph_inputs` | env 中本节点结果 |
| --- | --- | --- | --- |
| x | `["x"]` | `x → TensorBox(InputBuffer("x"))` | TensorBox |
| alpha | `["x","alpha"]` | 再加 `alpha → sympy.Integer(2)` | `sympy.Integer(2)` |

后面的乘法 lowering 从 env 得到 TensorBox 和 SymPy 2。它可以构造 `i ↦ load(x,i) * 2`，却不会为 alpha 准备另一块输入存储。

若当前 GraphLowering 是名为 `true_graph_0` 的条件分支子图，`qualify_name("x")` 交回 `true_graph_0_x`。注意 `graph_input_names` 在改名前已经追加原始 `x`，而 `graph_inputs` 使用限定后的名字；两者分别服务调用接口顺序和生成期名字唯一性。

## stride 不是可有可无的描述

同样逻辑值 `[[0,1,2],[3,4,5]]` 可以来自连续输入 `[2,3]`、stride `[3,1]`；也可以让转置视图表现为 shape `[3,2]`、stride `[1,3]`。`FixedLayout` 同时保存 size 与 stride，后续 `InputBuffer` 的位置读取才能把逻辑索引换成正确存储位置。

例如连续 x 的 `[1,2]` 对应元素位置 `1×3+2×1=5`。若错误登记成 `[1,3]`，同一个逻辑位置会变成 `1×1+2×3=7`，已经超出六元素存储。placeholder 因此不能只登记 shape。

实际 direct-lowering 结果与这份推演一致：连续 `[2,3]` 输入登记 size `[2,3]`、stride `[3,1]`；其转置登记 size `[3,2]`、stride `[1,3]`。两者均建立 `TensorBox→StorageBox→InputBuffer`，`graph_inputs_original` 保存的是 `InputBuffer`。

## 静态与动态 size/stride 分支

选择条件不是“这个 placeholder 在参数表中排第几个”，而是样例 tensor 的 `_has_symbolic_sizes_strides`：

- 普通 tensor 为 false，`static_sizes_strides` 把每项转成 SymPy Integer。
- 实验构造的动态 FakeTensor 为 `size=(u0,3)` 且标志为 true；复用同一个 ShapeEnv 时，`symbolic_sizes_strides` 实际登记 `[u0,3]` 与 stride `[3,1]`。

`reuse_shape_env=False` 的分支会用 `ConstantSource('__inductor_unknown_tensor_N')` 在本 GraphLowering 的 ShapeEnv 中创建符号，再剥掉外层 SymInt。例如动态输入 `[u0,3]` 作为第一份未知 tensor 时，来源名是 `__inductor_unknown_tensor_0`，返回的 size 列表保留代表首维的符号表达式；后续 shape guard 和尺寸计算都引用同一个表达式。

## 名字、设备与对齐分别登记

名为 `branch` 的子图输入 x 实际产生：

```text
graph_input_names = ['x']
graph_inputs keys = ['branch_x']
InputBuffer name  = branch_x
```

接口位置继续使用父调用合同中的原名，生成期对象使用限定名避免不同子图冲突。

used CPU 输入会让 `device_types=['cpu']`，`device_node_mapping[cpu]` 指向当前 placeholder x。相同 tensor 若完全没有 users，InputBuffer、两个输入表和接口名仍建立，但 device types 与 mapping 均为空；这正是源码注释为 cudagraph 保留的例外。

对齐判断在 suppress-shape-guards 上下文中读取样例输入。当前 helper 只关心 GPU：CPU 输入直接返回 false。GPU 输入的起始字节地址满足源码要求的对齐倍数，或配置明确假定输入对齐时，限定后的名字会加入 `aligned_inputs`；Triton 生成访存选项时读取这个集合，决定是否可以采用要求对齐的加载方式。若运行时输入不满足所需对齐，外围输入检查会复制到对齐存储；这里不额外为 storage offset 生成 guard。

## 用新情况检验理解

若增加一个未使用的 CPU tensor `unused`：它仍进入 `graph_input_names`、`graph_inputs` 和 `graph_inputs_original`，但因为当前 placeholder 没有 users，不调用 `add_device_info`。这个情况已由独立实验覆盖，使“接口中存在 CPU 参数”与“计算图需要在 CPU 执行”保持区别。

若动态输入的两个维度在 ShapeEnv 中是同一个符号 `s0`，复用 ShapeEnv 的分支把二者转换成同一 SymPy 表达式，而不是独立创造两个变量。实验已覆盖一个 unbacked `u0` 被保留；两个位置复用同符号的判断来自同一转换源码。

本叶子交出的 TensorBox 和标量表达式将由 Q212 的算子分派读取；输入 mutation 的最终处理则到 Q213 的 output 合同再完成。
