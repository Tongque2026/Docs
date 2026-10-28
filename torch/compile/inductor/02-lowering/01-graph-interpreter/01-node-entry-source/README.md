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

这里有三个表示包装和一份布局对象：

- `FixedLayout` 只描述 device、dtype、size、stride 和 offset，不是一块新存储；placeholder 没有显式传 offset，当前构造器使用默认值 0；
- `InputBuffer("arg0_1", layout)` 是对运行时传入存储的命名引用，不持有样例 x 的六个数据；
- `StorageBox` 包住这份底层表示，使后续 mutation 可以把 box 内的表示换掉；
- `TensorBox` 是解释器 env 交给后续 lowering 的张量值。

`TensorBox.create(InputBuffer(...))` 实际建立三层链：`TensorBox.data = StorageBox`、`StorageBox.data = InputBuffer`；InputBuffer 的 `layout` 字段再指向 FixedLayout。所以 `tensor.data.data` 正好取得最初的裸 InputBuffer，不需要再多取一层才能越过 layout。

TensorBox 进入 `graph_inputs["arg0_1"]`，裸 InputBuffer 进入 `graph_inputs_original["arg0_1"]`。`graph_inputs` 让后续 lowering 和 wrapper 按生成期名字找到当前输入表示；本次 placeholder 的返回值同时进入解释器 `env`，所以后续 add 节点会从 env 取得同一个 TensorBox。它生成的是“运行时读取 arg0_1”的规则，而不是读取上面列出的 0 到 5。

`graph_inputs_original` 只保留最初的 InputBuffer 身份。若后续 mutation 将 StorageBox 内的表示换成另一个结果，Q213 展开的 output 方法会比较当前内层表示与这份原始身份，并安排把最终值写回调用者传入的原存储。本页只建立这对对象和明确的接收者；替换与写回算法在 Q213 展开。

## 读源码前先固定四项状态

`compile_fx` 构造 GraphLowering 后调用 `graph.run(*example_inputs)`。FX Interpreter 把这些实参保存成一个按位置前进的迭代器；每遇到一个 placeholder，父类 `placeholder` 就取出下一项。因此第一个 FX placeholder 接住第一份 example input，第二个接住第二份。这里不是根据参数名重新搜索样例，也不是从 `node.meta` 取真实输入。

`ShapeEnv` 保存符号、样例值和约束之间的关系。例如两个维度已知相等时，它们可以引用同一个符号，而不是各建一个变量。进入 GraphLowering 以前，`compile_fx` 从 example inputs 找到它们所属的 FakeTensorMode，并取出同一个 `fake_mode.shape_env`，再把这个对象传给 `GraphLowering(..., shape_env=shape_env)`。GraphLowering 将它保存为 `self._shape_env` 并令 `reuse_shape_env=True`。因此动态 example 的 `size()[0]` 所携带的 u0，和当前 GraphLowering 查询约束、生成 guard 时使用的 u0 来自同一个 ShapeEnv，而不是碰巧同名的两个符号。

若 example inputs 没有可取得的 ShapeEnv，GraphLowering 自己新建一个并令 `reuse_shape_env=False`。此时 `symbolic_sizes_strides` 用 `ConstantSource('__inductor_unknown_tensor_N')` 表示“第 N 份未知 tensor 输入”这个来源，再让新的 ShapeEnv 为这份样例的 size、stride、storage offset 建立符号和值/范围关系。docstring 所说的 duck-shape，就是创建这些关系时，数值相同且条件允许的尺寸可以复用符号。

`SymTypes` 在这里包含 PyTorch 的符号整数、符号浮点数和符号布尔值。它们携带 `node.expr`，placeholder 直接把这个 SymPy 表达式交给后续 lowering。

`GraphLowering.run_node(n)` 在解释每个 FX 节点前进入 `set_current_node(n)`，把 `self.current_node` 临时设为 n；父类解释器随后根据 `n.op` 调用这里的 `placeholder`。离开该节点时，context manager 会恢复先前值。因此 placeholder 内的 `self.current_node` 就是当前 FX placeholder；它的 `users` 为空表示没有后续 FX 节点读取该输入。`V.graph` 指向当前激活的这一个 GraphLowering，所以 `V.graph.current_node` 是 `add_device_info` 保存设备来源时读取的同一节点。

输入名有两种用途。`graph_input_names` 按 FX 接口顺序保存原 target；wrapper 生成 `call(args)` 时按这张列表写出 `x, alpha = args`，所以 None 和 BackwardState 即使不产生 buffer，也仍占据调用参数位置。只有 tensor 分支继续调用 `qualify_name`，使子图中的 InputBuffer 名全局唯一。标量分支在限定名字之前已经返回，因此命名子图中的标量仍以原 target 作为 `graph_inputs` 的 key。

## 完整源码

先固定首个代码块会看到的类型和值。`ex: torch.Tensor` 是本次样例输入 x；它的 size=`[2,3]`、stride=`[3,1]`、dtype=float32、device=cpu。返回标注中的 `Tuple[List[...],List[...]]` 表示函数交回两张列表：第一张是各维 size，第二张是各维 stride；动态维可用符号表达式，静态维就是整数。`placeholder` 后续用这两张列表建立 layout，再建立名为 `arg0_1` 的 InputBuffer。

第一次逐行阅读先走静态主路径：`reuse_shape_env=True`，当前编译任务已有 `_shape_env`，x 没有符号尺寸，target 为 `arg0_1`，没有子图前缀，`current_node.users` 非空；CPU 输入不参与 GPU 对齐优化，所以 `aligned_inputs` 保持空。遇到另一分支时再换状态：动态 FakeTensor 的 size 为 `[u0,3]`，stride 为 `[3,1]`；动态 size 并不要求 stride 也含符号。

若 `reuse_shape_env=False`，来源名中的编号取 `len(self._shape_env.var_to_val)`。只有该表原先为空时，第一项来源才叫 `__inductor_unknown_tensor_0`；已有两项映射时，新来源从后缀 2 开始。

两条 symbolic 路径交出相同形式的结果，但建立过程不同：

```text
复用：example.size() 中的 SymInt u0
        → sympy.sympify
        → 当前 ShapeEnv 原有的 SymPy Symbol u0

新建：example tensor + ConstantSource("__inductor_unknown_tensor_N")
        → self._shape_env.create_symbolic_sizes_strides_storage_offset(...)
        → ShapeEnv 登记新符号的来源、样例值和范围
        → 从返回的 SymInt 取出 .node.expr
```

`convert_shape_to_inductor` 在复用路径中逐项执行 `sympy.sympify`：具体整数成为 SymPy Integer，SymInt 则交出它已经持有的 SymPy 表达式。新建路径调用的 ShapeEnv 方法同时返回 size、stride 和 storage offset；本方法只保留前两张列表，随后同样交给 FixedLayout。

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

## 五条正常路径和一个失败出口

| example input | `graph_input_names` | `graph_inputs[target]` | 返回给 FX env | InputBuffer |
| --- | --- | --- | --- | --- |
| Tensor | 追加原 target | TensorBox | 同一 TensorBox | 建立 |
| SymInt/SymFloat/SymBool | 追加 | SymPy expr | expr | 不建立 |
| int/bool/float | 追加 | `sympy.sympify` 结果 | expr | 不建立 |
| None | 追加 | 不登记 | None | 不建立 |
| BackwardState | 追加 | 不登记 | None | 不建立 |
| 其他对象 | 已追加 | assert 前尚未登记 | 无正常返回 | 不建立 |

这张表解释一个细节：`graph_input_names` 表示 FX 接口顺序，不等同于 `graph_inputs` 的键集合。`None` 与 BackwardState 仍占调用参数位置，但不会成为生成程序读取的 tensor 或标量表达式。

`graph_inputs_original` 只登记 tensor；标量、None 和 BackwardState 都不会进入这张表。BackwardState 分支直接信任上游“该参数未被使用”的条件，本方法自身不检查 users。

placeholder 接受某一类型，不等于后续 output 一定支持它。实验中 bool 在这里成功登记为 SymPy `BooleanTrue`，随后 output 处理报 `Unsupported inductor graph input type`。若名为 `unknown` 的 placeholder 收到 `object()`，它会先追加 `graph_input_names=['unknown']`，随后因为既不属于特殊分支也不是 Tensor，在 Tensor assert 处失败；`graph_inputs` 仍为空。这个案例只描述该 assert 位置的部分状态。Tensor 分支后续 helper 若抛出其他异常，方法同样没有事务回滚，本页不枚举那些外部异常。

## 具体状态推演：静态 tensor 与标量一起进入

设图有 `x` 和 `alpha` 两个 placeholder：`x` 是上面的 `[2,3]` tensor，`alpha=2`。

| 处理后 | `graph_input_names` | `graph_inputs` | env 中本节点结果 |
| --- | --- | --- | --- |
| x | `["x"]` | `x → TensorBox(StorageBox(InputBuffer("x")))` | 同一 TensorBox |
| alpha | `["x","alpha"]` | 再加 `alpha → sympy.Integer(2)` | `sympy.Integer(2)` |

后面的乘法 lowering 从 env 得到 TensorBox 和 SymPy 2。它可以构造 `i ↦ load(x,i) * 2`，却不会为 alpha 准备另一块输入存储。

若当前 GraphLowering 是名为 `true_graph_0` 的条件分支子图，`qualify_name("x")` 交回 `true_graph_0_x`。注意 `graph_input_names` 在改名前已经追加原始 `x`，而 `graph_inputs` 使用限定后的名字；两者分别服务调用接口顺序和生成期名字唯一性。

## stride 不是可有可无的描述

同样逻辑值 `[[0,1,2],[3,4,5]]` 可以来自连续输入 `[2,3]`、stride `[3,1]`；也可以让转置视图表现为 shape `[3,2]`、stride `[1,3]`。`FixedLayout` 同时保存 size 与 stride，后续 `InputBuffer` 的位置读取才能把逻辑索引换成正确存储位置。

例如连续 x 的 `[1,2]` 对应元素位置 `1×3+2×1=5`。若错误登记成 `[1,3]`，同一个逻辑位置会变成 `1×1+2×3=7`，已经超出六元素存储。placeholder 因此不能只登记 shape。

实际 direct-lowering 结果与这份推演一致：连续 `[2,3]` 输入登记 size `[2,3]`、stride `[3,1]`；其转置登记 size `[3,2]`、stride `[1,3]`。两者均建立 `TensorBox→StorageBox→InputBuffer`，`graph_inputs_original` 保存的是 `InputBuffer`。

## 静态与动态 size/stride 分支

样例 tensor 的 `_has_symbolic_sizes_strides` 直接选择分支：

- 普通 tensor 为 false，`static_sizes_strides` 把每项转成 SymPy Integer。
- 实验构造的动态 FakeTensor 为 `size=(u0,3)` 且标志为 true；复用同一个 ShapeEnv 时，`symbolic_sizes_strides` 实际登记 `[u0,3]` 与 stride `[3,1]`。

`reuse_shape_env=False` 的分支会用 `ConstantSource('__inductor_unknown_tensor_N')` 在本 GraphLowering 的 ShapeEnv 中创建符号，再剥掉外层 SymInt。N 是创建前 `var_to_val` 中已有映射的数量；表为空时才从 0 开始。返回的 size 列表保留代表首维的符号表达式，后续 shape guard 和尺寸计算引用同一个表达式。

源码注释“the first N inputs are weights”描述常见的调用方排列；实际 if 条件只检查 `_has_symbolic_sizes_strides`。任何没有符号 size/stride 的 Tensor 都走静态转换。

Tensor 这一条正常路径内部因此有三个 size/stride 来源：

| 样例状态 | 调用的方法 | size/stride 来源 |
| --- | --- | --- |
| 无符号 size/stride | `static_sizes_strides` | 每项转为 SymPy Integer |
| 有符号，复用 ShapeEnv | `symbolic_sizes_strides` | 转换现有符号关系 |
| 有符号，不复用 ShapeEnv | `symbolic_sizes_strides` | 当前 ShapeEnv 为样例建立新关系 |

## 名字、设备与对齐分别登记

名为 `branch` 的子图输入 x 实际产生：

```text
graph_input_names = ['x']
graph_inputs keys = ['branch_x']
InputBuffer name  = branch_x
```

接口位置继续使用父调用合同中的原名，生成期对象使用限定名避免不同子图冲突。

这个限定只发生在 tensor 分支。若同一子图还有普通标量 `alpha=2`，它会在调用 `qualify_name` 之前登记并返回，所以 `graph_inputs` 同时可能含 `branch_x` 和未加前缀的 `alpha`。

每个子图由自己的 GraphLowering 实例解释，也拥有自己的 `graph_inputs` 字典。因此不同子图都使用 key `alpha` 时不会写进同一个共享字典；tensor 前缀解决的是生成期对象跨图汇总时的命名唯一性。

假设处理第一个输入前，三个设备集合和 mapping 都为空。used CPU placeholder x 处理后，`device_types={'cpu'}`，`device_idxs` 仍为空，`device_node_mapping[torch.device('cpu')] = x`。随后再处理 used CPU placeholder y，type 集合不变，mapping 也继续指向首次登记该完整 device 的 x。若再来 used `cuda:1` 输入 z，则 type 集合增加 `'cuda'`、index 集合增加 1、mapping 增加 `torch.device('cuda',1) → z`。

unused tensor 仍建立 InputBuffer、两张 tensor 输入表和接口名，只是不调用 `add_device_info`。这些字段是整图累计状态：“不调用”表示当前输入不增加或改写设备记录，不表示处理前已有集合会被清空。

这三份设备状态在本页之后有不同消费者：

- `device_types` 决定 wrapper 选 CPU 还是相应设备的生成器，并随编译产物保存；
- `device_idxs` 随编译产物保存，运行时用它确定设备编号；
- `device_node_mapping` 保留“哪个 FX 节点首次引入完整 device”的来源，编译收尾用它判断 CUDA Graph 是否因 CPU 或多设备而禁用，并可把原因定位回该节点。

对齐判断与 users 条件无关。先把相邻 helper `should_assume_input_aligned(example)` 的返回值记为 A；placeholder 只负责在 A=true 时把限定名加入 `aligned_inputs`。当前 helper 对 CPU 返回 false；GPU 的具体判定读取配置和 storage offset。判断处使用 suppress-shape-guards 上下文，避免这次观察新增 shape guard。后续 GPU codegen 查询一个输入名是否在 `aligned_inputs`：在集合中便可采用要求该对齐条件的访问方式，不在集合中则按未对齐输入处理。具体生成指令留在 Codegen 篇；这里交出的合同就是“限定输入名 → 可否假定对齐”。

把 users 条件和 helper 结果记为一张状态表。A 表示 `should_assume_input_aligned(example)` 返回 true：

| 输入 | 有 users | `add_device_info` | alignment helper | `aligned_inputs` |
| --- | --- | --- | --- | --- |
| CPU | 是 | 调用 | A 恒为 false | 不增加 |
| CPU | 否 | 不调用 | A 恒为 false | 不增加 |
| GPU | 是 | 调用 | 独立计算 A | A=true 时加入限定名 |
| GPU | 否 | 不调用 | 仍独立计算 A | A=true 时加入限定名 |

所以 unused GPU 且 A=false 时，两张 tensor 输入表已经登记，当前输入不新增 device 或 alignment 记录；A=true 时当前输入只向 `aligned_inputs` 增加限定名。先前输入留下的累计集合仍然保留。A 的具体判定来自相邻 helper，output 对原输入身份的消费来自 Q213。本页用这两处当前源码解释 placeholder 所存状态的用途，不把它们列入五个完整方法的声明范围。

## 用新情况检验理解

若增加一个未使用的 CPU tensor `unused`：它仍进入 `graph_input_names`、`graph_inputs` 和 `graph_inputs_original`，但因为当前 placeholder 没有 users，不调用 `add_device_info`。这个情况已由独立实验覆盖，使“接口中存在 CPU 参数”与“计算图需要在 CPU 执行”保持区别。

本文把未知对象的 Tensor assert 作为具体失败出口。Tensor 路径没有事务回滚；异常发生在哪一步，决定已经留下哪些状态：

| 失败位置 | 已留下的图级状态 |
| --- | --- |
| size/stride helper、`qualify_name` 或三层对象构造期间 | 只有原 target 已进入 `graph_input_names` |
| 两张输入表写完、设备登记之前 | 再加限定名对应的 TensorBox 与原 InputBuffer |
| `add_device_info` 之后、alignment helper 期间 | 再加当前 used 输入贡献的设备记录 |

成功通过 alignment 判断后才返回 TensorBox。这里按源码写入顺序说明部分状态；具体 helper 自身还能抛出哪些异常，不把它们扩展成新的输入类别。

若动态输入的两个维度在 ShapeEnv 中是同一个符号 `s0`，复用 ShapeEnv 的分支把二者转换成同一 SymPy 表达式，而不是独立创造两个变量。实验已覆盖一个 unbacked `u0` 被保留；两个位置复用同符号的判断来自同一转换源码。

本叶子交出的 TensorBox 和标量表达式将由 Q212 的算子分派读取；输入 mutation 的最终处理则到 Q213 的 output 合同再完成。

