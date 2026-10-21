# GraphLowering 输出处理：固定返回存储、stride 与输入修改

[返回 GraphLowering](../README.md) · [上一篇：算子与常量分派](../02-dispatch-source/README.md) · [返回 Lowering](../../README.md)

最后一个 FX `output` 节点不再构造一项数学运算。它要把 env 中的结果整理成调用者可观察的接口：tensor 必须有实际存储，返回 stride 要与 FX 元数据兼容，被原地修改的输入必须写回调用者传入的那块存储，所有 buffer 的布局也要在离开 Lowering 前定下来。

本叶子完整保留当前 `torch/_inductor/graph.py` 中 `output`、`finalize`、`try_match_insignificant_strides` 三个方法，按产品文件顺序排列，方法内部无省略。教学注释统一以 `# 教学：` 开头。

## 返回前同时处理三种引用

对 `x.add_(1); return x`，Lowering 内部当前值、调用者原输入和函数返回其实指向同一个语义对象，却可能暂时由不同 IR 表示：

```text
graph_inputs["x"]       → 修改后的当前值
graph_inputs_original   → 调用者传入的原 InputBuffer
graph_outputs           → 函数要返回的值
```

出口必须把修改后的当前值写进原 InputBuffer；若返回项正是这份修改后的值，还要把返回引用换回原 InputBuffer。这样调用后既能看到 `x` 已更新，返回 tensor 也保持输入别名身份。

## 完整源码

进入 `output` 时，`target="output"`；`args` 是已经由 Interpreter 从 env 换成 IR 结果的 tuple，本例可记作 `(modified_x,)`；`kwargs={}`。`graph_inputs_original["x"]` 仍是调用者的 InputBuffer，而 `graph_inputs["x"]` 已指向修改后的值。返回类型 `None` 表示这个方法不把结果交回下一 FX node，而是把整理后的项目写入 `self.graph_outputs`。函数逐项判断返回值类型，必要时物化 TensorBox、核对 stride，并把修改后的输入引用换回原 InputBuffer。末尾日志中的 `num_channels_last_conv` 只是本图有多少 convolution 促使输入采用 channels-last 的计数，`graph_id` 是当前编译图编号；它们只用于 debug 输出，不参与 normalize、mutation 写回或 layout 决策。

```python
def output(
    self, target: str, args: Tuple[object], kwargs: Dict[str, object]  # type: ignore[override]
) -> None:
    # 教学：父 Interpreter 从 env 解引用 output 的参数。
    result = super().output(target, args, kwargs)  # type: ignore[arg-type]
    if not isinstance(result, (tuple, list)):
        # nested subgraphs can have singleton outputs
        # 教学：统一成序列，后面逐项与 FX output 参数配对。
        result = (result,)
    assert isinstance(result, (tuple, list)), type(result)
    # 教学：只允许这里列出的 Lowering 结果跨过图出口。
    assert all(
        isinstance(
            x,
            (
                TensorBox,
                ir.Constant,
                type(None),
                ir.ConstantBuffer,
                sympy.Expr,
                sympy.logic.boolalg.Boolean,
                int,
                ir.EffectfulKernel,
            ),
        )
        for x in result
    ), result

    # 教学：保留原 FX node，以便读取每个返回项的 fake stride。
    fx_node_args = V.graph.current_node.args[0]  # type: ignore[arg-type]
    if not isinstance(fx_node_args, (tuple, list)):
        # nested subgraphs can have singleton outputs
        fx_node_args = (fx_node_args,)
    # 教学：输出 tensor 不能只是一条延迟规则；先把它变成可供外部引用的存储输入。
    result = [ir.ExternKernel.realize_input(x) for x in result]
    result_correct_strides = []

    assert len(fx_node_args) == len(result)
    for r, fx_node in zip(result, fx_node_args):
        if not isinstance(r, (ir.TensorBox, ir.BaseView)):
            # 教学：标量、None 已在上一步变成专用 buffer；它们不做 tensor stride 修正。
            result_correct_strides.append(r)
        else:
            # AOT Autograd tries to detect stride divergence of inductor from output metadata.
            # Here, we try to avoid spurious divergence by matching insignificant strides such as
            # 教学：只修正 size 0/1 维度中不改变元素位置的 stride 差异。
            result_correct_strides.append(
                self.try_match_insignificant_strides(
                    r, fx_node.meta["val"].stride()
                )
            )

    # 教学：这里建立函数正式返回表，后续 wrapper/codegen 读取它。
    self.graph_outputs = result_correct_strides
    value: ir.IRNode
    # 教学：逐个检查 tensor 输入的当前表示是否仍是原 InputBuffer。
    for name, value in self.graph_inputs.items():
        assert isinstance(
            value, (TensorBox, sympy.Expr)
        ), f"Unsupported inductor graph input type: {type(value)}"
        if not isinstance(value, TensorBox):
            continue
        # 教学：先物化当前值，才能判断实际底层对象并安排写回。
        value.realize()
        assert isinstance(value, TensorBox)
        value = value.data
        assert isinstance(value, ir.StorageBox)
        # 教学：保留 StorageBox，因为 graph_outputs 中可能正引用它。
        value_storage_box = value
        value = value.data
        if not isinstance(value, InputBuffer) or value.get_name() != name:
            # one of our inputs was mutated, need to turn that into a copy
            # 教学：当前值不再是同名原输入，便把它实现进原 InputBuffer。
            ir.MutationLayoutSHOULDREMOVE.realize_into(
                value, self.graph_inputs_original[name]
            )
            # replace output with mutated input
            try:
                # 教学：若返回项就是当前 StorageBox，改为返回原输入对象以保持别名身份。
                ind = self.graph_outputs.index(value_storage_box)
                self.graph_outputs[ind] = self.graph_inputs_original[name]
            except ValueError:
                # 教学：输入可以被修改但不返回，此时只需写回，无返回项可替换。
                pass

    # 教学：所有输出与 mutation 安排完成后，冻结每个已登记 buffer 的布局。
    self.finalize()
    log.debug(
        "Force channels last inputs for %d conv for the current graph with id %d",
        self.num_channels_last_conv,
        self.graph_id if self.graph_id is not None else -1,
    )

def finalize(self) -> None:
    # 教学：每个 buffer 根据前面收集的读写和布局要求作最终决定。
    for buf in self.buffers:
        buf.decide_layout()

def try_match_insignificant_strides(
    self,
    tensor: Union[ir.TensorBox, ir.BaseView],
    meta_strides_inp: Tuple[Union[int, torch.SymInt], ...],
) -> Union[ir.TensorBox, ir.BaseView]:
    """
        Tries to match the strides of the tensor to those in the meta_strides. Strides of insignificant
        dimensions - size 0 or 1 - will be updated.

        If there are real stride differences (NHWC vs NCHW) then the input will be returned.
        """

    # should have already been realized
    # 教学：只有已有 storage/layout 的结果才能安全建立另一份布局视图。
    assert torch._inductor.ir.is_storage_and_layout(tensor)

    # 教学：把 torch.SymInt 解包成与 IR 比较使用的 SymPy 表达式。
    meta_strides = [
        s.node.expr if isinstance(s, torch.SymInt) else s for s in meta_strides_inp
    ]

    # 教学：所有 stride 已可静态证明相同，直接复用原对象。
    if all(
        self.sizevars.statically_known_equals(s1, s2)
        for s1, s2 in zip(meta_strides, tensor.get_stride())
    ):
        return tensor  # type: ignore[arg-type]

    def significant_strides_equal(
        shape: Sequence[Union[Expr, int]],
        meta_strides: Sequence[Union[Expr, int]],
        tensor_strides: Sequence[Union[Expr, int]],
    ) -> bool:
        for dim, s1, s2 in zip(shape, meta_strides, tensor_strides):
            # 教学：size 0/1 的维度没有两项不同元素可由 stride 区分，跳过。
            if self.sizevars.statically_known_leq(dim, 1):  # type: ignore[arg-type]
                continue

            # 教学：任何有效维度 stride 不同都是真实布局差异。
            if not self.sizevars.statically_known_equals(s1, s2):
                return False

        return True

    # 教学：真实维度存在差异时不伪装成 meta layout，原 tensor 原样返回。
    if not significant_strides_equal(
        tensor.get_size(), meta_strides, tensor.get_stride()
    ):
        return tensor

    # 教学：只剩无关维差异；取得同一 storage 和旧布局，复制 stride 列表。
    storage, old_layout = torch._inductor.ir.as_storage_and_layout(tensor)
    new_stride = list(old_layout.stride)
    for i, s in enumerate(tensor.get_size()):
        if self.sizevars.statically_known_leq(s, 1):  # type: ignore[arg-type]
            new_stride[i] = meta_strides[i]

    # 教学：device、dtype、size、offset 和 storage 不变，只替换无关维 stride。
    new_layout = torch._inductor.ir.FixedLayout(
        old_layout.device,
        old_layout.dtype,
        old_layout.size,
        new_stride,
        old_layout.offset,
    )
    return ir.TensorBox(torch._inductor.ir.ReinterpretView(storage, new_layout))
```

源码副本去掉类级四空格缩进；三个方法内部的产品代码、注释和顺序完整保留。

## 案例一：普通计算结果成为 graph_outputs

对 `return (x+y)*2`，最后的 mul 在进入 output 前可能还是延迟 Pointwise。`ExternKernel.realize_input` 使它成为具名 `ComputedBuffer`。假设其 stride 与 FX meta 都是 `[1]`：

| 步骤 | result / 图级状态 |
| --- | --- |
| 父类解引用 | `TensorBox(StorageBox(Pointwise))` |
| `realize_input` | `TensorBox(StorageBox(ComputedBuffer buf0))` |
| stride 比较 | 完全相同，返回同一对象 |
| 赋值 | `graph_outputs=[buf0 对应对象]` |
| 输入检查 | x、y 仍是各自同名 InputBuffer，无写回 |
| finalize | buf0 执行 `decide_layout()` |

`output()` 返回 `None`，因为结果已经写入 `self.graph_outputs`。后续生成 wrapper 时读取这张图级返回表。

这里还有一处容易漏掉的表示变化。`realize_input` 不只处理 tensor；图若返回 Python/SymPy 值，它会先把这些值变成有统一 IR 接口的对象：

| output 进入前 | `realize_input` 之后 | 为什么不进入 stride 分支 |
| --- | --- | --- |
| `3`、SymPy 表达式或布尔表达式 | `ShapeAsConstantBuffer` | 它表达形状/标量值，不是有多维布局的 tensor |
| `None` | `NoneAsConstantBuffer` | 它表达“没有 tensor”，没有 size 和 stride |
| `ir.Constant` | 由 `add_tensor_constant` 登记的 tensor 常量 | 它按 tensor 常量路径形成可引用存储 |
| 已有 `ConstantBuffer` | 原对象 | 已经满足外部输入形式 |

实验 04 直接运行 `GraphLowering`，实际观察到整数、浮点、SymInt 输出成为 `ShapeAsConstantBuffer`，`None` 与 `BackwardState` 占位输出成为 `NoneAsConstantBuffer`。所以上面循环里的“直接保留”是指保留**已包装后的对象**，不是把原始 `3` 或 `None` 原样放进 `graph_outputs`。

## 案例二：只修正 size-1 维度的 stride

设返回 tensor shape 为 `[2,1,3]`：

```text
实际 IR stride = [3,3,1]
FX meta stride = [3,99,1]
```

维度 0 的 size=2，stride 都为 3；维度 2 的 size=3，stride 都为 1。唯一差异在维度 1，而它的 size=1。无论 stride 是 3 还是 99，合法索引都只能取 0，对元素位置贡献都是 `0×stride=0`。

因此 `significant_strides_equal` 返回真，新布局只把第二项改成 99：

```text
同一 storage、同一 offset、同一 size
旧 layout stride [3,3,1]
新 view stride   [3,99,1]
```

若 shape 改为 `[2,2,3]`，第二维已有索引 0 和 1，stride 3 与 99 会读取不同位置；函数返回原 tensor，不建立这份 ReinterpretView。这里的“匹配”只消除无实际地址意义的差异，不转换真实 NCHW/NHWC 排列。

对这两个输入调用当前方法时，`[2,1,3]` 得到 `ReinterpretView`，底层名称仍是 `stride_src`，size、offset 不变而 stride 成为 `[3,99,1]`；`[2,2,3]` 则返回传入的同一个对象。

## 案例三：修改输入并返回它

设 x 初始 `[1,2]`，图语义为 `x.add_(10); return x`。placeholder 时：

```text
graph_inputs_original["x"] = InputBuffer x
graph_inputs["x"]          = TensorBox(StorageBox(InputBuffer x))
```

mutation lowering 后，`graph_inputs["x"]` 指向表示 `[11,12]` 的当前 StorageBox。output 先把它放进 `graph_outputs`，随后遍历输入：底层已不再是名为 x 的原 InputBuffer，于是 `realize_into(current_value, original_x)` 安排把 `[11,12]` 写回调用者存储。

返回表原本引用当前 StorageBox，查找成功后替换成 `graph_inputs_original["x"]`：

```text
执行副作用：原输入存储 x ← [11,12]
函数返回：  同一个原输入存储 x
```

若函数修改 x 但只返回另一个 y，`graph_outputs.index(value_storage_box)` 找不到 x，进入 `ValueError` 分支：写回仍保留，只是不改返回表。

实验 06 分别完整编译了这两种函数。第一种返回的 x 与调用者 x 的 `data_ptr()` 相同；第二种执行 `x.copy_(y); return y+1`，调用后 x 变成 `[7,8,9]`，返回 `[8,9,10]` 且不与 x 别名。运行结果证明了“写回不依赖 x 是否在返回值中”；具体由 `index` 找不到后吞掉 `ValueError` 完成，则由上面的当前源码分支支持。

## finalize 为什么放在最后

输出物化可能新增 buffer；输入写回也可能建立 mutation operation 或给布局增加约束。若在它们之前就决定布局，最后产生的消费者要求无法参与选择。当前源码因此先完成输出与 mutation 交接，再对 `self.buffers` 中每个 buffer 调用 `decide_layout()`。

布局选择算法属于 Q23；本叶子需要掌握的状态边界是：离开 output 后，`graph_outputs` 已稳定、输入写回已登记、现有 buffers 已决定布局。

实验 06 还给 `buffers` 放入两个只实现 `decide_layout()` 的接口探针，观察到调用顺序为 `buf_first`、`buf_second`。这个 mock 只能证明 `finalize` 会按列表顺序调用所有现有 buffer；“为何必须在输出和写回之后”来自 `output` 的调用顺序及这些步骤可能登记 buffer/约束的源码关系。

## 用新情况检验理解

1. 返回 `(tensor, 3, None)` 时，只有 tensor 进入 stride 修正；`3` 和 `None` 先分别包装成 `ShapeAsConstantBuffer` 与 `NoneAsConstantBuffer`，然后在 stride 循环中原样加入返回表。
2. shape `[0,5]` 的第 0 维 stride 可按 meta 修正，因为 size 0 没有任何合法元素；第 1 维 size 5 的差异不能忽略。
3. 当前输入底层仍是 `InputBuffer`，但名字不是 graph_inputs 中的 key，也会被判为修改，需要写回原输入。
4. 输入被修改但没有作为返回项时，`ValueError` 不取消 mutation，只表示没有返回引用需要替换。

Q213 已把 GraphLowering 的最终接口收口讲完整。Q214 将回到每个节点外围，解释 `run_node` 如何在调用这些入口前后处理 origin、布局约束、物化、users 和符号断言。
