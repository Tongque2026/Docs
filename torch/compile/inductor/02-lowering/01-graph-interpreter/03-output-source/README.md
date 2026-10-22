# GraphLowering 怎样形成返回值并写回修改过的输入

[返回 GraphLowering](../README.md) · [上一篇：算子与常量分派](../02-dispatch-source/README.md) · [返回 Lowering](../../README.md)

前两篇已经把 placeholder、算子和常量的 Lowering 结果放进 `env`。读到最后一个 FX `output` 节点时，`GraphLowering` 不再创建新的数学计算，而要把这些内部结果变成函数接口：

```text
env 中的返回结果
    │
    ├─ 需要存储的 tensor：物化成可由生成程序引用的 buffer
    ├─ 标量和 None：包装成生成程序能够返回的对象
    ├─ 仍是 TensorBox/BaseView 的结果：与 FX output 保存的 stride 核对
    └─ 被修改的输入：把新值写回调用者原来的存储
              │
              ▼
       self.graph_outputs
              │
              ▼
        决定现有 buffer 布局
```

本篇先让三组具体数据走到出口，再回到 `output`、`finalize` 和 `try_match_insignificant_strides` 的完整源码。这样读源码时，每个分支都有已经见过的对象。

## 普通返回值怎样进入 graph_outputs

先看：

```python
def f(x, y):
    return (x + y) * 2
```

处理 output 之前，最后的乘法结果可能仍是延迟的逐元素计算：

```text
env 中的结果
TensorBox(StorageBox(Pointwise))
```

函数已经要返回，生成程序必须有一份可以引用的结果存储。`ir.ExternKernel.realize_input` 沿 `TensorBox.data` 进入内部的 `StorageBox`，物化其中的计算，得到：

```text
StorageBox(ComputedBuffer buf0)
```

这里的“物化”是登记一份有名字的结果 buffer 及其产生方式，不是此刻执行用户计算。普通 `TensorBox(StorageBox(...))` 在 `realize_input` 中被解开一层，返回内部 `StorageBox`；它不属于随后检查的 `TensorBox/BaseView`，直接进入：

```text
self.graph_outputs = [StorageBox(ComputedBuffer buf0)]
```

`output()` 自身返回 `None`；后续 wrapper/codegen 读取 `self.graph_outputs`，生成运行时返回值。

不同类型的返回值都要先变成统一的出口对象。下面的表区分了三个时点：`super().output` 已从 env 取出的值、`realize_input` 的结果，以及它是否进入 tensor stride 检查。

| 从 env 取得的值 | `realize_input` 之后 | 是否检查 tensor stride |
| --- | --- | --- |
| `TensorBox(StorageBox(Pointwise))` | 物化后的 `StorageBox(ComputedBuffer)` | 否 |
| `int`、SymPy 表达式或布尔表达式 | `ShapeAsConstantBuffer` | 否 |
| `None` | `NoneAsConstantBuffer` | 否 |
| `ir.Constant` | `add_tensor_constant` 登记的 `TensorBox(StorageBox(ConstantBuffer))` | 是；该分支直接返回 TensorBox，不再递归解包 |
| 已有 `ConstantBuffer` | 原 `ConstantBuffer` | 否 |
| `EffectfulKernel` | `copy_input` 建立并物化一份 Pointwise，返回它的 `TensorBox` | 是 |

若传入的结果不属于 `output` 允许的类型，类型断言立即失败；若返回项数量与原 FX output 参数数量不同，长度断言失败。正常路径中，两边仍逐项配对；只有 `realize_input` 后仍是 `TensorBox` 或 `BaseView` 的项目才读取原 FX 节点 `meta["val"]` 中记录的接口 stride。普通计算输出已经是 `StorageBox`，在这一循环中直接保留。

## 只有 size 0/1 维的 stride 可以改写

这个 helper 不会接收已经解包成 `StorageBox` 的普通计算输出；调用它的条件是 `realize_input` 后的对象仍为 `TensorBox` 或 `BaseView`，例如 `ir.Constant` 形成的 TensorBox，或仍保留 view 身份的结果。下面直接给 helper 一份已有 storage/layout 的 tensor，观察它怎样处理接口 stride。

设返回 tensor 的 shape 是 `[2,1,3]`：

```text
实际 IR stride = [3,3,1]
FX meta stride = [3,99,1]
```

逻辑索引 `(i0,i1,i2)` 对应的存储位置为：

```text
offset + i0×stride0 + i1×stride1 + i2×stride2
```

第二维 size 为 1，所以该维唯一合法索引是 `i1=0`。这一维对地址的贡献始终是 `0×3=0` 或 `0×99=0`；把 stride 从 3 写成 99 不会改变任何元素的位置。`try_match_insignificant_strides` 因而建立一份新 view：

```text
底层 storage：同一份
device/dtype：不变
size：         [2,1,3]，不变
offset：       不变
stride：       [3,3,1] → [3,99,1]
外层对象：     ReinterpretView
```

这不是数据复制，也不是 NCHW/NHWC 转换。它只是让没有地址意义的 stride 与 FX 接口记录一致。

如果 shape 改成 `[2,2,3]`，第二维允许 `i1=1`。此时地址贡献会从 `1×3` 变成 `1×99`，已经指向不同位置。方法返回传入的原 tensor，storage、size、offset 和 stride 全都不改。

size 为 0 的维也没有任何合法元素，因此同样可以采用 meta stride。方法使用静态符号判断；如果编译期不能证明某维 size≤1，也不会把那一维当作无关维。

## 修改输入时要同时维护当前值、原存储和返回引用

再看：

```python
def f(x):
    x.add_(10)
    return x
```

placeholder 建立了两份用途不同的记录：

```text
graph_inputs_original["x"] = InputBuffer x
    调用者传入的原存储；函数结束时必须让调用者继续持有它

graph_inputs["x"] = TensorBox(StorageBox(InputBuffer x))
    当前计算应该读取的 x；发生 mutation 后可以改指新值
```

Lowering 处理完 mutation 后，`graph_inputs["x"]` 仍是外层 `TensorBox`，但它的 `.data` 已是表示 `[11,12]` 的当前 `StorageBox`；`graph_inputs_original["x"]` 仍是调用者传入的 `InputBuffer x`。output 处理返回值时，`realize_input` 解开外层 TensorBox，所以 `graph_outputs` 保存的恰是同一个当前 StorageBox。

输入检查把三个层次分别保存下来：

```text
current_tensor  = graph_inputs["x"]       # TensorBox
current_storage = current_tensor.data       # StorageBox；也在 graph_outputs 中
current_ir      = current_storage.data      # 当前值的底层 IRNode
original_x      = graph_inputs_original["x"]  # 原 InputBuffer
```

然后它做两件不同的事：

```text
当前值 [11,12]
    │
    ├─ realize_into(current_ir, original_x)
    │      登记：运行时把 [11,12] 写进调用者的原存储
    │
    └─ graph_outputs.index(current_storage) → original_x
           返回值继续代表调用者持有的那块存储
```

只有第一步，调用者的 x 会更新，但返回表仍代表临时的当前值；只有第二步，返回别名看似正确，却没有把新内容写回调用者存储。两步分别保证“内容已更新”和“返回身份仍是原输入”。

若函数修改 x，却返回另一个值：

```python
def g(x, y):
    x.copy_(y)
    return y + 1
```

输入检查仍发现 x 的当前表示不再是同名原 `InputBuffer`，所以 `realize_into` 仍登记写回。随后在 `graph_outputs` 中找不到 x 的当前 `StorageBox`，`.index()` 抛出并由本方法捕获 `ValueError`。这里的 `ValueError` 只表示“没有返回引用需要替换”，不会撤销前面已经登记的写回。

`.index(current_storage)` 能命中不是因为 TensorBox 与 StorageBox 存在特殊相等规则，而是因为 `realize_input` 已经剥掉 TensorBox，返回表中保存的就是这一个 StorageBox 对象。

未修改的 tensor 输入会沿 `TensorBox → StorageBox → InputBuffer` 检查到底层：对象仍是 `InputBuffer` 且名称等于输入 key，便不写回。符号输入在 `graph_inputs` 中是 `sympy.Expr`，没有 tensor 存储，直接跳过这项检查。底层是 `InputBuffer` 但名称不等于输入 key 时，也按已修改处理并写回原输入。

## 为什么最后才 finalize

普通输出的物化可能新增 buffer；输入写回也可能登记 mutation operation 或增加布局约束。它们完成之前就决定 layout，会漏掉刚刚出现的生产者、消费者或写回要求。

因此 output 最后才调用 `finalize()`：按 `self.buffers` 当前顺序，对每个已登记 buffer 调用 `decide_layout()`。离开 output 时形成三份稳定状态：

```text
graph_outputs：函数正式返回哪些 IR 对象
输入写回：     哪些当前值要实现进原 InputBuffer
buffers：      当前所有已登记 buffer 都完成 decide_layout
```

布局究竟如何选择属于存储与布局篇；这里需要掌握的是决定发生时已经收齐哪些信息。

## 完整源码：把四条路径映射回实现

进入 `output` 时，`args` 已由父 `Interpreter` 用 env 中的 IR 结果替换；`V.graph.current_node.args[0]` 仍能取得原 FX output 引用。下面保留当前 `torch/_inductor/graph.py` 中三个方法的全部语句和产品注释，只增加 `# 教学：` 注释，并去掉类级缩进。

```python
def output(
    self, target: str, args: Tuple[object], kwargs: Dict[str, object]  # type: ignore[override]
) -> None:
    # 教学：取得 env 已解引用的 IR 结果；单项输出也统一成序列。
    result = super().output(target, args, kwargs)  # type: ignore[arg-type]
    if not isinstance(result, (tuple, list)):
        # nested subgraphs can have singleton outputs
        result = (result,)
    assert isinstance(result, (tuple, list)), type(result)
    # 教学：不在列表中的对象不能跨过 GraphLowering 出口。
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

    # 教学：原 FX 引用保留 meta stride；同样把单项输出统一成序列。
    fx_node_args = V.graph.current_node.args[0]  # type: ignore[arg-type]
    if not isinstance(fx_node_args, (tuple, list)):
        # nested subgraphs can have singleton outputs
        fx_node_args = (fx_node_args,)
    # 教学：把延迟 tensor、标量、None 和常量变成生成程序可引用的 IR 对象。
    result = [ir.ExternKernel.realize_input(x) for x in result]
    result_correct_strides = []

    # 教学：每份实际 IR 结果必须与一份原 FX output 接口信息对应。
    assert len(fx_node_args) == len(result)
    for r, fx_node in zip(result, fx_node_args):
        if not isinstance(r, (ir.TensorBox, ir.BaseView)):
            # 教学：普通输出已是 StorageBox；标量/None 也无此类 layout，均直接保留。
            result_correct_strides.append(r)
        else:
            # AOT Autograd tries to detect stride divergence of inductor from output metadata.
            # Here, we try to avoid spurious divergence by matching insignificant strides such as
            # 教学：tensor 用实际 IR layout 与原 FX meta stride 进入下方 helper。
            result_correct_strides.append(
                self.try_match_insignificant_strides(
                    r, fx_node.meta["val"].stride()
                )
            )

    # 教学：物化和 stride 修正后的对象先形成正式返回表。
    self.graph_outputs = result_correct_strides
    value: ir.IRNode
    for name, value in self.graph_inputs.items():
        assert isinstance(
            value, (TensorBox, sympy.Expr)
        ), f"Unsupported inductor graph input type: {type(value)}"
        if not isinstance(value, TensorBox):
            # 教学：sympy.Expr 没有 tensor 存储，不参与 mutation 写回。
            continue
        # 教学：沿 TensorBox → StorageBox → 当前底层 IRNode 逐层检查。
        value.realize()
        assert isinstance(value, TensorBox)
        value = value.data
        assert isinstance(value, ir.StorageBox)
        # 教学：realize_input 剥掉 TensorBox，因此返回当前值时 graph_outputs 保存同一 StorageBox。
        value_storage_box = value
        value = value.data
        if not isinstance(value, InputBuffer) or value.get_name() != name:
            # one of our inputs was mutated, need to turn that into a copy
            # 教学：value 此时是 StorageBox.data，即当前底层 IRNode；先登记其写回。
            ir.MutationLayoutSHOULDREMOVE.realize_into(
                value, self.graph_inputs_original[name]
            )
            # replace output with mutated input
            try:
                # 教学：若当前值也被返回，再把该返回引用换成原 InputBuffer。
                ind = self.graph_outputs.index(value_storage_box)
                self.graph_outputs[ind] = self.graph_inputs_original[name]
            except ValueError:
                # 教学：未返回该输入只跳过引用替换，不撤销写回。
                pass

    # 教学：输出和输入写回已经登记齐全，现在才决定现有 buffer 的布局。
    self.finalize()
    log.debug(
        "Force channels last inputs for %d conv for the current graph with id %d",
        self.num_channels_last_conv,
        self.graph_id if self.graph_id is not None else -1,
    )

def finalize(self) -> None:
    # 教学：遍历此时已经收齐的 buffer；具体布局算法由各 buffer 实现。
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
    # 教学：只有已经拥有 storage 和 layout 的结果才能建立另一份布局 view。
    assert torch._inductor.ir.is_storage_and_layout(tensor)

    # 教学：torch.SymInt 解包成 sizevars 能比较的 SymPy 表达式。
    meta_strides = [
        s.node.expr if isinstance(s, torch.SymInt) else s for s in meta_strides_inp
    ]

    # 教学：全部 stride 已相同，不需要建立新 view。
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
            # 教学：size 0/1 的维没有可由 stride 区分的两个元素。
            if self.sizevars.statically_known_leq(dim, 1):  # type: ignore[arg-type]
                continue

            # 教学：任一 size>1 的维不同，便属于真实地址差异。
            if not self.sizevars.statically_known_equals(s1, s2):
                return False

        return True

    # 教学：存在真实地址差异时，保留原 tensor 的完整 layout。
    if not significant_strides_equal(
        tensor.get_size(), meta_strides, tensor.get_stride()
    ):
        return tensor

    # 教学：只剩无关维差异；复用同一 storage 和旧 layout 的其余字段。
    storage, old_layout = torch._inductor.ir.as_storage_and_layout(tensor)
    new_stride = list(old_layout.stride)
    for i, s in enumerate(tensor.get_size()):
        if self.sizevars.statically_known_leq(s, 1):  # type: ignore[arg-type]
            new_stride[i] = meta_strides[i]

    # 教学：新 view 只换 stride；device、dtype、size、offset 和 storage 不变。
    new_layout = torch._inductor.ir.FixedLayout(
        old_layout.device,
        old_layout.dtype,
        old_layout.size,
        new_stride,
        old_layout.offset,
    )
    return ir.TensorBox(torch._inductor.ir.ReinterpretView(storage, new_layout))
```

源码按四段执行：先统一并物化输出，再修正无关 stride，然后检查输入写回，最后 finalize。`num_channels_last_conv` 和 `graph_id` 只进入末尾 debug 日志，不改变上述状态。

## 沿源码再走四个边界

1. 单项返回时，`result` 和 `fx_node_args` 分别包装成一项 tuple；这样后续仍能一一配对。tuple/list 返回不再包装。
2. `meta_strides_inp` 中若有 `torch.SymInt`，先取出其 SymPy 表达式；后面的“相同”与“size≤1”都必须由 `sizevars` 静态证明。
3. 输入未修改时，两次 `.data` 得到 `StorageBox` 和同名 `InputBuffer`，条件为假，不调用 `realize_into`，也不搜索返回表。
4. 输入已修改但没有返回时，写回先登记，返回表查找后失败；捕获 `ValueError` 后仍继续处理其他输入并最终 finalize。

一次调用若有多个输入，循环按 `graph_inputs` 的顺序逐个完成上述检查。`graph_outputs.index` 负责源码当前这一次返回引用替换；完整编译 `x.add_(10); return x, x` 的两个运行时返回值仍都与调用者 x 共享同一 `data_ptr()`，但这个外部结果还包含上层图输出去重和 wrapper 处理，不能仅由本方法的一次 `.index()` 推出。

## 用新输入检查是否掌握出口合同

1. `return (tensor, 3, None)`：普通 TensorBox 物化并解包成 StorageBox，所以与另外两项一样跳过 stride helper；`3` 和 `None` 分别成为 `ShapeAsConstantBuffer` 与 `NoneAsConstantBuffer`，三项按原顺序进入 `graph_outputs`。若第一项是物化后仍为 TensorBox/BaseView 的常量或 view，才进入 helper。
2. shape `[0,5]`、实际 stride `[5,1]`、meta stride `[99,1]`：第 0 维没有合法元素，可以把 stride 改为 99；第 1 维 size=5，若两边不同就必须保留原 tensor。
3. 当前输入底层仍是 `InputBuffer`，但名称不是 `graph_inputs` 中的 key：条件仍为真，需要把当前值写回该 key 对应的原 InputBuffer。
4. `x.copy_(y); return y+1`：x 的写回发生；返回表查找 x 失败，只跳过别名替换；最后仍执行 finalize。

Q213 至此把 GraphLowering 的出口收口：`graph_outputs` 已形成，输入 mutation 已映射回调用者存储，现有 buffer 布局已决定。下一篇[每个节点执行后的统一收尾](../04-run-node-source/README.md)解释 `run_node` 如何在各节点入口和出口维护 origin、布局约束、物化、users 与符号断言。
