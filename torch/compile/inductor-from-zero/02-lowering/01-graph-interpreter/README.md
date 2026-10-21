# GraphLowering：逐节点把 FX 环境换成 Inductor IR 环境

[返回 Lowering](../README.md) · [返回总览](../../README.md)

实现叶子：[输入登记完整源码](01-node-entry-source/README.md)、[算子与常量分派完整源码](02-dispatch-source/README.md)、[怎样形成返回值并记录输入修改](03-output-source/README.md)、[`run_node` 怎样记录来源并处理异常](04-run-node-source/README.md)。

先看 `a=x+y; b=relu(a)`。处理 add 后，环境表保存 `add 节点 → a 的位置计算规则`；处理 relu 时，根据它的参数引用找到 add 节点，再从表中取出这条规则。沿节点顺序执行处理函数、用这样一张表连接前后结果，就是这里所说的 Interpreter（解释器）。它不会拿真实输入算出 a 和 b 的数值。

`GraphLowering` 继承 `torch.fx.Interpreter`，并把环境表里的结果从输入的 shape/dtype 等描述逐步换成 Inductor IR 对象。

本篇贯穿实验是：

```python
def fn(x, y):
    a = x + y
    b = torch.relu(a)
    return b * 2
```

脚本 [`experiments/04_graph_lowering_nodes.py`](../../experiments/04_graph_lowering_nodes.py) 包装当前 `GraphLowering.run_node`，在不修改 PyTorch 源码的情况下记录每个节点返回类型和图级注册表大小。

## 先区分五份“记录”

理解解释器前，先不要把下面五类状态混成一个 env：

| 状态 | key → value | 用途 |
| --- | --- | --- |
| FX Interpreter 的 `env` | FX node → 本次解释结果 | 后续 node 根据 FX 引用取得前面构造的 IR 对象 |
| `graph_inputs` | 输入名 → TensorBox 或符号表达式 | 记录生成程序的正式输入及其布局/符号信息 |
| `graph_inputs_original` | Tensor 输入名 → 最初的 InputBuffer | mutation 收尾时找到必须被写回的调用者存储身份 |
| `operations` / `buffers` | 顺序列表及名字表 | 记录已经物化、需要 Scheduler 处理的工作和存储 |
| `graph_outputs` | 顺序列表中的 IR/常量结果 | 记录 GraphLowering 交给生成阶段的普通图输出 |

后续 node 通过 FX `env` 取得参数；`graph_inputs` 记录生成函数将接收哪些输入，`graph_outputs` 记录它要返回哪些结果；`graph_inputs_original` 保存输入修改前对应的那块调用者数据；`operations` 并不会在每个算术 node 后都增加，因为延迟 IR 可以继续组合。

## 实际逐节点轨迹

实验得到：

| FX node | 返回类型链 | buffers | operations |
| --- | --- | ---: | ---: |
| `arg0_1` placeholder | `TensorBox → StorageBox → InputBuffer` | 0 | 0 |
| `arg1_1` placeholder | `TensorBox → StorageBox → InputBuffer` | 0 | 0 |
| `aten.add.Tensor` | `TensorBox → StorageBox → Pointwise` | 0 | 0 |
| `aten.relu.default` | `TensorBox → StorageBox → Pointwise` | 0 | 0 |
| `aten.mul.Tensor` | `TensorBox → StorageBox → ComputedBuffer` | 1 | 1 |
| `output` | 返回 `None`；`graph_outputs` 为 `StorageBox → ComputedBuffer` | 1 | 1 |

这张表揭示了三个关键阶段：输入被登记但不是待调度 operation；add 和 relu 仍是可组合的延迟计算；mul 要作为函数结果返回，因此必须拥有确定的存储和 stride，才出现第一个 buffer 和 operation。

## placeholder：由样例输入建立接口，不读取数值

FX Interpreter 把 `graph.run(*example_inputs)` 的对应参数交给 `placeholder`。对于 Tensor，当前实现读取 device、dtype、size、stride：

```text
example FakeTensor / tensor
  shape=(3,), stride=(1,), dtype=float32, device=cpu
            │
            ▼
FixedLayout(cpu, float32, size=[3], stride=[1])
            │
            ▼
InputBuffer("arg0_1", layout)
            │
            ▼
TensorBox(StorageBox(InputBuffer(...)))
```

它把结果同时写入：

- `graph_input_names`：保持输入顺序；
- `graph_inputs[target]`：供整张 Inductor 图按名字查输入；
- `graph_inputs_original[target]`：保留原 InputBuffer 身份，供 mutation 收尾比较；
- `device_types/device_idxs`：记录输入来自哪类设备和哪个设备编号；
- `aligned_inputs`：若 GPU 输入的起始地址能被 16 字节整除，就把输入名记在这里。后端据此决定能否生成要求 16 字节对齐的加载；运行时输入不满足对应检查时，不能直接复用这份假设。

静态 Tensor 用具体整数记录 size/stride。动态输入例如第一维未知时，ShapeEnv 为它建立符号 `s0`，placeholder 的 size 记成 `[s0,3]`；调用生成函数时，真实输入 shape `[19,3]` 会把 s0 绑定为 19，后续 layout、索引和输入检查都读取同一个符号表达式。

placeholder 并不只接 Tensor。Inductor 使用 Python 的 SymPy 库保存“暂时不能写成一个确定整数”的算术表达式。上面的 `s0` 可由样例输入的 shape `[19,3]` 得到暂用值 19，源码称为 backed symbol。另一类符号可能来自 `nonzero` 的结果个数等运行时数据；样例输入的 shape 不能预先给它一个值，源码称为 unbacked symbol，下面用 `u0` 表示。`ShapeAsConstantBuffer` 则是 Inductor IR 中一个很小的包装：它让整数、浮点数或符号值能像其他图结果一样进入输出列表；它不表示为这个标量申请了一块 tensor storage。实验另外直接运行四张最小 GraphLowering 图：

| example input | placeholder 交给 env 的值 | `graph_inputs` | 直接返回后的 `graph_outputs` |
| --- | --- | --- | --- |
| Python `3` | SymPy `Integer(3)` | `scalar → 3` | `ShapeAsConstantBuffer` |
| Python `2.5` | SymPy `Float(2.5)` | `scalar → 2.5` | `ShapeAsConstantBuffer` |
| unbacked `SymInt u0` | 符号表达式 `u0` | `scalar → u0` | `ShapeAsConstantBuffer` |
| `None` | `None` | 不登记 scalar | `NoneAsConstantBuffer` |

普通 int/float 与 `SymInt/SymFloat/SymBool` 都先转成 SymPy 表达式，后续符号算术在编译期操作表达式，不创建 InputBuffer。对于 `u0`，编译器保留这个名字和它后来获得的范围/等式约束，生成的程序在实际值出现后检查相应条件。`None` 和约定为未使用的 `BackwardState` 返回 `None`；后者必须没有实际用途。

当前源码还有一个需要精确表述的边界：最小实验让 Python `True` 成为 placeholder 并直接作为图输出时，placeholder 先把它登记成 SymPy `BooleanTrue`，但 `output()` 随后检查每个 graph input 只允许 `TensorBox` 或 `sympy.Expr`，于是报 `Unsupported inductor graph input type: BooleanTrue`。这只证明当前“布尔 placeholder 直接返回”的人工图出口，不应扩写成所有编译图都不能使用布尔条件；实际控制流通常有自己的 predicate 接口和上游专门化。

这里没有读取 `[-2,1,3]` 的三个数据值。InputBuffer 表达“运行时从 arg0_1 读取”，所以同一编译版本以后可以接收满足 guards 的其他数值。

## call_function：先从 env 取 IR 参数，再按 target 分派

处理 add 时，FX node 的参数是两个 placeholder node。`fetch_args_kwargs_from_env` 将它们换成刚才的两份 TensorBox：

```text
FX: add(arg0_1_node, arg1_1_node)
               │ 从 env 解引用
               ▼
IR 参数: add(TensorBox(InputBuffer arg0_1),
             TensorBox(InputBuffer arg1_1))
```

`GraphLowering.call_function` 的主分派是：

1. 容器上的 `operator.getitem` 直接用 Interpreter 的 Python 行为取项；
2. 带 `_inductor_lowering_function` 标记的 pattern handler 直接调用；
3. 普通 ATen target 在 `lowerings[target]` 查处理函数；
4. target 没有注册 lowering 时，先看它是否在允许交给外部调用的名单中：在名单中就建立 fallback，wrapper 运行时直接调用该 ATen operator。若开启 `implicit_fallbacks`，不在显式名单中的可用 ATen operator 也可以自动走这条路；关闭时则报 missing-operator。若该 operator 本应先被 decomposition 拆成已有的小算子，错误信息会指出应补 decomposition，而不是悄悄生成未知 IR。

lowering 内部异常会包装成 `LoweringException`，附上 target 和参数上下文。这样错误出口仍能指出正在解释哪个算子。

这些分支交出的对象并不相同：

| 当前 FX node | 分派时拿到什么 | 本层实际出口 |
| --- | --- | --- |
| 普通 add/relu/mul | target 已在 `lowerings`，参数是 env 解出的 TensorBox/常数 | Pointwise；最终输出时成为 ComputedBuffer |
| cond 之后的 `operator.getitem` | cond lowering 返回的 Python list | 直接取出其中一个 `TensorBox→MultiOutput`，不再查 ATen lowering 表 |
| pattern lowering handler | target 带 `_inductor_lowering_function` | 直接调用 handler；handler 构造约定的 IR |
| histc fallback | 已注册/按策略建立 fallback lowering | `FallbackKernel` operation 及其 `MultiOutput` |
| 未注册且禁止隐式 fallback | OpOverload，无可用 lowering | 有 decomposition 时抛 `MissingOperatorWithDecomp`，否则抛 `MissingOperatorWithoutDecomp` |

实验 07 实际观察到 histc 的 operations 为 `FallbackKernel, MultiOutput`，cond 父图先得到 `list[TensorBox→MultiOutput→Conditional]`，紧接着内置 getitem 取出该项。missing-operator 两种异常是当前源码的明确失败出口，本实验没有为了制造错误而篡改全局 lowering 注册表。

### add、relu、mul 如何串起来

add lowering 返回 `TensorBox(StorageBox(Pointwise))`。Interpreter 将它记为 add node 的结果。relu 取回这个 TensorBox，以其 loader 作为输入构造新的 Pointwise；mul 再取回 relu 的结果并接上常数 2。

概念上，env 中的内容依次是：

```text
arg0_1 → 读取输入 x 的说明
arg1_1 → 读取输入 y 的说明
add    → i ↦ load(x,i) + load(y,i)
relu   → i ↦ max(add_rule(i), 0)
mul    → i ↦ relu_rule(i) * 2
```

每一步都能引用前一步，但 add/relu 没有注册 buffer/operation。它们的计算规则嵌进后继规则。

## run_node 比 call_function 多做哪些图级工作

`call_function` 只解决 target 分派。外围 `run_node` 还负责让本节点产生的 IR 与全图约束一致。

### 记录 origin

当前 FX node 和输入 IR 已有的来源记录会合并进上下文。例如 FX 节点名是 `add`，它的 stack trace 指向用户文件 `model.py:12`，那么在这个上下文里新建的 Pointwise/ComputedBuffer 会记住这项来源；后续 Scheduler 的节点说明和 Codegen 的源码行映射可以把生成循环重新指回这次 add。这里记录的不是 tensor 数据，而是“这段 IR 由哪些 FX 节点和用户代码位置产生”。

### 先处理特殊分支

这一步有三类例外。遇到当前 codegen 不能读写的 dtype，就不调用普通 lowering，而是建立 fallback；某些库调用要求输入满足固定 stride，会先插入布局调整，如果该调用还会改写输入，则把“已被修改”的状态从调整后的参数传回原参数；对符号整数的加减比较，结果已经保存在 `node.meta["val"]` 的符号表达式中，可以直接取出，不必建立 tensor IR。

### 满足输出或消费者的布局要求

若当前结果直接作为 graph output，或要传给 `as_strided` 类操作，`run_node` 会依据 FX meta 的 stride 要求决定是否固定 stride、插入布局转换或物化 view。用户可见输出不能随意采用与 eager 合同不兼容的排列。

贯穿案例的 mul 是 output 的直接输入。它原先是 Pointwise，但输出必须有实际 storage，`require_exact_strides` 路径触发物化：

```text
TensorBox → StorageBox → Pointwise
                   realize()
                         ▼
TensorBox → StorageBox → ComputedBuffer(name=buf0,
                                        layout=Flexible/决定后的布局,
                                        data=Pointwise)
```

物化同时调用 `register_buffer` 和 `register_operation`，因此实验在 mul 之后第一次看到 buffers=1、operations=1。

### 按多用户和读取复杂度决定是否提前物化

如果一个结果有多个 users，而某个 user 要求真实输入，或 output 也使用这份多用户 Pointwise/Reduction，`run_node` 可以 `realize_hint/realize`。读次数过多、Pointwise inner function 过大也可能触发物化，避免无限复制计算或递归表达式过深。

这是一项存储/重算权衡，不改变 FX env 的引用含义。具体阈值和 layout 行为在 Q23 展开。

### 登记读者和新符号

`register_users_of(result)` 查看 TensorBox 将读取哪些 buffer，把当前结果加入 `name_to_users[read_name]`。以后某 buffer 被 mutation 时，`mark_buffer_mutated` 能先物化所有仍读取旧版本的结果。

`run_node` 还记录处理前的 buffers/operations 列表长度，处理后只检查新增部分。若某项数据相关计算第一次产生没有样例整数可代入的符号 u0，它会保存“u0 来自这项结果”的绑定；若后续要求 `u0>=0`，则把这条尚不能在编译期判断的条件登记成运行时标量断言，源码对象名为 `AssertScalar`。这样数据相关符号在运行时取得实际值后仍会接受必要检查。

因此，`run_node` 的出口不保证总是“一个 TensorBox”。符号 magic method 可直接交出 SymPy 表达式；cond lowering 交出 TensorBox 列表；unsupported dtype 的 call_function 可绕过普通 lowering 表走 fallback；新绑定的 unbacked symbol 还可能让本节点额外登记 `AssertScalar`。env 保存的是该 FX node 在 Lowering 层的实际结果，后续节点再按 Python 容器或 IR 对象类型消费它。

## output：把返回请求变成稳定的存储接口

FX `output` node 从 env 取得要返回的 IR 值。当前实现先统一为 tuple/list，并限制允许的返回对象类型。对 Tensor 结果执行 `ExternKernel.realize_input`，确保最终返回有可引用的 storage；再根据 FX output meta 调整 size-0/1 维度中不影响实际位置的 stride 差异。

贯穿案例此时 `graph_outputs` 保存 `StorageBox(ComputedBuffer buf0)`，而 `output()` 本身返回 `None`：它通过修改 GraphLowering 的图级状态交付结果，不把另一份 IR 继续传给后继 node，因为 output 已是终点。

随后它逐个检查 graph input 是否仍指向原 InputBuffer。若输入被修改，出口必须把新值写进 `graph_inputs_original[name]` 指向的原输入存储。当前源码完成这件事的内部类名是 `MutationLayoutSHOULDREMOVE`；名字里的 `SHOULDREMOVE` 只是维护者留下的实现命名，不代表一种需要学习的新布局。它的实际作用是把“新值的计算”与“写入原输入 storage”绑在一起。若 `graph_outputs` 中正好含有被替换的 StorageBox，还会将那一项改回原 InputBuffer 身份。最后 `finalize()` 让每个 buffer 决定布局。

实验 06 的用户函数是 `old=x+1; x.copy_(y); return old,x`。在实际 AOT/post-grad 图合同中，GraphLowering 的普通 `graph_outputs` 只含 old 的 `ComputedBuffer buf0`；x 的更新作为输入 mutation 副作用交回，外层编译合同再重建用户可见的第二个返回 alias。运行结果是 old=`[1,2,3]`、调用者 x=`[9,9,9]`，且 returned_x 与调用者 x 的 data pointer 相同。不能把用户函数的两个返回值误写成 GraphLowering 一定有两个普通 graph outputs。

## 一张表连接本层入口与出口

| 阶段 | 具体对象变化 | 下一消费者 |
| --- | --- | --- |
| placeholder | FakeTensor 属性 → InputBuffer/TensorBox，登记输入名 | 后续 FX node 通过 env 取用；wrapper 使用输入接口 |
| call_function | env 中 IR 参数 → lowering → 新 IR 结果 | 后续 node 或 output |
| run_node 收尾 | origin、layout、realize、users、symbol asserts | Scheduler 与调试/运行时约束 |
| output | 返回 IR → graph_outputs；处理 mutation 与最终 layout | `GraphLowering.codegen()` |

## 用新请求检查理解

请求 `a=x+y; return a, relu(a)` 时，add 有两个 users，其中 output 直接返回 a。`run_node` 不能只把 add 规则无限内联到 relu：a 自身必须有返回 storage；多用户/output 条件会促使 a 物化。relu 可以读取这个 buffer，并为第二项输出形成另一个 ComputedBuffer。Scheduler 因而可能看到两项 operation，而不是贯穿案例的一项。

请求只返回 `(x+y)*2` 时，add 没有独立的输出身份，保持延迟规则即可。这两种情况数值算式相近，但 FX users 和 output 合同改变了 Lowering 的物化结果。

再加入两类节点，可以检验 env 与 operation 注册表是否混淆：

- `histc(x)` 的 fallback node 处理完后，env 指向它的 MultiOutput TensorBox，同时 `operations` 已含 FallbackKernel 和 MultiOutput；这不是延迟 Pointwise。
- `torch.cond(pred, true_fn, false_fn, [x])` 会先分别 Lower 两个子图，再在父图登记 Conditional/MultiOutput；父图 getitem 从返回 list 取出 TensorBox。运行时只按 predicate 选择一支，但编译时两支都必须建立一致输出合同。

若把 `old=x+1` 从 mutation 例的返回值中删除，旧值是否仍需在 copy_ 前物化，取决于它是否还有其他实际消费者；不能继续用“它是 graph output”作为理由。若仍有 `old*2` 跨过 copy_ 使用，`register_users_of` 与 `mark_buffer_mutated` 会保护这个旧版本读者。这把 env 引用、用户关系和输入写回三份状态连接起来。
