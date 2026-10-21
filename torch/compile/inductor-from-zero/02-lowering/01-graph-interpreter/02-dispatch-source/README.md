# GraphLowering 分派：把 FX target 接到 lowering 或常量 IR

[返回 GraphLowering](../README.md) · [上一篇：输入登记](../01-node-entry-source/README.md) · [返回 Lowering](../../README.md)

解释器处理到 `a = aten.add.Tensor(x, y)` 时，FX 只给出 target 和已经从 env 取出的参数。GraphLowering 必须决定由哪个 lowering 构造 IR；处理到 `get_attr` 时，则要决定模块属性是内联成值、登记为 ConstantBuffer，还是保留子图/对象身份。

本叶子完整保留当前 `torch/_inductor/graph.py` 中这一职责的八个方法：`allocate_non_dup_const_name`、`add_tensor_constant`、`constant_name`、`call_function`、`can_inline_constant`、`get_attr`、`call_module`、`call_method`。按产品文件出现顺序排列，方法内部没有省略；教学注释以 `# 教学：` 标记。

## 两类 FX 节点先分开

```text
call_function 节点
  target + 已转换的 IR args
      ├─ Python getitem
      ├─ pattern 注入的 lowering
      ├─ lowerings[target]
      └─ 建立 fallback / 报缺失算子

get_attr 节点
  GraphModule 中 target 对应的对象
      ├─ 子 GraphModule → Subgraph
      ├─ ScriptObject → TorchBindObject
      └─ Tensor → 标量 Constant / 小向量内联 / ConstantBuffer
```

两条路径最后都给 FX Interpreter 一个可放入 env 的结果，但来源不同：函数节点执行一条构造规则，属性节点读取 GraphModule 已有对象。

## 完整源码

进入第一段源码时，`name: Optional[str]` 表示调用者可以给常量建议名字，也可以传 `None`；本例用 `name="weight"`。`data` 是一份 Tensor 常量，`use_runtime_constant_folding=False`，constants 与 constant_reprs 起初为空，也没有名称冲突或 device override。函数先检查现有常量是否已指向同一块数据；若没有，就登记为 weight，constant_reprs 保存用于缓存/诊断的常量表示。后面的 `call_function` 案例则固定 `target=aten.add.Tensor`、`args=(x_ir,y_ir)`、`kwargs={}`，最终从 `lowerings[target]` 取得处理函数并返回 add 的 IR。ScriptObject/TorchBind 分支不在这次调用中进入。

```python
def allocate_non_dup_const_name(
    self, name: Optional[str], data: Union[Tensor]
) -> str:
    orig_name = name
    # 教学：未启用运行时常量折叠时，先寻找指向同一存储位置的已登记常量。
    if not config.aot_inductor.use_runtime_constant_folding:
        for constant_name, value in self.constants.items():
            if (
                not data.is_mkldnn
                and data.size() == value.size()
                and data.stride() == value.stride()
                and data.dtype == value.dtype
                and data.device == value.device
                and data.untyped_storage().data_ptr()
                == value.untyped_storage().data_ptr()
                and data.storage_offset() == value.storage_offset()
            ):
                return constant_name

    # 教学：无名字时按当前常量数产生基础名；数字开头不能直接作生成代码变量名。
    if name is None:
        name = f"constant{len(self.constants)}"
    assert name is not None
    if name[0].isdigit():
        name = f"constant_{name}"
    name = self.qualify_name(name)
    # We may generate a var name for each constant in the codegen.
    # Let's only keep sane characters.
    # 教学：将点号等字符替换为下划线，再用后缀解决名字冲突。
    prefix = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = prefix
    cnt = 0
    while name in self.constants:
        name = f"{prefix}_{cnt}"
        cnt += 1
    # 教学：三张表分别保存实际 tensor、用于缓存辨识的表示和原属性名。
    self.constants[name] = data
    self.constant_reprs[name] = (
        f"{data.device!r} {data.dtype!r} "
        f"{tuple(data.size())!r} {tuple(data.stride())!r} "
        f"{hash(data):x}"
    )
    self.allocated_constant_name[name] = orig_name  # type: ignore[assignment]
    return name

def add_tensor_constant(
    self, data: Tensor, name: Optional[str] = None
) -> TensorBox:
    # 教学：先取得唯一注册名，再用真实 tensor 的设备、dtype、size、stride 建布局。
    new_name = self.allocate_non_dup_const_name(name, data)
    return TensorBox.create(
        ir.ConstantBuffer(
            new_name,
            FixedLayout(data.device, data.dtype, *self.static_sizes_strides(data)),
        )
    )

def constant_name(self, name: str, device_override: Optional[torch.device]) -> str:
    """
    We AOT copy constants to the devices they are needed on.
    If device_override doesn't match the constant's device, then
    copy it and return a different name.
    """
    # 教学：设备符合或调用方没有覆盖要求时，复用原登记名。
    if self.constants[name].device == device_override or device_override is None:
        return name
    # 教学：跨设备复制时关闭 Python dispatch mode，避免 .to() 只制造 fake tensor。
    with torch.utils._python_dispatch._disable_current_modes():
        # caller might have OrderedSet fake tensor mode which will create a fake tensor
        # when calling .to, so unset modes here
        return self.allocate_non_dup_const_name(
            f"{name}_{device_override.type}{device_override.index or 0}",
            self.constants[name].to(device_override),
        )

def call_function(self, target: Callable, args: Any, kwargs: Dict[str, Any]) -> Any:  # type: ignore[type-arg, override]
    # 教学：Python 容器取项不需要 ATen lowering，交给 FX Interpreter 直接执行。
    if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
        return super().call_function(target, args, kwargs)

    # hasattr on OpOverloadPacket is slow, check isinstance first
    if not isinstance(target, torch._ops.OpOverloadPacket) and hasattr(
        target, "_inductor_lowering_function"
    ):
        # passthrough lowerings from .pattern_matcher
        # 教学：pattern 可把已经是 lowering 的 callable 直接放成 target。
        return target(*args, **kwargs)

    if target not in lowerings:
        # 教学：走到这里的未知 target 必须是具体 OpOverload，不能是任意 callable。
        assert isinstance(
            target, torch._ops.OpOverload
        ), f"{target} is not an OpOverload"
        base_name = target.name().split(".")[0]
        if base_name in FALLBACK_ALLOW_LIST:
            # 教学：白名单操作允许当场注册 fallback lowering。
            make_fallback(target)
        elif config.implicit_fallbacks:
            # 教学：允许隐式 fallback 时，分解是否存在只影响日志错误类型；仍建立 fallback。
            error = (
                MissingOperatorWithDecomp
                if get_decompositions([target])
                else MissingOperatorWithoutDecomp
            )
            log.info(
                "Creating implicit fallback for:\n%s",
                error.operator_str(target, args, kwargs),
            )
            make_fallback(target)

        elif get_decompositions([target]):
            # There isn't a good way to dynamically patch this in
            # since AOT Autograd already ran.  The error message tells
            # the user how to fix it.
            raise MissingOperatorWithDecomp(target, args, kwargs)
        else:
            raise MissingOperatorWithoutDecomp(target, args, kwargs)

    try:
        # 教学：无论原生注册还是刚建立的 fallback，都从同一张 lowerings 表取出并调用。
        log.debug("  via %s", lowerings[target])  # type: ignore[index]
        out = lowerings[target](*args, **kwargs)  # type: ignore[index]
        return out
    except Exception as e:
        # 教学：保留原 traceback，同时把 target、args、kwargs 加进 lowering 错误上下文。
        raise LoweringException(e, target, args, kwargs).with_traceback(
            e.__traceback__
        ) from None

@staticmethod
def can_inline_constant(t: torch.Tensor) -> bool:
    """
    True if this is a small constant attr that will be inlined.
    """
    # 教学：当前条件只接受长度不超过 8 的一维 tensor。
    return len(t.shape) == 1 and t.shape[0] <= 8

def get_attr(
    self, target: str, args: Tuple[()], kwargs: Dict[str, object]  # type: ignore[override]
) -> Union[Constant, TensorBox, ir.Subgraph, TorchBindObject]:
    # this is a constant
    # 教学：target 可为点分路径，从 GraphModule 递归取得真实属性。
    value = getattr_recursive(self.module, target)  # type: ignore[arg-type]

    if isinstance(value, torch.fx.GraphModule):
        return ir.Subgraph(name=target, graph_module=value)

    if isinstance(value, torch._C.ScriptObject):
        # 教学：TorchBind 对象保留对象身份，并进入专用常量表。
        self.torchbind_constants[target] = value
        self.constant_reprs[target] = ""
        return TorchBindObject(target, value)

    assert isinstance(value, torch.Tensor)
    if (
        config.aot_inductor.use_runtime_constant_folding
        or config.always_keep_tensor_constants
        or unsupported_output_tensor(value)
    ):
        # 教学：这些配置/类型要求保留 tensor 存储，不能内联成算术值。
        return self.add_tensor_constant(value, target)

    with no_dispatch():
        if value.shape == ():
            # 教学：零维 tensor 取出单值，但继续保留 dtype 和 device。
            return Constant(value.item(), value.dtype, value.device)
        if self.can_inline_constant(value):
            log.debug("Inlining constant: %s ", str(target))
            # tensor lowering has constant inlining logic
            from .lowering import tensor

            # 教学：小一维常量交给 tensor lowering 形成可内联 IR。
            return tensor(value.tolist(), dtype=value.dtype, device=value.device)

    # 教学：其余 tensor 登记为具名 ConstantBuffer。
    return self.add_tensor_constant(value, target)

def call_module(self, target: Any, args: Any, kwargs: Any) -> NoReturn:
    # 教学：到 GraphLowering 时不应再有普通 call_module；出现即违反输入图前提。
    raise AssertionError

def call_method(self, target: Any, args: Any, kwargs: Any) -> NoReturn:
    # 教学：方法调用也应在更早阶段规范化为可分派的函数节点。
    raise AssertionError
```

源码副本去掉类级四空格缩进；产品注释、条件顺序与方法内部代码完整保留。

## 已注册 add 怎样走完分派

Q211 已将 x、y 登记为 TensorBox。解释器从 env 取参后调用：

```text
target = torch.ops.aten.add.Tensor
args   = (x_tensorbox, y_tensorbox)
kwargs = {}
```

该 target 已在全局 `lowerings` 表中，因此跳过未知算子分支，直接执行 `lowerings[target](*args, **kwargs)`。处理器交回 `TensorBox(StorageBox(Pointwise))`；`call_function` 不在这里登记 operation，也不计算真实 `[x0+y0,...]`。返回对象由 FX Interpreter 放进 add node 对应的 env 项，供下一个 node 取得。

如果处理器内部因广播不合法等原因抛出异常，外层改抛 `LoweringException`。错误携带“哪个 target、收到哪些 IR 参数”，而不是只剩处理器内部的一行错误。

## 未注册操作的四个出口

| 条件 | 动作 | 是否继续到统一调用点 |
| --- | --- | --- |
| base name 在 `FALLBACK_ALLOW_LIST` | `make_fallback(target)` | 是 |
| 开启 `implicit_fallbacks` | 记录相应提示并 `make_fallback` | 是 |
| 禁止 fallback，但存在 decomposition | `MissingOperatorWithDecomp` | 否，立即报错 |
| 禁止 fallback，且无 decomposition | `MissingOperatorWithoutDecomp` | 否，立即报错 |

`make_fallback` 的结果也登记进 `lowerings[target]`，所以成功分支最终共用同一个 try/except 调用点。存在 decomposition 不代表 GraphLowering 此刻还能自动回到 AOTAutograd 重跑分解；源码注释明确此阶段已经太晚，因此在禁用 fallback 时给出专门错误。

独立 probe 保持产品 `call_function` 不变，只在隐式 fallback 一例中把 `make_fallback` 换成可记录的最小注册器，实际得到：

```text
container getitem ((10,20),1) -> 20
带 _inductor_lowering_function 标记的 callable -> 直接返回 ('passthrough',7)
addcdiv（有 decomposition、无 lowering、禁止 fallback） -> MissingOperatorWithDecomp
linalg_eig（无 decomposition、无 lowering、禁止 fallback） -> MissingOperatorWithoutDecomp
linalg_eig（允许 implicit fallback） -> make_fallback 一次，再从 lowerings 统一调用
lowering 内部 ValueError -> LoweringException，首行保留 "ValueError: lowering boom"
call_module / call_method -> AssertionError
```

mock 只隔离“注册 fallback 后是否确实走统一 lowerings 调用点”；它不证明真实 `FallbackKernel` 对 linalg_eig 的数值执行。真实 fallback IR 已由 Q24 的 histc 案例覆盖，但两项证据不能混成一次实验。

## 同一组属性值怎样得到不同 IR

假设 GraphModule 含以下属性：

```text
bias0 = tensor(2.0)                  # 0 维
bias1 = tensor([1.,2.,3.])           # 长度 3 的 1 维
table = arange(12).reshape(3,4)      # 2 维
branch = 一个 GraphModule
obj = 一个 ScriptObject
```

在未强制保留 tensor 常量、未启用运行时常量折叠时：

| target | 分支 | env 中结果 | 图级状态变化 |
| --- | --- | --- | --- |
| bias0 | `shape == ()` | `Constant(2.0,dtype,device)` | 不登记 ConstantBuffer |
| bias1 | 一维且长度 ≤ 8 | tensor lowering 的内联 IR | 不走 `add_tensor_constant` |
| table | 其余 Tensor | `TensorBox(ConstantBuffer)` | `constants` 等三张表增加记录 |
| branch | GraphModule | `ir.Subgraph` | 保留子图模块 |
| obj | ScriptObject | `TorchBindObject` | 写入 `torchbind_constants` |

若开启 `always_keep_tensor_constants`，bias0 和 bias1 也会在到达内联判断前走 `add_tensor_constant`。配置改变的是表示路径，不是常量数值。

上述五类中，本轮实际构造并调用当前 `get_attr` 的四类结果是：

```text
scalar_const -> Constant
small_const  -> TensorBox→StorageBox→Pointwise
table_const  -> TensorBox→StorageBox→ConstantBuffer
branch       -> Subgraph
constants keys 只有 ['table_const']
```

在 `always_keep_tensor_constants=True` 下，scalar 与 small 都变为 ConstantBuffer，constants keys 为两者。ScriptObject 是由 TorchScript/C++ 注册、从 GraphModule 属性取出的有状态对象；TorchBindObject 是 Inductor 为它建立的 IR 包装，后续自定义 operator 可以把这份对象作为参数传回已注册的 C++ 实现。它不是普通 tensor 常量，因此不会走 size/stride layout 或小常量内联分支。

## 常量名字与共享存储的具体推演

模块属性名 `layer.weight` 先清理为 `layer_weight`。若该名字已在 `constants`，新记录依次尝试 `layer_weight_0`、`layer_weight_1`。子图名为 `true_graph` 时还会先限定成 `true_graph_layer.weight`，再清理点号。

若两个属性 tensor 具有相同 size、stride、dtype、device、底层 data pointer 和 storage offset，且没有启用运行时常量折叠，第二次登记直接返回第一份 `constant_name`。这不是比较数值相等：两份内容恰好相同但存储独立的 tensor 不会因此合并；同一 storage 上 offset 不同的 view 也不会合并。

实际命名状态为：

```text
base[:3] 以 layer.weight 登记 -> layer_weight
同一 view 再以 alias 登记      -> layer_weight（复用）
base[1:4]，同 storage 不同 offset -> layer_weight_0
独立 clone，数值相同          -> layer_weight_1
名字 1.bad-name              -> constant_1_bad_name
```

跨设备请求则由 `constant_name(name, cuda:1)` 创建实际 `.to(cuda:1)` 副本，并以类似 `name_cuda1` 的基础名重新登记。未启用运行时常量折叠时，后来同 storage 条件的登记可由前述去重逻辑找到已有记录；启用时这段去重循环按源码被跳过。当前环境未运行 CUDA 跨设备复制，这一段限于源码合同。

## 用新情况检验理解

1. 长度为 9 的一维常量不会内联，因为当前上限为 8；它进入 ConstantBuffer。
2. `operator.getitem` 的第一个参数若是 Python tuple，直接由父 Interpreter 取项；若是一个 IR 多输出对象，则不满足这个分支，需要产品注册的相应 lowering。
3. 一个未知普通 Python callable 既无 `_inductor_lowering_function` 标记、也不是 OpOverload，会在断言处失败，而不会悄悄生成 fallback。
4. 两个数值完全相同、但分别分配的 3×4 tensor 具有不同 data pointer，名字去重不会把它们当成同一常量。

Q212 至此交出可放入 env 的计算 IR、常量 IR 或明确错误。Q213 将继续解释 output 如何把 env 中的最终值变成 `graph_outputs`，并处理输入 mutation。
