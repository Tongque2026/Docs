# backend 分派循环源码：最终节点怎样交给生成后端

[返回后端分派篇](../README.md) · [返回 Scheduler](../../README.md) · [返回总览](../../../README.md)

Q34 已经让我们按节点类型画出提交时间线。本篇打开实际控制循环：它维护哪几份状态，为什么在 extern/template/device 边界先 flush，以及分派后哪个 `node` 被加入 available/completed。

先固定循环直接维护的状态：

| 状态 | 保存什么 | 谁使用它 |
| --- | --- | --- |
| `self.backends[device]` | 每个 device 已缓存的有状态生成器；待提交工作保存在生成器内部 | 分派调用继续向实例交任务；`flush()` 要求所有已创建 backend 提交 |
| `current_device` | 当前 wrapper device guard 所在设备 | 下一节点决定是否 flush、退出旧 guard、进入新 guard |
| `buffer_names_to_free` | 已到最后一次使用、等待安全提交点释放的 buffer 名 | `flush()` 在 backend 工作写出后调用 `free_buffers()` 兑现 |
| `available_buffer_names` / `completed_operations` | Scheduler 外层循环已经登记的 buffer 名和 operation 名 | available 在调度构造时用于裁掉已由输入、常量或先前结果满足的依赖；completed 在后续 inplace 判断中过滤已经处理过的使用者 |

这两份“完成”集合是编译期记账，不表示设备已经执行。具体 backend 还会在 `mark_run()` 中分配输出并记录生成状态；template 案例会展示 backend 处理范围与 Scheduler 外层记账范围为何不同。

### `_codegen` 接到的不是一组全空状态

`Scheduler` 构造时已经完成节点建立、依赖、DCE、融合和 last usage，随后 `codegen()` 才调用 `_codegen()`。入口各状态如下：

- `self.nodes` 是融合、组合和最终拓扑处理后的节点序列，每个节点已有 `last_usage`；
- `completed_operations` 在 Scheduler 构造时建为空集合，codegen 前没有正常路径向其中加入 operation；
- `available_buffer_names` 初始已经包含图输入、常量和 torchbind 常量，构造期间发现的新常量也会加入；它不是空集合；
- `current_device` 在进入 codegen 前被明确重置为 `None`，`buffer_names_to_free` 也建为空集合；
- `self.backends` 虽然最初为空，却不保证到 codegen 时仍为空。普通 `SchedulerNode` 构造时就可能为计算 group 取得 backend，融合检查也会访问 backend。因此 `_codegen` 负责复用或按需补建实例，不等于“第一个普通节点才首次创建 backend”；
- `origin_to_index` 是 `enter_context()` 按需填充的 FX node 顺序缓存。

这些对象都属于同一个 Scheduler。`_codegen` 没有另建事务副本；后文的 backend、wrapper 和 Scheduler 集合会按语句顺序逐项改变。

```text
最终节点序列
    -> 选择 wrapper 上下文
    -> 必要时 flush 与切换 device guard
    -> 加入本节点 last_usage 到待释放集合
    -> template / extern / foreach / ordinary / nop 五路分派
    -> 可选 debug sync
    -> 更新 available buffers 与 completed operations
    -> backend 主动请求时 flush
遍历结束：退出最外 guard，再 final flush
```

## K01：backend 按需缓存，flush 只提交实例内部已有的工作

### flush 先提交所有已创建 backend，再尝试释放

`BaseScheduling` 是各设备代码生成器共同实现的接口；`self.backends` 把具体 `torch.device` 映射到这类对象。映射里有实例，不等于实例内部已经积累待提交节点：一个 CPU backend 可能早在 SchedulerNode 构造时就已缓存，但其 codegen group 仍为空。下一次 flush 会调用所有已缓存实例的 `flush()`，随后 `free_buffers()` 把已经到最后使用的数据转成 wrapper 释放请求。

```python
def flush(self) -> None:
    # 教学：一个 Scheduler 可缓存多个 device backend；逐一提交它们的待生成工作。
    for backend in self.backends.values():
        backend.flush()
    # 教学：kernel 调用已经写出后，才把 pending last-usage 转成 wrapper free 请求。
    self.free_buffers()
```

`self.backends` 只含已经按需创建的实例。第一个节点前调用 `flush()` 时，字典可能为空，也可能已有一个内部没有待提交工作的 backend；两种情况都不必生成 kernel，但 `free_buffers()` 仍会运行。flush 不是“必然生成一个 kernel”，而是“让当前所有 backend 兑现各自已经积累的工作，并在其后兑现可释放项”。

### backend 创建与缓存的完整实现

```python
def create_backend(self, device: torch.device) -> BaseScheduling:
    # 教学：GPU device 必须在 Lowering 时补齐 index；Scheduler 不替它猜设备号。
    assert (
        not is_gpu(device.type) or device.index is not None
    ), f"{device} should have been normalized in lowering"
    V.graph.add_device_info(device)

    device_scheduling = get_scheduling_for_device(device.type)
    if device_scheduling is None:
        raise RuntimeError(f"Unsupported device type: {device.type}")

    if not has_triton():
        if (
            device.type == "cuda"
            and (device_props := torch.cuda.get_device_properties(device)).major < 7
        ):
            raise RuntimeError(
                f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability {device_props.major}.{device_props.minor}"  # noqa: B950
            )
        elif is_gpu(device.type):
            raise RuntimeError(
                "Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at https://github.com/openai/triton"  # noqa: B950
            )

    return device_scheduling(self)

def get_backend(self, device: torch.device) -> BaseScheduling:
    # 教学：完整 torch.device 是缓存键；首次创建，后续返回同一有状态实例。
    if device not in self.backends:
        self.backends[device] = self.create_backend(device)
    return self.backends[device]
```

创建顺序有可观察含义：先把 device 信息登记进图，再查已注册 Scheduling 类；不支持的 type 直接报错。没有可用 Triton 时，旧 CUDA capability 与其他 GPU 分别给出不同 RuntimeError。成功后以 `device_scheduling(self)` 建实例，使 backend 能回访同一 Scheduler 的依赖与 wrapper 状态。

缓存键是完整 `torch.device`。CPU 的重复请求复用同一 `CppScheduling`，因此前一个 `codegen_node` 放进 `kernel_group` 的内容能被稍后的 `flush` 看见；若每次创建新实例，积累状态就会丢失。

### wrapper 上下文选择最晚的 FX origin

先代入融合组 Fused(op0,op1)：op0 的 origins 是 FX 节点 add（图中序号 3），op1 的 origins 是 relu（序号 4）以及它继承的 add。字典去重后按序号排序，最后一项是 relu，因此 wrapper 的 current node 上下文设为 relu；该组生成的源码行与 debug 信息优先指向这个最晚的用户计算。

```python
def enter_context(self, node: BaseSchedulerNode) -> None:
    def get_order(n: torch.fx.Node) -> int:
        if n not in self.origin_to_index:
            self.origin_to_index.update({n: i for i, n in enumerate(n.graph.nodes)})
        return self.origin_to_index[n]

    # Use a dict to have ordering
    origins = {
        (get_order(e), e): None
        for n in node.get_nodes()
        if n.node is not None
        for e in n.node.get_origins()
    }
    origins = list(origins.keys())
    if origins:
        _, last = max(origins, key=operator.itemgetter(0))
        V.graph.wrapper_code.enter_context(last)
```

融合组可能含多个成员，每个成员又可能带多个 origin。字典按 `(图中顺序, FX node)` 去重并保序；非空时选择顺序号最大的 origin。若所有成员都没有 IR node/origin，函数什么也不提交。它只选择 wrapper 的源码/调试上下文，不改变调度依赖。

### pointwise 的首尾状态

实验 11 的最终列表只有 `op0_op1`。到 `_codegen` 入口，CPU backend 已可能在此前节点构造或融合检查中缓存；这里关心的是它尚未接到本轮 `codegen_node`。状态应写成：

```text
backends={cpu: CppScheduling}     # 可能已缓存；内部待生成组为空
current_device=None
buffer_names_to_free={}
available_buffer_names={arg0_1,...}  # 已含本图输入与常量
completed_operations={}
```

访问 CPU 节点时 `cpu != None`，主循环先调用 `flush`。已缓存 backend 的组仍为空，因而这次不生成 C++ kernel；随后设置 `current_device=cpu`。普通分派通过 `get_backend(cpu)` 复用该实例，再把 op0/op1 累积进它的 kernel group。遍历结束的 final flush 提交 `[op0,op1]` 并兑现 pending free。若某个只有 extern/nop 的图在 codegen 前确实尚未创建 backend，同一控制流也允许字典为空；那只是另一种入口，不是普通 pointwise 的必然状态。

这解释了实验顺序：

```text
scheduler.flush                    # backend 可已缓存，但内部组为空
get_backend(cpu)=已有 CppScheduling
cpp.codegen_node(op0_op1)          # 只积累
scheduler.flush
cpp.flush([op0,op1])               # 真正提交
```

这个普通节点同时经过三层状态：`codegen_node` 先改变 C++ backend 内部待提交组；随后外层循环把 `buf0/buf1` 与 `op0/op1` 分别加入 available/completed；最后 `flush` 才让 backend 把 kernel 调用写进 wrapper。三者发生在不同语句，不能用“节点已经生成”同时代替。

### K01 理解验收

若首节点是 extern，且此前节点构造与调度判断都没有需要某个普通 backend，前置 `flush()` 时 `backends` 仍可能为空；extern 分派本身不需要普通 CPU backend。K03 的完整循环会在该节点之后查询对应 backend 的 `ready_to_flush()`，这一步才可能创建并缓存它。若同一 Scheduler 后来出现普通 CPU 节点，会复用这个实例。

K01 到此完成。K02 打开 extern 调用和主循环中的边界顺序。

## K02：extern 在 wrapper 中直接成调用，因此先切断普通 kernel group

```python
def codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode) -> None:
    assert isinstance(scheduler_node, ExternKernelSchedulerNode)
    # 'decide_inplace_update' stores the inplace update decisions in
    # the current kernel from where 'allocate' retrieve those decisions.
    # We have to make sure there is a non-NULL kernel handler to store
    # those inplace update decisions.
    counters["inductor"]["extern_calls"] += 1
    # 教学：临时 Kernel 只承载 inplace 决策与 mark_run 状态，不增加普通 kernel 计数。
    with V.set_kernel_handler(Kernel(increase_kernel_count=False)):
        scheduler_node.decide_inplace_update()
        scheduler_node.mark_run()
    node = scheduler_node.node
    assert isinstance(node, ir.ExternKernel), f"{type(node)=}"
    # 教学：IR extern 自己向 wrapper 写调用，而非进入 backend kernel group。
    node.codegen(V.graph.wrapper_code)
    self.free_buffers()
```

临时 `Kernel` handler 解决一个具体状态归属问题：`decide_inplace_update()` 需要把决定存进当前 kernel，后续分配逻辑再读取；extern 没有普通 backend kernel，所以这里显式提供非空 handler。`mark_run()` 更新节点输出状态，之后 IR extern 直接写 wrapper 调用，并立即尝试 free。

mixed 案例的最终节点为普通融合组 `op0_op1`、mm extern `op2`、sin 普通节点 `op3`：

| 时刻 | backend 累积 | Scheduler 动作 | 出口状态 |
| --- | --- | --- | --- |
| op0_op1 后 | C++ 组含 op0/op1 | available 加 buf0/buf1，completed 加 op0/op1 | 尚未提交 kernel |
| 到 op2 前 | 同上 | extern 条件触发 flush | 写出首个 C++ kernel，pending free 清空 |
| op2 分派 | 空普通组 | 临时 handler 决定 inplace，写 mm wrapper 调用 | op2 自行 free，再更新 buf2/op2 |
| op3 后 | C++ 组含 op3 | 普通状态更新 | 末尾 final flush 写 sin kernel |

extern 前的 flush 与 device 是否改变无关；本例三项全是 CPU，仍必须把此前普通组提交完，wrapper 才能按依赖顺序写 mm 调用。fallback 的 `FallbackKernel` 和 `MultiOutput` 也属于这条 extern 路径。

### K02 理解验收

若两个 extern 连续出现，每一个前面都会调用 Scheduler.flush；第一次 extern 后普通 backend 可能仍是空组，第二次前的 flush 因而可以再次不生成 kernel。两次 extern 各自写 wrapper 调用、mark_run 和 free，不能据“相邻且同 device”把它们放进普通 loop group。

K02 到此完成。K03 用完整 `_codegen` 看五类分派和 template 的变量重绑定。

## K03：一个循环按固定顺序处理五类节点

公开 `codegen` 只增加计时范围，所有状态推进都在 `_codegen`：

```python
def codegen(self) -> None:
    with dynamo_timed("Scheduler.codegen"):
        return self._codegen()
```

下面按源码顺序阅读 `_codegen`。先固定一组用于手算控制流的假定节点：CPU 普通 A、CPU 普通 B、CPU extern C、CUDA foreach D、nop E。图输入为 x、y，并假定没有常量、torchbind 常量或构造中新发现的常量，所以入口恰为 `available={x,y}`；`completed={}`、`current_device=None`、pending free 为空。CPU/CUDA backend 可以已经缓存，但内部待提交组均为空；stack 检查、debug sync、freezing 均关闭。各节点的名字如下：

| 节点 | operation 名 | 输出 buffer | `last_usage` |
| --- | --- | --- | --- |
| A ordinary | `{A}` | `{a}` | `{x}` |
| B ordinary | `{B}` | `{b}` | `{a}` |
| C extern | `{C}` | `{c}` | `{b}` |
| D foreach | `{D0,D1}` | `{d0,d1}` | `{c,y}` |
| E nop | `{E}` | `{e}` | `{d0}` |

A/B 在同一个 CPU backend group 积累；C 前先提交 A/B，再直接写 extern；D 触发设备切换、进入 CUDA guard并走 foreach；E 不切设备，只更新逻辑完成状态；循环结束退出 guard 并最终 flush。开头的测试开关只在测试配置下检查调用栈没有重复 frame。

```python
def _codegen(self) -> None:
    if config.check_stack_no_cycles_TESTING_ONLY:
        import torch._dynamo.convert_frame

        stack = traceback.extract_stack()
        seen = set()
        for frame in reversed(stack):
            # This is where maybe_cprofile is
            if (
                frame.name == "_compile_inner"
                and frame.filename == torch._dynamo.convert_frame.__file__
            ):
                break
            key = (frame.filename, frame.lineno)
            assert key not in seen, (
                f"Duplicate stack frame {frame.filename}:{frame.lineno}; "
                "did you add a decorator to one of the functions in this stack "
                "trace?  If so, try using a context manager instead."
            )
            seen.add(key)

    for node in self.nodes:
        try:
            log.debug(
                "Generating code for node %s with estimated runtime %f",
                node.get_name(),
                node.get_estimated_runtime(),
            )
        except Exception as e:
            log.debug(
                "Generating code for node %s with estimated runtime 0.0",
                node.get_name(),
            )

        # 教学：在分派前先选择本节点对应的 wrapper 源码上下文。
        self.enter_context(node)

        if not isinstance(node, NopKernelSchedulerNode) and (
            device := node.get_device()
        ):
            if (
                device != self.current_device
                or node.is_extern()
                or node.is_template()
            ):
                # 教学：device、extern、template 任一边界都先提交旧工作。
                self.flush()
            if device != self.current_device:
                if self.current_device and device_need_guard(
                    self.current_device.type
                ):
                    V.graph.wrapper_code.codegen_device_guard_exit()
                if device_need_guard(device.type):
                    assert device.index is not None, "device should have an index"
                    V.graph.wrapper_code.codegen_device_guard_enter(device.index)

                self.current_device = device

        # 教学：这里只记下本节点用完了哪些 buffer；遇到 flush 或写出 extern 调用时，
        # wrapper 才为这些 buffer 生成实际的释放语句。
        self.buffer_names_to_free.update(node.last_usage)

        if node.is_template():
            # 教学：局部 node 被重绑定为主模板；epilogue 由 backend 一并接收。
            node, *epilogue = node.get_nodes()
            self.get_backend(device).codegen_template(node, epilogue)
        elif node.is_extern():
            node = typing.cast(ExternKernelSchedulerNode, node)
            self.codegen_extern_call(node)
        elif node.is_foreach():
            node = typing.cast(ForeachKernelSchedulerNode, node)
            backend_ = self.get_backend(device)
            from .codegen.cuda_combined_scheduling import CUDACombinedScheduling
            from .codegen.simd import SIMDScheduling

            if isinstance(backend_, (SIMDScheduling, CUDACombinedScheduling)):
                backend = backend_
            else:
                raise AssertionError(f"{type(self)=}")
            backend.codegen_combo_kernel(node)
        elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
            self.get_backend(device).codegen_node(node)
        else:
            assert isinstance(node, NopKernelSchedulerNode)
            node.mark_run()

        if config.triton.debug_sync_kernel:
            self.get_backend(device).codegen_sync()

        # 教学：注意 template 分支此处的 node 已是主模板，不再是原融合组。
        self.available_buffer_names.update(node.get_buffer_names())
        self.completed_operations.update(node.get_operation_names())

        if not isinstance(node, NopKernelSchedulerNode):
            device = node.get_device()
            if device is not None and self.get_backend(device).ready_to_flush():
                self.flush()

    if self.current_device and device_need_guard(self.current_device.type):
        # exit the outermost CUDA device guard. this is
        # important for nested indentation codegen-ing.
        V.graph.wrapper_code.codegen_device_guard_exit()

    self.flush()
```

把上面的具体名字代入，且假定 backend 都不主动请求提前 flush：

| 处理完成到 | backend / wrapper 新动作 | pending free | `available_buffer_names` | `completed_operations` |
| --- | --- | --- | --- | --- |
| 入口 | backend 组为空，wrapper 尚无本例调用 | `{}` | `{x,y}` | `{}` |
| A | CPU backend 积累 A | `{x}` | `{x,y,a}` | `{A}` |
| B | 同一 backend 再积累 B | `{x,a}` | `{x,y,a,b}` | `{A,B}` |
| C | C 前 flush 写 A+B kernel 并释放 x/a；extern 写 wrapper 调用后立即释放 b | `{}` | 再加 c | 再加 C |
| D | 切换前空 flush，进入 CUDA guard；combo backend 接收 D | `{c,y}` | 再加 d0/d1 | 再加 D0/D1 |
| E | `mark_run()`，不生成 kernel，也不查 ready | `{c,y,d0}` | 再加 e | 再加 E |
| 结束 | 先写 guard exit，再 final flush；backend 提交已有工作，随后 free | `{}` | 不再变化 | 不再变化 |

这里的 available 一栏是累计集合，表中“再加”表示保留前面的所有名字。backend 已接收任务、wrapper 已写调用、Scheduler 已登记 available/completed 是三件事：例如 D 的 combo backend 先接收 D，外层两集合随后才更新；若最终 flush 尚未发生，wrapper 还不一定已有该 backend 的 kernel 调用。

### 分派顺序决定同一个对象走哪条路

判断依次是 template、extern、foreach、普通 Scheduler/Fused、nop。template 放在最前，是因为带 epilogue 的 `FusedSchedulerNode` 同时也是融合组；若先按普通 fused 判断，它会错误进入 `codegen_node`。extern 直接调用上节方法；foreach 还要检查 backend 是 `SIMDScheduling` 或 `CUDACombinedScheduling`，否则抛 AssertionError；普通节点进入 backend `codegen_node`；剩余对象必须是 nop，只 `mark_run()`。

nop 在 device 边界块前被排除，所以不会切 guard，也不查询 `ready_to_flush()`；它仍更新待释放名字以及 available/completed，也不会向普通 backend 或 wrapper 添加一次 kernel 调用。

当前源码中，debug-sync 分支没有同样排除 nop，却读取只在前面非 nop 条件块中赋值的局部变量 `device`。独立实验让首节点为 `NopKernelSchedulerNode` 并开启 `triton.debug_sync_kernel`，编译在这里以 `UnboundLocalError: local variable 'device' referenced before assignment` 失败。若 nop 前已有节点，Python 还可能保留上一轮的局部 `device`。因此本篇把“debug sync 调用当前 backend”限定为非 nop 正常路径；nop+debug-sync 是当前实现的失败/疑点出口，而不是一条可解释成安全同步的路径。

### frozen template 的 `node` 重绑定

进入时 `node=op0_op1`，`node.get_nodes()` 返回主模板 op0 与 epilogue op1：

```text
分派前：node = Fused(op0,op1)
执行：node,*epilogue = [op0,op1]
分派：codegen_template(op0,[op1])
分派后：node = op0
```

当前 C++ template backend 的 `codegen_template` 遍历 `[op0,op1]` 并分别调用 `mark_run()`，所以两项输出都由 backend 分配，生成的同一个 template kernel 也包含 epilogue 的 relu。随后外层循环只读取重绑定后的 op0，因此实验状态是 available 新增 buf0、completed 新增 op0。

这两层状态承担不同工作：backend 的 `mark_run()` 和生成代码保证 op1 的结果真实产生；Scheduler 的 available 集合主要在更早的调度构造和 dependency pruning 中使用，此时最终节点顺序已经确定；completed 集合供后续 inplace 判断过滤已处理使用者。没有把 epilogue 加入 completed 会让这项过滤更保守，而不会撤销 backend 已生成的 epilogue。因而外层集合不是“template kernel 实际包含哪些 operation”的清单，它只记录当前局部 `node` 指向的主模板。

runtime-weight mm 则在 Q24 已回退成 extern op0 + 普通 relu op1；本循环不重新选择模板，它依次走 extern、ordinary 两支。

### available/completed 发生在分派之后

普通融合组没有重绑定，`get_buffer_names/get_operation_names` 覆盖所有成员。extern 被 cast 后仍是同一 scheduler node。foreach 也保留组合节点。只有 template 明确换成主节点。两份集合记录 Scheduler 外层看见的逻辑范围，不是设备执行完成事件，也不代替 backend 的 `mark_run()`。

foreach 的短时间线是：处理 device/guard → 登记整个 foreach 的 last usage → backend 类型检查 → `codegen_combo_kernel(foreach)` → 按 foreach 组合对象更新两集合 → 查询 backend ready。nop 则是：登记 last usage → `mark_run()` → 不产生 kernel/wrapper 调用 → 更新两集合 → 跳过 ready 查询。

### K03 理解验收

若一个节点同时是 FusedSchedulerNode 且 `is_template()` 为真，它走第一支；若 template backend 内部成功处理三个 epilogue，Scheduler 两集合仍按主模板节点更新。若 foreach backend 不是两种允许类型，即使设备和依赖都合法也在类型检查处失败，不会退回普通 `codegen_node`。

K03 到此完成。K04 连接 guard、主动 flush、debug sync 和最终收尾。

## K04：device 切换、主动 flush 与最终收尾共享同一状态机

边界块先比较 device。发生切换时，顺序固定为：

```text
flush 旧 backend 与 pending free
    -> 若旧 device 需要 guard，发出 guard exit
    -> 若新 device 需要 guard，检查 index 并发出 guard enter
    -> current_device = 新 device
```

extern/template 即使 device 未变也先 flush，但不会退出再进入同一个 guard。CPU 不需要 guard；CUDA 等需要 guard 的 device 必须已有 index，否则 assert。

非 nop 节点分派后，若 `config.triton.debug_sync_kernel` 为真，当前 backend 收到 `codegen_sync()`。然后更新 available/completed，再查询 `ready_to_flush()`。因此主动 flush 发生在本节点已交给 backend、状态集合已更新之后；它可把刚积累的组立即提交。nop 开启 debug-sync 的当前失败行为已在 K03 单列。

遍历结束时，若当前 device 需要 guard，先发出最外层 exit 请求，再执行 final `flush()`。这是 Scheduler 调用这些生成接口的顺序；wrapper 如何组织缓冲区和缩进由 Q43/Q431 展开。

### 同一五节点案例中的 device 与提交边界

继续使用 K03 的 CPU ordinary A/B、CPU extern C、CUDA foreach D、nop E：

| 节点 | 进入前动作 | 分派与状态 | 进入下项前状态 |
| --- | --- | --- | --- |
| A | `None→CPU`，flush 已缓存但空闲的 backend，无 guard | CPU backend 积累 A | current=CPU；available 加 buffer a，completed 加 operation A |
| B | 无边界 | 同一 backend 积累 B | CPU group=A+B；两集合分别再加 b 与 B |
| C | extern 边界 flush A+B | 直接写 extern，随后 free | CPU group 空；两集合分别再加 c 与 C |
| D | CPU→CUDA，先做一次边界 flush；CPU 无 exit；CUDA enter(index) | CUDA combo backend 接收 D | current=CUDA；两集合分别加 d0/d1 与 D0/D1 |
| E | nop 不切 device、不 flush | 只 `mark_run()` | current 仍为 CUDA；两集合分别加 e 与 E，跳过 ready |
| 结束 | 先请求 CUDA guard exit | final flush CPU/CUDA backend，再 free | pending 集合被兑现 |

若 CUDA backend 在 D 后 `ready_to_flush()` 为真，会先中途 flush D；E 不查询 ready，末尾 final flush 仍执行，并兑现 E 加入的 pending free。

### 失败与退化出口

`_codegen` 不是事务。除 runtime 估算日志这一处主动捕获外，控制循环没有用 `try/finally` 恢复 guard、backend、pending free 或两份完成集合。异常会向上传播；源码没有承诺在同一个 Scheduler 上从失败节点继续或重试。各失败点留下的状态不同：

| 异常位置 | 已经发生 | 尚未发生 / 不会自动恢复 |
| --- | --- | --- |
| `get_estimated_runtime()` | 改写一条 debug 日志为 0.0 | 异常在这里被捕获，正常 codegen 继续 |
| stack 测试检查 | 只完成局部 stack/seen 扫描 | 发现重复 frame 后 assert，节点循环尚未开始 |
| `create_backend(device)` 失败 | `add_device_info(device)` 已可能写入图；若来自节点 device 切换，旧工作可能已 flush，guard exit/enter 和 `current_device` 也可能已逐步改变 | `self.backends[device] = ...` 的右侧没有成功返回，因此该 key 不会插入；没有统一撤销 device info 或 wrapper guard 语句 |
| ordinary/template/foreach backend 分派中失败 | 当前节点的 last usage 已加入 pending；backend 可能已部分修改自己的组、输出或生成状态 | 本节点的 available/completed 尚未更新；backend 内部是否能恢复只能由具体实现回答 |
| extern 分派中失败 | extern counter、inplace 决定、`mark_run`、wrapper 调用或部分 free 可能已按失败位置发生 | 外层 available/completed 只在整个 extern helper 返回后更新；没有回滚 |
| nop + debug sync | pending 已加入，`mark_run()` 已执行 | 首节点 nop 因 `device` 未绑定抛 `UnboundLocalError`；若前面有非 nop，则可能错误复用上一轮 device 去同步旧 backend。sync 抛出时两集合尚未更新 |
| `flush()` 的第 k 个 backend 失败 | 排在前面的 backend 已 flush；第 k 个可能部分提交 | 后面的 backend 未 flush，`free_buffers()` 尚未调用；没有反向撤销已写 wrapper 内容 |
| `free_buffers()` 中途失败 | 前面若干名字可能已经写出 free 请求 | pending 集合在函数末尾才整体 clear，因此失败时仍保留原集合；重试会怎样处理重复请求取决于 wrapper，本文不能假定幂等 |
| `ready_to_flush()` 失败 | 当前节点分派和 available/completed 更新都已完成 | 本轮主动 flush 未发生 |
| 最终 `flush()` 失败 | 需要 guard 的 device 已先写出最外层 exit 请求 | backend 可能部分提交，pending 可能未兑现；没有函数内 cleanup |

测试专用 stack 检查遇到 Dynamo `_compile_inner` 边界会停止反向扫描。节点没有 origin 时 `enter_context` 不调用 wrapper；空 backend 或空 pending 集合本身仍是合法状态。外层编译框架可能整体放弃这次生成产物，但本篇所示 Scheduler 调用链没有把这些局部字段恢复成入口快照。

### K04 理解验收

若普通 GPU 节点 G1 后接同 device template T，再接普通 G2：T 前因 template flush G1，但不切 guard；T 当场生成；G2 积累新组。遍历末尾先请求 guard exit，再 final flush G2。若 T 之后 backend 报 ready，T 分派后的查询还会再做一次可能为空的 flush。

若一个 backend 直到首个 extern 完成后的 ready 查询才创建，首 extern 前的 Scheduler.flush 不会调用它；一旦缓存存在，后续每次 Scheduler.flush 都会遍历它，即使其内部当前没有工作。

K04 到此完成。本文完整展示的七个外层方法是 `flush`、`codegen_extern_call`、`create_backend`、`get_backend`、`enter_context`、`codegen` 和 `_codegen`；`enter_context` 内另有嵌套函数 `get_order`。它们共同形成一条连续控制链：按需缓存有状态 backend，在边界提交，按节点身份分派，更新生成状态，并在安全点兑现 last usage。
