# backend 分派循环源码：最终节点怎样交给生成后端

[返回后端分派篇](../README.md) · [返回 Scheduler](../../README.md) · [返回总览](../../../README.md)

Q34 已经让我们按节点类型画出提交时间线。本篇打开实际控制循环：它维护哪几份状态，为什么在 extern/template/device 边界先 flush，以及分派后哪个 `node` 被加入 available/completed。

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

## K01：backend 延迟创建，首节点前的 flush 可以是空操作

### flush 先提交所有已创建 backend，再尝试释放

`BaseScheduling` 是各设备代码生成器共同实现的接口；`self.backends` 把具体 `torch.device` 映射到这类对象。本例刚开始时字典为空，首次空 flush 不调用任何 backend；创建 CPU 生成器并暂存节点后，字典为 `{cpu: CppScheduling}`，下一次 flush 会调用它的 `flush()`，随后 `free_buffers()` 把已经到最后使用的数据转成 wrapper 释放请求。

```python
def flush(self) -> None:
    # 教学：一个 Scheduler 可缓存多个 device backend；逐一提交它们的待生成工作。
    for backend in self.backends.values():
        backend.flush()
    # 教学：kernel 调用已经写出后，才把 pending last-usage 转成 wrapper free 请求。
    self.free_buffers()
```

`self.backends` 只含已经按需创建的实例。第一个节点前调用 `flush()` 时它可以仍为空；循环不执行，但 `free_buffers()` 仍安全运行。flush 不是“必然生成一个 kernel”，而是“让当前所有 backend 兑现已积累工作，并在其后兑现可释放项”。

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

实验 11 的最终列表只有 `op0_op1`，初始状态可写成：

```text
backends={}
current_device=None
buffer_names_to_free={}
```

访问 CPU 节点时 `cpu != None`，主循环先调用 `flush`。此时没有 backend，因而没有 C++ kernel；随后设置 `current_device=cpu`。普通分派第一次调用 `get_backend(cpu)`，才创建并缓存 `CppScheduling`，再把 op0/op1 累积进它的 kernel group。遍历结束的 final flush 看到这个实例，提交 `[op0,op1]` 并兑现 pending free。

这解释了实验顺序：

```text
scheduler.flush                    # 空 backend 集合
create_backend(cpu)=CppScheduling
cpp.codegen_node(op0_op1)          # 只积累
scheduler.flush
cpp.flush([op0,op1])               # 真正提交
```

### K01 理解验收

若首节点是 extern，前置 `flush()` 时 `backends` 仍可能为空；extern 路径本身不需要先创建普通 CPU backend。只有节点完成后的 `ready_to_flush()` 查询调用 `get_backend(cpu)` 时，才可能创建它。若同一 Scheduler 后来出现普通 CPU 节点，会复用这个实例。

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

下面按源码顺序阅读 `_codegen`。先固定节点序列：CPU 普通 A、CPU 普通 B、CPU extern C、CUDA foreach D、nop E；`current_device=None`，backend 缓存为空，available/completed 与待释放集合都为空，stack 检查、debug sync、freezing 均关闭。A 首先创建 CPU backend，A/B 在同组积累；C 前 flush 后走 extern；D 触发设备切换、进入 CUDA guard 并走 foreach；E 不切设备，只更新逻辑完成状态；循环结束退出 guard 并最终 flush。开头的测试开关只在测试配置下检查调用栈没有重复 frame。

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

### 分派顺序决定同一个对象走哪条路

判断依次是 template、extern、foreach、普通 Scheduler/Fused、nop。template 放在最前，是因为带 epilogue 的 `FusedSchedulerNode` 同时也是融合组；若先按普通 fused 判断，它会错误进入 `codegen_node`。extern 直接调用上节方法；foreach 还要检查 backend 是 `SIMDScheduling` 或 `CUDACombinedScheduling`，否则抛 AssertionError；普通节点进入 backend `codegen_node`；剩余对象必须是 nop，只 `mark_run()`。

nop 在 device 边界块前被排除，所以不会切 guard，也不查询 `ready_to_flush()`；它仍更新待释放名字以及 available/completed。Triton debug sync 是调试生成路径，普通配置关闭；理解正常分派时无需让这个调试选项替代 nop 的实际职责。

### frozen template 的 `node` 重绑定

进入时 `node=op0_op1`，`node.get_nodes()` 返回主模板 op0 与 epilogue op1：

```text
分派前：node = Fused(op0,op1)
执行：node,*epilogue = [op0,op1]
分派：codegen_template(op0,[op1])
分派后：node = op0
```

具体 C++ template backend 在内部对 op0、op1 都 `mark_run()`，生成内容含 relu；但当前循环随后的两行只读取重绑定后的 op0，所以实验状态是 available 新增 buf0、completed 新增 op0，而不是整个融合组。理解这一步必须追踪 Python 局部变量，不能只看“模板含 epilogue”这一概念。

runtime-weight mm 则在 Q24 已回退成 extern op0 + 普通 relu op1；本循环不重新选择模板，它依次走 extern、ordinary 两支。

### available/completed 发生在分派之后

普通融合组没有重绑定，`get_buffer_names/get_operation_names` 覆盖所有成员。extern 被 cast 后仍是同一 scheduler node。foreach 也保留组合节点。只有 template 明确换成主节点。两份集合表示生成阶段已经可以引用哪些逻辑结果、哪些 operation 已处理；不是设备执行完成事件。

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

每次分派后，若 `config.triton.debug_sync_kernel` 为真，当前 backend 收到 `codegen_sync()`。然后更新 available/completed，再查询 `ready_to_flush()`。因此主动 flush 发生在本节点已交给 backend、状态集合已更新之后；它可把刚积累的组立即提交。

遍历结束时，若当前 device 需要 guard，先发出最外层 exit 请求，再执行 final `flush()`。这是 Scheduler 调用这些生成接口的顺序；wrapper 如何组织缓冲区和缩进由 Q43/Q431 展开。

### CPU 普通 A、extern C、CUDA E 的完整推演

假定节点序列为 CPU ordinary A、CPU ordinary B、CPU extern C、CPU ordinary D、CUDA ordinary E：

| 节点 | 进入前动作 | 分派与状态 | 进入下项前状态 |
| --- | --- | --- | --- |
| A | `None→CPU`，空 flush，无 guard | CPU backend 首次创建，积累 A | current=CPU，available/completed 加 A |
| B | 无边界 | 同一 backend 积累 B | CPU group=A+B |
| C | extern 边界 flush A+B | 直接写 extern，自己 free | CPU group 空，状态加 C |
| D | 无 device 切换 | 新 CPU group 积累 D | pending free 等待安全提交 |
| E | CPU→CUDA，先 flush D；CPU 无 exit；CUDA enter(index) | CUDA backend 积累 E | current=CUDA |
| 结束 | 先请求 CUDA guard exit | final flush CPU/CUDA backend，再 free | pending 集合被兑现 |

若 CUDA backend 在 E 后 `ready_to_flush()` 为真，会先中途 flush E；末尾 final flush 仍执行，但可以是空操作。

### 失败与退化出口

- `get_estimated_runtime()` 仅用于 debug log；异常被捕获后记 0.0，不阻断 codegen。
- 测试专用 stack 检查发现相同 `(filename,lineno)` 重复 frame 时 assert；遇到 Dynamo `_compile_inner` 边界停止反向扫描。
- 不支持 device、缺 Triton、旧 CUDA、foreach backend 类型错误、需要 guard 却无 index 都明确失败，不静默换路径。
- 节点没有 origin 时 `enter_context` 不调用 wrapper；空 backend/pending 集合的 flush 合法。

### K04 理解验收

若普通 GPU 节点 G1 后接同 device template T，再接普通 G2：T 前因 template flush G1，但不切 guard；T 当场生成；G2 积累新组。遍历末尾先请求 guard exit，再 final flush G2。若 T 之后 backend 报 ready，T 分派后的查询还会再做一次可能为空的 flush。

若一个 backend 直到首个 extern 完成后的 ready 查询才创建，首 extern 前的 Scheduler.flush 不会调用它；一旦缓存存在，后续每次 Scheduler.flush 都会遍历它，即使其内部当前没有工作。

K04 到此完成。七个方法共同形成一条连续控制链：延迟创建有状态 backend，在边界提交，按节点身份分派，更新生成状态，并在安全点兑现 last usage。
