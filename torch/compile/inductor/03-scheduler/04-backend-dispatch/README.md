# 后端分派：Scheduler 按边界提交生成工作

[上一项：融合](../03-fusion/README.md) · [返回 Scheduler](../README.md) · [返回总览](../../README.md) · 下一项：Codegen 中的虚拟化操作

融合完成后，Scheduler 已有一列最终节点。最后一项职责不是亲自写 C++ 或 Triton，而是维护当前 device 与 backend 状态，遇到边界时提交已积累的工作，再调用对应的生成函数。例如 CPU 普通节点交给 `CppScheduling.codegen_node(node)`；已有函数调用则先让 CPU 生成器写完当前循环，再由 wrapper 记录该调用。

```text
最终 scheduler nodes
       │
       ├─ 选择/复用 device backend
       ├─ device、extern、template 边界前 flush
       ├─ 按节点类型调用对应的 codegen 函数并传入当前节点
       ├─ 更新 available buffers 与 completed operations
       └─ backend 主动请求或遍历结束时 flush
       │
       ▼
kernel 定义 + wrapper 中的分配、调用、释放和 device guard
```

本篇讲清 Scheduler 怎样递交工作；backend 内部怎样把循环翻译成具体指令，从 Q41 开始展开。

## device type 先映射到一组 backend 能力

`codegen.common` 维护 `device_codegens` 注册表，每个 device type 对应：

- 一个 Scheduling 实现，负责 kernel 级生成、融合能力和 flush；
- 一个 Python wrapper 生成类；
- 可选的 C++ wrapper 生成类。

当前内置注册的主路径是：

| device type | Scheduling 选择 | wrapper |
| --- | --- | --- |
| CPU | 配置为 `cpp` 时由 `CppScheduling` 生成 C++ 循环；若用户启用 Halide 后端，则改由 Halide 的调度器生成对应代码 | `WrapperCodeGen` / `CppWrapperCpu` |
| CUDA | `CUDACombinedScheduling` 在 Triton、模板等 CUDA 路径间分派；启用 Halide 时也可改走 Halide 调度器 | `WrapperCodeGen` / `CppWrapperCuda` |
| XPU | `TritonScheduling` | `WrapperCodeGen` |
| 已注册 PrivateUse1 | PyTorch 为第三方设备预留的设备类型；扩展必须注册自己的 Scheduling 与 wrapper 生成类 | 由扩展提供 |

`Scheduler.get_backend(device)` 第一次请求某个规范化 device 时调用 `create_backend()`，随后缓存在 `self.backends[device]`。这里的 key 是完整 `torch.device`，不是只有字符串 `"cpu"` 或 `"cuda"`。创建时还把 device 信息登记到图中；不支持的 device type，以及源码列出的无可用 Triton GPU 情况会在这里失败，而不是静默改用 CPU。

CPU pointwise 实验实际记录：

```text
create_backend device=cpu type=CppScheduling
```

同一份 Scheduler 后续反复请求 CPU backend 时复用该对象，因为其中保存着尚未 flush 的 kernel group。

## 为什么第一个节点之前也会 flush

`_codegen()` 遍历每个最终节点。对于有 device 的非 nop 节点，只要满足任一条件就先 `flush()`：

```text
节点 device != current_device
或节点是 extern
或节点是 template
```

初始 `current_device=None`，所以第一个 CPU 节点也满足 device 改变。此时 flush 的组为空，不会生成空 kernel；它先清理所有 backend 的待提交状态和可释放 buffer，再把 `current_device` 设为 CPU。

pointwise 案例是 Q33 的两成员融合节点：

```text
visit op0_op1, current_device=None
scheduler.flush
  cpp.flush scheduled_nodes=[]
dispatch cpp.codegen_node(op0_op1)
遍历结束 scheduler.flush
  cpp.flush scheduled_nodes=[op0, op1]
```

第一次 flush 建立干净的 device 边界；`codegen_node` 把两个成员加入 C++ `kernel_group`；末尾 flush 才把该组变成 kernel 定义和 wrapper 调用。

## 普通节点、extern、template、foreach 分别去哪里

当前分派表是：

| 最终节点 | Scheduler 调用 | 交付含义 |
| --- | --- | --- |
| `SchedulerNode` / `FusedSchedulerNode` | `backend.codegen_node(node)` | 生成一个普通或预融合循环组 |
| template 节点或带 epilogue 的融合组 | `backend.codegen_template(template, epilogues)` | 第一个成员是模板主体，其余成员作为 epilogue |
| `ExternKernelSchedulerNode` | `Scheduler.codegen_extern_call(node)` | 让 IR extern 直接向 wrapper 写调用 |
| foreach/combo 节点 | `SIMDScheduling` 或 `CUDACombinedScheduling.codegen_combo_kernel(node)` | 多个子 kernel 由组合生成器安排 |
| `NopKernelSchedulerNode` | `node.mark_run()` | 更新分配/运行状态，不生成普通 kernel |

分派顺序很重要：template 可能也是融合节点，必须先命中 template；extern 不走普通 backend loop codegen；foreach 只接受具备 SIMD/combo 接口的后端。

## extern 为什么切断正在积累的 kernel group

混合案例：

```python
pointwise_result = realize(x + 1)
mm_result = torch.mm(matrix, weight)
return pointwise_result, torch.sin(mm_result)
```

当前 CPU 最终顺序是：

```text
op0_op1  FusedSchedulerNode       # pointwise 分支
op2      ExternKernelSchedulerNode # mm
op3      SchedulerNode             # sin
```

实际调用时间线：

```text
op0_op1 -> cpp.codegen_node，暂存在 kernel_group

到 op2 extern：
    scheduler.flush
      cpp.flush [op0, op1]       # 先提交此前普通循环
    codegen_extern_call(op2)     # wrapper 写 mm 调用

到 op3：
    cpp.codegen_node(op3)，进入新的 kernel_group

遍历结束：
    cpp.flush [op3]              # 提交 sin kernel
```

这保证 wrapper 的实际调用顺序与依赖一致：先得到 pointwise 输出；调用 mm 得到 `buf2`；随后 sin kernel 才读取它。即便 extern 与前一节点 device 都是 CPU，extern 条件仍强制 flush，因为它不能被留在同一个普通 C++ loop group 中。

`codegen_extern_call()` 会在一个临时 `Kernel` handler 下决定原地复用并标记输出分配，然后让 `ir.ExternKernel.codegen()` 写 wrapper 调用，最后处理可释放 buffer。

fallback 也属于 extern 分派。`histc` 实验实际依次访问 `FallbackKernel` 和取结果的 `MultiOutput`，两者都是 `ExternKernelSchedulerNode`，因此每个节点前都有 extern flush；数值结果为 `[1,2,0,1]`。

## 同一个 mm 请求可以交成 extern，也可以交成 template

实验开启 CPU max autotune 并只保留 CPP GEMM backend，但把 mm 权重作为运行时输入。当前 `use_cpp_packed_gemm_template()` 要求第二个矩阵是 module buffer，本例没有合法 CPP choice，于是 Lowering 发出警告并回退到 ATen extern：

```text
op0 ExternKernelSchedulerNode(aten.mm)
op1 SchedulerNode(relu)
```

Scheduler 随后只能按 extern→普通节点分派，不会在 Q34 阶段把它重新变成 template。这说明算法候选选择属于 Q24，Q34 消费的是已经确定的节点身份。

为了实际进入模板分支，第二个案例把 64×64 单位矩阵登记成模块 buffer，并在 inference/no-grad 下开启 freezing。此时权重成为 `constant0`，最终调度节点是包含两个成员的 `op0_op1`：

```text
op0：CppTemplateBuffer 完成 mm
op1：ComputedBuffer 完成 relu epilogue
```

实际分派记录为：

```text
visit op0_op1 type=FusedSchedulerNode
先做一次空 cpp.flush
dispatch cpp.codegen_template op0 epilogues=['op1']
```

`CppScheduling.codegen_template()` 接收主模板与 epilogue 列表，渲染模板源码，在同一 kernel context 中对二者执行 `mark_run()`，定义 kernel 并写 wrapper 调用，随后兑现可释放 buffer。这里与普通 `codegen_node()` 的另一个区别是：模板调用当场完成定义和调用，不把成员留到 `CppScheduling.kernel_group` 等下一次 flush；所以末尾的 `cpp.flush scheduled_nodes=[]` 是空的。

## 每个节点交付后更新两份状态

普通分支分派完成后，Scheduler 执行：

```text
available_buffer_names += node.get_buffer_names()
completed_operations   += node.get_operation_names()
```

混合案例在进入 extern op2 前已经是：

```text
available 包含 buf0、buf1
completed = [op0, op1]
```

op2 完成后，进入 op3 时：

```text
available 又包含 buf2
completed = [op0, op1, op2]
```

op3 分派后再加入 `buf3` 和 op3。普通融合节点的 `get_buffer_names()` 和 `get_operation_names()` 包含所有成员，因此一次加入整个普通融合组。这些集合让后端和 wrapper 知道哪些结果已经可以引用，也参与原地复用、断言与调试状态；它们不是运行时事件计数器。

template 分支有一个必须单独看的当前实现差异。`_codegen()` 先执行：

```python
node, *epilogue = node.get_nodes()
backend.codegen_template(node, epilogue)
```

局部变量 `node` 从原来的融合组重新绑定为 template 主节点。后面的两个集合也就按这个主节点更新。冻结模板案例在调用结束后实际是：

```text
available：原有 arg1_1、constant0，再加入 buf0
completed：加入 op0
```

epilogue `op1` 并没有因此漏算：`CppScheduling.codegen_template()` 已在内部对 `[op0, op1]` 都调用 `mark_run()`，而模板生成的最终代码包含 relu。只是不能把“成员都已生成”误写成“Scheduler 的 available/completed 集合也加入全部模板成员”。

进入节点之前，`enter_context()` 还会从成员的 FX origins 中选图顺序最靠后的 origin，交给 wrapper 建立源码/调试上下文。这样融合节点生成的 wrapper 行仍能追溯到用户图位置。

## `last_usage` 怎样在提交边界兑现

访问一个节点时，Scheduler 先把该节点的 `last_usage` 加入 `buffer_names_to_free`。普通 C++ `codegen_node` 可能只是积累循环，不能立刻释放；`Scheduler.flush()` 会：

1. 调用所有已创建 backend 的 `flush()`，把积累的 kernel 和调用写完；
2. 调用 `free_buffers()`，为真正可释放的 buffer 生成释放动作；
3. 清空 pending-free 集合。

混合案例中，pointwise 节点访问后 pending-free 包含其中间结果与输入；到 extern 前 flush 完 C++ 组后集合清空。extern 自己又处理其输入最后使用。op3 的 `buf2` 最后使用则在末尾 flush 后清空。

因此 Q32 的 `last_usage` 到这里才转成 wrapper 释放安排，并且释放不会越过仍未提交的 kernel。

## backend 也可以主动要求 flush

节点状态更新后，Scheduler 查询 `backend.ready_to_flush()`。CPU `CppScheduling` 在累计参数数超过保守上限时设置该标志；Scheduler 随即 flush，避免继续扩大一个 C++ kernel group。SIMD/Triton 后端也可按自己的资源和生成策略实现该接口。

不论中途是否触发，遍历结束总会执行最后一次 flush，空组安全地不产生 kernel。

## device guard 只在需要的 device 上生成

device 改变时，Scheduler 先退出旧 device guard，再进入新 device guard。`device_need_guard(device.type)` 决定该类型是否需要 guard；CPU 实验没有生成 guard。对于需要 guard 的 CUDA 等设备，wrapper 入口形如：

```text
退出旧 device 上下文（若有）
进入新 device index 的上下文
在其中取得/使用对应 stream 并发出调用
```

遍历结束时，Scheduler 先发出“退出当前 device guard”的请求，再要求各 backend 把尚未输出的工作 flush 到 wrapper。这里描述的是生成器接收请求的顺序；最终生成代码怎样把缓冲内容放进 guard 作用域，由 wrapper 生成器负责。阅读这一层时只需记住：设备切换前会 flush 旧设备工作，遍历结束也一定会 flush，因而没有节点会留在 backend 的暂存组里。

## 用新节点序列推算

假设最终顺序是：CPU 普通 A、CPU 普通 B、CPU extern C、CPU 普通 D、CUDA 普通 E。

可以推出：

1. A 前因 `None→CPU` 做一次空 flush；A、B 可在同一 CPU backend group 中积累。
2. C 前即使 device 不变也 flush A、B；C 直接写 extern 调用。
3. D 开始新的 CPU group。
4. E 前因 `CPU→CUDA` flush D，再进入 CUDA device guard，交给 CUDA backend。
5. E 完成后退出 guard，最后 flush CUDA backend 与待释放 buffer。

能按这条规则画出调用顺序，就已经掌握 Scheduler 怎样把全局计划交到不同 codegen 后端。

## 继续阅读实现

[Backend 分派循环源码：设备切换、flush 与节点出口](01-dispatch-source/README.md)
