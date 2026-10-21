# Wrapper 与内存计划：把 kernel 串成一次可调用执行

[上一项：CPU C++ 与 GPU/Triton](../02-backends/README.md) · [返回 Codegen](../README.md) · 下一项：缓存、编译与加载

kernel 只知道若干地址参数上的局部计算。生成的 C++ 或 Triton 代码把这种地址参数叫 pointer：它指向一块 tensor 存储的起始位置，kernel 再用索引找到要读写的元素。wrapper 负责把这些局部计算接成一次完整函数调用：接收 tensor 参数，核对 shape、stride 和 dtype，分配中间结果与输出空间，把各块存储的起始地址传给 kernel，按 Scheduler 的顺序调用它们，最后返回输出 tuple。

内存计划会把这些动作放在一条时间线上。例如 buf0 在第 1 行分配，第 2 行被 kernel A 写入，第 4 行最后一次被 kernel B 读取，第 5 行以后就可释放。如果第 6 行需要同设备、同 dtype 且空间足够的 buf1，planner 可以让 buf1 复用 buf0 的同一块内存，而不是再次申请。若多份逻辑 buffer 从一块较大的 allocation pool 取空间，则每份记录自己的 offset，例如 buf0 使用字节 0..255、buf1 使用字节 256..511。

```text
装有所有实参的 Python 列表 args
   │ 取出实参、清空列表、检查 shape/stride/dtype
   ▼
输入 tensor ──→ 分配/复用 buffer ──→ kernel / extern 调用
                       ▲                    │
                       └── last use 后释放 ┘
                                            │
                                            ▼
                                       tuple(outputs)
```

## Scheduler 写入的是“待规划的 wrapper 行”

后端 codegen 进行时，`WrapperCodeGen` 并非立即把所有内容写成最终字符串。分配和释放先保存为带 buffer 对象的行：

- `AllocateLine(buffer)`：此处开始需要一块存储；
- `FreeIfNotReusedLine(buffer)`：此处以后不再使用，可释放，也可留给后续 buffer；
- `ReuseLine(old,new)`：新逻辑名字接管旧存储；
- 普通文本/调用行：形成规划时间线的边界；
- device/subgraph 行：建立嵌套作用域。

等整个 Scheduler codegen 完成，wrapper 才拥有全局顺序，可以判断一次 free 是否应变成 reuse。若在 `codegen_allocation()` 当场分配，就看不到后面即将出现的同规格 buffer。

## 真实案例：两次 mm 中间夹 pointwise

实验计算：

```python
first = x @ weight0
activated = relu(first)
second = activated @ weight1
return sin(second)
```

当前 CPU 实际 wrapper 主线为：

```python
buf0 = empty_strided_cpu((8, 8), (8, 1), torch.float32)
extern_kernels.mm(arg1_1, arg0_1, out=buf0)
del arg0_1
del arg1_1

buf1 = buf0; del buf0  # reuse
cpp_fused_relu_0(buf1)

buf2 = empty_strided_cpu((8, 8), (8, 1), torch.float32)
extern_kernels.mm(buf1, arg2_1, out=buf2)
del arg2_1
del buf1

buf3 = buf2; del buf2  # reuse
cpp_fused_sin_1(buf3)
return (buf3,)
```

按对象状态逐步看：

| 时刻 | 活跃存储 | 为什么还活着 |
| --- | --- | --- |
| 第一 mm 后 | 输入、`buf0` | `buf0` 将供 relu/第二 mm 使用 |
| relu 前后 | `buf1` 与 `buf0` 是同一存储 | relu 可原地覆盖；只换逻辑名 |
| 第二 mm 时 | `buf1`、新 `buf2` | mm 输入输出不能占同一块存储 |
| 第二 mm 后 | `buf2` | `buf1` 最后一次使用结束，可删引用 |
| sin 后 | `buf3` 与 `buf2` 是同一存储 | sin 原地覆盖，且该存储成为图输出 |

所以这里只为两个 mm 结果真正申请两块 8×8 存储，relu 与 sin 没有各自申请完整输出。`buf1 = buf0; del buf0` 不是复制 64 个元素，而是 Python 名字接管同一 tensor/storage；对应 C++ kernel 参数也是 `float* in_out_ptr0`。

实验输出与 eager 相同。这个案例证明当前图的复用决定，不能推广成“所有 pointwise 都可原地”：旧值若仍有后续用户、输出存在 alias 约束、dtype/layout/容量不兼容或 buffer 被列入 `never_reuse_buffers`，都不能这样复用。

## `last_usage` 怎样变成 `del`

Q32 中 Scheduler 已为每个节点计算最后使用集合。节点 codegen 后，Scheduler 让 wrapper 对这些 buffer 执行 `codegen_free()`：

- 图输入不能作为新中间 buffer 的可复用存储，但 Python 局部引用可在最后一次使用后 `del`；
- 常量、图输出、已经释放、禁止复用或特殊 alias buffer 不进入普通复用池；
- 普通中间 buffer 生成 `FreeIfNotReusedLine`，规划阶段才决定输出 `del name` 还是让后续 `AllocateLine` 变成 `ReuseLine`。

`del` 的直接含义是放掉 wrapper 局部引用，不等于此刻强制把设备内存归还操作系统。allocator 可能继续缓存底层块；本节讨论的是生成图内部的逻辑生存期和存储复用。

## 简单复用计划：按兼容 key 接住最近释放块

普通 `memory_plan_reuse()` 顺序扫描 wrapper 行。遇到可复用 free 时，把它按 `buffer_reuse_key` 压入 planning state；后续 allocate 若命中兼容 key，就将 free 标为已复用，并把 allocate 换成 `ReuseLine`。

完全相同 shape/stride 时生成：

```python
new = old; del old  # reuse
```

当前 `buffer_reuse_key` 的内容可以明确写成：

```text
(device, dtype, simplify(layout.storage_size()))
```

它比较的是符号存储容量表达式，而不是 shape/stride 本身。shape/stride 完全相同时生成直接别名；key 相同但 shape/stride 不同时，可用 `reinterpret_tensor` 在同一 storage 上建立新 view。`can_reuse()` 另外排除图输入、常量、已经释放、`never_reuse_buffers` 等特殊身份。因此真正受保护的是 device、dtype、存储容量和身份约束；不同布局本身并不必然禁止复用，也不能只凭两个 tensor 恰有相同元素数就自行判定可复用。

## inference 的 pool memory planning

开启 `config.memory_planning` 且当前生成是 inference 时，wrapper 改用 `MemoryPlanner`：

1. 丢掉已被优化删除的 buffer 行；
2. 把共享存储的 allocate/reuse 名字归入同一个 `BufferGroup`；
3. 沿完整 wrapper 时间线计算每组 first/last usage；图输出的生存期延长到调用末尾；
4. 按 device 建立 pool；`TemporalSplit` 让生存期不重叠的 allocation 使用同一空间区域，`SpatialSplit` 为生存期重叠或大小不同的对象安排不同 offset；
5. 把每个 tensor 表达成 pool 基址、offset、shape、stride 的 view，仅在 pool 首/末次使用处创建/销毁 pool。

其关键判据是生存期是否重叠，而不只是源码顺序相邻。训练路径目前不会使用这套完整 pool planner，因为源码明确注明它当前会增加训练峰值内存；训练或未开启该配置时使用前述 reuse planner。

同一条两次 mm pipeline 在 `memory_planning=True` 下实际生成：

```python
pool1 = empty_strided_cpu((8, 8), (8, 1), torch.float32)
buf0 = pool1  # alloc
extern_kernels.mm(..., out=buf0)
buf1 = buf0;  # reuse
cpp_fused_relu_0(buf1)

pool0 = empty_strided_cpu((8, 8), (8, 1), torch.float32)
buf2 = pool0  # alloc
extern_kernels.mm(buf1, ..., out=buf2)
del pool1, buf0, buf1
buf3 = buf2;  # reuse
cpp_fused_sin_1(buf3)
return (buf3,)
```

`pool1` 对应中间 storage group `buf0/buf1`，最后使用发生在第二次 mm，所以随后整体销毁；`pool0` 对应输出 group `buf2/buf3`，生存期延长到返回。本例每个 pool 只有一个 8×8 storage，所有 offset 都是 0；它实际证明了分组、live range 和 pool 首末使用，没有证明复杂空间切分已经在本例发生。

## 输入检查、符号尺寸与输出协议

实验入口先执行：

```python
arg0_1, arg1_1, arg2_1 = args
args.clear()
assert_size_stride(arg0_1, (8,8), (8,1))
...
```

`args.clear()` 让 boxed 参数容器不再额外持有 tensor；局部变量仍保持所需引用。静态版本用 `assert_size_stride` 检查编译所依赖的 shape/stride。动态版本会把运行时尺寸绑定到符号：例如编译时把第一维记为 `s0`，本次输入 shape 为 `[19]`，入口便令 `s0=19`。若所选 kernel 要求长度能被 16 整除，就生成 `s0 % 16 == 0` 之类的检查；19 不满足时不能调用该版本，而要选择或编译带尾部处理的版本。动态形状不是简单跳过检查。

返回始终按图接口组装 tuple。若输出是输入 alias、view 或 mutated input，`get_output_refs()` 会返回对应引用/view，而不是无条件分配新 tensor。内存计划必须保留这种可观察语义。

### 修改输入时，返回合同包含“新结果”和“原 storage”两条线

沿用 Q23/Q32 的实际请求：

```python
old = x + 1
x.copy_(y)
return old, x
```

输入为 `x=[0,1,2]`、`y=[9,9,9]`。当前 CPU 实验生成的调用核心是：

```python
buf0 = empty_strided_cpu((3,), (1,), torch.float32)
cpp_fused_add_0(arg0_1, arg1_1, buf0, arg0_1)
del arg0_1
del arg1_1
return (buf0,)
```

这里的四个指针对应两项工作：从旧 `arg0_1` 读取 `x`，把 `x+1` 写入新 storage `buf0`；同时从 `arg1_1` 读取 `y`，把它写回 `arg0_1` 指向的调用者 storage。执行后的状态是：

| storage | 调用前 | kernel 写入后 | 谁继续观察 |
| --- | --- | --- | --- |
| 调用者的 x storage | `[0,1,2]` | `[9,9,9]` | 调用者原来的 `x` 对象 |
| 新建的 buf0 storage | 尚不存在 | `[1,2,3]` | 显式函数结果 `old` |

生成模块的 `return (buf0,)` 只列出新增的显式结果。编译调用协议还知道 `arg0_1` 是 mutated input；外层把调用者原来的 `x` 对象作为第二项语义结果接回，因此实验中 `returned_x.data_ptr()==x.data_ptr()`。wrapper 中 `del arg0_1` 只是删除局部 Python 名，不会撤销调用者持有的对象，也不会把这块输入 storage 放进普通中间复用池。

若函数只执行 `x.copy_(y)`、另返回 `y+1`，调用者仍能观察 x 被改成 y，但新增返回值使用独立 storage。可见“图输出”不能只靠生成模块最后那行 tuple 判断：新增返回 buffer、mutated input 和 alias/view 都由图的输出/修改合同共同组装。

## device guard 与 stream 位于 wrapper 层

Scheduler 的 `_codegen()` 观察当前节点 device；设备变化时先结束旧 guard，再调用 `codegen_device_guard_enter(device.index)`。Python wrapper 对 GPU 生成大意如下的作用域：

```python
with device_ops.device_guard(device_index):
    device_ops.set_device(device_index)
    # 获取当前 raw stream，并按顺序调用 Triton/extern kernels
```

Triton 调用由 wrapper 写成 `kernel.run(..., grid=..., stream=streamN)`。这里的 stream 是设备按提交顺序执行工作的队列；`streamN` 是 wrapper 保存的局部变量，里面装着运行时取得的底层设备 stream 句柄，源码有时称它为 raw stream。相邻 kernel 使用同一个 `streamN` 时，提交顺序也规定了它们的执行先后。guard 确保“当前设备”与取得的 stream、分配和 kernel launch 一致；CPU-only 图不需要这段。

C++ wrapper 和 AOT 模式会使用相应的 CUDA guard/stream guard 类型，语法不同但职责相同。多 device 图在 Scheduler 顺序中切换 guard，不能让单个 kernel 自行猜测当前 device。

## 从一段 wrapper 反推执行计划

阅读生成代码时可按下面顺序：

1. 从 guards 确认该编译版本接受的输入合同；
2. 给每次 allocation/reuse 标出 storage 身份，而不只看 `bufN` 名字；
3. 沿 kernel 调用记录每块存储的最后用户；
4. 检查 `del` 是否恰在最后用户之后、图输出是否一直存活；
5. GPU 情况再检查 device guard、grid 和 stream 是否包住对应调用。

对本例而言，逻辑 buffer 名是 `buf0..buf3`，物理上只有两次新分配：`buf0/buf1` 一组、`buf2/buf3` 一组。这正是 wrapper 内存计划要表达的区别。

换到修改输入案例，物理身份则是“输入 x 的原 storage + 新建 old 的 storage”；即使 wrapper 局部名被 `del`，调用者持有的 x 仍须保活并呈现写回值。先标 storage 身份，再读名字和 `del`，两类案例便能用同一方法解释。

## 继续阅读实现

[Wrapper 复用规划源码：LIFO、live range 与 allocation pool](01-planner-source/README.md)
