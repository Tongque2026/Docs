# 后端分派：按最终节点顺序填充 kernel 与 wrapper

[上一篇：融合判断与代价](../03-fusion/README.md) · [返回 Scheduler](../README.md) · [下一篇：Codegen](../../04-codegen/README.md) · [返回总览](../../README.md)

实现叶子：[设备切换、flush 与节点出口](01-dispatch-source/README.md)

Q33 融合结束后，父级 Scheduler 还会合并循环、固定 MultiTemplate 的实现、重排通信并建立
combo；临时的 `GroupedSchedulerNode` 会在这里拆回它包装的节点。Q32 随后为最终节点计算
`last_usage`。本篇接收的就是这列最终节点，因此 `_codegen()` 不会再遇到 Grouped 节点。

这一步仍发生在编译期间。它沿节点列生成 kernel 定义和调用、已有函数调用、设备 guard、
空间安排与释放语句；它没有让 CPU 或 GPU 立即执行这些计算。

本篇包含四项依次配合的工作：

```text
最终 scheduler nodes
  │
  ├─ 1. 按节点的完整 device 取得或创建 backend
  ├─ 2. 识别提交边界，并维护三份生成状态
  ├─ 3. 按节点身份调用普通、extern、template、combo 或 nop 入口
  └─ 4. flush 剩余工作，把 kernel/wrapper 缓冲区交回 GraphLowering
```

## 第一项：从 device 找到可复用的 backend 实例

全局的 device codegen 注册表按 device type 保存生成器类别。例如 CPU 对应
`CppScheduling`；当前默认 CUDA 路径对应 `CUDACombinedScheduling`，它再把普通 Triton
节点交给内部的 SIMD/Triton scheduling。Scheduler 第一次请求某个完整
device，例如 `cpu` 或 `cuda:1` 时，先用 `device.type` 找到生成器类别，再把创建的实例
缓存到 `self.backends[device]`。后续同一完整 device 的节点复用同一个实例；另一张 GPU
或另一种 device 使用另一个实例。

```text
node.device = cuda:1
        │
        ├─ device.type = "cuda" ──> 注册表中的 scheduling 类
        └─ 完整 device cuda:1 ───> Scheduler.backends[cuda:1] 实例
```

普通 scheduler node 在建立循环 `group` 时就可能请求 backend，因此分派开始前实例可能
已经存在。也有相反情况：若第一项工作是普通 extern，它不需要普通循环 backend；实例可能
到节点分派后的 `ready_to_flush()` 查询才按需建立。

创建 backend 时会检查 device：

- GPU device 必须带 index；
- device type 必须已经注册生成器；
- GPU 路径要求可用且版本合适的 Triton；过旧 CUDA 设备也会被拒绝。

这些条件失败会终止编译，不会在本层自动换成另一个 backend。

backend 与 wrapper generator 保存的内容不同。backend 接收普通循环或模板，形成 kernel
定义和调用；wrapper 按运行顺序记录空间分配或复用、kernel 调用、ATen 等已有函数调用、
device guard 和释放语句。

CPU 的 `CppScheduling` 可以把连续普通节点暂存在当前 kernel group 中，等到
`flush()` 才提交。当前 CUDA combined backend 把普通 Triton 节点委派给内部 SIMD 路径；
后者在一次 `codegen_node()` 内完成 kernel 定义和调用，不保留一个等待 Scheduler flush
的普通组。

## 第二项：边界与三份状态共同决定生成顺序

Scheduler 每访问一个节点，都维护下面三份编译期状态：

| 状态 | 初始内容 | 分派时怎样变化 | 直接消费者 |
| --- | --- | --- | --- |
| `available_buffer_names` | 函数输入、常量和 TorchBind 常量 | 节点分派后加入它的结果名，只增不删 | 建立调度节点时曾用于裁掉入口已有依赖；当前分派只继续维护这份历史记录，后续节点不会拿它重新授权读取 |
| `completed_operations` | 空 | 节点分派后加入它的 operation 名，只增不删 | 后续原地复用判断用它排除已经处理过的 user |
| `buffer_names_to_free` | 空 | 分派节点前加入该节点的 `last_usage`；清理点检查后清空 | `free_buffers()` 判断哪些逻辑名能写成释放语句 |

`buffer_names_to_free` 只是释放候选。图输出、已处理名字、没有独立存储的逻辑名以及仍由
复用关系持有的存储都不会产生释放语句；不论是否真的写出释放，检查结束后候选集合都会
清空。

对 CPU C++ 普通节点，一次访问的顺序是：

```text
1. 当前节点要求边界时，先 flush 旧工作
2. device 改变时，退出旧 guard、进入新 guard
3. pending 加入当前节点的 last_usage
4. backend.codegen_node(node)
   └─ 决定原地复用，mark_run 安排输出空间，把循环留在当前 C++ group
5. available 加入节点结果名；completed 加入 operation 名
6. backend.ready_to_flush() 为真时，立即 flush
```

`mark_run()` 逐个处理节点输出，让 wrapper 记录分配或复用；它不表示 operation 已执行。
`completed_operations` 则是 Scheduler 的“已经越过分派位置”记录。两者不是同一份状态。

Scheduler 的 `flush()` 会遍历所有已经创建的 backend，让各自提交剩余工作，然后统一调用
`free_buffers()` 清空 pending。以下情况触发它：

- 当前节点的 device 与上一节点不同；
- 当前节点是 extern 或 template；
- 普通节点分派后，backend 报告当前组已满；
- 所有节点遍历结束。

第一次访问 CPU 普通节点时，`current_device=None`，所以先执行一次 `None→CPU` 的
flush。此时 CPU backend 可能已经存在，但它的 kernel group 为空，pending 也为空，因此
不会写出调用。

## 主案例：pointwise → mm → sin

实验计算：

```python
pointwise = test_operators.realize(matrix + 1)
product = torch.mm(pointwise, weight)
output = torch.sin(product)
```

`test_operators.realize` 是实验使用的测试算子：它读取传入的值，再写出一个具名结果，用来
强制观察这次中间结果交接。因此 `matrix+1` 形成 op0 和 buf0，realize 再读取 buf0、形成
op1 和 buf1。当前 CPU 路径的最终节点是：

```text
op0_op1：普通融合节点
  op0 计算 matrix+1，写 buf0
  op1 读取 buf0，写 buf1

op2：extern mm(buf1, weight)，写 buf2
op3：普通 sin(buf2)，写 buf3
```

wrapper 最终让 buf1 复用 buf0 的存储，并让返回值 buf3 接管 buf2 的存储。调度记录仍保留
四个 operation 和四个结果名；逻辑名不同不等于一定分配四块物理存储。

两个输入在生成代码中记为 `arg0_1=matrix`、`arg1_1=weight`。入口状态为：

```text
C++ group = 空
available = {arg0_1, arg1_1}
completed = {}
pending   = {}
```

### 访问普通融合节点 op0_op1

`None→CPU` 先做空 flush。随后：

- `last_usage={arg0_1,buf0}` 加入 pending；
- backend 为 op0、op1 决定复用并安排 buf0、buf1 的空间，把两个循环留在当前 C++ group；
- available 加入 buf0、buf1，completed 加入 op0、op1。

```text
C++ group = {op0, op1}
available = {arg0_1, arg1_1, buf0, buf1}
completed = {op0, op1}
pending   = {arg0_1, buf0}
```

此时输出空间已经安排，但产生 buf1 的 kernel 调用还没写出，所以 pending 要等这个组提交后
才能检查。

### 访问 extern op2

op2 是 extern，即使仍在 CPU，也先触发 flush。backend 写出 pointwise kernel 定义与调用，
然后检查 `arg0_1、buf0` 并清空旧 pending。buf0 的存储已由 buf1 接管，所以无需再释放
一块独立的 buf0 存储。

接着才处理 op2：

- pending 加入 `op2.last_usage={arg1_1,buf1}`；
- extern 入口安排 buf2，并直接向 wrapper 写 mm 调用；
- mm 调用之后立即检查并清空当前 pending；
- available 加入 buf2，completed 加入 op2。

```text
C++ group = 空
available = {arg0_1, arg1_1, buf0, buf1, buf2}
completed = {op0, op1, op2}
pending   = {}
```

### 访问普通 op3 并结束遍历

op3 与上一节点同为 CPU，且不是 extern 或 template，所以没有前置 flush：

- pending 加入 `op3.last_usage={buf2}`；
- backend 安排 buf3 复用 buf2，把 sin 留在新的 C++ group；
- available 加入 buf3，completed 加入 op3。

遍历结束触发最终 flush：backend 写出 sin kernel 定义与调用，再检查 buf2。buf2 的存储已由
函数输出 buf3 接管，因此不会释放返回值的物理空间；pending 清空。

wrapper 中相关语句的运行顺序是：

```text
安排 buf0；让 buf1 复用 buf0
调用 pointwise kernel(matrix → buf1)
结束 matrix 的输入引用
安排 buf2
调用 mm(buf1, weight → buf2)
结束 weight 和 buf1 的引用
让输出名 buf3 接管 buf2
调用 sin kernel 原地更新 buf3
返回 buf3
```

这条时间线展示的是编译器写入 wrapper 的顺序。程序以后调用 wrapper 时，设备才按这个顺序
执行。

## 第三项：节点身份与 backend 共同决定生成入口

所有节点都先经过前置边界和 `last_usage` 登记，再按下面五种身份分派：

| 节点与 backend 条件 | 本层动作 | 当前节点登记的 pending 在哪里清理 |
| --- | --- | --- |
| 普通或融合节点 | 对应完整 device 的 `backend.codegen_node()` | 由该 backend 决定；默认 CPU C++ 等待 Scheduler flush，默认 CUDA Triton/SIMD 路径在入口末尾清理 |
| extern | wrapper 直接写已有函数调用 | extern 调用写完后 |
| template 或带 epilogue 的模板融合组 | backend 同时处理模板主体和 epilogue，写一个模板 kernel 调用 | template 入口末尾 |
| foreach/combo | 允许的 SIMD 或 CUDA combined backend 写组合 kernel 调用 | combo 入口末尾 |
| nop | `mark_run()` 处理其输出，不写 kernel | 下一个清理点 |

普通/融合节点这一行也适用于其他已注册 backend；表中的 CPU 与 CUDA 行为只是本篇随后用来
推演时间线的两个当前实例。自定义 device backend 同样接收 `codegen_node()`，但何时提交和
清理由它自己的实现决定。

foreach/combo 只允许交给 SIMD 或 CUDA combined backend；遇到其他 backend 会直接报错。
它本身不触发前置 flush。Grouped 节点已在入口前拆开，因此表中没有 Grouped 分支；若未知
类型意外进入 `_codegen()`，它不会被当作普通节点处理，而会在要求 Nop 的断言处失败。

nop 表示无需 kernel 调用的 IR 工作。它的 `mark_run()` 会逐个处理输出，但某个输出是否
需要独立分配仍由该输出的 layout 和复用关系决定。nop 分支不调用 `free_buffers()`：

- 后继是 extern 或 template 时，后继的前置 Scheduler flush 先清理 nop 留下的 pending；
- 若后继 device 与 Scheduler 当前的 `current_device` 不同，先由 device-change flush 清理；
- 若 device 与 `current_device` 相同，后继是 SIMD 普通节点或 combo 时，由其生成入口末尾
  清理累积的 pending；后继是 CPU C++ 普通节点时，候选继续积累到下一次边界或最终 flush。

Nop 自己不更新 `current_device`。因此若它是整列第一个节点，后继即使与 Nop 报告同一种
device，也会因为 Scheduler 的 `current_device` 仍为 None 而先触发 device-change flush。

template backend 会对主体和 epilogue 都安排输出并写入同一个调用。当前 Scheduler 分支随后
只把模板主体的结果名和 operation 名加入公共 available/completed；epilogue 的空间和计算
没有遗漏，因为它们已经由 template backend 写入 wrapper。completed 少记 epilogue 可能使
后续原地复用判断保守地保留一个“尚未处理”的 user，从而放弃一次复用机会；它不会重复生成
epilogue，也不会提前覆盖仍需读取的数据。

普通 backend 还可在节点分派后用 `ready_to_flush()` 报告当前组已满。Scheduler 只读取这个
结果；具体资源上限由 backend 自己判断。

## 第四项：device 边界、空列表与最终出口

若已经积累 CPU 普通节点 A、B，下一节点 C 在 `cuda:0`，顺序是：

```text
flush 所有已创建 backend，提交 CPU {A,B} 并清理旧 pending
退出旧 device guard（若需要），进入 cuda:0 guard
登记 C.last_usage
把 C 交给 cuda:0 backend
更新全局 available/completed
```

device 切换发生在登记 C 的 `last_usage` 之前，所以旧 flush 不会处理 C 的释放候选。
available 和 completed 跨 device 保留。guard 也是将来写入 wrapper 的上下文代码；编译器
此刻没有进入 GPU。

遍历结束时，Scheduler 先退出最后一个需要的 device guard，再执行最终 flush。该 flush
遍历全部已创建 backend，提交任何剩余组，并进行最后一次 pending 清理。

若最终节点列表为空，逐节点循环、device guard 和节点分派全部跳过；Scheduler 仍执行最终
flush。通常这时没有 backend 工作和 pending，因此不产生 kernel、extern 或释放请求。
GraphLowering 仍会继续让 wrapper 汇总一个可调用模块。

除第一项列出的 backend 创建错误外，分派还可能在以下位置终止：

- foreach/combo 遇到不支持它的 backend；
- 未知节点类型进入身份分派；
- GPU guard 所需的 device index 缺失；
- 任一 backend 的 codegen、flush 或 wrapper 写入抛出异常。

本层没有为这些错误建立 fallback。当前源码还有一个可复现的边界：Nop 节点配合
`triton.debug_sync_kernel=True` 时，debug-sync 路径读取了该分支没有设置的局部 device，
会触发 `UnboundLocalError`；这是编译失败，不是 Nop 的正常出口。

正常结束时，`Scheduler.codegen()` 没有返回一份新的节点列表。它通过副作用留下：

| 交给后续 Codegen 的内容 | 当前状态 |
| --- | --- |
| 各 backend 的 kernel 缓冲 | 所有待提交组已经 flush；kernel 定义与调用已经写入相应缓冲 |
| wrapper 缓冲 | 已按顺序记录 allocation/reuse、extern、kernel call、guard 与可释放语句 |
| `available/completed` | 保留为本轮编译历史；不会变成运行时状态 |
| pending | 最终 `free_buffers()` 后为空 |

控制返回 GraphLowering 后，Codegen 篇继续打开这些缓冲怎样形成 kernel 源码、怎样规划 wrapper
内存，以及怎样汇总、编译和加载最终 Python 模块。

## 改变条件后重新推演

把主案例中的 extern mm 换成一个与前后节点兼容的 CPU 普通节点，并令
`ready_to_flush()` 始终返回 false。mm 不再制造前置边界，三个阶段都进入同一个 C++
group；最终只需一次非空 flush。

若只把 `ready_to_flush()` 改为在 mm 后返回 true，Scheduler 会在 mm 分派和公共状态更新
之后提交 `{op0,op1,op2}`，清理当时的 pending；sin 从空组开始，在最终 flush 单独提交。

若再把 mm 移到另一 device，则先 flush 旧 CPU 组、切换 guard，之后才登记 mm 自己的
`last_usage` 并交给新 backend。节点身份若仍为 extern，它的调用写完后还会立即清理自己
登记的 pending。

面对一列新节点，始终按同一顺序推演：选择完整 device 的 backend；检查当前节点前是否
flush 和切 guard；登记 `last_usage`；按身份生成；更新 available/completed；检查 backend
是否要求 flush；遍历结束再退出 guard 并最终 flush。
