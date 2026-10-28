# Wrapper 与内存计划：把 kernel 串成一次可调用执行

[上一项：CPU C++ 与 GPU/Triton](../02-backends/README.md) · [返回 Codegen](../README.md) · [下一篇：缓存、编译与加载](../04-cache-load/README.md)

Q42 已经交出 kernel 定义和调用合同：kernel 名、tensor 参数顺序、逻辑范围，以及 GPU
调用所需的 grid、device 和 stream。Scheduler 驱动 codegen 时还按执行顺序向 wrapper 写入
allocation、free、kernel/extern call 和 device 行，并提供 buffer layout、`last_usage` 与图的
输入、输出和修改关系。本篇把这两路对象合成一份可调用的 wrapper 源码。

kernel 只知道若干地址参数上的局部计算。生成的 C++ 或 Triton 代码把这种地址参数叫
pointer：它指向一块 tensor 存储的起始位置，kernel 再用索引找到要读写的元素。wrapper
接收 tensor 参数，显式核对当前调用所需的 shape/stride。当前 Python wrapper 的
`assert_size_stride` 不检查 dtype/device；这些专门化条件用于进入 `module.call(args)` 之前
的 callable 选择。常规调用由 Dynamo guard 选择现有版本；进入编译/缓存查找时，FX Graph
Cache 再检查候选条件。不满足时，外层寻找其他候选或重新编译，当前 wrapper 不会在调用中
改写自己的 kernel。wrapper 依据已经成立的设备条件设置 device guard、分配空间，把各块
storage 的起始地址传给 kernel，按 Scheduler 的顺序调用它们，最后返回模块 tuple。因此
本篇从“进入 wrapper 时 dtype/device 合同已经成立”开始；Dynamo guard 与 FX Graph Cache
分别怎样查找候选，由 Q44 沿缓存调用链继续展开。

本篇依次完成四项工作：

```text
kernel 调用合同 + 按顺序写下的 wrapper 行 + 输入/输出/修改关系
  │
  ├─ 1. 形成包含 allocation、free、call 和 device 的完整时间线
  ├─ 2. 选择普通 reuse 或 inference pool，确定每个逻辑 buffer 使用哪块 storage
  ├─ 3. 生成 boxed 入口、检查、分配/view、调用、释放和模块返回 tuple
  └─ 4. 加入 device/stream 作用域，把完整 wrapper 源码及输出合同交给 Q44
```

这里要始终分开三层对象：

| 层次 | 例子 | 含义 |
| --- | --- | --- |
| 逻辑 buffer | `buf0`、`buf1` | 编译图中两项结果的身份 |
| Python tensor 引用 | wrapper 局部变量 `buf0` | 指向 tensor 对象的 Python 名字，可被 `del` |
| storage | 下文记为 S0、S1 | 真正保存元素的底层存储，可被多个 tensor/view 引用 |

后文统一使用三份合同：fresh-output contract 说明哪些新 storage 必须由生成模块 tuple 交出；mutation contract 说明 kernel 要对哪个输入 storage 产生副作用；output-alias contract 说明某个用户输出就是原输入或它的 view。三份合同可以同时出现，也彼此独立：函数可以修改 x 却不返回 x，也可以不修改 x 而只返回它的 view。

内存计划会把这些动作放在一条时间线上。例如 buf0 在第 1 行分配，第 2 行被 kernel A 写入，第 4 行最后一次被 kernel B 读取，第 5 行以后就可释放。如果第 6 行需要的 buf1 命中该 planner 的兼容 key，planner 可以让 buf1 复用 buf0 的同一块内存，而不是再次申请。本文的普通 reuse planner 要求 storage-size 表达式相等；pool planner 才另行安排不同大小对象的空间位置。若多份逻辑 buffer 从一块较大的 allocation pool 取空间，则每份记录自己的 offset，例如 buf0 使用字节 0..255、buf1 使用字节 256..511。

```text
装有所有实参的 Python 列表 args（boxed 调用：多个实参装入一个列表）
   │ 取出实参、清空列表、检查 shape/stride；外层 guard 约束 dtype/device 等条件
   ▼
输入 tensor ──→ 分配/复用 buffer ──→ kernel / extern 调用
                       ▲                    │
                       └── last use 后释放 ┘
                                            │
                                            ▼
                                 生成模块 tuple + 写回副作用
                                            │
                                            ▼
                              外层按输出元数据恢复用户结果
```

## Scheduler 写入的是“待规划的 wrapper 行”

后端 codegen 进行时，`WrapperCodeGen` 并非立即把所有内容写成最终字符串。分配和释放先保存为带 buffer 对象的行：

- `AllocateLine(buffer)`：此处开始需要一块存储；
- `FreeIfNotReusedLine(buffer)`：此处以后不再使用，可释放，也可留给后续 buffer；
- `ReuseLine(old,new)`：新逻辑名字接管旧存储。它有两个生产时机：Scheduler 已经批准
  pointwise 等节点原地覆盖时，会直接写入这种行；普通 planner 扫描完整时间线后，也会把
  匹配成功的 `FreeIfNotReusedLine(old)+AllocateLine(new)` 改成同一种最终行；
- 普通文本/调用行：保留实际调用顺序，使规划扫描不能把分配、释放任意移过这些动作；
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

这次生成物中的局部实参顺序与 Python 函数签名顺序不同，先固定对应关系：

| wrapper 局部名 | 原函数参数 | 首次用途 |
| --- | --- | --- |
| `arg0_1` | `weight0` | 第一 mm 的右输入 |
| `arg1_1` | `x` | 第一 mm 的左输入 |
| `arg2_1` | `weight1` | 第二 mm 的右输入 |

因此第一条调用是 `mm(arg1_1, arg0_1)`；它结束后 x 和 weight0 都不再使用，可以删除这两个
wrapper 局部引用。第二条调用才消费 `arg2_1`。

Scheduler 和 backend 完成 codegen 后，规划器先看到的是下面这条时间线，而不是已经决定好
所有 `empty` 和 `del` 的 Python 文本。这里省略 kernel 定义，只保留规划所需的行：

```text
unpack/guard inputs
AllocateLine(buf0)
Call extern mm(inputs → buf0)
free last-used input references
ReuseLine(buf0 → buf1)             # Scheduler 已批准 relu 原地覆盖
Call cpp_fused_relu_0(buf1)
AllocateLine(buf2)
Call extern mm(buf1, weight1 → buf2)
free last-used weight1 reference
FreeIfNotReusedLine(buf1)
ReuseLine(buf2 → buf3)             # Scheduler 已批准 sin 原地覆盖
Call cpp_fused_sin_1(buf3)
Return module tuple (buf3,)
```

这条输入时间线中的两个 `ReuseLine` 都由 Scheduler 的原地决策产生：relu 接管 `buf0`，
sin 接管 `buf2`。它们不是普通 planner 从 free-key 栈中选出的。普通 planner 随后另会尝试
把较早的 `FreeIfNotReusedLine` 与后续兼容的 `AllocateLine` 配对，匹配成功时也输出
`ReuseLine`。本例第二次 mm 分配 `buf2` 时，`buf1` 仍要作为输入，所以两者不能配对。

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

规划完成后，入口解包和检查、上面的分配/复用/调用行以及返回共同进入同一个 boxed
`call(args)`。把不影响本例的模块级定义折叠后，结构是：

```python
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(...)                 # 三个输入的当前合同
    # buf0 empty → mm → buf1 接管 → relu
    # buf2 empty → mm → buf3 接管 → sin
    return (buf3,)
```

Q44 接收的是含 kernel binding/import 和这段 `call(args)` 的完整 wrapper 源码；上面折叠的
注释位置就是紧邻代码块中已经列出的实际行，并没有省掉另一轮运行时规划。

按对象状态逐步看：

| 时刻 | 活跃存储 | 为什么还活着 |
| --- | --- | --- |
| 第一 mm 后 | `buf0` 引用 S0；尚需使用的输入引用 | S0 将供 relu/第二 mm 使用 |
| relu 前后 | `buf1` 接管 S0；局部名 `buf0` 已删除 | relu 可原地覆盖 S0 |
| 第二 mm 时 | `buf1` 引用 S0；新 `buf2` 引用 S1 | mm 同时读 S0、写 S1，不能共用一块存储 |
| 第二 mm 后 | `buf2` 引用 S1 | S0 的最后用户结束，`buf1` 可删除 |
| sin 后 | `buf3` 接管 S1；局部名 `buf2` 已删除 | sin 原地覆盖 S1，S1 成为图输出 |

所以这里只为两个 mm 结果真正申请两块 8×8 存储，relu 与 sin 没有各自申请完整输出。`buf1 = buf0; del buf0` 不是复制 64 个元素：Python 赋值先增加一个指向同一 tensor 对象的局部引用，再删除旧局部引用，storage 始终是 S0；对应 C++ kernel 参数也是 `float* in_out_ptr0`。

实验输出与 eager 相同。这个案例证明当前图的复用决定，不能推广成“所有 pointwise 都可原地”：旧值若仍有后续用户、输出存在 alias 约束、dtype/layout/容量不兼容或 buffer 被列入 `never_reuse_buffers`，都不能这样复用。

## `last_usage` 怎样变成 `del`

`last_usage` 是 Scheduler 为当前节点记录的集合：其中每个 buffer 在该节点之后都不再有用户。节点 codegen 后，Scheduler 让 wrapper 对这些 buffer 执行 `codegen_free()`：

- 图输入不能作为新中间 buffer 的可复用存储，但 Python 局部引用可在最后一次使用后 `del`；
- 常量、图输出、已经释放、禁止复用或特殊 alias buffer 不进入普通复用池；
- 普通中间 buffer 生成 `FreeIfNotReusedLine`，规划阶段才决定输出 `del name` 还是让后续 `AllocateLine` 变成 `ReuseLine`。

逻辑 last use 先让 buffer 在静态规划中成为复用候选。若没有后续 allocation 与它配对，最终代码输出 `del old`；若成功配对，最终代码直接生成新引用接管旧 storage，而不是运行时先 `del`、再动态寻找内存。Python 引用计数和设备 allocator 是否真正回收或缓存底层块又是更外层的事件。因此一行 `del` 不等于此刻强制把设备内存归还操作系统。

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

同一个 key 下的候选按栈保存。假设 A、B 都是同 device、dtype、storage size 的中间
buffer，并依次结束最后一次使用：

```text
FreeIfNotReusedLine(A) → 候选栈 [A]
FreeIfNotReusedLine(B) → 候选栈 [A, B]，B 在栈顶
AllocateLine(C)        → 弹出 B，生成 ReuseLine(B → C)；A 仍留作后续候选
```

所以“最近释放块”指同一 key 内最后压栈的 B。A、B 都满足容量合同，选择顺序只决定 C
接管哪块 storage，不改变 C 的逻辑结果。

它比较的是符号存储容量表达式相等，不是“大块容量至少够用”就算命中，也不直接比较 shape/stride。shape/stride 完全相同时生成直接别名；key 相同但 shape/stride 不同时，可用 `reinterpret_tensor` 在同一 storage 上建立新 view。

例如旧结果是连续 `(2,3)`、stride `(3,1)`，新结果需要连续 `(3,2)`、stride `(2,1)`；两者都是 float32、同 device、storage size 都是 6 个元素。旧结果最后使用后，可以生成：

```python
new = reinterpret_tensor(old, (3, 2), (2, 1), 0)
del old
```

`new[2,1]` 的线性位置是 `2*2+1=5`，仍落在这块 6 元素 storage 内。旧逻辑值已经死亡，所以换 shape 解释不会改变仍需观察的结果。`can_reuse()` 还会排除图输入、常量、已经释放、`never_reuse_buffers` 等特殊身份；相同元素数若不能得到相同的 storage-size 表达式，或身份受保护，也不能自行判定可复用。

## inference 的 pool memory planning

开启 `config.memory_planning` 且当前生成是 inference 时，wrapper 改用 `MemoryPlanner`：

1. 丢掉已被优化删除的 buffer 行；
2. 把共享同一 storage 的逻辑 buffer 名归入一个 `BufferGroup`；一个 group 是 pool 放置时不可再拆开的 storage 身份；
3. 沿完整 wrapper 时间线计算每组 first/last usage；必须由生成模块交出的输出 storage/group 生存期延长到调用末尾；输入 alias/view 可由外层保存的 base 恢复，不必成为普通 pool 输出 group；
4. 按 device 建立 pool；`TemporalSplit` 让生存期不重叠的 allocation 使用同一空间区域，`SpatialSplit` 为生存期重叠或大小不同的对象安排不同 offset；
5. 把每个 tensor 表达成 pool 基址、offset、shape、stride 的 view，仅在 pool 首/末次使用处生成/删除 pool 基底引用；底层 allocation 何时回到 allocator 或系统仍由引用和 allocator 决定。

其关键判据是生存期是否重叠，而不只是源码顺序相邻。训练路径目前不会使用这套完整 pool planner，因为源码明确注明它当前会增加训练峰值内存；训练或未开启该配置时使用前述 reuse planner。

先用一个忽略对齐的最小放置稿看清两种切分。三个 group 都在同一 device：

| group | 大小 | 生存时刻 |
| --- | ---: | --- |
| A | 64 B | 1..3 |
| B | 32 B | 4..6 |
| C | 48 B | 2..5 |

A 与 B 的生存期不重叠，可以在时间上复用 pool 的字节 `0..63`：A 结束后，B 仍从 offset 0 开始。C 在时刻 2..3 与 A 重叠、在 4..5 与 B 重叠，不能占它们正在使用的区域，因此可放在 offset 64，使用字节 `64..111`。

```text
pool 0..63   ：时刻 1..3 放 A；时刻 4..6 放 B   （时间共享）
pool 64..111 ：时刻 2..5 放 C                   （空间分开）
```

时刻 2 有 A+C 同时存活，因此 112 B 是这个例子的容量下界；上述放置达到该下界，而不是声称任意 planner 必然选出同一组 offset。实际 planner 还要处理对齐、符号大小和更复杂的嵌套切分；这个数值稿只展示决定 offset 的两条约束：live range 不重叠可以占同一区域，重叠的 group 必须位于不同空间。

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

`pool1` 对应中间 storage group `buf0/buf1`，最后使用发生在第二次 mm，所以随后整体销毁；`pool0` 对应输出 group `buf2/buf3`，生存期延长到返回。本例每个 pool 只有一个 8×8 storage，所有 offset 都是 0；它实际证明了分组、live range 和 pool 首末使用。上面的 A/B/C 数值稿负责解释时间共享与非零 offset，不能冒充这次实验的生成结果。

## 输入检查、符号尺寸与输出协议

实验入口先执行：

```python
arg0_1, arg1_1, arg2_1 = args
args.clear()
assert_size_stride(arg0_1, (8,8), (8,1))
...
```

boxed 调用指把原本多个位置参数装进一个列表传给生成模块。`args.clear()` 让这个容器不再额外持有 tensor；拆出的局部变量仍保持所需引用。静态版本用 `assert_size_stride` 检查编译所依赖的 shape/stride。dtype/device 用于外层选择这个已编译版本；进入当前 Python wrapper 后，代码按照该设备条件设置 guard 和分配，不再发出 dtype/device 断言。

动态版本会把运行时尺寸绑定到符号：例如编译时把第一维记为 `s0`，本次输入 shape 为 `[19]`，入口便令 `s0=19`。若所选 kernel 要求长度能被 16 整除，就生成 `s0 % 16 == 0` 之类的检查。guard 不匹配时不是这段 wrapper 临时改写自己的 kernel；wrapper 外的编译缓存/调度路径需要寻找匹配版本，或为新条件重新编译。动态形状不是简单跳过检查。

生成模块只需交出不能由外层现有引用恢复的新结果及必要基底。外层运行元数据为每个用户输出记录“新结果”“就是某个输入”“某个输入的 view”等关系，并在调用结束后恢复用户可见的 tuple。内存计划必须保留这些关系所要求的 storage 生存期。

### 修改输入时，返回合同包含“新结果”和“原 storage”两条线

看一个同时产生新结果、修改输入并返回该输入的请求：

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

这个函数同时触发三份独立合同：

| 合同 | 本例内容 | 即使没有另外两项是否仍成立 |
| --- | --- | --- |
| fresh output | `old` 位于新 S1，生成模块必须交出 `buf0` | 是 |
| input mutation | kernel 把 y 写入输入 x 的 S0 | 是；即使函数不返回 x，调用者仍观察到修改 |
| output alias | 用户输出第 2 项就是输入 0 本身 | 是；即使 x 未被修改，也应返回同一输入对象 |

在本次 inference 编译入口中，data mutation 被保留在生成模块里；生成模块因此直接写 S0，不需要把“更新后的 x”作为新 tensor 返回。输出 alias 元数据则让运行层知道用户还要求返回输入 0。按这两个事实，本例的语义流程可以写成：

```text
运行层保留用户传入 x 的引用，供输出 alias 恢复使用
→ 调用生成模块 [x,y]
    kernel 把 old 写入新 S1，同时把 y 写回 x 的 S0       # mutation contract
    生成模块只返回 (buf0,)                 # 新结果 old
→ 外层读取模块结果 old=buf0
→ 按 output-alias 元数据把原输入引用作为第 2 项
→ 向用户返回 (old, x)
```

因此实验中 `returned_x.data_ptr()==x.data_ptr()`。上面是这次编译入口和当前实验的合同解释，不承诺所有 ABI 都用完全相同的 tuple 形态；关键不变量是三份合同分别成立。wrapper 中 `del arg0_1` 只删除生成模块内部的局部 Python 名；调用者与运行层所需的输入引用仍然存在，所以这行不会撤销输入对象，也不会把 S0 放进普通中间复用池。

view 输出使用同一原则。若函数返回 `x.view(3,1)`，外层元数据记录“该输出以输入 x 为 base，并采用 shape `(3,1)` 与对应 stride”；生成模块不必复制三项数据，运行协议可从保存的原输入重建 view。若返回的是新计算 `x+1`，它没有现成输入 base，生成模块才需要交出新 buffer。

若函数只执行 `x.copy_(y)`、另返回 `y+1`，调用者仍能观察 x 被改成 y，但新增返回值使用独立 storage。可见“图输出”不能只靠生成模块最后那行 tuple 判断：新增返回 buffer、mutated input 和 alias/view 都由图的输出/修改合同共同组装。

## 没有中间 buffer 时怎样结束

wrapper 不要求每次调用都产生可规划的中间 storage：

| 请求 | wrapper 中仍要做的事 | 模块 tuple 与用户结果 |
| --- | --- | --- |
| 零元素输入 | 解包输入；当前实现跳过该输入的 size/stride 断言 | 按输出合同返回或恢复结果 |
| 零元素计算结果 | 这类 `ComputedBuffer` 更早成为空节点，通常没有普通 allocate/free 行 | 若输出能由输入 alias/view 恢复，由外层恢复；否则遵守对应空结果合同 |
| 只有输入 mutation | 调用写回输入 storage；无需为修改后的输入新建中间 buffer | 模块 tuple 可以为空，调用者仍从原输入观察副作用 |
| 只有输入 alias/view 输出 | 保存输入引用，不申请复制数据的 storage | 模块 tuple 可以为空，外层按元数据重建用户输出 |
| 没有可复用块 | 正常生成新的 allocation | 这不是规划失败，只是没有 reuse |

没有 planning lines 时，两种 planner 都能直接处理空列表，wrapper 继续生成相应 return。选择
pool 也不是一次“先试 pool、失败再退普通 reuse”的运行时探测：只有 inference 且
`config.memory_planning` 开启时才进入 pool planner；训练或配置关闭时从一开始就选普通
reuse。若规划数据违反内部必须成立的约束、kernel 调用合同无法形成合法 wrapper，编译会
以异常结束，不会在已生成的 wrapper 运行时悄悄换一个 planner 或转回 eager。

## device guard 与 stream 位于 wrapper 层

Scheduler 的 `_codegen()` 观察当前节点 device；设备变化时先结束旧 guard，再调用 `codegen_device_guard_enter(device.index)`。Python wrapper 对 GPU 生成大意如下的作用域：

```python
with device_ops.device_guard(device_index):
    device_ops.set_device(device_index)
    # 获取当前 raw stream，并按顺序调用 Triton/extern kernels
```

Triton 调用由 wrapper 写成 `kernel.run(..., grid=..., stream=streamN)`。这里的 stream 是设备按提交顺序执行工作的队列；`streamN` 是 wrapper 保存的局部变量，里面装着运行时取得的底层设备 stream 句柄，源码有时称它为 raw stream。相邻 kernel 使用同一个 `streamN` 时，提交顺序也规定了它们的执行先后。guard 确保“当前设备”与取得的 stream、分配和 kernel launch 一致；CPU-only 图不需要这段。

这保证的是同一 stream 上 device 工作的先后，不表示 host 在每次 launch 后等待 kernel 完成；若工作位于不同 stream，仅凭 wrapper 文本相邻也不能推出依赖，必须另有同步关系。

因此 GPU 上的逻辑 last use 与设备完成时刻不是一回事，两条复用路径要分别判断：

- 直接用 alias/reinterpret 让新逻辑 buffer 接管同一 storage：同一 stream 依靠先读后写的提交顺序；若后续覆盖提交到另一 stream，提交方必须建立 event/wait 或其他等价执行依赖，allocator tracking 不能替代这条依赖。
- Python 引用归零、storage 先交还缓存 allocator，之后又被分配：allocator/运行时必须记录旧 allocation 曾在哪些 stream 使用，避免把仍被异步工作访问的块过早交给无序的使用者。

两条路径都不能只靠一行 `del` 推出安全。

C++ wrapper 和 AOT 模式会使用相应的 CUDA guard/stream guard 类型，语法不同但职责相同。多 device 图在 Scheduler 顺序中切换 guard，不能让单个 kernel 自行猜测当前 device。

例如 Python wrapper 的时间线中依次出现 CPU、`cuda:0`、`cuda:1` 三段工作时，作用域可以
画成：

```text
CPU allocation/call
enter guard(cuda:0) → 取得 stream0 → cuda:0 allocation → launch kernel0(stream0) → exit
enter guard(cuda:1) → 取得 stream1 → cuda:1 allocation → launch kernel1(stream1) → exit
return outputs
```

默认 reuse key 含 device，pool 也按 device 分组，所以 `cuda:0` 的 storage 不会成为
`cuda:1` allocation 的复用候选。若 `kernel1` 需要 `kernel0` 的结果，图中还必须存在合法的
跨设备复制/通信与同步安排；两个 guard 在 wrapper 文本中相邻本身不传送数据。当前 AOT C++
wrapper 只接受单个 CUDA device，第二个 CUDA device guard 会触发编译期检查；上面的多设备
时间线描述的是支持这种切换的 Python wrapper 路径。

## 本篇交给缓存与加载阶段的内容

四项工作完成后，`GraphLowering.codegen()` 可以汇总一份完整 wrapper source：

| 源码或元数据部分 | 已确定的内容 |
| --- | --- |
| 模块顶层 | kernel bindings、imports、常量与辅助定义 |
| boxed `call(args)` 入口 | 参数解包、`args.clear()`、size/stride 检查和符号尺寸绑定 |
| 已规划时间线 | allocation 或 pool view、reinterpret/reuse、kernel/extern calls、`del` |
| 设备执行上下文 | device guards、raw stream 获取和 GPU launch 参数 |
| 模块返回 | 不能只靠既有输入引用恢复的新结果 tuple |
| Inductor 调用元数据 | mutated-input 索引、output strides、guards 等字段 |

Q44 接住 wrapper source，把它写入缓存并加载为 `module.call`；随后把 callable、适用条件、
mutated-input 索引和 output strides 等 Inductor 字段装入 `CompiledFxGraph`。input
alias/view 的用户输出重建关系由更外层 AOTAutograd 的运行元数据继续保存，不属于
`CompiledFxGraph`。缓存和加载不会重新做本篇的 storage 规划。

## 从一段 wrapper 反推执行计划

阅读生成代码时可按下面顺序：

1. 从 guards 确认该编译版本接受的输入合同；
2. 给每次 allocation/reuse 标出 storage 身份，而不只看 `bufN` 名字；
3. 沿 kernel 调用记录每块存储的最后一次 host 侧提交使用；
4. 检查 free/reuse 是否位于该提交之后、图输出是否一直存活；GPU 还要结合 stream/allocator 规则判断设备侧安全性；
5. GPU 情况再检查 device guard、grid 和 stream 是否包住对应调用。

对本例而言，逻辑 buffer 名是 `buf0..buf3`，物理上只有两次新分配：`buf0/buf1` 一组、`buf2/buf3` 一组。这正是 wrapper 内存计划要表达的区别。

换到修改输入案例，物理身份则是“输入 x 的原 storage + 新建 old 的 storage”；即使 wrapper 局部名被 `del`，调用者持有的 x 仍须保活并呈现写回值。先标 storage 身份，再读名字和 `del`，两类案例便能用同一方法解释。

## 用一段新时间线检查 storage 判断

假设所有中间结果都在同一 device、dtype、容量为 6 个元素，输入 `x` 另有自己的 S_in。下面先直接写 planner 收到的抽象行；`AllocateLine` 尚未发射 `empty`，`FreeIfNotReusedLine` 也尚未发射 `del`：

```text
AllocateLine(buf0, shape=(2,3), stride=(3,1))             # 申请逻辑 buf0
CallLine(kernel_a, x, buf0)
InplaceAliasLine(buf0 → buf1, shape=(3,2), stride=(2,1))  # 已批准原地覆盖，同一 S0

AllocateLine(buf2, shape=(3,2), stride=(2,1))             # 此时 S0 仍被读，需 S1
CallLine(kernel_b, buf1, buf2)
FreeIfNotReusedLine(buf1)                                 # S0 成为候选

AllocateLine(buf3, shape=(6,), stride=(1,))               # 尚未决定 empty 或 reuse
CallLine(kernel_c, buf2, buf3)
FreeIfNotReusedLine(buf3)
ReturnLine(buf2)
```

`InplaceAliasLine` 是本题使用的教学标记：它表示 Scheduler/codegen 已经根据值的最后用户和原地操作条件批准 `buf1` 覆盖 `buf0`，作为同一 alias group 交给全局 memory planning；它不是普通 planner 将 free 与后续 allocate 配对后产生的 `ReuseLine`。

先自行标注，再核对：`buf0/buf1` 都引用 S0；第二个 allocation 发生时 S0 仍被 `kernel_b` 读取，所以 `buf2` 必须得到 S1；`FreeIfNotReusedLine(buf1)` 使 S0 进入静态复用候选，`buf3` 可用 `(6,)` view 接管它；`buf3` 最终没有用户，free 记录不会改变输出。生成模块只返回引用 S1 的 `buf2`。

规划阶段会消费较早的 free 行，并把 `buf3` 的 allocation 行替换成下面这个局部片段：

```python
buf3 = reinterpret_tensor(buf1, (6,), (1,), 0); del buf1  # S0
kernel_c(buf2, buf3)
del buf3
```

若用户函数的第二个输出还是 `x.view(2,3)`，外层会另外保存 S_in 的输入引用并按输出元数据重建该 view；它不应把已死亡的 S0 或输出 S1 误当作输入 view 的 base。这个新例同时检查了不同 layout reinterpret、同时活跃时不可复用、最后使用后的再复用，以及模块 tuple 与用户语义 tuple 的区别。

## 继续阅读实现

[Wrapper 复用规划源码：LIFO、live range 与 allocation pool](01-planner-source/README.md)
