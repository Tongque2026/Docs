# Codegen：在 Scheduler 驱动下写出并加载可调用函数

[上一篇：Scheduler](../03-scheduler/README.md) · [返回总览](../README.md)

[虚拟化 ops](01-kernel-ops/README.md) · [CPU C++ 与 GPU/Triton](02-backends/README.md) · [Wrapper 与内存计划](03-wrapper-memory/README.md) · [缓存、编译与加载](04-cache-load/README.md)

Scheduler 已经形成最终节点顺序；它不会先返回一列工作，再启动一个独立的 Codegen
遍历。`Scheduler.codegen()` 沿这列节点前进时，直接调用 backend 和 wrapper 的生成
接口：普通节点带来逐位置计算，extern 节点要求 wrapper 写已有函数调用，template 节点
进入专用生成器。这里把这些被调用的生成工作统称为 Codegen。Scheduler 的循环结束时，
kernel 与 wrapper 的代码缓冲区已经被逐步填充；`GraphLowering.codegen()` 随后让 wrapper
汇总完整模块源码，再由缓存和加载流程取得 PyTorch 可以调用的函数。

本篇主线限定为普通 `torch.compile` 的非 AOT 路径：产物是当前 Python 进程里的
`module.call`。AOTInductor 也复用部分生成机制，但最后由 AOT 编译器交出可部署文件路径，
不是下面这条 Python module 加载路线；该分支留到缓存与加载子篇比较。

最终产物不只是 kernel。一个 kernel 只知道怎样计算；调用者还需要输入检查、输出空间、kernel 与已有函数的调用顺序、结果返回，以及一份已经加载进 Python 进程的函数对象。

本篇用同一个 CPU 例子走完这条交付链：

```python
def fn(x, y):
    return torch.relu(x + y) * 2

x = [-2, 1,  3]
y = [ 1, 4, -1]
结果 = [0, 10, 4]
```

## 同一个计算会依次出现六种对象

阅读后面的过程时，要区分下面六种对象：

| 对象 | 在本例中的内容 | 谁继续使用 |
| --- | --- | --- |
| `LoopBody` | `load x → load y → add → relu → mul 2 → store` | 后端 handler |
| handler 局部值 | `tmp0`、`tmp1`、…、`tmp5` | 组成 kernel 计算体 |
| kernel 源码 | 带循环、指针读取和写回的 C++ 函数 | 模块顶层的异步编译绑定 |
| wrapper 模块源码 | kernel 编译绑定、guards、分配、调用、引用结束和 `return` | 模块缓存与加载器 |
| loaded module | 执行模块顶层并等待 kernel 编译后的 Python 模块对象 | 提供模块全局 kernel 名与 `call` |
| loaded callable | 已加载模块中的 `module.call` | `torch.compile` 的运行入口 |

这些对象不是同一个东西的不同名字。`LoopBody` 还没有 C++ 语法；wrapper 源码仍是一段文本；加载模块以后，模块顶层先建立可调用的 kernel 绑定，`module.call` 才是调用者取得的函数对象。

## 四项职责怎样协作并交接

最终的 callable 采用 boxed 调用约定：它接收一个参数列表 `args`，而不是把每个 Tensor 写成独立的位置参数。

启用 FX Graph Cache 时，Q44 在一次单图编译中出现两个时刻。入口先查缓存候选：命中时
不会创建本次 Scheduler，而是恢复已有 wrapper 的 key 与源码信息；没有合法候选或进入
缓存后发现本图不支持保存时，才走下面的完整生成路线。若本地、远端 FX Graph Cache 都
未启用，或缓存被强制关闭，则不查询候选，直接进入完整生成；这不叫 bypass。各条路径最后
都要经过 `PyCodeCache` 加载 module；若当前
进程尚未加载该 module，执行模块顶层时仍可能触发底层 kernel 编译。cache hit 省掉的是
Scheduler、LoopBody replay 和 wrapper 源码重新生成，不是无条件省掉加载与 kernel 编译。
cache miss 还会保存以后恢复 callable 所需的信息，bypass 不保存普通候选。FX 图候选的
key 与 wrapper 源码按内容得到的 key 是两种对象，不能只用一个“缓存 key”概括。

```text
单图编译请求
    ├─ FX Graph Cache 未启用/被强制关闭 ──→ 完整生成路线
    └─ 已启用：Q44 查询候选
           ├─ miss / bypass ─────────────→ 完整生成路线
           └─ hit ───────────────────────→ 恢复 wrapper cache_key/source

完整生成路线：
      Scheduler.codegen() 进入内部 _codegen() 循环
           │
           ├─ 普通节点：backend.codegen_node(node)
           │      │ LoopBody、位置范围、读写关系
           │      v
           │   Q41 虚拟 ops 协议 + Q42 device handler/backend
           │      └─ 在 replay 中共同形成局部值和 kernel 源码
           │
           ├─ extern 等节点：直接调用 wrapper 生成入口
           └─ 生存期候选：过滤后让 wrapper 写引用结束/复用动作
                  │
                  v
      Q43 wrapper 汇总 Python 模块源码：guards、空间、调用、返回
                  │
                  ├──────────────────────────────┐
恢复 wrapper cache_key/source ──────────────────┤
                                                 v
      Q44 末端：确保 wrapper 源码文件存在 → PyCodeCache 加载 module
                                     │ 首次加载时执行模块顶层，可能编译 kernel
                                     │ 只有 miss 保存 FX cache 条目
                                     v
                           boxed module.call(args)
```

Q41 与 Q42 不是两个前后运行的批处理步骤。Q41 是 replay 使用的统一 ops 协议和当前
handler 路由；Q42 安装 CPU、Triton 或其他目标的 handler/backend。LoopBody 遇到
`load`、`add`、`relu`、`store` 时通过 Q41 找到当前 Q42 handler，handler 当场返回局部
表达式并将它们组织成目标 kernel。不存在一份先由 Q41 物化、再整体交给 Q42 的 tmp 清单。

Q43 的 wrapper 缓冲区接收 kernel 源码、调用要求和 extern 调用。Scheduler 还会根据最终
节点上的生存期候选集合，在生成边界直接驱动 wrapper 尝试写引用结束或存储复用动作；这个集合
综合使用、写入、alias 和输出身份，表示“执行到当前节点后哪些 buffer 可以结束生存期”，
不只是“最后一次被读”，也不保证每项候选最终都会释放或复用。Scheduler 先排除 removed、
已经处理、图输出或没有真实分配等不能结束生存期的候选。wrapper 对输入可以写 `del` 来
结束当前 Python 局部引用，但不能把输入 storage 交给内部输出复用；常量、标记为
never-reuse 或已经释放的 buffer 也不能成为复用来源。Q43 最终把已经积累的这些内容汇总成完整模块源码。Q44 接过这份
源码，将它写入内容缓存、加载模块并取得 callable。

下面沿本例逐步代入。

## 第一步：LoopBody 在 CPU handler 下变成局部值

Scheduler 的生成循环把普通节点交给 CPU backend 时，该节点包含这串后端无关操作：

```text
load(arg0_1, x0)
load(arg1_1, x0)
add
relu
constant(2)
mul
store(buf0, x0)
```

这里的 replay 是**编译器调用 handler 方法来构造代码**，不是拿真实 Tensor 再执行一次 eager 计算。当前 handler 是 CPU C++ handler，因此每一步返回一个可供后一步引用的 C++ 局部值：

```text
load x       → tmp0
load y       → tmp1
add          → tmp2 = tmp0 + tmp1
relu         → tmp3 = max(tmp2, 0)
constant 2   → tmp4 = 2.0
mul          → tmp5 = tmp3 * tmp4
store        → out_ptr0[x0] = tmp5
```

代入位置 `x0=0`，两个 load 对应 -2 和 1，tmp2=-1，tmp3=0，tmp5=0。`x0=1` 时 tmp2=5、tmp3=5、tmp5=10。tmp 只表示当前循环位置的局部值，不是完整中间 Tensor。

这里没有“Q41 先交出 tmp 关系、Q42 再消费”的中间对象。LoopBody 每调用一次统一 ops
接口，当前 CPU handler 就立即返回对应局部表达式；Q41 的协议与 Q42 的目标 handler 在
同一次 replay 中共同形成 kernel 计算体。

## 第二步：backend 把局部关系放入 kernel

本例 kernel 省略函数签名后的计算体可以简化为：

```cpp
for (int64_t x0 = 0; x0 < 3; x0++) {
    auto tmp0 = in_ptr0[x0];
    auto tmp1 = in_ptr1[x0];
    auto tmp2 = tmp0 + tmp1;
    auto tmp3 = std::max(tmp2, 0.0f);
    auto tmp5 = tmp3 * 2.0f;
    out_ptr0[x0] = tmp5;
}
```

循环依次写出 0、10、4。kernel 的接口是输入指针和输出指针；它不负责接收 `torch.compile` 的参数列表，也不决定返回 tuple。

CPU backend 还要决定向量循环和尾部，Triton backend 则使用 program id、block 索引和
mask。这些是“同一位置语义怎样落到不同执行模型”的 Q42 内部问题，在后端子篇展开。
backend 生成 kernel 时直接向同一个 wrapper 生成对象登记 kernel 源码、参数类型和调用
要求；不是 Q42 先返回一份独立产物，再由 Q43 重新读取。Q43 最后把这些已登记内容、下面
的编译绑定和 `call` 汇总成完整模块源码。

模块顶层用下面的结构把 C++ 源码变成 Python 可调用名字：

```python
async_compile = AsyncCompile()

cpp_fused_add_mul_relu_0 = async_compile.cpp_pybinding(
    ['const float*', 'const float*', 'float*'],
    r'''...上面的 C++ kernel 源码...''',
)

async_compile.wait(globals())
```

`cpp_pybinding` 接收参数类型和 C++ 源码，安排编译并返回一个会绑定到该 Python 全局名的结果；`wait(globals())` 等待编译完成并更新模块全局命名空间。所以下面的 `call(args)` 执行时，`cpp_fused_add_mul_relu_0` 已经是可调用的 kernel 入口，而不是一个尚未解释的函数名。

## 第三步：wrapper 把 kernel 放进图级调用

Q43 为同一个例子生成的主调用可以简化为：

```python
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (3,), (1,))
    assert_size_stride(arg1_1, (3,), (1,))
    buf0 = empty_strided_cpu((3,), (1,), torch.float32)
    cpp_fused_add_mul_relu_0(arg0_1, arg1_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0,)
```

这段源码补齐了 kernel 不处理的图级合同：

| 时刻 | wrapper 做了什么 | 此后有什么 |
| --- | --- | --- |
| 入口 | 从参数列表取出 x、y，并清空原容器引用 | `arg0_1`、`arg1_1` |
| guards | 检查 shape `[3]`、stride `[1]` | 确认这份代码适用于当前输入布局 |
| 分配 | 为最终结果准备 `buf0` | kernel 有可写输出空间 |
| 调用 | 传入两输入和 buf0 | buf0 得到 `[0,10,4]` |
| 收尾 | 删除最后使用完的输入局部引用 | 返回 `(buf0,)` |

`def call` 此时仍只是完整 Python 模块源码中的一个函数定义，还不是调用者手里的函数对象。Q43 将整份模块文本交给 Q44。

## 第四步：同一份 wrapper 源码被写入并加载

`GraphLowering._compile_to_module()` 不是接收一份由别处预先生成好的源码。它先调用
`self.codegen()`，让上述 Scheduler/backend/wrapper 协作得到完整源码和 source line map，
再在同一个方法中写入并加载。source line map 记录生成文件行号对应的 FX 节点或用户源码
位置，供加载和报错映射使用：

```text
完整 wrapper source + source line map
    → PyCodeCache.write(source)
    → 得到内容 key K 和源码路径 P
    → PyCodeCache.load_by_key_path(K, P, ...)
    → 执行模块顶层并完成/等待 kernel 编译
    → loaded module
    → module.call
```

内容 key 由这份完整源码决定；源码路径 P 指向缓存中的 Python 文件。这个 key 在当前
进程尚未加载时，加载器执行该文件的模块顶层：创建 `AsyncCompile`，安排 C++ binding，
等待 kernel 编译并建立全局 kernel 名，同时执行 `def call` 定义；若同一个 key 已在进程内
的 `PyCodeCache` 中，加载器直接返回已有 module，不重复执行模块顶层。最终得到的 loaded
module 中，`module.__file__` 对应 P，`module.call` 是函数对象。`GraphLowering` 同时保存
K、P 和 source line map。

随后 `torch.compile` 用这个入口运行 `module.call([x,y])`，得到 tuple 中的 `[0,10,4]`。

因此同一个例子已经从本层入口走到出口：

```text
op0 的 LoopBody
  → CPU handler 的 tmp0..tmp5
  → C++ kernel
  → 含 def call(args) 的 wrapper source
  → 内容 key/path
  → loaded module.call
  → 运行得到 ([0,10,4],)
```

因此调用形式是 `module.call([x,y])`，而不是 `call(x,y)` 这样的普通多位置参数接口。

## 三条分类轴不要混在一起

Codegen 中经常同时出现 pointwise、CPU、extern 等名称。它们回答不同问题：

| 分类轴 | 例子 | 决定什么 |
| --- | --- | --- |
| 计算形态 | pointwise、reduction | LoopBody 中有哪些位置变量和 ops；是否需要累计状态 |
| 目标 backend | CPU C++、GPU Triton | handler 返回哪种局部表达式，kernel 使用循环还是 program/block |
| 节点生成路径 | 普通 LoopBody、template、extern | 是 replay 普通计算、由 template renderer 生成，还是直接在 wrapper 写已有函数调用 |

三条轴可以组合，并非一组互斥选项。例如普通 reduction 可以由 CPU backend 生成，也可以
由 Triton backend 生成；在当前 CPU 普通 mm 案例中，mm 选择 extern 路径，不把内部数学
重新表达成普通 LoopBody，而是在 wrapper 中写已有库调用；冻结权重的 CPU mm+relu 案例
则实际进入 template 路径。选择结果依赖 device、配置、输入和候选，不能把“mm=extern”
当作无条件规则。template renderer 是第三条“生成路径”中的专用生成器，不是与 CPU/Triton
并列的新设备轴，并可把 relu 一类 epilogue 接进模板 kernel。

这些路径最后都汇入同一份 wrapper 模块源码，再由 Q44 加载成 callable。

本教程实际运行了 CPU 生成代码；Triton 的 program id、block offset 和 mask 只由当前
源码与索引公式推演，当前环境没有 CUDA/Triton 实机结果。MultiTemplate 实验也只验证
延迟测时结果会缓存并选出最小项，没有覆盖真实 GPU template 编译执行。

## 另外两类案例改变哪项职责

主案例展示了普通 CPU pointwise。下面只打开足以判断职责边界的变化，内部算法留给四个子篇。

### reduction 改变 Q41/Q42 的计算形态

对 `(x+1).sum(dim=1)`，LoopBody 除逐位置 ops 外还带规约位置和累计操作：

```text
load(128*x0+x1) → add 1 → reduction(sum) → store_reduction(x0)
```

LoopBody 仍通过 Q41 的统一 ops 协议 replay；Q42 的 handler 需要形成累计值和规约循环。
Q43 仍负责输入检查、输出空间和 kernel 调用，Q44 仍加载完整模块。变化的是 kernel 内怎样
遍历和累计，不是 boxed callable 的基本职责。

### extern 与生成 kernel 在 wrapper 中相遇

对 `mm → relu → mm → sin`，两个 mm 作为 extern 调用写进 wrapper，relu/sin 由生成 kernel 完成。Scheduler 已给出顺序，Q43 将它们接成：

```text
分配 storage S0 → extern mm 写 S0
relu kernel 原地更新 S0
分配 storage S1 → extern mm 读取 S0、写 S1
sin kernel 原地更新 S1
返回 S1
```

第一 mm 的输出逻辑名是 buf0。Scheduler 的最后使用信息表明：relu 读完 buf0 后，没有其他节点还要观察 relu 前的旧值；relu 又允许原地计算。因此 Q43 不为 relu 输出再写一次 `empty_strided_cpu`，而是生成：

```python
buf0 = empty_strided_cpu(...)          # 分配 S0
extern_kernels.mm(..., out=buf0)       # S0 保存第一 mm 结果
buf1 = buf0; del buf0                  # 新逻辑名接管 S0
cpp_fused_relu_0(buf1)                 # 在 S0 上原地写 relu
```

这四行把决定链连起来了：旧值在 relu 处最后使用 → 允许覆盖 → buf1 接管 S0 → buf0 这个旧逻辑名结束。`del buf0` 不释放 S0，因为 buf1 仍指向它。

如果函数还要额外返回 relu 前的第一 mm 结果，旧值在 relu 之后仍可观察，前述覆盖就不合法。wrapper 必须让 relu 写另一块 storage，例如分配 buf1、让 relu 读取 buf0 并写 buf1，最后分别返回旧的 buf0 和新的 buf1。若额外返回的是 relu 后的 buf1，则 relu 仍可接管 S0，但 S0 必须一直保留到函数返回，不能再交给其他覆盖操作。

第二个 mm 同时读取 S0 和写 S1，所以两块 storage 必须并存。逻辑名变化而 storage 不变叫复用，不是再分配。

这一变化不改 relu/sin 的 ops 数学，却改变 Q43 的生存期和复用安排；完整 wrapper 源码随之改变，Q44 得到的内容 key 和 callable 合同也随之改变。

## 外层 FX Graph Cache 怎样改变取得 callable 的路线

前面的四步描述一次完整 codegen-and-compile。更外层的 FX Graph Cache 可以在相同图与输入条件已有合法产物时缩短路线：

| 结果 | 本次怎样取得 callable |
| --- | --- |
| miss | 没有合法候选，执行完整生成、源码写入和加载，并保存以后恢复所需的信息 |
| hit | FX cache 候选条件通过，用条目保存的 wrapper `cache_key`、`source_code`、line map 等信息恢复 callable；源码路径由 key 重新推导，文件缺失时可由保存的源码重写 |
| bypass | 当前图不支持存入这层缓存，仍执行完整编译，但不形成普通 hit 候选 |

hit 缩短的是“取得 callable”的编译路径。这里的候选条件用于编译期选择哪份产物；真正
调用恢复出的 `module.call(args)` 时，空间安排、kernel/extern 调用和返回仍会执行；若该
wrapper 实际生成了 shape/stride 检查，它们也会执行。是否生成这类检查受配置和输入条件
影响，不能把每份 wrapper 都写成必有 `assert_size_stride`。FX cache 候选条件不满足时不能
拿旧 callable 强行运行；查找其他候选仍失败后才重新编译。

候选组织、动态符号 guards、缓存禁用、AOT 与无计算快速出口属于 Q44 的内部边界，在缓存与加载子篇展开。

## 改变条件后怎样判断四项职责

先假设主案例现有 callable 专用于长度 3，其 guards 不接受长度 19；改为 19 后需要重新编译。
若这次仍生成长度 19 的静态 specialization：

- Q41 的 `load→add→relu→mul→store` 关系不变；
- Q42 必须生成覆盖 19 个合法位置的循环或 block/mask；
- Q43 的输出分配大小改为 19；开启相应 size 检查时，wrapper guard 也改为 19；
- Q44 写入的是相应的新 wrapper source，并交出符合这份输入合同的 callable。

重新编译时也可能发生动态泛化：此时 Q42 的循环边界、Q43 的 guards 和输出大小使用符号
表达式，新源码可接受一组尺寸，而不只固定为 19。还有第三种情况：若此前已经存在 guards
接受 19 的符号候选，Q44 可以直接恢复并复用 callable，本次不会重新运行 Q41–Q43。
因此尺寸变化本身只能触发重新检查候选，不能推出“一定生成静态 19 源码”。

把混合案例的 relu 结果增加为第二个图输出：

- relu 的 Q41 ops 和 Q42 kernel 数学不变；
- Q43 延长 S0 的生存期，不能让后续工作覆盖它；
- wrapper 的返回结构和源码改变；
- Q44 加载后交出的 callable 必须返回两个结果。

这两项变化说明四项职责处理的是同一编译请求的不同部分：Q41/Q42 决定设备计算怎样表达，Q43 决定整张图怎样调用和保存结果，Q44 决定怎样把这份源码变成当前进程中的 callable。

## 本层出口

Codegen 不是在 Scheduler 返回后才接收整列节点。Scheduler 逐节点调用 backend 与 wrapper
生成接口，这些接口共同写出 kernel 和 wrapper 源码；缓存与加载流程再从完整源码取得一个
boxed callable。运行这个 callable 时，wrapper 检查输入合同，准备或复用 storage，按顺序
调用生成 kernel 和已有函数，最后按图接口返回结果。

下一层分别打开这四项职责：

- [虚拟化 ops 与 kernel 表达](01-kernel-ops/README.md)：LoopBody 如何 replay；
- [CPU C++ 与 GPU/Triton backend](02-backends/README.md)：不同执行模型怎样形成 kernel；
- [Wrapper 与内存计划](03-wrapper-memory/README.md)：guards、分配、复用、生存期和返回；
- [缓存、编译与加载](04-cache-load/README.md)：源码缓存、kernel 编译、FX Graph Cache 与 callable 恢复。
