# Codegen：把调度计划写成 kernel、wrapper 与可调用模块

[上一篇：Scheduler](../03-scheduler/README.md) · [返回总览](../README.md) · [深入：虚拟化 ops](01-kernel-ops/README.md) · [CPU C++ 与 GPU/Triton](02-backends/README.md) · [Wrapper 与内存计划](03-wrapper-memory/README.md) · [缓存、编译与加载](04-cache-load/README.md)

Codegen 的最终目标不是一段孤立 kernel，而是可被 PyTorch 调用的模块。它既要生成设备计算，也要检查输入、分配输出、处理 device、调用 kernel、管理生存期并返回约定结构。

先沿用总览中的 `relu(x+y)*2`。Scheduler 交来一条已经排好位置范围和读写关系的工作记录，其中保存的计算顺序是：读取两个输入、相加、relu、乘 2、写输出。源码把这份与具体 C++ 或 Triton 语法无关的计算体称为 `LoopBody`。

Codegen 会再次走过这串操作。走到 `load` 时，CPU 生成器返回 `in_ptr0[x0]` 这样的 C++ 表达式；走到 `add` 时返回前两个表达式的和；最后把结果放进 store 语句。同一串 `LoopBody` 换成 Triton 生成器时，会得到 Triton 的 load、mask 和 store。这种“用某个目标后端重新执行一遍统一操作记录”的过程，源码称为 replay（重放）；负责回答每项操作怎样写的对象称为 handler。

外围代码还会在运行时检查输入，例如本次代码只接受 shape `[3]`、stride `[1]` 的 float32 张量；条件不满足时不能直接使用这份代码。这类运行条件称为 guard。建立了这些具体对象后，再看四项职责怎样合作：

## 四项职责共同形成产物

```text
Scheduler node
   │
   ├─ Q41 kernel ops：按顺序取出 LoopBody 中的 load/add/reduction/store
   │         │
   │         v
   ├─ Q42 device backend：由 CPU、Triton 等 handler 把操作写成目标代码
   │         │ kernel 定义及调用它需要的参数
   │         v
   ├─ Q43 wrapper：检查输入、分配/复用内存、选择设备、调用 kernel/已有函数、释放、返回
   │         │ 完整模块源码
   │         v
   └─ Q44 cache/load：按源码内容保存结果，编译 C++/Triton 产物，加载 Python module
             │
             v
        接收参数列表的 call(args)
```

`GraphLowering.codegen()` 先初始化 wrapper，创建 Scheduler 并让其 codegen。普通循环节点先由 Q41 以统一 ops 语义交给 Q42；extern 节点可直接给 Q43 登记已有调用，template 由相应后端渲染。各 backend 在这个过程中向 wrapper 登记 kernel 定义和调用，Q43 最后用 `wrapper_code.generate(is_inference)` 拼成完整模块源码；Q44 才把源码变成可复用、可调用的模块。

## 同一 LoopBody 怎样从统一操作变成后端局部值

对 `relu(x+y)*2`，Scheduler node 交来的 LoopBody 不是已经写好的 C++：

```text
load(arg0,index0)
load(arg1,index0)
add
relu
constant(2)
mul
store(buf0,index0)
```

Q41 在索引变量 `x0` 下重放这串虚拟 ops。CPU handler 依次交回 `tmp0` 到 `tmp5`，所以后一步拿到的是前一步的后端表达式，而非 eager Tensor：`load→tmp0/tmp1`，`add→tmp2`，`relu→tmp3`，常数→tmp4，`mul→tmp5`，最后 store。Q42 再把这些表达式放进实际 C++ 循环。

把位置 `x0=0` 代入，两个 load 分别读取 -2、1；tmp2 表示 -1，tmp3 表示 relu 后的 0，tmp4 是常数 2，tmp5 为 0，store 写出 buf0[0]。换到 `x0=1`，同一操作序列把 1、4 变成 tmp2=5、tmp3=5、tmp5=10。LoopBody 保存的是这一串关系；tmp 名是当前 CPU handler 为表达式选择的局部身份。

规约例 `sum(x+1,dim=1)` 多一组 `x1`：load 索引为 `128*x0+x1`，add 产生当前项，`ops.reduction(...,'sum',value)` 更新 `tmp_acc0`，`store_reduction(buf0,x0,tmp_acc0)` 在一行完成后写出。当前 CPU 生成器会先用“一个数”的处理方式推导标量表达式，也会用“一组数”的处理方式推导向量表达式；源码分别称它们为 scalar handler 和 vector handler。标量方式产生单个 C++ 值，向量方式产生可同时容纳多项的向量值，并据此判断主循环和尾部怎样生成。

对 `x0=0`，x1=0、1、2 时 load 读取 0、1、2，add 后当前项为 1、2、3，累计状态依次是 1、3、6；x1 到 127 后为 `1+...+128=8256`，随后 store_reduction 写 buf0[0]。对 `x0=1`，load 从位置 128 开始，加入 1 后是 129..256，最终写 buf0[1]=24640。

采集中同一 reduction LoopBody 被重放三次：一次取得标量写法；两次取得向量写法，其中一次用于分析能否向量化，另一次用于生成所选向量循环。三次都走过 `load→constant→add→reduction→store_reduction`，但产生的值类型和用途不同。这是编译期反复询问“同一计算怎样写”，不表示运行时把相同规约执行三遍。

## CPU 案例：一个 kernel 内的具体数据流

贯穿实验实际生成：

```cpp
for (int64_t x0 = 0; x0 < 3; x0++) {
    tmp0 = in_ptr0[x0];
    tmp1 = in_ptr1[x0];
    tmp2 = tmp0 + tmp1;
    tmp3 = max(tmp2, 0);
    tmp5 = tmp3 * 2.0f;
    out_ptr0[x0] = tmp5;
}
```

对位置 0：读取 -2 与 1，`tmp2=-1`，`tmp3=0`，`tmp5=0`。位置 1 得到 `(1+4)×2=10`；位置 2 得到 `(3-1)×2=4`。局部 `tmp` 只活在当前迭代，不是三份完整中间 Tensor。

这段代码把三个更早阶段的决定具体化：Lowering 提供位置计算，Scheduler 把它们安排为一个生成单元，CPU backend 选择 C++ 循环及标量表达。

尺寸不整除向量宽度时，后端还要覆盖尾部。实验对 19 个 float32 元素生成并实际运行 CPU 代码：主循环一次处理 16 项，尾部向量处理剩余位置 16、17、18；结果与 eager 一致。尾部只允许读取和写回这 3 项，不能把候选向量中越界位置当成输入。

当前生成体的两个循环边界是：

```text
主循环：x0=0；条件 x0<16；步长 16；load/store 数量 16
尾循环：x0=16；条件 x0<19；步长 3；load/store 数量 3
```

输入是 -3..15。主循环覆盖位置 0..15；尾循环从 `in_ptr0+16` 只加载 3 项，即输入 13、14、15，经 `relu(value+1)*2` 得到 28、30、32，再只写 output 16、17、18。生成代码中的 `loadu(...,3)` 与 `store(...,3)` 是防止访问位置 19..31 的具体机制，不只是“另有一个尾循环”的名称。

当前环境没有 CUDA/Triton，因此 GPU 部分只按当前源码公式推演，不能写成实机结果。若 `xnumel=19`、`XBLOCK=8`：

| pid | 候选 `pid*8+[0..7]` | mask 为真的位置 |
| --- | --- | --- |
| 0 | 0..7 | 0..7 |
| 1 | 8..15 | 8..15 |
| 2 | 16..23 | 16、17、18 |

三个 program 合计恰好覆盖 0..18。CPU tail 和 Triton mask 解决同一个合法范围问题，但采用不同执行模型：CPU 在循环结构中处理尾部，Triton 为整个 block 计算候选索引，再用 `xindex<xnumel` 屏蔽越界 lane。

若把 xnumel 改成 17、XBLOCK 保持 8，program 数仍为 3，但最后一组只有位置 16 的 mask 为真；若改成 16，则只需两个 program，不产生第三组。这个推算来自当前 Triton 索引公式，没有在本机执行 Triton kernel。

## wrapper 怎样把 kernel 变成函数

实验 wrapper 的主线是：

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

逐步看状态：

| 时刻 | 已有对象 | 动作后的变化 |
| --- | --- | --- |
| 入口 | `args=[x,y]` | 解包，清空容器以释放引用 |
| guards 后 | 两输入 shape=(3,), stride=(1,) | 输入符合该编译版本 |
| 分配后 | 新 `buf0` | 有最终输出存储 |
| kernel 后 | `buf0=[0,10,4]` | 输入最后使用完可删除局部引用 |
| 返回 | `(buf0,)` | 符合 Inductor 单图 tuple 输出约定 |

动态形状时 guards 和尺寸表达会不同；输入 mutation、alias output、多个 device 或 extern call 也会改变 wrapper。共同点是 wrapper 落实图级调用约定，而 kernel 专注计算。

## 后端不是只有一种

Scheduler 按 node device 取得 `BaseScheduling` 实现。普通 CPU pointwise 可生成 C++；GPU 常由 Triton/SIMD scheduling 生成程序；template 节点还可把主计算后的逐元素操作接进同一个模板，例如 mm 的累加结果紧接 relu，而不先写完整 mm 输出再启动 relu kernel，这段后接计算称为 epilogue；extern node 生成已有库调用。

所以“Codegen”是统一职责，不代表所有算子经过同一种字符串模板。每种后端都必须接住 Scheduler 给出的分组、读写、布局和调用位置。

## 混合图：kernel、wrapper 和加载怎样共同交付结果

Q43 的实际案例执行 `mm → relu → mm → sin`。Codegen 在同一模块中产生两类计算：mm 保留为 `extern_kernels.mm(..., out=buf)` 调用，relu/sin 生成 C++ kernel。wrapper 按 Scheduler 顺序把它们连接起来：

```text
分配 buf0 → mm 写 buf0
buf1 接管 buf0 storage → relu 原地更新
分配 buf2 → 第二次 mm 读取 buf1、写 buf2
buf3 接管 buf2 storage → sin 原地更新
返回 buf3
```

可以把两次真实分配叫作 storage S0、S1，避免把四个 buf 名误认为四次分配：

| 语句后 | 逻辑名字到 storage | 活跃内容 |
| --- | --- | --- |
| `buf0=empty_strided` | buf0→S0 | 第一 mm 的输出空间 |
| 第一 mm | buf0→S0 | S0 写入第一 mm 结果；两个已最后使用的输入引用删除 |
| `buf1=buf0; del buf0` | buf1→S0 | 只转移逻辑名字，没有新分配 |
| relu kernel | buf1→S0 | 原地把 S0 改成 activated |
| `buf2=empty_strided` | buf1→S0，buf2→S1 | 第二 mm 同时需要输入 S0 和不同的输出 S1 |
| 第二 mm 后 | buf2→S1 | S1 写入结果；buf1 与最后一个 weight 引用删除 |
| `buf3=buf2; del buf2` | buf3→S1 | 再次只转移逻辑名字 |
| sin 与 return | buf3→S1 | 原地写 sin，tuple 返回 S1 |

`del buf1` 表示 wrapper 的这个 Python 引用在最后使用后移除，使存储可以按计划回收或复用；它不是一条“立即释放物理内存”的设备指令。相反，`buf1=buf0` 后删除 buf0 明确表示两个逻辑名字先后指向同一 S0。

第二次 mm 开始时 buf1 与 buf2 必须并存，因为一个是输入、一个是输出；relu 和 sin 则在实际生成中原地复用前一项 storage。模块加载完成后，调用者不需要分别调用四项工作，只调用 boxed `call(args)`。这个案例把三类产物连起来：backend 生成/引用计算，wrapper 实现图级时间线，compile/load 把整个时间线变成 callable。

| wrapper 时刻 | 活跃 storage | 变化 |
| --- | --- | --- |
| 第一 mm 后 | buf0 | mm 写新分配 |
| relu | buf1 与 buf0 是同一 storage | buf1 接管并原地更新，旧名字删除 |
| 第二 mm 期间 | buf1、buf2 | 输入仍要读，输出必须另分配，二者不能复用 |
| 第二 mm 后 | buf2 | buf1 最后使用结束，可删除 |
| sin | buf3 与 buf2 是同一 storage | buf3 接管并原地更新，最后返回 buf3 |

如果 buf1 还作为第二项图输出，第二次 mm 后不能删除或让后续覆盖它；它从产生到最后使用的生存区间必须延长到函数返回，负责安排内存复用的代码也必须取消与这段区间重叠的复用。源码分别把这两个概念称为 live range 和 memory planner。数值相同不构成复用许可，关键是旧值是否仍有消费者。

这里仍可让 buf1 接管 buf0，因为用户没有要求观察 relu 前的第一 mm 结果；不能做的是在第二 mm 之后把 S0 交给其他会覆盖它的结果。若额外返回的是 relu 前的 first，而不是 activated，那么 relu 本身也不能在 S0 上原地覆盖，必须为两个可观察值保留不同 storage。

另一条缓存案例在两个独立进程中使用同一临时缓存：第一次 FX Graph Cache miss 并生成上述类型的模块，第二次 hit 后从已保存的产物恢复 `module.call`。这里的产物包括 wrapper Python 源码及其 cache key/path、输入输出信息和 shape 检查表达式；C++ kernel 另有对应编译缓存。源码把这类可保存、可恢复的结果统称为 artifact。缓存改变的是获得 callable 的路径，不改变 callable 内的输入检查、分配和实际计算。

若第二次输入的 dtype、stride、device、相关配置或符号 guard 与候选不相容，不能仅因 FX 节点文本相同就复用。查找应继续尝试其他合法候选，或 miss 后重新编译；源码明确不支持缓存的图则走 bypass。三种结果的含义分别是“合法复用”“没有合法候选而生成”“本次不参与该缓存”，都不等于运行时跳过 wrapper 和实际计算。

## 代码怎样被写入和加载

`GraphLowering._compile_to_module()`：

1. 调用 `codegen()` 得到完整源码和 source line map；后者记录“生成文件的某一行来自哪个 FX 节点或用户源码位置”，加载模块和报错映射会读取它；
2. 用 `PyCodeCache.write(code)` 根据内容写入缓存路径；
3. 用 `PyCodeCache.load_by_key_path(...)` 加载模块，并附上 constants；
4. 保存 cache key/path/linemap；
5. 编译入口最终返回可 boxed 调用的函数。

这是一次完整编译内部的 source/module cache：生成出的同一份完整 Python wrapper 源码决定 `PyCodeCache` 的内容 key；`load_by_key_path` 执行模块顶层、等待其中的异步 kernel 编译，并取得 `module.call`。它与更外层的 FX Graph Cache 不是同一张表。

外围的 `FxGraphCache` 还可能在 `_compile_fx_inner` 避免重复 codegen-and-compile。缓存命中条件、guards 与动态形状复用将在 Q44 单独展开。

首次 miss 与后续 hit 的状态交接不同，但出口相同：

```text
miss：Scheduler/Codegen → wrapper source → 内容 key/path
      → 加载模块顶层并编译/等待 kernel → module.call

hit：FX cache candidate + guards 通过 → wrapper cache key/source
     → PyCodeCache 加载模块 → 恢复 current_callable=module.call

共同运行出口：boxed call(args) → guards → 分配/复用 → kernel/extern → tuple
```

三种 FX Graph Cache 结果要分开：

| 结果 | 当前实现做了什么 | 是否进入本次完整 codegen |
| --- | --- | --- |
| miss | 没找到 guard 合法的候选；调用 `fx_codegen_and_compile`，保存 CompiledFxGraph 元数据 | 是 |
| hit | 从同一 graph hash 的候选中找到 guards 通过者；用保存的 source/cache key 经 PyCodeCache 恢复 `current_callable` | 否 |
| bypass | 图包含该缓存明确不支持的对象；记录 bypass 原因，再直接调用 compile callback | 是，但不保存为普通 hit 候选 |

实验在独占临时 cache 的两个独立进程中得到 `miss=1,hit=0,bypass=0`，随后 `miss=0,hit=1,bypass=0`，两次结果和都为 272。第三个独立进程编译含 `torch.cond` 的图；`cond` 属于把两张子图作为参数的高阶算子（源码基类名为 `HigherOrderOperator`），当前缓存检查拒绝这类对象并走 bypass，counter 为 `bypass=1`，结果 `[3,5]`。这证明的是当前三条控制路径，不用耗时猜测是否命中。

hit 也不是“只看 FX 文本相同”。一个 graph hash 目录可以有多份带不同输入条件的候选；例如动态长度记作符号 s0，本次输入长度 19 就提供 hint `s0=19`，查找时把 19 代入每份候选的条件求值。没有任何候选通过就是 miss。若候选通过但对应 PyCodeCache 产物恰好缺失且恢复失败，当前实现把它当 miss 重新编译。

还要区分“没有启用这层 cache”和“bypass”：强制禁用缓存、配置关闭，或者正在生成可保存并独立部署的 AOT 产物时，会在 `_compile_fx_inner` 外围直接走 `codegen_and_compile`，没有进入 `FxGraphCache.load`，因而不属于其 bypass counter。bypass 是已经尝试该缓存、但 `_check_can_cache` 明确拒绝当前图。

无计算节点且非 AOT mode 的图还有更短出口：`_compile_fx_inner` 可把 `gm.forward` 包成 boxed callable，不为它生成空 kernel。这个分支同样满足“交回 callable”，但不能用来证明普通计算图经过了 kernel codegen。

这个 `count_calls==0` 检查发生在 FX Graph Cache 选择之前；因此它既不是 hit、miss，也不是 bypass。AOT mode 明确排除该快速返回，因为 AOT 输出必须包含可独立部署时使用的生成源码、输入输出信息和相应加载入口，不能只返回当前 Python 进程里的临时 callable。

## 用变化检查四项职责

若把 19 元素输入改为 16，Q41 的逐位置语义不变，Q42 不再需要 3 元素尾部；Q43 的输出大小与 guard 改为 16；Q44 的输入/生成相关信息必须让不兼容旧产物的情况重新选择候选。改变尺寸并非只影响一个 `for` 上界。

若在 `mm→relu→mm→sin` 中把第一次 relu 结果也返回，Q41/Q42 的 relu 数学不变；Q43 因旧值仍活跃而不能让后续覆盖其 storage；生成源码改变又会产生相应内容 key，Q44 恢复的 callable 必须是这份新合同。由此可以区分“kernel 算什么”“storage 何时可复用”和“取得哪份 callable”三个问题。

## 本层出口

最终调用者得到的是一个接收参数列表的 compiled callable。它运行时先落实输入合同，再依照已生成的 storage 时间线调用 backend/extern，最后按图接口返回结果。贯穿实验实际验证连续 CPU float32；GPU 与动态形状的目标代码在子篇按当前源码边界解释。

至此四个组件连起来了：FX 图优化改变“要解释哪些节点”；Lowering 给出“这些节点代表怎样的计算和存储”；Scheduler 决定“哪些工作按什么顺序组成生成单元”；Codegen 把计划变成设备代码和可调用 wrapper。

本层内部从[虚拟化 ops 与 kernel 表达](01-kernel-ops/README.md)开始，再比较[CPU C++ 与 GPU/Triton backend](02-backends/README.md)，随后展开[wrapper 与内存计划](03-wrapper-memory/README.md)以及[缓存、编译与加载](04-cache-load/README.md)。
