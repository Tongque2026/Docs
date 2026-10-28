# CPU C++ 与 GPU/Triton：同一逻辑位置怎样分给循环和 program

[上一项：虚拟化 `ops`](../01-kernel-ops/README.md) · [返回 Codegen](../README.md) · [下一篇：wrapper 与内存计划](../03-wrapper-memory/README.md)

Q41 和本篇观察同一次 backend 调用。backend 已经接到 Scheduler 分好的节点、其中调整后的
`LoopBody`、迭代范围和规约范围。CPU 路径先用标量 handler 重放，建立循环草稿，再尝试用
向量 handler 重放并替换合法区段；Triton 路径先建立 grid/block 空间，再在其中安装 handler
重放 `LoopBody`。两条路径都在重放过程中产生目标代码里的临时值。

例如长度为 `N` 的逐元素计算保存了这串位置级操作：

```text
tmp0 = load(input, x0)
tmp1 = add(tmp0, 1)
tmp2 = relu(tmp1)
tmp3 = mul(tmp2, 2)
store(output, x0, tmp3)
```

它还没有规定谁来处理 `x0=0..N-1`。本篇的四项职责是：

```text
Scheduler node + LoopBody + iteration/reduction ranges
  │
  ├─ 1. 建立逻辑域、tiling、索引和有效条件
  ├─ 2. CPU：选择标量/向量 loop，并在其中重放 LoopBody
  ├─ 3. Triton：选择 grid/block/mask，并在 program 中重放 LoopBody
  └─ 4. 登记 kernel 定义与调用合同，交给 Q43 排列 wrapper
```

CPU 和 Triton 平行接收同一类输入。CPU 后端生成循环，让一个
CPU 线程在每一步处理一个或一组连续位置；Triton 后端生成 GPU program，让每个 program
取得一块候选位置，再用 mask 去掉超出逻辑范围的 lane。

```text
同一逻辑域 0..N-1 与同一 LoopBody
              │
       ┌──────┴──────┐
       ▼             ▼
CPU：循环边界      Triton：启动多少个 program
     + SIMD 宽度            + 每个 program 的 block
       │             │
       ▼             ▼
覆盖 0..N-1       覆盖 0..N-1
```

下面始终追踪同一个问题：每个逻辑位置由谁负责，尾部怎样避免越界，位置上的 `load/add/relu/mul/store` 又变成什么目标表达式。

## 第一项：把逻辑范围变成后端执行域

先固定后文反复使用的对象：

| 对象 | 本篇含义 |
| --- | --- |
| iteration range / `numel` | pointwise 要覆盖的逻辑位置数；规约时是输出位置数 |
| reduction range / `rnumel` | 每个输出要累计的逻辑位置数 |
| index variable | 当前 loop 或 program 中代表逻辑位置的符号，例如 `x0`、`xindex`、`rindex` |
| tiling | 把逻辑轴分配给 CPU 向量维或 Triton 的 x/y/z/r block 维的方案 |
| vector width / block | 一次生成的候选 lane 数；例如 CPU 16 lane、Triton `XBLOCK=8` |
| grid | 要启动多少个 Triton program |
| mask | 候选 lane 是否对应真实逻辑位置，以及 masked/间接访问附带的有效条件 |

本篇说的“补齐”或 padding lane，是执行块为了整宽覆盖而多出的候选位置。例如 N=19、
XBLOCK=8 会产生候选位置 19..23，但 mask 禁止它们访存；tensor storage 仍只有原来的 19
个逻辑元素。layout 中的物理 padding 则体现在 stride/index expression 中，逻辑位置据此
映射到含空隙的物理地址。

## 第二项：CPU 用循环步长切分 19 个位置

CPU 后端先用标量 handler 重放 `LoopBody`，得到“一个位置怎样计算”的标量计算草稿，并建立循环边界。随后如果当前 CPU 指令集和 dtype 支持向量计算，它再用向量 handler 重放同一 `LoopBody`。向量 handler 的一个值包含多个 lane，例如 16 个 float32 值；这不是把已经生成的标量 C++ 文本再改写一遍。

循环结构可以先想成下面的可替换框架：

```text
loop 0..N:
    使用标量 kernel 处理当前位置
```

当向量宽度选为 16 后，后端把其中能够整块覆盖的区间换成向量 kernel。剩余位置再选择“只启用部分 lane 的尾向量”或标量 kernel：

```text
主段 0..15   → 一个 16-lane 向量步骤
尾段 16..18  → 一个只启用 3 个 lane 的尾向量步骤
```

当前 CPU 环境编译 `relu(x+1)*2`、float32 长度 19 时，生成了下面两个循环：

```cpp
for (int64_t x0 = 0; x0 < 16; x0 += 16) {
    auto v = Vectorized<float>::loadu(in_ptr0 + x0, 16);
    v = clamp_min(v + Vectorized<float>(1.0f), 0) * Vectorized<float>(2.0f);
    v.store(out_ptr0 + x0);
}
for (int64_t x0 = 16; x0 < 19; x0 += 3) {
    auto v = Vectorized<float>::loadu(in_ptr0 + x0, 3);
    v = clamp_min(v + Vectorized<float>(1.0f), 0) * Vectorized<float>(2.0f);
    v.store(out_ptr0 + x0, 3);
}
```

第二个循环中的长度 3 告诉向量访存只读取、写回前三个有效 lane，所以它覆盖 16、17、18，不碰 19 以后的地址。当前配置允许这种尾向量；不允许时，同一尾段会改用三个标量步骤。因此 16+3 是这个环境、dtype 和配置下的生成结果，不是所有 CPU 的固定切法。

生成过程中用过标量 kernel，也不表示最终文件必须保留标量循环。本例主段和尾段都被向量版本接管，最终只打印上面的两个循环。若循环范围足够大，外层还可用多个 CPU 线程分段；线程分段与 SIMD 是两层工作：线程领取一段区间，每个线程内部再以多个 lane 同时处理连续元素。

## 第三项：Triton 用 grid 启动 program，再在 program 内生成 block

Triton 先选定 launch 配置中的 block 参数，grid 函数再据此计算 program 数量；kernel 内部由每个 program 计算自己的候选位置。对长度 19，先固定一个便于推演的 `XBLOCK=8`：

```text
program 数量 = ceil_div(19, 8) = 3
grid = (3,)
```

于是 GPU 会启动 `program_id` 为 0、1、2 的三个 program。每个 program 中：

```python
xoffset = tl.program_id(0) * XBLOCK
lane = tl.arange(0, XBLOCK)       # [0,1,2,3,4,5,6,7]
xindex = xoffset + lane
xmask = xindex < xnumel
```

代入三个 program：

| `program_id` | `xoffset` | `xindex` 候选位置 | 有效 mask | 实际位置 |
| ---: | ---: | --- | --- | --- |
| 0 | 0 | 0..7 | 8 个 true | 0..7 |
| 1 | 8 | 8..15 | 8 个 true | 8..15 |
| 2 | 16 | 16..23 | 前 3 个 true | 16,17,18 |

kernel 用同一个 mask 保护输入读取和输出写回：

```python
tmp0 = tl.load(in_ptr0 + xindex, xmask)
tmp1 = tmp0 + 1.0
tmp2 = tl.maximum(tmp1, 0.0)
tmp3 = tmp2 * 2.0
tl.store(out_ptr0 + xindex, tmp3, xmask)
```

三个 program 的有效位置合起来正好是 0..18。`tl.arange` 产生的是一个 block tensor，表中的 8 个元素是八个候选 lane；Python 级 Triton 源码没有规定“一个 lane 对应一个 CUDA thread”，具体线程映射由 Triton 编译器完成。

这里的 8 是为了把公式算清楚而固定的示例值。实际生成物由当前 size hints、启发式配置或 autotune 选择 block 参数；选定 `XBLOCK` 后，grid 再由 `ceil_div(xnumel, XBLOCK)` 得到 program 数量。若生成时能够证明所有 lane 都有效，某些 mask 可以省掉。

## 同一串 ops 在两端分别产生什么

以刚才的一个位置或一个 block 为单位，对应关系是：

| `LoopBody` 操作 | CPU 标量/向量结果 | Triton block-tensor 结果 |
| --- | --- | --- |
| `load(input,x0)` | `in_ptr0[x0]` 或 `Vectorized::loadu` | `tl.load(in_ptr0+xindex, xmask)` |
| `add(value,1)` | 标量加法或向量加法 | `tmp0 + 1.0` |
| `relu(value)` | `std::max` 或向量 `clamp_min` | `tl.maximum(tmp1,0.0)` |
| `mul(value,2)` | 标量乘法或向量乘法 | `tmp2 * 2.0` |
| `store(output,x0,value)` | 标量赋值或向量 `store` | `tl.store(...,tmp3,xmask)` |

`LoopBody` 决定值流仍然是 load→add→relu→mul→store；CPU/Triton handler 决定每个值是标量、CPU 向量还是 Triton block tensor；CPU loop nest 或 Triton grid 决定这串计算覆盖哪些位置。在本篇的分析顺序中，先确定哪些节点共享一个 kernel，再看目标后端怎样表达和切分其中的位置。

## 规约还要安排组内位置和累计状态

现在令 `input=torch.arange(256, dtype=torch.float32).reshape(2,128)`，计算形状 2×128 的 `(input+1).sum(dim=1)`：

```text
x0 = 输出行，取 0、1
r0 = 当前行内的位置，取 0..127
value = input[128*x0+r0] + 1
output[x0] = 这一行 128 个 value 的总和
```

### CPU：八个向量步骤汇入一个向量累计器

仍假设 float32 向量宽度为 16。对 `x0=0`，运行时状态如下：

```text
进入第 0 行：acc_vec = [0,0,...,0]                 # 16 个 lane
r0=0..15：   vec = input[0..15] + 1；acc_vec += vec
r0=16..31：  vec = input[16..31] + 1；acc_vec += vec
...
r0=112..127：vec = input[112..127] + 1；acc_vec += vec
八个向量步骤结束：横向相加 acc_vec 的 16 个 lane → 标量 8256
写回 output[0] = 8256
```

每个 lane 累加了相隔 16 的八个值；八次向量更新全部完成后，`vec_reduce_all` 才把 16 个 lane 横向合成一个标量。随后 `x0=1` 建立新的零累计器，读取输入 128..255，最终写 `output[1]=24640`。因此输出行循环、组内循环、横向归并和一次写回各有明确位置。

### Triton：固定一个分块案例观察跨块累计

为了只看分块关系，固定 `XBLOCK=1、RBLOCK=64`。grid 在 x 轴启动两个 program：pid 0 负责输出行 0，pid 1 负责输出行 1。每个 program 的规约范围分成两块：0..63 和 64..127。

在这里固定 `XBLOCK=1`，所以生成的累计对象可以画成与规约 block 对应的一维 64-lane 值。一般 `XBLOCK>1` 时，每个 x 候选位置都有一组 r lanes，累计对象还会带 x 方向的 block 维度。对 pid 0：

```text
xindex = 0
acc_lane[0..63] = 0

roffset=0：
  rindex = 0..63
  对每个 lane j：acc_lane[j] += input[j] + 1

roffset=64：
  rindex = 64..127
  对每个 lane j：acc_lane[j] += input[64+j] + 1

两个规约块完成：result = sum(acc_lane[0..63]) = 8256
写回 output[0]
```

pid 1 使用同样的两个 `rindex` 块，但输入地址加上 `128*xindex`，因此读取 128..255 并写 `output[1]=24640`。若规约长度改成 130，第三个块的候选 `rindex` 是 128..191；`rmask=rindex<130` 只让 lane 0、1 把位置 128、129 加进各自的 `acc_lane`，其余 lane 加求和单位值 0。三个块完成后才横向求和 64 个 lane并写回。

这张执行稿只展开当前源码中 non-persistent 分块规约的累计生命周期；真实 kernel 还会根据 tiling 形成具体的 tensor 形状和归并层级。核心关系是：program 内的累计状态跨过多个 `roffset` 块，一直保留到完整规约范围结束。

另一种 persistent 形态尝试让一个 reduction block 覆盖完整的当前 r 域，从而不再使用上面的多个 `roffset` 扫描块；无效 lane 用求和的单位值 0 填充，最后完成规约。两种形态都会保留累计状态，区别是一次覆盖完整 r 域，还是让同一累计对象跨多批 `roffset` 扫描继续更新。persistent 不保证任意大的规约都能塞进一个 block；哪种形态合法并被选中还受范围、资源限制和后端启发式影响，本篇不展开选择算法。

这些候选不是 Scheduler 预先交来的。Triton backend 先生成主 kernel；只有主 kernel 被判为
persistent、`multi_kernel` 配置开启且 schedule 不含 sort 时，backend codegen 才额外生成
non-persistent kernel，并用 `MultiKernel` 包装两者。这个生成物尚无已缓存选择时，运行调用
会对候选做基准测试、选出更快者，并把选择写入该 multi-kernel 对应的选择缓存；在同一缓存
条目所代表的编译专门化条件下，以后可直接调用赢家。这里缓存的是候选 kernel 编号，不是
按 tensor 中的实际数值重新选择；尺寸、设备或生成配置改变并形成另一编译/缓存条目时，本段
不保证沿用原选择。它不改变上面的求和语义。

`DisableReduction/EnableReduction` 是 node schedule 中供 codegen 消费的状态标记，不是新的数学算子，也不会成为设备端执行的一条指令。它们只在 schedule 需要暂时关闭当前规约循环、生成位于 r 循环外层的 pointwise 后缀时成对包住这段区域；不是每个规约都必然出现。若一个融合 schedule 包含“规约求和→结果乘 2”，这段生成区域可以记成：

```text
在 reduction 区域生成累计
→ DisableReduction：关闭 r 循环，最终和现在可用
→ 在 pointwise 区域生成 result*2
→ EnableReduction：关闭这个临时上下文，恢复后续节点所需的规约生成状态
```

基础规约仍是“建立累计状态→遍历完整 r 域→横向归并→写一次结果”；这些机制只决定候选选择和后缀计算放在哪个生成区域。

## 第四项：kernel 定义怎样接到 wrapper 调用

CPU kernel 可以简化成下面的合同：

```cpp
void kernel(const float* in_ptr0, float* out_ptr0) {
    // 上面的 16+3 循环
}
```

模块顶层用 `cpp_pybinding` 声明参数类型并编译这段函数。wrapper 调用：

```python
cpp_fused_add_mul_relu_0(arg0_1, buf0)
```

绑定层从输入 tensor `arg0_1` 取得只读数据指针传给 `in_ptr0`，从输出 tensor `buf0` 取得可写数据指针传给 `out_ptr0`。因此 Python 形参和 C++ 指针按位置一一对应。

这个指针合同还依赖本例已经固定的调用前提：tensor 是 float32，所以指针类型是 `float*`；19 个元素采用单位步长的连续扁平索引，所以地址可以写成 `in_ptr0+x0`。编译结果的专门化条件和调用 guard 负责保证 shape、stride、dtype、device 等前提仍成立，索引式负责把逻辑位置变成实际地址。`loadu` 中的 `u` 只表示无需按向量边界对齐，不表示可以忽略任意 tensor stride；若输入是转置或带步长的 slice，地址索引必须体现相应 layout。

Triton pointwise kernel 的代表性合同是：

```python
def triton_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # program_id、xindex、xmask 和计算体
```

wrapper 发起调用时提供 tensor pointer、`xnumel=19`、grid 和当前 CUDA stream；选中的 launch 配置提供 `XBLOCK`：

```text
selected XBLOCK = 8
grid = (ceil_div(19, 8),) = (3,)
stream = 当前 device 上用于排列这次工作的 CUDA stream
```

于是 grid 启动三个 program。wrapper 先进入输入 tensor 所在设备的 device context，再把这次 kernel launch 排入该设备的当前 stream；stream 约束的是这次 launch 与同一 stream 中其他工作的先后，不保证 kernel 内三个 program 按 pid 串行执行。device 和 stream 不是新的数学输入。真实生成的调用可能把静态 numel 内联或由 launcher/meta 参数提供 block 值；无论具体签名怎样，wrapper launch 都要共同提供四类信息：数据指针、逻辑范围、启动网格和执行上下文。

规约 kernel 也要形成同样的定义与调用合同，只是多了规约范围。前面的 2×128 CPU
kernel 接收输入和输出指针；函数体的一个外层输出循环迭代两次，每次完成八次向量累计、
横向求和及一次写回。前面的 Triton 规约 kernel 需要输入/输出指针、`xnumel=2`、
`rnumel=128`、`XBLOCK`、grid、device 和 stream。non-persistent 形态的 launch 配置还提供
`RBLOCK`，program 内用 `roffset` 循环按这个宽度扫描多批。persistent 形态则在 kernel 中
根据静态规约范围确定覆盖完整 r 域的 `RBLOCK`，wrapper 不再传这个 block 参数。

若 backend 生成 `MultiKernel`，它登记的调用入口会持有多个候选 kernel。第一次没有缓存
选择时，这个入口测量候选并调用胜者；以后相同选择缓存条目直接调用已选编号。无论选中
哪个候选，2×128 案例最终都只向两个有效输出位置各写一次。

## 普通 LoopBody 之外的两个入口

上面的比较适用于普通 pointwise/reduction `LoopBody`。矩阵乘等操作可能进入 template：这条路径提供并实例化更专用的算法、block 和布局候选，编译与选择流程再结合实际尺寸和配置确定使用者。另一些 fallback/extern 节点直接调用已有库函数。Scheduler 仍负责把这些 kernel 或调用与其他节点排好顺序，但它们不需要把普通逐位置 LoopBody 翻译成上述 SIMD 循环。

模板可能带可融合的 epilogue，extern 周围也仍有布局和调度处理。遇到新节点时，先看它
最终交给后端的是普通位置规则、模板候选还是已有调用合同，再沿相应入口分析。

## 融合、动态范围和特殊索引怎样改变执行几何

融合没有创造另一套位置系统。Q33 已经决定哪些节点共用一个 kernel，并排好其中的读写
顺序；backend 为这组节点建立一份 CPU loop nest 或 Triton program 结构，再依次重放各成员
的 `LoopBody`。若一个规约结果后面融合了逐元素计算，前文的
`DisableReduction/EnableReduction` 把后缀放在 r 循环完成以后，后缀仍使用同一个输出位置。

静态范围可直接写入生成代码；动态范围则作为 kernel 参数进入循环边界、grid 计算和 mask。
例如运行时才知道 `xnumel=N` 时，Triton 仍按
`ceil_div(N, XBLOCK)` 启动 program，并以 `xindex < N` 判断最后一块的有效 lane。生成的
wrapper 可以在 launch 前执行当前调用所需的 size/stride 断言；Dynamo guard 和 FX Graph
Cache guard 决定已有编译产物能否复用，这一缓存路径在 Q44 讲解。本篇从已经选定的编译
产物和本次范围参数开始。

同一位置还可能带第二个有效条件或非连续地址：

| 输入情况 | CPU 后端中的变化 | Triton 后端中的变化 |
| --- | --- | --- |
| 间接索引，例如先读取 `index[x]` 再访问 `data[index[x]]` | 生成两次读取；不能合法向量化时改用标量实现 | 先得到临时索引值，再用它构造地址；临时值继承产生它时的有效 mask |
| 带条件读取 | 在条件成立的路径读取，否则使用规定的替代值 | 把该条件与范围 mask 一起用于 `tl.load` |
| 非连续 stride | 地址式按 stride 计算；连续 `loadu` 不一定可用 | block 中每个 lane 仍按索引式得到地址，不再假定地址相邻 |

因此 mask 不只表示“最后一个 block 越界”。对
`masked(mask, body, other)` 一类位置规则，最终访存条件同时要求当前位置处于逻辑范围且
`mask` 为真。间接索引也不是先生成不受保护的地址再补 mask；产生索引临时值时已有的有效
条件会跟着它进入后续访问。

空范围会更早结束。静态 `numel=0` 的 pointwise loader 变成空操作，backend 没有位置可
重放，也不生成一次假的位置计算。静态规约长度为 0 时，IR 创建阶段已经把规约改写成
“在每个输出位置写单位值”的 pointwise 计算，例如空集合求和写 0；普通规约 backend
不会收到一个要求进入 `rnumel=0` 累计循环的节点。动态范围在运行时可能为 0 时，生成的
边界、grid 和 mask 必须让它没有有效位置。

## 后端可能换实现，也可能拒绝生成

“选择了 CPU 或 Triton backend”不保证每个快速实现都可用。CPU 后端先有标量
`LoopBody` 重放结果；若 dtype、索引式或操作不适合当前向量 handler，它可以保留标量循环，
只放弃 SIMD，而不是自动退回 eager。Triton 的合法配置也可能改用另一 tiling，或者在满足
前文门槛时生成 persistent/non-persistent 两个候选。

有些情况不能在本后端中表示，此时编译直接报错，而不是悄悄转到 CPU：同一个 SIMD kernel
中的节点无法形成兼容迭代组，缺少生成所需的 buffer，或者某个 store mode、dtype/操作没有
对应 handler，都会沿各自检查抛出异常。设备没有注册 Q34 所需的 backend 时同样无法开始
生成。是否事先把某个算子留作 extern/fallback 是更早的 Lowering 和调度决定；到了这里，
backend 只为自己收到的节点生成合法实现或报告失败。

当前环境实际运行了 N=19 的 CPU 代码并核对输出；Triton 的 grid、block、mask 和规约例只
依据当前源码与测试中的公式推演，因为当前环境没有可运行的 CUDA/Triton 设备。这个证据
可以说明索引和覆盖关系，不能说明真实设备会选中示例里的 block 参数或获得何种性能。

## 本篇交给 wrapper 的内容

backend 完成后，Q43 接到的是可调用内容及其调用要求：

| 交出内容 | 本例中的具体值 |
| --- | --- |
| kernel 定义或可调用入口 | CPU `cpp_fused_add_mul_relu_0`，或选定/包装后的 Triton kernel |
| tensor 参数顺序 | `arg0_1` 提供输入指针，`buf0` 提供输出指针 |
| 逻辑范围参数 | pointwise 的 `xnumel=19`；规约还带 `rnumel=128` |
| 启动信息 | CPU 调用不需要 grid；Triton 带 grid、block 配置、device 和 stream |
| 结果位置 | kernel 向已分配的 `buf0` 写入有效输出位置 |

Q43 直接使用已经确定的位置计算。Scheduler 驱动 codegen 写下 allocation/free 候选和
last-usage 顺序；Q43 据此形成内存计划，分配或复用 `buf0`，生成运行时 guard 和 device
context，按照既定顺序调用这些入口，在最后一次使用后释放可复用存储，并把用户可见结果
返回给 Python。

## 用两个新尺寸检查切分

第一题：`N=21`，假设 CPU 向量宽度为 8、允许尾向量；Triton 选择 `XBLOCK=16`。

```text
CPU：主段 0..15，共两个 8-lane 步骤；尾段 16..20，启用 5 个 lane。
Triton：grid=ceil_div(21,16)=2；pid 0 覆盖 0..15，pid 1 只有 16..20 有效。
```

第二题换一个正文没有展开过的双轴尾部：形状 3×130 的按行求和，Triton 固定 `XBLOCK=2、RBLOCK=64`。

```text
x 轴 grid=ceil_div(3,2)=2。
pid 0 的两个 x lane 对应行 0、1，均有效；pid 1 的两个 x lane 对应行 2、3，只有行 2 有效。
每个有效 x lane 都有 64 个 r 累计 lane，并依次处理 0..63、64..127、128..191 三个候选块。
第三个 r block 只有 r=128、129 有效。
完整 r 域结束后，三个有效输出行各写一次，无效的行 3 不读取也不写回。
```

两个轴的 mask 通过广播组合，而不是把无效 lane 数简单相加。对 `pid=1、roffset=128`：

```text
xindex = 1*2 + [0,1] = [2,3]                    # 形状 [2,1]
rindex = 128 + [0,1,...,63]                     # 形状 [1,64]
valid[x,r] = (xindex < 3) AND (rindex < 130)    # 广播成 [2,64]
address[x,r] = in_ptr0 + 130*xindex + rindex
```

这 128 个二维候选中，只有 `(行2,r128)`、`(行2,r129)` 两项有效；行 3 的 64 项全部无效，行 2 的其余 62 个 r 候选也无效。

再判断一个布局变化：若输入是 `base[::2]`，逻辑位置 `x0` 的物理地址不再是 `in_ptr0+x0`，而是 `in_ptr0+2*x0`。改变的是索引合同；逻辑域和上面的 CPU/Triton 切分问题仍然存在，但后端未必还能采用前文连续 `loadu(in_ptr0+x0)` 的指令形态，它可能选择另一种向量访存或标量方案。

读完本篇，应能从 `N`、向量宽度、block 参数和 stride 写出每个步骤的候选位置、有效 mask、实际地址、累计器生命周期和写回次数，而不是只记住 CPU 和 Triton 的类名。

## 继续阅读实现

[CPU/Triton backend 源码：索引、访存、kernel 与向量尾部](01-backend-source/README.md)
