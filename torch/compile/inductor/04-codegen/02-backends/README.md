# CPU C++ 与 GPU/Triton：同一逻辑位置怎样分给循环和 program

[上一项：虚拟化 `ops`](../01-kernel-ops/README.md) · [返回 Codegen](../README.md) · 下一项：wrapper 与内存计划

Codegen 后端接到 Scheduler 分好的节点和其中的 `LoopBody`。例如长度为 `N` 的逐元素计算保存了这串位置级操作：

```text
tmp0 = load(input, x0)
tmp1 = add(tmp0, 1)
tmp2 = relu(tmp1)
tmp3 = mul(tmp2, 2)
store(output, x0, tmp3)
```

它还没有规定谁来处理 `x0=0..N-1`。CPU 后端生成循环，让一个 CPU 线程在每一步处理一个或一组连续位置；Triton 后端生成 GPU program，让每个 program 取得一块候选位置，再用 mask 去掉超出逻辑范围的 lane。

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

## CPU 用循环步长切分 19 个位置

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

## Triton 用 grid 启动 program，再在 program 内生成 block

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

如果调度器同时保留 persistent 与 non-persistent 两种候选，`MultiKernel` 会包装这些已经生成的 kernel。这个生成物尚无已缓存选择时，运行调用会对候选做基准测试、选出更快者，并把选择写入该 multi-kernel 对应的选择缓存；在同一缓存条目所代表的编译专门化条件下，以后可直接调用赢家。这里缓存的是候选 kernel 编号，不是按 tensor 中的实际数值重新选择；尺寸、设备或生成配置改变并形成另一编译/缓存条目时，本段不保证沿用原选择。它不改变上面的求和语义。

`DisableReduction/EnableReduction` 是 node schedule 中供 codegen 消费的状态标记，不是新的数学算子，也不会成为设备端执行的一条指令。它们只在 schedule 需要暂时关闭当前规约循环、生成位于 r 循环外层的 pointwise 后缀时成对包住这段区域；不是每个规约都必然出现。若一个融合 schedule 包含“规约求和→结果乘 2”，这段生成区域可以记成：

```text
在 reduction 区域生成累计
→ DisableReduction：关闭 r 循环，最终和现在可用
→ 在 pointwise 区域生成 result*2
→ EnableReduction：关闭这个临时上下文，恢复后续节点所需的规约生成状态
```

基础规约仍是“建立累计状态→遍历完整 r 域→横向归并→写一次结果”；这些机制只决定候选选择和后缀计算放在哪个生成区域。

## kernel 定义怎样接到 wrapper 调用

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

## 普通 LoopBody 之外的两个入口

上面的比较适用于普通 pointwise/reduction `LoopBody`。矩阵乘等操作可能进入 template：这条路径提供并实例化更专用的算法、block 和布局候选，编译与选择流程再结合实际尺寸和配置确定使用者。另一些 fallback/extern 节点直接调用已有库函数。Scheduler 仍负责把这些 kernel 或调用与其他节点排好顺序，但它们不需要把普通逐位置 LoopBody 翻译成上述 SIMD 循环。

这是一条选择阅读入口的导航，不是三种互斥节点类型的完整判别式：模板可能带可融合的 epilogue，extern 周围也仍有布局和调度处理。遇到新节点时，先找它最终交给后端的是普通位置规则、模板候选，还是已有调用合同，再沿相应部分分析。

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
