# CPU C++ 与 GPU/Triton：同一串计算怎样分给 CPU 循环和 GPU program

[上一项：虚拟化 `ops`](../01-kernel-ops/README.md) · [返回 Codegen](../README.md) · 下一项：wrapper 与内存计划

先抓住共同输入：两种后端接到的都不是原始 `torch.relu`，而是 Scheduler 分组后的节点及其中的 `LoopBody`。`LoopBody` 保存同一串逐位置操作，例如：

```text
tmp0 = load(input, x0)
tmp1 = add(tmp0, 1)
tmp2 = relu(tmp1)
tmp3 = mul(tmp2, 2)
store(output, x0, tmp3)
```

后端的核心差异不是“算式不同”，而是怎样把逻辑位置 `x0=0..N-1` 分给执行资源：CPU 生成普通/并行循环并用 SIMD 一次处理若干连续位置；Triton 生成许多 program，每个 program 用一个 block tensor 处理一块位置。

```text
Scheduler node + LoopBody
          │
          ├─ CppScheduling
          │    └─ CppKernelProxy → CppKernel / CppVecKernel → C++ loops
          │
          └─ SIMDScheduling
               └─ TritonKernel → program_id / block / mask → Triton program
```

## CPU：先生成单个元素的写法，再选择向量循环

`CppScheduling.codegen_node()` 取得节点后交给 `CppKernelProxy.codegen_nodes()`。proxy 将每个 SchedulerNode 包装成可用索引变量重放的函数，再由 `codegen_functions()` 完成两步：

1. 用 `CppKernel` 重放，建立可作为通用实现的标量底稿和完整 loop nest。
2. 若当前 CPU 提供可一次处理多个数的机器指令，再用 `CppVecKernel` 重放相同 LoopBody。源码把 CPU 提供的这组机器指令能力称为 ISA（instruction set architecture）；它与数据类型共同决定一次能处理几个元素。随后选择分块轴和宽度，把循环拆成向量主段与尾段。

这不是先产生标量 C++、再对文本做向量化。相同 `ops.load/add/store` 被不同 handler 解释一次，返回标量或向量 `CppCSEVariable`。

“先构造标量底稿”也不等于最终源文件一定保留标量循环。vector kernel 合法时，loop nest 的主段会改用 vector kernel；尾段再按配置和 dtype 能力选择 masked vector 或标量底稿。最终输出只打印选中的 loop 实现。

### 长度 19 的真实生成结果

独立实验实际编译 `relu(x+1)*2`，输入有 19 个 float32。当前机器选择 16 元素向量宽度，生成两个循环：

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

第一轮处理位置 0 到 15；尾部 masked vector 只装载并写回位置 16、17、18。源码中 `enable_loop_tail_vec` 和 dtype 能力决定是否使用这种 masked vector；条件不满足时尾段可以退回标量 kernel。因此“CPU 尾部必定是标量”并不成立。

本例最终文件只有上述两个 vector 循环，看不到构造过程中用过的 scalar kernel。这正是“标量底稿被 vector 主段和 vector 尾段替换”的具体结果。

更大的外层范围还可由 loop nest 决定 OpenMP 并行深度。这里要区分两层并行：线程把循环区间分段，SIMD 在每个循环步中同时计算多个元素。

## Triton：program 取得一块索引，mask 屏蔽越界 lane

`SIMDScheduling.codegen_node()` 先从融合节点取 `numel/rnumel`，用 `generate_node_schedule()` 排列规约内部和外部节点，再让 `TritonKernel` 重放 LoopBody。pointwise 的典型索引骨架来自当前 `TritonKernel` 源码，也出现在当前测试中：

```python
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)
xmask = xindex < xnumel
x0 = xindex
value = tl.load(in_ptr0 + x0, xmask)
# 与 CPU 相同的 add、relu、mul 语义
tl.store(out_ptr0 + x0, result, xmask)
```

以 `xnumel=19, XBLOCK=8` 独立代入公式：

| program id | 候选位置 | `xmask` | 实际位置 |
| ---: | --- | --- | --- |
| 0 | 0..7 | 8 个 true | 0..7 |
| 1 | 8..15 | 8 个 true | 8..15 |
| 2 | 16..23 | true,true,true,false×5 | 16,17,18 |

三个 program 合起来恰好覆盖 0..18，且不会访问 19..23。这里的 `XBLOCK=8` 为解释公式而固定；真实 Inductor 会根据 size hints、heuristic/autotune 配置选择 block 参数，某些能静态证明整除或尺寸为 1 的情形还可省略 mask。

本环境没有 CUDA 与 Triton 包，所以上表不是 GPU 运行结果，而是对当前源码和当前测试所用公式的独立求值。它证明索引覆盖关系，不证明某块 GPU 上最终选出的 `XBLOCK` 或性能。

## 同一 ops 怎样落成不同目标表达

| LoopBody 语义 | CPU C++ | Triton |
| --- | --- | --- |
| `load(name,index)` | 指针下标，或 `Vectorized::loadu` | `tl.load(ptr + index, mask, ...)` |
| `add/relu/mul` | 标量 C++ 或 `at::vec` 表达 | Triton tensor 表达 |
| `store` | 标量赋值或向量 `store` | `tl.store(ptr + index, value, mask)` |
| 循环位置 | C++ induction variable | `program_id*BLOCK + arange` |
| 越界尾部 | scalar/masked-vector tail | lane mask |

公共 `CSEProxy` 仍负责临时表达式复用、bounds 和 store cache；具体 kernel 的 override 与 load/store/reduction 方法生成目标表达。于是“同一 LoopBody”不表示目标代码逐字相似，只表示它们实现同一位置级数据流。

## 规约：输出轴 `x` 与规约轴 `r`

对形状 2×128 的 `(input+1).sum(dim=1)`，共同逻辑是：

```text
x0 in [0,2)       # 选择输出行
r0 in [0,128)     # 遍历该行
value = load(input, 128*x0+r0) + 1
output[x0] = sum_r(value)
```

Q41 的 CPU 实验实际生成外层 `x0` 循环、内层宽 16 的 `r0` 向量循环，再用 `vec_reduce_all` 横向归并向量累加器。

Triton 把范围树分成 pointwise 轴和 reduction 轴。典型非持久规约建立 `rbase=tl.arange(0,RBLOCK)`，按 `roffset` 分块遍历规约范围，用 `rmask=rindex<rnumel` 屏蔽末块，累积后再做 `tl.sum(..., dim)`；持久规约在当前启发式允许时构造覆盖规约范围的 block、用默认值替换无效 lane，再执行最终规约。是否持久化取决于规约 hint、静态范围、配置以及 multi-kernel 条件，并非所有 `sum` 固定走同一形态；启用 multi-kernel 时，当前调度还可能同时生成非持久候选供后续选择。

`generate_node_schedule()` 还可能插入 `DisableReduction/EnableReduction`：规约结果只有规约循环结束后才可供依赖它的后缀 pointwise 使用。这说明 GPU codegen 不只是翻译一条 `ops.reduction`，还要安排值在哪个循环区域产生。

## kernel 定义与调用约定也不同

CPU 产物通过 `async_compile.cpp_pybinding([...], source)` 编译成可从 Python 调用的函数；wrapper 直接传入 tensor data pointer 对应的 tensor 参数。19 元素实验中调用为：

```python
cpp_fused_add_mul_relu_0(arg0_1, buf0)
```

Triton kernel 定义除输入输出 pointer 外，还可能带 `xnumel/rnumel` 和 `XBLOCK/RBLOCK: tl.constexpr`。`TritonKernel.call_kernel()` 收集实参和 numel，生成默认 grid，再要求 wrapper 以当前 device、grid、stream 相关约定发起 Triton kernel 调用。block 参数和 grid 因而是调用协议的一部分，不只是 kernel 内部局部变量。

模板 kernel 和 extern kernel 不属于这条普通 LoopBody 对照线。矩阵乘可能由 Triton/CPP template 选择具体算法，fallback/extern 则直接调用已有实现；它们仍由 Scheduler 在 backend 边界排序，但不必经过上述 pointwise SIMD 生成过程。

## 三个容易混淆的判断

1. **融合不等于向量化。** 融合决定哪些节点共享 kernel；CPU SIMD 或 Triton block 决定一个 kernel 内怎样并行执行位置。
2. **block 不等于 CUDA thread 的逐一映射。** `tl.arange` 产生 block tensor，Triton 编译器再把它映射到底层线程；Python 级源码没有声明“lane i 就是 thread i”。
3. **mask 不改变计算域。** 逻辑域仍是 0..N-1；mask 只是让物理 block 的补齐 lane 不执行越界访存。

据此读 backend 代码时，先写出 LoopBody 的逻辑索引和操作，再找后端如何构造循环位置、尾部条件及规约轴，最后才看具体语法。这样 CPU 的 `x0+=16` 与 Triton 的 `pid*XBLOCK+arange` 会成为同一逻辑域的两种切分，而不是两套无关实现。

## 继续阅读实现

[CPU/Triton backend 源码：索引、访存、kernel 与向量尾部](01-backend-source/README.md)
