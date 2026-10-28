# Scheduler 节点与依赖：谁必须等谁完成

[上一层：Scheduler](../README.md) · [下一篇：排序、mutation 版本与生存期](../02-order-lifetime/README.md) · [返回总览](../../README.md)

Lowering 已经登记了一组 IR operation。每项 operation 说明一项运行时工作，例如“按循环
计算 buf0”或“调用 mm，写出 buf1”。Scheduler 接到这些工作后，先建立一张依赖图：

```text
Lowering operations
        │
        ▼
1. 包成 scheduler nodes，并取得每项工作的读写清单
        │
        ▼
2. 按 buffer 名登记生产者、使用者和未满足的读取
        │
        ▼
3. 补上共享 storage 与 mutation 的顺序和版本关系
        │
        ▼
4. 补符号依赖，并接住 ReadWrites 已携带的 effect 顺序
        │
        ▼
scheduler nodes + ReadWrites + users + unmet dependencies + 版本表
```

这些状态回答“谁必须等谁”，还没有决定最终执行次序。下一篇才据此拓扑排序、删除无用工作
并计算生存期；融合篇还会继续使用具体的访问位置。

## 第一步：把 operation 包成带调度状态的节点

IR operation 保存怎样计算。Scheduler 还要为它保存未满足的读取、输出使用者、是否已经
生成等全局安排状态，所以每项 operation 都再包成一个 scheduler node：

| Lowering operation | Scheduler 包装 | 怎样取得读写 |
| --- | --- | --- |
| `ComputedBuffer`、`TemplateBuffer` | `SchedulerNode` | 重放普通循环；模板调用自己的提取接口 |
| `ExternKernel` | `ExternKernelSchedulerNode` | 读取 operation 已保存的整块读写关系 |
| no-op operation | `NopKernelSchedulerNode` | 读取 operation 已保存的读写关系 |

上一篇的 `FallbackKernel`、`Conditional`、`WhileLoop` 和 `MultiOutput` 都属于
`ExternKernel` 这一族，因此每项 operation 各自得到一个 `ExternKernelSchedulerNode`。
父 operation 表示一次调用或控制流；`MultiOutput` 只表示从父结果中取一项，不会再次执行
父 operation。

no-op operation 不需要普通 kernel，但仍可能携带必须保留的关系。类型判断先检查 no-op，
再检查普通、模板和 extern 类型。operation 没有关联的 FX 来源时会触发断言；不属于已支持
类型时会抛 `NotImplementedError`，Scheduler 不会猜测怎样包装它。

### ReadWrites 是这项工作的读写清单

每个 scheduler node 都有一份 `ReadWrites`。其中最重要的两组是：

```text
reads  = 这项工作读取哪些 buffer，以什么方式读取
writes = 这项工作写出哪些 buffer，以什么方式写入
```

对 `ComputedBuffer`，`extract_read_writes()` 用符号循环变量重放计算体。重放中的 load/store
不读取真实数据，只登记 buffer 名、线性索引和变量范围。以
`(x+1).sum(dim=1)*2`、`x.shape=[2,3]` 为例，Lowering 已把整个表达式放进一个规约 operation：

```text
读 arg0_1[3*c0]、arg0_1[3*c0+1]、arg0_1[3*c0+2]，c0 ∈ [0,2)
写 buf0[c0]，c0 ∈ [0,2)
```

`c0=0` 时读取位置 0、1、2，得到 `(0+1 + 1+1 + 2+1)×2=12`，写 `buf0[0]`；
`c0=1` 时读取位置 3、4、5，得到 30，写 `buf0[1]`。这种知道具体访问位置的记录叫
`MemoryDep`。

### 三种记录表达不同精度的关系

| 记录 | 保存什么 | 本例用途 |
| --- | --- | --- |
| `MemoryDep(name,index,var_names,size,mode)` | buffer 名、线性索引、循环变量范围和访问方式 | 普通循环精确读写某些位置 |
| `StarDep(name)` | 只知道依赖整块 buffer，不提供逐位置索引 | extern 输入输出、mutation 目标、符号顺序 |
| `WeakDep(name,mutating_buf)` | 修改必须等待某个旧 reader；不代表真实内存读取 | 保护覆盖前仍要完成的读取 |

例如 `MemoryDep(arg0_1, 3*c0+1, {c0:2})` 在两次循环中读取物理位置 1 和 4。
连续、广播、转置和间接索引会产生不同的 `index`；后面的融合判断会使用这些差异。

外部 mm 没有交出普通循环体，因此 `sin(mm(x,w))` 的入口状态是：

```text
op0 ExternKernelSchedulerNode
  reads  = StarDep(x), StarDep(w)
  writes = StarDep(buf0)

op1 SchedulerNode
  reads  = MemoryDep(buf0, c0, {c0:8})
  writes = MemoryDep(buf1, c0, {c0:8})
```

生产端和消费端的记录精度不同，仍可通过共同的名字 `buf0` 相连。

## 第二步：由名字建立生产者、使用者和等待关系

Scheduler 开始时先把图输入、常量和 TorchBind 常量的名字放进
`available_buffer_names`。这些值在编译函数开始运行时已经存在，不需要等待图内 operation
产生。节点初始化时把全部 `reads` 复制成 `unmet_dependencies`，再从中删除这些已可用名字。

看一条内部结果链：

```text
op0: buf0 = x + 1
  reads  = MemoryDep(x,    c0, {c0:8})
  writes = MemoryDep(buf0, c0, {c0:8})

op1: buf1 = buf0 * 2
  reads  = MemoryDep(buf0, c0, {c0:8})
  writes = MemoryDep(buf1, c0, {c0:8})
```

`x` 已经可用，所以 op0 不再等待它；`buf0` 是图内结果，op1 仍要等待：

```text
op0.unmet_dependencies = {}
op1.unmet_dependencies = {MemoryDep(buf0, c0, {c0:8})}

buf0.defining_op = op0
users[buf0]      = {op1}
users[buf1]      = {OUTPUT}
```

`defining_op` 指向写出该 buffer 的节点；`users` 记录谁要使用它；`unmet_dependencies` 记录
当前节点仍需等待哪些图内结果。三份状态共同表达 `op0 → op1`，并没有另外创建一个叫
“edge”的产品对象。

`OUTPUT` 是虚拟使用者，不执行计算。Scheduler 为函数返回值登记它，使通向返回值的工作
不会在后续死节点删除中消失。输入或常量被修改时，也会为其最新版本登记 `OUTPUT`，因为
写回本身是调用者可见的结果。

若一项读取既不在 `available_buffer_names` 中，也没有图内 producer，Q31 不会把它自动当成
外部输入。这个名字会继续留在 `unmet_dependencies`；后续建立祖先关系时无法找到它的
`defining_op`，编译失败。

## 第三步：共享 storage 和 mutation 补充普通名字关系

普通读写默认“一个 buffer 名代表一份结果”。alias 和 mutation 会打破这个假设：两个名字
可能指向同一 storage，同一个外部可见名字也可能先后表示修改前后的值。

### alias 名共享同一份 users

Scheduler 先读取每个 operation 输出声明的 alias 名。假设 IR 已说明 `view_out` 与 `base`
共享 storage，消费者 C 读取 `view_out`：

```text
登记 alias 前：users[base] 和 users[view_out] 是两份列表
登记 alias 后：users[base] ─┐
                             ├─→ 同一份列表 {C}
             users[view_out] ┘
```

随后节点 M 声明覆盖 `base`。M 查询 `users[base]` 时会看到经 `view_out` 读取的 C。假设 C
写出 `c_result`，M 就增加：

```text
WeakDep(c_result, mutating_buf=m_result)
```

这条记录表示 C 必须先读完共享 storage，M 才能覆盖它。若 `base` 本身由图内 P 产生，M
还会增加 `StarDep(base)`，表示覆盖也要等待 P。于是当前依赖状态表达：

```text
P → M          # 目标 storage 先存在
C → M          # alias reader 先完成
```

若 C1、C2 都已通过 `base` 或任一 alias 名读取这块 storage，共享 users 列表会同时包含二者。
M 分别对 C1、C2 的输出增加 WeakDep，于是两名 reader 都必须先于 M；这组读取本身不会在
C1 与 C2 之间建立先后边。

这里的案例按当前 alias 合并逻辑推演；实验 08 没有单独运行 alias 图。Q31 只能处理 IR 已经
报告的 alias 名，不能发现一项未写进 IR 的隐藏共享关系。

### mutation 先保护旧 reader，再更新逻辑版本名

考虑：

```python
old = x + 1
x.copy_(y)
return old, x
```

Lowering 交出：

```text
op0：读 arg0_1(x)，写 buf0                 # old
op1：读 arg1_1(y)，写 buf1，mutations=[arg0_1]
```

处理 op1 的修改声明时，Scheduler 做三件事：

1. 增加 `StarDep(arg0_1)`，表明被覆盖的目标；本例目标是已可用图输入，所以该记录随后从
   `unmet_dependencies` 删除；若目标由图内节点产生，它会留下并要求等待 producer。
2. 查看 `users[arg0_1]`，发现 op0 已读取旧 x；为 op1 增加
   `WeakDep(buf0, mutating_buf=buf1)`，并把 op1 登记为 buf0 的 weak user。
3. 更新版本表：后续依赖图中对 `arg0_1` 的读取改指向 `buf1`；生成代码仍知道 buf1 实际
   写回调用者的 `arg0_1`。

结果是：

```text
op1.unmet_dependencies = {WeakDep(buf0, mutating_buf=buf1)}
users[buf0]             = {op1(weak), OUTPUT}
mutation_renames        = {arg0_1: buf1}
mutation_real_name      = {buf1: arg0_1}
```

因此 op0 先得到 `old=[1,2,3]`，op1 再把 y 写入 x，最终 x 为 `[9,9,9]`。若没有旧 x reader，
就不会凭空增加这条 WeakDep。例如 `x.copy_(y); return x*2` 在进入 Scheduler 前，后继乘法
已经被改成直接读取 y；两个 operation 都读 y，版本表仍记录写回，但没有需要保护的旧 x
reader。

## 第四步：补符号依赖，并接住已有的 effect 顺序

有两种值或行为不一定通过普通 Tensor buffer 的读取自然形成边：运行时才能得到的符号整数，
以及必须维持先后的副作用。符号关系由 Scheduler 在本步新增；同类 effect 的前驱已经由
IR 写进 `ReadWrites`，Scheduler 在本步沿普通读关系把它接入 users 和 unmet 状态。

### unbacked SymInt：使用者等待符号的产生者

有些 shape 整数无法从输入 shape 预先推导，只能由图中一次计算得到。这里把它记为 `u0`。
假设 op0 产生 `u0`，op1 用 `u0` 决定输出 size：

```text
op0：产生 buffer buf0，并声明自己定义 u0
op1：在 size/index 中使用 u0，但没有普通 load(buf0)
```

若只看普通 `ReadWrites`，op1 与 op0 没有共同 buffer 名。Scheduler 另建一张
“符号 → 首个定义节点”的表，记录 `u0 → op0`；看到 op1 使用 u0 时，为 op1 增加对 op0
所有输出 buffer 的 `StarDep`：

```text
op1.unmet_dependencies += StarDep(buf0)
```

这是一条只为排序建立的 fake dependency。它不表示 op1 真的读取 buf0 元素，只保证 u0
已经产生。若同一个 `MultiOutputLayout` 让多个输出都声明定义 u0，当前实现使用遇到的第一
个定义节点作为来源。

来自符号图输入的自由符号一开始就标成“输入提供”，使用者不需要等待图内节点。图内节点
使用尚未登记来源的 unbacked symbol 时，依赖构造立即触发断言。若最终图输出的 shape 使用
u0，Scheduler 还给定义 u0 的节点输出登记 `OUTPUT`，避免产生这个返回 shape 所需符号的
工作被删除。

### effect chain：后一个同类副作用读取前一个结果

带副作用的调用不能只凭“没有普通使用者”判断为无用。`EffectfulKernel` 在 IR 建立时按
effect type 保存同类调用的前一个 operation。若 E0、E1 是同一种 effect：

```text
E0：该 effect type 的前一个 operation
E1：get_read_writes() 额外加入 StarDep(E0 的输出)
```

因此 E1 的普通依赖处理会建立 `E0 → E1`。不同 effect type 不会仅因“都有副作用”自动
得到彼此的顺序边。第一项同类 effect 没有前驱，也不会凭空得到一条读取；它仍通过
`has_side_effects=True` 告诉下一篇的死节点删除逻辑保留自己。collective 等其他 IR 也可能
报告副作用；只有 operation 明确提供的读写或 effect chain 才决定本篇的顺序关系。

## 本层交给后续阶段什么

依赖构造结束时，每类状态都有明确消费者：

| 本层结果 | 内容 | 谁继续使用 |
| --- | --- | --- |
| scheduler nodes + `ReadWrites` | 节点类型及精确/整块读写 | Q32 排序和删除；Q33 比较访问与融合条件 |
| `defining_op`、`users`、`unmet_dependencies` | producer、消费者和仍需等待的内部结果 | Q32 拓扑排序、祖先和死节点删除 |
| alias 共用 users、WeakDep | 共享 storage 上的 reader 与覆盖顺序 | Q32 保持合法顺序；删除后再裁剪失去来源的 weak 关系 |
| `mutation_renames`、`mutation_real_name` | 当前逻辑版本与实际写回名字 | Q32 继续解释版本、排序与后续生成 |
| unbacked symbol fake deps、`OUTPUT` users | 符号顺序；函数返回、输出 shape 和外部可见 mutation 的保活 | Q32 排序和删除 |
| effect `StarDep`、`has_side_effects` | 同类 effect 的前驱关系；一般副作用的保留标志 | Q32 排序和死节点删除 |

本篇完成的标志是：已可用输入已经从 unmet 集合删除，所有已知消费者、alias reader、
mutation、symbol use、effect 前驱和输出保活都已登记。合法图中的内部 unmet 关系应能在
后续找到 producer；缺失 producer 的名字也可能暂时留在这里，直到 Q32 建立祖先关系时
失败。本篇仍没有选出最终次序，也没有删除节点或计算最后使用位置。

下一篇从这些状态开始，说明怎样得到合法顺序、怎样删除无用节点，以及 mutation 版本和
生存期怎样随最终节点序列变化。完整构造源码与逐句状态见
[依赖构造核心源码](01-dependency-source/README.md)。
