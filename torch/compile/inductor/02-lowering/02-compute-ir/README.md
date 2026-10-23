# 计算 IR：一个逻辑索引最终读到哪里、算出什么

[返回 Lowering](../README.md) · [上一篇：GraphLowering 解释器](../01-graph-interpreter/README.md) · [返回总览](../../README.md)

实现叶子：[Pointwise、Reduction 与 View 构造源码](01-ir-construction-source/README.md)。

Graph interpreter 看到 add、relu 会返回 `Pointwise`，sum 可返回 `Reduction`，permute 返回 view。本篇打开这些对象内部共同使用的语言：范围说明合法逻辑索引，loader 说明怎样从输入取值，`inner_fn` 把索引和算术组合起来，view 的 reindexer 则先把新坐标换回旧坐标。

本篇出现两种 layout 记录。`FixedLayout` 已确定 size、stride 和 offset，因此可以直接算物理位置；`FlexibleLayout` 只承诺 size，具体 stride 由后续存储职责确定。

先把三条路径放在同一张图里：

```text
Pointwise：输出索引 i ──────────────→ loaders(i) ─→ 算术 ─→ 一个输出值

Reduction：输出索引 i + 组内索引 r ─→ loader(i,r) ─→ combine ─→ 一个输出值

View：新索引 j ─→ reindex(j)=旧索引 i ─→ 原数据 loader(i) ─→ 一个观察值
```

独立实验 [`experiments/05_compute_ir_paths.py`](../../experiments/05_compute_ir_paths.py) 在当前源码中分别编译三条路径，并观测到：

```text
add      → ComputedBuffer → Pointwise(ranges=[2,3])
sum      → ComputedBuffer → Reduction(ranges=[2], reduction_ranges=[128], type=sum)
permute  → ReinterpretView(size=[3,2], stride=[1,3]) → InputBuffer
```

这三条轨迹分别展示普通逐点计算、规约和只改坐标解释的 view。后文会看到，小型静态规约有时会直接展开成 pointwise 计算，因此不能只凭函数名 `sum` 推断最终 IR 类型。

## Pointwise：同一个输出位置驱动所有输入 loader

`make_pointwise(fn)` 先把 Tensor 输入转成可调用的 loaders，取第一项输入的 size 作为 `ranges`，再建立：

```python
# 对应当前源码结构的教学展开
def inner_fn(index):
    inputs_loaded = [load(index) for load in loaders]
    return fn(*inputs_loaded)
```

随后 `Pointwise.create(device, dtype, inner_fn, ranges)` 把它包装成 TensorBox。注意 `inner_fn` 在 Lowering 时主要是一份规则；真正 codegen 时，ops handler 才把 `load/add/store` 等动作写成目标后端代码。

### 同形状 add 的六个位置

给定：

```text
x = [[0,1,2],        y = [[10,10,10],
     [3,4,5]]             [20,20,20]]
```

两者 shape 都是 `[2,3]`，所以 `ranges=[2,3]`。合法 index 是 `[row,column]`，row 为 0、1，column 为 0、1、2。

| 输出 index | x loader | y loader | inner_fn 的值 |
| --- | ---: | ---: | ---: |
| `[0,0]` | 0 | 10 | 10 |
| `[0,1]` | 1 | 10 | 11 |
| `[0,2]` | 2 | 10 | 12 |
| `[1,0]` | 3 | 20 | 23 |
| `[1,1]` | 4 | 20 | 24 |
| `[1,2]` | 5 | 20 | 25 |

`Pointwise._index([2,3])` 在 IR 格式化/分析时构造两个符号索引；若某维 size=1，则该维索引直接用常数 0。这个规则与广播 view 配合：广播维无论输出 index 是多少，都映回输入的 0。

### 广播改变 loader，不改变输出 index 的职责

若 y 变成 shape `[1,3]`，广播到 `[2,3]` 后，其 `ExpandView` reindexer 会把第 0 维清零：

```text
输出 index [0,2] → x[0,2] + y[0,2]
输出 index [1,2] → x[1,2] + y[0,2]
```

外层 Pointwise 仍把 `[row,column]` 同时交给两个 loader；差别藏在 y loader 内部的 view reindex。这样算术构造函数无需为每种广播形状重写循环。

实验实际记录到 `ExpandView.create` 的输入 size `[1,3]`、目标 size `[2,3]`。因为 y 已有固定 storage/layout，构造器立即把它表示为：

```text
ReinterpretView(size=[2,3], stride=[0,1], offset=0)
    → 原 InputBuffer(size=[1,3], stride=[3,1])
```

stride 0 使两行的同一列落到相同物理元素：`[0,2]` 与 `[1,2]` 的位置都是 `0×row+2×1=2`。随后 add 的 Pointwise 仍是 `ranges=[2,3]`。这份实际对象把“外层 index 不变、输入 loader 清零广播维”连接起来。

若另一项是标量常数 2，`promote_constants` 将其变成与 tensor device/dtype 协调的 Constant/ExpandView；Pointwise loader 在每个 index 都取得同一个常数，而不是为 2 建立运行时 Tensor 输入。

### 规则组合为何能消掉完整中间 Tensor

对 `relu(x+y)*2`：

```text
add_loader(i)  = load_x(i) + load_y(i)
relu_loader(i) = max(add_loader(i), 0)
mul_inner(i)   = relu_loader(i) * 2
```

每层 `make_pointwise` 捕获前一层 loader。只要没有提前物化，最终 inner_fn 调用链就包含全部算术。生成程序为当前位置保留局部临时值，而不是为 add 和 relu 分别分配完整 buffer。

## Reduction：把输出坐标与组内坐标分开

`_make_reduction_inner` 先规范化 axis，然后把输入各维分为：

- kept dimensions：决定产生哪一个输出；
- reduced dimensions：决定一个输出要组合哪些输入。

对于 shape `[2,3]` 的 `x.sum(dim=1)`：

```text
kept_idx=[0]，ranges=[2]
reduced_idx=[1]，reduction_ranges=[3]
reduction_type="sum"
```

构造的 loader 接受两组坐标：

```python
def loader(index, reduction_index):
    # index=[row]，reduction_index=[column]
    new_index = [row, column]
    return input_loader(new_index)
```

### 输出 row=1 的完整累计

| 输出索引 row | 规约索引 column | 原输入坐标 | 读到 | 累计后 |
| ---: | ---: | --- | ---: | ---: |
| 1 | 0 | x[1,0] | 3 | 3 |
| 1 | 1 | x[1,1] | 4 | 7 |
| 1 | 2 | x[1,2] | 5 | 12 |

row=0 同理得到 3，因此结果为 `[3,12]`。这里 column 不能并入普通输出 ranges：若把 `[row,column]` 都当输出范围，就会生成六个输出值，而不是两个和。

`Reduction.store_reduction` 把 `inner_fn(vars,reduction_vars)` 的单项值交给 `ops.reduction(dst_dtype, src_dtype, reduction_type, value)`，再把组合结果存到只由输出 vars 索引的位置。

### keepdim 只改变输出接口，不改变输入组合集合

`keepdim=False` 时 ranges 是 `[2]`，输出 shape `[2]`。`keepdim=True` 时被规约维保留为 size 1，ranges 是 `[2,1]`；loader 收到输出 `[row,0]` 后只取 kept dimension 的 row，再与 column 组合成原输入 `[row,column]`。

实验用 `[2,128]` 同时运行两种情况：非 keepdim 为 `ranges=[2]`，keepdim 为 `ranges=[2,1]`；二者的 `reduction_ranges` 都是 `[128]`、type 都是 sum。两种情况对每个 row 都读取相同 128 个元素，差别只是输出坐标系统和输出 layout（本例分别为 size `[2]` 与 `[2,1]`）。

### Reduction 构造可能有三种出口

Reduction 的坐标模型不保证最终一定保留为单层 `Reduction`。组内没有元素或只有一个元素时，构造器可用恒等值或唯一输入建立等价 Pointwise；较小静态规约也可能展开为 Pointwise。较大的规约可以保留为一层 Reduction，或先产生若干部分结果，再由第二层 Reduction 合并。

本实验的 `[2,128]` sum 实际保留 `ranges=[2]`、`reduction_ranges=[128]`、`type=sum`。构造器怎样处理 0 项、1 项、静态展开和 split 决策，留到[构造实现叶子](01-ir-construction-source/README.md)。本层只要求后继按实际返回对象继续处理。

## View：先改逻辑坐标，不必新增计算 operation

给定连续 x：

```text
x.shape  = [2,3]
x.stride = [3,1]
x = [[0,1,2],
     [3,4,5]]
```

`x.permute(1,0)` 调用 `PermuteView.create(x.data, dims=[1,0])`。如果底层已经具有 storage 和 layout，当前实现直接建立：

```text
ReinterpretView(
    同一个 storage,
    FixedLayout(size=[3,2], stride=[1,3], offset=0)
)
```

实验实际观测到这条类型链，且 buffers=0、operations=0。它没有运行一个“转置 kernel”；只是让同一物理存储接受新的逻辑坐标解释。

完整的共同物理槽位图见 [transpose-layout.html](transpose-layout.html)。图中固定六个 float32 物理位置，直接标出了 x 和 v 在每个槽位上的逻辑坐标。

### 三个具体位置

FixedLayout 的 indexer 使用：

```text
元素位置 = offset + Σ(index[d] × stride[d])
```

对 v 的 stride `[1,3]`：

| v 位置 | 元素位置 `p×1+q×3` | 原 x 位置 | 值 |
| --- | ---: | --- | ---: |
| v[0,0] | 0 | x[0,0] | 0 |
| v[0,1] | 3 | x[1,0] | 3 |
| v[2,1] | 5 | x[1,2] | 5 |

float32 每元素 4 byte，所以 v[2,1] 的 byte 偏移为 `5×4=20`。

如果底层还不是带固定 layout 的 storage，`PermuteView` 会保留 `dims`，其 `make_reindexer()` 构造逆置换：新 index `[p,q]` 先变回旧 index `[q,p]`，再调用原数据 loader。两种表示的含义一致：改变坐标到原数据的映射，而不改变数值来源。

## 三类对象怎样组合

View 可以出现在 Pointwise loader 里面，Reduction 的输入 loader 也可以先经过 View：

```text
v = x.permute(1,0)  # v[p,q] → x[q,p]
z = v + 10          # Pointwise inner([p,q]) → load_x([q,p]) + 10
s = z.sum(dim=1)    # Reduction output p，组内 q → load_x([q,p]) + 10
```

以 x 为前述 2×3：v shape `[3,2]`。输出 `s[1]` 组合：

```text
q=0: v[1,0]=x[0,1]=1，+10 → 11
q=1: v[1,1]=x[1,1]=4，+10 → 14
sum → 25
```

这条组合说明每层只承担一种坐标职责：view 换坐标，pointwise 对同一输出坐标做算术，reduction 增加组内坐标并组合。

综合实验 17 将规模扩大到 x shape `[128,3]`，实际轨迹是：

```text
permute → ReinterpretView(size=[3,128])       operations=[]
add     → Pointwise(size=[3,128])             operations=[]
relu    → Pointwise(size=[3,128])             operations=[]
sum     → ComputedBuffer buf0(size=[3])       operations=[ComputedBuffer]
mm      → ExternKernelOut buf1(size=[1,2])    operations=[ComputedBuffer, ExternKernelOut]
```

本篇在 sum 出口关心的是：Reduction 的三个输出位置 p=0、1、2，各自让组内 q 遍历 0…127；它的输入 loader 依次执行 permute 的 `[q,p]` 映射、加 1 和 relu。实际 rows 为 `[24512,24640,24768]`。后续特殊调用把已经物化的 buf0 作为 mm 输入。

迁移到 `sum(dim=0)` 时，输出不再按 p 选转置后的行，而是按 q 选列：输出范围变为 `[128]`，每个 q 的 reduction index p 遍历三项 `[0,q]、[1,q]、[2,q]`。改变 axis 同时改变 kept/reduced 集合，不能只把最终输出 shape 改掉而沿用原 loader。

## 本层出口

| IR | 必要信息 | 交给下一职责的内容 |
| --- | --- | --- |
| Pointwise | device、dtype、ranges、inner_fn | 一条可继续被后继 loader 组合的位置规则 |
| Reduction | 上述信息 + reduction_ranges/type/src_dtype | 输出空间、组内空间和组合规则 |
| View/ReinterpretView | 新 size、reindex 或固定 stride/offset、共享 data | 读取原 storage 的坐标关系，通常不新增 operation |

这些对象先交给[存储与数据版本](../03-storage-layout/README.md)。可继续组合的规则仍由后继 loader 引用；图输出、外部调用或其他物化条件会让规则进入 `ComputedBuffer` 并登记 operation。Scheduler 只读取已经登记的 operations，并在分析或生成时重放其中的计算/索引规则；共享原 storage 的 view 没有独立 operation，也不会单独成为调度节点。

