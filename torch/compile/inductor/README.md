# 从 `torch.compile` 的图到可调用函数：Inductor 总览

> 源码基线：`/work/pytorch`，commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。本文只依据该工作区当前源码、相关测试与本教程的独立实验。

你已经知道 `torch.compile` 会捕获图，但要读懂 Inductor，先要看清“图”里究竟保存了什么。以贯穿案例为例：

```python
def fn(x, y):
    a = x + y
    b = torch.relu(a)
    return b * 2
```

编译器收到的计算关系可以先记成这张表：

| 名字 | 这一步做什么 | 输入引用谁的结果 |
| --- | --- | --- |
| `x`、`y` | 函数输入 | 调用者提供 |
| `a` | 相加 | `x`、`y` |
| `b` | 把负数变为 0 | `a` |
| `c` | 乘 2 | `b`、常数 2 |
| 返回 | 返回结果 | `c` |

FX 是 PyTorch 保存这类计算关系的一种表示。每个 FX node 记录一项输入、操作或返回；边由“后一步引用前一步的结果”形成。`GraphModule` 把这张图、相关模块属性和可执行的 `forward` 组织在一起。比如把 relu 的输入从 `a` 改成 `x`，就不再计算 `relu(x+y)`，而是计算 `relu(x)`；一条边连向谁，直接决定程序算什么。

现在才进入四个组件。先不记内部类名，只看同一份计算关系每一步多了什么。

对上面的三个算术操作，FX 图只记录“先加、再 relu、最后乘 2”。Lowering 会把它补成一条可以代入位置的规则：

```text
给定输出位置 i
  读取 x[i] 和 y[i]
  相加并把负数变成 0
  再乘 2
  写到输出位置 i
```

这份“有哪些输入、输出范围多大、每个位置怎样计算”的编译器记录，后文称为 Inductor IR。对本例，它至少让后一步知道：读取 x、y，写一份输出，位置 i 从 0 到 2。

Scheduler 再把这项工作包装成一条待生成记录，可以先把它想成：

```text
工作：计算最终输出
读取：x、y
写入：out
位置范围：0、1、2
必须排在谁后面：没有其他计算
```

这样的记录后文称为 scheduler node。Scheduler 会比较多条记录之间谁读谁写、谁必须先执行，以及能否放进同一个循环。假设三项算术原本各自生成一次完整数组，就需要三个循环；把它们放进同一个循环后，每个位置依次完成 add、relu、mul，只写最终结果。后文把这种“多项工作合成一个生成单元”称为 fusion（融合）。本例的实际编译会直接保留可组合的位置规则，最终生成下面这一类循环：

```cpp
for (int64_t i = 0; i < 3; i++) {
    out[i] = max(x[i] + y[i], 0.0f) * 2.0f;
}
```

Codegen 做的就是把已经排好顺序和分组的记录变成这种目标代码，并生成负责检查输入、准备输出空间、调用代码和返回结果的外围函数。

本教程把 Inductor 看成一条逐步补全执行细节的流水线：

```text
torch.compile 前半段交来一张 FX GraphModule
                 │
                 ▼
1. 图优化：在 FX 节点层改写、删除、规范化算子
                 │  仍是 FX 图
                 ▼
2. Lowering：逐个解释 FX 节点，构造上述位置规则和待执行工作
                 │  怎样计算、读写哪份数据、是否要调用已有函数
                 ▼
3. Scheduler：比较各项工作的读写，安排先后并决定哪些放进同一生成单元
                 │  已排好次序和分组的工作记录
                 ▼
4. Codegen：生成设备或 CPU 执行代码；外围函数准备内存、调用并返回
                 │
                 ▼
一个可调用的编译结果
```

`torch.compile` 的前半段先把 Python 计算捕获成 FX 图。对本例，Dynamo 交出的区域包含 add、relu、mul 和 return，不包含函数外的下一条 Python 语句。训练时，AOTAutograd 再根据这张前向计算图建立求梯度所需的反向计算，并交出可分别编译的前向图与反向图。它们为 Inductor 准备输入，但不属于下面四项 Inductor 职责。本篇追踪的是：其中每一张 FX 图怎样成为可调用函数。

另一端也不能停在执行加法的 C++ 循环。调用者最终需要一个 Python 可调用对象；外围代码还要检查输入是否满足这次编译采用的 shape 和 stride。stride 表示逻辑坐标每增加 1，内存偏移增加多少；一维连续输入的 stride=`[1]`，所以位置 i 读取内存偏移 i。检查通过后，外围代码准备输出空间，调用生成代码或已有函数，再把结果交回调用者。源码把这层外围代码称为 wrapper。

后文还会反复用到三个词，先用本例把它们落到具体内容上：

- **layout（排布记录）**说明一个张量有多大，以及逻辑位置怎样换算成一维内存位置。例如 size=`[2,3]`、stride=`[3,1]` 时，位置 `[1,2]` 对应偏移 `1×3+2×1=5`。
- **storage（实际数据空间）**是保存元素的那块内存。同一块 storage 可以被两个张量名字共同使用；例如 `x.view(3,2)` 没有复制六个元素，只是用另一套 size/stride 读取 x 的 storage。
- **backend（代码生成后端）**是把排好顺序的工作翻译成某种目标代码的实现。CPU backend 可以把本例写成 C++ 循环；GPU 路径可交给 Triton backend 写成 Triton kernel。

四个阶段并不保证每次都各自产生一个新节点。它们更像四次补充问题答案：

| 处理到哪里 | 当前对象还缺什么 | 本阶段补上的信息 |
| --- | --- | --- |
| FX 图优化前 | 节点关系可能仍可整理或替换 | 不改变程序结果的节点改写，以及引用关系合法的图 |
| Lowering 前 | FX 节点只说“调用哪个算子”，没有说明每个输出位置怎样读输入 | 位置计算规则；需要独立保存的数据；要调用的已有函数 |
| Scheduler 前 | 各项待执行工作已经形成，但还没有比较它们的读写关系 | 谁必须先执行；哪些工作放在同一个生成单元；每份数据最后由谁读取 |
| Codegen 前 | 工作已经排好顺序和分组，但还不是机器能执行的程序 | CPU/CUDA 等目标代码，以及检查输入、准备内存、调用和返回的外围代码 |

图优化本身还会在编译流程的三个位置运行。AOTAutograd 之前称为 pre-grad；训练时前向和反向仍在一张图里时称为 joint；拆成独立的前向图或反向图以后、进入 Lowering 前称为 post-grad。FX Graph Cache 保存的正是某张单图从 post-grad 往后的编译结果；命中后可以恢复已有结果，不再重做这张图的 post-grad、Lowering、Scheduler 和 Codegen。

如果某一步已经不缺对应信息，它可以不产生新工作。例如纯转置可能只保留共享视图，没有独立转置 kernel。

理解上面的对象以后，源码中的主干调用可用来定位实现：

下面调用树中的 `GraphLowering` 是“正在处理这一张 FX 图”的对象。它一边逐个读取 FX node，一边保存“该 node 当前对应哪条位置计算规则或哪份有名结果”，最后把需要执行的工作收集起来。先知道这个用途即可；它的字段和分支在 Lowering 篇展开。

```text
compile_fx
  ├─ pre_grad_passes
  ├─ AOTAutograd（把训练拆成前向、反向图；把 add_ 这类原地计算改成普通 add 加显式 copy_ 写回）
  └─ _compile_fx_inner
       └─ fx_codegen_and_compile
            ├─ joint_graph_passes / post_grad_passes
            ├─ GraphLowering(...).run(...)
            └─ GraphLowering.compile_to_fn()
                 └─ GraphLowering.codegen()
                      ├─ Scheduler(self.operations)
                      ├─ Scheduler.codegen()
                      └─ wrapper_code.generate(...)
```

`compile_fx` 并不是只收一张最初的 Dynamo 图然后直线走到底。训练时 AOTAutograd 会准备前向与反向图；动态形状也会增加符号尺寸和输入条件。普通 `torch.compile` 在当前 Python 进程中交回可调用函数；AOT 编译则预先生成可保存、可独立部署的源码、输入输出信息和加载入口。本篇先抓住每一张交给 `_compile_fx_inner` 的单图共有的主干。

## 一个逐元素案例走完四个阶段

输入 `x=[-2,1,3]`、`y=[1,4,-1]`，数学过程是：

```text
x+y       = [-1, 5, 2]
relu(...) = [ 0, 5, 2]
乘 2      = [ 0,10, 4]
```

独立实验保存在 [`experiments/01_pointwise_pipeline.py`](experiments/01_pointwise_pipeline.py)，实际输出也为 `[0,10,4]`。

### 图优化交出什么

实验在 `post_grad_custom_post_pass` 保存了 post-grad 图：

```text
arg0_1, arg1_1
   └─ aten.add.Tensor
        └─ aten.relu.default
             └─ aten.mul.Tensor(..., 2)
                  └─ return
```

三项算术仍是三个 FX 节点。这说明本例的“最后融合成一个循环”不是 post-grad FX 优化已经做完的事。图优化在这一层整理或替换节点、删除无效节点，并保证节点先后和引用仍然合法；交出的仍是 FX 图。

### Lowering 交出什么

`GraphLowering` 按 FX 顺序解释节点：placeholder 变成输入说明；读到 `aten.add`、`aten.relu`、`aten.mul` 时，分别找到负责建立相应计算规则的处理函数。

对于这个逐元素链，Lowering 的核心结果可以先理解为一条“给定输出位置，怎样取数和计算”的延迟规则：

```python
# 教学伪代码，不是产品源码
def value_at(i):
    t0 = read_x(i)
    t1 = read_y(i)
    return relu(t0 + t1) * 2
```

这里保存的是计算方法，不是已经算出的三元素张量。只有后继操作或函数输出需要一份有名字的数据时，Lowering 才登记一项负责产生该数据的工作。对应的内部类名和登记过程在 Lowering 篇展开。

### Scheduler 交出什么

`GraphLowering.codegen()` 用 `self.operations` 创建 `Scheduler`。例如 op0 写 buf0、op1 读 buf0 时，即使输入列表反过来，Scheduler 也会把 op0 排在 op1 前。若 op1 和 op2 都读 buf0，则最终顺序中较后的那一个是最后读取者，此后 buf0 才可释放或复用。Scheduler 还会删除结果无人使用且没有写入等副作用的节点，并尝试融合兼容的工作。

本例最终只有一个已融合的 CPU 调度工作项。注意，Scheduler 并不是把数值算出来；它决定哪些计算应当一起交给哪个设备后端，以及它们的先后与存活关系。

### Codegen 交出什么

实验捕获到一个 C++ kernel。关键循环等价于：

```cpp
for (int64_t x0 = 0; x0 < 3; x0++) {
    auto tmp0 = in_ptr0[x0];
    auto tmp1 = in_ptr1[x0];
    auto tmp2 = tmp0 + tmp1;
    auto tmp3 = max(tmp2, 0);
    auto tmp5 = tmp3 * 2.0f;
    out_ptr0[x0] = tmp5;
}
```

wrapper 则检查输入 shape/stride，分配 `buf0`，调用 `cpp_fused_add_mul_relu_0`，释放不再需要的输入引用，再返回 `(buf0,)`。所以 Codegen 不只指 kernel 文本，也包括把 kernel 变成可调用程序所需的外围代码和加载过程。

## 四个组件分别回答什么问题

| 组件 | 进入时拥有 | 核心问题 | 交出给下一步 |
| --- | --- | --- | --- |
| 图优化 | 带元信息的 FX 图 | 能否在不改变程序结果的前提下整理、删减或替换节点？ | 改写后的 FX 图 |
| Lowering | FX 节点、样例输入的 shape/device/dtype 信息 | 每个算子在 Inductor 中如何表示计算、布局、存储、修改或外部调用？ | Inductor IR 操作和图级状态 |
| Scheduler | 已物化的 IR 操作及其读写 | 哪些操作有依赖，按何顺序执行，哪些能融合？ | 排序和融合后的调度节点 |
| Codegen | 调度节点、目标 device、布局与生存期 | 如何生成目标后端 kernel，并由 wrapper 正确调用？ | 可编译、可加载、可调用的模块 |

同一个词在不同层可能指不同对象。“node”可能是 FX node、IR operation 或 scheduler node；“fusion”可能是 FX 模式替换，也可能是 scheduler 将循环合成一个 kernel。阅读源码时先问当前对象属于哪一层。

## 其他请求也完整经过四阶段，但交接内容不同

逐元素链只能说明一条理想路径。下面用三个对照案例走到本层出口，先建立全貌；每个组件的内部算法留给对应子篇。

### 按行求和：Lowering 必须增加组内位置

输入 `x=[[0,1,2],[3,4,5]]`，请求 `x.sum(dim=1)`，数学结果为 `[3,12]`。

| 阶段 | 本例的具体变化 |
| --- | --- |
| 图优化 | 交出一张包含 `aten.sum.dim_IntList` 请求的规范化 FX 图；本例没有把求和改成三个加法节点 |
| Lowering | 建立两个坐标：`row` 选择输出 0 或 1，`column` 遍历组内 0、1、2；记录组合操作是 sum |
| Scheduler | 看到一项读取 6 个输入位置、写 2 个输出位置的规约工作，安排依赖、融合和执行顺序 |
| Codegen | 当前 CPU 实验生成外层两行、内层规约循环及累计值；wrapper 分配长度 2 的结果并调用 kernel |

对第 1 行，累计值按 `0→3→7→12` 变化。Lowering 交出的不只是“算子叫 sum”，还必须让后续知道一个输出对应哪一组输入。

### 转置视图：逻辑结果变化，可能没有新计算 operation

同一输入请求 `x.t()`。输出形状为 3×2，位置 `[2,1]` 对应原输入 `[1,2]`，值为 5。

| 阶段 | 本例的具体变化 |
| --- | --- |
| 图优化 | 保留规范化的转置/维度变换关系 |
| Lowering | 建立 size=`[3,2]`、stride=`[1,3]` 的读取视图，保留它与输入共享 storage；实验中没有新增 operation |
| Scheduler | 没有需要执行的转置计算时，不会为了“经过调度”虚构一个搬运节点 |
| Codegen | wrapper 可按输出接口返回对应 view；只有消费者要求另一种物理布局时才需要额外转换或复制 |

这个案例说明“阶段存在”不等于每个阶段必然产生新节点或 kernel。编译器保留共享和索引关系，同样是在完成工作。

### 矩阵乘接逐元素：已有实现与生成循环在 wrapper 汇合

对 `return torch.relu(x @ w)`，当前 CPU 路径把矩阵乘保留为已有 `extern_kernels.mm` 调用，relu 可成为生成的逐元素 C++ kernel：

```text
FX：mm → relu → output
  │
  ├─ Lowering：mm 形成外部/候选实现，relu 形成位置规则
  ├─ Scheduler：外部调用形成边界，安排 mm 先完成，再提交 relu
  └─ Codegen wrapper：分配 mm 输出 → 调用 mm → 调用 relu kernel → 返回
```

若实现选择或设备不同，mm 也可能进入 template：其中保存 mm 的输入、输出 layout、分块参数候选和待填的 kernel 代码；算法选择确定一组参数后，后端填入模板并生成 kernel。共同点是 Lowering 保留“调用什么、参数和输出是谁”，Scheduler 按依赖排序，Codegen 在同一 wrapper 中连接它与生成 kernel。

### 输入覆盖：还要保护旧值和外部可见身份

请求 `old=x+1; x.copy_(y); return old,x`，取 `x=[0,1,2]`、`y=[9,9,9]`。正确出口是 `old=[1,2,3]`，调用者的原 x storage 变成 `[9,9,9]`。

Lowering 在建立覆盖写入前让仍读取旧 x 的 `old` 物化；Scheduler 由读写和 mutation 依赖把旧读排在覆盖前；Codegen 生成先读旧 x、再写原 x storage 的程序。若颠倒顺序，old 会错误变成 `[10,10,10]`。这条路径说明编译不仅保存数值公式，还必须保存存储身份和数据版本。

### 运行时条件：两张分支图共享一个父图出口

请求可以写成：predicate 为真时返回 `x+1`，否则返回 `x-1`。取 `x=[2,4]`，两种运行结果分别是 `[3,5]` 和 `[1,3]`。

| 阶段 | 本例的具体变化 |
| --- | --- |
| 图优化 | 父 FX 图保留一项 `higher_order.cond`，两支计算仍以子图存在；编译期不能根据一次样例 predicate 删除另一支 |
| Lowering | 分别解释 true/false 子图，并核对两支输出的数量、shape、stride、dtype 和 device 能否形成同一个父图接口 |
| Scheduler | 父图看到一项带子图接口的控制流工作，按它对 predicate、operands 和输出的依赖安排位置，而不是把 add、sub 当作父图中必定连续执行的两项工作 |
| Codegen | wrapper 保留运行时条件，只调用被 predicate 选中的分支；最终仍通过统一输出接口返回一份结果 |

这里“编译两支”与“运行两支”不同。编译器必须先确认两支返回同样数量的结果，而且对应结果的 shape、stride、dtype 和 device 能组成同一个父图输出，运行时才能只选其中一支。如果 false 分支改成返回 shape `[1,2]`，而 true 分支仍返回 shape `[2]`，Lowering 无法确定父图究竟返回哪种形状，本例会在进入正常调度和生成前失败。

### 无计算图与缓存命中：主干可以有快速出口

若单图没有需要 Inductor 生成的计算，`_compile_fx_inner` 可以把原 `GraphModule.forward` 包成 boxed callable。普通 Python 函数接收多个位置参数；这里的 boxed callable 统一接收一个列表 `args=[x,y,...]`，取出参数后返回单图约定的结果 tuple。它不虚构 Scheduler 节点或空 kernel。

另一种快速路径是 FX Graph Cache 命中。它发生在进入 `fx_codegen_and_compile` 之前：此时 pre-grad 已经在 AOTAutograd 之前运行，训练图的 joint passes 也已经在拆分前运行。当前单图的节点和常量、输入的 dtype/device/shape/stride，以及影响生成代码的配置共同参与定位缓存候选；候选自己的 shape 检查再判断它能否用于本次尺寸。合法候选保存 wrapper 源码的 cache key、代码路径、输入输出信息和 shape 检查表达式；由此加载 wrapper 并恢复 callable。因此命中时省去的是这张单图后续的 post-grad、Lowering、Scheduler 和 Codegen，不是前面的所有图处理，也不是实际运行。调用时 wrapper 和 kernel/已有函数仍会执行。

例如同一张 `relu(x+1)*2` 图在两个独立进程共用一个新缓存目录：本教程实验第一次观察到一次 FX cache miss，第二次观察到一次 hit，两次都计算相同结果。正确结果只说明程序行为一致；明确的 hit/miss counter 才说明第二次走了缓存出口。

这些案例合起来给出本教程的处理地图：逐元素、规约、视图、已有调用、mutation 和控制流都可能进入四项职责，但每次交接的对象不同；无计算图和缓存命中则展示了合法快速出口。后续子篇分别打开这些差异。

## 综合案例：一条请求同时改变索引、合并数据并跨过已有调用

最后把前面分开建立的机制放到同一个真实 CPU 案例中：

```python
def composite(x, weight):
    view = x.permute(1, 0)
    shifted = torch.relu(view + 1)
    rows = shifted.sum(dim=1)
    projected = torch.mm(rows.unsqueeze(0), weight)
    return rows, projected
```

输入 `x=torch.arange(384.).reshape(128,3)`，所以第一行是 `[0,1,2]`，最后一行是 `[381,382,383]`。转置后有 3 行、每行 128 项；它仍读取 x 的原 storage。所有数非负，`relu(view+1)` 不改变加 1 后的值。第一条规约行依次是 `1,4,7,...,382`：

```text
项数 = 128
首项 = 1，末项 = 382
rows[0] = 128 × (1+382) / 2 = 24512

同理 rows = [24512, 24640, 24768]
```

令 `weight=[[1,0],[0,1],[1,1]]`。矩阵乘得到：

```text
projected[0,0] = rows[0] + rows[2] = 49280
projected[0,1] = rows[1] + rows[2] = 49408
```

独立实验 [`experiments/17_root_composite.py`](experiments/17_root_composite.py) 采集到的 Lowering 结果如下。箭头左边是 FX 操作；右边先说明它留下什么，再在括号中给出源码类名：

```text
permute      → 只换 size/stride、继续读原输入（ReinterpretView）；operations 仍为空
add / relu   → 可继续组合的位置计算规则（Pointwise）；operations 仍为空
sum          → 需要保存的有名结果 buf0（ComputedBuffer）；operations=[buf0]
unsqueeze    → 对 buf0 的 view；不增加 operation
mm           → 已有矩阵乘调用的输出 buf1（ExternKernelOut）；operations=[buf0, buf1]
output       → 返回 buf0 和 buf1
```

为什么 sum 在这里必须成为 `buf0`？它一方面是最终第一个输出，另一方面又是 mm 的输入。mm 不能接收一条尚未执行的 Python 位置规则；它需要可传给已有实现的真实张量。于是 Lowering 在这个交接处给三项和稳定的存储身份。

接下来 Scheduler 面对的不是原来的六个 FX 算子，而是两个需要执行的 IR operation：先产生 `buf0`，再由 mm 读取它和 weight、写 `buf1`。视图、加法和 relu 已经包含在产生 `buf0` 的索引与计算中。Codegen 因而生成一个完成转置读取、加一、relu 和规约的 CPU kernel，再由 wrapper 调用 mm，最后同时返回 `buf0`、`buf1`。实验数值正是 `rows=[24512,24640,24768]`、`projected=[[49280,49408]]`。

这个例子展示的不是“算子越多越复杂”，而是三项要求相互作用：非连续 view 改变读取位置，规约把 128 项合成一项，已有调用和多输出共同要求规约结果物化。若删掉 mm，`rows` 仍因作为输出而需要独立保存；若不再返回 rows，mm 仍要求在调用前拿到一份可传入的 rows 数据。两项消费者需求任留其一，本例的这个物化理由仍成立。

## 先用本层规则判断三个变化

下面三题只需要本篇的交接关系，不要求记住内部类名。

1. `a=x+y; b=relu(a); return b*2` 改成同时返回 `a` 和 `b*2`。图优化仍交出两条使用 a 的边；Lowering 不能只把 a 当成 b 内部可随意消失的临时规则，还要保留能交给输出的结果身份；Scheduler 和 Codegen 必须保证该结果在返回前有效。
2. `x.t()` 后面接一个只接受连续输入的已有函数。视图给出的 stride 是 `[1,3]`，不满足这个函数对输入排布的要求；Lowering 需要先准备一份连续数据，Scheduler 再把准备工作排在已有函数调用之前，wrapper 按这个顺序连接两者。
3. 上述 `cond` 的两支 shape 相同，但一支返回 float32、另一支返回 int64。数值个数相同仍不能形成统一接口；Lowering 比较两支输出时会拒绝，而不是让 Scheduler 猜测运行时会选哪种 dtype。

这三题分别改变输出身份、消费者布局要求和控制流接口。能够指出“哪份对象发生什么变化、为什么由哪个阶段处理”，才算掌握本层协作；只背四个组件名称还不够。

## 阅读路线

现在按同一层的四项职责继续阅读：

1. [图优化：为什么 Lowering 前还要反复改 FX 图](01-graph-optimization/README.md)
2. [Lowering：怎样把 FX 节点变成可调度的计算](02-lowering/README.md)
3. [Scheduler：怎样从读写依赖得到执行与融合计划](03-scheduler/README.md)
4. [Codegen：怎样生成 kernel、wrapper 并加载](04-codegen/README.md)

读完这四篇，你应能拿到一张简单 FX 图，指出每个节点引用谁、说明四阶段增加或改变了什么，并判断“一个 kernel”究竟在哪一层形成。更深一层的对象、分支和实现按本目录 [`_readme.md`](_readme.md) 中的“全系列广度优先讲解队列”展开；那份文件是教程编写与续写记录，不是学习正文。
