# 后端分派：把调度节点变成有顺序的生成请求

[返回 Scheduler](../README.md) · [返回总览](../../README.md)

实现叶子：[设备切换、flush 与节点出口](01-dispatch-source/README.md)

融合结束后，Scheduler 手里已有一列顺序合法的节点。每个节点记录自己的 device、计算种类、输入结果、输出结果，以及哪些输入在自己这里最后一次使用。

后端分派沿这列节点走一遍。普通计算交给对应 device 的 backend 形成 kernel；已有函数调用直接写进 wrapper；同时，Scheduler 记录后面的生成步骤已经可以引用哪些结果、已经处理过哪些 operation，以及哪些数据可以安排释放。

这里发生的都是**生成代码**。`completed` 表示某项 operation 已经走过分派，不表示 CPU 或 GPU 已经运行完它。

## backend 积累 kernel，wrapper 排列调用

device backend 接收普通计算。例如 CPU 默认使用 `CppScheduling`。它可以把连续且兼容的节点暂存在一个 kernel group 中，等到边界出现时再结束这组 kernel。

wrapper generator 写最终的调用顺序：准备结果空间、调用生成的 kernel、调用 ATen 等已有函数，以及在合适的位置写释放代码。

```text
普通节点 A ─┐
普通节点 B ─┴─> CPU backend 的当前组 ──flush──> kernel AB 的定义和调用

extern 节点 C ───────────────────────────────> wrapper 中的已有函数调用
```

把 CPU 普通节点交给 `CppScheduling` 时，kernel 可能还没结束。`flush()` 才会让它提交当前组，然后把当前积累的释放请求交给 wrapper。提交后，旧组结束；下一个 CPU 普通节点从空组继续。其他 backend 不一定积累组；当前 CUDA SIMD backend 在一次 `codegen_node` 中直接生成并调用 kernel，也在该调用之后处理释放请求。

以下情况会在继续处理前提交当前组：

- 节点换到另一 device；
- 当前节点是 extern 或 template；
- backend 在接收一个节点后报告当前组已满；
- 所有节点遍历完毕。

Scheduler 为每个完整 device 缓存一个 backend。开始时 `current_device=None`，所以第一个 CPU 节点会触发一次 `None→CPU` 的 flush。此时还没有 backend 工作，也没有释放请求，因而这次 flush 不生成调用。

## 每访问一个节点，状态按固定顺序改变

理解后面的案例需要三份编译期状态：

| 状态 | 保存什么 | 何时改变 |
| --- | --- | --- |
| `available_buffer_names` | Scheduler 建立节点时用来删去“结果已经存在”的未满足依赖；进入分派后继续累计已经越过的结果名 | 节点分派后加入该节点的结果名；本轮分派中不删除 |
| `completed_operations` | 已经走过分派的 operation 名；生成器决定原地复用时据此排除已经处理完的使用者 | 节点分派后加入该节点的 operation 名；本轮分派中不删除 |
| `buffer_names_to_free` | 已到最后一次使用、等待 `free_buffers()` 检查的候选名字 | 分派当前节点前加入；`free_buffers()` 检查后清空 |

对本篇主案例采用的 CPU C++ 普通节点，顺序是：

```text
1. 如果 device 或节点身份要求边界，先 flush 旧组
   └─ backend 提交旧组；free_buffers() 交付旧 pending，并清空 pending
2. pending 加入当前节点的 last_usage
3. backend.codegen_node(node)：逐个成员决定原地复用，mark_run 安排输出空间，再把生成工作放进当前组
4. available 加入当前节点的结果名
5. completed 加入当前节点的 operation 名
6. 若 backend 报告组已满，再 flush 当前组并清空 pending
```

CPU C++ 普通节点在第 3 步已经让 wrapper 安排输出的分配或复用，但读取这些输入的 kernel 调用还没有写出，所以此时不能处理释放候选。等组被提交后再调用 `free_buffers()`，wrapper 中的顺序才会是“准备输出空间，调用 kernel，然后释放它最后读取的数据”。

`free_buffers()` 会逐项过滤候选：图输出不能释放；已经被删除或处理过的名字不重复生成；没有独立可释放存储的逻辑名也不会产生释放语句。检查结束后 pending 集合仍会清空。因此 pending 中出现 `buf0`，只表示它到达最后使用并要接受检查，不保证 wrapper 一定出现一行 `free(buf0)`。

extern 的前两步相同，但第 3 步不同：它直接把已有函数调用写进 wrapper，随后立即调用 `free_buffers()`。因此 extern 当前节点登记的 pending 不必等下一次 Scheduler flush；已有函数调用已经写在它们前面。

`available` 和 `completed` 跨 flush、跨 device 保留。flush 结束的是 backend 当前组，并清空 pending；它不会让 Scheduler 忘掉已经分派的结果和 operation。它们是 Scheduler 的建图与生成记录，不是 wrapper 中某个结果能否被读取的运行时门禁：`available` 的直接依赖裁剪已经在建立调度节点时完成，当前 `_codegen` 循环只继续维护这份集合；`completed` 还会被后续节点的原地复用判断读取。节点依赖和顺序在进入分派前已经确定，backend 按节点 IR 中的结果名生成读写。

## 完整案例：pointwise → mm → sin

看下面的计算：

```python
pointwise = matrix + 1
product = torch.mm(pointwise, weight)
output = torch.sin(product)
```

实验用 `realize` 要求 `matrix + 1` 在进入 mm 前成为有名字的结果。当前 CPU 路径形成四个 operation：

```text
op0：matrix + 1                         写 buf0
op1：realize 读取 buf0，写具名结果       写 buf1
op2：extern mm(buf1, weight)            写 buf2
op3：sin(buf2)                          写 buf3
```

op0 和 op1 已融合成最终节点 `op0_op1`。在 Scheduler 的逻辑记录中，`buf0` 是加法结果，`buf1` 是 realize 后交给 mm 的结果。生成的 wrapper 最终让 `buf1` 复用 `buf0` 的存储，所以运行程序不必真的保留两块相同形状的空间；但分派时两项 operation 和两个结果名仍然存在。

两个函数输入在生成代码中名为：

```text
arg0_1 = matrix
arg1_1 = weight
```

初始状态为：

```text
group = 空
available = {arg0_1, arg1_1}
completed = {}
pending = {}
```

### 访问融合节点 `op0_op1`

它是第一个 CPU 节点。`None→CPU` 先触发空 flush，三份集合不变。然后执行普通节点的固定顺序：

- `op0_op1.last_usage = {arg0_1, buf0}`：matrix 只在加法中读取，buf0 只在组内被 op1 读取；二者加入 pending。
- backend 遍历 op0、op1，决定输出是否复用，并用 `mark_run()` 让 wrapper 安排 buf0、buf1 的空间；随后当前组变成 `{op0, op1}`。
- available 加入 `buf0、buf1`；completed 加入 `op0、op1`。

此时完整快照是：

```text
group = {op0, op1}
available = {arg0_1, arg1_1, buf0, buf1}
completed = {op0, op1}
pending = {arg0_1, buf0}
```

pending 还没有交给 wrapper，因为产生并最后读取它们的普通 kernel 仍在当前组中。

### 访问 extern `op2`

extern 是提交边界。Scheduler 先 flush `{op0, op1}`：backend 写出产生 `buf1` 的 pointwise kernel 及其调用；随后检查 `arg0_1、buf0` 的释放候选并清空 pending。当前生成结果中 buf1 已复用 buf0 的空间，因此 buf0 不需要另一条独立释放语句；matrix 输入 `arg0_1` 的释放排在 pointwise 调用之后。`available` 和 `completed` 不变。

接下来才处理 op2 自己：

- `op2.last_usage = {arg1_1, buf1}`，因此 pending 先变为这两个名字；
- wrapper 写 `mm(buf1, arg1_1) → buf2`；
- extern 路径立即检查 `arg1_1、buf1` 的释放候选，把需要的释放写到 mm 调用之后，并清空 pending；
- available 加入 `buf2`，completed 加入 `op2`。

```text
group = 空
available = {arg0_1, arg1_1, buf0, buf1, buf2}
completed = {op0, op1, op2}
pending = {}
```

集合仍保留旧名字，是因为它们记录“生成过程中曾经可引用或已处理”，不是当前仍存活的运行时 buffer 清单。是否释放由 wrapper 中已经写下的语句表达。

### 访问普通节点 `op3`

op3 与 op2 都在 CPU 上，且 op3 不是 extern 或 template，所以没有前置 flush：

- `op3.last_usage = {buf2}`，pending 变为 `{buf2}`；
- backend 判断 sin 可以让输出 buf3 原地复用 buf2 的空间；wrapper 先写下 `buf3 = buf2` 这项复用安排，再把 op3 放入新的当前组；
- available 加入 `buf3`，completed 加入 `op3`。

遍历结束触发最终 flush。backend 提交原地更新 buf3 的 sin kernel，随后检查 `buf2` 的释放候选。buf2 的空间已经转交给 buf3，而 buf3 是函数输出，因此这里不会再释放同一块空间；检查只清空 pending。

```text
group = 空
available = {arg0_1, arg1_1, buf0, buf1, buf2, buf3}
completed = {op0, op1, op2, op3}
pending = {}
```

wrapper 中与本例有关的空间、调用和逻辑名结束顺序是：

```text
分配 buf0；让 buf1 复用 buf0
pointwise kernel(matrix → buf1)
结束 matrix 的输入引用
    → 分配 buf2
      mm(buf1, weight → buf2)
      结束 weight 和 buf1 的引用
        → 让输出名 buf3 接管 buf2 的空间
          sin kernel 原地更新 buf3
          返回 buf3；这块输出空间不释放
```

“结束逻辑名引用”与“释放物理存储”不是一回事。buf0 的空间继续由 buf1 使用，buf2 的空间继续由 buf3 使用；只有 `free_buffers()` 检查后确认不再被输出或复用关系持有的存储，才会得到真正的释放语句。

## 不同节点身份改变第 3 步的出口

所有节点都先经过“必要时提交旧组”和“登记当前 last_usage”。随后根据身份选择出口：

| 节点身份 | 分派动作 | 当前节点的 pending 何时清空 |
| --- | --- | --- |
| CPU C++ 普通或融合节点 | `CppScheduling.codegen_node(node)`，进入当前组 | 下一次 flush |
| CUDA SIMD 普通或融合节点 | SIMD backend 在 `codegen_node` 内生成并调用 kernel | kernel 调用写完后由 backend 立即清空 |
| extern | wrapper 写已有函数调用 | 调用写完后立即 `free_buffers()` |
| template 或带 epilogue 的模板融合组 | backend 生成并调用模板 kernel | 模板调用写完后立即 `free_buffers()` |
| foreach/combo | backend 生成并调用组合 kernel | 组合调用写完后立即 `free_buffers()` |
| nop | `mark_run()`，不生成 kernel | 自身不清空；等待后续分支内的 `free_buffers()` 或之后的 flush |

身份在进入这次遍历前已经确定。Scheduler 在这里不重新比较实现候选。template 的主体怎样渲染、combo 怎样组成多个子 kernel，属于 Codegen；本层只关心它们已经直接写出调用，因此可以接着交付当前 pending。

foreach/combo 身份本身不会触发 Scheduler 的前置 flush。它只允许交给 SIMD 或 CUDA combined backend；这些 backend 的普通节点和 combo 都在各自的生成调用内写出 kernel 调用并清空 pending，没有一个像 CPU `CppScheduling.kernel_group` 那样等待提交的旧组。于是同一 backend 上的普通 A 后接 combo B 时，A 的调用已在处理 A 时写入 wrapper，B 再写在后面。

例如 SIMD 普通 A 读取 `x` 写 `a`，随后 combo B 的两个子节点读取 `a、y` 并写 `b、c`。A 处理后，wrapper 已有 `kernel_A`，pending 已清空，available 加 `a`、completed 加 A。B 再登记自己的 last usage，写组合调用并清空 pending；返回后，`get_buffer_names()` 汇总两个子节点的 `b、c` 加入 available，`get_operation_names()` 汇总两个子 operation 加入 completed。

`mark_run()` 与 `completed_operations` 不是同一份记录。前者逐个处理 operation 的输出：普通输出会调用 wrapper generator 的 allocation 接口，原地复用输出会调用相应的复用接口；后者只是 Scheduler 在这轮生成中维护的“已经越过分派位置”的名字集合。合法执行顺序在进入本篇前已经排好，后续节点不会因为一个名字没进入 completed 就跳过已经生成的工作。

每条分支返回后，Scheduler 都更新 available 和 completed。template 有一个当前实现的特例。假设融合节点由模板主体 `opT` 和 epilogue `opE` 组成，分别写 `bufT` 和 `bufE`：template backend 在同一次生成中对二者调用 `mark_run()`，所以 wrapper 收到 `bufT`、`bufE` 的分配或复用安排；随后写出的同一个模板 kernel 调用会产生最终的 `bufE`。`_codegen` 返回后只按拆出的模板主体更新公共集合，因此 available 加入 `bufT`，completed 加入 `opT`，不会在这里再加入 `bufE`、`opE`。

若后继节点 `opN` 读取 `bufE`，它的依赖和位置在本轮分派开始前已经确定；backend 从 `opN` 的 IR 读取 `bufE` 这个名字生成代码，不会重新查询 available 来决定是否准许读取。wrapper 也已经在 template backend 内收到 `bufE` 的空间安排和生产调用。因此 available 漏记 epilogue 结果不会丢掉 `bufE` 的分配、生产或后继读取。

completed 漏记 `opE` 会影响后续原地复用判断：当某个候选输入的 users 中仍列着 `opE` 时，判断可能把它当成尚未处理的使用者，从而放弃本可进行的一次存储复用。这是保守地少复用，不是提前覆盖仍需读取的数据；template kernel 已经生成的 epilogue 也不会因此重复分派。

例如后继 `opN` 正在判断能否原地复用输入 `old`，而 `old.users={opE, opN}`，两项都允许原地使用。判断会删掉名字已在 completed 中的 users：

```text
若 completed 含 opE：remaining_users={opN}
    → 只剩当前 opN，可以继续检查并选择复用

当前 template 特例下 completed 不含 opE：remaining_users={opE, opN}
    → 仍有两个使用者，不选择复用，为 opN 另行安排输出空间
```

第二条路径多保留一块空间，却不会覆盖 `old`；所以它影响复用机会，不影响计算顺序和结果。

nop 的作用是完成不需要 kernel 调用的 IR 工作，例如只重新安排已有输入存储的 concat。假设 nop 读取 `part0、part1` 并登记结果 `joined`：`mark_run()` 让 wrapper 安排 `joined` 的存储，分支返回后 available 加 `joined`、completed 加 nop 的 operation 名；`part0、part1` 若属于该节点的 last usage，则留在 pending。这个分支自己不调用 `free_buffers()`，所以这些名字等待下一个清理点：若后继是 extern、template、SIMD 普通节点或 combo，就由那个生成入口末尾的 `free_buffers()` 一并检查；若后继是 CPU C++ 普通节点，则继续积累到下一次边界或最终 flush。这里并不存在一个尚未提交的 nop kernel。

backend 也可以在接收普通节点后用 `ready_to_flush()` 报告“当前组已满”。Scheduler 把这个返回值当作可观察信号；一旦为真，立即执行与其他 flush 相同的“提交当前组、交付并清空 pending”。本篇不需要根据节点形状自行猜 backend 的资源上限。

## device 改变时，当前节点仍在 flush 之后登记

假设已经处理 CPU 普通节点 A、B，下一项是 CUDA 普通节点 C：

```text
处理 A、B
    → CPU group={A,B}，pending 保存 A、B 的最后使用
看到 C 的 device 不同
    → flush CPU：提交 {A,B}，交付并清空旧 pending
    → wrapper 退出旧 guard、进入 CUDA guard
    → 登记 C.last_usage
    → 把 C 交给 CUDA backend；当前 SIMD 路径当场写出 C 的 kernel 调用并清空 C 的 pending
    → 更新全局 available/completed
遍历结束
    → 退出 CUDA guard
    → 最终 flush：处理各 backend 的剩余状态；当前 SIMD 路径没有尚未提交的 C 组
```

device 切换发生在登记当前节点 C 的 `last_usage` 之前。因此旧 device 的 flush 不会误处理 C 的释放请求。available 和 completed 是整个遍历共享的集合，不会因为换 backend 而清空。当前 CUDA SIMD backend 在 guard 已进入后直接写 kernel 调用，所以遍历末尾退出 guard 时，没有一组尚未写出的 CUDA 普通调用留给最终 flush。

guard 也是 wrapper 中将来执行的上下文代码；编译器此刻没有真的进入 GPU。

## 改变条件后重新推演

把主案例中的 extern mm 换成与前后节点兼容的 CPU 普通节点，并先固定 `ready_to_flush()` 始终为 false。访问它之前不再强制 flush：`{op0, op1}` 继续留在当前组，旧 pending `{arg0_1, buf0}` 继续等待；随后并入 mm 的 `last_usage={arg1_1, buf1}`，再并入 sin 的 `last_usage={buf2}`。最终 flush 一次提交整组，并交付 `{arg0_1, buf0, arg1_1, buf1, buf2}`。

再只改变一个条件：若 backend 在接收普通 mm 后让 `ready_to_flush()` 返回 true，Scheduler 就在 mm 分派后提交 `{op0, op1, op2}`，交付并清空当时的 pending；sin 随后从空组开始，在最终 flush 时单独提交并交付 `{buf2}`。这样，两条轨迹的差异只来自一个明确的 backend 返回值。

把 op2 移到另一 device。处理顺序变为：先 flush 旧 CPU 组和旧 pending，切换 guard，再登记 op2 自己的 last_usage，最后交给新 device 的生成入口。extern 身份还会使 op2 的调用写完后立即清空它自己的 pending。

如果 `buf1` 同时是函数输出，`compute_last_usage()` 会从图输出名开始反向计算仍需保留的结果，因此不会把这份输出当作可以在 mm 后释放的普通中间值。分派循环只消费已经算好的 `node.last_usage`；它不会在这里重新决定输出能否释放。

面对一列新节点，可以逐项套用同一个顺序：先看当前节点是否要求在它之前 flush；再登记当前节点的 `last_usage`；按身份生成；更新 available/completed；最后检查 backend 是否要求 flush。只要每次交付后把 pending 清空，就能同时推出 kernel group、wrapper 调用顺序和三份状态。
