# 普通循环以外的 Lowering：选择实现、保留调用、组织子图

[返回 Lowering](../README.md) · [上一篇：存储与数据版本](../03-storage-layout/README.md) · [返回总览](../../README.md)

Pointwise 和 Reduction 适合用“给定索引怎样算”表达，但并非所有算子都应该被拆成通用 loops。矩阵乘已有高度优化的库和模板；某些 ATen 算子没有原生 Inductor lowering，直接调用 eager/ATen 实现更实际；`cond/while_loop` 必须保留运行时分支或循环，而不是编译时任选一条路径。

本篇分别追踪三类结果最终怎样交给 Scheduler：

```text
专用高性能算子
  → 建立候选（extern / Triton / C++ / CUDA template）
  → algorithm selection
  → 选中的 ExternKernel 或 TemplateBuffer

未原生实现但允许执行的算子
  → FallbackKernel
  → wrapper 运行时调用对应 operator

高阶控制流
  → 分别 Lower 子 GraphModule
  → Conditional / WhileLoop + MultiOutput 接口
```

先把两个后文类名落到具体对象上。`ExternKernel` 记录“一次已有函数调用”：调用目标（如 `at::mm_out`）、输入 buffer、输出 buffer 及其 size/stride；Scheduler 会把它当作普通生成循环之外的一项独立工作。`TemplateBuffer` 则保存一份待填的 kernel 模板、输入、输出 layout 和选择出的配置；后端把这些参数填入模板后才得到实际 kernel。

控制流或某些外部调用一次可以产生多个 Tensor，但父图仍要分别引用第 0、1 项。源码用一个调用节点保存共同工作，再用多个 `MultiOutput` 分别表示“从这一次调用的返回结构中取第几项”；它们不是重复执行调用。

独立实验 [`experiments/07_special_lowering_paths.py`](../../experiments/07_special_lowering_paths.py) 在当前 CPU 环境观察到：

```text
aten.mm       → ExternKernelOut
aten.histc    → FallbackKernel → MultiOutput
higher_order.cond
  true graph  → ComputedBuffer(Pointwise add)
  false graph → ComputedBuffer(Pointwise sub)
  parent      → Conditional → MultiOutput
```

## 矩阵乘：Lowering 的工作是选实现，不是手写三重循环

`aten.mm` 注册到 `kernel/mm.py:tuned_mm`。它先用 `mm_args` 确定 m、n、k、输入/layout，再根据设备、配置、shape 和可用后端建立候选列表。

当前实现可能加入：

- `ExternKernelChoice(torch.mm, "at::mm_out")`；
- Triton mm template 的多个 tile/config；
- CUTLASS、CK 或 C++ packed GEMM template；
- autoheuristic 根据 m/n/k、dtype、device 等输入特征为候选打分，可过滤明显不合适的配置或调整候选顺序；留下的候选仍由后续选择过程验证或比较。

候选不是输出 Tensor。它是一份 `ChoiceCaller`，包含输入、候选 layout、可执行 benchmark、hash 和“选中后构造哪个 IR node”的方法。

### Extern 与 template 候选交出不同 IR

| 候选 | `output_node()` | 后续含义 |
| --- | --- | --- |
| `ExternKernelCaller` | `ExternKernelOut` 或 `ExternKernelAlloc` | wrapper 生成已有函数/库调用 |
| `TritonTemplateCaller` | `TritonTemplateBuffer` | 后端渲染专用模板，可接 epilogue |
| 多个候选延迟决定 | `MultiTemplateBuffer` | Scheduler 结合 epilogue 后再确定选择 |

`autotune_select_algorithm` 根据配置可能 benchmark、从 cache 取选择，或保留延迟选择。这里的“autotune”属于实现选择，不是 Scheduler 的普通 pointwise fusion。

这里有三个不同时间点，不能都叫“模板已选中”：

1. `ChoiceCaller` 只是候选，尚未成为图中的输出 node；
2. 普通选择已经调用某个 caller 的 `output_node()`，图里出现具体 ExternKernelOut 或 TemplateBuffer；
3. `MultiTemplateBuffer` 已作为 operation 进入图，但仍保存 `choice_timings`，等 Scheduler 看见是否有合法 epilogue 后再选最快基础实现或连 epilogue 一起比较。

`MultiTemplateBuffer.get_min_choice()` 只返回当前 timings 中最小者；选择 Triton caller 时，`finalize_as_triton_caller` 才把该 caller 的 render function 安装到 buffer。若最终是 extern choice，则后续对应逻辑会把多候选节点完成为外部调用。因而 Scheduler 收到 MultiTemplateBuffer 时仍承担一次真实决定，而不是只生成早已确定的模板。

### 当前 CPU 实验具体选择

输入是 4×4 float32 矩阵与单位矩阵。实际 Lowering 结果：

```text
TensorBox → StorageBox → ExternKernelOut(name=buf0)
operations = [ExternKernelOut]
```

结果等于原矩阵。这个证据只说明当前配置/CPU/shape 选择 external mm；GPU 或打开 max-autotune 后可能产生 template 或多候选，不能把本例选择推广成 `mm` 永远是 ExternKernel。

### 为什么 template 能融合 epilogue

`TemplateBuffer` 记录 layout、inputs 和 `make_kernel_render`，其 read dependencies 先保守表示为对每个输入的 `StarDep`。Scheduler 识别 template node 后调用 `codegen_template(template_node, epilogue_nodes)`；这样矩阵乘后的 pointwise，例如 bias/relu，有机会成为模板 epilogue，而无需把矩阵乘拆成普通 Pointwise。

“有机会”取决于依赖、设备、layout 和模板支持：若 epilogue 不合法或实际选择 extern，relu 仍是独立后继 kernel。实验 07 的 CPU 4×4 mm 只出现 ExternKernelOut；实验 11 中虽把 case 命名为 `template_mm`，实际同样是 ExternKernelOut，二者都没有执行 TemplateBuffer render。

## Fallback：保留 operator 语义，由 wrapper 调用

当前 `lowering.py` 显式为 `aten.histc`、部分 linalg、pool backward、nonzero 等注册 fallback。`make_fallback(op)` 把 target 放入 lowerings；命中后 `FallbackKernel.create` 会：

1. 把真实输入换成只保存 shape、stride、dtype 和 device 的 FakeTensor，再在 FakeTensorMode 下调用 operator；该模式拦截算子执行，只计算输出结构，不读取真实元素；
2. 拆分 tensor 与非 tensor 参数，并保存如何重新组装参数；
3. 建立一个 packed `FallbackKernel` operation；
4. 对每个 Tensor 输出建立 `MultiOutput`，记录如何从 packed return 取出该项；
5. 根据 operator schema 登记 alias 和 mutation。

### histc 的实际对象交接

请求：

```python
x = [0.5, 1.5, 1.8, 3.2]
torch.histc(x, bins=4, min=0, max=4)
```

实际 IR：

```text
FallbackKernel(name=buf0, op=aten.histc.default)
  └─ MultiOutput(name=buf1, indices=对应 Tensor 输出)
```

运行结果为 `[1,2,0,1]`。FallbackKernel 是 operation；MultiOutput 让后续图像使用普通 buffer 一样引用这项输出。

### fallback 不是 eager graph break

整张 Inductor 图仍继续编译。Scheduler 把 FallbackKernel 当 extern node，wrapper 在正确位置调用 operator；它的前后仍可有 Inductor kernels。边界处必须物化输入，因为外部调用不能接收一条尚未执行的 Pointwise Python closure。

这与 Dynamo graph break 不同：graph break 会把捕获图切开，回到 Python 再开始另一张图；fallback 是同一已编译图内部的一项 external operation。

### 为什么 fallback 仍要理解 schema

外部调用可能返回 tuple/list，可能 alias 输入，也可能 mutation。FallbackKernel 检查 operator schema：函数式 op、view、in-place op和可 auto-functionalize 的 mutation 有相应处理；无法安全表达的 mutable/alias 组合会报 `NotImplementedError`，而不是假装成纯函数调用。

因此 fallback 省去的是原生计算实现，不省去 shape/layout/依赖/副作用建模。

## cond：两张子图都编译，运行时只选一张

实验：

```python
def true_fn(v):  return v + 1
def false_fn(v): return v - 1
return torch.cond(pred, true_fn, false_fn, [x])
```

输入 x=`[2,4]`。同一编译函数运行：

```text
pred=True  → [3,5]
pred=False → [1,3]
```

FX 图中的 true_fn/false_fn 作为 `get_attr` GraphModule 被 `GraphLowering.get_attr` 转成 `ir.Subgraph`，尚未立即摊平进父图。

### Conditional.create 的完整交接

1. predicate 与 operands 先 `realize_input`，因为父级控制节点需要真实接口；
2. 对 true/false Subgraph 分别调用 `make_subgraph` 建立子 GraphLowering；
3. 在各自 graph handler 上运行子图 Lowering；
4. 取得两支 `graph_outputs`；
5. 检查输出数量、size、stride、device、dtype、offset 相同；
6. 检查当前实现支持的 branch output alias 条件；
7. 父图注册一个 `Conditional` operation；
8. 每项分支结果用 `MultiOutput` 暴露给父图后续 users。

实验轨迹：

```text
true_graph_0  : add → ComputedBuffer
false_graph_0 : sub → ComputedBuffer
root          : cond → Conditional(buf0) → MultiOutput(buf1)
```

父图不是把 add 和 sub 同时执行再二选一。`Conditional.codegen` 让 wrapper 生成运行时条件，只执行 predicate 选中的子图；但编译期必须 Lower 两支并确认接口等价，才能让父图把输出当成统一值使用。

当前实现会检查两支输出数量以及每项的 size、stride、device、dtype 和 offset，并拒绝它明确发现的不兼容共享关系。读者在这一层只需据此判断：两支必须能通过同一组父图输出槽位表示；任一项的结构不同，父图就不能把它们当作同一个结果接口。

## while_loop：循环携带值必须形成稳定接口

`higher_order.while_loop` 走相同的“子图不是普通算术 node”原则。Lowering 收到：

- `cond_fn` Subgraph；
- `body_fn` Subgraph；
- `carried_inputs`：每轮更新并传给下一轮的值；
- `additional_inputs`：循环体读取但不作为循环状态更新的值。

`WhileLoop.create` 先物化输入，再分别 Lower cond/body。它检查：

- cond 只有一个输出；
- cond 输出是 bool 标量 Tensor；
- body 输出不能含当前不支持的 alias；
- body 的输出数量和每项 size/stride/device/dtype 必须与 carried inputs 对应，从而能作为下一轮输入。

父图登记 `WhileLoop` extern operation，并用 MultiOutput 代表最终 carried values。运行时 wrapper 反复调用 cond 子图，真时执行 body 并更新 carried values，假时把当前 carried values交回父图。

还有一个零轮循环带来的存储约束：若某个 carried input 本身是 graph input，循环可能一次也不执行，输出就直接是该输入。当前实现将对应 MultiOutput 名加入 `never_reuse_buffers`，防止 wrapper 把可能仍代表调用者输入的存储误作可复用输出空间。

## 三条路径的共同出口

| 路径 | Lowering 保留什么 | Scheduler 看见什么 |
| --- | --- | --- |
| algorithm selection | 候选输入/layout/benchmark 或已选 extern/template | Template scheduler node 或 ExternKernel scheduler node |
| fallback | operator identity、参数结构、输出、alias/mutation schema | ExternKernel scheduler node |
| cond/while | 子 GraphLowering、父级输入、统一输出接口 | Conditional/WhileLoop extern node及 MultiOutput |

它们都不是普通 Loops，但仍必须拥有明确的 inputs、outputs、layout、device、依赖和副作用。Scheduler 不需要重新理解 histc 数学或 cond 分支源码；它只需按这些接口安排 external/template/control-flow node。

## 用新情况检查理解

若 `torch.cond` 的 true branch 返回连续 `[2,3]` Tensor，而 false branch 返回同 shape 但 stride `[1,2]` 的 view，数值类型和 shape 相同仍不够；当前 `Conditional.create` 会在 stride 等价检查处拒绝。父图不能用一个统一 MultiOutput layout 同时代表两种不一致接口。

若 mm 后接 relu，可能有三种结果：选择 extern mm，relu 作为后继 kernel；选择能接 epilogue 的 template，把 relu 接入模板；或 MultiTemplateBuffer 延迟到 Scheduler 结合 epilogue 再比较。最终路径需观察实际 IR 和选择记录，不能只凭 Python 出现 `mm+relu` 判断。

若 while body 将一个 carried `[2,3]` tensor 转置为 `[3,2]` 后返回，即使元素数仍为 6，也不能成为下一轮的同一 carried slot：size 和 stride 合同都不相等，`WhileLoop.create` 会在父 operation 注册前失败。若 body 返回两项而 carried 只有一项，则更早在数量检查失败。

至此 Lowering 的第二层四项直接职责全部覆盖：解释图、表达计算索引、处理存储/数据版本、接入特殊实现与控制流。广度优先队列下一项进入 Scheduler 的依赖构造。

## 继续阅读实现

[特殊调用分派源码：模板、extern 与控制流出口](01-special-source/README.md)
