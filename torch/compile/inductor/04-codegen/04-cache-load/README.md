# 缓存、编译与加载：生成代码怎样成为下一次调用

[上一项：Wrapper 与内存计划](../03-wrapper-memory/README.md) · [返回 Codegen](../README.md) · [返回教程总览](../../README.md)

Codegen 最终返回的不是一段打印出来的字符串，而是 `call(args)` 这样的 Python 可调用函数；`args` 是装有本次所有实参的列表。首次编译的主线是：

先区分两个容易混淆的保存对象。`cache key` 是由生成源码内容或 FX 图、输入属性、相关配置等信息计算出的查找标识；它用于定位能否复用旧结果。`artifact` 是按这个标识保存的具体产物，例如 wrapper Python 源码及路径、已编译 C++ 动态库，或恢复 `CompiledFxGraph` 所需的代码与元数据。key 相同只是找到了候选；对应文件缺失时仍要重新生成。

```text
GraphLowering.codegen()
      │ 完整 Python wrapper source
      ▼
PyCodeCache.write(source) ──内容哈希──→ <cache>/<prefix>/<key>.py
      │
      ▼
PyCodeCache.load_by_key_path() ──执行模块顶层──→ module
      │                                      │
      │                                      ├─ 编译/加载 C++ kernels
      │                                      ├─ 准备 Triton/extern 定义
      │                                      └─ 定义 call(args)
      ▼
module.call ──→ CompiledFxGraph.current_callable ──→ boxed 调用入口
```

这里至少有四层不同复用，不能统称为“torch.compile 有缓存”：Dynamo 层可复用捕获图，FX Graph Cache 可复用整个 Inductor 编译结果，PyCodeCache 可复用 wrapper 模块，C++/Triton 子缓存可复用具体 kernel 编译产物。

## 首次路径：生成源码、按内容写入、加载模块

`GraphLowering._compile_to_module()` 依次执行：

1. `codegen()` 生成完整 wrapper source，同时记录“生成文件的这一行来自哪个 FX 节点或源码位置”；这张生成行号到原始位置的对应表就是 source line map，编译报错和调试信息用它把生成代码的位置指回用户程序；
2. `PyCodeCache.write(code)` 对去除首尾空白的源码做哈希，得到 `key` 和 `.py` 路径；
3. 文件不存在时用原子写写入；相同内容映射到相同路径；
4. `load_by_key_path(key,path,linemap,attrs)` 加载 Python 模块，并注入常量；
5. GraphLowering 保存 `cache_key/cache_path/cache_linemap`，返回模块；
6. `compile_to_fn()` 选择模块的 `call` 函数。

生成模块顶层通常包含：

```python
async_compile = AsyncCompile()
cpp_fused_... = async_compile.cpp_pybinding(types, cpp_source)
async_compile.wait(globals())
del async_compile

def call(args):
    ...
```

因此“加载 `.py`”会执行模块顶层：它不仅解析 `call`，也会触发或等待其中登记的 kernel 编译。CPU C++ 源码还有自己的 `CppCodeCache`/扩展加载过程；Triton 也有相应 kernel cache。wrapper 内容哈希相同不意味着每次都重新调用外部编译器。

`PyCodeCache` 还维护进程内 `key → ModuleType` 字典：同一 key 再加载时直接返回模块对象。磁盘文件解决跨进程产物定位，内存字典解决同一进程重复 import；两者是两层命中。

## `CompiledFxGraph` 保存的不是只有函数

正常 JIT 路径用 `module.call` 构造 `CompiledFxGraph.current_callable`。其余字段按用途分成三组：

- **恢复可执行模块：** wrapper 的 `cache_key`、source、line map 和 constants；
- **恢复调用条件：** device 类型/编号、被修改输入及其索引、输出 strides、`fx_kwargs`、输入检查集合和符号 guards；这些决定当前输入能否使用产物以及怎样接回输出；
- **恢复附加状态：** cudagraph 禁用原因和编译期间 metrics/counters 增量；它们恢复诊断与统计，不参与 wrapper 源码内容的哈希命中判断。

`CompiledFxGraph.__call__(inputs)` 只把 boxed list 交给 `current_callable(inputs)`。`_compile_fx_inner()` 最后设置 `_boxed_call=True`，告诉 AOTAutograd/外围调用者该函数要求参数列表，而不是 `fn(*args)`。这与生成 wrapper 的 `def call(args)` 以及入口处 `args.clear()` 对齐。

无计算节点的图是例外：`_compile_fx_inner()` 可直接 `make_boxed_func(gm.forward)`，不必为一个空计算图生成设备 kernel。

## FX Graph Cache：复用 post-grad 之前的单图编译结果

在未强制禁用缓存、启用本地或远端 FX graph cache、且不是 AOT mode 时，`_compile_fx_inner()` 不直接调用 `fx_codegen_and_compile()`，而是进入 `FxGraphCache.load()`。

缓存查询到这里时，pre-grad 已经在 AOTAutograd 之前运行，训练路径的 joint passes 也已经在 partition 前运行。命中省掉的是传给 `fx_codegen_and_compile()` 的这张单图后续工作：post-grad、Lowering、Scheduler、wrapper/kernel Codegen 以及相应编译；它不会倒回去省掉 pre-grad 或 joint。

cache key 由规范化后的 FX Graph、example inputs 中影响代码生成的信息、Inductor 配置/kwargs 和输入检查信息共同哈希。命中流程不是只看一串函数名：

1. 用 key 找到一个或多个序列化 `CompiledFxGraph` 候选；
2. 对每个候选，用当前 backed SymInt hints 求值其 `guards_expr`；
3. guard 为真才可命中；否则继续找候选或重新编译；
4. 命中对象在磁盘中不保存活 callable，因此从它的 `cache_key` 定位 wrapper `.py`；必要时用缓存内保存的 source 重建该文件；
5. 经 `PyCodeCache.load_by_key_path(...).call` 恢复 `current_callable`；
6. 再用真实符号向当前 ShapeEnv 安装对应 guards，并恢复缓存的 metrics/counters。

miss 时才执行完整 `fx_codegen_and_compile`。保存时 `_save_graph()` 先复制一份 `CompiledFxGraph`，只把这份磁盘副本中不可序列化的 `current_callable` 设为 `None`，再保存其余信息、source 和 guard 表达式；当前进程正在返回的原对象仍保留 callable，可以立即运行。

某些图会 bypass。`cond`、`while_loop` 这类节点自身携带子图，PyTorch 把它们称为 HigherOrderOperator；torchbind 节点则引用由 C++ 注册并绑定到 Python 的自定义对象。普通 FX Graph Cache 当前没有承诺把这些额外对象完整序列化和恢复，因此遇到尚未支持的情形会绕过缓存；freezing/runtime constant folding 模式也有各自的 bypass 条件。bypass 不是 hit，也不是“编译失败”，只是本次走完整编译。

独立实验的第三个进程实际编译了 `torch.cond`。它包含 HigherOrderOperator，当前 `_check_can_cache()` 主动抛出 `BypassFxGraphCache`，记录为：

```text
fxgraph_cache_miss=0
fxgraph_cache_hit=0
fxgraph_cache_bypass=1
result=[3.0, 5.0]
```

异常在 `FxGraphCache.load()` 内被捕获，随后照常调用完整 compile function。因此 bypass 计数与正确结果同时出现：它表达“该图不参与这层缓存”，不表达编译或执行失败。

## 跨进程独立实验：第一次 miss，第二次 hit

实验为两个独立 Python 进程指定同一个新建临时 `TORCHINDUCTOR_CACHE_DIR`，每个进程只编译并运行一次相同的 `relu(x+1)*2`：

```text
run 1: fxgraph_cache_miss=1, hit=0, bypass=0
run 2: fxgraph_cache_miss=0, hit=1, bypass=0
两次 result_sum 都是 272.0
```

分成两个进程很重要：如果在同一 `torch.compile` 对象上调用两次，第二次可能早在 Dynamo/运行时 dispatch 层就复用 callable，无法证明 FX Graph Cache 的磁盘命中。该实验的新临时目录确保第一次没有旧条目；退出后只删除实验自己创建的临时缓存，不修改用户现有 Inductor cache。

第二个进程也没有第一个进程的 `PyCodeCache.cache` 内存字典，所以 `hit=1` 证明的是 FX Graph Cache 从共享临时目录恢复编译图；恢复过程中仍要加载对应 wrapper artifact，并在新进程得到新的 `module.call` 对象。

## 本地与远端 FX cache 怎样组成同一次查找

`FxGraphCache.load(..., local, remote)` 至少要求两者之一启用。当前 `_lookup_graph()` 的候选顺序是：

```text
同一个 graph key
   │
   ├─ local=True：遍历本地 key 子目录中的所有序列化候选
   │                 └─ 对每个候选求 guards_expr
   │
   └─ 尚未命中且 remote 可用：remote_cache.get(key)
                     └─ base64 解码、反序列化、求 guards_expr
```

因此本地合法候选会先命中，不再请求后面的远端候选；本地目录不存在、文件损坏或所有 guard 都拒绝时，仍可继续远端。远端当前按 key 取回一份编码后的 serialized `CompiledFxGraph`，它和本地候选一样不能直接携带 callable，命中后仍要恢复 wrapper artifact、`module.call` 和当前 ShapeEnv guards。

若全部候选都不合法，系统执行完整编译。`_save_graph()` 对原 `CompiledFxGraph` 做一份不可调用的磁盘副本，序列化一次：`local=True` 时写入 key 子目录，远端对象存在时把同一内容编码后 `put(key, data)`。两处可以同时保存，不是“远端模式完全替换本地模式”。

远端开关由 `should_use_remote_fx_graph_cache()` 与配置决定。源码中的 `fbcode` 判断表示 Meta 内部构建环境；本教程使用的开源构建不属于它，因此没有显式配置时远端缓存默认关闭。本地实验没有连接远端服务，只验证了本地缓存的 miss、保存和 hit。远端缓存连接或读写失败时，源码会记录日志并继续尝试其他缓存层，最终也可以按 cache miss 重新编译；但真正的 wrapper 或 kernel 编译错误仍会向上报告，不能靠 cache miss 掩盖。

## 动态形状：一个 key 下可能有多个 guard 候选

符号形状不是“任意尺寸都用同一机器码”。例如第一维在图中记为符号 `s0`，本次实际输入长度 32 就为它提供 hint `s0=32`。若某个候选使用没有尾部的 16 元素向量循环，它会保存 `guards_expr: s0 % 16 == 0`；代入 32 得到真，可以命中。输入长度改成 30 时，同一表达式为假，只能继续查找带尾部处理的候选，或重新编译并保存一个新候选。

查找时 cache key 先定位同类候选集合，`guards_expr` 再代入当前这些有具体 hint 的符号整数（backed SymInt）选择可用项。若都不满足，会新编译并保存另一个候选。命中之后还必须把同一项 guard 安装回当前 ShapeEnv，让外层运行时继续保证这个 callable 只接收其有效范围内的输入。

这也解释两种表面相似但原因不同的重编译：Dynamo guard 失败可能重新捕获 FX 图；Inductor FX cache 的候选 guard 失败则在已进入单图编译后无法复用已有 `CompiledFxGraph`。

## Source cache、FX cache 与运行时 dispatch 的职责表

| 层 | key/判据大意 | 命中省掉什么 | 仍需做什么 |
| --- | --- | --- | --- |
| Dynamo/上层图缓存 | Python frame、guards | 重新 trace/capture | 调用已选 compiled callable |
| FX Graph Cache | 已经过 pre/joint 的单图、输入/配置信息 + shape guards | post-grad、Lowering、Scheduler、Codegen 和子编译主流程 | 恢复模块/callable，安装 guards，运行 |
| PyCodeCache 内存 | wrapper source hash key | 重复加载 Python module | 调用 `module.call` |
| PyCodeCache 磁盘 | wrapper 内容哈希 | 重写相同 `.py` | 需要时 import/执行模块 |
| C++/Triton code cache | kernel source + 编译相关参数 | 重复编译具体 kernel | 加载产物并 launch |

缓存正确性的共同原则是：凡是会改变生成代码或运行合同的信息，必须进入 key、候选 guards 或显式 bypass 条件。只按 FX 节点文本缓存会错误复用不同 dtype、stride、device 或配置的产物。

## 从 `torch.compile` 调用到输出的最后一跳

把全教程主线收束起来：

1. Dynamo/AOTAutograd 把一个可编译区域交给 `_compile_fx_inner`；
2. pre-grad 和 joint 已经完成；FX Graph Cache 尝试为当前单图取得合法 `CompiledFxGraph`，miss 才进入 post-grad、Lowering、Scheduler、Codegen；
3. Codegen 写出带 kernel 定义和 `call(args)` 的 wrapper；
4. PyCodeCache 加载模块，`module.call` 成为 current callable；
5. 运行时用 boxed list 调用它；wrapper 做 guards、分配/复用、device/stream 和 kernel 调用；
6. kernel 按后端执行 LoopBody 语义；wrapper 返回 tuple；外围再恢复用户所见的调用/输出结构。

所以四个组件不是四个独立“优化器”：图优化确定规范化计算，Lowering 把节点变成带布局与索引的 IR，Scheduler 把 IR 排成合法且尽量融合的生成单元，Codegen 将这些单元、存储时间线和调用合同固化为可缓存、可加载、可运行的模块。

## 继续阅读实现

[FX Graph Cache 查找源码：key、候选 guard、artifact 与 boxed call](01-cache-source/README.md)
