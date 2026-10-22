# 缓存、编译与加载：生成代码怎样成为下一次调用

[上一项：Wrapper 与内存计划](../03-wrapper-memory/README.md) · [返回 Codegen](../README.md) · [返回教程总览](../../README.md)

上一篇已经得到完整的 wrapper Python 源码，其中定义了 `call(args)`；`args` 是装有本次所有实参的列表。本篇接住这份源码，完成三件事：把源码和 kernel 产物写入缓存，在当前进程加载出可调用的 `module.call`，再把这个函数连同其适用条件装进一个 `CompiledFxGraph` 对象交给运行时。首次调用和后续缓存命中最终都要得到这样的可调用对象。

artifact 指能够单独保存、以后再取回的产物。这里有三种 artifact，它们各有自己的 key，不能混用；FX Graph 候选引用 wrapper，wrapper 模块顶层再取得 kernel：

| 保存对象 | key 根据什么计算 | 保存的 artifact |
| --- | --- | --- |
| FX Graph candidate | FX 图、影响代码生成的输入属性和配置 | 不含活函数的 `CompiledFxGraph` 数据 |
| wrapper Python 模块 | wrapper 源码内容 | `.py` 文件；进程内还保存已加载的 module |
| 一个 C++/Triton kernel | kernel 源码与编译参数 | 动态库或后端编译产物 |

`CompiledFxGraph` 是本篇的交接对象：它的 `current_callable` 字段指向当前进程内真正可调用的 `module.call`，其他字段记录恢复该函数所需的 wrapper source key/source、输入条件和诊断信息。磁盘不能保存一个仍然活着的 Python 函数，所以写入 FX Graph Cache 的副本不含 `current_callable`；命中后必须重新从 wrapper artifact 得到它。下文固定把定位候选集合的标识叫 **graph key**，把定位 wrapper `.py` 的标识叫 **wrapper source key**。这两个 key 会沿 candidate 恢复 wrapper 的链路传播；kernel cache 另用 kernel source 和编译参数计算自己的后端 key，由 wrapper 模块顶层重新查询。

首次编译的主线是：

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
module.call ──→ 构造可调用的 CompiledFxGraph ──→ 本次立即运行
                         │
                         └─复制并清空副本的 current_callable
                                      │
                                      └─按 graph key 保存 candidate
```

这种“把所有实参装进一个 list 再调用”的形式称为 boxed call。它与普通的 `fn(x, y)` 不同：这里调用的是 `fn([x, y])`。本文只追踪进入 Inductor 单图编译后的三层 artifact；更外面的 Dynamo 可以复用已经捕获的图和 compiled callable，末尾职责表再把它列作相邻层。

## 首次路径：生成源码、按内容写入、加载模块

`GraphLowering._compile_to_module()` 依次执行：

1. `codegen()` 生成完整 wrapper source，同时记录“生成文件的这一行来自哪个 FX 节点或源码位置”；这张生成行号到原始位置的对应表就是 source line map，编译报错和调试信息用它把生成代码的位置指回用户程序；
2. `PyCodeCache.write(code)` 对去除首尾空白的源码做哈希，得到 wrapper source key（源码局部变量名为 `key`）和 `.py` 路径；
3. 文件不存在时用原子写写入，避免两个并发进程写同一路径时留下只写了一半的文件；相同内容映射到相同路径；
4. `load_by_key_path(wrapper_source_key,path,linemap,attrs)` 加载 Python 模块；若生成代码引用了没有直接写入源码的 tensor 常量，参数 `attrs` 就携带后文 `CompiledFxGraph.constants` 中的这些对象，并把它们放进模块的全局名字表；
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

因此“加载 `.py`”会执行模块顶层：`async_compile.cpp_pybinding(...)` 先登记一个任务；任务用 kernel 源码和编译参数查询 C++ 子缓存，命中就加载已有动态库，miss 才调用外部编译器；`wait(globals())` 等待任务完成，并把 kernel 函数放回模块的全局名字 `cpp_fused_...`。Triton 任务也先查询自己的 kernel cache。后面的 `call(args)` 才能调用这些全局函数。

`PyCodeCache` 还维护进程内 `key → ModuleType` 字典：同一 key 再加载时直接返回模块对象。磁盘文件解决跨进程产物定位，内存字典解决同一进程重复 import；两者是两层命中。

## `CompiledFxGraph` 保存的不是只有函数

普通的即时编译路径会先由 `GraphLowering` 取得 `module.call`、wrapper source key、source 和 line map；`fx_codegen_and_compile()` 再把这些结果与当前图的输入条件一起构造为 `CompiledFxGraph`。其字段按用途分成三组：

- **恢复可执行模块：** wrapper 的 `cache_key`、source、line map 和 constants；
- **恢复调用条件：** device 类型/编号、被修改输入及其索引、输出 strides，以及稍后用于选择动态尺寸候选的符号条件；这些决定当前输入能否使用产物以及怎样接回输出；
- **恢复附加状态：** 编译期间的统计增量和运行方式限制；它们恢复诊断状态，不参与 wrapper 源码内容的哈希命中判断。

调用链是 `外围 → CompiledFxGraph.__call__(inputs) → current_callable(inputs) → module.call(inputs)`。`_compile_fx_inner()` 最后设置 `_boxed_call=True`，告诉外围调用者该函数要求参数列表，而不是 `fn(*args)`。wrapper 取出局部输入名后执行 `args.clear()`，释放这个临时列表持有的 tensor 引用，让后续内存生存期不再被列表额外延长。

## FX Graph Cache：从单图编译入口复用后续产物

`_compile_fx_inner()` 接到一张准备交给 Inductor 后半程处理的 FX 单图。启用本地或远端 FX Graph Cache 时，它不立即调用 `fx_codegen_and_compile()`，而是先进入 `FxGraphCache.load()`。

这次查询只包住从 `fx_codegen_and_compile()` 开始的工作：对当前单图做剩余图改写，再运行 Lowering、Scheduler、wrapper/kernel Codegen 及相应编译。命中会省掉这整段工作；在单图到达 `_compile_fx_inner()` 之前已经完成的处理不在本次命中范围内。

动态尺寸需要先定义三个词。图里某一维可以不用固定整数，而用符号 `s0` 表示；当前输入长度 32 是这个符号本次的具体值，源码称 hint。候选保存的 `guards_expr` 是适用条件，例如 `s0 % 16 == 0`。ShapeEnv 是外围维护符号及其约束的状态。查询时先把 hint 代入表达式选候选；命中后再把同一条件登记到 ShapeEnv，使外围只在条件成立时使用这个 callable。

graph key 由规范化后的 FX Graph、example inputs 中影响代码生成的信息、Inductor 配置和输入检查信息共同哈希。命中流程不是只看一串函数名：

1. 用 key 找到一个或多个序列化 `CompiledFxGraph` 候选；
2. 对每个候选，把当前符号尺寸的具体 hint 代入其 `guards_expr`；
3. guard 为真才可命中；否则继续找候选或重新编译；
4. 命中对象在磁盘中不保存活 callable，因此从它的 wrapper source key（源码字段名为 `cache_key`）定位 wrapper `.py`；必要时用候选保存的 source 重建该文件；
5. 经 `PyCodeCache.load_by_key_path(...).call` 恢复 `current_callable`；
6. 再把对应条件登记到当前 ShapeEnv，并恢复缓存的 metrics/counters。

miss 时才执行完整 `fx_codegen_and_compile`。代码生成如果依赖某个符号条件，例如选择了没有尾部处理的 16 元素向量循环，就把该选择成立所需的 `s0 % 16 == 0` 记录为 guard；保存候选时，这些条件组成它的 `guards_expr`。`_save_graph()` 先复制一份 `CompiledFxGraph`，只把这份磁盘副本中不可序列化的 `current_callable` 设为 `None`，再保存其余信息、source 和 guard 表达式；当前进程正在返回的原对象仍保留 callable，可以立即运行。

例如第一维记为 `s0`，长度 32 的输入可以使用要求 `s0 % 16 == 0` 的候选；长度 30 代入后为假，只能继续寻找带尾部处理的候选，或重新编译。某一维已被捕获为符号时，graph key 记录的是同一个符号形状问题以及 dtype、stride、device 等属性，不把每次的 32 或 30 都当作新的固定尺寸图；具体范围由候选 guard 区分。

允许缓存、但没找到合法候选，称为 miss；系统判定这类图不应参加该缓存，称为 bypass。`cond`、`while_loop` 这类节点自身携带子图，PyTorch 把它们称为 HigherOrderOperator；torchbind 节点则引用由 C++ 注册并绑定到 Python 的自定义对象。普通 FX Graph Cache 当前没有承诺把这些额外对象完整序列化和恢复，因此遇到尚未支持的情形会绕过缓存。bypass 后直接走完整编译。

另一个独立进程实际编译了 `torch.cond`。它包含 HigherOrderOperator，当前 `_check_can_cache()` 主动抛出 `BypassFxGraphCache`，记录为：

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

把实验里的文件与三层 artifact 对上，可以看到两个进程分别做了什么：

| artifact | 第一个进程 miss 后 | 第二个进程 hit 后 |
| --- | --- | --- |
| FX Graph 候选 | 写入不含活 callable 的候选 | 读取候选并验证 guard |
| wrapper `.py` | 按源码哈希写入并加载 | 从相同路径加载；缺失时可用候选保存的 source 重建 |
| kernel 产物 | 首次编译并写入子缓存 | 子缓存命中时加载，不重新运行外部编译器 |

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

## 各层缓存与运行时 dispatch 的职责表

| 层 | key/判据大意 | 命中省掉什么 | 仍需做什么 |
| --- | --- | --- | --- |
| Dynamo/上层图缓存 | Python frame、guards | 重新 trace/capture | 调用已选 compiled callable |
| FX Graph Cache | 到达单图编译入口的 FX 图、输入/配置信息 + shape guards | 剩余图改写、Lowering、Scheduler、Codegen 和子编译主流程 | 恢复模块/callable，登记 guards，运行 |
| PyCodeCache（进程内） | wrapper source hash key | 重复加载 Python module | 调用 `module.call` |
| PyCodeCache（磁盘 artifact） | 同一个 wrapper 内容哈希 | 重写相同 `.py` | 新进程仍需加载/执行模块 |
| C++/Triton code cache | kernel source + 编译相关参数 | 重复编译具体 kernel | 加载产物并 launch |

缓存正确性的共同原则是：凡是会改变生成代码或运行合同的信息，必须进入 key、候选 guards 或显式 bypass 条件。只按 FX 节点文本缓存会错误复用不同 dtype、stride、device 或配置的产物。

## 从单图编译入口到输出

把全教程主线收束起来：

1. `_compile_fx_inner` 用 graph key 为当前单图查找 FX Graph candidate；
2. hit 时验证 guard 并恢复 wrapper；miss 时才运行剩余图改写、Lowering、Scheduler、Codegen，再保存 candidate；资格检查判定 bypass 时不查找或保存 candidate，转而完整编译；
3. Codegen 写出带 kernel 定义和 `call(args)` 的 wrapper；
4. PyCodeCache 加载模块，`module.call` 成为 current callable；
5. 运行时用 boxed list 调用它，wrapper 依次调用已经加载的 kernel 并返回结果。

## 继续阅读实现

[FX Graph Cache 查找源码：key、候选 guard、artifact 与 boxed call](01-cache-source/README.md)
