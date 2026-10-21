# Q421 CPU/Triton 索引与 kernel 生成核心源码备课记录

## 读者、层次与边界

- 读者已从 Q42 能把 N=19 映射到 CPU 16+3 与 Triton `pid*XBLOCK+arange`，从 Q411 知道 LoopBody 怎样把当前索引交给 handler；尚不知道两类 kernel 如何保存范围、生成实际 load/store 索引、组织 body 与完整 kernel 定义。
- 本篇是第 3 层叶子。输入为当前 Cpp/Triton kernel context、LoopBody 重放产生的 SymPy index/CSEVariable、范围树与 wrapper 参数；出口为 C++ load/store 片段或完整 Triton kernel body/定义。
- CPU 完整 loop-nest 选择还涉及 `CppKernelProxy` 和 vector kernel；本篇以 `CppKernel` 的共同标量访问合同为 CPU 核心，再引用 Q42 已实测的 proxy 16+3 出口。Triton 以 range tree、indexing、load/store 和 body/kernel 定义形成闭合职责。若继续完整展开 CppKernelProxy 的候选搜索，应作为同一叶子的后续知识点，不另建“更深入”目录。
- 产品基线 `/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`；CPU 可实跑，Triton 无设备，只能用当前源码、测试和公式实验。

## 初始声明范围

| 平台 | 声明 | 作用 |
| --- | --- | --- |
| CPU `CppKernel` | `set_ranges`、`load`、`store`、`store_reduction` | 建立迭代/规约变量并把线性 index 变成指针访问 |
| CPU `CppKernelProxy` | `codegen_functions`、`codegen_nodes`、`codegen_loops` | 建 scalar 底稿，选择 1D/2D vector 与 tail，把 SchedulerNode 变成最终 loop nest |
| Triton `TritonKernel` | `codegen_range_tree`、`indexing` | 分派范围树并将 SymPy index 变成 Triton offset/mask |
| Triton `TritonKernel` | `load`、`store`、`store_reduction` | 生成带 eviction/cache/mask/broadcast/atomic 条件的访存 |
| Triton `TritonKernel` | `codegen_body`、`codegen_kernel` | 拼接索引、计算、suffix 与完整 decorator/signature/body |
| Triton `TritonKernel` range helpers | `iteration_ranges_ranges_code`、`iteration_ranges_scalar_code`、`iteration_ranges_get_pid`、`_has_constant_mask`、`filter_masks`、`iteration_ranges_codegen_header` | 实际生成 arange/scalar、program id、offset 与可省略/必需 mask |

共 20 个方法。`TritonKernel.indexing` 约 336 行、`codegen_kernel` 约 182 行，`CppKernelProxy.codegen_functions` 约 160 行，仍须完整保留，不能用分支摘要替代；长实现按连续机制分段恢复状态。研究已确认：只讲 `CppKernel.load/store` 无法解释 Q42 实测的 16+3，只讲 `codegen_range_tree` 也无法解释 pid/offset/mask，因此 proxy 三方法与六个 range helper 都属于本叶子声明范围。

## 篇内知识点队列

| 编号 | 完成条件 | 案例与证据 | 状态 |
| --- | --- | --- | --- |
| K01 | 逐句推演 CppKernel ranges 与普通/规约 load/store，明确 pointer、dtype、CSE 与 removed buffer 出口 | N=19 CPU 产物、2×128 reduction | 完成，已复读 |
| K02 | 从 range tree 生成 x/r program/block/mask，代入 N=19 与 2×128，覆盖 constant mask 和 persistent/non-persistent 前提 | 当前源码/测试/公式实验 | 完成，已复读 |
| K03 | 分段讲完整 Triton indexing：precomputed size、simplification、mask、indirect、block pointer、dense/reshape/broadcast 等主要出口 | 多组符号索引推演 | 完成，已复读 |
| K04 | 完整讲 load/store/store_reduction 的 dtype、mask、eviction/cache、atomic、broadcast与失败/断言 | 源码分支 + 测试定位 | 完成，已复读 |
| K05 | 连接 codegen_body/kernel 的参数、decorator、heuristic、代码段顺序与 wrapper 消费；完成声明与理解验收 | N=19 Triton 假定产物 | 完成，已复读 |
| K06 | 完整推演 CppKernelProxy 从 scalar 底稿到 1D/2D vector、masked/scalar tail 与 loop 输出，并完成 20/20 AST 验收 | 实验 13 + cpp.py | 完成，已复读 |

## 覆盖与证据原则

- CPU N=19 与 2×128 reduction 可用实验 13/12 的实际生成代码；CPU 结论不外推 GPU。
- Triton N=19/XBLOCK=8 只证明公式覆盖；真实 block 参数、生成文件和性能没有运行证据。
- 每个长函数先列全部顶层判断与出口，再选能命中机制的具体索引；不能只展示常见 `x0` 直线后把其余分支写成“类似”。
- `indexing` 的内部缓存、mask 合成、block pointer 与 indirect 分支必须分别给状态；最终 AST 等价只作范围证据，不替代讲解。
- actual reader 反馈尚无；作者迁移题必须能从正文推导新 N、不同 stride/offset、原子 mode 或规约形态。

## 当前恢复点

Q421 六个知识点均已完成并复读。声明范围的 20 个方法全部与当前产品源码 AST 等价；正文另外完成 CPU 1D/2D tail、Triton ordinary/block-pointer/atomic、loop/persistent reduction 的状态推演。根队列下一项为 Q431。

## 第五轮事实验收

- 源码范围：从 Markdown 的 Python fence 提取函数，与 commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402` 的 `CppKernel`、`CppKernelProxy`、`TritonKernel` 对照，20/20 AST 相同。教学注释不改变语义；曾发现 `indexing` 的 docstring 空白不同导致 19/20，已恢复产品原文后达到 20/20。
- CPU 实验：`PYTHONPATH=/work/pytorch python3 experiments/13_backend_models.py` 通过，N=19 结果与 eager 一致；既有生成产物显示 16 元素 vector 主循环与 3 元素 tail。它支持当前机器的 CPU 路径，不证明其他 ISA 都采用 16。
- Triton 公式实验：同一脚本用 N=19、XBLOCK=8 得到 pid 三组 0..7、8..15、16..23，mask 后覆盖恰为 0..18。当前环境没有 CUDA/Triton，未声称实际编译或运行 GPU kernel；decorator、block pointer 和 eviction 结论来自当前源码。
- 结构检查：源码前先建立范围树、offset/mask、代码缓冲区与 scalar 底稿的用途；长函数后均有具体状态和条件对照，不以 AST 或实验成功替代理解验收。

## 第五轮作者理解验收

以下回答只使用正文中的机制：

1. N=35、CPU factor=16 时，main 覆盖 0..31，tail `steps=3` 覆盖 32..34；masked dtype 不支持只使 tail 回退 scalar，不取消完整块的 vector kernel。
2. Triton 普通 store 改为 `atomic_add` 后，`block_ptr=mode is None` 变为 False，因此索引回到普通 offset/mask 并生成 relaxed atomic；这不是 block pointer store 的一个字符串变体。
3. 规约 load 从 sum 改为 max 时，地址和 mask 可以不变，但无效 r lane 的 `other` 需改为 max 单位元；最终 store 仍在 suffix，并以临时退出 reduction 状态后得到的外层索引写出。
4. 动态 N 没有 hint 时 size hint 取 8192，只影响 heuristic；实际 numel 仍作为参数，边界正确性仍由 index/mask 决定。
5. 18×18、factor=16 的 2D masked tiling 分成 16×16、16×2、2×16、2×2，后三块分别携带 inner、outer、双 tail size，因此没有重复或遗漏。

实际读者反馈仍未取得；这里登记的是作者复读，不能表述为读者已理解。
