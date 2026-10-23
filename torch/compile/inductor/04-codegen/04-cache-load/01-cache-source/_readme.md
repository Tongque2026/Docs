# Q441 FX Graph Cache 查找完整源码备课记录

## 读者、层次与边界

- 读者已从 Q44 区分 Dynamo cache、FX Graph Cache、PyCodeCache 与 kernel cache，也走过跨进程 miss/hit 和 cond bypass；尚不知道 key 怎样稳定生成、同 key 多候选怎样验证、磁盘对象怎样恢复 callable。
- 本篇是第 3 层最后一个叶子。入口为 FX GraphModule、example inputs、fx kwargs、inputs_to_check 以及 local/remote 开关；出口为可调用的 `CompiledFxGraph`，并已把候选 guards、metrics 和 post-compile 合同接入当前进程。
- PyCodeCache import 实现、C++/Triton 编译器缓存、cudagraph 内部算法和 remote service 实现不在本篇展开；本篇完整解释 FxGraphCache 对它们的调用与失败出口。
- 唯一依据为 `/work/pytorch` commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402`、当前测试和实验 15。

## 声明范围

完整保留 24 个函数/方法：

1. key 规范化：`_ident`、`extract_tensor_metadata_for_cache_key`、四个 `_reduce_*`；`FxGraphCachePickler.dumps/get_hash/debug_lines`；`FxGraphHashDetails.__init__/debug_lines`；`compiled_fx_graph_hash`，共 12 个。
2. 查找编排：`FxGraphCache` 全部 10 个方法：目录、SymInt/ShapeEnv、lookup、post-compile、save、bypass check、load、clear。
3. artifact 合同：`CompiledFxGraph.__init__/__call__`，共 2 个。

这些声明闭合“输入细节→稳定 key→local/remote candidates→guard→wrapper artifact/callable→miss save 或 hit restore→post compile→boxed call”。被调用的 PyCodeCache/cudagraph helpers 只讲当前接口，不复制到本叶子。

## 篇内知识点队列

| 编号 | 完成条件 | 案例 | 状态 |
| --- | --- | --- | --- |
| K01 | 从具体 graph/input/config 建立 hash details，解释 tensor/SymInt/config 规范化与 bypass | 相同值不同对象、s0 hint 改变 | 完成，已复读 |
| K02 | 完整讲 local/remote candidate 顺序与 guards hints，覆盖损坏文件和全拒绝 | 两个 guard 候选 | 完成，22/22 已入源码，已复读 |
| K03 | 完整讲 wrapper artifact 缺失重建、PyCodeCache `.call`、真实符号 guard 回装与 metrics | 跨进程 hit | 完成，已复读 |
| K04 | 完整讲 copy-save、guards_expr、多本地叶子/远端单值、pickle/write failure | miss/save | 完成，已复读 |
| K05 | 完整讲 `_check_can_cache/load/post_compile/CompiledFxGraph` 的 hit/miss/bypass 统一出口 | run1/run2/cond | 完成，已复读 |
| K06 | 24/24 AST、实验 15、链接与迁移题验收；随后执行全系列完成审计 | 完成审计 | 完成，已复读 |

## 处理情况覆盖

| 情况 | 关键状态与出口 | 知识点 |
| --- | --- | --- |
| 稳定 key | graph、输入 metadata/常量值、kwargs、系统与配置序列化 | K01 |
| FakeTensor/real constant/SymInt | metadata、metadata+values、symbol name | K01 |
| 配置或对象不可序列化 | `BypassFxGraphCache` | K01/K05 |
| local 多候选 | 排序遍历文件、逐个反序列化和 guard | K02 |
| remote 候选 | local 未命中后 get/decode/unpickle | K02 |
| 无 guard/guard true/false | 立即命中、命中、继续候选 | K02 |
| wrapper 文件缺失 | 用 source 原子重建；修正 cpp prefix | K03 |
| artifact 加载 OSError | 本层退化为 miss | K03 |
| 保存 | copy、callable=None、pruned guards、local/remote 写入 | K04 |
| freezing/no ShapeEnv/HOO/torchbind | bypass 后完整编译 | K05 |
| miss/hit/bypass | compile+save / restore / compile without cache | K05 |
| post compile | strides、cudagraph 状态、realign | K05 |

## 证据边界与恢复点

- 实验 15 实际覆盖新临时目录 run1 miss、独立 run2 hit、cond bypass，均核对数值；不覆盖 remote、动态多候选或 artifact 丢失。
- remote 顺序、动态 guards、损坏文件/OSError 和保存失败只按当前源码及独立状态推演说明，不声称服务实测。
- K01–K06 全部完成；Q441 叶子完成。根记录保存全系列审计结论。
- 实际读者反馈尚无；作者复读与实验事实分别登记。

## 第五轮叶子验收

- **源码完整性**：声明的 24 个函数/方法全部在正文，Python fence 无语法错误，和当前 `codecache.py` AST 24/24 相同。
- **运行证据**：实验 15 重跑通过；新目录 run1 为 miss=1，独立 run2 为 hit=1，两次 sum=272；cond 为 bypass=1、结果 `[3,5]`。remote、动态多候选和 artifact 丢失仍明确限定为源码推演。
- **过程覆盖**：key reducers、local/remote 顺序、无 guard/true/false、损坏候选、artifact 重建/OSError、copy-save、pickle/write failure、四类 bypass、miss/hit/bypass、post-compile 和 boxed call 均有入口、状态与出口。
- **理解迁移**：正文可推出 hint 改变但 key 不变时的 guard 选择、local artifact 缺失与加载 OSError 的不同结果、remote 失败后的 local 降级、保存失败不破坏原 callable，以及 hit 后仍必须 post-compile。
- **导航**：父篇已增加本叶子入口；本篇返回父篇并链接上一叶子。

实际读者反馈尚无，本状态只表示作者复读与事实核对完成。

## 第五轮事实与理解验收

- **源码**：正文 Python fence 无语法错误；声明的 key 规范化、hash details、FxGraphCache 和 CompiledFxGraph 共 24/24 函数/方法与 commit `8332c5622aabd9535969e9a8d0ed60e4a90e4402` AST 相同。
- **实验**：重跑实验 15，run1 为 miss=1/hit=0/bypass=0，run2 为 miss=0/hit=1/bypass=0，两次 result_sum=272.0；cond 为 bypass=1、结果 `[3,5]`。这证明本地跨进程 miss/hit 与 HOO bypass，不证明 remote 或动态多候选。
- **作者迁移 1**：s0 hint 从 32 改为 19 时 key 可保持，`s0%16==0` 候选 guard 拒绝，继续尾部候选或 miss；符号名字负责归类，guard 负责版本有效性。
- **作者迁移 2**：local artifact 缺失时 source 可重建；PyCodeCache OSError 则 lookup 返回 None 并完整重编译，不能执行 pickle 中的空 callable。
- **作者迁移 3**：remote 构造失败不阻止 local hit；local/remote 都拒绝才 miss。HOO 在形成安全 key 前 bypass，不增加 miss。
- **作者迁移 4**：保存失败不破坏原 compiled_graph callable；compile 本身失败则没有可返回的成功对象，不能由 cache write 容错掩盖。
- **作者迁移 5**：hit 仍执行 post_compile，因为 tracing strides、cudagraph BoxedBool 和 input realignment 属于当前进程/本次输入合同，不是磁盘 artifact 的持久状态。

实际读者反馈仍未取得；以上只登记作者复读。

## 第六轮隔离冷读回修

每轮只将本篇 `README.md` 交给一名 `fork_turns=none` 的全新读者；禁止读取作者
记录、其他教程、源码、实验、旧报告或聊天。提示只限定读前背景，不能用于补正文。

- `140_CACHE_SOURCE_COLD_REVIEW_1.md`（B）：把 `FxGraphCache.load` 的
  check/key/lookup/hit/miss/bypass/post-compile 总控状态机移到篇首；把静态
  metadata、动态符号身份和本次 hint 分为三类；前置连续 run1/run2 跨进程案例；
  解释 HOO 当前为何采用逐类支持前的保守 bypass，而不是只复述检查结果；同步收紧
  后文“shape 已进 hash”的表述。
- `141_CACHE_SOURCE_COLD_REVIEW_2.md`（A）：全新读者可独立复述并迁移完整主线，
  hit/miss/bypass、两次 guard、local/remote、artifact 恢复、保存容错、HOO 与
  post-compile 均闭合。随后修正一项真实控制流：local 写和 remote put 在同一
  `try`，local 先失败时不会继续 remote；同时收紧篇首损坏候选、H0 来源、规范化
  符号身份和 remote 单值边界。

修改后 2 个 Python fence 均可解析；实验 15 重跑通过：run1 miss=1、run2 hit=1、
cond bypass=1，数值结果保持正确。
