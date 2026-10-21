# Inductor 教程备课与续写记录

## 2026-10-21 目标读者逐句审阅回修

- 用户要求先提交基线，再由独立 AI 严格扮演最初读者画像逐篇复制正文、逐句批注，随后按反馈回修。
- 基线提交：独立本地仓库分支 `codex_docs_inductor_261021`，commit `81a7e0c docs: add Inductor source tutorial`；没有 push。
- 审阅目录：`/work/learning/1021/56sol/inductor-reader-review/`。审阅者未读取任何 `_readme.md`，只按 BFS 顺序阅读 40 篇正式 README。
- 覆盖结果：6,310 个自然语言句/标题/表格知识行、441 个代码或图示块、6,866 条批注；其中 591 条指出看不懂、部分看懂或表达生僻。40→40 路径核对、逐句覆盖、去批注后逐文件字节一致均通过。
- 回修顺序：先处理会阻断后文的 S1（根总览与四个一级组件）；再处理 S2 的具体对象前提；最后逐叶子把参数/状态表移到长源码之前并清理 S3 抽象表达。当前项：根总览 S1。
- 完成条件：每条问题批注都必须对应正文修订或有证据说明误判；修订后由同一读者画像重新通读，不以词频下降替代理解验收。

## 2026-10-21 第五轮全量回修：以技能原文重新判定

用户再次要求完整阅读技能并重新整改。本轮已经重新完整阅读 `SKILL.md`，并继续读取备课、建立解释、图示与实现、复读方法及 `teaching-shots.md`。此前“第三轮严格回修”“第四轮整改”的标签不再自动构成完成证据；每项仍须回到实际正文，分别通过事实核对和只凭正文的理解推演。

本轮暂停从 Q331 向后扩写。恢复点仍保留为 Q331/K04，但只有完成下面的全量回修顺序后才回到该点：

```text
Q00 总览
  → Q10、Q20、Q30、Q40（全部第 1 层直接模块）
  → Q11…Q44（全部第 2 层模块）
  → 已存在的第 3 层叶子逐篇复核
  → Q331/K04，并继续尚未创建的后续叶子
```

本轮完成状态只接受以下证据组合：

1. **范围证据**：系统图、处理情况表和正文都包含本层主要路径、重要条件及失败/快速出口；只有名称不算覆盖。
2. **过程证据**：至少一个入门例、一个条件对照和需要时的综合例从本层入口走到本层出口，展示具体对象和关键状态变化。
3. **事实证据**：源码、测试、独立实验各自只支持其能够证明的结论；实验结果正确不替代中间机制证据。
4. **理解证据**：作者只依赖正文完成改变关键条件的迁移题，答案必须包含对象、变化和因果；摘抄标题或接口不通过。
5. **叶子附加证据**：声明范围内源码按产品顺序完整保留，教学注释可区分；每个重要分支都有进入条件、状态、出口和具体推演。AST 等价只证明没有漏改语句，不能单独证明讲透。

### 本轮全量盘点

| 范围 | 文件现状 | 本轮首先检查的风险 | 当前判定 |
| --- | --- | --- | --- |
| Q00 | 根正文与记录齐全 | 四阶段对象是否先于调用树建立；快速出口与综合例是否都走到 callable | 已复核，保留本层完成 |
| Q10 | 正文、记录及三个子目录齐全 | 三个处理阶段与 matcher/不变量是否真是同层直接职责；mutation、接受、拒绝、共享 user 是否走完整 | 已复核，保留本层完成 |
| Q20–Q40 | 四篇正文和记录齐全 | 不能沿用旧“严格回修”标签，须逐篇重做范围、案例、迁移题检查 | Q20、Q30、Q40 均已复核 |
| Q11–Q44 | 14 篇正文和记录齐全 | 系统图与直接子职责、案例层次和出口是否一致 | Q11–Q44 全部已完成第五轮复核 |
| 已存在叶子 | 原 14 个叶子已全部完成第五轮复核；Q341 已新建并完成 | 完整源码之外，重要分支状态推演是否充分 | Q111–Q341 均已逐篇复核 |
| 后续叶子 | Q411、Q421、Q431、Q441 均已按队列新建并完成 | 逐篇保留完整源码、状态推演和事实/理解验收 | 全部完成，待全系列审计 |

### Q00 与 Q10 的重新复核结果

- Q00 正文先用 `a=x+y; b=relu(a); return b*2` 建立 FX node 和结果引用，再引入四阶段；逐元素、规约、视图、extern、mutation、cond、空图和 cache hit 都有本层出口。综合例把 permute、pointwise、reduction、extern 与双输出接在一条实际 CPU 路径中。作者仅凭正文可以判断：删掉 `rows` 输出但保留 mm 时，rows 仍因 extern 输入合同而必须可用；同时删掉 mm 和 rows 输出后，原先的物化理由消失，需重新观察消费者。事实与理解两类证据成立，保留“本层完成”。
- Q10 正文把三项直接职责统一为“安排 pass 的处理阶段、模式替换、恢复图不变量”，与三个子目录一致。mutation 例从 pre 的 `add_` 走到 joint/post 的函数式 `add+copy_`；replacement 例分别走通 float32 接受、int64 拒绝和旧 add 仍有第二 user；拓扑/meta/recompile/lint 各有消费者。只凭正文可以判断：若规则只重接旧 mul 的 user，显式 `copy_` 不会因匹配 add/mul 自动消失。保留“本层完成”。
- 实际读者仍未提供逐篇作答；以上只登记作者复读，不写成“读者已经理解”。

当前项：全系列完成审计。Q00–Q441 的正式讲解均已交付。
篇内当前点：无；正在核对目录、队列、链接、源码声明、实验与事实/理解验收记录。
审计通过后：结束本次持续目标；发现缺口则退回所属原篇，不新建“更深入”目录。

## 范围与基线

- 读者已知：基础 PyTorch；知道 `torch.compile`；只听过图优化、Lowering、Scheduler、Codegen 名称。
- 学习目标：能从 FX 图追踪到 Inductor IR、调度节点、kernel 与 wrapper；最终能依据当前源码解释主要路径和关键实现。
- 唯一项目资料：`/work/pytorch` 当前源码、其测试、本目录独立实验；不读取已有教程。
- 源码 commit：`8332c5622aabd9535969e9a8d0ed60e4a90e4402`。
- 工作区已有改动：`aten/src/ATen/native/dlc/SpectralOps.cpp`、`native_functions.yaml` 以及未跟踪教程目录；均不是本教程修改，研究时不据其推断主线。

## 职责树

```text
Q00 Inductor 单图编译主干
├─ Q10 FX 图准备与优化
│  ├─ Q11 pre-grad / joint / post-grad 三个处理阶段
│  ├─ Q12 pattern matcher 的匹配与替换
│  └─ Q13 图合法性、元信息与拓扑维护
├─ Q20 Lowering
│  ├─ Q21 GraphLowering 的 FX 解释与环境
│  ├─ Q22 计算和索引 IR（Pointwise / Reduction / View）
│  ├─ Q23 物化、布局、别名与修改
│  └─ Q24 模板、算法选择、外部调用与控制流
├─ Q30 Scheduler
│  ├─ Q31 IR operation 到 scheduler node 与依赖
│  ├─ Q32 拓扑、死节点、mutation 重命名与生存期
│  ├─ Q33 融合候选、合法性和代价
│  └─ Q34 device backend 分派与 codegen 顺序
└─ Q40 Codegen 与加载
   ├─ Q41 虚拟化 ops 到后端 kernel
   ├─ Q42 CPU C++ 与 GPU/Triton 后端分支
   ├─ Q43 wrapper、内存规划与设备守卫
   └─ Q44 缓存、编译、加载和运行入口
```

## 全系列广度优先队列

| 顺序·编号 | 层次/父项 | 目标 | 前置 | 正文/记录 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 1·Q00 | 0/无 | 解释单图输入输出及四阶段协作 | 基础 torch.compile | `README.md` / 本文件 | 本层完成（第二轮回修） |
| 2·Q10 | 1/Q00 | 图优化内部直接职责协作 | Q00 | `01-graph-optimization/` | 本层完成（第二轮回修） |
| 3·Q20 | 1/Q00 | Lowering 内部直接职责协作 | Q00,Q10 | `02-lowering/` | 本层完成（第五轮复核） |
| 4·Q30 | 1/Q00 | Scheduler 内部直接职责协作 | Q00,Q20 | `03-scheduler/` | 本层完成（第五轮复核） |
| 5·Q40 | 1/Q00 | Codegen 与加载直接职责协作 | Q00,Q30 | `04-codegen/` | 本层完成（第五轮复核） |
| 6·Q11 | 2/Q10 | 三个 FX pass 处理阶段及训练/推理差异 | Q10 | `01-graph-optimization/01-pass-stages/` | 本层完成（第五轮复核） |
| 7·Q12 | 2/Q10 | pattern matcher 完整替换机制 | Q10,Q11 | `01-graph-optimization/02-pattern-matching/` | 本层完成（第五轮复核） |
| 8·Q13 | 2/Q10 | 图元信息、排序和合法性维护 | Q10,Q11 | `01-graph-optimization/03-graph-invariants/` | 本层完成（第五轮复核） |
| 9·Q21 | 2/Q20 | GraphLowering 解释器完整节点交接 | Q20 | `02-lowering/01-graph-interpreter/` | 本层完成（第五轮复核） |
| 10·Q22 | 2/Q20 | Pointwise/Reduction/View 的索引计算 | Q20,Q21 | `02-lowering/02-compute-ir/` | 本层完成（第五轮补缺后复核） |
| 11·Q23 | 2/Q20 | realize、layout、alias、mutation | Q20,Q22 | `02-lowering/03-storage-layout/` | 本层完成（第五轮复核） |
| 12·Q24 | 2/Q20 | template/extern/control-flow | Q20,Q21 | `02-lowering/04-special-calls/` | 本层完成（第五轮复核） |
| 13·Q31 | 2/Q30 | Scheduler 节点与依赖构造 | Q30,Q23 | `03-scheduler/01-dependencies/` | 本层完成（第五轮复核） |
| 14·Q32 | 2/Q30 | 排序、删除、修改和生存期 | Q30,Q31 | `03-scheduler/02-order-lifetime/` | 本层完成（第五轮复核） |
| 15·Q33 | 2/Q30 | 融合判断与代价 | Q30,Q31 | `03-scheduler/03-fusion/` | 本层完成（第五轮复核） |
| 16·Q34 | 2/Q30 | 后端分派与生成顺序 | Q30,Q31 | `03-scheduler/04-backend-dispatch/` | 本层完成（第五轮复核） |
| 17·Q41 | 2/Q40 | ops 虚拟化和 kernel 表达 | Q40,Q34 | `04-codegen/01-kernel-ops/` | 本层完成（第五轮复核） |
| 18·Q42 | 2/Q40 | CPU 与 GPU 后端比较 | Q40,Q41 | `04-codegen/02-backends/` | 本层完成（第五轮复核） |
| 19·Q43 | 2/Q40 | wrapper 和内存计划 | Q40,Q34 | `04-codegen/03-wrapper-memory/` | 本层完成（第五轮补缺后复核） |
| 20·Q44 | 2/Q40 | code cache、加载、运行 | Q40,Q43 | `04-codegen/04-cache-load/` | 本层完成（第五轮补缺后复核） |
| 21·Q111 | 3/Q11 | 三个 recursive pass 入口的完整源码 | Q11 | `01-graph-optimization/01-pass-stages/01-recursive-entry/` | 叶子完成（第五轮复核） |
| 22·Q121 | 3/Q12 | matcher 候选扫描与应用完整源码 | Q12 | `01-graph-optimization/02-pattern-matching/01-apply-source/` | 叶子完成（第五轮复核） |
| 22b·Q122 | 3/Q12 | replacement graph 嵌入完整源码 | Q121 | `01-graph-optimization/02-pattern-matching/02-replace-source/` | 叶子完成（第五轮复核） |
| 23·Q131 | 3/Q13 | stable topological sort 完整源码 | Q13 | `01-graph-optimization/03-graph-invariants/01-stable-sort-source/` | 叶子完成（第五轮复核） |
| 23b·Q132 | 3/Q13 | FakeTensorUpdater 增量更新完整源码 | Q131 | `01-graph-optimization/03-graph-invariants/02-fake-meta-source/` | 叶子完成（第五轮复核） |
| 24·Q211 | 3/Q21 | GraphLowering 输入登记完整源码 | Q21 | `02-lowering/01-graph-interpreter/01-node-entry-source/` | 叶子完成（第五轮复核） |
| 24b·Q212 | 3/Q21 | GraphLowering 算子与常量分派完整源码 | Q211 | `02-lowering/01-graph-interpreter/02-dispatch-source/` | 叶子完成（第五轮复核） |
| 24c·Q213 | 3/Q21 | GraphLowering 输出与输入 mutation 合同完整源码 | Q212 | `02-lowering/01-graph-interpreter/03-output-source/` | 叶子完成（第五轮补缺后复核） |
| 24d·Q214 | 3/Q21 | GraphLowering `run_node` 外围收尾完整源码 | Q213 | `02-lowering/01-graph-interpreter/04-run-node-source/` | 叶子完成（第五轮复核） |
| 25·Q221 | 3/Q22 | Pointwise/Reduction/View 构造核心源码 | Q22 | `02-lowering/02-compute-ir/01-ir-construction-source/` | 叶子完成（第五轮补缺后复核） |
| 26·Q231 | 3/Q23 | realize 与布局决定完整源码 | Q23 | `02-lowering/03-storage-layout/01-storage-source/` | 叶子完成（第五轮补缺后复核） |
| 27·Q241 | 3/Q24 | 特殊调用分派核心源码 | Q24 | `02-lowering/04-special-calls/01-special-source/` | 叶子完成（第五轮复核） |
| 28·Q311 | 3/Q31 | 依赖构造核心源码 | Q31 | `03-scheduler/01-dependencies/01-dependency-source/` | 叶子完成（第五轮补缺后复核） |
| 29·Q321 | 3/Q32 | 排序与 last usage 核心源码 | Q32 | `03-scheduler/02-order-lifetime/01-order-source/` | 叶子完成（第五轮补缺后复核） |
| 30·Q331 | 3/Q33 | 融合主循环完整源码 | Q33 | `03-scheduler/03-fusion/01-fusion-source/` | 叶子完成（第五轮补缺后复核） |
| 31·Q341 | 3/Q34 | backend 分派循环完整源码 | Q34 | `03-scheduler/04-backend-dispatch/01-dispatch-source/` | 叶子完成（第五轮新建并复核） |
| 32·Q411 | 3/Q41 | LoopBody 重放完整源码 | Q41 | `04-codegen/01-kernel-ops/01-replay-source/` | 叶子完成（第五轮新建并复核） |
| 33·Q421 | 3/Q42 | CPU/Triton 索引生成核心源码 | Q42 | `04-codegen/02-backends/01-backend-source/` | 叶子完成（第五轮新建并复核） |
| 34·Q431 | 3/Q43 | wrapper 复用规划完整源码 | Q43 | `04-codegen/03-wrapper-memory/01-planner-source/` | 叶子完成（第五轮新建并复核） |
| 35·Q441 | 3/Q44 | FX Graph Cache 查找完整源码 | Q44 | `04-codegen/04-cache-load/01-cache-source/` | 叶子完成（第五轮新建并复核） |

第二轮整改前位置：原队列继续到了 Q221/K03c2，但“完成”主要依赖源码覆盖、实验和作者结论，未逐篇核对当前层完整案例、迁移推演及作者记录是否具备模块级计划。因此暂停 Q221，新反馈触发全系列回修。

当前项：全系列完成审计；正式 BFS 队列已无待讲项。

## 2026-10-21 第五轮全系列完成审计

本轮按最初目标而不是已有完成标签重新核对：

| 明确要求 | 当前权威证据 | 审计结论 |
| --- | --- | --- |
| 只使用当前源码、测试和独立实验 | `/work/pytorch` HEAD=`8332c5622aabd9535969e9a8d0ed60e4a90e4402`；所有作者记录引用该基线；未使用旧教程作事实依据 | 通过 |
| 从零建立前提与逐层结构 | Q00 先建立 FX node/引用；目录按 Q00→Q10…Q44→20 个叶子嵌套；所有 40 个模块目录均有 README/_readme 配对 | 通过 |
| BFS 明确队列 | 根队列 40 个稳定项全部为“本层完成”或“叶子完成”；239 个篇内 K 行无待讲/进行中 | 通过 |
| 主要情况与完整案例 | 各篇覆盖表分别登记普通、条件、失败/快速出口；主线案例覆盖 pointwise、reduction、view、extern、mutation、control flow、CPU/Triton、wrapper 与 cache | 通过 |
| 叶子完整源码与状态推演 | 20 个叶子正文均含声明源码与具体状态；各相邻记录保存 AST/逐句对照证据、重要分支和迁移题 | 通过 |
| 事实与理解分别验收 | 40 份作者记录均有依据/证据；篇内理解题按对象、变化和因果作答；实际读者反馈均单独注明尚无 | 通过 |
| 独立实验 | 24 个实验脚本通过语法解析，artifacts 目录有 33 份采集产物；第五轮在同一全量会话逐项重跑 01–24，全部 PASS；各结论按实际运行或源码推演分别限定 | 通过 |
| 导航与文件结构 | 80 个 Markdown 文件、40 对 README/_readme；剔除代码块后本地链接均存在；所有父篇包含已写直接子篇入口 | 通过 |
| 产品源码不被教程修改 | PyTorch tracked diff 只有原有 DLC `SpectralOps.cpp` 与 `native_functions.yaml`，教程工作未修改 `/work/pytorch` 产品源码 | 通过 |

审计中发现并修复了 10 个叶子缺少父篇入口的导航缺口；发现 Q431 初始声明漏掉 allocation tree 支撑方法后，将范围从 49 扩为 87 并完成 87/87 AST；Q441 完成 24/24 AST 与跨进程实验。现在正式队列、篇内队列和实际文件状态一致，没有剩余待讲项。

完成审计后，实际读者又指出两类问题：表示执行位置的词不直观，以及正文用作者分类代替了图的实际变化。回修没有只换同义词：Q10/Q11 先画出 pre-grad、joint、post-grad 在实际流水线中的位置，再写明各处实际能看见的对象——`add_`、尚未拆分的前向与反向、拆分后的普通 ATen 单图；入口总览和 Q20 同时把未先解释的“合同/语义”式概括改成具体 shape、dtype、stride、存储与消费者要求。反馈与修订记录保存在 Q11 相邻 `_readme.md`。

最终源码复查还纠正了一项缓存阶段边界：FX Graph Cache 查询发生在 `_compile_fx_inner` 内，pre-grad 与训练路径 joint passes 此时已经运行；hit 跳过的是当前单图的 `fx_codegen_and_compile`，即 post-grad、Lowering、Scheduler、Codegen 和子编译，而不是全部图优化。根总览、Q44 正文及 Q44 作者记录已同步修正。

## 第四轮全量整改判据（2026-10-21）

本轮因用户要求重新完整阅读技能而启动。上表中的“第三轮严格回修”只保留历史记录含义；它不能自动证明对应正文已经满足当前技能。逐篇复核时必须同时找到以下六项，缺一项便将该篇退回“待回修”，不得用链接检查、实验成功或源码行数替代：

1. 相邻 `_readme.md` 先声明读者起点、当前层、输入、出口、直接职责和暂不展开边界；
2. 处理情况从当前源码入口、分派条件、状态和失败出口得到，而非从已有案例反推；
3. 每个知识点用真实或明确标注的推演案例从本层入口走到本层出口；
4. 叶子保持声明范围内全部源码、产品顺序和语义，并加入代码内教学注释与具体状态推演；
5. 事实验收逐项说明源码、测试或独立实验分别证明什么；
6. 理解验收由正文回答迁移题，答案必须包含对象、状态变化和因果，不能只复述类名。

整改恢复点保持 Q241。先完成 Q241 的模块计划和声明范围，再按篇内首个未完成知识点继续；不会因 Q24 上层已完成而跳过叶子实现。

## 处理情况与案例覆盖

| 情况 | 本层变化 | 主案例 | 当前深度 | 后续归属 |
| --- | --- | --- | --- | --- |
| 逐元素链 | FX 多节点 → 延迟位置规则 → 一个融合循环 | add-relu-mul CPU 实验 | 机制已讲并复读 | Q22,Q33,Q42 |
| 广播/类型提升 | 输入位置映射及 dtype 规则变化 | Q22 用具体索引式解释；未单列运行实验 | 机制已讲 | 可扩展实验 |
| 规约 | 增加输出范围、规约范围和组合 | `sum(dim=1)` | 已介绍 | Q22,Q33 |
| 视图/布局 | 逻辑索引与物理索引改变，可共享存储 | transpose | 已介绍 | Q23 |
| 外部调用/模板 | 保留调用或候选，而非普通点式规则 | mm | 已介绍 | Q24,Q34,Q42 |
| mutation/alias | 保护版本、顺序和返回别名 | copy_ 旧读与写回案例 | 依赖建模已讲 | Q23,Q31,Q32,Q43 |
| 动态形状 | 符号尺寸、guards、复用边界 | Q13/Q21/Q44 源码路径 | 机制已讲；未单列动态实验 | 可扩展实验 |
| 控制流/collective | 子图、effect token、特殊调度 | cond 子图案例 | Lowering 已讲 | Q24,Q31 |
| 空图/缓存/AOT | 快速返回、缓存命中、AOT 产物 | Q44 缓存实验与源码分支 | cache 已实测；空图/AOT 为源码边界 | 可扩展专题 |

## 证据记录

- 主干：`torch/_inductor/compile_fx.py` 的 `compile_fx`、`_compile_fx_inner`、`fx_codegen_and_compile`。
- 图优化：`fx_passes/pre_grad.py`、`joint_graph.py`、`post_grad.py`。
- Lowering：`graph.py:GraphLowering`，`lowering.py:lowerings/register_lowering`，`ir.py`。
- 调度：`scheduler.py:Scheduler._init` 和 `_codegen`。
- 生成加载：`graph.py:codegen/_compile_to_module`、`codegen/`、`codecache.py`。
- 实验：`experiments/01_pointwise_pipeline.py`；产物为 post-grad 图、输出代码和结果。实验实际证明三 FX 算子生成一个 CPU kernel；不据此推广所有设备和算子。
- Pattern matcher 实验：`experiments/02_pattern_matcher.py` 与 `artifacts/pattern_matcher.txt`；实际覆盖成功替换、无锚点、常量结构失败、extra_check 拒绝、共享内部节点保留、跨 mutation region 拒绝、重复 PatternExpr 身份约束，以及 apply 的空表/输入类型、CPU fallback、call_module 过滤、逆序多锚点、擦除保护和 counters。
- FX 不变量实验：`experiments/03_graph_invariants.py` 与 `artifacts/graph_invariants.txt`；覆盖稳定拓扑修复、重挂与非法依赖，`recompile()` 前后，FakeTensor shape 传播与 hash 缓存、属性等价重试、缺 fake 参数/不支持 target、alias 改变、当前 dtype 比较漏项，以及 DCE 删除纯 sin/保留副作用 copy_。
- GraphLowering 节点实验：`experiments/04_graph_lowering_nodes.py` 与 `artifacts/graph_lowering_nodes.txt`；覆盖 Tensor、int/float/None/unbacked SymInt placeholder，延迟 Pointwise、output 物化、graph_outputs，以及 bool 直返最小图的当前失败边界。
- 计算 IR 实验：`experiments/05_compute_ir_paths.py` 与 `artifacts/compute_ir_paths.txt`；覆盖同形状 Pointwise、广播 stride-0 ReinterpretView、真实 Reduction、keepdim Reduction 和零 operation 的 transpose view。
- Mutation 实验：`experiments/06_mutation_storage.py` 及 post-grad graph、output code、`mutation_storage.txt`；覆盖旧值输出、原地写回和用户层返回 alias。old 在本例也是图输出，不能用该实验单独证明 copy_ 首次触发物化。
- 物化因果实验：`experiments/18_lowering_materialization.py` 与 `lowering_materialization.txt`；观察普通 pointwise、mm consumer、默认 old-read 顺序，并新增合法重排 add→copy_→mul，实际证明 copy_ 节点处的旧 reader 保护会将延迟 add 物化。
- 特殊路径实验：`experiments/07_special_lowering_paths.py` 与 `special_lowering_paths.txt`；覆盖 mm extern、histc fallback、cond 子图与父图接口。
- Scheduler 依赖实验：`experiments/08_scheduler_dependencies.py` 与 `scheduler_dependencies.txt`；覆盖 MemoryDep、StarDep、WeakDep、producer/users、输入 mutation 版本名。
- Scheduler 排序实验：`experiments/09_scheduler_order_lifetime.py` 与 `scheduler_order_lifetime.txt`；覆盖拓扑序、反向 DCE、ancestors、mutation 真实名与 last_usage。
- Scheduler 融合实验：`experiments/10_scheduler_fusion.py` 与 `scheduler_fusion.txt`；覆盖纵向/横向融合、候选分数与索引、extern、规模拒绝路径。
- Scheduler 分派实验：`experiments/11_scheduler_backend_dispatch.py` 与 `scheduler_backend_dispatch.txt`；覆盖 backend 创建、flush 边界、普通/extern 分派和完成状态。
- 虚拟化 ops 实验：`experiments/12_codegen_virtualized_ops.py` 与 `codegen_virtualized_ops.txt`；覆盖 LoopBody FX、handler 重放、CSE 临时值以及 scalar/vector reduction。
- 后端模型实验：`experiments/13_backend_models.py`、`backend_models.txt` 与 `backend_models_cpu_output.py`；实际覆盖 CPU 宽 16 主循环和 3 元素向量尾部，并独立核算当前 Triton `program_id/arange/xmask` 公式的 19 元素覆盖。当前环境无 CUDA/Triton，GPU 结论限定为源码和公式证据。
- Wrapper 内存实验：`experiments/14_wrapper_memory.py`、`wrapper_memory.txt` 与 `wrapper_memory_output.py`；实际覆盖 mm/pointwise 交替调用、两次原地存储复用、输入/中间值最后使用释放和 tuple 输出。
- 缓存加载实验：`experiments/15_cache_load.py` 与 `cache_load.txt`；在独占临时 cache 下用两个进程得到第一次 FX graph cache miss、第二次 hit，且两次结果一致。
- pass 阶段与递归入口实验：`experiments/16_pass_stage_graphs.py` 与 `pass_stage_graphs.txt`；同一 mutation 请求在 pre hook 保留 `add_`，joint/post hook 为函数式 `add` 加显式 `copy_` 写回；另用 while/cond 嵌套 FX 图实际调用三个产品递归包装器，核对后序顺序、pre 子层 `example_inputs=None`、对象替换写回，以及 joint/post 原地对象身份。
- 训练/推理阶段实验：`experiments/19_pass_training_inference.py` 与 `pass_training_inference.txt`；inference 捕获 joint/post 各一张图，training 捕获 joint 一张以及 partition 后 forward/backward 两张 post 图。

## 样篇复读

- 只凭根正文可回答：四阶段各自输入输出；三 FX 节点何时成为一个 kernel；wrapper 做什么。通过。
- Q10 可回答：三组 passes 所处的处理阶段不同；post-grad 的输入已经 normalized/functionalized；本例 scheduler fusion 不属于 FX pass。通过。
- 案例连贯：root 的 post-grad 图与 Q10 出口一致；Q20 从同一图开始；Q30 以 Lowering operations 为入口；Q40 的代码与实验产物一致。通过。
- 迁移题：若 post-grad 图已经把一组模式替成单个节点，Scheduler 看见的是替换后的 Lowering operations，而不是原节点。正文可以推出。通过。
- 实际读者反馈：尚无。

## 2026-10-21 用户反馈与整改记录

- 原反馈：用户要求“完整阅读一次 skill 的要求，重新整改”。
- 确认缺口：原稿跳过 FX 前提；上层 pattern 案例隐藏具体匹配；Lowering 先堆包装类；布局只列决策函数；综合案例不足；复读题多为结论摘抄；作者验证语气混入正文。
- 状态处理：撤销此前“Q00–Q44 全完成”的结论，整套队列回退为待回修；先按 BFS 修 Q00、Q10、Q20、Q30、Q40，再进入 Q11。
- 已完成回修：根篇新增 FX node/引用前提和 sum/view/mm/mutation 四阶段交接；Q10 使用实际 distribute-mul 规则展示节点重接；Q20 展示规则到 buffer 的具体变化及五类出口；Q30/Q40 新增混合 extern/pointwise 的完整时间线。
- 实际读者反馈：只有本次“需按技能重新整改”，尚无逐篇理解回答；不得以作者复读替代。

### 作者仅依据正文完成的迁移题

1. 若 relu 改为直接读取 x，图中的哪条边改变？答：relu 的输入引用由 add 结果 a 改为输入 x；后续 mul 仍引用 relu 结果。
2. 若 `a=x+y` 同时作为图输出且供 relu 使用，哪项安排可能改变？答：Lowering 必须给 a 保留稳定输出身份；Scheduler/Codegen 不能只让 a 成为仅供 relu 的局部临时值。
3. 若 transpose view 后接只接受连续输入的 extern，问题在哪个交接处理？答：Lowering 的存储/布局职责需物化或转换成符合合同的输入，再由 Scheduler 排序调用。
4. 在 `op0→op1→mm→op2` 中为何 mm 前必须 flush？答：mm 要读取前一 CPU group 产生的 b；不 flush 就还没有可调用的生产程序。
5. 两次独立进程都运行正确，为什么不能只凭耗时说第二次命中 FX cache？答：需像实验一样读取明确的 hit/miss counter；正确结果只证明行为，不证明复用层次。

## 2026-10-21 第二轮全量整改

用户再次要求完整阅读技能并重新整改。本轮已重新逐字读取主技能、备课、解释、图示与实现、复读与教学回修示范；不能继承上一轮的完成判断。

### 全量盘点结论

| 范围 | 已有材料 | 按技能仍缺什么 | 状态处理 |
| --- | --- | --- | --- |
| Q00 根篇 | 四阶段图、逐元素主例、规约/视图/extern/mutation 对照 | 处理地图尚未覆盖控制流、空图/缓存等已列范围；部分阶段主张只在作者记录有证据；需重新做只凭正文的迁移复读 | 进行中 |
| Q10–Q40 | 四个一级模块均有正文和相邻记录 | 多份记录只列知识点与“复读通过”，缺模块所在层、承诺深度、完整覆盖表、案例入口/出口及下一层边界 | 全部待回修 |
| Q11–Q44 | 14 个二级模块均有正文 | 要逐篇检查系统图是否覆盖全部直接职责，案例是否从本层入口走到出口；不能以实验通过代替机制讲解 | 全部待回修 |
| Q111–Q214 | 9 个叶子已有源码和注释 | 源码顺序审计只能证明没漏行；仍须检查每个重要分支是否有条件、关键状态、出口和迁移推演 | 全部待回修 |
| Q221 | 已写 Pointwise、Reduction 及 split 决策的大段源码 | 原稿未完成 multilayer helpers 与 View；此前分段完成记录也须按完整状态推演复核 | 待回修，保留 K03c2 恢复点 |
| Q231–Q431 | 页面均已创建并完成第五轮叶子复核 | 完整源码之外仍逐篇保留状态推演、事实边界与作者迁移题 | 已完成 |
| Q441 | 页面已创建并完成第五轮叶子复核 | 24/24 AST、跨进程 miss/hit、HOO bypass 与迁移题均已验收 | 已完成 |

### Q00 模块级回修计划

本篇位于第 0 层，输入是读者已知的 `torch.compile` 使用经验，出口是能区分 FX graph、Inductor IR operation、scheduler node、kernel/wrapper，并能追踪一个请求在四个一级职责间交接。下一层只打开各职责内部，不在本篇解释 matcher、IR 类字段、融合算法或后端循环实现。

| 知识点 | 要解决的问题与中间变化 | 案例与完成条件 | 状态 |
| --- | --- | --- | --- |
| K00-R1 | FX node、结果引用与四阶段各自接收/交出什么 | `add→relu→mul`；读者可指出换边后语义变化，并逐阶段说出具体对象 | 完成，已复读 |
| K00-R2 | 主要处理情况怎样改变交接内容 | sum、transpose、mm+relu、copy_、cond、无计算图和缓存；每例走到 callable 或明确的本层快速出口 | 完成，已复读 |
| K00-R3 | 多种机制共同出现时怎样贯通四阶段 | 非连续 view + pointwise + reduction + extern/输出身份的综合例；必须先由当前源码或实验确认实际路径 | 完成，已实测并复读 |
| K00-R4 | 只凭正文完成理解验收 | 改输入引用、增加一个图输出、要求连续 extern、改变控制流接口等迁移题；答案含对象、变化和因果 | 完成，已复读 |

Q00 第二轮回修完成；全系列当前项转为同层 Q10。综合案例的稳定事实记录为：输入 `[128,3]` 经 permute/add/relu/sum/mm；实际 Lowering 在 sum 才登记 `ComputedBuffer buf0`，mm 登记 `ExternKernelOut buf1`，输出同时保留两者。实验与数值见 `experiments/17_root_composite.py` 和 `experiments/artifacts/root_composite.txt`。

### K00-R1 / K00-R2 事实与理解验收

- 当前源码证据：`compile_fx.py:_compile_fx_inner` 在 `count_calls==0 and not aot_mode` 时直接返回 `make_boxed_func(gm.forward)`；缓存启用时调用 `FxGraphCache.load`，miss 才由其回调进入完整编译。
- 当前源码证据：`lowering.py` 将 `higher_order.cond` 注册到 `ir.Conditional.create`；`ir.py:Conditional.create` 分别 Lower 两支，并逐项断言 size、stride、device、dtype、offset 一致，随后建立 `Conditional` 和 `MultiOutput`。
- 已有独立实验：`experiments/07_special_lowering_paths.py` 覆盖 true/false 两次运行和父/子图 IR；`experiments/15_cache_load.py` 覆盖两个独立进程的一次 miss 与一次 hit。
- 作者只凭正文复读：若 cond 两支 shape 相同而 dtype 不同，父图无法得到统一输出合同，失败发生在 Lowering，Scheduler 不会根据本次 predicate 猜一支；若纯 view 没有新 operation，Scheduler 不应虚构搬运工作；若 FX cache 命中，实际运行仍发生，只跳过本图的重新编译主流程。这些答案均可由根正文中的具体交接推出。
- 实际读者反馈：尚无。本记录只报告作者复读，不声称读者已经理解。

### K00-R3 / K00-R4 事实与理解验收

- 实际采集：permute 得到 `ReinterpretView`，add/relu 为延迟 `Pointwise`，三步均未登记 operation；sum 后得到 `ComputedBuffer(name=buf0)`；unsqueeze 仍为 view；mm 后得到 `ExternKernelOut(name=buf1)`。最终 `graph_outputs` 是 buf0 与 buf1。
- 数学核对：转置后第一行 `1,4,...,382` 的和为 24512，另外两行为 24640、24768；给定 weight 后结果为 `[49280,49408]`，编译结果与 eager 一致。
- 作者只凭正文迁移：删去 mm 但仍返回 rows，输出身份仍要求 rows 可交回；不返回 rows 但保留 mm，已有调用的输入合同仍要求 rows 在调用前可用；两者都删去时，本篇不能再用这两个理由断言同一物化安排，需观察新的消费者。答案包含改变的消费者、对象身份和原因。
- Q00 完成范围仅为四个一级职责的协作及主要路径。本记录不把 Q10–Q44 的内部机制一并标成完成。

### 第二轮验收规则

- 状态只按正文承诺深度登记：列名或接口不算机制完成；实验通过不算读者理解；叶子无漏行不算重要分支已讲透。
- 每篇 `_readme.md` 必须补齐所在层、输入/出口、直接职责、篇内有序知识点、覆盖表、证据边界、作者复读和实际读者反馈来源。
- 作者迁移题必须只能依赖正文作答，并至少改变一个会影响决定的条件；能从标题抄结论的题不通过。
- 先修 Q00，再按 Q10、Q20、Q30、Q40 处理同层；随后处理 Q11–Q44；最后恢复叶子队列。原 Q221 的源码写作暂停，不以“已经写了很多”改变 BFS 顺序。

## 整改前的一致性审计（不再作为完成判据）

- 结构：20 个队列项 Q00、Q10–Q44 均有读者正文；每个正文目录均有相邻 `_readme.md`。
- 导航：40 个 Markdown 文件，共 86 个本地链接；检查结果 0 个断链。
- 实验：24 个独立脚本均通过语法检查；产物覆盖图优化、Lowering、Scheduler、CPU Codegen、wrapper 内存、跨进程 FX Graph Cache、训练/推理阶段、布局冻结、多模板、fallback 输出树与 while_loop。
- 术语：全文保持 FX node、IR operation、scheduler node、logical buffer/storage、kernel/wrapper 的层次区分。
- 事实边界：CPU 结论来自实际生成和数值验证；GPU/Triton 来自当前源码、当前测试和索引公式求值，明确未作实机运行；动态形状、AOT 与 pool planner 未伪装成实验结果。
- 案例交接：pointwise 主案例贯穿四阶段；reduction、view、mutation、extern/template/control-flow 在对应分支补齐；Q43/Q44 接住 last usage 与最终 callable。
- 遗留范围：动态 shape 多 guard 候选、GPU 实机代码、AOTInductor 部署适合各自成为后续专题，不影响本轮从零理解单图主干和四组件边界。

## 2026-10-21 目标读者逐句复读与整改

- 读者画像：会基础 torch、了解 `torch.compile`，只听过图优化、Lowering、Scheduler、Codegen 四个组件；审阅者不读任何 `_readme.md`，不使用旧教程或作者研究背景补全正文。
- 初次全量副本：`/work/learning/1021/56sol/inductor-reader-review/`。40/40 篇按相对路径复制，逐句、标题、表格行和代码块批注；去批注后逐字节一致。
- 一级复读：`inductor-reader-review-round2/` 与 `inductor-reader-review-round3/`。第二次复读确认具体案例已前置，但仍指出若干作者自用的分类词；正文现已改成三个实际改图位置及 `add_ → add + copy_` 的变化。
- 二级复读：`/work/learning/1021/56sol/inductor-reader-review-level2-round2/`。15/15 篇、2127 条批注、69 个断点；主要问题是首次出现的 handler、ISA、pointer、source line map、动态 guard、fusion score 等没有当前对象和值。
- 叶子复读：`/work/learning/1021/56sol/inductor-reader-review-leaves-round2/`。20/20 篇、4006 条批注、81 个断点，去批注后逐字节一致 20/20。主要问题是数百行源码先于参数/集合/配置状态，以及 planner 多组同名 `plan/codegen` 省略类归属。
- 已实施整改：把作者术语换成实际对象变化；在主要长源码块前增加一次确定调用的输入、集合、配置、分支和预期出口；恢复 planner 方法的类归属；将 AST 数量、写作取舍和证据审计移回作者记录；补齐动态符号、alignment、checkpoint 标签、device stream、缓存候选等消费者。
- 最终全量复读：`/work/learning/1021/56sol/inductor-reader-review-final/`。同一位目标读者复读整改后的 40/40 篇，共检查 7010 个标题、自然段、表格行和代码块；去批注后与正式正文逐字节一致 40/40。该轮提出 21 个剩余断点。
- 修订后复查：`/work/learning/1021/56sol/inductor-reader-review-final/FIX_RECHECK.md`。21/21 已实质修复，部分修复 0、未修复 0、remaining=0；全文正式 README 中“时点”“按语义时机”“语义时机”均为 0 处。
