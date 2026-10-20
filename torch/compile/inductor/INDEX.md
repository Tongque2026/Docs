# 当前阅读索引

模块身份、状态与顺序来自 [state.json](maintenance/state.json)。本层完成表示已讲清直接子模块交接；只有“叶子·逐行完成”才表示相应完整源码范围已逐行解释。

- [Inductor](README.md)：本层完成
  - [图准备](01-graph-preparation/README.md)：本层完成
    - [属性与布局描述](01-graph-preparation/01-properties/README.md)：本层完成
      - [初次属性建立](01-graph-preparation/01-properties/01-initial/README.md)：叶子·逐行完成
      - [增量属性维护](01-graph-preparation/01-properties/02-incremental/README.md)：叶子·逐行完成
    - [图重写](01-graph-preparation/02-rewrites/README.md)：本层完成
      - [清理与局部顺序](01-graph-preparation/02-rewrites/01-cleanup-order/README.md)：叶子·逐行完成
      - [无效操作与存储边界](01-graph-preparation/02-rewrites/02-noop-storage/README.md)：叶子·逐行完成
      - [模式匹配与子图替换](01-graph-preparation/02-rewrites/03-patterns/README.md)：本层完成
        - 规则构造与登记：待展开
        - 模式与匹配状态：待展开
        - 候选应用与图维护：待展开
      - [批量计算与通信改写](01-graph-preparation/02-rewrites/04-batch-communication/README.md)：本层完成
        - 同类算子批处理：待展开
        - TP 计算与通信组合：待展开
        - DDP 通信分桶与等待安排：待展开
    - [写入语义恢复](01-graph-preparation/03-mutation/README.md)：本层完成
      - [View/scatter 表示与分解](01-graph-preparation/03-mutation/01-view-scatter/README.md)：叶子·逐行完成
      - [存储可写性与原地化](01-graph-preparation/03-mutation/02-storage-reinplace/README.md)：叶子·逐行完成
      - [自动功能化包装分解](01-graph-preparation/03-mutation/03-functional-wrappers/README.md)：叶子·逐行完成
    - [图收尾](01-graph-preparation/04-finalization/README.md)：本层完成
      - [顺序与入口一致性](01-graph-preparation/04-finalization/01-order-entry/README.md)：叶子·逐行完成
      - [构造节点设备迁移](01-graph-preparation/04-finalization/02-constructor-devices/README.md)：叶子·逐行完成
  - [Lowering](02-lowering/README.md)：本层完成
    - [解释图与登记输入](02-lowering/01-interpretation/README.md)：本层完成
      - [图上下文与符号状态](02-lowering/01-interpretation/01-context-symbols/README.md)：叶子·逐行完成
      - 输入和常量登记：待展开
      - 节点派发与结果维护：待展开
    - [表达计算与布局](02-lowering/02-expressions/README.md)：本层完成
      - 点表达式与类型处理：待展开
      - View、广播与布局索引：待展开
      - 归约与扫描：待展开
    - [物化与副作用](02-lowering/03-materialization/README.md)：本层完成
      - 存储实现与复用提示：待展开
      - 输出与布局约束：待展开
      - 写入与旧值保护：待展开
    - [专用算子路径](02-lowering/04-special-ops/README.md)：本层完成
      - [外部调用与 fallback 协议](02-lowering/04-special-ops/01-external-fallback/README.md)：本层完成
        - 注册、派发与布局约束：待展开
        - 参数物化与输出描述：待展开
        - 调用生成与副作用：待展开
      - [模板与复合算子接入](02-lowering/04-special-ops/02-template-integration/README.md)：本层完成
        - 矩阵与卷积的候选接入：待展开
        - 列表与复合计算：待展开
        - 用户 kernel 与专用调用：待展开
      - 条件与循环子图：待展开
  - [调度与融合](03-scheduling/README.md)：本层完成
    - [依赖与合法顺序](03-scheduling/01-dependencies/README.md)：本层完成
      - 读写提取与调度节点：待展开
      - 别名、写入与符号依赖：待展开
      - 排序、活跃性与祖先关系：待展开
    - [融合与分组](03-scheduling/02-fusion/README.md)：本层完成
      - 候选与优先级：待展开
      - [合法性与后端判断](03-scheduling/02-fusion/02-legality-backends/README.md)：本层完成
        - 通用约束与读写匹配：待展开
        - 设备融合判断：待展开
        - 环检测与融合收益：待展开
      - 合并与分组维护：待展开
    - [生命周期与生成驱动](03-scheduling/03-lifetime/README.md)：本层完成
      - 最后使用分析：待展开
      - 生成遍历与 flush：待展开
      - 释放与局部缓冲区消除：待展开
  - [代码生成与运行封装](04-codegen-runtime/README.md)：本层完成
    - [设备 kernel 与方案选择](04-codegen-runtime/01-kernels/README.md)：本层完成
      - [后端与循环组织](04-codegen-runtime/01-kernels/01-backends-loops/README.md)：本层完成
        - [后端注册与能力查询](04-codegen-runtime/01-kernels/01-backends-loops/01-registration/README.md)：叶子·逐行完成
        - [调度实例与后端委托](04-codegen-runtime/01-kernels/01-backends-loops/02-dispatch/README.md)：叶子·逐行完成
        - 循环展开与 kernel 分组：待展开
      - [元素表达式与目标代码](04-codegen-runtime/01-kernels/02-expressions-code/README.md)：本层完成
        - ops 解释与类型语义：待展开
        - 参数索引与访存：待展开
        - 临时值复用与生成区：待展开
      - [模板候选与方案选择](04-codegen-runtime/01-kernels/03-template-selection/README.md)：本层完成
        - 候选对象与模板生成：待展开
        - 预编译、校验与计时：待展开
        - 计时缓存与结果交付：待展开
    - [Wrapper 与存储计划](04-codegen-runtime/02-wrapper/README.md)：本层完成
      - [入口与设备调用](04-codegen-runtime/02-wrapper/01-entry-device-calls/README.md)：本层完成
        - 入口参数与程序装配：待展开
        - 设备上下文与 stream：待展开
        - kernel 调用与启动参数：待展开
      - 存储计划与复用：待展开
      - 输出、别名与子图：待展开
    - [编译与加载](04-codegen-runtime/03-build-load/README.md)：本层完成
      - [编译任务与等待](04-codegen-runtime/03-build-load/01-tasks-wait/README.md)：本层完成
        - 编译入口与后端任务：待展开
        - 线程池、进程池与 worker：待展开
        - Future 解析与 scope 交接：待展开
      - [源码和二进制缓存](04-codegen-runtime/03-build-load/02-source-binary-cache/README.md)：本层完成
        - 缓存键、路径与原子写入：待展开
        - 编译条件与锁内构建：待展开
        - 后端产物与进程内复用：待展开
      - [JIT 与 AOT 装载](04-codegen-runtime/03-build-load/03-jit-aot-loading/README.md)：本层完成
        - JIT 模块与入口：待展开
        - AOT 产物构建：待展开
        - AOT 容器装载与运行：待展开
    - [缓存与运行适配](04-codegen-runtime/04-cache-runtime/README.md)：本层完成
      - 键与候选适用性：待展开
      - 恢复与持久化：待展开
      - [调用适配与 CUDA Graph](04-codegen-runtime/04-cache-runtime/03-call-adaptation/README.md)：本层完成
        - 编译后策略与运行状态：待展开
        - 输入对齐与调用转交：待展开
        - 捕获、重放与 Graph Trees：待展开

按用户要求，优先展开支持新设备相关模块：后端注册与调度、图重写的模式注册/匹配/子图替换（自定义融合算子）、wrapper 设备调用、Lowering 设备状态、后端能力及编译加载，以及布局/物化/alias/mutation、依赖/副作用/生命周期与异步释放。已完成当前图收尾；设备相关子树优先，其余范围保留。


当前待写顺序（设备接入优先，其余保持广度优先）：

1. 点表达式与类型处理（表达计算与布局）〔设备接入优先〕
2. View、广播与布局索引（表达计算与布局）〔设备接入优先〕
3. 归约与扫描（表达计算与布局）〔设备接入优先〕
4. 存储实现与复用提示（物化与副作用）〔设备接入优先〕
5. 输出与布局约束（物化与副作用）〔设备接入优先〕
6. 写入与旧值保护（物化与副作用）〔设备接入优先〕
7. 读写提取与调度节点（依赖与合法顺序）〔设备接入优先〕
8. 别名、写入与符号依赖（依赖与合法顺序）〔设备接入优先〕
9. 排序、活跃性与祖先关系（依赖与合法顺序）〔设备接入优先〕
10. 最后使用分析（生命周期与生成驱动）〔设备接入优先〕
11. 生成遍历与 flush（生命周期与生成驱动）〔设备接入优先〕
12. 释放与局部缓冲区消除（生命周期与生成驱动）〔设备接入优先〕
13. 循环展开与 kernel 分组（后端与循环组织）〔设备接入优先〕
14. 规则构造与登记（模式匹配与子图替换）〔设备接入优先〕
15. 模式与匹配状态（模式匹配与子图替换）〔设备接入优先〕
16. 候选应用与图维护（模式匹配与子图替换）〔设备接入优先〕
17. 入口参数与程序装配（入口与设备调用）〔设备接入优先〕
18. 设备上下文与 stream（入口与设备调用）〔设备接入优先〕
19. kernel 调用与启动参数（入口与设备调用）〔设备接入优先〕
20. 通用约束与读写匹配（合法性与后端判断）〔设备接入优先〕
21. 设备融合判断（合法性与后端判断）〔设备接入优先〕
22. 环检测与融合收益（合法性与后端判断）〔设备接入优先〕
23. ops 解释与类型语义（元素表达式与目标代码）〔设备接入优先〕
24. 参数索引与访存（元素表达式与目标代码）〔设备接入优先〕
25. 临时值复用与生成区（元素表达式与目标代码）〔设备接入优先〕
26. 注册、派发与布局约束（外部调用与 fallback 协议）〔设备接入优先〕
27. 参数物化与输出描述（外部调用与 fallback 协议）〔设备接入优先〕
28. 调用生成与副作用（外部调用与 fallback 协议）〔设备接入优先〕
29. 矩阵与卷积的候选接入（模板与复合算子接入）〔设备接入优先〕
30. 列表与复合计算（模板与复合算子接入）〔设备接入优先〕
31. 用户 kernel 与专用调用（模板与复合算子接入）〔设备接入优先〕
32. 候选对象与模板生成（模板候选与方案选择）〔设备接入优先〕
33. 预编译、校验与计时（模板候选与方案选择）〔设备接入优先〕
34. 计时缓存与结果交付（模板候选与方案选择）〔设备接入优先〕
35. 编译入口与后端任务（编译任务与等待）〔设备接入优先〕
36. 线程池、进程池与 worker（编译任务与等待）〔设备接入优先〕
37. Future 解析与 scope 交接（编译任务与等待）〔设备接入优先〕
38. 缓存键、路径与原子写入（源码和二进制缓存）〔设备接入优先〕
39. 编译条件与锁内构建（源码和二进制缓存）〔设备接入优先〕
40. 后端产物与进程内复用（源码和二进制缓存）〔设备接入优先〕
41. JIT 模块与入口（JIT 与 AOT 装载）〔设备接入优先〕
42. AOT 产物构建（JIT 与 AOT 装载）〔设备接入优先〕
43. AOT 容器装载与运行（JIT 与 AOT 装载）〔设备接入优先〕
44. 编译后策略与运行状态（调用适配与 CUDA Graph）〔设备接入优先〕
45. 输入对齐与调用转交（调用适配与 CUDA Graph）〔设备接入优先〕
46. 捕获、重放与 Graph Trees（调用适配与 CUDA Graph）〔设备接入优先〕
47. 输入和常量登记（解释图与登记输入）
48. 节点派发与结果维护（解释图与登记输入）
49. 条件与循环子图（专用算子路径）
50. 候选与优先级（融合与分组）
51. 合并与分组维护（融合与分组）
52. 存储计划与复用（Wrapper 与存储计划）
53. 输出、别名与子图（Wrapper 与存储计划）
54. 键与候选适用性（缓存与运行适配）
55. 恢复与持久化（缓存与运行适配）
56. 同类算子批处理（批量计算与通信改写）
57. TP 计算与通信组合（批量计算与通信改写）
58. DDP 通信分桶与等待安排（批量计算与通信改写）
