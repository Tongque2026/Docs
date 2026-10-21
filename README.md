# PyTorch 源码学习

[Inductor 讲解入口](torch/compile/inductor/README.md) · [当前模块树与阅读顺序](torch/compile/inductor/INDEX.md) · [Inductor 从零教程](torch/compile/inductor-from-zero/README.md)

中文讲解基于本地 PyTorch 源码，重点覆盖 Inductor 的输入输出、内部实现与新设备接入；不以 PyTorch 文档为依据。按层建立流程，再逐层完成叶子实现的完整逐行解释，每层最多四个直接子模块。

设备接入优先范围包括后端注册与调度、模式注册/匹配/子图替换（自定义融合算子）、wrapper、设备状态、后端能力、编译与运行适配，以及布局/物化/alias/mutation、依赖/副作用/生命周期和异步释放。优先范围沿用现有树，不另增目录。系列仍在继续，完成状态与待写范围见索引。

源码基线为 `8332c5622aabd9535969e9a8d0ed60e4a90e4402`。源码链接和实验环境以 `/work/pytorch` 为位置；原生库版本差异和实验限制见各篇及维护记录。真实运行、设备元数据、受控 fixture 和未验证路径分别注明。

仓库保存文档、SVG/PNG 图、图生成器、实验脚本、被引用的生成源码快照与结果记录。编译缓存、动态库和临时构建产物不提交；脚本可重新生成本机产物。

在原源码环境中执行结构、引用、完整源码范围及已采集证据检查：

```bash
python3 torch/compile/inductor/maintenance/verify.py
```

该检查不重新运行全部实验，也不表示待展开源码已经完成讲解。
