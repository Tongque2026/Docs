# PyTorch 源码学习

[Inductor 从零教程](torch/compile/inductor/README.md)

本分支只保留这一版 Inductor 教程。教程面向会使用基础 `torch`、了解
`torch.compile`，但尚不了解 Inductor 内部实现的读者。

正文从整体编译流程开始，依次讲解图优化、Lowering、Scheduler 和 Codegen，
再进入各组件的叶子实现。内容只依据当前 `/work/pytorch` 源码、相关测试和独立实验。

教程目录同时保存相邻 `_readme.md` 备课记录、独立实验脚本和实验产物。
