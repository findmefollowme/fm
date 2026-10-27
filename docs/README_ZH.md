# 中文注释代码索引

所有文件都是新增的注释副本；原始代码路径未被覆盖。

- [Kronos 核心模型](annotated/model/kronos.py)：Tokenizer、预测器、自回归推理。
- [底层神经网络模块](annotated/model/module.py)：量化器、Transformer 相关模块。
- [Tokenizer 训练脚本](annotated/finetune/train_tokenizer.py)：重建损失、量化损失、验证与优化器。
- [顺序微调脚本](annotated/finetune_csv/train_sequential.py)：Tokenizer 阶段和预测器阶段的编排。
- [完整中文导读](KRONOS_CODE_GUIDE_ZH.md)：架构流程、张量形状、训练目标和研究边界。

建议先读完整中文导读，再按导读里的顺序打开注释代码。
