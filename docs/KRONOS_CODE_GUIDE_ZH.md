# Kronos 源码中文导读（FM 项目）

> 本文对应仓库 `findmefollowme/fm` 的 `master` 源码。中文注释版本放在 `docs/annotated/` 下，独立于原始文件；没有覆盖原始模型代码。  
> 当前分支：`chinese-comments`

## 1. 先用一句话理解 Kronos

Kronos 不是直接从 K 线数字预测下一个收盘价的普通回归模型。它先把一段行情压缩成离散 token，再用自回归 Transformer 预测未来 token，最后把 token 解码回连续的行情特征。

整体流程：

```text
历史 OHLCV / 其他输入特征
          │
          ▼
  KronosTokenizer.encode
          │
          ▼
   两段离散 token：s1、s2
          │
          ▼
 Kronos Predictor（时间嵌入 + Transformer）
          │
          ▼
 逐时间点预测 s1，再条件预测 s2
          │
          ▼
  tokenizer.decode
          │
          ▼
未来行情特征（再做反归一化）
```

## 2. 重要文件地图

| 文件 | 主要作用 | 什么时候看 |
|---|---|---|
| `model/kronos.py` | `KronosTokenizer`、`Kronos` 预测器及自回归推理 | 理解完整模型流程，优先读 |
| `model/module.py` | 量化器、Transformer block、时间嵌入、层级 token 嵌入、输出头等底层组件 | 想知道每个模块内部原理时读 |
| `model/__init__.py` | 统一导出模型类 | 了解外部如何导入模型 |
| `finetune/train_tokenizer.py` | Tokenizer 训练与验证循环 | 理解第一阶段训练目标 |
| `finetune_csv/finetune_tokenizer.py` | CSV/自定义数据场景的 Tokenizer 微调逻辑 | 适配自己的数据时读 |
| `finetune_csv/train_sequential.py` | 串联 Tokenizer 与预测器两个微调阶段 | 理解整个微调流程 |
| `finetune_csv/README.md` | 自定义数据训练说明 | 先看运行配置和数据格式 |

## 3. 三个最关键的概念

### A. Tokenizer：行情的“离散编码器”

在 `model/kronos.py` 中，`KronosTokenizer` 大致做四件事：

1. `embed`：把输入的每个时间点从原始特征维度映射到隐藏维度。
2. `encoder`：让序列中不同时间位置的信息相互作用。
3. `BSQuantizer`：把连续表示量化为离散的二值码，并得到整数索引。
4. `decoder` 与 `head`：尝试从离散码还原输入行情。

常见张量形状：

- 输入行情：`[B, T, D]`，B 是批次大小，T 是时间步数，D 是每个时间点的特征数。
- 隐藏表示：`[B, T, d_model]`。
- 离散 token：形状通常保留批次与时间维，最后由量化器产生整数索引或分段索引。
- 重建结果：`[B, T, D]`。

注意：这里的 token 是模型学到的压缩编码，不应直接把某个 token 解释成“恐惧”“贪婪”“吸筹”或“上涨”。要做这种语义解释，需要额外设计实验验证。

### B. BSQ：把连续向量变成离散码

在 `model/module.py` 中，`BinarySphericalQuantizer` / `BSQuantizer` 相关实现负责离散化。直观上，隐藏向量每个维度按正负号变成 -1 或 +1，再组合成整数索引。

量化器不仅关心重建，还会加入 commitment loss 和熵相关正则项，以减少编码器输出与量化码偏离，并鼓励码的使用更有效。训练细节以代码中具体的损失实现为准。

### C. Predictor：预测 token，而非直接预测原始价格

`Kronos` 使用层级 token 表示，通常先预测 `s1`，再在 `s1` 条件下预测 `s2`。时间嵌入把分钟、小时、星期、日期、月份等时间信息加入模型。Transformer 处理上下文后，输出头为 token 候选类别生成 logits。

推理时，`auto_regressive_inference` 逐步生成未来 token；完成后再调用 Tokenizer 解码，并由预测器包装层处理归一化/反归一化。

## 4. 训练流程和损失在优化什么？

### 第一阶段：Tokenizer

`finetune/train_tokenizer.py` 中主要看到：

- `recon_loss_pre`：使用第一段编码重建原始输入的误差。
- `recon_loss_all`：使用完整编码重建原始输入的误差。
- `bsq_loss`：量化器的正则/量化损失。
- `loss`：把重建误差与量化损失组合起来训练。

因此，Tokenizer 阶段的直接目标是学到“可压缩、可重建、适合后续预测”的行情表示；它本身并不是直接以买卖收益或未来涨跌方向为目标。

### 第二阶段：预测器

预测器训练使用 Tokenizer 生成的 token 序列，学习按时间顺序预测后续 token。训练中常见的 teacher forcing 是把真实的前序 token 作为上下文来训练；推理时则必须用模型自己生成的 token 继续往后滚动，因此训练与真实生成会有差异。

### 微调总控

`finetune_csv/train_sequential.py` 中的 `SequentialTrainer` 负责按顺序执行：

1. 加载或微调 Tokenizer；
2. 使用第一阶段得到的 Tokenizer；
3. 训练/微调 Predictor；
4. 保存模型与日志。

“微调”与“从零训练”不同：加载预训练权重后再用自有数据更新参数，一般比从随机初始化开始更现实。是否真正从头训练，要检查配置里是否禁用预训练权重以及相应数据、训练预算是否足够。

## 5. 从哪里开始读，最省力？

建议按这个顺序：

1. 打开 `docs/annotated/model/kronos.py`，先读 `KronosTokenizer` 的 `encode` / `decode`。
2. 接着读 `Kronos.forward`，确认输入的 `s1_ids`、`s2_ids` 和时间戳如何进入模型。
3. 再读自回归推理函数，跟踪“预测 s1 → 预测 s2 → 追加 token → 解码”的循环。
4. 只有遇到不懂的模块时，再跳到 `docs/annotated/model/module.py`。
5. 最后看训练脚本，区分 Tokenizer 的重建损失和 Predictor 的 token 预测损失。

## 6. 需要特别注意的研究边界

- 重建得好，不等于预测得准；预测得准，也不等于策略能赚钱。
- 多次采样后取平均，是一种生成聚合方式，不自动等于经过校准的概率或置信区间。
- 模型默认主要从价格、成交量/成交额等数值序列与时间特征学习规律，并没有天然显式的“市场情绪”“资金意图”或订单流理解模块。
- 如果 FM/FMFM 的研究假设是“蜡烛图是情绪累积的外在表现，情绪背后是资金”，应把它作为待检验的研究假设：增加可观测的情绪/资金流特征，设计基线模型与消融实验，并在时间切分的样本外数据上评估，避免未来数据泄漏。
- 回测至少要考虑手续费、滑点、流动性、交易限制和不同市场阶段；不要只看训练损失或单一时期的收益。

## 7. 本分支新增内容

- `docs/annotated/model/kronos.py`：核心模型中文注释版
- `docs/annotated/model/module.py`：底层模块中文注释版
- `docs/annotated/finetune/train_tokenizer.py`：Tokenizer 训练循环中文注释版
- `docs/annotated/finetune_csv/train_sequential.py`：两阶段微调总控中文注释版
- `docs/KRONOS_CODE_GUIDE_ZH.md`：本中文导读

原始文件仍保留在原路径，方便逐行对照。中文注释旨在解释代码意图，不代表对模型预测效果或交易收益的保证。
