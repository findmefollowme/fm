# FMFM — Funding Flow Market Model

FMFM 是一个面向 Web3 永续合约与去中心化交易市场的量化研究项目。研究主轴只有**资金流与市场微观结构**；不以“情绪”作为模型概念或输入。

> 当前阶段：MVP 骨架 / 研究原型。默认只采集公开行情、计算特征、输出研究信号和纸面回测，不会下真实订单。

## 目标

输入历史 K 线与可获得的资金流相关数据，输出可验证的量化决策：
- 方向：做多 / 做空 / 观望
- 预期持有周期与失效时间
- 入场参考、止损、目标价和风险收益比
- 信号置信度/校准状态（必须由样本外验证获得，不能把模型 softmax 直接当概率）
- 资金费率、手续费、滑点与流动性约束后的预期收益

## 设计原则

1. **先验证数据，再训练模型。** 先做特征和基线，不直接把 Kronos 当成盈利系统。
2. **资金流可观测优先。** 使用逐笔成交方向、主动买卖量、订单簿深度、持仓量、资金费率、清算等实际可获得数据；缺失字段必须显式标记，不用假数据填补。
3. **严格时间切分。** 使用 walk-forward / expanding-window 评估，禁止未来数据泄漏。
4. **先纸面交易。** 没有通过样本外、费用后、多个市场阶段的验证，不开放实盘执行。
5. **交易所适配器隔离。** 第一数据源 Hyperliquid；后续按统一接口增加其他 DEX / 链上市场。
6. **模型和执行分离。** 预测模型给出估计，风险引擎决定是否允许下单。

## 研究路线

- M0 数据契约与历史数据采集
- M1 资金流特征、数据质量检查、基线信号
- M2 时间序列模型（先简单模型，再评估是否采用 Kronos 式 tokenizer + autoregressive Transformer）
- M3 严格样本外回测、手续费/滑点/资金费率、持仓期限评估
- M4 纸面交易与实时监控
- M5 经过明确验收后才考虑实盘执行

## 多时间周期预测

FMFM 将同时研究以下期限：**1 分钟、5 分钟、15 分钟、30 分钟、1 小时、2 小时、4 小时、1 天、1 周、1 个月**。不同期限分别建立标签与评估结果，不把短周期信号简单放大成长期预测。第一版提供统一周期注册表、OHLCV 重采样和多周期未来收益/波动率标签；实际训练前仍需核验数据缺口、周期边界和样本覆盖。月线是日历周期，不等于固定 30 天。

- `src/fmfm/horizons.py`：周期定义与 OHLCV 重采样。
- `src/fmfm/multi_horizon.py`：按多个期限生成训练标签。
- `src/fmfm/targets.py`：未来收益与未来波动率目标函数。

## 数据校验与自动测试

`src/fmfm/data/quality.py` 提供 OHLCV 完整性诊断：重复时间戳、时间顺序、缺失区间、异常 OHLC 和负成交量。诊断只报告问题，不会悄悄修复数据。`.github/workflows/fmfm-tests.yml` 配置了 Python 3.10/3.11/3.12 的自动化测试工作流；需要以 GitHub Actions 实际运行结果为准，不能仅凭工作流文件存在就认为测试通过。

## 数据落盘

`src/fmfm/data/storage.py` 提供 JSONL 原始事件追加写入，以及带 UTC 时间戳、去重和原子替换的 CSV K 线存储。当前先用轻量格式建立可追溯原型；当数据量、并发和查询需求明确后，再评估 Parquet/数据库方案。

## 当前结构

    fmfm/
      src/fmfm/data/       # 交易场所公开数据适配器
      src/fmfm/features/   # 资金流和微观结构特征
      src/fmfm/signals/    # 可解释基线信号
      src/fmfm/backtest/   # 费用后回测
      tests/

## 第一阶段的数据源：Hyperliquid

官方 Info API 支持查询 K 线快照、L2 订单簿、历史资金费率和当前合约状态。**官方文档说明 K 线历史只提供最近 5000 根**，把请求切段不能获取更久远的历史；适配器会拒绝超过 5000 根的请求。资金费率历史使用时间戳分页；当前 open interest 等合约状态是快照，不是历史序列，需要持续采集才能形成历史。WebSocket 适合从现在开始持续保存公开成交和行情事件。参见 [Info endpoint](https://hyperliquid.gitbook.io/Hyperliquid-docs/for-developers/api/info-endpoint) 和 [WebSocket](https://hyperliquid.gitbook.io/Hyperliquid-docs/for-developers/api/websocket)。

## 安装与数据准备

    cd fmfm
    python -m venv .venv
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    # macOS/Linux: source .venv/bin/activate
    pip install -e ".[dev]"
    pytest

### 1. 下载 Hyperliquid 公开 K 线

时间必须包含时区，建议使用 UTC。示例下载 BTC 的一段 1 分钟历史数据：

    fmfm fetch-candles --coin BTC --interval 1m --start 2026-10-24T00:00:00Z --end 2026-10-27T00:00:00Z --output data/BTC_1m.csv

程序会保存 CSV 和对应的 JSON manifest，并输出数据质量报告。官方 API 的 K 线历史只有最近 5000 根；此示例时间范围约 3 天，适用于 1 分钟线。更久远的数据需要历史归档或从现在开始持续采集。

### 2. 开始采集实时公开市场事件

    fmfm collect-live --coin BTC --output data/raw/BTC_events.jsonl --duration-seconds 3600

这会订阅公开成交、L2 盘口、合约上下文和 1 分钟 K 线，将原始消息以 JSONL 追加保存。若希望持续运行直到手动中断，省略 `--duration-seconds`。程序会记录连接、心跳与错误事件；断线重连不等于保证零缺口，必须将连接中断作为数据质量标记。此命令不需要钱包地址、私钥或交易权限。

### 3. 下载历史资金费率

    fmfm fetch-funding --coin BTC --start 2026-01-01T00:00:00Z --end 2026-01-08T00:00:00Z --output data/BTC_funding.csv

资金费率作为独立时间序列存储，后续应按时间对齐到各 K 线周期；不允许把未来时点的费率泄漏到当前特征中。

### 4. 检查已有数据

    fmfm inspect --input data/BTC_1m.csv --expected-frequency 1min

### 5. 生成多周期训练标签

    fmfm build-targets --coin BTC --input data/BTC_1m.csv --source-interval 1m --horizons 1m 5m 15m 30m 1h 2h 4h --output-dir data/targets_intraday

# 单独获取更长的日线历史，支持日/周/月研究
fmfm fetch-candles --coin BTC --interval 1d --start 2023-01-01T00:00:00Z --end 2026-10-27T00:00:00Z --output data/BTC_1d.csv
fmfm build-targets --coin BTC --input data/BTC_1d.csv --source-interval 1d --horizons 1d 1w 1mo --output-dir data/targets_long

默认只生成与输入周期相容的目标。输入为 1m 时可生成所有周期的标签，但最近 5000 根 1m K 线不足以支撑日、周、月模型训练；这些长周期的样本会很少或为空。固定周期只保留完整时间桶，并对跨越数据缺口的目标置空。

### 6. 生成特征并评估一个简单的样本外预测基线

    fmfm build-features --input data/BTC_1m.csv --output data/BTC_features.csv
    fmfm build-targets --coin BTC --input data/BTC_1m.csv --source-interval 1m --horizons 1m 5m 15m 30m 1h 2h 4h --output-dir data/targets_intraday
    fmfm evaluate-ridge --features data/BTC_features.csv --targets data/targets/BTC_1m_targets.csv --target-column forward_log_return_1m --output data/BTC_1m_predictions.csv --min-train-size 500 --test-size 100

Ridge 基线采用扩展窗口训练、按时间向前预测，并只在训练集上拟合特征标准化参数。输出包括 MAE、RMSE、零收益基线 MAE、方向准确率和相关性。它只评估点预测，不等于可盈利交易策略；下一步还要用真实成本、资金费率和滑点做交易层回测。

## 重要限制

- K 线只能反映聚合后的价格和成交结果，无法单独还原每一笔资金的意图。
- 资金流特征不是自动盈利保证；交易方向、持仓时间与止损都必须经过样本外测试。
- 当前代码是起始骨架，不是已验证的自动交易机器人；未实现签名交易、钱包私钥管理或真实下单。
