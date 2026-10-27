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

## 当前结构

    fmfm/
      src/fmfm/data/       # 交易场所公开数据适配器
      src/fmfm/features/   # 资金流和微观结构特征
      src/fmfm/signals/    # 可解释基线信号
      src/fmfm/backtest/   # 费用后回测
      tests/

## 第一阶段的数据源：Hyperliquid

官方 Info API 支持查询 K 线快照和 L2 订单簿等信息，WebSocket 支持实时订阅行情与成交。历史 K 线端点有每次最多返回 5000 根的限制；抓取时应按时间范围分页，并在断线后补齐缺口。参见 [Info endpoint](https://hyperliquid.gitbook.io/Hyperliquid-docs/for-developers/api/info-endpoint) 和 [WebSocket](https://hyperliquid.gitbook.io/Hyperliquid-docs/for-developers/api/websocket)。

## 安装（开发中）

    cd fmfm
    python -m venv .venv
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    # macOS/Linux: source .venv/bin/activate
    pip install -e ".[dev]"
    pytest

## 重要限制

- K 线只能反映聚合后的价格和成交结果，无法单独还原每一笔资金的意图。
- 资金流特征不是自动盈利保证；交易方向、持仓时间与止损都必须经过样本外测试。
- 当前代码是起始骨架，不是已验证的自动交易机器人；未实现签名交易、钱包私钥管理或真实下单。
