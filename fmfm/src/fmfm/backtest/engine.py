"""Small vectorized baseline backtest with explicit next-open timing.

This is a research harness, not an exchange simulator. It assumes the signal
is computed after bar t closes, the position is entered at open[t+1], and
that position earns open-to-next-open returns. Funding, liquidation, order-book
impact and partial fills must be modelled before any performance claims.
"""
from __future__ import annotations

import pandas as pd


def run_next_bar_backtest(
    bars: pd.DataFrame,
    signals: pd.Series,
    fee_bps_per_side: float = 5.0,
    slippage_bps_per_side: float = 2.0,
) -> pd.DataFrame:
    """Backtest -1/0/+1 signals with next-open execution and open-to-open PnL."""
    if "open" not in bars.columns:
        raise ValueError("Missing required column: 'open'")
    if not bars.index.equals(signals.index):
        raise ValueError("bars and signals must have identical indexes")
    if fee_bps_per_side < 0 or slippage_bps_per_side < 0:
        raise ValueError("fees and slippage must be non-negative")

    out = bars[["open"]].copy()
    out["desired_signal"] = pd.to_numeric(signals, errors="coerce").fillna(0.0).clip(-1, 1)
    out["position"] = out["desired_signal"].shift(1).fillna(0.0)
    out["bar_return"] = out["open"].shift(-1).div(out["open"]).sub(1.0)
    out["gross_return"] = out["position"] * out["bar_return"]

    turnover = out["position"].diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(out["position"].iloc[0])
    cost_bps = fee_bps_per_side + slippage_bps_per_side
    out["turnover"] = turnover.fillna(0.0)
    out["cost_return"] = out["turnover"] * cost_bps / 10_000
    out["net_return"] = out["gross_return"] - out["cost_return"]
    out["equity"] = (1.0 + out["net_return"].fillna(0.0)).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out
