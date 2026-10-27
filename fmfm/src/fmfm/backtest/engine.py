"""Small vectorized baseline backtest.

Signals are assumed to be computed at bar close and executed at the next bar open.
This is a research harness, not an exchange simulator; funding and liquidation
modelling must be added for perpetual futures before any performance claims.
"""
from __future__ import annotations

import pandas as pd


def run_next_bar_backtest(
    bars: pd.DataFrame,
    signals: pd.Series,
    fee_bps_per_side: float = 5.0,
    slippage_bps_per_side: float = 2.0,
) -> pd.DataFrame:
    """Backtest a -1/0/+1 position signal with next-open entry and close-to-close PnL.

    At time t, signal[t] is generated after bar t closes. It is shifted and held
    over the next bar to avoid using bar-t close information for a bar-t fill.
    Transaction costs are charged when position changes. The simplified cost
    model is intentionally explicit and should be replaced with venue-specific
    fees, funding and depth-aware slippage before strategy evaluation.
    """
    required = {"open", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not bars.index.equals(signals.index):
        raise ValueError("bars and signals must have identical indexes")

    out = bars[["open", "close"]].copy()
    out["desired_signal"] = signals.astype(float).clip(-1, 1)
    out["position"] = out["desired_signal"].shift(1).fillna(0.0)
    out["bar_return"] = out["close"].pct_change().fillna(0.0)
    out["gross_return"] = out["position"] * out["bar_return"]

    turnover = out["position"].diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(out["position"].iloc[0])
    cost_bps = fee_bps_per_side + slippage_bps_per_side
    out["turnover"] = turnover.fillna(0.0)
    out["cost_return"] = out["turnover"] * cost_bps / 10_000
    out["net_return"] = out["gross_return"] - out["cost_return"]
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out
