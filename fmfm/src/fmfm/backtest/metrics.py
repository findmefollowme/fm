"""Performance metrics for fee-aware research backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_returns(
    net_returns: pd.Series,
    *,
    periods_per_year: float = 365.0 * 24.0 * 60.0,
    turnover: pd.Series | None = None,
) -> dict[str, float]:
    """Summarize net returns; caller must set annualization to bar frequency."""
    returns = pd.to_numeric(net_returns, errors="coerce").dropna().astype(float)
    if returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "mean_turnover": 0.0,
        }
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(returns) / periods_per_year if periods_per_year > 0 else 0.0
    annualized_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(mean / std * np.sqrt(periods_per_year)) if std > 0 and periods_per_year > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    mean_turnover = float(turnover.reindex(returns.index).fillna(0.0).mean()) if turnover is not None else 0.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
        "mean_turnover": mean_turnover,
    }
