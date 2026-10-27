import numpy as np
import pandas as pd

from fmfm.backtest.metrics import summarize_returns


def test_metrics_report_net_return_and_drawdown():
    returns = pd.Series([0.1, -0.2, 0.05])
    result = summarize_returns(returns, periods_per_year=3)
    assert np.isclose(result["total_return"], (1.1 * 0.8 * 1.05) - 1)
    assert result["max_drawdown"] < 0
    assert 0 <= result["win_rate"] <= 1
