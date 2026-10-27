import pandas as pd

from fmfm.backtest.engine import run_next_bar_backtest


def test_signal_is_delayed_and_open_to_open_return_is_used():
    bars = pd.DataFrame({"open": [100.0, 110.0, 121.0, 110.0]})
    signals = pd.Series([1.0, 0.0, -1.0, 0.0])
    result = run_next_bar_backtest(bars, signals, fee_bps_per_side=0, slippage_bps_per_side=0)
    assert result["position"].tolist() == [0.0, 1.0, 0.0, -1.0]
    assert result.iloc[1]["gross_return"] == 0.1
    assert result.iloc[2]["gross_return"] == 0.0
