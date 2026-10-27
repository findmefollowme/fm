import pandas as pd

from fmfm.backtest.engine import run_next_bar_backtest


def test_signal_is_shifted_to_next_bar():
    bars = pd.DataFrame(
        {"open": [100.0, 100.0, 110.0], "close": [100.0, 110.0, 121.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
    )
    signals = pd.Series([1, 1, 0], index=bars.index)
    result = run_next_bar_backtest(bars, signals, fee_bps_per_side=0, slippage_bps_per_side=0)
    assert result["position"].tolist() == [0.0, 1.0, 1.0]
    assert result.loc[bars.index[1], "gross_return"] == 0.1
