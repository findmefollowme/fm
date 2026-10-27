import numpy as np
import pandas as pd

from fmfm.multi_horizon import build_multi_horizon_targets


def test_multi_horizon_targets_include_requested_periods():
    timestamps = pd.date_range("2026-01-01", periods=10, freq="min", tz="UTC")
    close = np.arange(100.0, 110.0)
    bars = pd.DataFrame({
        "timestamp": timestamps,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close + 0.5,
        "volume": np.ones(10),
        "trades": np.ones(10),
    })
    result = build_multi_horizon_targets(bars, horizons=("1m", "5m"))
    assert len(result["1m"]) == 10
    assert len(result["5m"]) == 2
    assert "forward_log_return_5m" in result["5m"]
    assert "forward_range_volatility_5m" in result["5m"]
    assert np.isnan(result["5m"]["forward_log_return_5m"].iloc[-1])
