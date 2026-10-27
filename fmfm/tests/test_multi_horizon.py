import numpy as np
import pandas as pd

from fmfm.multi_horizon import build_multi_horizon_targets


def _bars(timestamps):
    close = np.arange(100.0, 100.0 + len(timestamps))
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close + 0.5,
        "volume": np.ones(len(timestamps)),
        "trades": np.ones(len(timestamps)),
    })


def test_multi_horizon_targets_include_requested_periods():
    timestamps = pd.date_range("2026-01-01", periods=10, freq="min", tz="UTC")
    result = build_multi_horizon_targets(_bars(timestamps), horizons=("1m", "5m"))
    assert len(result["1m"]) == 10
    assert len(result["5m"]) == 2
    assert "forward_log_return_5m" in result["5m"]
    assert "forward_range_volatility_5m" in result["5m"]
    assert np.isnan(result["5m"]["forward_log_return_5m"].iloc[-1])


def test_missing_minute_does_not_create_false_one_minute_target():
    timestamps = pd.to_datetime([
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:01:00Z",
        "2026-01-01T00:03:00Z",
        "2026-01-01T00:04:00Z",
    ])
    result = build_multi_horizon_targets(_bars(timestamps), horizons=("1m",))["1m"]
    assert np.isnan(result["forward_log_return_1m"].iloc[1])
    assert np.isfinite(result["forward_log_return_1m"].iloc[2])
