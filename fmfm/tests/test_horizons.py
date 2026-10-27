import pandas as pd
import pytest

from fmfm.horizons import HORIZONS, resample_ohlcv


def test_requested_horizons_are_registered():
    assert [h.key for h in HORIZONS] == [
        "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1mo"
    ]


def test_resample_one_minute_bars_to_five_minutes():
    timestamps = pd.date_range("2026-01-01", periods=10, freq="min", tz="UTC")
    bars = pd.DataFrame({
        "timestamp": timestamps,
        "open": range(100, 110),
        "high": range(101, 111),
        "low": range(99, 109),
        "close": range(100, 110),
        "volume": [1.0] * 10,
        "trades": [2] * 10,
    })
    result = resample_ohlcv(bars, "5m")
    assert len(result) == 2
    assert result.iloc[0]["open"] == 100
    assert result.iloc[0]["close"] == 104
    assert result.iloc[0]["high"] == 105
    assert result.iloc[0]["low"] == 99
    assert result.iloc[0]["volume"] == 5


def test_unknown_horizon_fails_explicitly():
    with pytest.raises(ValueError):
        resample_ohlcv(pd.DataFrame(), "7m")
