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


def test_empty_resample_preserves_timestamp_schema():
    bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    result = resample_ohlcv(bars, "5m")
    assert "timestamp" in result.columns
    assert result.empty


def test_daily_source_can_build_complete_weekly_and_monthly_bars():
    daily_times = pd.date_range("2026-02-01", periods=28, freq="D", tz="UTC")
    bars = pd.DataFrame({
        "timestamp": daily_times,
        "open": range(100, 128),
        "high": range(101, 129),
        "low": range(99, 127),
        "close": range(100, 128),
        "volume": [1.0] * 28,
    })
    weekly = resample_ohlcv(bars, "1w", source_interval="1d", drop_incomplete=True)
    monthly = resample_ohlcv(bars, "1mo", source_interval="1d", drop_incomplete=True)
    assert len(weekly) == 3
    assert len(monthly) == 1
