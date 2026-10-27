import pandas as pd

from fmfm.data.quality import inspect_ohlcv


def test_quality_detects_bad_rows_duplicates_and_gaps():
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:03:00Z",
        ]),
        "open": [100, 100, 105],
        "high": [101, 101, 104],
        "low": [99, 99, 106],
        "close": [100, 100, 105],
        "volume": [1, 1, -1],
    })
    report = inspect_ohlcv(bars, expected_frequency="1min")
    assert report["duplicate_timestamps"] == 1
    assert report["invalid_ohlc_rows"] == 1
    assert report["negative_volume_rows"] == 1
    assert report["missing_intervals"] >= 1
