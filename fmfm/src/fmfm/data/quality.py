"""Data quality diagnostics for time-indexed OHLCV bars."""
from __future__ import annotations

import pandas as pd


def inspect_ohlcv(
    bars: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    expected_frequency: str | None = None,
) -> dict[str, object]:
    """Return deterministic integrity diagnostics; never silently repair data."""
    required = {"open", "high", "low", "close", "volume"}
    missing_columns = sorted(required - set(bars.columns))
    report: dict[str, object] = {
        "row_count": len(bars),
        "missing_columns": missing_columns,
        "duplicate_timestamps": 0,
        "invalid_timestamps": 0,
        "non_monotonic_timestamps": False,
        "invalid_ohlc_rows": 0,
        "negative_volume_rows": 0,
        "null_required_values": 0,
        "missing_intervals": 0,
        "first_timestamp": None,
        "last_timestamp": None,
    }
    if missing_columns or bars.empty:
        return report

    frame = bars.copy()
    if timestamp_col in frame.columns:
        times = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        times = pd.Series(pd.to_datetime(frame.index, utc=True), index=frame.index)
    else:
        report["missing_columns"] = sorted([*missing_columns, timestamp_col])
        return report

    report["duplicate_timestamps"] = int(times.duplicated().sum())
    report["invalid_timestamps"] = int(times.isna().sum())
    valid_times = times.dropna()
    report["non_monotonic_timestamps"] = bool(not valid_times.is_monotonic_increasing)
    if not valid_times.empty:
        report["first_timestamp"] = valid_times.min().isoformat()
        report["last_timestamp"] = valid_times.max().isoformat()

    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    report["null_required_values"] = int(numeric.isna().any(axis=1).sum())
    invalid = (
        (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
        | (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    report["invalid_ohlc_rows"] = int(invalid.sum())
    report["negative_volume_rows"] = int((numeric["volume"] < 0).sum())

    if expected_frequency and len(valid_times) > 1:
        ordered = valid_times.sort_values().drop_duplicates()
        expected = pd.Timedelta(expected_frequency)
        deltas = ordered.diff().dropna()
        report["missing_intervals"] = int(((deltas / expected).round().astype("Int64") - 1).clip(lower=0).sum())
    return report
