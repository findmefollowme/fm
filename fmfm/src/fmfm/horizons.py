"""Canonical prediction horizons for FMFM.

The model can forecast multiple horizons from one canonical time-indexed OHLCV
dataset. Intraday bars are resampled from 1-minute data; longer bars are
resampled with explicit UTC boundaries. This does not imply every venue has
complete 1-minute history.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Horizon:
    key: str
    label: str
    pandas_rule: str
    minutes: int | None


HORIZONS: tuple[Horizon, ...] = (
    Horizon("1m", "1 minute", "1min", 1),
    Horizon("5m", "5 minutes", "5min", 5),
    Horizon("15m", "15 minutes", "15min", 15),
    Horizon("30m", "30 minutes", "30min", 30),
    Horizon("1h", "1 hour", "1h", 60),
    Horizon("2h", "2 hours", "2h", 120),
    Horizon("4h", "4 hours", "4h", 240),
    Horizon("1d", "1 day", "1D", 1440),
    Horizon("1w", "1 week", "1W-MON", 10080),
    Horizon("1mo", "1 month", "1MS", None),
)

HORIZON_BY_KEY = {item.key: item for item in HORIZONS}
OHLCV_AGGREGATION = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "trades": "sum",
}


def resample_ohlcv(
    bars: pd.DataFrame,
    horizon: str,
    *,
    timestamp_col: str = "timestamp",
    drop_incomplete: bool = False,
) -> pd.DataFrame:
    """Resample OHLCV bars to a supported horizon using UTC timestamps.

    Input timestamps represent the opening time of each source bar. Empty
    buckets are omitted rather than forward-filled. A monthly horizon is
    calendar-based, so its elapsed duration varies.
    """
    if horizon not in HORIZON_BY_KEY:
        raise ValueError(f"Unsupported horizon {horizon!r}; choose from {tuple(HORIZON_BY_KEY)}")
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if bars.empty:
        return pd.DataFrame(columns=[timestamp_col, *OHLCV_AGGREGATION])

    frame = bars.copy()
    if timestamp_col in frame.columns:
        frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="raise")
        frame = frame.set_index(timestamp_col)
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"Provide {timestamp_col!r} or a DatetimeIndex")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    agg = {key: value for key, value in OHLCV_AGGREGATION.items() if key in frame.columns}
    result = frame.resample(HORIZON_BY_KEY[horizon].pandas_rule, label="left", closed="left").agg(agg)
    result = result.dropna(subset=["open", "high", "low", "close"])
    if drop_incomplete and not result.empty and horizon != "1m":
        # This completeness check assumes the canonical source is 1-minute bars.
        counts = frame["close"].resample(
            HORIZON_BY_KEY[horizon].pandas_rule, label="left", closed="left"
        ).count()
        expected: dict[pd.Timestamp, int] = {}
        if horizon == "1mo":
            for bucket in result.index:
                next_bucket = bucket + pd.DateOffset(months=1)
                expected[bucket] = int((next_bucket - bucket).total_seconds() // 60)
        else:
            expected_count = HORIZON_BY_KEY[horizon].minutes
            if expected_count is not None:
                expected = {bucket: expected_count for bucket in result.index}
        expected_series = pd.Series(expected, dtype="int64")
        actual = counts.reindex(result.index).fillna(0).astype("int64")
        required = expected_series.reindex(result.index).fillna(0)
        result = result.loc[actual >= required]
    return result.rename_axis(timestamp_col).reset_index()


def supported_horizons() -> list[dict[str, str | int | None]]:
    """Return a serializable list for CLIs, dashboards, and model configs."""
    return [
        {"key": item.key, "label": item.label, "pandas_rule": item.pandas_rule, "minutes": item.minutes}
        for item in HORIZONS
    ]
