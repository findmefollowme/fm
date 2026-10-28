"""Multi-horizon training targets for FMFM."""
from __future__ import annotations

import pandas as pd

from fmfm.horizons import HORIZON_BY_KEY, resample_ohlcv
from fmfm.targets import forward_log_return, forward_range_volatility


def build_multi_horizon_targets(
    bars: pd.DataFrame,
    horizons: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1mo"),
    *,
    source_interval: str = "1m",
) -> dict[str, pd.DataFrame]:
    """Build future-outcome labels from source OHLCV bars.

    Choose source_interval to match the input CSV. For example, use 1m data for
    intraday horizons and 1d data for daily, weekly and monthly targets. Labels
    are valid only when the next bar starts exactly one target interval later.
    """
    outputs: dict[str, pd.DataFrame] = {}
    for key in horizons:
        if key not in HORIZON_BY_KEY:
            raise ValueError(f"Unsupported horizon: {key}")
        frame = resample_ohlcv(
            bars, key, source_interval=source_interval, drop_incomplete=True
        )
        times = pd.Series(pd.to_datetime(frame["timestamp"], utc=True), index=frame.index)
        close = pd.Series(frame["close"].to_numpy(), index=frame.index, dtype=float)
        high = pd.Series(frame["high"].to_numpy(), index=frame.index, dtype=float)
        low = pd.Series(frame["low"].to_numpy(), index=frame.index, dtype=float)
        returns = forward_log_return(close, 1)
        range_volatility = forward_range_volatility(high, low, 1)

        if key == "1mo":
            expected_next = times + pd.DateOffset(months=1)
        else:
            minutes = HORIZON_BY_KEY[key].minutes
            expected_next = times + pd.to_timedelta(minutes, unit="min")
        actual_next = times.shift(-1)
        contiguous = actual_next.eq(expected_next)
        returns = returns.where(contiguous)
        range_volatility = range_volatility.where(contiguous)

        labels = pd.DataFrame(index=frame.index)
        labels[f"forward_log_return_{key}"] = returns
        labels[f"forward_range_volatility_{key}"] = range_volatility
        if key == "1mo":
            forecast_time = pd.to_datetime(frame["timestamp"], utc=True) + pd.DateOffset(months=1)
        else:
            forecast_time = pd.to_datetime(frame["timestamp"], utc=True) + pd.to_timedelta(
                HORIZON_BY_KEY[key].minutes, unit="min"
            )
        outputs[key] = labels.reset_index(drop=True).assign(timestamp=forecast_time.to_numpy())
    return outputs
