"""Multi-horizon training targets for FMFM."""
from __future__ import annotations

import pandas as pd

from fmfm.horizons import HORIZON_BY_KEY, resample_ohlcv
from fmfm.targets import forward_log_return, forward_realized_volatility


def build_multi_horizon_targets(
    one_minute_bars: pd.DataFrame,
    horizons: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1mo"),
) -> dict[str, pd.DataFrame]:
    """Build one target frame per horizon from canonical 1-minute OHLCV bars.

    For each horizon, output forward close-to-close log return and realized
    volatility over the next horizon bar. This first version is a transparent
    label baseline, not a claim of predictive accuracy. Monthly horizons are
    calendar-based and therefore are handled by resampling before labeling.
    """
    outputs: dict[str, pd.DataFrame] = {}
    for key in horizons:
        if key not in HORIZON_BY_KEY:
            raise ValueError(f"Unsupported horizon: {key}")
        frame = resample_ohlcv(one_minute_bars, key)
        close = pd.Series(frame["close"].to_numpy(), index=frame.index, dtype=float)
        steps = 1
        labels = pd.DataFrame(index=frame.index)
        labels[f"forward_log_return_{key}"] = forward_log_return(close, steps)
        labels[f"forward_realized_volatility_{key}"] = forward_realized_volatility(close, steps)
        outputs[key] = labels.reset_index(drop=True).assign(timestamp=frame["timestamp"].to_numpy())
    return outputs
