"""Multi-horizon training targets for FMFM."""
from __future__ import annotations

import pandas as pd

from fmfm.horizons import HORIZON_BY_KEY, resample_ohlcv
from fmfm.targets import forward_log_return, forward_range_volatility


def build_multi_horizon_targets(
    one_minute_bars: pd.DataFrame,
    horizons: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1mo"),
) -> dict[str, pd.DataFrame]:
    """Build one target frame per horizon from canonical 1-minute OHLCV bars.

    Each timeframe gets its own next-bar close-to-close log return and a
    next-bar range-volatility label estimated from that future candle's high
    and low. These are labels for training/evaluation only, not features.
    """
    outputs: dict[str, pd.DataFrame] = {}
    for key in horizons:
        if key not in HORIZON_BY_KEY:
            raise ValueError(f"Unsupported horizon: {key}")
        frame = resample_ohlcv(one_minute_bars, key, drop_incomplete=True)
        close = pd.Series(frame["close"].to_numpy(), index=frame.index, dtype=float)
        high = pd.Series(frame["high"].to_numpy(), index=frame.index, dtype=float)
        low = pd.Series(frame["low"].to_numpy(), index=frame.index, dtype=float)
        labels = pd.DataFrame(index=frame.index)
        labels[f"forward_log_return_{key}"] = forward_log_return(close, 1)
        labels[f"forward_range_volatility_{key}"] = forward_range_volatility(high, low, 1)
        outputs[key] = labels.reset_index(drop=True).assign(timestamp=frame["timestamp"].to_numpy())
    return outputs
