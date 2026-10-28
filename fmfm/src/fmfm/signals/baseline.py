"""Interpretable baseline signal; not a production strategy."""
from __future__ import annotations

import pandas as pd


def flow_momentum_signal(
    features: pd.DataFrame,
    flow_z_threshold: float = 1.5,
    min_return: float = 0.0,
) -> pd.DataFrame:
    """Produce a simple research label from rolling signed-flow proxies."""
    required = {"signed_flow_zscore_rolling", "bar_return"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = pd.DataFrame(index=features.index)
    flow, ret = features["signed_flow_zscore_rolling"], features["bar_return"]
    out["signal"] = 0
    out.loc[(flow > flow_z_threshold) & (ret > min_return), "signal"] = 1
    out.loc[(flow < -flow_z_threshold) & (ret < -min_return), "signal"] = -1
    out["side"] = out["signal"].map({1: "long", -1: "short", 0: "flat"})
    return out
