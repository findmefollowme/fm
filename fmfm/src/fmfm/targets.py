"""Leakage-safe target builders for FMFM research.

Targets are future outcomes for training/evaluation only. Never join these
columns back into same-time model features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Log return from current close to close horizon bars ahead."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    values = pd.to_numeric(close, errors="coerce").astype(float)
    future = values.shift(-horizon)
    ratio = future / values
    result = np.log(ratio.where(ratio > 0))
    return result.replace([np.inf, -np.inf], np.nan)


def forward_realized_volatility(close: pd.Series, horizon: int) -> pd.Series:
    """Root-sum-square of the next horizon one-bar log returns."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    values = pd.to_numeric(close, errors="coerce").astype(float)
    ratio = values / values.shift(1)
    one_bar = np.log(ratio.where(ratio > 0)).replace([np.inf, -np.inf], np.nan)
    future = pd.concat([one_bar.shift(-step) for step in range(1, horizon + 1)], axis=1)
    return np.sqrt(future.pow(2).sum(axis=1, min_count=horizon))
