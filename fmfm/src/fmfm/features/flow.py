"""First-pass features derived from bars, trades and L2 snapshots.

These are proxies for observable trading pressure, not direct measurements of all
capital intentions. Missing inputs are not fabricated.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def candle_flow_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build OHLCV proxies. These cannot identify actual aggressor side."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = pd.DataFrame(index=bars.index)
    price_range = (bars["high"] - bars["low"]).replace(0, np.nan)
    out["bar_return"] = bars["close"].pct_change()
    out["close_location"] = ((bars["close"] - bars["low"]) / price_range).clip(0, 1)
    out["signed_volume_proxy"] = (2 * out["close_location"] - 1) * bars["volume"]
    out["volume_change"] = bars["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    out["turnover_proxy"] = bars["close"] * bars["volume"]
    out["range_pct"] = (bars["high"] - bars["low"]) / bars["close"].replace(0, np.nan)
    out["volume_zscore_rolling"] = _rolling_zscore(bars["volume"], window=48)
    out["signed_flow_zscore_rolling"] = _rolling_zscore(
        out["signed_volume_proxy"], window=48
    )
    return out


def trade_flow_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade-level signed notional into one-minute bars.

    Expected columns: timestamp, side, size, price. Verify the venue's side
    convention before production use; unknown side values remain NaN.
    """
    required = {"timestamp", "side", "size", "price"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = trades.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["size"] = pd.to_numeric(data["size"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    side = data["side"].astype(str).str.upper()
    signed = np.where(side.isin(["B", "BUY", "BUYER"]), 1.0,
                      np.where(side.isin(["A", "S", "SELL", "SELLER"]), -1.0, np.nan))
    data["signed_notional"] = signed * data["size"] * data["price"]
    data["notional"] = data["size"] * data["price"]
    data = data.set_index("timestamp").sort_index()
    return data[["signed_notional", "notional"]].resample(
        "1min", label="right", closed="left"
    ).sum(min_count=1).rename(
        columns={"signed_notional": "net_aggressive_notional", "notional": "total_notional"}
    )


def order_book_features(snapshot: dict[str, Any]) -> dict[str, float]:
    """Summarize an L2 snapshot into spread and depth imbalance features."""
    levels = snapshot.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("Unexpected L2 snapshot: expected bid and ask levels")

    bids, asks = levels[0], levels[1]
    if not bids or not asks:
        return {"mid_price": np.nan, "spread_bps": np.nan, "depth_imbalance": np.nan}

    bid_px, ask_px = float(bids[0]["px"]), float(asks[0]["px"])
    mid = (bid_px + ask_px) / 2
    bid_depth = sum(float(level["sz"]) for level in bids)
    ask_depth = sum(float(level["sz"]) for level in asks)
    depth_total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / depth_total if depth_total else np.nan
    return {
        "mid_price": mid,
        "spread_bps": ((ask_px - bid_px) / mid * 10_000) if mid else np.nan,
        "depth_imbalance": imbalance,
    }


def _rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    min_periods = max(8, window // 4)
    mean = values.rolling(window, min_periods=min_periods).mean()
    std = values.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((values - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
