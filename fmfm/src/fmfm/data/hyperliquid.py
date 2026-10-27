"""Read-only adapter for Hyperliquid public Info API.

This module intentionally does not implement signing or order placement.
"""
from __future__ import annotations

from typing import Any

import httpx
import pandas as pd


class HyperliquidPublicData:
    """Read-only access to public market data on Hyperliquid."""

    def __init__(self, base_url: str = "https://api.hyperliquid.xyz", timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post_info(self, payload: dict[str, Any]) -> Any:
        response = httpx.post(
            f"{self.base_url}/info",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch candle snapshots; the API returns at most 5000 candles per request."""
        rows = self._post_info(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": int(start_ms),
                    "endTime": int(end_ms),
                },
            }
        )
        columns = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
        if not rows:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame(rows).rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low",
                     "c": "close", "v": "volume", "n": "trades"}
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "trades"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return frame[columns]

    def order_book(self, coin: str) -> dict[str, Any]:
        """Fetch one L2 order-book snapshot."""
        return self._post_info({"type": "l2Book", "coin": coin})

    def all_mids(self) -> dict[str, float]:
        """Fetch current mid prices."""
        payload = self._post_info({"type": "allMids"})
        return {symbol: float(price) for symbol, price in payload.items()}
