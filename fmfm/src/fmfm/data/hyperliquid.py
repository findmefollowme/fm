"""Read-only adapter for Hyperliquid public Info API.

This module intentionally does not implement signing or order placement.
"""
from __future__ import annotations

from typing import Any

import httpx
import pandas as pd


_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}
_CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trades"]


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

    @staticmethod
    def _normalize_candles(rows: Any) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=_CANDLE_COLUMNS)
        frame = pd.DataFrame(rows).rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "n": "trades",
            }
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "trades"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return (
            frame.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)[_CANDLE_COLUMNS]
        )

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch one candle snapshot request (venue limit: up to 5000 candles)."""
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
        return self._normalize_candles(rows)

    def candles_range(
        self,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        max_candles_per_request: int = 5000,
    ) -> pd.DataFrame:
        """Fetch a long fixed-interval range in chunks and deduplicate boundaries.

        Calendar-month intervals are intentionally excluded because they do not
        have a fixed millisecond duration; monthly bars can be resampled locally.
        """
        if interval not in _INTERVAL_MS:
            raise ValueError(
                f"Unsupported fixed interval {interval!r}; choose from {tuple(_INTERVAL_MS)}"
            )
        if end_ms < start_ms:
            raise ValueError("end_ms must be >= start_ms")
        if not 1 <= max_candles_per_request <= 5000:
            raise ValueError("max_candles_per_request must be between 1 and 5000")

        step_ms = _INTERVAL_MS[interval]
        cursor = int(start_ms)
        end_ms = int(end_ms)
        frames: list[pd.DataFrame] = []
        while cursor <= end_ms:
            chunk_end = min(
                end_ms,
                cursor + (max_candles_per_request - 1) * step_ms,
            )
            chunk = self.candles(coin, interval, cursor, chunk_end)
            if not chunk.empty:
                frames.append(chunk)
            cursor = chunk_end + step_ms

        if not frames:
            return pd.DataFrame(columns=_CANDLE_COLUMNS)
        result = pd.concat(frames, ignore_index=True)
        return (
            result.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)[_CANDLE_COLUMNS]
        )

    def order_book(self, coin: str) -> dict[str, Any]:
        """Fetch one L2 order-book snapshot."""
        return self._post_info({"type": "l2Book", "coin": coin})

    def all_mids(self) -> dict[str, float]:
        """Fetch current mid prices."""
        payload = self._post_info({"type": "allMids"})
        return {symbol: float(price) for symbol, price in payload.items()}
