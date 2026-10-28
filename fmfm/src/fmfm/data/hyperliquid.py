"""Read-only adapter for Hyperliquid public Info API.

The candle endpoint exposes only the latest 5000 bars. Chunking can split a
request within that accessible window; it cannot recover older candles.
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
_FUNDING_COLUMNS = ["timestamp", "coin", "funding_rate", "premium"]


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

    @staticmethod
    def _normalize_funding(rows: Any) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=_FUNDING_COLUMNS)
        frame = pd.DataFrame(rows).rename(
            columns={"time": "timestamp", "fundingRate": "funding_rate"}
        )
        if "premium" not in frame:
            frame["premium"] = pd.NA
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        for col in ["funding_rate", "premium"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return (
            frame.sort_values("timestamp")
            .drop_duplicates(["timestamp", "coin"], keep="last")
            .reset_index(drop=True)[_FUNDING_COLUMNS]
        )

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch one candle snapshot request (venue history limit: latest 5000 bars)."""
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
        """Fetch a range within the latest-5000-bar retention window.

        Requests longer than 5000 bars are rejected. Splitting a request into
        chunks does not bypass the venue's historical retention limit.
        """
        if interval == "1M":
            if end_ms < start_ms:
                raise ValueError("end_ms must be >= start_ms")
            if not 1 <= max_candles_per_request <= 5000:
                raise ValueError("max_candles_per_request must be between 1 and 5000")
            start = pd.to_datetime(start_ms, unit="ms", utc=True)
            end = pd.to_datetime(end_ms, unit="ms", utc=True)
            month_count = (end.year - start.year) * 12 + end.month - start.month + 1
            if month_count > 5000 or month_count > max_candles_per_request:
                raise ValueError("Monthly candles use one calendar-based request; keep the range within the request limit")
            return self.candles(coin, interval, start_ms, end_ms)

        if interval not in _INTERVAL_MS:
            raise ValueError(
                f"Unsupported interval {interval!r}; choose from {tuple(_INTERVAL_MS)} or '1M'"
            )
        if end_ms < start_ms:
            raise ValueError("end_ms must be >= start_ms")
        if not 1 <= max_candles_per_request <= 5000:
            raise ValueError("max_candles_per_request must be between 1 and 5000")

        step_ms = _INTERVAL_MS[interval]
        requested_count = (int(end_ms) - int(start_ms)) // step_ms + 1
        if requested_count > 5000:
            raise ValueError(
                f"Requested range spans about {requested_count} {interval} bars; "
                "Hyperliquid's candle API only exposes the latest 5000 bars. "
                "Use an archive or collect data prospectively for longer history."
            )

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

    def funding_history(
        self,
        coin: str,
        start_ms: int,
        end_ms: int,
        *,
        page_size: int = 500,
    ) -> pd.DataFrame:
        """Fetch historical funding in timestamp pages (API returns up to 500 rows/page)."""
        if end_ms < start_ms:
            raise ValueError("end_ms must be >= start_ms")
        if not 1 <= page_size <= 500:
            raise ValueError("page_size must be between 1 and 500")

        cursor = int(start_ms)
        end_ms = int(end_ms)
        pages: list[pd.DataFrame] = []
        while cursor <= end_ms:
            rows = self._post_info(
                {
                    "type": "fundingHistory",
                    "coin": coin,
                    "startTime": cursor,
                    "endTime": end_ms,
                }
            )
            if not rows:
                break
            page = self._normalize_funding(rows)
            if page.empty:
                break
            pages.append(page)
            if len(rows) < page_size:
                break
            last_ms = int(pd.to_datetime(page["timestamp"], utc=True).max().timestamp() * 1000)
            next_cursor = last_ms + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        if not pages:
            return pd.DataFrame(columns=_FUNDING_COLUMNS)
        result = pd.concat(pages, ignore_index=True)
        return (
            result.sort_values("timestamp")
            .drop_duplicates(["timestamp", "coin"], keep="last")
            .reset_index(drop=True)[_FUNDING_COLUMNS]
        )

    @staticmethod
    def drop_unclosed_candles(
        bars: pd.DataFrame,
        interval: str,
        *,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Remove the currently forming candle using an explicit UTC as-of time."""
        if bars.empty:
            return bars.copy()
        if interval == "1M":
            close_times = pd.to_datetime(bars["timestamp"], utc=True) + pd.DateOffset(months=1)
        elif interval in _INTERVAL_MS:
            close_times = pd.to_datetime(bars["timestamp"], utc=True) + pd.to_timedelta(
                _INTERVAL_MS[interval], unit="ms"
            )
        else:
            raise ValueError(f"Unsupported interval {interval!r}")
        cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            raise ValueError("as_of must include a timezone")
        cutoff = cutoff.tz_convert("UTC")
        return bars.loc[close_times <= cutoff].reset_index(drop=True)

    def order_book(self, coin: str) -> dict[str, Any]:
        """Fetch one L2 order-book snapshot."""
        return self._post_info({"type": "l2Book", "coin": coin})

    def all_mids(self) -> dict[str, float]:
        """Fetch current mid prices."""
        payload = self._post_info({"type": "allMids"})
        return {symbol: float(price) for symbol, price in payload.items()}

    def meta_and_asset_contexts(self) -> Any:
        """Fetch current perp metadata and current asset contexts.

        Contexts include current funding and open interest, but are snapshots,
        not historical time series. Persist repeated snapshots to build history.
        """
        return self._post_info({"type": "metaAndAssetCtxs"})
