"""Parse the raw Hyperliquid WebSocket JSONL stream into normalized events."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fmfm.features.flow import order_book_features


def read_jsonl_events(path: str | Path) -> list[dict[str, Any]]:
    """Read append-only JSONL events and report malformed line numbers."""
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at line {line_number}: {exc}") from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def trades_from_events(
    events: list[dict[str, Any]],
    *,
    coin: str | None = None,
) -> pd.DataFrame:
    """Normalize public trades into timestamp/side/size/price rows."""
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("channel") != "trades":
            continue
        payload = event.get("payload")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            continue
        for trade in data:
            if not isinstance(trade, dict):
                continue
            trade_coin = str(trade.get("coin", event.get("coin", "")))
            if coin and trade_coin.upper() != coin.upper():
                continue
            if not {"time", "side", "sz", "px"}.issubset(trade):
                continue
            rows.append({
                "timestamp": pd.to_datetime(trade["time"], unit="ms", utc=True),
                "coin": trade_coin,
                "side": str(trade["side"]).upper(),
                "size": pd.to_numeric(trade["sz"], errors="coerce"),
                "price": pd.to_numeric(trade["px"], errors="coerce"),
                "trade_id": trade.get("tid"),
                "hash": trade.get("hash"),
            })
    columns = ["timestamp", "coin", "side", "size", "price", "trade_id", "hash"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("timestamp").reset_index(drop=True)


def book_features_from_events(
    events: list[dict[str, Any]],
    *,
    coin: str | None = None,
) -> pd.DataFrame:
    """Normalize public L2 snapshots to timestamped spread/depth features."""
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("channel") != "l2Book":
            continue
        payload = event.get("payload")
        snapshot = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(snapshot, dict) or "time" not in snapshot:
            continue
        snapshot_coin = str(snapshot.get("coin", event.get("coin", "")))
        if coin and snapshot_coin.upper() != coin.upper():
            continue
        features = order_book_features(snapshot)
        rows.append({
            "timestamp": pd.to_datetime(snapshot["time"], unit="ms", utc=True),
            "coin": snapshot_coin,
            **features,
        })
    columns = ["timestamp", "coin", "mid_price", "spread_bps", "depth_imbalance"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def asset_contexts_from_events(
    events: list[dict[str, Any]],
    *,
    coin: str | None = None,
) -> pd.DataFrame:
    """Normalize activeAssetCtx snapshots, timestamped by local ingestion time.

    The active-asset context feed is a snapshot stream. Ingestion time is used
    because the payload may not include a separate event timestamp.
    """
    context_fields = [
        "funding", "openInterest", "markPx", "midPx", "oraclePx",
        "premium", "dayNtlVlm", "prevDayPx",
    ]
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("channel") != "activeAssetCtx":
            continue
        payload = event.get("payload")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict) or not isinstance(data.get("ctx"), dict):
            continue
        context_coin = str(data.get("coin", event.get("coin", "")))
        if coin and context_coin.upper() != coin.upper():
            continue
        observed_at = event.get("ingest_time")
        if not observed_at:
            continue
        context = data["ctx"]
        row: dict[str, Any] = {
            "timestamp": pd.to_datetime(observed_at, utc=True),
            "coin": context_coin,
        }
        for field in context_fields:
            if field in context:
                row[field] = pd.to_numeric(context[field], errors="coerce")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "coin", *context_fields])
    return (
        pd.DataFrame(rows)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
