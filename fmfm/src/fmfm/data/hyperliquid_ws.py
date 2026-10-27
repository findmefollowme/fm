"""Append-only collector for Hyperliquid public WebSocket market streams."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

from fmfm.data.storage import append_raw_events

DEFAULT_WS_URL = "wss://api.hyperliquid.xyz/ws"
SUPPORTED_CHANNELS = {"trades", "l2Book", "activeAssetCtx", "candle"}


def build_subscriptions(coin: str, channels: tuple[str, ...]) -> list[dict[str, Any]]:
    """Build validated public subscriptions for one coin."""
    unknown = set(channels) - SUPPORTED_CHANNELS
    if unknown:
        raise ValueError(f"Unsupported channels: {sorted(unknown)}")
    subscriptions: list[dict[str, Any]] = []
    for channel in channels:
        if channel == "candle":
            subscriptions.append({"type": "candle", "coin": coin, "interval": "1m"})
        else:
            subscriptions.append({"type": channel, "coin": coin})
    return subscriptions


def _event_record(coin: str, channel: str, payload: Any, connection_epoch: int) -> dict[str, Any]:
    return {
        "source": "hyperliquid_websocket",
        "coin": coin,
        "channel": channel,
        "ingest_time": datetime.now(timezone.utc).isoformat(),
        "connection_epoch": connection_epoch,
        "payload": payload,
    }


async def collect_public_stream(
    coin: str,
    output: str | Path,
    *,
    channels: tuple[str, ...] = ("trades", "l2Book", "activeAssetCtx", "candle"),
    duration_seconds: float = 0,
    reconnect_seconds: float = 5,
    heartbeat_seconds: float = 25,
    websocket_url: str = DEFAULT_WS_URL,
) -> None:
    """Collect public messages until duration elapses or the process is stopped.

    duration_seconds=0 means run until interrupted. Every received message is
    appended as a raw JSONL event. Reconnects are recorded, but this collector
    does not guarantee gap-free trade history or backfill missed trades.
    """
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be >= 0")
    if reconnect_seconds <= 0 or heartbeat_seconds <= 0:
        raise ValueError("reconnect_seconds and heartbeat_seconds must be > 0")
    subscriptions = build_subscriptions(coin, channels)
    started = time.monotonic()
    epoch = 0
    output_path = Path(output)

    while duration_seconds == 0 or time.monotonic() - started < duration_seconds:
        try:
            async with websockets.connect(
                websocket_url,
                open_timeout=15,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
                max_size=8 * 1024 * 1024,
            ) as socket:
                epoch += 1
                append_raw_events(
                    output_path,
                    [_event_record(coin, "collector_connected", {"url": websocket_url}, epoch)],
                )
                for subscription in subscriptions:
                    await socket.send(json.dumps({"method": "subscribe", "subscription": subscription}))

                while duration_seconds == 0 or time.monotonic() - started < duration_seconds:
                    timeout = heartbeat_seconds
                    if duration_seconds > 0:
                        remaining = duration_seconds - (time.monotonic() - started)
                        if remaining <= 0:
                            return
                        timeout = min(timeout, remaining)
                    try:
                        raw_message = await asyncio.wait_for(socket.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        if duration_seconds > 0 and time.monotonic() - started >= duration_seconds:
                            return
                        await socket.send(json.dumps({"method": "ping"}))
                        append_raw_events(
                            output_path,
                            [_event_record(coin, "collector_heartbeat", {"sent": True}, epoch)],
                        )
                        continue

                    try:
                        message = json.loads(raw_message)
                    except (json.JSONDecodeError, TypeError):
                        message = {"unparsed_message": str(raw_message)}
                    channel = str(message.get("channel", "unknown")) if isinstance(message, dict) else "unknown"
                    append_raw_events(
                        output_path,
                        [_event_record(coin, channel, message, epoch)],
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            append_raw_events(
                output_path,
                [
                    _event_record(
                        coin,
                        "collector_error",
                        {"error_type": type(exc).__name__, "error": str(exc)},
                        epoch,
                    )
                ],
            )
            if duration_seconds > 0 and time.monotonic() - started >= duration_seconds:
                return
            await asyncio.sleep(reconnect_seconds)
