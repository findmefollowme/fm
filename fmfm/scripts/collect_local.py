"""One-command local Hyperliquid data collection and verification.

Run after installing FMFM with: pip install -e ".[dev]"
This script performs historical downloads first, then captures live WebSocket events.
It never places orders and never needs wallet credentials.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_cli(*args: str) -> None:
    command = [sys.executable, "-m", "fmfm.cli", *args]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def count_events(path: Path) -> tuple[int, Counter[str], list[dict[str, Any]]]:
    total = 0
    channels: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": str(exc)})
                continue
            total += 1
            channels[str(record.get("channel", "missing"))] += 1
            if record.get("channel") == "collector_error":
                errors.append({
                    "line": line_number,
                    "error": record.get("payload", {}),
                })
    return total, channels, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and verify local FMFM public market data")
    parser.add_argument("--coin", default="BTC", help="Hyperliquid coin, default BTC")
    parser.add_argument("--output-root", type=Path, default=Path("data/collection_runs"))
    parser.add_argument("--candle-days", type=int, default=3,
                        help="Recent 1m candle history; max 3 days by default to stay below 5000 bars")
    parser.add_argument("--funding-days", type=int, default=30)
    parser.add_argument("--live-minutes", type=float, default=60,
                        help="Live WebSocket capture duration; use 5 for a quick smoke test")
    args = parser.parse_args()

    if not 1 <= args.candle_days <= 3:
        parser.error("--candle-days must be 1..3 for 1m candles under the 5000-bar API limit")
    if args.funding_days < 1 or args.live_minutes <= 0:
        parser.error("--funding-days and --live-minutes must be positive")

    coin = args.coin.upper()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    out = args.output_root / f"{coin}_{run_id}"
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    candle_path = out / f"{coin}_1m.csv"
    funding_path = out / f"{coin}_funding.csv"
    events_path = raw_dir / f"{coin}_events.jsonl"
    end = iso_z(now)
    candle_start = iso_z(now - timedelta(days=args.candle_days))
    funding_start = iso_z(now - timedelta(days=args.funding_days))

    run_cli("fetch-candles", "--coin", coin, "--interval", "1m",
            "--start", candle_start, "--end", end, "--output", str(candle_path))
    run_cli("fetch-funding", "--coin", coin, "--start", funding_start,
            "--end", end, "--output", str(funding_path))
    run_cli("collect-live", "--coin", coin, "--output", str(events_path),
            "--duration-seconds", str(int(args.live_minutes * 60)))

    bars = pd.read_csv(candle_path)
    funding = pd.read_csv(funding_path)
    event_total, channel_counts, collector_errors = count_events(events_path)
    manifest_path = candle_path.with_suffix(candle_path.suffix + ".manifest.json")
    candle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required_channels = ["trades", "l2Book", "activeAssetCtx", "candle"]
    report = {
        "run_id_utc": run_id,
        "coin": coin,
        "status": "market_messages_received" if len(bars) > 0 and channel_counts["trades"] > 0 and channel_counts["candle"] > 0 else "incomplete",
        "note": "This report verifies local files and received messages, not gap-free coverage or predictive value.",
        "files": {
            "candles_csv": str(candle_path),
            "candles_manifest": str(manifest_path),
            "funding_csv": str(funding_path),
            "raw_events_jsonl": str(events_path),
        },
        "coverage": {
            "candle_rows": len(bars),
            "candle_first_timestamp": bars["timestamp"].min() if len(bars) else None,
            "candle_last_timestamp": bars["timestamp"].max() if len(bars) else None,
            "funding_rows": len(funding),
            "funding_first_timestamp": funding["timestamp"].min() if len(funding) and "timestamp" in funding else None,
            "funding_last_timestamp": funding["timestamp"].max() if len(funding) and "timestamp" in funding else None,
            "raw_event_lines_parsed": event_total,
            "channel_counts": dict(channel_counts),
            "channels_with_no_messages": [c for c in required_channels if channel_counts[c] == 0],
            "collector_errors_or_malformed_lines": collector_errors,
        },
        "candle_quality": candle_manifest.get("quality", {}),
    }
    report_path = out / "collection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== LOCAL COLLECTION REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")
    return 0 if report["status"] == "market_messages_received" else 2


if __name__ == "__main__":
    raise SystemExit(main())
