"""Command-line tools for reproducible FMFM data preparation."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import httpx
import pandas as pd

from fmfm.data.hyperliquid import HyperliquidPublicData
from fmfm.data.quality import inspect_ohlcv
from fmfm.data.storage import read_bars_csv, write_bars_csv
from fmfm.features.flow import candle_flow_features
from fmfm.horizons import HORIZONS
from fmfm.models.walk_forward import regression_metrics, walk_forward_ridge
from fmfm.multi_horizon import build_multi_horizon_targets


def _utc_ms(value: str) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise argparse.ArgumentTypeError("Timestamp must include a timezone, e.g. 2026-01-01T00:00:00Z")
    return int(stamp.tz_convert("UTC").timestamp() * 1000)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def command_fetch(args: argparse.Namespace) -> int:
    start_ms, end_ms = _utc_ms(args.start), _utc_ms(args.end)
    if end_ms < start_ms:
        raise ValueError("--end must be later than or equal to --start")
    client = HyperliquidPublicData(timeout=args.timeout)
    bars = client.candles_range(
        args.coin.upper(),
        args.interval,
        start_ms,
        end_ms,
        max_candles_per_request=args.chunk_size,
    )
    output = write_bars_csv(bars, args.output)
    frequency = "1min" if args.interval == "1m" else None
    report = inspect_ohlcv(bars, expected_frequency=frequency)
    manifest = {
        "source": "Hyperliquid public Info API",
        "coin": args.coin.upper(),
        "interval": args.interval,
        "requested_start": args.start,
        "requested_end": args.end,
        "rows": len(bars),
        "output": str(output),
        "quality": report,
        "note": "A successful API response does not guarantee complete history; inspect gaps and coverage.",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0 if not bars.empty else 2


def command_fetch_funding(args: argparse.Namespace) -> int:
    start_ms, end_ms = _utc_ms(args.start), _utc_ms(args.end)
    client = HyperliquidPublicData(timeout=args.timeout)
    funding = client.funding_history(
        args.coin.upper(), start_ms, end_ms, page_size=args.page_size
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    funding.to_csv(output, index=False)
    manifest = {
        "source": "Hyperliquid public Info API",
        "endpoint_type": "fundingHistory",
        "coin": args.coin.upper(),
        "requested_start": args.start,
        "requested_end": args.end,
        "rows": len(funding),
        "output": str(output),
        "note": "Funding history is a realized historical series; current open interest is a separate snapshot.",
    }
    _write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not funding.empty else 2


def command_inspect(args: argparse.Namespace) -> int:
    bars = read_bars_csv(args.input)
    report = inspect_ohlcv(bars, expected_frequency=args.expected_frequency)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report["missing_columns"] or report["null_required_values"] or report["invalid_ohlc_rows"] or report["negative_volume_rows"]:
        return 2
    return 0


def command_targets(args: argparse.Namespace) -> int:
    bars = read_bars_csv(args.input)
    requested = tuple(args.horizons or [h.key for h in HORIZONS])
    target_frames = build_multi_horizon_targets(bars, requested)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"input": str(args.input), "outputs": {}}
    for horizon, labels in target_frames.items():
        path = out_dir / f"{args.coin.upper()}_{horizon}_targets.csv"
        labels.to_csv(path, index=False)
        summary["outputs"][horizon] = {
            "path": str(path),
            "rows": len(labels),
            "valid_forward_returns": int(labels[f"forward_log_return_{horizon}"].notna().sum()),
        }
    _write_json(out_dir / f"{args.coin.upper()}_targets_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_build_features(args: argparse.Namespace) -> int:
    bars = read_bars_csv(args.input)
    features = candle_flow_features(bars)
    features.insert(0, "timestamp", bars["timestamp"].to_numpy())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(json.dumps({"rows": len(features), "feature_columns": list(features.columns[1:]), "output": str(output)}, indent=2))
    return 0


def command_evaluate_ridge(args: argparse.Namespace) -> int:
    feature_frame = read_bars_csv(args.features).set_index("timestamp")
    target_frame = read_bars_csv(args.targets).set_index("timestamp")
    if args.target_column not in target_frame.columns:
        raise ValueError(f"Target column not found: {args.target_column}")
    joined = feature_frame.join(target_frame[[args.target_column]], how="inner")
    x = joined.drop(columns=[args.target_column])
    y = joined[args.target_column]
    predictions = walk_forward_ridge(
        x,
        y,
        min_train_size=args.min_train_size,
        test_size=args.test_size,
        purge=args.purge,
        alpha=args.alpha,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index_label="timestamp")
    metrics = regression_metrics(predictions)
    report = {
        "features_file": str(args.features),
        "targets_file": str(args.targets),
        "target_column": args.target_column,
        "parameters": {
            "min_train_size": args.min_train_size,
            "test_size": args.test_size,
            "purge": args.purge,
            "alpha": args.alpha,
        },
        "metrics": metrics,
        "predictions_file": str(output),
        "warning": "Out-of-sample forecast metrics are not proof of profitability; include costs in a separate trading simulation.",
    }
    _write_json(output.with_suffix(output.suffix + ".metrics.json"), report)
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, default=str))
    return 0 if metrics["n"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fmfm", description="FMFM read-only quant research CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch-candles", help="Download public Hyperliquid candles")
    fetch.add_argument("--coin", required=True, help="Hyperliquid coin symbol, e.g. BTC")
    fetch.add_argument("--interval", default="1m", help="Venue interval; default 1m")
    fetch.add_argument("--start", required=True, help="Timezone-aware ISO timestamp")
    fetch.add_argument("--end", required=True, help="Timezone-aware ISO timestamp")
    fetch.add_argument("--output", type=Path, required=True, help="Output CSV path")
    fetch.add_argument("--chunk-size", type=int, default=5000, help="Candles per request (1..5000)")
    fetch.add_argument("--timeout", type=float, default=15.0)
    fetch.set_defaults(func=command_fetch)

    funding = commands.add_parser("fetch-funding", help="Download public historical funding rates")
    funding.add_argument("--coin", required=True)
    funding.add_argument("--start", required=True, help="Timezone-aware ISO timestamp")
    funding.add_argument("--end", required=True, help="Timezone-aware ISO timestamp")
    funding.add_argument("--output", type=Path, required=True)
    funding.add_argument("--page-size", type=int, default=500, help="API page size (1..500)")
    funding.add_argument("--timeout", type=float, default=15.0)
    funding.set_defaults(func=command_fetch_funding)

    inspect = commands.add_parser("inspect", help="Inspect a normalized OHLCV CSV")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--expected-frequency", default="1min")
    inspect.set_defaults(func=command_inspect)

    features = commands.add_parser("build-features", help="Build OHLCV-derived flow proxy features")
    features.add_argument("--input", type=Path, required=True)
    features.add_argument("--output", type=Path, required=True)
    features.set_defaults(func=command_build_features)

    evaluate = commands.add_parser("evaluate-ridge", help="Run expanding-window ridge forecast evaluation")
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--targets", type=Path, required=True)
    evaluate.add_argument("--target-column", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--min-train-size", type=int, default=500)
    evaluate.add_argument("--test-size", type=int, default=100)
    evaluate.add_argument("--purge", type=int, default=1)
    evaluate.add_argument("--alpha", type=float, default=1.0)
    evaluate.set_defaults(func=command_evaluate_ridge)

    targets = commands.add_parser("build-targets", help="Build future labels for multiple horizons")
    targets.add_argument("--coin", required=True)
    targets.add_argument("--input", type=Path, required=True, help="Canonical fine-grained OHLCV CSV")
    targets.add_argument("--output-dir", type=Path, required=True)
    targets.add_argument("--horizons", nargs="*", choices=[h.key for h in HORIZONS])
    targets.set_defaults(func=command_targets)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError, httpx.HTTPError) as exc:
        print(f"fmfm: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
