import pandas as pd
import pytest

from fmfm.cli import _json_safe, _merge_asof_features, _utc_ms, build_parser


def test_cli_exposes_data_pipeline_commands():
    parser = build_parser()
    args = parser.parse_args([
        "fetch-candles", "--coin", "BTC", "--start", "2026-01-01T00:00:00Z",
        "--end", "2026-01-02T00:00:00Z", "--output", "btc.csv",
    ])
    assert args.command == "fetch-candles"
    assert args.coin == "BTC"
    assert args.interval == "1m"

    collector = parser.parse_args([
        "collect-live", "--coin", "BTC", "--output", "data/raw/BTC_events.jsonl",
        "--duration-seconds", "60",
    ])
    assert collector.command == "collect-live"
    assert "trades" in collector.channels

    funding = parser.parse_args([
        "fetch-funding", "--coin", "BTC", "--start", "2026-01-01T00:00:00Z",
        "--end", "2026-01-02T00:00:00Z", "--output", "funding.csv",
    ])
    assert funding.command == "fetch-funding"

    features = parser.parse_args([
        "build-features", "--input", "bars.csv", "--output", "features.csv",
        "--source-interval", "1m", "--events", "events.jsonl", "--funding", "funding.csv",
    ])
    assert features.command == "build-features"
    assert features.source_interval == "1m"
    assert str(features.events) == "events.jsonl"
    assert str(features.funding) == "funding.csv"

    evaluate = parser.parse_args([
        "evaluate-ridge", "--features", "features.csv", "--targets", "targets.csv",
        "--target-column", "forward_log_return_5m", "--output", "predictions.csv",
    ])
    assert evaluate.command == "evaluate-ridge"

    targets = parser.parse_args([
        "build-targets", "--coin", "BTC", "--input", "daily.csv",
        "--output-dir", "targets", "--source-interval", "1d",
    ])
    assert targets.source_interval == "1d"


def test_asof_feature_merge_uses_only_recent_prior_snapshots():
    index = pd.DatetimeIndex(
        ["2026-01-01T00:01:00Z", "2026-01-01T00:02:00Z"], name="timestamp"
    )
    base = pd.DataFrame({"candle_feature": [1.0, 2.0]}, index=index)
    extra = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:30Z"]),
        "spread_bps": [3.0],
    })
    merged = _merge_asof_features(base, extra, max_staleness="1min")
    assert merged.iloc[0]["spread_bps"] == 3.0
    assert pd.isna(merged.iloc[1]["spread_bps"])


def test_non_finite_metrics_serialize_as_json_null():
    assert _json_safe({"metric": float("nan")}) == {"metric": None}


def test_utc_ms_requires_timezone():
    with pytest.raises(ValueError):
        _utc_ms("2026-01-01 00:00:00")
