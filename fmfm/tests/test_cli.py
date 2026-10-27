import argparse

import pytest

from fmfm.cli import _utc_ms, build_parser


def test_cli_exposes_data_pipeline_commands():
    parser = build_parser()
    args = parser.parse_args([
        "fetch-candles", "--coin", "BTC", "--start", "2026-01-01T00:00:00Z",
        "--end", "2026-01-02T00:00:00Z", "--output", "btc.csv",
    ])
    assert args.command == "fetch-candles"
    assert args.coin == "BTC"
    assert args.interval == "1m"


def test_utc_ms_requires_timezone():
    with pytest.raises(argparse.ArgumentTypeError):
        _utc_ms("2026-01-01 00:00:00")
