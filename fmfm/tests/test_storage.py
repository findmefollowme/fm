import json

import pandas as pd

from fmfm.data.storage import append_raw_events, read_bars_csv, write_bars_csv


def test_bars_csv_deduplicates_and_sorts(tmp_path):
    bars = pd.DataFrame({
        "timestamp": ["2026-01-01T00:01:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"],
        "close": [2, 1, 3],
    })
    path = write_bars_csv(bars, tmp_path / "bars.csv")
    loaded = read_bars_csv(path)
    assert len(loaded) == 2
    assert loaded.iloc[0]["close"] == 1
    assert loaded.iloc[1]["close"] == 3


def test_raw_events_append_jsonl(tmp_path):
    path = tmp_path / "raw" / "events.jsonl"
    assert append_raw_events(path, [{"event": "swap"}, {"event": "trade"}]) == 2
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["swap", "trade"]
