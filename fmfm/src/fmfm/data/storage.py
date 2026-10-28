"""Simple durable local storage for FMFM research datasets.

Raw events are append-only JSONL. Normalized bars are CSV snapshots written
atomically and deduplicated by timestamp. This avoids adding a heavyweight
database before the ingestion schema and volume requirements are measured.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def append_raw_events(path: str | Path, events: Iterable[dict[str, Any]]) -> int:
    """Append raw event dictionaries as JSON Lines; return number written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def write_bars_csv(
    bars: pd.DataFrame,
    path: str | Path,
    *,
    timestamp_col: str = "timestamp",
) -> Path:
    """Normalize timestamps, deduplicate and atomically replace a bar CSV."""
    if timestamp_col not in bars.columns:
        raise ValueError(f"Missing timestamp column {timestamp_col!r}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = bars.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="raise")
    frame = frame.sort_values(timestamp_col).drop_duplicates(timestamp_col, keep="last")
    temp = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, target)
    return target


def read_bars_csv(path: str | Path, *, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Read a normalized bar CSV with UTC timestamps and sorted unique rows."""
    frame = pd.read_csv(path)
    if timestamp_col not in frame.columns:
        raise ValueError(f"Missing timestamp column {timestamp_col!r}")
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="raise")
    return frame.sort_values(timestamp_col).drop_duplicates(timestamp_col, keep="last").reset_index(drop=True)
