"""Durable per-iteration metrics history (``history.jsonl`` + state copy)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.io import write_jsonl

def _upsert_history_file(path: Path, record: dict[str, Any]) -> None:
    """Atomically add/replace one iteration record, tolerating a torn last line."""
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    nonempty = [(line_number, line) for line_number, line in enumerate(lines, 1) if line.strip()]
    for index, (line_number, line) in enumerate(nonempty):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(nonempty) - 1:
                break
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(parsed)

    iteration = record["iteration"]
    for index, existing in enumerate(records):
        if existing.get("iteration") == iteration:
            records[index] = record
            break
    else:
        records.append(record)
    write_jsonl(path, records)


def _upsert_state_history(
    history: list[dict[str, Any]],
    record: dict[str, Any],
) -> None:
    iteration = record["iteration"]
    for index, existing in enumerate(history):
        if existing.get("iteration") == iteration:
            history[index] = record
            return
    history.append(record)
