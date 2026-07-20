"""Filesystem, JSON/JSONL, and subprocess primitives (task-agnostic)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# This directory's parent is the standalone project root
# (``cd loop && python -m …``).
LOOP_ROOT = Path(__file__).resolve().parent.parent
# All math scripts and data live here too (no parent-sandbox dependency).
MATH_ROOT = LOOP_ROOT
REPO_ROOT = LOOP_ROOT


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def atomic_text_writer(path: Path, *, newline: str | None = None):
    """Write a text file through a same-directory temp file and atomic replace."""
    ensure_dir(path.parent)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", newline=newline, encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def run_python(args: Sequence[str], *, cwd: Path | None = None) -> None:
    """Run a python module/script; raise on non-zero exit."""
    cmd = [sys.executable, *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or LOOP_ROOT))


def save_json(path: Path, data: Any) -> None:
    payload = json.dumps(
        data,
        indent=2,
        sort_keys=False,
        allow_nan=False,
    ) + "\n"
    with atomic_text_writer(path) as handle:
        handle.write(payload)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file with strict-JSON records."""
    lines = [json.dumps(record, allow_nan=False) for record in records]
    payload = "\n".join(lines) + ("\n" if lines else "")
    with atomic_text_writer(path) as handle:
        handle.write(payload)
