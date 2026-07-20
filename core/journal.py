"""Crash-safe per-iteration phase journal (``iteration_state.json``)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.io import load_json, save_json
from core.runstate import LoopState

def _load_or_create_iteration_journal(
    *,
    iter_dir: Path,
    iteration: int,
    config: dict[str, Any],
    state: LoopState,
    ensure_before: Callable[[Path], None],
) -> tuple[Path, dict[str, Any]]:
    """Create a durable phase journal or validate an interrupted iteration."""
    journal_path = iter_dir / "iteration_state.json"
    before_path = iter_dir / "train_data_before.csv"
    if journal_path.is_file():
        journal = load_json(journal_path)
        if not isinstance(journal, dict) or journal.get("iteration") != iteration:
            raise ValueError(f"Invalid iteration journal: {journal_path}")
        if journal.get("config") != config:
            raise ValueError(
                f"Cannot change configuration while resuming incomplete iteration "
                f"{iteration}; finish it with the original settings first"
            )
        if not before_path.is_file():
            raise FileNotFoundError(
                f"Missing pre-iteration train snapshot required for safe resume: {before_path}"
            )
        return journal_path, journal

    unexpected = [
        path.name
        for path in iter_dir.iterdir()
        if path.name != before_path.name
        and not (path.name.startswith(".") and path.name.endswith(".tmp"))
    ]
    if unexpected:
        raise RuntimeError(
            f"Refusing to reuse unjournaled iteration directory {iter_dir}; "
            f"found {sorted(unexpected)}"
        )
    if not before_path.is_file():
        ensure_before(before_path)
    journal = {
        "iteration": iteration,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "policy_before": {
            "state_path": state.last_policy_state_path,
            "sampler_path": state.last_policy_sampler_path,
            "judge_model_path": state.last_judge_model_path,
        },
        "train_data_before": str(before_path),
        "completed_phases": [],
        "phase_data": {},
    }
    save_json(journal_path, journal)
    return journal_path, journal


def _phase_done(journal: dict[str, Any], phase: str) -> bool:
    return phase in journal.get("completed_phases", [])


def _phase_data(journal: dict[str, Any], phase: str) -> dict[str, Any]:
    raw = journal.get("phase_data", {}).get(phase, {})
    return raw if isinstance(raw, dict) else {}


def _mark_phase(
    journal_path: Path,
    journal: dict[str, Any],
    phase: str,
    **data: Any,
) -> None:
    phases = journal.setdefault("completed_phases", [])
    if phase not in phases:
        phases.append(phase)
    journal.setdefault("phase_data", {})[phase] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    save_json(journal_path, journal)
