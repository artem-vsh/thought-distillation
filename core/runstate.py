"""Persisted, resumable loop state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import os

from core.io import load_json, save_json


@dataclass
class LoopState:
    """Persisted state for resumable autoresearch runs."""

    run_dir: str
    seed_data: str
    train_data: str
    # Fixed eval sample drawn from the *train* split (in-distribution / can overfit).
    eval_sample: str
    # Full held-out test pool + fixed sample for evals (never used in training).
    heldout_test: str = ""
    eval_heldout_sample: str = ""
    heldout_fraction: float = 0.10
    iteration: int = 0
    policy_log_dirs: list[str] = field(default_factory=list)
    # Policy resume (weights) and sampler/judge (inference) from last train.
    last_policy_state_path: str | None = None
    last_policy_sampler_path: str | None = None
    # Explicit judge pointer (usually same as last_policy_sampler_path).
    last_judge_model_path: str | None = None
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    marginal_streak: int = 0
    stopped: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopState:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        # Tolerate older state.json missing newer fields.
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def load_state(path: Path) -> LoopState:
    return LoopState.from_dict(load_json(path))


def save_state(path: Path, state: LoopState) -> None:
    save_json(path, state.to_dict())




# Held per-process so re-entrant init_run calls (tests, resume after fresh
# init in one process) do not deadlock on their own lock.
_RUN_LOCK_FDS: dict[Path, int] = {}


def _acquire_run_lock(run_dir: Path) -> None:
    """Hold an exclusive advisory lock on the run dir for process lifetime.

    Prevents a second autoresearch process from resuming/initializing the
    same run concurrently and interleaving state.json / history writes.
    """
    key = run_dir.resolve()
    if key in _RUN_LOCK_FDS:
        return
    import fcntl  # POSIX; the loop already assumes a Unix environment

    fd = os.open(key / ".autoresearch.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(
            f"Run directory is locked by another autoresearch process: {key}. "
            "Stop that process before resuming this run."
        ) from exc
    _RUN_LOCK_FDS[key] = fd
