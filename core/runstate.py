"""Persisted, resumable loop state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
