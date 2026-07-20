"""Live status + event stream for autoresearch runs (dashboard-ready).

Files written under each run directory:

  status.json          — current phase / iteration / latest numbers (overwrite)
  events.jsonl         — append-only timeline of phase transitions
  checkpoints.jsonl    — every Tinker checkpoint seen across iterations
  train_progress.json  — latest train metrics snapshot (overwrite while/after train)
  data_snapshots.jsonl — train pool size after each generate/validate step
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.io import append_jsonl, ensure_dir, load_json, save_json, write_jsonl


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunStatus:
    """Point-in-time status for a dashboard to poll."""

    run_dir: str
    phase: str = "init"
    iteration: int = 0
    message: str = ""
    started_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    stopped: bool = False
    stop_reason: str | None = None
    # Live counters
    train_pool_size: int | None = None
    heldout_size: int | None = None
    # Train-seed (in-domain) eval
    last_instant_accuracy: float | None = None
    last_high_accuracy: float | None = None
    last_accuracy_gap: float | None = None
    last_instant_delta: float | None = None
    # Held-out (never-train) eval — primary generalization signal
    last_heldout_instant_accuracy: float | None = None
    last_heldout_high_accuracy: float | None = None
    last_heldout_accuracy_gap: float | None = None
    last_heldout_instant_delta: float | None = None
    # Overfitting gap: train_seed instant − heldout instant (post)
    last_overfit_gap: float | None = None
    last_policy_state_path: str | None = None
    last_policy_sampler_path: str | None = None
    last_judge_model_path: str | None = None
    last_train_log: str | None = None
    last_checkpoint_name: str | None = None
    policy_source: str | None = None  # "base" | "checkpoint"
    judge_source: str | None = None
    # Adaptive eval (CI) live snapshot for the dashboard
    eval_in_progress: bool = False
    eval_progress: dict[str, Any] = field(default_factory=dict)
    # Last completed CI summaries (heldout / train_seed)
    last_eval_ci: dict[str, Any] = field(default_factory=dict)
    # Paths for the dashboard to open
    paths: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressTracker:
    """Writes dashboard-oriented status/events under ``run_dir``."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        ensure_dir(self.run_dir)
        self.status_path = self.run_dir / "status.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.checkpoints_path = self.run_dir / "checkpoints.jsonl"
        self.train_progress_path = self.run_dir / "train_progress.json"
        self.data_snapshots_path = self.run_dir / "data_snapshots.jsonl"
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoint_index")

        if self.status_path.is_file():
            try:
                raw = load_json(self.status_path)
                known = {f.name for f in fields(RunStatus)}
                self.status = RunStatus(
                    **{k: v for k, v in raw.items() if k in known}
                )
                self.status.run_dir = str(self.run_dir)
            except (TypeError, KeyError, json.JSONDecodeError, ValueError):
                self.status = RunStatus(run_dir=str(self.run_dir))
        else:
            self.status = RunStatus(run_dir=str(self.run_dir))

    def _flush_status(self) -> None:
        self.status.updated_at = _utc_now()
        save_json(self.status_path, self.status.to_dict())
        # Dedicated file so dashboards can poll without rewriting whole status history
        save_json(self.run_dir / "eval_progress.json", {
            "eval_in_progress": self.status.eval_in_progress,
            "eval_progress": self.status.eval_progress,
            "last_eval_ci": self.status.last_eval_ci,
            "updated_at": self.status.updated_at,
            "iteration": self.status.iteration,
            "phase": self.status.phase,
        })

    def set_eval_progress(self, snapshot: dict[str, Any] | None) -> None:
        """Publish live adaptive-eval progress (or clear when None)."""
        if snapshot is None:
            self.status.eval_in_progress = False
            self.status.eval_progress = {}
        else:
            self.status.eval_in_progress = snapshot.get("status") == "in_progress"
            self.status.eval_progress = snapshot
            if snapshot.get("status") == "complete":
                tag = str(snapshot.get("tag") or "eval")
                self.status.last_eval_ci[tag] = {
                    "n": snapshot.get("n"),
                    "n_pool": snapshot.get("n_pool"),
                    "target_ci_pp": snapshot.get("target_ci_pp"),
                    "p_value": snapshot.get("p_value"),
                    "target_met": snapshot.get("target_met"),
                    "exhausted": snapshot.get("exhausted"),
                    "instant_ci": snapshot.get("instant_ci"),
                    "high_ci": snapshot.get("high_ci"),
                    "comparisons": snapshot.get("comparisons"),
                    "instant": snapshot.get("instant"),
                    "high": snapshot.get("high"),
                }
                self.status.eval_in_progress = False
        self._flush_status()

    def event(
        self,
        kind: str,
        message: str = "",
        **payload: Any,
    ) -> None:
        """Append a timeline event and refresh status.updated_at."""
        record = {
            "ts": _utc_now(),
            "kind": kind,
            "iteration": self.status.iteration,
            "phase": self.status.phase,
            "message": message,
            **payload,
        }
        append_jsonl(self.events_path, record)
        if message:
            self.status.message = message
        self._flush_status()

    def set_phase(
        self,
        phase: str,
        message: str = "",
        *,
        iteration: int | None = None,
        **extra: Any,
    ) -> None:
        if iteration is not None:
            self.status.iteration = iteration
        self.status.phase = phase
        if message:
            self.status.message = message
        if extra:
            self.status.extra.update(extra)
        self.event("phase", message or phase, phase=phase, **extra)

    def set_paths(self, **paths: str | Path) -> None:
        for key, value in paths.items():
            self.status.paths[key] = str(value)
        self._flush_status()

    def update(self, **fields: Any) -> None:
        for key, value in fields.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)
            else:
                self.status.extra[key] = value
        self._flush_status()

    def snapshot_train_pool(self, size: int, *, source: str, path: Path | None = None) -> None:
        self.status.train_pool_size = size
        append_jsonl(
            self.data_snapshots_path,
            {
                "ts": _utc_now(),
                "iteration": self.status.iteration,
                "source": source,
                "size": size,
                "path": str(path) if path else None,
            },
        )
        self._flush_status()

    def record_eval(
        self,
        tag: str,
        *,
        instant_accuracy: float | None,
        high_accuracy: float | None,
        accuracy_gap: float | None,
        out_dir: Path,
        instant_answers: Path | None = None,
        high_answers: Path | None = None,
        differential_path: Path | None = None,
        split: str = "train_seed",
    ) -> None:
        """Persist eval pointers + latest headline numbers for the dashboard.

        ``split`` is ``train_seed`` (in-domain) or ``heldout`` (never-train).
        """
        if split == "heldout":
            if instant_accuracy is not None:
                self.status.last_heldout_instant_accuracy = instant_accuracy
            if high_accuracy is not None:
                self.status.last_heldout_high_accuracy = high_accuracy
            if accuracy_gap is not None and accuracy_gap == accuracy_gap:
                self.status.last_heldout_accuracy_gap = accuracy_gap
        else:
            if instant_accuracy is not None:
                self.status.last_instant_accuracy = instant_accuracy
            if high_accuracy is not None:
                self.status.last_high_accuracy = high_accuracy
            if accuracy_gap is not None and accuracy_gap == accuracy_gap:  # not NaN
                self.status.last_accuracy_gap = accuracy_gap

        # Overfitting diagnostic when both splits are known.
        if (
            self.status.last_instant_accuracy is not None
            and self.status.last_heldout_instant_accuracy is not None
        ):
            self.status.last_overfit_gap = (
                self.status.last_instant_accuracy
                - self.status.last_heldout_instant_accuracy
            )

        eval_paths = {
            f"eval_{tag}_dir": str(out_dir),
        }
        if differential_path:
            eval_paths[f"eval_{tag}_differential"] = str(differential_path)
        if instant_answers:
            eval_paths[f"eval_{tag}_instant_answers"] = str(instant_answers)
        if high_answers:
            eval_paths[f"eval_{tag}_high_answers"] = str(high_answers)
        self.status.paths.update(eval_paths)

        self.event(
            "eval",
            f"{tag}/{split}: instant={instant_accuracy} high={high_accuracy} "
            f"gap={accuracy_gap}",
            tag=tag,
            split=split,
            instant_accuracy=instant_accuracy,
            high_accuracy=high_accuracy,
            accuracy_gap=accuracy_gap,
            out_dir=str(out_dir),
            overfit_gap=self.status.last_overfit_gap,
        )

    def index_train_log(
        self,
        train_log: Path,
        *,
        iteration: int,
    ) -> dict[str, Any]:
        """Index checkpoints + metrics from a Tinker cookbook train log dir.

        Copies ``checkpoints.jsonl`` and the tail of ``metrics.jsonl`` into
        ``checkpoint_index/`` so a dashboard can read them without walking
        nested iter dirs. Also appends each checkpoint to the run-level
        ``checkpoints.jsonl``.
        """
        train_log = Path(train_log)
        self.status.last_train_log = str(train_log)
        self.status.paths["last_train_log"] = str(train_log)

        summary: dict[str, Any] = {
            "ts": _utc_now(),
            "iteration": iteration,
            "train_log": str(train_log),
            "checkpoints": [],
            "metrics_tail": [],
            "config": None,
        }

        # Per-iteration snapshot directory
        snap = ensure_dir(self.checkpoint_dir / f"iter_{iteration:03d}")

        ckpt_src = train_log / "checkpoints.jsonl"
        if ckpt_src.is_file():
            dest = snap / "checkpoints.jsonl"
            shutil.copy2(ckpt_src, dest)
            existing_checkpoint_keys: set[tuple[Any, ...]] = set()
            existing_checkpoint_records: list[dict[str, Any]] = []
            if self.checkpoints_path.is_file():
                existing_lines = [
                    line
                    for line in self.checkpoints_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
                for line_index, existing_line in enumerate(existing_lines):
                    try:
                        existing = json.loads(existing_line)
                    except json.JSONDecodeError:
                        if line_index == len(existing_lines) - 1:
                            break
                        continue
                    if not isinstance(existing, dict):
                        continue
                    existing_checkpoint_records.append(existing)
                    existing_checkpoint_keys.add(
                        (
                            existing.get("iteration"),
                            existing.get("name"),
                            existing.get("state_path"),
                            existing.get("sampler_path"),
                        )
                    )
            for line in ckpt_src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                indexed = {
                    "ts": _utc_now(),
                    "iteration": iteration,
                    "train_log": str(train_log),
                    "name": rec.get("name"),
                    "batch": rec.get("batch"),
                    "state_path": rec.get("state_path"),
                    "sampler_path": rec.get("sampler_path"),
                }
                checkpoint_key = (
                    indexed["iteration"],
                    indexed["name"],
                    indexed["state_path"],
                    indexed["sampler_path"],
                )
                if checkpoint_key not in existing_checkpoint_keys:
                    existing_checkpoint_records.append(indexed)
                    existing_checkpoint_keys.add(checkpoint_key)
                summary["checkpoints"].append(indexed)
                if rec.get("state_path"):
                    self.status.last_policy_state_path = rec["state_path"]
                if rec.get("sampler_path"):
                    self.status.last_policy_sampler_path = rec["sampler_path"]
                    self.status.last_judge_model_path = rec["sampler_path"]
                    self.status.policy_source = "checkpoint"
                    self.status.judge_source = "checkpoint"
                if rec.get("name") is not None:
                    self.status.last_checkpoint_name = str(rec["name"])
            write_jsonl(self.checkpoints_path, existing_checkpoint_records)

        metrics_src = train_log / "metrics.jsonl"
        if metrics_src.is_file():
            shutil.copy2(metrics_src, snap / "metrics.jsonl")
            lines = [
                ln for ln in metrics_src.read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
            tail = lines[-50:]
            for ln in tail:
                try:
                    summary["metrics_tail"].append(json.loads(ln))
                except json.JSONDecodeError:
                    continue

        config_src = train_log / "config.json"
        if config_src.is_file():
            shutil.copy2(config_src, snap / "config.json")
            try:
                summary["config"] = load_json(config_src)
            except json.JSONDecodeError:
                summary["config"] = None

        # Latest cookbook eval artifacts (eval_test under iterations)
        eval_dirs = sorted(train_log.glob("iteration_*/eval_test*"))
        summary["cookbook_eval_paths"] = [str(p) for p in eval_dirs[-10:]]

        save_json(snap / "summary.json", summary)
        save_json(self.train_progress_path, summary)
        self.event(
            "checkpoint_index",
            f"indexed {len(summary['checkpoints'])} checkpoints from {train_log}",
            train_log=str(train_log),
            n_checkpoints=len(summary["checkpoints"]),
            last_checkpoint=self.status.last_checkpoint_name,
        )
        self._flush_status()
        return summary

    def mark_stopped(self, reason: str) -> None:
        self.status.stopped = True
        self.status.stop_reason = reason
        self.status.phase = "stopped"
        self.event("stopped", reason, reason=reason)
