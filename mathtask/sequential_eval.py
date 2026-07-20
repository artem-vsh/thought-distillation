"""Adaptive eval: grow sample until CI half-widths meet target or pool exhausted."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from common import (
    DifferentialMetrics,
    EvalMetrics,
    ensure_dir,
    load_math_csv,
    save_json,
    write_operations_csv,
)
from core.loop_config import EvalCIConfig
from mathtask.run_evals import eval_effort
from core.stats_ci import DiffCI, ProportionCI, diff_ci, proportion_ci


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PrevInstantRef:
    """Previous post-eval instant counts for delta CI (same split)."""

    correct: int
    total: int
    accuracy: float | None = None
    label: str = "previous_instant"


ProgressCb = Callable[[dict[str, Any]], None]


@dataclass
class SequentialEvalResult:
    differential: DifferentialMetrics
    n_used: int
    n_pool: int
    exhausted: bool
    target_met: bool
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    instant_ci: dict[str, Any] | None = None
    high_ci: dict[str, Any] | None = None
    progress_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "differential": self.differential.to_dict(),
            "n_used": self.n_used,
            "n_pool": self.n_pool,
            "exhausted": self.exhausted,
            "target_met": self.target_met,
            "comparisons": self.comparisons,
            "instant_ci": self.instant_ci,
            "high_ci": self.high_ci,
            "progress_path": self.progress_path,
        }


def _write_subset_ops(pool_ops: list[str], n: int, path: Path) -> None:
    write_operations_csv(path, pool_ops[:n])


def _comparisons(
    *,
    instant: EvalMetrics,
    high: EvalMetrics | None,
    skip_high: bool,
    prev: PrevInstantRef | None,
    cfg: EvalCIConfig,
) -> list[DiffCI]:
    out: list[DiffCI] = []
    target = cfg.target_half_width
    p = cfg.p_value
    if cfg.compare_instant_vs_high and not skip_high and high is not None:
        # high − instant (same n)
        out.append(
            diff_ci(
                high.correct,
                high.total,
                instant.correct,
                instant.total,
                name="high_minus_instant",
                p_value=p,
                target_half_width=target,
            )
        )
    if cfg.compare_instant_vs_previous and prev is not None and prev.total > 0:
        out.append(
            diff_ci(
                instant.correct,
                instant.total,
                prev.correct,
                prev.total,
                name="instant_minus_previous",
                p_value=p,
                target_half_width=target,
            )
        )
    return out


def _all_required_met(comps: list[DiffCI], cfg: EvalCIConfig, *, skip_high: bool) -> bool:
    """True when every *configured* comparison that can run has met the CI target.

    If a comparison cannot run yet (e.g. no previous instant), it is not required.
    If no comparisons are active, we treat the min_sample_size as sufficient once reached.
    """
    needed: list[str] = []
    if cfg.compare_instant_vs_high and not skip_high:
        needed.append("high_minus_instant")
    if cfg.compare_instant_vs_previous:
        # only required if present in comps
        if any(c.name == "instant_minus_previous" for c in comps):
            needed.append("instant_minus_previous")
    if not needed:
        return True
    by_name = {c.name: c for c in comps}
    return all(by_name[n].meets_target for n in needed if n in by_name)


def run_sequential_differential(
    *,
    pool_csv: Path,
    out_dir: Path,
    model: str,
    instant_model_path: str | None,
    high_model_path: str | None,
    skip_high: bool,
    cfg: EvalCIConfig,
    tag: str = "",
    prev_instant: PrevInstantRef | None = None,
    progress_path: Path | None = None,
    on_progress: ProgressCb | None = None,
) -> SequentialEvalResult:
    """Grow eval sample until CIs meet ``cfg`` or the pool is exhausted.

    Uses ``ask_arithmetic`` resume semantics: answers CSVs accumulate as n grows.
    """
    ensure_dir(out_dir)
    pool = load_math_csv(pool_csv)
    pool_ops = [ex.operation for ex in pool]
    n_pool = len(pool_ops)
    if n_pool == 0:
        raise ValueError(f"Empty eval pool: {pool_csv}")

    max_n = min(n_pool, cfg.max_sample_size)
    min_n = min(max(cfg.min_sample_size, cfg.batch_size), max_n)
    n = min_n

    progress_path = progress_path or (out_dir / "eval_progress.json")
    subset_csv = out_dir / "eval_sample_growing.csv"
    # Full pool pin for audit
    if pool_csv.resolve() != (out_dir / "eval_pool.csv").resolve():
        shutil.copy2(pool_csv, out_dir / "eval_pool.csv")

    last_diff: DifferentialMetrics | None = None
    last_comps: list[DiffCI] = []
    last_inst_ci: ProportionCI | None = None
    last_high_ci: ProportionCI | None = None
    target_met = False
    exhausted = False

    while True:
        _write_subset_ops(pool_ops, n, subset_csv)
        # Stable pin name used by dual-eval reuse checks
        shutil.copy2(subset_csv, out_dir / "eval_sample.csv")

        instant = eval_effort(
            effort="instant",
            sample_csv=subset_csv,
            out_dir=out_dir,
            model=model,
            model_path=instant_model_path,
        )
        if skip_high:
            high = EvalMetrics(
                effort="high",
                total=instant.total,
                completed=0,
                correct=0,
                accuracy=0.0,
                incomplete=instant.total,
                model_path=high_model_path,
            )
            gap = None
        else:
            high = eval_effort(
                effort="high",
                sample_csv=subset_csv,
                out_dir=out_dir,
                model=model,
                model_path=high_model_path,
            )
            gap = high.accuracy - instant.accuracy

        last_diff = DifferentialMetrics(
            sample_size=instant.total,
            instant=instant,
            high=high,
            accuracy_gap=gap,
        )
        last_inst_ci = proportion_ci(instant.correct, instant.total, p_value=cfg.p_value)
        last_high_ci = (
            None
            if skip_high
            else proportion_ci(high.correct, high.total, p_value=cfg.p_value)
        )
        last_comps = _comparisons(
            instant=instant,
            high=high,
            skip_high=skip_high,
            prev=prev_instant,
            cfg=cfg,
        )
        target_met = _all_required_met(last_comps, cfg, skip_high=skip_high)
        exhausted = n >= max_n
        # If we have no prev and only prev-comparison is required, still need min n
        if not last_comps and n >= min_n:
            target_met = True

        snapshot = {
            "ts": _utc_now(),
            "tag": tag,
            "status": "complete" if (target_met or exhausted) else "in_progress",
            "n": n,
            "n_pool": n_pool,
            "max_n": max_n,
            "min_n": min_n,
            "batch_size": cfg.batch_size,
            "target_ci_pp": cfg.target_ci_pp,
            "p_value": cfg.p_value,
            "target_half_width": cfg.target_half_width,
            "target_met": target_met,
            "exhausted": exhausted,
            "instant": instant.to_dict(),
            "high": high.to_dict() if not skip_high else None,
            "accuracy_gap": gap,
            "instant_ci": last_inst_ci.to_dict() if last_inst_ci else None,
            "high_ci": last_high_ci.to_dict() if last_high_ci else None,
            "comparisons": [c.to_dict() for c in last_comps],
            "prev_instant": (
                {
                    "correct": prev_instant.correct,
                    "total": prev_instant.total,
                    "accuracy": prev_instant.accuracy,
                    "label": prev_instant.label,
                }
                if prev_instant
                else None
            ),
        }
        save_json(progress_path, snapshot)
        if on_progress:
            on_progress(snapshot)

        print(
            f"[seq-eval {tag}] n={n}/{max_n} instant={instant.accuracy:.4f}"
            f"{'' if skip_high else f' high={high.accuracy:.4f} gap={gap:+.4f}'}"
            f" | target_met={target_met} exhausted={exhausted}",
            flush=True,
        )
        for c in last_comps:
            print(
                f"  CI {c.name}: diff={c.diff:+.4f} ±{c.half_width:.4f} "
                f"(target ±{c.target_half_width:.4f}) "
                f"{'OK' if c.meets_target else 'WIDE'}",
                flush=True,
            )

        if target_met or exhausted:
            break
        next_n = min(n + cfg.batch_size, max_n)
        if next_n == n:
            exhausted = True
            break
        n = next_n

    assert last_diff is not None
    # Final differential + CI payload for dashboards
    payload = last_diff.to_dict()
    if tag:
        payload["tag"] = tag
    payload["ci"] = {
        "p_value": cfg.p_value,
        "target_ci_pp": cfg.target_ci_pp,
        "target_half_width": cfg.target_half_width,
        "instant": last_inst_ci.to_dict() if last_inst_ci else None,
        "high": last_high_ci.to_dict() if last_high_ci else None,
        "comparisons": [c.to_dict() for c in last_comps],
        "n_used": n,
        "n_pool": n_pool,
        "target_met": target_met,
        "exhausted": exhausted,
    }
    save_json(out_dir / "differential.json", payload)
    save_json(
        out_dir / "eval_manifest.json",
        {
            "tag": tag,
            "pool_csv": str(pool_csv),
            "out_dir": str(out_dir),
            "sequential": True,
            "ci": payload["ci"],
            "files": {
                "answers_instant": str(out_dir / "answers_instant.csv"),
                "scored_instant": str(out_dir / "scored_instant.csv"),
                "answers_high": str(out_dir / "answers_high.csv") if not skip_high else None,
                "scored_high": str(out_dir / "scored_high.csv") if not skip_high else None,
                "differential": str(out_dir / "differential.json"),
                "eval_progress": str(progress_path),
            },
        },
    )
    # Mark progress complete
    final_prog = {
        **payload.get("ci", {}),
        "ts": _utc_now(),
        "tag": tag,
        "status": "complete",
        "instant": last_diff.instant.to_dict(),
        "high": last_diff.high.to_dict() if not skip_high else None,
        "accuracy_gap": last_diff.accuracy_gap,
        "instant_ci": last_inst_ci.to_dict() if last_inst_ci else None,
        "high_ci": last_high_ci.to_dict() if last_high_ci else None,
        "comparisons": [c.to_dict() for c in last_comps],
        "n": n,
        "n_pool": n_pool,
        "max_n": max_n,
    }
    save_json(progress_path, final_prog)

    return SequentialEvalResult(
        differential=last_diff,
        n_used=n,
        n_pool=n_pool,
        exhausted=exhausted,
        target_met=target_met,
        comparisons=[c.to_dict() for c in last_comps],
        instant_ci=last_inst_ci.to_dict() if last_inst_ci else None,
        high_ci=last_high_ci.to_dict() if last_high_ci else None,
        progress_path=str(progress_path),
    )
