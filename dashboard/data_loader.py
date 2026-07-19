"""Aggregate autoresearch run artifacts into a dashboard payload."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any  # noqa: F401 — used in nested helpers

# Project root is loop/ (parent of dashboard/).
LOOP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = LOOP_ROOT / "output" / "autoresearch"
# Alias for older callers.
REPO_ROOT = LOOP_ROOT


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if limit is not None and limit > 0:
        lines = lines[-limit:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def list_runs(runs_root: Path | None = None) -> list[dict[str, Any]]:
    """List autoresearch run directories (newest first)."""
    root = Path(runs_root or DEFAULT_RUNS_ROOT)
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        status = _read_json(path / "status.json") or {}
        runs.append(
            {
                "id": path.name,
                "path": str(path),
                "mtime": _mtime(path),
                "phase": status.get("phase"),
                "iteration": status.get("iteration"),
                "stopped": status.get("stopped"),
                "updated_at": status.get("updated_at"),
                "message": status.get("message"),
            }
        )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def resolve_run_dir(
    run_id: str | None = None,
    *,
    runs_root: Path | None = None,
    run_dir: Path | None = None,
) -> Path | None:
    if run_dir is not None:
        p = Path(run_dir)
        return p if p.is_dir() else None
    root = Path(runs_root or DEFAULT_RUNS_ROOT)
    if run_id:
        p = root / run_id
        return p if p.is_dir() else None
    runs = list_runs(root)
    if not runs:
        return None
    return Path(runs[0]["path"])


def _acc(blob: Any) -> float | None:
    if blob is None:
        return None
    if isinstance(blob, (int, float)):
        return float(blob)
    if not isinstance(blob, dict):
        return None
    instant = blob.get("instant")
    if isinstance(instant, dict) and instant.get("accuracy") is not None:
        return float(instant["accuracy"])
    if blob.get("accuracy") is not None:
        return float(blob["accuracy"])
    return None


def _extract_eval_point(diff: dict[str, Any] | None) -> dict[str, Any]:
    if not diff:
        return {
            "instant": None,
            "high": None,
            "gap": None,
            "instant_correct": None,
            "instant_completed": None,
        }
    instant = diff.get("instant") or {}
    high = diff.get("high") or {}
    return {
        "instant": instant.get("accuracy"),
        "high": high.get("accuracy"),
        "gap": diff.get("accuracy_gap"),
        "instant_correct": instant.get("correct"),
        "instant_completed": instant.get("completed"),
        "high_correct": high.get("correct"),
        "high_completed": high.get("completed"),
    }


def _nested_high_acc(diff: Any) -> float | None:
    if not isinstance(diff, dict):
        return None
    high = diff.get("high")
    if isinstance(high, dict) and high.get("accuracy") is not None:
        return float(high["accuracy"])
    return None


def _is_imported_history_record(rec: dict[str, Any]) -> bool:
    """True for the imported parent-checkpoint generation (gen 1)."""
    kind = str(rec.get("kind") or "")
    if kind in {
        "imported",
        "imported_checkpoint",
        "parent_import",
        "parent_checkpoint",
    }:
        return True
    if rec.get("source") in {"parent_train_promoted", "imported_checkpoint"}:
        return True
    try:
        if int(rec.get("iteration")) == -1:  # type: ignore[arg-type]
            return True
    except (TypeError, ValueError):
        pass
    return False


def _history_has_imported_slot(history: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(r, dict) and _is_imported_history_record(r) for r in history
    )


def _history_record_generation(
    rec: dict[str, Any],
    *,
    has_imported_slot: bool,
) -> int:
    """Map a history record onto the chart generation axis.

    Convention when a parent checkpoint was imported:
      gen 0 = baseline, gen 1 = imported, gen (loop_iter + 2) = loop train k
    Without import:
      gen 0 = baseline, gen (loop_iter + 1) = loop train k
    Explicit ``generation`` on the record always wins.
    """
    if rec.get("generation") is not None:
        try:
            return int(rec["generation"])
        except (TypeError, ValueError):
            pass
    if _is_imported_history_record(rec):
        return 1
    try:
        loop_it = int(rec.get("iteration"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return loop_it + (2 if has_imported_slot else 1)


def _is_interim_history_record(rec: dict[str, Any]) -> bool:
    """True when post-eval was early-finalized so later gens can start.

    These points still belong on their own generation axis slot; CI refine
    continues in the background and must update *that* gen, never the next one.
    """
    if rec.get("ci_refine_pending") is True:
        return True
    post = None
    heldout = rec.get("heldout")
    if isinstance(heldout, dict):
        post = heldout.get("post")
    if post is None:
        post = rec.get("post_heldout") or rec.get("post")
    if isinstance(post, dict) and post.get("role") in {
        "post_interim",
        "interim",
        "ci_refine_pending",
    }:
        return True
    note = str(rec.get("note") or "").lower()
    if "interim" in note and "ci refine" in note:
        return True
    return False


def _loop_iter_dir(run_dir: Path, loop_it: int) -> Path:
    return run_dir / f"iter_{int(loop_it):03d}"


def _ci_refine_live_from_history(
    run_dir: Path | None,
    history: list[dict[str, Any]],
    *,
    has_imported_slot: bool,
) -> dict[str, Any] | None:
    """Live overlay for an interim history gen whose CI refine is still running.

    Reads ``iter_XXX/eval_post/*/eval_progress.json`` (and optional
    ``ci_refine_job.json``) so gen N interim metrics stay on gen N while the
    loop advances to generate/train later iterations.
    """
    if run_dir is None or not run_dir.is_dir():
        return None

    interim_rec: dict[str, Any] | None = None
    for rec in history:
        if isinstance(rec, dict) and _is_interim_history_record(rec):
            interim_rec = rec
            # Prefer the latest interim (highest generation) if several exist.
    if interim_rec is None:
        # Also pick up refine job even if history flag already cleared mid-write.
        for rec in reversed(history):
            if not isinstance(rec, dict) or _is_imported_history_record(rec):
                continue
            try:
                lit = int(rec.get("iteration"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if lit < 0:
                continue
            job = _read_json(_loop_iter_dir(run_dir, lit) / "ci_refine_job.json")
            prog = _read_json(
                _loop_iter_dir(run_dir, lit) / "eval_post" / "heldout" / "eval_progress.json"
            )
            if job or (isinstance(prog, dict) and prog.get("status") == "in_progress"):
                interim_rec = rec
                break
        if interim_rec is None:
            return None

    gen = _history_record_generation(interim_rec, has_imported_slot=has_imported_slot)
    try:
        loop_it = int(interim_rec.get("iteration"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        loop_it = max(0, gen - (2 if has_imported_slot else 1))

    iter_dir = _loop_iter_dir(run_dir, loop_it)
    job = _read_json(iter_dir / "ci_refine_job.json")
    held_prog = _read_json(iter_dir / "eval_post" / "heldout" / "eval_progress.json") or {}
    # Prefer refine tag progress; fall back to whatever is in heldout progress.
    prog = held_prog if isinstance(held_prog, dict) else {}
    refine_running = bool(
        (isinstance(job, dict) and interim_rec.get("ci_refine_pending") is not False)
        or prog.get("status") == "in_progress"
        or "ci_refine" in str(prog.get("tag") or "")
        or interim_rec.get("ci_refine_pending") is True
    )
    if not refine_running and not _is_interim_history_record(interim_rec):
        return None

    # Seed from interim history, then overlay live refine progress when present.
    heldout = interim_rec.get("heldout") if isinstance(interim_rec.get("heldout"), dict) else {}
    post_h = heldout.get("post") or interim_rec.get("post_heldout")
    point: dict[str, Any] = {
        "generation": gen,
        "kind": "live",
        "label": f"gen {gen} · interim (CI refine)",
        "loop_iteration": loop_it,
        "kinds": ["eval", "interim"],
        "heldout_instant": _acc(post_h),
        "heldout_high": _nested_high_acc(post_h),
        "ci_refine_pending": True,
    }
    if isinstance(post_h, dict):
        ci = post_h.get("ci") if isinstance(post_h.get("ci"), dict) else {}
        inst_ci = ci.get("instant") if isinstance(ci.get("instant"), dict) else {}
        high_ci = ci.get("high") if isinstance(ci.get("high"), dict) else {}
        if inst_ci.get("half_width") is not None:
            point["heldout_instant_ci_half"] = float(inst_ci["half_width"])
        if high_ci.get("half_width") is not None:
            point["heldout_high_ci_half"] = float(high_ci["half_width"])
        n = post_h.get("sample_size") or (post_h.get("instant") or {}).get("total")
        if n is not None:
            point["n"] = n

    if prog.get("status") == "in_progress" or prog.get("instant"):
        inst = prog.get("instant") if isinstance(prog.get("instant"), dict) else {}
        high = prog.get("high") if isinstance(prog.get("high"), dict) else {}
        inst_ci = prog.get("instant_ci") if isinstance(prog.get("instant_ci"), dict) else {}
        high_ci = prog.get("high_ci") if isinstance(prog.get("high_ci"), dict) else {}
        prog_n = prog.get("n")
        hist_n = point.get("n")
        # Only replace interim metrics when refine has at least as many samples.
        # A restarted refine batch (n=40 after interim n=80) must not regress the chart.
        can_replace = hist_n is None or prog_n is None or float(prog_n) >= float(hist_n)
        point["tag"] = prog.get("tag") or point.get("tag")
        point["status"] = prog.get("status") or point.get("status")
        point["max_n"] = prog.get("max_n") or prog.get("n_pool") or point.get("max_n")
        if can_replace and inst.get("accuracy") is not None:
            point["heldout_instant"] = float(inst["accuracy"])
            if high.get("accuracy") is not None:
                point["heldout_high"] = float(high["accuracy"])
            if inst_ci.get("half_width") is not None:
                point["heldout_instant_ci_half"] = float(inst_ci["half_width"])
            if high_ci.get("half_width") is not None:
                point["heldout_high_ci_half"] = float(high_ci["half_width"])
            if prog_n is not None:
                point["n"] = prog_n
        display_n = point.get("n") if point.get("n") is not None else prog_n
        display_max = point.get("max_n") or prog.get("n_pool") or 200
        point["label"] = f"gen {gen} · interim CI refine n={display_n}/{display_max}"

    if point.get("heldout_instant") is None and point.get("heldout_high") is None:
        return None
    return point


def _history_series(history: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Build chart series from history.jsonl dual-eval records."""
    series: dict[str, list[Any]] = {
        "iteration": [],
        "train_seed_instant_pre": [],
        "train_seed_instant_post": [],
        "train_seed_high_post": [],
        "heldout_instant_pre": [],
        "heldout_instant_post": [],
        "heldout_high_post": [],
        "overfit_gap": [],
        "train_pool_size": [],
        "new_examples": [],
    }
    for rec in history:
        train_seed = rec.get("train_seed") if isinstance(rec.get("train_seed"), dict) else {}
        heldout = rec.get("heldout") if isinstance(rec.get("heldout"), dict) else {}

        pre_t = train_seed.get("pre") or rec.get("pre_train") or rec.get("pre")
        post_t = train_seed.get("post") or rec.get("post_train") or rec.get("post")
        pre_h = heldout.get("pre") or rec.get("pre_heldout")
        post_h = heldout.get("post") or rec.get("post_heldout")

        series["iteration"].append(rec.get("iteration"))
        series["train_pool_size"].append(rec.get("train_pool_size"))
        series["new_examples"].append(rec.get("new_examples"))
        series["train_seed_instant_pre"].append(_acc(pre_t))
        series["train_seed_instant_post"].append(_acc(post_t))
        series["heldout_instant_pre"].append(_acc(pre_h))
        series["heldout_instant_post"].append(_acc(post_h))
        series["train_seed_high_post"].append(_nested_high_acc(post_t))
        series["heldout_high_post"].append(_nested_high_acc(post_h))

        delta = rec.get("delta") if isinstance(rec.get("delta"), dict) else {}
        overfit = delta.get("overfit_gap")
        if overfit is None and _acc(post_t) is not None and _acc(post_h) is not None:
            overfit = float(_acc(post_t)) - float(_acc(post_h))  # type: ignore[arg-type]
        series["overfit_gap"].append(overfit)
    return series


def _checkpoints_view(
    run_dir: Path,
    indexed: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify finalized (per-iter last) vs intermediate checkpoints."""
    by_iter: dict[int, list[dict[str, Any]]] = {}
    for rec in indexed:
        it = rec.get("iteration")
        if it is None:
            continue
        try:
            it_i = int(it)
        except (TypeError, ValueError):
            continue
        by_iter.setdefault(it_i, []).append(rec)

    finalized: list[dict[str, Any]] = []
    intermediate: list[dict[str, Any]] = []

    for it_i in sorted(by_iter):
        group = by_iter[it_i]
        # Prefer explicit name/batch order
        def _key(r: dict[str, Any]) -> tuple:
            name = str(r.get("name") or "")
            batch = r.get("batch")
            try:
                batch_n = int(batch) if batch is not None else -1
            except (TypeError, ValueError):
                batch_n = -1
            return (batch_n, name)

        group_sorted = sorted(group, key=_key)
        for r in group_sorted[:-1]:
            intermediate.append({**r, "role": "intermediate", "iteration": it_i})
        if group_sorted:
            finalized.append({**group_sorted[-1], "role": "finalized", "iteration": it_i})

    # Also scan train logs for live intermediate progress (metrics tail)
    live_intermediate = None
    train_progress = _read_json(run_dir / "train_progress.json")
    if isinstance(train_progress, dict):
        ckpts = train_progress.get("checkpoints") or []
        if ckpts:
            last = ckpts[-1]
            live_intermediate = {
                "name": last.get("name"),
                "batch": last.get("batch"),
                "state_path": last.get("state_path"),
                "sampler_path": last.get("sampler_path"),
                "iteration": train_progress.get("iteration"),
                "train_log": train_progress.get("train_log"),
                "role": "live_or_latest",
                "source": "train_progress.json",
            }

    # Latest overall
    last_finalized = finalized[-1] if finalized else None
    last_intermediate = intermediate[-1] if intermediate else None
    if live_intermediate and (
        not last_finalized
        or str(live_intermediate.get("name")) != str(last_finalized.get("name"))
    ):
        # Prefer live as "last intermediate" when it differs from last finalized
        last_intermediate = live_intermediate

    # From history train meta
    history_finals: list[dict[str, Any]] = []
    for rec in history:
        train = rec.get("train") if isinstance(rec.get("train"), dict) else None
        if not train:
            continue
        history_finals.append(
            {
                "iteration": rec.get("iteration"),
                "state_path": train.get("state_path"),
                "sampler_path": train.get("sampler_path"),
                "log_path": train.get("log_path"),
                "policy_source": train.get("policy_source"),
                "judge_source": train.get("judge_source"),
                "role": "finalized",
                "source": "history.train",
            }
        )
        if history_finals and not last_finalized:
            last_finalized = history_finals[-1]

    return {
        "finalized": finalized or history_finals,
        "intermediate": intermediate,
        "last_finalized": last_finalized or (history_finals[-1] if history_finals else None),
        "last_intermediate": last_intermediate,
        "live_train_progress": train_progress,
        "count_finalized": len(finalized or history_finals),
        "count_intermediate": len(intermediate),
    }


def _latest_eval_from_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "train_seed": {"pre": None, "post": None},
            "heldout": {"pre": None, "post": None},
            "overfit_gap": None,
            "iteration": None,
        }
    last = history[-1]
    train_seed = last.get("train_seed") if isinstance(last.get("train_seed"), dict) else {}
    heldout = last.get("heldout") if isinstance(last.get("heldout"), dict) else {}
    pre_t = train_seed.get("pre") or last.get("pre_train") or last.get("pre")
    post_t = train_seed.get("post") or last.get("post_train") or last.get("post")
    pre_h = heldout.get("pre") or last.get("pre_heldout")
    post_h = heldout.get("post") or last.get("post_heldout")
    delta = last.get("delta") if isinstance(last.get("delta"), dict) else {}
    overfit = delta.get("overfit_gap")
    if overfit is None and _acc(post_t) is not None and _acc(post_h) is not None:
        overfit = float(_acc(post_t)) - float(_acc(post_h))  # type: ignore[arg-type]
    return {
        "iteration": last.get("iteration"),
        "train_seed": {
            "pre": _extract_eval_point(pre_t if isinstance(pre_t, dict) else None),
            "post": _extract_eval_point(post_t if isinstance(post_t, dict) else None),
            "delta_instant": (train_seed.get("delta") or {}).get("instant_accuracy")
            if isinstance(train_seed.get("delta"), dict)
            else delta.get("train_seed_instant"),
        },
        "heldout": {
            "pre": _extract_eval_point(pre_h if isinstance(pre_h, dict) else None),
            "post": _extract_eval_point(post_h if isinstance(post_h, dict) else None),
            "delta_instant": (heldout.get("delta") or {}).get("instant_accuracy")
            if isinstance(heldout.get("delta"), dict)
            else delta.get("heldout_instant"),
        },
        "overfit_gap": overfit,
    }


def _data_growth(data_snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ts": r.get("ts"),
            "iteration": r.get("iteration"),
            "source": r.get("source"),
            "size": r.get("size"),
        }
        for r in data_snapshots
    ]


def _scan_iter_dirs(run_dir: Path) -> list[dict[str, Any]]:
    iters: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("iter_*")):
        if not path.is_dir():
            continue
        metrics = _read_json(path / "metrics.json")
        iters.append(
            {
                "name": path.name,
                "path": str(path),
                "has_metrics": metrics is not None,
                "has_train": (path / "train").is_dir(),
                "has_eval_pre": (path / "eval_pre").is_dir(),
                "has_eval_post": (path / "eval_post").is_dir(),
                "n_generated": _count_csv_rows(path / "generated.csv"),
                "n_validated": _count_csv_rows(path / "validated.csv"),
            }
        )
    return iters


def _count_csv_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        # header + rows
        n = sum(1 for _ in path.open(encoding="utf-8"))
        return max(0, n - 1)
    except OSError:
        return None


def status_baseline_fallback(run_dir: Path) -> dict[str, Any] | None:
    """If baseline.json missing, try status.json embedded baseline."""
    status = _read_json(run_dir / "status.json") or {}
    b = status.get("baseline")
    return b if isinstance(b, dict) else None


def _metric_acc(blob: Any) -> float | None:
    if isinstance(blob, dict):
        if blob.get("accuracy") is not None:
            return float(blob["accuracy"])
        inst = blob.get("instant")
        if isinstance(inst, dict) and inst.get("accuracy") is not None:
            return float(inst["accuracy"])
    if isinstance(blob, (int, float)):
        return float(blob)
    return None


def _metric_half(blob: Any) -> float | None:
    if not isinstance(blob, dict):
        return None
    ci = blob.get("ci")
    if isinstance(ci, dict) and ci.get("half_width") is not None:
        return float(ci["half_width"])
    return None


def _build_accuracy_chart(
    *,
    history: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    status: dict[str, Any],
    eval_progress_live: dict[str, Any],
    ckpt_view: dict[str, Any],
    headline: dict[str, Any],
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a clean generation timeline for the accuracy chart.

    - generation 0: parent baseline only (never mixed with run metrics)
    - generation 1: imported parent checkpoint (when present)
    - generation k+offset: loop iteration k (finalized or interim)
    - interim gens keep their slot while background CI refine runs
    - live: active eval for current loop work, or CI refine for interim gen
    """
    points: list[dict[str, Any]] = []

    if baseline:
        ti = baseline.get("train_seed_instant")
        hi = baseline.get("heldout_instant")
        points.append(
            {
                "generation": 0,
                "kind": "baseline",
                "label": "gen 0 · baseline",
                "heldout_instant": hi,
                "heldout_high": baseline.get("heldout_high"),
                "train_seed_instant": ti,
                "train_seed_high": baseline.get("train_seed_high"),
                "heldout_instant_ci_half": baseline.get("heldout_instant_ci_half"),
                "heldout_high_ci_half": baseline.get("heldout_high_ci_half"),
                "train_seed_instant_ci_half": baseline.get("train_seed_instant_ci_half"),
                "overfit_gap": (
                    float(ti) - float(hi)
                    if ti is not None and hi is not None
                    else None
                ),
            }
        )

    has_imported_slot = _history_has_imported_slot(history)
    finalized_gens: set[int] = set()
    interim_gens: set[int] = set()
    for rec in history:
        raw_it = rec.get("iteration")
        gen = _history_record_generation(rec, has_imported_slot=has_imported_slot)
        train_seed = rec.get("train_seed") if isinstance(rec.get("train_seed"), dict) else {}
        heldout = rec.get("heldout") if isinstance(rec.get("heldout"), dict) else {}
        post_t = train_seed.get("post") or rec.get("post_train") or rec.get("post")
        post_h = heldout.get("post") or rec.get("post_heldout")
        ti = _acc(post_t)
        hi = _acc(post_h)
        th = _nested_high_acc(post_t)
        hh = _nested_high_acc(post_h)

        def _half_from_post(post: Any, key: str = "instant") -> float | None:
            if not isinstance(post, dict):
                return None
            ci = post.get("ci")
            if not isinstance(ci, dict):
                return None
            side = ci.get(key)
            if isinstance(side, dict) and side.get("half_width") is not None:
                return float(side["half_width"])
            return None

        is_import = _is_imported_history_record(rec)
        is_interim = (not is_import) and _is_interim_history_record(rec)
        if is_import:
            kind = "imported"
            label = f"gen {gen} · imported"
        elif is_interim:
            kind = "interim"
            n = None
            if isinstance(post_h, dict):
                n = post_h.get("sample_size") or (post_h.get("instant") or {}).get("total")
            label = f"gen {gen} · interim" + (f" n={n}" if n is not None else "")
            interim_gens.add(gen)
        else:
            kind = "finalized"
            label = f"gen {gen} · finalized"
            finalized_gens.add(gen)
        # Interim still occupies its generation slot so live for later iters
        # cannot collide / replot the same metrics one step to the right.
        if is_import or is_interim:
            finalized_gens.add(gen)

        points.append(
            {
                "generation": gen,
                "kind": kind,
                "label": label,
                "loop_iteration": raw_it,
                "heldout_instant": hi,
                "heldout_high": hh,
                "train_seed_instant": ti,
                "train_seed_high": th,
                "heldout_instant_ci_half": _half_from_post(post_h, "instant"),
                "heldout_high_ci_half": _half_from_post(post_h, "high"),
                "train_seed_instant_ci_half": _half_from_post(post_t, "instant"),
                "train_seed_high_ci_half": _half_from_post(post_t, "high"),
                "ci_refine_pending": bool(is_interim),
                "overfit_gap": (
                    float(ti) - float(hi)
                    if ti is not None and hi is not None
                    else None
                ),
            }
        )

    # Live for the *current* loop phase (eval_pre/post/train of status.iteration).
    live = _live_generation_point(
        status=status,
        eval_progress_live=eval_progress_live,
        ckpt_view=ckpt_view,
        headline=headline,
        finalized_gens=finalized_gens,
        has_baseline=baseline is not None,
        has_imported_slot=has_imported_slot,
    )

    # Background CI refine for interim gens (e.g. gen2 n=80 while iter1 generates).
    # Must stay on the interim generation — never appear as the next gen.
    refine_live = _ci_refine_live_from_history(
        run_dir, history, has_imported_slot=has_imported_slot
    )
    if refine_live is not None:
        if live is None:
            live = refine_live
        else:
            live_gen = live.get("generation")
            refine_gen = refine_live.get("generation")
            # Drop a "next gen" live that only echoes interim metrics / no own eval.
            live_has_own_eval = bool(
                live.get("tag")
                or (live.get("n") is not None and "eval" in (live.get("kinds") or []))
            ) and live_gen not in interim_gens
            if live_gen == refine_gen:
                # Same slot: prefer refine progress overlay.
                live = refine_live
            elif not live_has_own_eval:
                live = refine_live
            # else keep main live (real eval for a later gen); interim stays in points

    gens = [p["generation"] for p in points]
    max_final = max(gens) if gens else (0 if baseline else -1)
    live_gen_x = live["generation"] if live and live.get("heldout_instant") is not None else None
    x_max = max(max_final, live_gen_x if live_gen_x is not None else max_final, 1)

    return {
        "points": points,
        "live": live,
        "x_min": 0 if baseline else (min(gens) if gens else 0),
        "x_max": x_max,
        "x_suggested_max": max(x_max, 2),  # avoid single-tick crumple
    }


def _live_generation_point(
    *,
    status: dict[str, Any],
    eval_progress_live: dict[str, Any],
    ckpt_view: dict[str, Any],
    headline: dict[str, Any],
    finalized_gens: set[int],
    has_baseline: bool,
    has_imported_slot: bool = False,
) -> dict[str, Any] | None:
    """Current generation point not yet in history.jsonl.

    Only emits while eval/train work is actually active. Stale headline
    accuracies or leftover checkpoints must not create a ghost generation
    that re-plots the previous gen's metrics one step to the right.
    """
    prog = eval_progress_live.get("eval_progress") or status.get("eval_progress") or {}
    phase = str(status.get("phase") or headline.get("phase") or "")
    # Active when explicitly in progress, or mid eval_pre/eval_post with a progress
    # snapshot (batch boundaries briefly flip status to "complete").
    mid_eval_phase = phase.startswith("eval_") or phase in {
        "eval_pre",
        "eval_post",
        "eval_post_prelim",
    }
    prog_active = isinstance(prog, dict) and (
        prog.get("status") == "in_progress"
        or (
            mid_eval_phase
            and (prog.get("instant") is not None or prog.get("n") is not None)
        )
    )
    eval_active = bool(
        status.get("eval_in_progress")
        or eval_progress_live.get("eval_in_progress")
        or prog_active
        # First batch of a dual eval may take minutes before any progress
        # snapshot is written; still show a live marker from phase alone.
        or mid_eval_phase
    )
    train_progress = ckpt_view.get("live_train_progress")
    train_active = bool(
        isinstance(train_progress, dict)
        and (
            train_progress.get("in_progress")
            or train_progress.get("status") == "in_progress"
        )
    )
    # No active work → no live point. Headline metrics belong to history already.
    if not eval_active and not train_active:
        return None

    mid = ckpt_view.get("last_intermediate")
    live_ckpt = mid if isinstance(mid, dict) else None
    fin = ckpt_view.get("last_finalized")
    # Only surface checkpoints as "live" while training is actually running.
    if live_ckpt and not train_active:
        live_ckpt = None
    if (
        live_ckpt
        and isinstance(fin, dict)
        and live_ckpt.get("name") is not None
        and live_ckpt.get("name") == fin.get("name")
    ):
        live_ckpt = None

    raw_it = headline.get("iteration")
    # Prefer the iteration stamped on the live eval progress (CI refine can
    # still run after status.iteration has advanced to the next gen).
    prog_it = None
    if isinstance(prog, dict) and prog.get("iteration") is not None:
        prog_it = prog.get("iteration")
    elif eval_progress_live.get("iteration") is not None:
        prog_it = eval_progress_live.get("iteration")
    if eval_active and prog_it is not None:
        raw_it = prog_it
    try:
        loop_it = int(raw_it) if raw_it is not None else 0
    except (TypeError, ValueError):
        loop_it = 0
    # Gen 0 = baseline. With imported parent as gen 1, loop iter k → gen k+2.
    import_offset = 1 if has_imported_slot else 0
    gen = loop_it + 1 + import_offset

    # Training the next iteration: place live checkpoint on the upcoming gen.
    if gen in finalized_gens and not eval_active:
        if train_active and live_ckpt:
            gen = max(finalized_gens) + 1 if finalized_gens else gen
        else:
            return None

    point: dict[str, Any] = {
        "generation": gen,
        "kind": "live",
        "label": f"gen {gen} · live",
        "loop_iteration": loop_it,
        "kinds": [],
    }

    if eval_active and isinstance(prog, dict) and prog.get("instant"):
        point["kinds"].append("eval")
        tag = str(prog.get("tag") or "")
        inst = prog.get("instant") if isinstance(prog.get("instant"), dict) else {}
        high = prog.get("high") if isinstance(prog.get("high"), dict) else {}
        inst_ci = prog.get("instant_ci") if isinstance(prog.get("instant_ci"), dict) else {}
        high_ci = prog.get("high_ci") if isinstance(prog.get("high_ci"), dict) else {}
        acc_i = inst.get("accuracy")
        acc_h = high.get("accuracy")
        half_i = inst_ci.get("half_width")
        half_h = high_ci.get("half_width")
        if "heldout" in tag or ("heldout" not in tag and "train" not in tag):
            point["heldout_instant"] = acc_i
            point["heldout_high"] = acc_h
            point["heldout_instant_ci_half"] = half_i
            point["heldout_high_ci_half"] = half_h
        if "train" in tag:
            point["train_seed_instant"] = acc_i
            point["train_seed_high"] = acc_h
            point["train_seed_instant_ci_half"] = half_i
        point["n"] = prog.get("n")
        point["max_n"] = prog.get("max_n") or prog.get("n_pool")
        point["tag"] = tag
        point["status"] = prog.get("status")
        point["label"] = (
            f"gen {gen} · {tag} n={prog.get('n')}/{prog.get('max_n') or prog.get('n_pool')}"
        )
    # Fill gaps from status headline only while an eval is actually running.
    # Never copy last-gen headline metrics onto the next generation.
    if eval_active:
        if point.get("heldout_instant") is None and headline.get("heldout_instant") is not None:
            # Only borrow headline when live gen matches the iteration those
            # headline numbers came from (same loop iter → same chart gen).
            head_it = headline.get("iteration")
            try:
                head_it_i = int(head_it) if head_it is not None else None
            except (TypeError, ValueError):
                head_it_i = None
            if head_it_i is None or head_it_i == loop_it:
                point["heldout_instant"] = headline.get("heldout_instant")
                point["heldout_high"] = headline.get("heldout_high")
                point.setdefault("kinds", []).append("eval")
        if (
            point.get("train_seed_instant") is None
            and headline.get("train_seed_instant") is not None
        ):
            head_it = headline.get("iteration")
            try:
                head_it_i = int(head_it) if head_it is not None else None
            except (TypeError, ValueError):
                head_it_i = None
            if head_it_i is None or head_it_i == loop_it:
                point["train_seed_instant"] = headline.get("train_seed_instant")
                point["train_seed_high"] = headline.get("train_seed_high")

    if live_ckpt and train_active:
        point["kinds"].append("checkpoint")
        point["checkpoint"] = {
            "name": live_ckpt.get("name"),
            "batch": live_ckpt.get("batch"),
            "role": live_ckpt.get("role"),
        }
        y_ref = point.get("heldout_instant")
        if y_ref is None and eval_active:
            y_ref = headline.get("heldout_instant")
        point["checkpoint_y"] = y_ref if y_ref is not None else 0.5

    # Need real accuracy (or an active eval marker) — never emit empty/ghost gens.
    if point.get("heldout_instant") is None and point.get("train_seed_instant") is None:
        # Allow a bare mid-eval marker (no samples yet) so the axis extends.
        if eval_active and mid_eval_phase:
            point.setdefault("kinds", []).append("eval")
            point["label"] = f"gen {gen} · {phase}"
            return point
        return None
    if not point.get("kinds"):
        point["kinds"] = ["eval"]
    return point


def _baseline_view(baseline_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize baseline for the dashboard (aligned heldout/train-seed preferred)."""
    if not baseline_payload:
        return None
    # Full import shape
    primary = baseline_payload.get("primary")
    if isinstance(primary, dict) and primary.get("heldout_instant"):
        hi = primary.get("heldout_instant") or {}
        hh = primary.get("heldout_high") or {}
        ti = primary.get("train_seed_instant") or {}
        th = primary.get("train_seed_high") or {}
        return {
            "source": baseline_payload.get("source_root") or baseline_payload.get("path"),
            "imported_at": baseline_payload.get("imported_at"),
            "heldout_instant": hi.get("accuracy"),
            "heldout_high": hh.get("accuracy"),
            "train_seed_instant": ti.get("accuracy"),
            "train_seed_high": th.get("accuracy"),
            "heldout_instant_ci_half": (hi.get("ci") or {}).get("half_width"),
            "heldout_high_ci_half": (hh.get("ci") or {}).get("half_width"),
            "train_seed_instant_ci_half": (ti.get("ci") or {}).get("half_width"),
            "train_seed_high_ci_half": (th.get("ci") or {}).get("half_width"),
            "heldout_instant_n": hi.get("total"),
            "heldout_high_n": hh.get("total"),
            "gap_high_minus_instant": (
                (primary.get("heldout_instant_vs_high") or {}).get("diff")
            ),
            "note": baseline_payload.get("note"),
            "raw_primary": primary,
        }
    # Compact status.json shape
    if baseline_payload.get("heldout_instant") is not None or isinstance(
        baseline_payload.get("heldout_instant"), dict
    ):
        hi = baseline_payload.get("heldout_instant")
        hh = baseline_payload.get("heldout_high")
        ti = baseline_payload.get("train_seed_instant")
        th = baseline_payload.get("train_seed_high")

        def _acc(x: Any) -> float | None:
            if isinstance(x, dict):
                return x.get("accuracy")
            if isinstance(x, (int, float)):
                return float(x)
            return None

        return {
            "source": baseline_payload.get("path"),
            "imported_at": baseline_payload.get("imported_at"),
            "heldout_instant": _acc(hi),
            "heldout_high": _acc(hh),
            "train_seed_instant": _acc(ti),
            "train_seed_high": _acc(th),
            "gap_high_minus_instant": baseline_payload.get(
                "heldout_gap_high_minus_instant"
            ),
            "note": "Baseline from status.json",
        }
    return None


def _in_progress_point(
    *,
    status: dict[str, Any],
    eval_progress_live: dict[str, Any],
    ckpt_view: dict[str, Any],
    history: list[dict[str, Any]],
    headline: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a single provisional chart point for live eval / intermediate ckpt.

    Plotted on the same iteration axis as completed history, with a distinct
    style on the dashboard (diamond / dashed).
    """
    prog = eval_progress_live.get("eval_progress") or status.get("eval_progress") or {}
    eval_live = bool(
        status.get("eval_in_progress")
        or eval_progress_live.get("eval_in_progress")
        or (isinstance(prog, dict) and prog.get("status") == "in_progress")
    )
    mid = ckpt_view.get("last_intermediate")
    live_ckpt = None
    if isinstance(mid, dict) and mid.get("role") in {"intermediate", "live_or_latest"}:
        live_ckpt = mid
    # Only surface intermediate if it is not the same as last finalized name
    fin = ckpt_view.get("last_finalized")
    if (
        live_ckpt
        and isinstance(fin, dict)
        and live_ckpt.get("name") is not None
        and live_ckpt.get("name") == fin.get("name")
        and live_ckpt.get("role") != "live_or_latest"
    ):
        live_ckpt = None

    if not eval_live and not live_ckpt:
        return None

    it = headline.get("iteration")
    if it is None and history:
        it = history[-1].get("iteration")
    if it is None:
        it = 0
    # Place provisional point at current iteration index (may equal last history)
    try:
        it_f = float(it)
    except (TypeError, ValueError):
        it_f = float(len(history))

    point: dict[str, Any] = {
        "iteration": it_f,
        "kinds": [],
        "label": "",
    }
    if eval_live and isinstance(prog, dict):
        point["kinds"].append("eval")
        tag = str(prog.get("tag") or "eval")
        instant = prog.get("instant") if isinstance(prog.get("instant"), dict) else {}
        high = prog.get("high") if isinstance(prog.get("high"), dict) else {}
        inst_ci = prog.get("instant_ci") if isinstance(prog.get("instant_ci"), dict) else {}
        high_ci = prog.get("high_ci") if isinstance(prog.get("high_ci"), dict) else {}
        # Prefer heldout tag for primary series; train_seed goes to train series
        is_heldout = "heldout" in tag
        is_train = "train" in tag
        acc_i = instant.get("accuracy")
        acc_h = high.get("accuracy")
        half_i = inst_ci.get("half_width")
        half_h = high_ci.get("half_width")
        if is_heldout or not is_train:
            point["heldout_instant"] = acc_i
            point["heldout_high"] = acc_h
            point["heldout_instant_ci_half"] = half_i
            point["heldout_high_ci_half"] = half_h
        if is_train or not is_heldout:
            point["train_seed_instant"] = acc_i
            point["train_seed_high"] = acc_h
            point["train_seed_instant_ci_half"] = half_i
        point["n"] = prog.get("n")
        point["max_n"] = prog.get("max_n") or prog.get("n_pool")
        point["target_ci_pp"] = prog.get("target_ci_pp")
        point["p_value"] = prog.get("p_value")
        point["comparisons"] = prog.get("comparisons") or []
        point["tag"] = tag
        point["label"] = f"eval {tag} n={prog.get('n')}/{prog.get('max_n') or prog.get('n_pool')}"

    if live_ckpt:
        point["kinds"].append("checkpoint")
        point["checkpoint"] = {
            "name": live_ckpt.get("name"),
            "batch": live_ckpt.get("batch"),
            "iteration": live_ckpt.get("iteration"),
            "role": live_ckpt.get("role"),
            "sampler_path": live_ckpt.get("sampler_path"),
            "state_path": live_ckpt.get("state_path"),
        }
        # Marker y: last known heldout instant, else mid accuracy scale
        y_ref = point.get("heldout_instant")
        if y_ref is None and history:
            last = history[-1]
            y_ref = _acc(
                (last.get("heldout") or {}).get("post")
                if isinstance(last.get("heldout"), dict)
                else last.get("post_heldout")
            )
        if y_ref is None:
            y_ref = headline.get("heldout_instant")
        point["checkpoint_y"] = y_ref if y_ref is not None else 0.5
        ck_lab = f"ckpt {live_ckpt.get('name') or live_ckpt.get('batch')}"
        point["label"] = (point["label"] + " · " if point["label"] else "") + ck_lab

    return point


def build_snapshot(
    run_dir: Path | None,
    *,
    events_limit: int = 80,
) -> dict[str, Any]:
    """Full dashboard payload for one run (or empty shell if none)."""
    now = datetime.now(timezone.utc).isoformat()
    if run_dir is None or not run_dir.is_dir():
        return {
            "ok": False,
            "error": "No autoresearch run found",
            "generated_at": now,
            "run": None,
            "runs": list_runs(),
        }

    status = _read_json(run_dir / "status.json") or {}
    config = _read_json(run_dir / "config.json") or {}
    state = _read_json(run_dir / "state.json") or {}
    split = _read_json(run_dir / "split_manifest.json") or {}
    integration = _read_json(run_dir / "integration.json") or {}
    history = _read_jsonl(run_dir / "history.jsonl")
    events = _read_jsonl(run_dir / "events.jsonl", limit=events_limit)
    checkpoints_indexed = _read_jsonl(run_dir / "checkpoints.jsonl")
    data_snapshots = _read_jsonl(run_dir / "data_snapshots.jsonl")
    eval_progress_live = _read_json(run_dir / "eval_progress.json") or {}
    baseline_payload = (
        _read_json(run_dir / "baseline" / "baseline.json")
        or status_baseline_fallback(run_dir)
    )

    ckpt_view = _checkpoints_view(run_dir, checkpoints_indexed, history)
    latest_eval = _latest_eval_from_history(history)
    series = _history_series(history)

    # Headline numbers prefer status, fall back to latest history
    headline = {
        "phase": status.get("phase") or ("stopped" if status.get("stopped") else "unknown"),
        "iteration": status.get("iteration", state.get("iteration")),
        "message": status.get("message") or "",
        "stopped": bool(status.get("stopped")),
        "stop_reason": status.get("stop_reason"),
        "updated_at": status.get("updated_at"),
        "started_at": status.get("started_at"),
        "train_pool_size": status.get("train_pool_size"),
        "heldout_size": status.get("heldout_size") or split.get("n_heldout"),
        "policy_source": status.get("policy_source"),
        "judge_source": status.get("judge_source"),
        "train_seed_instant": status.get("last_instant_accuracy")
        or (latest_eval["train_seed"]["post"] or {}).get("instant"),
        "train_seed_high": status.get("last_high_accuracy")
        or (latest_eval["train_seed"]["post"] or {}).get("high"),
        "heldout_instant": status.get("last_heldout_instant_accuracy")
        or (latest_eval["heldout"]["post"] or {}).get("instant"),
        "heldout_high": status.get("last_heldout_high_accuracy")
        or (latest_eval["heldout"]["post"] or {}).get("high"),
        "overfit_gap": status.get("last_overfit_gap") or latest_eval.get("overfit_gap"),
        "last_instant_delta": status.get("last_instant_delta"),
        "last_heldout_instant_delta": status.get("last_heldout_instant_delta"),
        "eval_in_progress": bool(
            status.get("eval_in_progress")
            or eval_progress_live.get("eval_in_progress")
        ),
        "target_ci_pp": (eval_progress_live.get("eval_progress") or {}).get(
            "target_ci_pp"
        )
        or (status.get("eval_progress") or {}).get("target_ci_pp")
        or config.get("target_ci_pp"),
        "p_value": (eval_progress_live.get("eval_progress") or {}).get("p_value")
        or config.get("p_value"),
    }

    baseline = _baseline_view(baseline_payload)
    if baseline:
        headline["baseline_heldout_instant"] = baseline.get("heldout_instant")
        headline["baseline_heldout_high"] = baseline.get("heldout_high")
        headline["baseline_gap"] = baseline.get("gap_high_minus_instant")

    # Clean generation timeline for the accuracy chart (linear x-axis).
    # Gen 0 = baseline only; gen k+1 = completed loop iteration k; live = current.
    chart = _build_accuracy_chart(
        history=history,
        baseline=baseline,
        status=status,
        eval_progress_live=eval_progress_live,
        ckpt_view=ckpt_view,
        headline=headline,
        run_dir=run_dir,
    )
    series["chart"] = chart
    series["in_progress"] = chart.get("live")
    # Keep legacy arrays in sync for older UI bits / tests
    series["iteration"] = [p["generation"] for p in chart["points"]]
    series["is_baseline"] = [p["kind"] == "baseline" for p in chart["points"]]
    series["heldout_instant_post"] = [p.get("heldout_instant") for p in chart["points"]]
    series["heldout_high_post"] = [p.get("heldout_high") for p in chart["points"]]
    series["train_seed_instant_post"] = [p.get("train_seed_instant") for p in chart["points"]]
    series["train_seed_high_post"] = [p.get("train_seed_high") for p in chart["points"]]
    series["heldout_instant_ci_half"] = [
        p.get("heldout_instant_ci_half") for p in chart["points"]
    ]
    series["train_seed_instant_ci_half"] = [
        p.get("train_seed_instant_ci_half") for p in chart["points"]
    ]
    series["overfit_gap"] = [p.get("overfit_gap") for p in chart["points"]]

    return {
        "ok": True,
        "generated_at": now,
        "run": {
            "id": run_dir.name,
            "path": str(run_dir),
        },
        "runs": list_runs(run_dir.parent),
        "headline": headline,
        "status": status,
        "eval_live": eval_progress_live,
        "in_progress": chart.get("live"),
        "baseline": baseline,
        "state": {
            "iteration": state.get("iteration"),
            "last_policy_state_path": state.get("last_policy_state_path"),
            "last_policy_sampler_path": state.get("last_policy_sampler_path"),
            "last_judge_model_path": state.get("last_judge_model_path"),
            "heldout_fraction": state.get("heldout_fraction"),
            "train_data": state.get("train_data"),
            "heldout_test": state.get("heldout_test"),
            "eval_sample": state.get("eval_sample"),
            "eval_heldout_sample": state.get("eval_heldout_sample"),
        },
        "split": {
            "n_train_initial": split.get("n_train_initial"),
            "n_heldout": split.get("n_heldout"),
            "n_eval_train_sample": split.get("n_eval_train_sample"),
            "n_eval_heldout_sample": split.get("n_eval_heldout_sample"),
            "heldout_fraction": split.get("heldout_fraction"),
            "note": split.get("note"),
        },
        "checkpoints": ckpt_view,
        "evals": {
            "latest": latest_eval,
            "series": series,
            "legend": {
                "train_seed": (
                    "In-domain sample from the train/seed split "
                    "(checkpoint-seeded distribution; may overfit)"
                ),
                "heldout": (
                    "Never-train held-out test carved from original seed "
                    "(generalization)"
                ),
            },
            "target_ci_pp": headline.get("target_ci_pp"),
            "p_value": headline.get("p_value"),
        },
        "data_growth": _data_growth(data_snapshots),
        "events": events,
        "iterations": _scan_iter_dirs(run_dir),
        "config": config,
        "integration": {
            "policy_renderer": (integration.get("policy_renderer")),
            "judge_renderer": (integration.get("judge_renderer")),
            "scripts": integration.get("scripts"),
        },
    }
