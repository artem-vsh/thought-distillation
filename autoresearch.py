#!/usr/bin/env python3
"""Autoresearch loop for math: generate → validate → eval → train → re-eval.

Integrates with existing math scripts (not reimplemented):

  - ``ask_arithmetic.py``        — eval sampling (instant / high)
  - ``test_arithmetic.py``       — eval scoring
  - ``train_math_llm_judge.py``  — RL train (instant policy, high judge)
  - ``data/arithmetic_operations.csv`` — seed (copy of ``data/``)

Each iteration persists intermediate artifacts under the run dir so a
real-time dashboard can poll ``status.json`` / ``events.jsonl``:

  status.json, events.jsonl, checkpoints.jsonl, train_progress.json,
  history.jsonl, data_snapshots.jsonl, integration.json,
  iter_XXX/{generated,validated,eval_pre,eval_post,train,metrics.json}

Example::

    source .venv/bin/activate
    python -m autoresearch \\
        --seed-data data/arithmetic_operations.csv \\
        --max-iters 5 \\
        --eval-sample-size 100 \\
        --gen-target 100 \\
        --train-max-steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Standalone project root (this directory).
_LOOP_ROOT = Path(__file__).resolve().parent
if str(_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOOP_ROOT))

from core.defaults import (
    DEFAULT_EVAL_SAMPLE_SIZE,
    DEFAULT_GEN_TARGET,
    DEFAULT_GEN_TEMPERATURE,
    DEFAULT_HELDOUT_FRACTION,
    DEFAULT_MARGINAL_DELTA,
    DEFAULT_MARGINAL_STREAK,
    DEFAULT_MAX_ITERS,
    DEFAULT_MODEL,
    DEFAULT_NOISE_Z,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SEEDS_PER_BATCH,
    DEFAULT_SEED_DATA,
    DEFAULT_SEED_SAMPLES,
    DEFAULT_TRAIN_EVAL_EVERY,
    DEFAULT_TRAIN_GROUPS_PER_BATCH,
    DEFAULT_TRAIN_GROUP_SIZE,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_LORA_RANK,
    DEFAULT_TRAIN_MAX_STEPS,
    DEFAULT_TRAIN_SAVE_EVERY,
    DEFAULT_VARIATIONS_PER_SEED,
    MIN_HELDOUT_EVAL_SAMPLE,
)
from core.io import (
    append_jsonl,
    ensure_dir,
    load_json,
    save_json,
    utc_now_tag,
    write_jsonl,
)
from core.metrics import DifferentialMetrics
from core.runstate import LoopState, load_state, save_state
from core.stopping import marginal_improvement_streak
from mathtask.dataset import (
    assert_disjoint_train_heldout,
    filter_out_operations,
    load_math_csv,
    merge_unique_examples,
    operation_keys,
    sample_examples,
    split_train_heldout,
    write_math_csv,
    write_operations_csv,
    write_split_manifest,
)
from mathtask.generate_data import generate_variations
from core.loop_config import EvalCIConfig, LoopConfig, load_loop_config, resolve_config_path
from mathtask.math_integration import check_model_consistency, write_integration_manifest
from core.progress import ProgressTracker
from mathtask.sequential_eval import PrevInstantRef, run_sequential_differential
from mathtask.train_step import run_train_step
from mathtask.validate_data import validate_examples


_RESUMABLE_CONFIG_FIELDS = (
    "model",
    "max_iters",
    "marginal_delta",
    "marginal_streak",
    "noise_z",
    "gen_target",
    "seed_samples",
    "variations_per_batch",
    "seeds_per_batch",
    "gen_temperature",
    "eval_sample_size",
    "heldout_fraction",
    "train_max_steps",
    "groups_per_batch",
    "group_size",
    "save_every",
    "eval_every",
    "learning_rate",
    "lora_rank",
    "rng_seed",
    "skip_generate",
    "skip_train",
    "skip_high_eval",
    "target_ci_pp",
    "p_value",
    "eval_batch_size",
    "eval_max_sample_size",
)
_IMMUTABLE_RESUME_FIELDS = {"heldout_fraction", "eval_sample_size"}
# Explicit flags that are meaningful on resume even though they are not
# restored config: --resume itself and per-invocation controls like --force.
_RESUME_CONTROL_FIELDS = {"resume", "force"}


def _serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }


def _explicit_cli_dests(argv: list[str]) -> set[str]:
    """Return dests of options explicitly present on the command line.

    Implemented as a second parse with all defaults suppressed, so argparse
    features like unambiguous abbreviations (``--max-it``) and
    ``--no-`` boolean negations resolve to the correct dest — raw token
    scanning silently misses those and would drop user intent on resume.
    """
    parser = build_parser(suppress_defaults=True)
    namespace, _unknown = parser.parse_known_args(argv)
    return set(vars(namespace))


def resolve_resume_config(
    args: argparse.Namespace,
    *,
    argv: list[str],
) -> tuple[argparse.Namespace, dict[str, Any] | None]:
    """Restore saved run settings unless the user explicitly overrides them."""
    if args.resume is None:
        return args, None

    config_path = args.resume.resolve() / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot safely resume without the original config: {config_path}"
        )
    saved = load_json(config_path)
    if not isinstance(saved, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")

    explicit = _explicit_cli_dests(argv)
    overrides: dict[str, dict[str, Any]] = {}
    for field_name in _RESUMABLE_CONFIG_FIELDS:
        if field_name not in saved:
            continue
        current = getattr(args, field_name)
        previous = saved[field_name]
        if field_name in explicit:
            if current != previous:
                if field_name in _IMMUTABLE_RESUME_FIELDS:
                    raise ValueError(
                        f"{field_name} is fixed when a run is created "
                        f"(saved={previous!r}, requested={current!r})"
                    )
                overrides[field_name] = {"previous": previous, "effective": current}
        else:
            setattr(args, field_name, previous)

    # Options the user passed that have no effect on a resumed run
    # (e.g. --seed-data, --runs-root, --run-name). Warn instead of silently
    # dropping the intent.
    ignored_fields = sorted(
        explicit - set(_RESUMABLE_CONFIG_FIELDS) - _RESUME_CONTROL_FIELDS
    )
    if ignored_fields:
        print(
            f"WARNING: ignored on resume (run is defined by {config_path}): "
            + ", ".join(f"--{name.replace('_', '-')}" for name in ignored_fields),
            flush=True,
        )

    return args, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resume_dir": str(args.resume.resolve()),
        "explicit_fields": sorted(explicit & set(_RESUMABLE_CONFIG_FIELDS)),
        "ignored_fields": ignored_fields,
        "overrides": overrides,
        "effective_config": _serialize_args(args),
    }


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


def init_run(
    *,
    runs_root: Path,
    run_name: str | None,
    seed_data: Path,
    resume_dir: Path | None,
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
    eval_sample_size: int = DEFAULT_EVAL_SAMPLE_SIZE,
    rng_seed: int = 0,
) -> tuple[Path, LoopState, ProgressTracker]:
    """Create or resume a run directory, state, and progress tracker.

    On first start, carves a **held-out test set** from the seed that is never
    merged into ``train_data`` and never used as generation few-shot seeds.
    Fixed eval samples are taken from:
      - train split (in-domain / checkpoint-seeded track)
      - held-out split (generalization track)
    """
    if resume_dir is not None:
        run_dir = resume_dir.resolve()
        _acquire_run_lock(run_dir)
        state_path = run_dir / "state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"No state.json in {run_dir}")
        state = load_state(state_path)
        state_run_dir = Path(state.run_dir).resolve()
        if state_run_dir != run_dir:
            raise ValueError(
                f"Resume directory mismatch: state.json belongs to {state_run_dir}, "
                f"not {run_dir}"
            )
        progress = ProgressTracker(run_dir)
        # A max-iters completion marks dashboard status only, not LoopState.
        # Reset stale dashboard flags while preserving a genuine early stop.
        progress.update(stopped=state.stopped, stop_reason=state.stop_reason)
        progress.set_phase(
            "resumed",
            f"Resuming at iteration {state.iteration}",
            iteration=state.iteration,
        )
        print(f"Resuming run at iteration {state.iteration}: {run_dir}", flush=True)
        return run_dir, state, progress

    name = run_name or f"math-{utc_now_tag()}"
    ensure_dir(runs_root)
    run_dir = runs_root / name
    try:
        run_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Use --resume for that run or choose a new --run-name."
        ) from exc
    _acquire_run_lock(run_dir)
    train_data = run_dir / "train_data.csv"
    heldout_test = run_dir / "heldout_test.csv"
    # Train-seed eval sample (in-domain); kept as eval_sample for back-compat.
    eval_train_sample = run_dir / "eval_train_sample.csv"
    eval_sample = run_dir / "eval_sample.csv"  # alias copy of eval_train_sample
    eval_heldout_sample = run_dir / "eval_heldout_sample.csv"

    seed_copy = run_dir / "seed_data.csv"
    shutil.copy2(seed_data, seed_copy)

    seed_examples = load_math_csv(seed_data)
    train_ex, heldout_ex = split_train_heldout(
        seed_examples,
        heldout_fraction=heldout_fraction,
        seed=rng_seed,
    )
    # The seed itself may contain canonical duplicates (e.g. "12+34" and
    # "34+12") that the random split could separate across the boundary.
    # Drop them from the train side so the held-out set stays truly unseen.
    heldout_keys = operation_keys(heldout_ex)
    deduped_train = filter_out_operations(train_ex, heldout_keys)
    if len(deduped_train) != len(train_ex):
        dropped = len(train_ex) - len(deduped_train)
        print(
            f"WARNING: dropped {dropped} train rows canonically matching "
            "held-out operations (near-duplicate leak in seed data)",
            flush=True,
        )
        train_ex = deduped_train
    assert_disjoint_train_heldout(train_ex, heldout_ex)

    train_eval_ex = sample_examples(
        train_ex, min(eval_sample_size, len(train_ex)), seed=rng_seed + 1
    )
    heldout_eval_ex = sample_examples(
        heldout_ex, min(eval_sample_size, len(heldout_ex)), seed=rng_seed + 2
    )
    if len(heldout_eval_ex) < MIN_HELDOUT_EVAL_SAMPLE:
        print(
            f"WARNING: held-out eval sample has only {len(heldout_eval_ex)} "
            f"rows (< {MIN_HELDOUT_EVAL_SAMPLE}); held-out deltas and early "
            "stopping will be noise-dominated. Provide a larger seed dataset "
            "or raise --heldout-fraction.",
            flush=True,
        )

    write_math_csv(train_data, train_ex)
    write_math_csv(heldout_test, heldout_ex)
    write_operations_csv(eval_train_sample, [ex.operation for ex in train_eval_ex])
    write_math_csv(
        eval_train_sample.with_name("eval_train_sample_with_solutions.csv"),
        train_eval_ex,
    )
    write_operations_csv(eval_heldout_sample, [ex.operation for ex in heldout_eval_ex])
    write_math_csv(
        eval_heldout_sample.with_name("eval_heldout_sample_with_solutions.csv"),
        heldout_eval_ex,
    )
    # Back-compat alias: eval_sample == train-seed sample
    shutil.copy2(eval_train_sample, eval_sample)

    write_split_manifest(
        run_dir / "split_manifest.json",
        seed_data=seed_data,
        train=train_ex,
        heldout=heldout_ex,
        eval_train_sample=train_eval_ex,
        eval_heldout_sample=heldout_eval_ex,
        heldout_fraction=heldout_fraction,
        rng_seed=rng_seed,
    )

    state = LoopState(
        run_dir=str(run_dir),
        seed_data=str(seed_data.resolve()),
        train_data=str(train_data),
        eval_sample=str(eval_sample),
        heldout_test=str(heldout_test),
        eval_heldout_sample=str(eval_heldout_sample),
        heldout_fraction=heldout_fraction,
        iteration=0,
    )
    save_state(run_dir / "state.json", state)

    progress = ProgressTracker(run_dir)
    progress.set_paths(
        seed_data=seed_data,
        seed_copy=seed_copy,
        train_data=train_data,
        heldout_test=heldout_test,
        eval_train_sample=eval_train_sample,
        eval_sample=eval_sample,
        eval_heldout_sample=eval_heldout_sample,
        split_manifest=run_dir / "split_manifest.json",
        state=run_dir / "state.json",
        history=run_dir / "history.jsonl",
        events=run_dir / "events.jsonl",
        checkpoints=run_dir / "checkpoints.jsonl",
        train_progress=run_dir / "train_progress.json",
        integration=run_dir / "integration.json",
    )
    progress.snapshot_train_pool(len(train_ex), source="bootstrap_train", path=train_data)
    progress.update(heldout_size=len(heldout_ex))
    progress.set_phase(
        "init",
        f"Started run: train={len(train_ex)} heldout={len(heldout_ex)} "
        f"(fraction={heldout_fraction})",
    )
    progress.event(
        "split",
        "carved never-train held-out set",
        n_train=len(train_ex),
        n_heldout=len(heldout_ex),
        n_eval_train=len(train_eval_ex),
        n_eval_heldout=len(heldout_eval_ex),
        heldout_fraction=heldout_fraction,
    )
    write_integration_manifest(run_dir / "integration.json")
    print(
        f"Started run: {run_dir}\n"
        f"  train pool: {len(train_ex)} → {train_data}\n"
        f"  held-out (never train): {len(heldout_ex)} → {heldout_test}\n"
        f"  eval train-seed sample: {len(train_eval_ex)} → {eval_train_sample}\n"
        f"  eval held-out sample: {len(heldout_eval_ex)} → {eval_heldout_sample}",
        flush=True,
    )
    return run_dir, state, progress


def _prev_instant_from_metrics(
    metrics: DifferentialMetrics | None,
    *,
    label: str,
) -> PrevInstantRef | None:
    if metrics is None:
        return None
    inst = metrics.instant
    if inst.total <= 0:
        return None
    return PrevInstantRef(
        correct=inst.correct,
        total=inst.total,
        accuracy=inst.accuracy,
        label=label,
    )


def _run_dual_evals(
    *,
    tag: str,
    iter_dir: Path,
    eval_train_sample: Path,
    eval_heldout_sample: Path,
    model: str,
    sampler_path: str | None,
    skip_high: bool,
    progress: ProgressTracker,
    eval_ci: EvalCIConfig,
    prev_train_instant: DifferentialMetrics | None = None,
    prev_heldout_instant: DifferentialMetrics | None = None,
) -> tuple[DifferentialMetrics, DifferentialMetrics]:
    """Run train-seed + held-out adaptive CI evals; return (train_diff, heldout_diff)."""
    base = ensure_dir(iter_dir / f"eval_{tag}")
    train_dir = ensure_dir(base / "train_seed")
    heldout_dir = ensure_dir(base / "heldout")

    print(
        f"[eval/{tag}] adaptive CI eval "
        f"(target ±{eval_ci.target_ci_pp}pp @ p<{eval_ci.p_value}) "
        f"train-seed + held-out",
        flush=True,
    )

    def _cb(snapshot: dict[str, Any]) -> None:
        progress.set_eval_progress(snapshot)

    # Held-out first (primary generalization track). When CI is not required
    # for a split, cap at min_sample_size so we still get a preliminary dual
    # read without the adaptive growth crawl.
    heldout_cfg = (
        eval_ci
        if eval_ci.require_heldout_ci
        else replace(eval_ci, max_sample_size=eval_ci.min_sample_size)
    )
    train_seed_cfg = (
        eval_ci
        if eval_ci.require_train_seed_ci
        else replace(eval_ci, max_sample_size=eval_ci.min_sample_size)
    )

    heldout_seq = run_sequential_differential(
        pool_csv=eval_heldout_sample,
        out_dir=heldout_dir,
        model=model,
        instant_model_path=sampler_path,
        high_model_path=sampler_path,
        skip_high=skip_high,
        cfg=heldout_cfg,
        tag=f"{tag}_heldout",
        prev_instant=_prev_instant_from_metrics(
            prev_heldout_instant, label="prev_heldout_instant"
        ),
        progress_path=heldout_dir / "eval_progress.json",
        on_progress=_cb,  # always publish live status for the dashboard
    )
    heldout_diff = heldout_seq.differential
    progress.record_eval(
        f"{tag}_heldout",
        instant_accuracy=heldout_diff.instant.accuracy,
        high_accuracy=None if skip_high else heldout_diff.high.accuracy,
        accuracy_gap=None if skip_high else heldout_diff.accuracy_gap,
        out_dir=heldout_dir,
        instant_answers=heldout_dir / "answers_instant.csv",
        high_answers=None if skip_high else heldout_dir / "answers_high.csv",
        differential_path=heldout_dir / "differential.json",
        split="heldout",
    )

    train_seq = run_sequential_differential(
        pool_csv=eval_train_sample,
        out_dir=train_dir,
        model=model,
        instant_model_path=sampler_path,
        high_model_path=sampler_path,
        skip_high=skip_high,
        cfg=train_seed_cfg,
        tag=f"{tag}_train_seed",
        prev_instant=_prev_instant_from_metrics(
            prev_train_instant, label="prev_train_seed_instant"
        ),
        progress_path=train_dir / "eval_progress.json",
        on_progress=_cb,  # always publish live status for the dashboard
    )
    train_diff = train_seq.differential
    progress.record_eval(
        f"{tag}_train_seed",
        instant_accuracy=train_diff.instant.accuracy,
        high_accuracy=None if skip_high else train_diff.high.accuracy,
        accuracy_gap=None if skip_high else train_diff.accuracy_gap,
        out_dir=train_dir,
        instant_answers=train_dir / "answers_instant.csv",
        high_answers=None if skip_high else train_dir / "answers_high.csv",
        differential_path=train_dir / "differential.json",
        split="train_seed",
    )

    progress.set_eval_progress(None)
    # Persist CI summaries under the tag for metrics.json
    save_json(
        base / "sequential_ci.json",
        {
            "heldout": heldout_seq.to_dict(),
            "train_seed": train_seq.to_dict(),
            "eval_ci": {
                "target_ci_pp": eval_ci.target_ci_pp,
                "p_value": eval_ci.p_value,
                "batch_size": eval_ci.batch_size,
                "max_sample_size": eval_ci.max_sample_size,
            },
        },
    )

    overfit = train_diff.instant.accuracy - heldout_diff.instant.accuracy
    print(
        f"[eval/{tag}] train-seed instant={train_diff.instant.accuracy:.4f} "
        f"(n={train_seq.n_used}) | heldout instant={heldout_diff.instant.accuracy:.4f} "
        f"(n={heldout_seq.n_used}) | overfit_gap={overfit:+.4f}",
        flush=True,
    )
    return train_diff, heldout_diff


def _load_dual_evals(
    *,
    tag: str,
    iter_dir: Path,
) -> tuple[DifferentialMetrics, DifferentialMetrics]:
    """Load a completed dual eval without resampling."""
    base = iter_dir / f"eval_{tag}"
    train_path = base / "train_seed" / "differential.json"
    heldout_path = base / "heldout" / "differential.json"
    if not train_path.is_file() or not heldout_path.is_file():
        raise FileNotFoundError(
            f"Iteration journal says eval_{tag} completed, but artifacts are missing"
        )
    return (
        DifferentialMetrics.from_dict(load_json(train_path)),
        DifferentialMetrics.from_dict(load_json(heldout_path)),
    )


def _reusable_prev_post(
    *,
    run_dir: Path,
    iteration: int,
    sampler_path: str | None,
    need_high: bool,
    eval_train_sample: Path,
    eval_heldout_sample: Path,
) -> Path | None:
    """Previous iteration's ``eval_post`` dir when it measured the same thing.

    Evals sample at temperature 0 on fixed CSVs, so when the policy checkpoint
    and both samples are unchanged, re-running the pre-train eval would only
    repeat the previous post-train measurement. Returns None whenever any
    condition cannot be verified, in which case the eval runs normally.
    """
    if iteration <= 0:
        return None
    prev_post = run_dir / f"iter_{iteration - 1:03d}" / "eval_post"
    for split, sample_csv in (
        ("train_seed", eval_train_sample),
        ("heldout", eval_heldout_sample),
    ):
        split_dir = prev_post / split
        try:
            diff = DifferentialMetrics.from_dict(
                load_json(split_dir / "differential.json")
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if (diff.instant.model_path or None) != (sampler_path or None):
            return None
        if need_high:
            if diff.accuracy_gap is None:
                return None
            if (diff.high.model_path or None) != (sampler_path or None):
                return None
        pinned = split_dir / "eval_sample.csv"
        if not pinned.is_file() or not sample_csv.is_file():
            return None
        if pinned.read_text(encoding="utf-8") != sample_csv.read_text(
            encoding="utf-8"
        ):
            return None
    return prev_post


def _load_or_create_iteration_journal(
    *,
    iter_dir: Path,
    iteration: int,
    train_data_path: Path,
    config: dict[str, Any],
    state: LoopState,
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
        write_math_csv(before_path, load_math_csv(train_data_path))
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


def run_iteration(
    state: LoopState,
    progress: ProgressTracker,
    *,
    model: str,
    gen_target: int,
    seed_samples: int,
    variations_per_batch: int,
    seeds_per_batch: int,
    eval_sample_size: int,
    train_max_steps: int,
    groups_per_batch: int,
    group_size: int,
    save_every: int,
    eval_every: int,
    learning_rate: float,
    lora_rank: int,
    rng_seed: int,
    skip_generate: bool,
    skip_train: bool,
    skip_high_eval: bool,
    gen_temperature: float,
    eval_ci: EvalCIConfig | None = None,
) -> LoopState:
    """Execute one full autoresearch iteration; mutate and return state."""
    if eval_ci is None:
        eval_ci = EvalCIConfig()
    run_dir = Path(state.run_dir)
    it = state.iteration
    train_data_path = Path(state.train_data)
    iter_dir = ensure_dir(run_dir / f"iter_{it:03d}")
    iteration_config = {
        "model": model,
        "gen_target": gen_target,
        "seed_samples": seed_samples,
        "variations_per_batch": variations_per_batch,
        "seeds_per_batch": seeds_per_batch,
        "eval_sample_size": eval_sample_size,
        "train_max_steps": train_max_steps,
        "groups_per_batch": groups_per_batch,
        "group_size": group_size,
        "save_every": save_every,
        "eval_every": eval_every,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "rng_seed": rng_seed,
        "skip_generate": skip_generate,
        "skip_train": skip_train,
        "skip_high_eval": skip_high_eval,
        "gen_temperature": gen_temperature,
        "target_ci_pp": eval_ci.target_ci_pp,
        "p_value": eval_ci.p_value,
        "eval_batch_size": eval_ci.batch_size,
        "eval_max_sample_size": eval_ci.max_sample_size,
    }
    journal_path, journal = _load_or_create_iteration_journal(
        iter_dir=iter_dir,
        iteration=it,
        train_data_path=train_data_path,
        config=iteration_config,
        state=state,
    )
    progress.set_phase("iteration_start", f"Starting iteration {it}", iteration=it)
    progress.set_paths(**{f"iter_{it:03d}": iter_dir})
    print("\n" + "=" * 72, flush=True)
    print(f"ITERATION {it}  →  {iter_dir}", flush=True)
    print("=" * 72 + "\n", flush=True)

    eval_train_sample = Path(state.eval_sample)
    # Prefer explicit path; fall back for older runs.
    eval_heldout_sample = Path(
        state.eval_heldout_sample
        or (Path(state.run_dir) / "eval_heldout_sample.csv")
    )
    heldout_path = Path(
        state.heldout_test or (Path(state.run_dir) / "heldout_test.csv")
    )
    heldout_keys: set[str] = set()
    if heldout_path.is_file():
        heldout_keys = operation_keys(load_math_csv(heldout_path))

    # ------------------------------------------------------------------
    # 1) Data generation (train pool only — never held-out)
    # ------------------------------------------------------------------
    generated_path = iter_dir / "generated.csv"
    validated_path = iter_dir / "validated.csv"
    audit_path = iter_dir / "validation_audit.jsonl"
    new_examples_added = 0

    if skip_generate:
        if not _phase_done(journal, "generate"):
            progress.event("skip", "skip generate/validate", reason="--skip-generate")
            _mark_phase(journal_path, journal, "generate", skipped=True)
        if not _phase_done(journal, "validate"):
            _mark_phase(
                journal_path,
                journal,
                "validate",
                skipped=True,
                new_examples_added=0,
            )
        print("Skipping generation/validation (--skip-generate)", flush=True)
    else:
        pool = load_math_csv(train_data_path)
        # Safety: strip any held-out ops if they ever leaked into the pool.
        if heldout_keys:
            cleaned = filter_out_operations(pool, heldout_keys)
            if len(cleaned) != len(pool):
                print(
                    f"[gen] stripped {len(pool) - len(cleaned)} held-out leaks "
                    f"from train pool",
                    flush=True,
                )
                write_math_csv(train_data_path, cleaned)
                pool = cleaned

        if _phase_done(journal, "generate"):
            generated = load_math_csv(generated_path)
            print(f"[gen] reusing {len(generated)} → {generated_path}", flush=True)
        else:
            progress.set_phase("generate", "Generating data variations (high reasoning)")
            with_sol = [ex for ex in pool if ex.solution]
            seeds = sample_examples(
                with_sol or pool,
                min(seed_samples, len(with_sol or pool)),
                seed=rng_seed + it * 1009,
            )
            write_math_csv(iter_dir / "generation_seeds.csv", seeds)
            print(
                f"[gen] {len(seeds)} seed exemplars from train pool "
                f"({len(pool)} rows; held-out banned={len(heldout_keys)})",
                flush=True,
            )
            generated = generate_variations(
                seeds,
                target=gen_target,
                variations_per_seed_batch=variations_per_batch,
                seed_batch_size=seeds_per_batch,
                model=model,
                model_path=None,
                temperature=gen_temperature,
                rng_seed=rng_seed + it,
            )
            # Never promote held-out operations (or copies) into training.
            generated = filter_out_operations(generated, heldout_keys)
            write_math_csv(generated_path, generated)
            progress.event(
                "generate_done",
                f"generated {len(generated)} examples",
                n=len(generated),
                path=str(generated_path),
            )
            _mark_phase(
                journal_path,
                journal,
                "generate",
                generated=len(generated),
            )
            print(f"[gen] wrote {len(generated)} → {generated_path}", flush=True)

        # ------------------------------------------------------------------
        # 2) Data validation
        # ------------------------------------------------------------------
        if _phase_done(journal, "validate"):
            validated = load_math_csv(validated_path)
            new_examples_added = int(
                _phase_data(journal, "validate").get("new_examples_added", 0)
            )
            resumed_pool = load_math_csv(train_data_path)
            assert_disjoint_train_heldout(
                resumed_pool,
                load_math_csv(heldout_path) if heldout_path.is_file() else [],
            )
            progress.update(train_pool_size=len(resumed_pool))
            print(
                f"[val] reusing {len(validated)} validated examples "
                f"(+{new_examples_added} added) → {validated_path}",
                flush=True,
            )
        else:
            progress.set_phase(
                "validate", "Validating with programmatic + high reasoning"
            )
            # A retry before the durable phase marker should not duplicate audit rows.
            audit_path.unlink(missing_ok=True)
            validated = validate_examples(
                generated,
                model=model,
                model_path=None,
                require_programmatic=True,
                audit_path=audit_path,
            )
            validated = filter_out_operations(validated, heldout_keys)
            write_math_csv(validated_path, validated)

            pool = load_math_csv(train_data_path)
            merged = merge_unique_examples(pool, validated)
            heldout = load_math_csv(heldout_path) if heldout_path.is_file() else []
            assert_disjoint_train_heldout(merged, heldout)
            write_math_csv(train_data_path, merged)
            write_math_csv(iter_dir / "train_data_snapshot.csv", merged)
            progress.snapshot_train_pool(
                len(merged), source="post_validate", path=train_data_path
            )
            before_keys = operation_keys(
                load_math_csv(iter_dir / "train_data_before.csv")
            )
            new_examples_added = len(operation_keys(validated) - before_keys)
            progress.event(
                "validate_done",
                f"kept {len(validated)}/{len(generated)}",
                kept=len(validated),
                total=len(generated),
                added=new_examples_added,
                path=str(validated_path),
                audit=str(audit_path),
            )
            _mark_phase(
                journal_path,
                journal,
                "validate",
                kept=len(validated),
                generated=len(generated),
                new_examples_added=new_examples_added,
            )
            print(
                f"[val] train pool {len(pool)} → {len(merged)} "
                f"(+{len(merged) - len(pool)} now; "
                f"{new_examples_added} vs iteration start)",
                flush=True,
            )

    if not eval_train_sample.is_file():
        raise FileNotFoundError(
            f"Missing train-seed eval sample: {eval_train_sample}. "
            "Re-init the run so a held-out split is carved from seed data."
        )
    if not eval_heldout_sample.is_file():
        raise FileNotFoundError(
            f"Missing held-out eval sample: {eval_heldout_sample}. "
            "Re-init the run so a held-out split is carved from seed data."
        )

    # ------------------------------------------------------------------
    # 3) Pre-train evals — train-seed + held-out
    # ------------------------------------------------------------------
    if _phase_done(journal, "eval_pre"):
        pre_train, pre_heldout = _load_dual_evals(tag="pre", iter_dir=iter_dir)
        print("[eval/pre] reusing completed eval artifacts", flush=True)
    else:
        prev_post = _reusable_prev_post(
            run_dir=run_dir,
            iteration=it,
            sampler_path=state.last_policy_sampler_path,
            need_high=not skip_high_eval,
            eval_train_sample=eval_train_sample,
            eval_heldout_sample=eval_heldout_sample,
        )
        if prev_post is not None:
            shutil.copytree(prev_post, iter_dir / "eval_pre", dirs_exist_ok=True)
            pre_train, pre_heldout = _load_dual_evals(tag="pre", iter_dir=iter_dir)
            progress.set_phase(
                "eval_pre",
                "Pre-train evals carried forward from previous post-train evals",
            )
            for split, diff in (("train_seed", pre_train), ("heldout", pre_heldout)):
                split_dir = iter_dir / "eval_pre" / split
                progress.record_eval(
                    f"pre_{split}",
                    instant_accuracy=diff.instant.accuracy,
                    high_accuracy=None if skip_high_eval else diff.high.accuracy,
                    accuracy_gap=None if skip_high_eval else diff.accuracy_gap,
                    out_dir=split_dir,
                    instant_answers=split_dir / "answers_instant.csv",
                    high_answers=(
                        None if skip_high_eval else split_dir / "answers_high.csv"
                    ),
                    differential_path=split_dir / "differential.json",
                    split=split,
                )
            print(
                f"[eval/pre] carried forward {prev_post} "
                "(same checkpoint + samples; temperature-0 evals repeat)",
                flush=True,
            )
            _mark_phase(
                journal_path, journal, "eval_pre", reused_from=str(prev_post)
            )
        else:
            progress.set_phase(
                "eval_pre",
                "Pre-train evals on train-seed sample AND held-out test",
            )
            # Previous post metrics for instant-vs-previous CI (if any)
            prev_post_train = None
            prev_post_heldout = None
            if state.metrics_history:
                last = state.metrics_history[-1]
                try:
                    prev_post_train = DifferentialMetrics.from_dict(
                        last.get("post_train")
                        or (last.get("train_seed") or {}).get("post")
                        or last.get("post")
                    )
                except (TypeError, KeyError, ValueError):
                    prev_post_train = None
                try:
                    prev_post_heldout = DifferentialMetrics.from_dict(
                        last.get("post_heldout")
                        or (last.get("heldout") or {}).get("post")
                    )
                except (TypeError, KeyError, ValueError):
                    prev_post_heldout = None
            pre_train, pre_heldout = _run_dual_evals(
                tag="pre",
                iter_dir=iter_dir,
                eval_train_sample=eval_train_sample,
                eval_heldout_sample=eval_heldout_sample,
                model=model,
                sampler_path=state.last_policy_sampler_path,
                skip_high=skip_high_eval,
                progress=progress,
                eval_ci=eval_ci,
                prev_train_instant=prev_post_train,
                prev_heldout_instant=prev_post_heldout,
            )
            _mark_phase(journal_path, journal, "eval_pre")

    # ------------------------------------------------------------------
    # 4) Train (instant policy from last checkpoint, high judge)
    # ------------------------------------------------------------------
    train_meta: dict | None = None
    train_log = ensure_dir(iter_dir / "train")
    if _phase_done(journal, "train"):
        if skip_train:
            print("Skipping train (--skip-train; already journaled)", flush=True)
        else:
            train_meta_path = train_log / "train_step_meta.json"
            if not train_meta_path.is_file():
                raise FileNotFoundError(
                    f"Iteration journal says train completed, but metadata is missing: "
                    f"{train_meta_path}"
                )
            train_meta = load_json(train_meta_path)
            print(f"[train] reusing completed train step → {train_log}", flush=True)
    else:
        if skip_train:
            print("Skipping train (--skip-train)", flush=True)
            progress.event("skip", "skip train", reason="--skip-train")
            _mark_phase(journal_path, journal, "train", skipped=True)
        else:
            # Freeze the sources recorded when this iteration first started.
            policy_before = journal.get("policy_before", {})
            policy_ckpt = policy_before.get("state_path")
            judge_ckpt = (
                policy_before.get("judge_model_path")
                or policy_before.get("sampler_path")
            )
            progress.set_phase(
                "train",
                "Training instant policy with high-reasoning judge "
                f"(policy_from={policy_ckpt or 'base'!r}, "
                f"judge_from={judge_ckpt or 'base'!r})",
            )
            progress.event(
                "auto_switch",
                "selected train sources",
                policy_source="checkpoint" if policy_ckpt else "base",
                judge_source="checkpoint" if judge_ckpt else "base",
                policy_checkpoint=policy_ckpt,
                judge_model_path=judge_ckpt,
            )
            artifact_dir = ensure_dir(iter_dir / "train_artifacts")
            train_meta = run_train_step(
                data_path=train_data_path,
                log_path=train_log,
                load_checkpoint=policy_ckpt,
                judge_model_path=judge_ckpt,
                model=model,
                max_steps=train_max_steps,
                groups_per_batch=groups_per_batch,
                group_size=group_size,
                save_every=save_every,
                eval_every=eval_every,
                learning_rate=learning_rate,
                lora_rank=lora_rank,
                seed=rng_seed,
                behavior_if_log_dir_exists="resume",
                artifact_dir=artifact_dir,
            )
            # Index checkpoints + metrics for dashboard (run-level + per-iter).
            progress.index_train_log(train_log, iteration=it)
            progress.event(
                "train_done",
                "train finished",
                state_path=train_meta.get("state_path"),
                sampler_path=train_meta.get("sampler_path"),
                judge_model_path=train_meta.get("sampler_path"),
                policy_source=train_meta.get("policy_source"),
                judge_source=train_meta.get("judge_source"),
                log_path=str(train_log),
                artifacts=str(artifact_dir),
            )
            _mark_phase(
                journal_path,
                journal,
                "train",
                state_path=train_meta.get("state_path"),
                sampler_path=train_meta.get("sampler_path"),
            )

    if train_meta is not None:
        if str(train_log) not in state.policy_log_dirs:
            state.policy_log_dirs.append(str(train_log))
        if train_meta.get("state_path"):
            state.last_policy_state_path = train_meta["state_path"]
        if train_meta.get("sampler_path"):
            state.last_policy_sampler_path = train_meta["sampler_path"]
            state.last_judge_model_path = train_meta["sampler_path"]
        progress.update(
            last_policy_state_path=state.last_policy_state_path,
            last_policy_sampler_path=state.last_policy_sampler_path,
            last_judge_model_path=state.last_judge_model_path,
        )

    # ------------------------------------------------------------------
    # 5) Post-train evals — train-seed + held-out
    # ------------------------------------------------------------------
    if _phase_done(journal, "eval_post"):
        post_train, post_heldout = _load_dual_evals(tag="post", iter_dir=iter_dir)
        print("[eval/post] reusing completed eval artifacts", flush=True)
    else:
        progress.set_phase(
            "eval_post",
            "Post-train evals on train-seed sample AND held-out test",
        )
        # Post vs pre (this iter) for instant-vs-previous CI after training
        post_train, post_heldout = _run_dual_evals(
            tag="post",
            iter_dir=iter_dir,
            eval_train_sample=eval_train_sample,
            eval_heldout_sample=eval_heldout_sample,
            model=model,
            sampler_path=state.last_policy_sampler_path,
            skip_high=skip_high_eval,
            progress=progress,
            eval_ci=eval_ci,
            prev_train_instant=pre_train,
            prev_heldout_instant=pre_heldout,
        )
        _mark_phase(journal_path, journal, "eval_post")

    train_instant_delta = post_train.instant.accuracy - pre_train.instant.accuracy
    heldout_instant_delta = (
        post_heldout.instant.accuracy - pre_heldout.instant.accuracy
    )
    if skip_high_eval:
        train_gap_delta = None
        heldout_gap_delta = None
    else:
        if pre_train.accuracy_gap is None or post_train.accuracy_gap is None:
            raise ValueError("Missing train-seed high-eval accuracy gap")
        if pre_heldout.accuracy_gap is None or post_heldout.accuracy_gap is None:
            raise ValueError("Missing held-out high-eval accuracy gap")
        train_gap_delta = post_train.accuracy_gap - pre_train.accuracy_gap
        heldout_gap_delta = post_heldout.accuracy_gap - pre_heldout.accuracy_gap
    overfit_gap = post_train.instant.accuracy - post_heldout.instant.accuracy
    progress.update(
        last_instant_delta=train_instant_delta,
        last_heldout_instant_delta=heldout_instant_delta,
        last_overfit_gap=overfit_gap,
    )

    # Chart axis: gen 0 baseline, gen 1 imported parent (if any), then loop iters.
    run_cfg = load_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else {}
    imported_parent = bool(
        run_cfg.get("imported_from")
        or run_cfg.get("promoted_state_path")
        or run_cfg.get("imported_checkpoint")
        or (progress.status.extra or {}).get("promoted")
    )
    record = {
        "iteration": it,
        "generation": it + (2 if imported_parent else 1),
        "new_examples": new_examples_added,
        "train_pool_size": progress.status.train_pool_size,
        "heldout_size": progress.status.heldout_size
        or (len(heldout_keys) if heldout_keys else None),
        "train": train_meta,
        # Nested dual-eval metrics
        "train_seed": {
            "pre": pre_train.to_dict(),
            "post": post_train.to_dict(),
            "delta": {
                "instant_accuracy": train_instant_delta,
                "accuracy_gap": train_gap_delta,
            },
        },
        "heldout": {
            "pre": pre_heldout.to_dict(),
            "post": post_heldout.to_dict(),
            "delta": {
                "instant_accuracy": heldout_instant_delta,
                "accuracy_gap": heldout_gap_delta,
            },
        },
        # Flat keys for early-stopping helper + back-compat
        "pre": pre_train.to_dict(),
        "post": post_train.to_dict(),
        "pre_train": pre_train.to_dict(),
        "post_train": post_train.to_dict(),
        "pre_heldout": pre_heldout.to_dict(),
        "post_heldout": post_heldout.to_dict(),
        "delta": {
            "train_seed_instant": train_instant_delta,
            "heldout_instant": heldout_instant_delta,
            "overfit_gap": overfit_gap,
            "train_seed_accuracy_gap": train_gap_delta,
            "heldout_accuracy_gap": heldout_gap_delta,
        },
        "paths": {
            "iter_dir": str(iter_dir),
            "eval_pre": str(iter_dir / "eval_pre"),
            "eval_post": str(iter_dir / "eval_post"),
            "train_log": str(train_log),
            "generated": str(generated_path),
            "validated": str(validated_path),
            "heldout_test": str(heldout_path),
            "metrics": str(iter_dir / "metrics.json"),
        },
        "eval_ci": {
            "target_ci_pp": eval_ci.target_ci_pp,
            "p_value": eval_ci.p_value,
            "batch_size": eval_ci.batch_size,
            "max_sample_size": eval_ci.max_sample_size,
        },
    }
    # Attach sequential CI payloads onto post differentials for dashboard series
    post_ci_path = iter_dir / "eval_post" / "sequential_ci.json"
    if post_ci_path.is_file():
        try:
            seq = load_json(post_ci_path)
            if isinstance(seq, dict):
                if isinstance(record.get("heldout"), dict) and isinstance(
                    record["heldout"].get("post"), dict
                ):
                    h_ci = (seq.get("heldout") or {}).get("differential", {})
                    # Prefer full sequential result ci block
                    record["heldout"]["post"]["ci"] = (seq.get("heldout") or {}).get(
                        "comparisons"
                    ) and {
                        "comparisons": (seq.get("heldout") or {}).get("comparisons"),
                        "instant": (seq.get("heldout") or {}).get("instant_ci"),
                        "high": (seq.get("heldout") or {}).get("high_ci"),
                        "n_used": (seq.get("heldout") or {}).get("n_used"),
                        "target_met": (seq.get("heldout") or {}).get("target_met"),
                        "exhausted": (seq.get("heldout") or {}).get("exhausted"),
                        "p_value": eval_ci.p_value,
                        "target_ci_pp": eval_ci.target_ci_pp,
                        "target_half_width": eval_ci.target_half_width,
                    }
                if isinstance(record.get("train_seed"), dict) and isinstance(
                    record["train_seed"].get("post"), dict
                ):
                    record["train_seed"]["post"]["ci"] = {
                        "comparisons": (seq.get("train_seed") or {}).get("comparisons"),
                        "instant": (seq.get("train_seed") or {}).get("instant_ci"),
                        "high": (seq.get("train_seed") or {}).get("high_ci"),
                        "n_used": (seq.get("train_seed") or {}).get("n_used"),
                        "target_met": (seq.get("train_seed") or {}).get("target_met"),
                        "exhausted": (seq.get("train_seed") or {}).get("exhausted"),
                        "p_value": eval_ci.p_value,
                        "target_ci_pp": eval_ci.target_ci_pp,
                        "target_half_width": eval_ci.target_half_width,
                    }
                record["post_heldout"] = record["heldout"]["post"]
                record["post_train"] = record["train_seed"]["post"]
                record["post"] = record["train_seed"]["post"]
        except (OSError, TypeError, ValueError):
            pass
    save_json(iter_dir / "metrics.json", record)
    history_path = run_dir / "history.jsonl"
    _upsert_history_file(history_path, record)
    _upsert_state_history(state.metrics_history, record)
    _mark_phase(journal_path, journal, "metrics")
    progress.event(
        "iteration_done",
        f"heldout Δ={heldout_instant_delta:+.4f} train-seed Δ={train_instant_delta:+.4f}",
        train_instant_delta=train_instant_delta,
        heldout_instant_delta=heldout_instant_delta,
        overfit_gap=overfit_gap,
        pre_train_instant=pre_train.instant.accuracy,
        post_train_instant=post_train.instant.accuracy,
        pre_heldout_instant=pre_heldout.instant.accuracy,
        post_heldout_instant=post_heldout.instant.accuracy,
    )

    print(
        f"\n[iter {it}] TRAIN-SEED instant "
        f"{pre_train.instant.accuracy:.4f} → {post_train.instant.accuracy:.4f} "
        f"(Δ {train_instant_delta:+.4f})\n"
        f"[iter {it}] HELDOUT    instant "
        f"{pre_heldout.instant.accuracy:.4f} → {post_heldout.instant.accuracy:.4f} "
        f"(Δ {heldout_instant_delta:+.4f}) | overfit_gap={overfit_gap:+.4f}",
        flush=True,
    )

    state.iteration = it + 1
    _mark_phase(
        journal_path,
        journal,
        "complete",
        committed_iteration=state.iteration,
    )
    save_state(run_dir / "state.json", state)
    return state


def build_parser(
    *,
    suppress_defaults: bool = False,
    loop_cfg: LoopConfig | None = None,
) -> argparse.ArgumentParser:
    """Build the CLI parser.

    With ``suppress_defaults=True`` every default becomes ``argparse.SUPPRESS``
    so a second parse reveals exactly which options the user typed (used for
    resume-override detection). Defaults come from ``loop_config.toml`` when present.
    """
    cfg = loop_cfg or load_loop_config()
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=resolve_config_path(),
        help=f"Path to loop_config.toml (default: {resolve_config_path()})",
    )
    p.add_argument(
        "--seed-data",
        type=Path,
        default=DEFAULT_SEED_DATA,
        help=f"Initial math CSV (default: {DEFAULT_SEED_DATA})",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Parent directory for runs (default: {DEFAULT_RUNS_ROOT})",
    )
    p.add_argument("--run-name", default=None, help="Optional run folder name")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an existing run directory (reads state.json)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-iters", type=int, default=cfg.stop.max_iters)
    p.add_argument(
        "--marginal-delta",
        type=float,
        default=cfg.stop.marginal_delta,
        help=(
            "Stop when instant accuracy gain is below this fraction "
            f"(default from config: {cfg.stop.marginal_delta})"
        ),
    )
    p.add_argument(
        "--marginal-streak",
        type=int,
        default=cfg.stop.marginal_streak,
        help=(
            "Consecutive marginal iterations required to stop "
            f"(default from config: {cfg.stop.marginal_streak})"
        ),
    )
    p.add_argument(
        "--noise-z",
        type=float,
        default=cfg.stop.noise_z,
        help=(
            "An instant-accuracy gain only counts as progress when it also "
            "exceeds noise_z × the binomial standard error of the pre/post "
            "comparison. 0 disables the noise band "
            f"(default from config: {cfg.stop.noise_z})"
        ),
    )
    p.add_argument(
        "--gen-target",
        "--step-size",
        type=int,
        default=cfg.generation.step_size,
        dest="gen_target",
        help=(
            "New data rows to aim for each iteration (generation step size). "
            f"Default from config generation.step_size: {cfg.generation.step_size}"
        ),
    )
    p.add_argument("--seed-samples", type=int, default=cfg.generation.seed_samples)
    p.add_argument(
        "--variations-per-batch",
        type=int,
        default=cfg.generation.variations_per_batch,
    )
    p.add_argument("--seeds-per-batch", type=int, default=cfg.generation.seeds_per_batch)
    p.add_argument("--gen-temperature", type=float, default=cfg.generation.temperature)
    p.add_argument(
        "--eval-sample-size",
        type=int,
        default=max(cfg.eval.pool_size_heldout, cfg.eval.pool_size_train_seed),
        help=(
            "Max ops reserved for each eval pool at run start "
            f"(default max of config pools: "
            f"{max(cfg.eval.pool_size_heldout, cfg.eval.pool_size_train_seed)})"
        ),
    )
    p.add_argument(
        "--target-ci-pp",
        type=float,
        default=cfg.eval.target_ci_pp,
        help=(
            "Target CI half-width in percentage points for adaptive eval "
            f"(default from config: {cfg.eval.target_ci_pp} @ p<{cfg.eval.p_value})"
        ),
    )
    p.add_argument(
        "--p-value",
        type=float,
        default=cfg.eval.p_value,
        help=f"α for CI / significance (default from config: {cfg.eval.p_value})",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=cfg.eval.batch_size,
        help=f"Adaptive eval growth step (default: {cfg.eval.batch_size})",
    )
    p.add_argument(
        "--eval-max-sample-size",
        type=int,
        default=cfg.eval.max_sample_size,
        help=f"Hard cap on adaptive eval n (default: {cfg.eval.max_sample_size})",
    )
    p.add_argument(
        "--heldout-fraction",
        type=float,
        default=cfg.data.heldout_fraction,
        help=(
            "Fraction of seed carved into a never-train held-out test set "
            f"(default from config: {cfg.data.heldout_fraction}). Progress evals "
            "run on both a train-seed sample and this held-out sample."
        ),
    )
    p.add_argument("--train-max-steps", type=int, default=cfg.train.max_steps)
    p.add_argument(
        "--groups-per-batch",
        type=int,
        default=cfg.train.groups_per_batch,
    )
    p.add_argument("--group-size", type=int, default=cfg.train.group_size)
    p.add_argument("--save-every", type=int, default=cfg.train.save_every)
    p.add_argument("--eval-every", type=int, default=cfg.train.eval_every)
    p.add_argument("--learning-rate", type=float, default=cfg.train.learning_rate)
    p.add_argument("--lora-rank", type=int, default=cfg.train.lora_rank)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "When resuming a run that early-stopped, clear the stop decision "
            "and continue iterating (otherwise resume is a no-op)"
        ),
    )
    p.add_argument(
        "--skip-generate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip data generation/validation (train/eval only)",
    )
    p.add_argument(
        "--skip-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip Tinker training (eval-only loop for debugging)",
    )
    p.add_argument(
        "--skip-high-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only score instant on the eval sample (cheaper)",
    )
    if suppress_defaults:
        for action in p._actions:
            if action.dest != "help":
                action.default = argparse.SUPPRESS
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # First pass: discover --config if present, then rebuild defaults from it.
    raw = list(argv) if argv is not None else sys.argv[1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(raw)
    cfg = load_loop_config(pre_args.config) if pre_args.config else load_loop_config()
    return build_parser(loop_cfg=cfg).parse_args(raw)


def eval_ci_from_args(args: argparse.Namespace, base: LoopConfig | None = None) -> EvalCIConfig:
    """Merge CLI eval CI knobs onto the loaded config section."""
    cfg = base or load_loop_config(getattr(args, "config", None))
    e = cfg.eval
    return EvalCIConfig(
        target_ci_pp=float(args.target_ci_pp),
        p_value=float(args.p_value),
        batch_size=int(args.eval_batch_size),
        min_sample_size=min(int(args.eval_batch_size), int(args.eval_max_sample_size)),
        max_sample_size=int(args.eval_max_sample_size),
        compare_instant_vs_high=e.compare_instant_vs_high and not args.skip_high_eval,
        compare_instant_vs_previous=e.compare_instant_vs_previous,
        require_heldout_ci=e.require_heldout_ci,
        require_train_seed_ci=e.require_train_seed_ci,
        pool_size_heldout=max(e.pool_size_heldout, int(args.eval_sample_size)),
        pool_size_train_seed=max(e.pool_size_train_seed, int(args.eval_sample_size)),
    )


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "max_iters": args.max_iters,
        "marginal_streak": args.marginal_streak,
        "gen_target": args.gen_target,
        "seed_samples": args.seed_samples,
        "variations_per_batch": args.variations_per_batch,
        "seeds_per_batch": args.seeds_per_batch,
        "eval_sample_size": args.eval_sample_size,
        "eval_batch_size": args.eval_batch_size,
        "eval_max_sample_size": args.eval_max_sample_size,
        "target_ci_pp": args.target_ci_pp,
        "train_max_steps": args.train_max_steps,
        "groups_per_batch": args.groups_per_batch,
        "group_size": args.group_size,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Expected positive configuration values, got {invalid}")
    if args.marginal_delta < 0:
        raise ValueError("marginal_delta must be >= 0")
    if args.noise_z < 0:
        raise ValueError("noise_z must be >= 0")
    if not 0.0 < args.p_value < 1.0:
        raise ValueError("p_value must be in (0, 1)")
    if not args.skip_train and args.save_every > args.train_max_steps:
        raise ValueError(
            f"save_every ({args.save_every}) must be <= train_max_steps "
            f"({args.train_max_steps}), otherwise training saves no checkpoint"
        )
    if args.gen_temperature < 0:
        raise ValueError("gen_temperature must be >= 0")
    if not 0.0 < args.heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    args, resume_record = resolve_resume_config(args, argv=raw_argv)
    validate_args(args)
    eval_ci = eval_ci_from_args(args)
    print(
        f"Config: step_size/gen_target={args.gen_target} | "
        f"eval CI ±{args.target_ci_pp}pp @ p<{args.p_value} | "
        f"batch={args.eval_batch_size} max_n={args.eval_max_sample_size} | "
        f"source={getattr(args, 'config', None)}",
        flush=True,
    )
    for warning in check_model_consistency(args.model):
        print(f"WARNING: {warning}", flush=True)
    run_dir, state, progress = init_run(
        runs_root=args.runs_root,
        run_name=args.run_name,
        seed_data=args.seed_data,
        resume_dir=args.resume,
        heldout_fraction=args.heldout_fraction,
        eval_sample_size=args.eval_sample_size,
        rng_seed=args.rng_seed,
    )
    state_path = run_dir / "state.json"

    if resume_record is None:
        # Immutable original configuration. Resume invocations are audited below.
        save_json(run_dir / "config.json", _serialize_args(args))
    else:
        resume_record["iteration"] = state.iteration
        resume_history = run_dir / "resume_history.jsonl"
        append_jsonl(resume_history, resume_record)
        progress.set_paths(resume_history=resume_history)

    if state.stopped and not args.force:
        print(
            f"\nRun is already stopped: {state.stop_reason}\n"
            "Resume with --force to clear the stop decision and continue.",
            flush=True,
        )
        return
    if state.stopped and args.force:
        print(
            f"\n--force: clearing early stop ({state.stop_reason})",
            flush=True,
        )
        state.stopped = False
        state.stop_reason = None
        save_state(state_path, state)
        progress.update(stopped=False, stop_reason=None)
        progress.event("force_resume", "cleared early stop via --force")

    # Derive the streak from durable history instead of trusting a possibly stale
    # counter written just before an interruption. Skipped under --force so a
    # forced run gets to attempt at least one new iteration.
    state.marginal_streak = marginal_improvement_streak(
        state.metrics_history,
        min_delta=args.marginal_delta,
        noise_z=args.noise_z,
    )
    if (
        not args.force
        and not state.stopped
        and state.marginal_streak >= args.marginal_streak
    ):
        reason = (
            f"held-out instant accuracy gains below progress threshold "
            f"(marginal_delta={args.marginal_delta}, noise_z={args.noise_z}) "
            f"for {args.marginal_streak} consecutive iterations"
        )
        state.stopped = True
        state.stop_reason = reason
        save_state(state_path, state)
        progress.mark_stopped(reason)

    start_iter = state.iteration
    end_iter = start_iter + args.max_iters
    if args.resume is not None and not state.stopped and start_iter > 0:
        print(
            f"Resume extends the run by {args.max_iters} iteration(s): "
            f"{start_iter} → {end_iter} (use --max-iters to change)",
            flush=True,
        )
    while state.iteration < end_iter and not state.stopped:
        state = run_iteration(
            state,
            progress,
            model=args.model,
            gen_target=args.gen_target,
            seed_samples=args.seed_samples,
            variations_per_batch=args.variations_per_batch,
            seeds_per_batch=args.seeds_per_batch,
            eval_sample_size=args.eval_sample_size,
            train_max_steps=args.train_max_steps,
            groups_per_batch=args.groups_per_batch,
            group_size=args.group_size,
            save_every=args.save_every,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            rng_seed=args.rng_seed,
            skip_generate=args.skip_generate,
            skip_train=args.skip_train,
            skip_high_eval=args.skip_high_eval,
            gen_temperature=args.gen_temperature,
            eval_ci=eval_ci,
        )
        state.marginal_streak = marginal_improvement_streak(
            state.metrics_history,
            min_delta=args.marginal_delta,
            noise_z=args.noise_z,
        )
        if state.marginal_streak:
            progress.event(
                "marginal",
                f"streak {state.marginal_streak}/{args.marginal_streak}",
                streak=state.marginal_streak,
            )
            print(
                f"Marginal improvement streak: "
                f"{state.marginal_streak}/{args.marginal_streak}",
                flush=True,
            )
        if state.marginal_streak >= args.marginal_streak:
            reason = (
                f"held-out instant accuracy gains below progress threshold "
                f"(marginal_delta={args.marginal_delta}, noise_z={args.noise_z}) "
                f"for {args.marginal_streak} consecutive iterations"
            )
            state.stopped = True
            state.stop_reason = reason
            progress.mark_stopped(reason)
            print(f"\nStopping: {reason}", flush=True)
            save_state(state_path, state)
            break
        save_state(state_path, state)

    if not state.stopped:
        progress.set_phase(
            "completed",
            f"Finished {state.iteration - start_iter} iteration(s)",
            iteration=state.iteration,
        )
        progress.update(stopped=False, stop_reason=None)

    print(
        f"\nDone. iterations completed={state.iteration - start_iter} "
        f"total_iteration_index={state.iteration} run_dir={run_dir}",
        flush=True,
    )
    print(f"Dashboard poll targets: {run_dir}/status.json  {run_dir}/events.jsonl", flush=True)
    if state.metrics_history:
        last = state.metrics_history[-1]
        post = last.get("post", {})
        instant = post.get("instant", {})
        print(
            f"Latest instant accuracy: {instant.get('accuracy')} "
            f"({instant.get('correct')}/{instant.get('total')})",
            flush=True,
        )


if __name__ == "__main__":
    main()
