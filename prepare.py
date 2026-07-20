"""Frozen harness for the autoresearch loop: data, evals, train, run setup.

This file is the fixed contract the loop builds on (Karpathy-autoresearch
style ``prepare.py``): run initialization with a never-train held-out split,
adaptive-CI dual evals with carry-forward, the RL train step, and the
crash-safe iteration bookkeeping. Iterate on ``loop.py`` (and
``loop_config.toml``), not on this file.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from core.defaults import (
    DEFAULT_EVAL_SAMPLE_SIZE,
    DEFAULT_HELDOUT_FRACTION,
    MIN_HELDOUT_EVAL_SAMPLE,
)
from core.history import _upsert_history_file, _upsert_state_history
from core.io import ensure_dir, load_json, save_json, utc_now_tag
from core.journal import (
    _load_or_create_iteration_journal,
    _mark_phase,
    _phase_data,
    _phase_done,
)
from core.loop_config import EvalCIConfig
from core.metrics import DifferentialMetrics
from core.progress import ProgressTracker
from core.runstate import LoopState, _acquire_run_lock, load_state, save_state
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
from mathtask.math_integration import write_integration_manifest
from mathtask.sequential_eval import PrevInstantRef, run_sequential_differential
from mathtask.train_step import run_train_step
from mathtask.validate_data import validate_examples

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


@dataclass
class IterationContext:
    """Everything one iteration's phases share (paths, journal, config)."""

    state: LoopState
    progress: ProgressTracker
    cfg: dict[str, Any]
    eval_ci: EvalCIConfig
    run_dir: Path
    it: int
    iter_dir: Path
    train_data_path: Path
    journal_path: Path
    journal: dict[str, Any]
    eval_train_sample: Path
    eval_heldout_sample: Path
    heldout_path: Path
    heldout_keys: set[str] = field(default_factory=set)
    generated_path: Path | None = None
    validated_path: Path | None = None
    audit_path: Path | None = None
    new_examples_added: int = 0
    train_meta: dict | None = None
    train_log: Path | None = None


def begin_iteration(
    state: LoopState,
    progress: ProgressTracker,
    cfg: dict[str, Any],
    eval_ci: EvalCIConfig,
) -> IterationContext:
    """Open (or re-open) this iteration: journal, dirs, samples, leak guards."""
    run_dir = Path(state.run_dir)
    it = state.iteration
    train_data_path = Path(state.train_data)
    iter_dir = ensure_dir(run_dir / f"iter_{it:03d}")
    iteration_config = {
        **cfg,
        "target_ci_pp": eval_ci.target_ci_pp,
        "p_value": eval_ci.p_value,
        "eval_batch_size": eval_ci.batch_size,
        "eval_max_sample_size": eval_ci.max_sample_size,
    }
    journal_path, journal = _load_or_create_iteration_journal(
        iter_dir=iter_dir,
        iteration=it,
        config=iteration_config,
        state=state,
        ensure_before=lambda before_path: write_math_csv(
            before_path, load_math_csv(train_data_path)
        ),
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

    return IterationContext(
        state=state,
        progress=progress,
        cfg=cfg,
        eval_ci=eval_ci,
        run_dir=run_dir,
        it=it,
        iter_dir=iter_dir,
        train_data_path=train_data_path,
        journal_path=journal_path,
        journal=journal,
        eval_train_sample=eval_train_sample,
        eval_heldout_sample=eval_heldout_sample,
        heldout_path=heldout_path,
        heldout_keys=heldout_keys,
        generated_path=iter_dir / "generated.csv",
        validated_path=iter_dir / "validated.csv",
        audit_path=iter_dir / "validation_audit.jsonl",
    )


def generate_and_validate(ctx: IterationContext) -> None:
    """Phases 1-2: grow the train pool with validated generations (or skip)."""
    progress, journal, journal_path = ctx.progress, ctx.journal, ctx.journal_path
    iter_dir, it = ctx.iter_dir, ctx.it
    train_data_path = ctx.train_data_path
    heldout_keys, heldout_path = ctx.heldout_keys, ctx.heldout_path
    generated_path = ctx.generated_path
    validated_path = ctx.validated_path
    audit_path = ctx.audit_path
    cfg = ctx.cfg
    model = cfg["model"]
    gen_target = cfg["gen_target"]
    seed_samples = cfg["seed_samples"]
    variations_per_batch = cfg["variations_per_batch"]
    seeds_per_batch = cfg["seeds_per_batch"]
    gen_temperature = cfg["gen_temperature"]
    rng_seed = cfg["rng_seed"]
    skip_generate = cfg["skip_generate"]
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

    ctx.new_examples_added = new_examples_added


def evaluate_pre(ctx: IterationContext) -> tuple[DifferentialMetrics, DifferentialMetrics]:
    """Phase 3: pre-train dual evals (carried forward when nothing changed)."""
    state, progress = ctx.state, ctx.progress
    journal, journal_path = ctx.journal, ctx.journal_path
    run_dir, it, iter_dir = ctx.run_dir, ctx.it, ctx.iter_dir
    eval_train_sample = ctx.eval_train_sample
    eval_heldout_sample = ctx.eval_heldout_sample
    model = ctx.cfg["model"]
    skip_high_eval = ctx.cfg["skip_high_eval"]
    eval_ci = ctx.eval_ci

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

    return pre_train, pre_heldout


def train_policy(ctx: IterationContext) -> None:
    """Phase 4: RL train step from the last checkpoint; advance policy pointers."""
    state, progress = ctx.state, ctx.progress
    journal, journal_path = ctx.journal, ctx.journal_path
    iter_dir, it = ctx.iter_dir, ctx.it
    train_data_path = ctx.train_data_path
    cfg = ctx.cfg
    model = cfg["model"]
    train_max_steps = cfg["train_max_steps"]
    groups_per_batch = cfg["groups_per_batch"]
    group_size = cfg["group_size"]
    save_every = cfg["save_every"]
    eval_every = cfg["eval_every"]
    learning_rate = cfg["learning_rate"]
    lora_rank = cfg["lora_rank"]
    rng_seed = cfg["rng_seed"]
    skip_train = cfg["skip_train"]

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

    ctx.train_meta = train_meta
    ctx.train_log = train_log


def evaluate_post(
    ctx: IterationContext,
    pre_train: DifferentialMetrics,
    pre_heldout: DifferentialMetrics,
) -> tuple[DifferentialMetrics, DifferentialMetrics]:
    """Phase 5: post-train dual evals with the fresh checkpoint."""
    state, progress = ctx.state, ctx.progress
    journal, journal_path = ctx.journal, ctx.journal_path
    iter_dir = ctx.iter_dir
    eval_train_sample = ctx.eval_train_sample
    eval_heldout_sample = ctx.eval_heldout_sample
    model = ctx.cfg["model"]
    skip_high_eval = ctx.cfg["skip_high_eval"]
    eval_ci = ctx.eval_ci

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

    return post_train, post_heldout


def finish_iteration(
    ctx: IterationContext,
    pre_train: DifferentialMetrics,
    pre_heldout: DifferentialMetrics,
    post_train: DifferentialMetrics,
    post_heldout: DifferentialMetrics,
) -> LoopState:
    """Phase 6: deltas, durable metrics/history, commit the iteration."""
    state, progress = ctx.state, ctx.progress
    journal, journal_path = ctx.journal, ctx.journal_path
    run_dir, it, iter_dir = ctx.run_dir, ctx.it, ctx.iter_dir
    heldout_keys, heldout_path = ctx.heldout_keys, ctx.heldout_path
    generated_path = ctx.generated_path
    validated_path = ctx.validated_path
    new_examples_added = ctx.new_examples_added
    train_meta, train_log = ctx.train_meta, ctx.train_log
    skip_high_eval = ctx.cfg["skip_high_eval"]
    eval_ci = ctx.eval_ci

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
