#!/usr/bin/env python3
"""The research loop: generate → validate → eval → train → re-eval, until
gains go marginal. **This is the file you iterate on** — the harness
primitives it composes live in ``prepare.py`` and stay fixed; knobs live in
``loop_config.toml``; goal and ground rules in ``program.md``.

Example::

    source .venv/bin/activate
    python loop.py --max-iters 5 --train-max-steps 20
"""

from __future__ import annotations

import sys
from pathlib import Path

# Standalone project root (this directory).
_LOOP_ROOT = Path(__file__).resolve().parent
if str(_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOOP_ROOT))

import prepare
from core.cli import (
    _serialize_args,
    eval_ci_from_args,
    parse_args,
    resolve_resume_config,
    validate_args,
)
from core.io import append_jsonl, save_json
from core.loop_config import EvalCIConfig
from core.progress import ProgressTracker
from core.runstate import LoopState, save_state
from core.stopping import marginal_improvement_streak
from mathtask.math_integration import check_model_consistency


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
    """One full iteration: each phase resumes from the journal if interrupted."""
    if eval_ci is None:
        eval_ci = EvalCIConfig()
    cfg = {
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
    }
    ctx = prepare.begin_iteration(state, progress, cfg, eval_ci)
    prepare.generate_and_validate(ctx)
    pre_train, pre_heldout = prepare.evaluate_pre(ctx)
    prepare.train_policy(ctx)
    post_train, post_heldout = prepare.evaluate_post(ctx, pre_train, pre_heldout)
    return prepare.finish_iteration(
        ctx, pre_train, pre_heldout, post_train, post_heldout
    )


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
    run_dir, state, progress = prepare.init_run(
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
