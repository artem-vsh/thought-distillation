#!/usr/bin/env python3
"""One Tinker RL train step via ``train_math_llm_judge.py``.

Same setup as a manual train run:

  - **Policy:** instant (``gpt_oss_no_sysprompt`` + final-channel prefill)
  - **Judge:** high reasoning (``gpt_oss_high_reasoning``)
  - **Data:** ``operation,solution`` CSV (validated with ``load_math_problems``)

After training, indexes checkpoints + metrics into the run directory so a
dashboard can track progress without scraping nested cookbook paths.

Example::

    source .venv/bin/activate
    python -m train_step \\
        --data output/autoresearch/run/train_data.csv \\
        --log-path output/autoresearch/run/iter_000/train \\
        --load-checkpoint tinker://.../weights/000020 \\
        --max-steps 20
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

# Standalone project root (this directory).
_LOOP_ROOT = Path(__file__).resolve().parent.parent
if str(_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOOP_ROOT))

from core.defaults import (
    DEFAULT_MODEL,
    DEFAULT_TRAIN_EVAL_EVERY,
    DEFAULT_TRAIN_GROUPS_PER_BATCH,
    DEFAULT_TRAIN_GROUP_SIZE,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_LORA_RANK,
    DEFAULT_TRAIN_MAX_STEPS,
    DEFAULT_TRAIN_SAVE_EVERY,
)
from core.io import ensure_dir, run_python, save_json
from mathtask.math_integration import (
    TRAIN_MATH_LLM_JUDGE_SCRIPT,
    load_train_problems,
    math_model_defaults,
)


def build_train_argv(
    *,
    data_path: Path,
    log_path: Path,
    load_checkpoint: str | None,
    judge_model_path: str | None,
    model: str,
    max_steps: int,
    groups_per_batch: int,
    group_size: int,
    save_every: int,
    eval_every: int,
    learning_rate: float,
    lora_rank: int,
    seed: int,
    behavior_if_log_dir_exists: str,
) -> list[str]:
    """Build chz-style CLI args for train_math_llm_judge.py.

    Auto-switch semantics:
      - ``load_checkpoint`` None → policy starts from base model
      - ``load_checkpoint`` set → continue training from that state path
      - ``judge_model_path`` None → judge is base model @ high reasoning
      - ``judge_model_path`` set → judge samples that checkpoint @ high reasoning
    """
    defaults = math_model_defaults()
    args = [
        "-m",
        "mathtask.train_math_llm_judge",
        f"data_path={data_path}",
        f"log_path={log_path}",
        f"model_name={model}",
        f"max_steps={max_steps}",
        f"groups_per_batch={groups_per_batch}",
        f"group_size={group_size}",
        f"save_every={save_every}",
        f"eval_every={eval_every}",
        f"learning_rate={learning_rate}",
        f"lora_rank={lora_rank}",
        f"seed={seed}",
        f"behavior_if_log_dir_exists={behavior_if_log_dir_exists}",
        # Explicit so train_step always matches the documented math defaults
        # even if train_math_llm_judge defaults change later.
        f"renderer_name={defaults['policy_renderer']}",
        f"judge_model={defaults['judge_model']}",
        f"judge_renderer_name={defaults['judge_renderer']}",
        f"judge_max_tokens={defaults['judge_max_tokens']}",
    ]
    if load_checkpoint:
        args.append(f"load_checkpoint_path={load_checkpoint}")
    if judge_model_path:
        args.append(f"judge_model_path={judge_model_path}")
    return args


def _snapshot_train_artifacts(log_path: Path, dest: Path) -> dict[str, Any]:
    """Copy cookbook intermediate files into dest for durable access."""
    ensure_dir(dest)
    copied: list[str] = []
    for name in (
        "checkpoints.jsonl",
        "metrics.jsonl",
        "config.json",
        "logs.log",
        "timing_spans.jsonl",
    ):
        src = log_path / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied.append(name)

    # Copy in-training eval HTML/jsonl if present (iteration_*/eval_test*)
    eval_copies: list[str] = []
    for path in sorted(log_path.glob("iteration_*/eval_test*")):
        rel = path.relative_to(log_path)
        target = dest / "cookbook_evals" / rel
        ensure_dir(target.parent)
        if path.is_file():
            shutil.copy2(path, target)
            eval_copies.append(str(rel))

    return {"copied": copied, "cookbook_evals": eval_copies, "dest": str(dest)}


def run_train_step(
    *,
    data_path: Path,
    log_path: Path,
    load_checkpoint: str | None,
    judge_model_path: str | None = None,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_TRAIN_MAX_STEPS,
    groups_per_batch: int = DEFAULT_TRAIN_GROUPS_PER_BATCH,
    group_size: int = DEFAULT_TRAIN_GROUP_SIZE,
    save_every: int = DEFAULT_TRAIN_SAVE_EVERY,
    eval_every: int = DEFAULT_TRAIN_EVAL_EVERY,
    learning_rate: float = DEFAULT_TRAIN_LEARNING_RATE,
    lora_rank: int = DEFAULT_TRAIN_LORA_RANK,
    seed: int = 0,
    behavior_if_log_dir_exists: str = "delete",
    artifact_dir: Path | None = None,
) -> dict:
    """Run one training job and return checkpoint + artifact metadata."""
    # Fail fast if CSV is not trainer-compatible (operation + non-empty solution).
    problems = load_train_problems(data_path)
    print(
        f"Train data OK via train_math_llm_judge.load_math_problems: "
        f"{len(problems)} problems from {data_path}",
        flush=True,
    )

    ensure_dir(log_path)
    argv = build_train_argv(
        data_path=data_path,
        log_path=log_path,
        load_checkpoint=load_checkpoint,
        judge_model_path=judge_model_path,
        model=model,
        max_steps=max_steps,
        groups_per_batch=groups_per_batch,
        group_size=group_size,
        save_every=save_every,
        eval_every=eval_every,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        seed=seed,
        behavior_if_log_dir_exists=behavior_if_log_dir_exists,
    )
    policy_src = load_checkpoint or "base model"
    judge_src = judge_model_path or "base model (high reasoning)"
    print(
        f"Training instant policy | data={data_path} | log={log_path} | "
        f"policy_from={policy_src!r} | judge_from={judge_src!r} | "
        f"max_steps={max_steps}",
        flush=True,
    )
    # Persist the exact CLI for dashboard / resume debugging.
    save_json(
        log_path / "train_step_argv.json",
        {"argv": argv, "n_problems": len(problems)},
    )
    run_python(argv)

    state_path, sampler_path = get_checkpoint_paths(log_path)
    if state_path is None or sampler_path is None:
        # Without this, the loop would keep going and silently eval/train the
        # base model, which reads as "converged" instead of "misconfigured".
        raise RuntimeError(
            f"Training finished but no usable checkpoint was found in {log_path} "
            f"(state_path={state_path!r}, sampler_path={sampler_path!r}). "
            f"Ensure save_every ({save_every}) <= max_steps ({max_steps}) so at "
            "least one checkpoint is saved."
        )
    snap_dest = artifact_dir or (log_path / "artifacts_snapshot")
    snapshot = _snapshot_train_artifacts(log_path, snap_dest)

    meta = {
        "log_path": str(log_path),
        "data_path": str(data_path),
        "n_problems": len(problems),
        "load_checkpoint_path": load_checkpoint,
        "judge_model_path": judge_model_path,
        "policy_source": "checkpoint" if load_checkpoint else "base",
        "judge_source": "checkpoint" if judge_model_path else "base",
        "state_path": state_path,
        "sampler_path": sampler_path,
        "max_steps": max_steps,
        "groups_per_batch": groups_per_batch,
        "group_size": group_size,
        "save_every": save_every,
        "eval_every": eval_every,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "seed": seed,
        "artifacts_snapshot": snapshot,
        "integration": {
            "script": str(TRAIN_MATH_LLM_JUDGE_SCRIPT),
            "policy_renderer": math_model_defaults()["policy_renderer"],
            "judge_renderer": math_model_defaults()["judge_renderer"],
        },
    }
    save_json(log_path / "train_step_meta.json", meta)
    # Convenience: also drop last checkpoint pointers at a stable name.
    save_json(
        log_path / "last_checkpoint.json",
        {
            "state_path": state_path,
            "sampler_path": sampler_path,
            "log_path": str(log_path),
        },
    )
    print(
        f"Train step done. state_path={state_path!r} sampler_path={sampler_path!r}",
        flush=True,
    )
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Train CSV (operation,solution)")
    p.add_argument("--log-path", type=Path, required=True, help="Tinker/cookbook log dir")
    p.add_argument(
        "--load-checkpoint",
        default=None,
        help="Resume policy weights from this tinker:// state path",
    )
    p.add_argument(
        "--judge-model-path",
        default=None,
        help=(
            "Optional tinker:// sampler path for the high-reasoning judge. "
            "Omit to use the base model."
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-steps", type=int, default=DEFAULT_TRAIN_MAX_STEPS)
    p.add_argument("--groups-per-batch", type=int, default=DEFAULT_TRAIN_GROUPS_PER_BATCH)
    p.add_argument("--group-size", type=int, default=DEFAULT_TRAIN_GROUP_SIZE)
    p.add_argument("--save-every", type=int, default=DEFAULT_TRAIN_SAVE_EVERY)
    p.add_argument("--eval-every", type=int, default=DEFAULT_TRAIN_EVAL_EVERY)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_TRAIN_LEARNING_RATE)
    p.add_argument("--lora-rank", type=int, default=DEFAULT_TRAIN_LORA_RANK)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--behavior-if-log-dir-exists",
        default="delete",
        choices=["delete", "resume", "ask", "raise"],
        help="What train_math_llm_judge does if log_path exists",
    )
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Where to copy checkpoints.jsonl / metrics.jsonl snapshots",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    return run_train_step(
        data_path=args.data,
        log_path=args.log_path,
        load_checkpoint=args.load_checkpoint,
        judge_model_path=args.judge_model_path,
        model=args.model,
        max_steps=args.max_steps,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        save_every=args.save_every,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        seed=args.seed,
        behavior_if_log_dir_exists=args.behavior_if_log_dir_exists,
        artifact_dir=args.artifact_dir,
    )


if __name__ == "__main__":
    main()


def find_latest_train_log(log_root: Path) -> Path | None:
    """Pick the newest math-llm-judge run directory under log_root."""
    if not log_root.is_dir():
        return None
    candidates = [p for p in log_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def get_checkpoint_paths(log_dir: Path) -> tuple[str | None, str | None]:
    """Return (state_path, sampler_path) from the last checkpoint in log_dir."""
    from tinker_cookbook.checkpoint_utils import get_last_checkpoint

    rec = get_last_checkpoint(str(log_dir), required_key="state_path")
    if rec is None:
        # Fall back to sampler-only checkpoints.
        rec = get_last_checkpoint(str(log_dir), required_key="sampler_path")
    if rec is None:
        return None, None
    return rec.state_path, rec.sampler_path
