"""CLI, config resolution, and resume-override auditing for the loop.

Defaults come from ``loop_config.toml``; explicit CLI flags override the file,
and on ``--resume`` the saved run config wins unless a flag was typed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.defaults import DEFAULT_MODEL, DEFAULT_RUNS_ROOT, DEFAULT_SEED_DATA
from core.io import load_json
from core.loop_config import (
    EvalCIConfig,
    LoopConfig,
    load_loop_config,
    resolve_config_path,
)

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
