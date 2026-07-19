#!/usr/bin/env python3
"""Run ask+test for instant vs high and report the performance differential.

Integrates with the existing math scripts:

  - ``ask_arithmetic.py``  — sample answers (instant / high, optional --model-path)
  - ``test_arithmetic.py`` — score answers (same CLI as manual runs)

Builds (or reuses) an eval sample CSV, runs both efforts, scores, and writes:

  answers_{effort}.csv
  scored_{effort}.csv          # test_arithmetic output
  differential.json
  eval_manifest.json           # paths + metrics for dashboards

Example::

    source .venv/bin/activate
    python -m run_evals \\
        --data data/arithmetic_operations.csv \\
        --sample-size 200 \\
        --out-dir output/autoresearch/run/iter_000/eval_pre \\
        --tag pre
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Project root is this package dir; parent holds ask/test/train math scripts.
_LOOP_ROOT = Path(__file__).resolve().parent
_MATH_ROOT = _LOOP_ROOT.parent
for _p in (_LOOP_ROOT, _MATH_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import (
    DEFAULT_EVAL_SAMPLE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_SEED_DATA,
    DifferentialMetrics,
    EvalMetrics,
    ensure_dir,
    load_math_csv,
    sample_examples,
    save_json,
    score_answers_csv,
    write_math_csv,
    write_operations_csv,
)
from math_integration import run_ask_arithmetic


def ensure_eval_sample(
    data_path: Path,
    sample_path: Path,
    *,
    sample_size: int,
    rng_seed: int,
) -> list[str]:
    """Create a fixed operation sample CSV if missing; return operations."""
    if sample_path.is_file():
        examples = load_math_csv(sample_path)
        ops = [ex.operation for ex in examples]
        print(f"Reusing eval sample ({len(ops)} ops): {sample_path}", flush=True)
        return ops

    examples = load_math_csv(data_path)
    sampled = sample_examples(examples, sample_size, seed=rng_seed)
    write_operations_csv(sample_path, [ex.operation for ex in sampled])
    # Keep solutions alongside when available (debugging + dashboard).
    sol_path = sample_path.with_name(sample_path.stem + "_with_solutions.csv")
    write_math_csv(sol_path, sampled)
    print(
        f"Wrote eval sample ({len(sampled)} ops) → {sample_path}",
        flush=True,
    )
    return [ex.operation for ex in sampled]


def eval_effort(
    *,
    effort: str,
    sample_csv: Path,
    out_dir: Path,
    model: str,
    model_path: str | None,
) -> EvalMetrics:
    ensure_dir(out_dir)
    answers = out_dir / f"answers_{effort}.csv"
    scored = out_dir / f"scored_{effort}.csv"

    # 1) ask_arithmetic.py (same flags as manual instant/high runs)
    run_ask_arithmetic(
        effort=effort,
        input_csv=sample_csv,
        output_csv=answers,
        model=model,
        model_path=model_path,
    )
    # 2) test_arithmetic.py → scored_{effort}.csv
    metrics = score_answers_csv(answers, scored, effort=effort, use_cli=True)
    metrics.effort = effort
    metrics.model_path = model_path
    expected_total = len(load_math_csv(sample_csv))
    if metrics.total != expected_total:
        raise RuntimeError(
            f"Eval coverage mismatch for {effort}: scored {metrics.total} rows "
            f"from a {expected_total}-row sample"
        )
    if metrics.completed + metrics.incomplete != metrics.total:
        raise RuntimeError(
            f"Inconsistent eval accounting for {effort}: total={metrics.total}, "
            f"completed={metrics.completed}, incomplete={metrics.incomplete}"
        )
    # Also keep a copy named like the manual convention for familiarity.
    manual_style = out_dir / f"arithmetic_run_{effort}_test.csv"
    if scored.is_file():
        shutil.copy2(scored, manual_style)

    print(
        f"[{effort}] {metrics.correct}/{metrics.total} correct overall "
        f"({100.0 * metrics.accuracy:.1f}%); "
        f"completed={metrics.completed}/{metrics.total}"
        f"{f' model_path={model_path}' if model_path else ''}",
        flush=True,
    )
    return metrics


def run_differential(
    *,
    sample_csv: Path,
    out_dir: Path,
    model: str,
    instant_model_path: str | None,
    high_model_path: str | None,
    skip_high: bool,
    tag: str = "",
) -> DifferentialMetrics:
    ensure_dir(out_dir)
    # Pin the sample used for this eval dir (dashboard / resume).
    pinned = out_dir / "eval_sample.csv"
    if sample_csv.resolve() != pinned.resolve() and sample_csv.is_file():
        shutil.copy2(sample_csv, pinned)

    instant = eval_effort(
        effort="instant",
        sample_csv=sample_csv,
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
            sample_csv=sample_csv,
            out_dir=out_dir,
            model=model,
            model_path=high_model_path,
        )
        gap = high.accuracy - instant.accuracy

    diff = DifferentialMetrics(
        sample_size=instant.total,
        instant=instant,
        high=high,
        accuracy_gap=gap,
    )
    payload = diff.to_dict()
    if tag:
        payload["tag"] = tag
    metrics_path = out_dir / "differential.json"
    save_json(metrics_path, payload)

    # Dashboard-oriented manifest with absolute-ish relative paths.
    manifest = {
        "tag": tag,
        "sample_csv": str(sample_csv),
        "out_dir": str(out_dir),
        "instant": instant.to_dict(),
        "high": high.to_dict(),
        "accuracy_gap": gap,
        "files": {
            "answers_instant": str(out_dir / "answers_instant.csv"),
            "scored_instant": str(out_dir / "scored_instant.csv"),
            "answers_high": str(out_dir / "answers_high.csv") if not skip_high else None,
            "scored_high": str(out_dir / "scored_high.csv") if not skip_high else None,
            "differential": str(metrics_path),
        },
    }
    save_json(out_dir / "eval_manifest.json", manifest)

    if gap is None:
        print(
            f"Differential: high eval skipped "
            f"(instant={instant.accuracy:.4f}) → {metrics_path}",
            flush=True,
        )
    else:
        print(
            f"Differential: high - instant = {gap:+.4f} "
            f"(instant={instant.accuracy:.4f}, high={high.accuracy:.4f}) "
            f"→ {metrics_path}",
            flush=True,
        )
    return diff


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_SEED_DATA,
        help="Source pool to sample eval problems from (if sample missing)",
    )
    p.add_argument(
        "--sample",
        type=Path,
        default=None,
        help="Fixed eval sample CSV (created if missing). Default: <out-dir>/eval_sample.csv",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_EVAL_SAMPLE_SIZE,
        help=f"Eval sample size when creating a new sample (default: {DEFAULT_EVAL_SAMPLE_SIZE})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for answers, scores, and differential.json",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--instant-model-path",
        default=None,
        help="Optional tinker:// path for the instant policy (post-train)",
    )
    p.add_argument(
        "--high-model-path",
        default=None,
        help="Optional tinker:// path for high-reasoning eval (default: base model)",
    )
    p.add_argument(
        "--skip-high",
        action="store_true",
        help="Only run instant eval (cheaper re-checks)",
    )
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument(
        "--tag",
        default="",
        help="Optional label stored in differential.json / eval_manifest.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> DifferentialMetrics:
    args = parse_args(argv)
    out_dir = ensure_dir(args.out_dir)
    sample_path = args.sample or (out_dir / "eval_sample.csv")
    ensure_eval_sample(
        args.data,
        sample_path,
        sample_size=args.sample_size,
        rng_seed=args.rng_seed,
    )
    return run_differential(
        sample_csv=sample_path,
        out_dir=out_dir,
        model=args.model,
        instant_model_path=args.instant_model_path,
        high_model_path=args.high_model_path,
        skip_high=args.skip_high,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
