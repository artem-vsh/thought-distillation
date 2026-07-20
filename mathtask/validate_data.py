#!/usr/bin/env python3
"""Validate generated arithmetic examples with program checks + high reasoning.

Pipeline per example:
  1. Programmatic eval via test_arithmetic.evaluate_operation (when possible)
  2. High-reasoning model cross-check; must emit keep/discard + confidence
  3. Retain only rows we are sure about:
       - programmatic solution matches claimed solution (or model-corrected), AND
       - model verdict is keep with high confidence

Example::

    source .venv/bin/activate
    python -m validate_data \\
        --input output/autoresearch/run/iter_000/generated.csv \\
        --output output/autoresearch/run/iter_000/validated.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Standalone project root (this directory).
_LOOP_ROOT = Path(__file__).resolve().parent.parent
if str(_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOOP_ROOT))

from core.defaults import DEFAULT_MODEL
from core.io import append_jsonl, ensure_dir
from mathtask.dataset import (
    MathExample,
    canonicalize_solution_string,
    load_math_csv,
    programmatic_solution,
    solutions_match,
    write_math_csv,
)
from mathtask.parsing import confidence_is_high, parse_validation_verdict
from mathtask.tinker_sample import SampleRequest, sample_many_sync


def build_validation_prompt(ex: MathExample) -> str:
    return "\n".join(
        [
            "You are a careful math data cleaner.",
            "Check whether the proposed solution is correct for the operation.",
            "Recompute the result step by step. Be conservative.",
            "",
            f"Operation: {ex.operation}",
            f"Proposed solution: {ex.solution}",
            "",
            "Rules:",
            "- If the operation is invalid, ambiguous, or you are unsure: discard.",
            "- If the proposed solution is wrong but you know the correct value,",
            "  you may keep it only if you are highly confident, and put the",
            "  corrected value in <solution>.",
            "- Prefer integer answers when the true value is an integer.",
            "",
            "Respond with exactly these tags (plus brief reasoning outside them):",
            "<verdict>keep</verdict> or <verdict>discard</verdict>",
            "<confidence>high</confidence> or medium or low",
            "<solution>NUMBER</solution>",
        ]
    )


def validate_examples(
    examples: list[MathExample],
    *,
    model: str,
    model_path: str | None,
    require_programmatic: bool,
    audit_path: Path | None,
) -> list[MathExample]:
    """Return only high-confidence, cross-checked examples."""
    if not examples:
        return []

    # Stage 1: programmatic pre-filter / solution repair.
    candidates: list[MathExample] = []
    for ex in examples:
        prog = programmatic_solution(ex.operation)
        if prog is not None:
            if ex.solution and solutions_match(prog, ex.solution):
                candidates.append(
                    MathExample(operation=ex.operation, solution=prog)
                )
            elif not ex.solution:
                candidates.append(
                    MathExample(operation=ex.operation, solution=prog)
                )
            else:
                # Claimed solution disagrees with the evaluator — still show to
                # the model only if we are not requiring programmatic agreement
                # up front; default is to drop mismatches early.
                if require_programmatic:
                    if audit_path:
                        append_jsonl(
                            audit_path,
                            {
                                "operation": ex.operation,
                                "proposed": ex.solution,
                                "programmatic": prog,
                                "stage": "programmatic_mismatch",
                                "kept": False,
                            },
                        )
                    continue
                candidates.append(ex)
        else:
            # Unevaluable programmatically — keep for LLM-only check.
            if require_programmatic:
                if audit_path:
                    append_jsonl(
                        audit_path,
                        {
                            "operation": ex.operation,
                            "proposed": ex.solution,
                            "stage": "not_programmatically_evaluable",
                            "kept": False,
                        },
                    )
                continue
            candidates.append(ex)

    print(
        f"Programmatic stage: {len(candidates)}/{len(examples)} candidates "
        f"for high-reasoning review",
        flush=True,
    )
    if not candidates:
        return []

    # Stage 2: high-reasoning cross-check.
    requests = [
        SampleRequest(key=str(i), user_content=build_validation_prompt(ex))
        for i, ex in enumerate(candidates)
    ]
    results = sample_many_sync(
        requests,
        reasoning_effort="high",
        model=model,
        model_path=model_path,
        temperature=0.0,
    )

    kept: list[MathExample] = []
    by_key = {r.key: r for r in results}
    for i, ex in enumerate(candidates):
        raw = by_key.get(str(i))
        text = raw.text if raw else ""
        verdict = parse_validation_verdict(text, fallback_solution=ex.solution)
        solution = verdict.solution or ex.solution
        solution = canonicalize_solution_string(solution)

        prog = programmatic_solution(ex.operation)
        if prog is not None:
            # Prefer the programmatic ground truth when available.
            solution = prog

        sure = (
            verdict.keep
            and confidence_is_high(verdict.confidence)
            and bool(solution)
        )

        if audit_path:
            append_jsonl(
                audit_path,
                {
                    "operation": ex.operation,
                    "proposed": ex.solution,
                    "final_solution": solution,
                    "programmatic": prog,
                    "verdict_keep": verdict.keep,
                    "confidence": verdict.confidence,
                    "kept": sure,
                },
            )

        if sure:
            kept.append(MathExample(operation=ex.operation, solution=solution))

    print(
        f"High-reasoning stage: kept {len(kept)}/{len(candidates)} "
        f"(high confidence + consistent)",
        flush=True,
    )
    return kept


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", type=Path, required=True, help="Generated CSV")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Validated CSV (operation, solution)",
    )
    p.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Optional JSONL audit log of keep/discard decisions",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-path", default=None)
    p.add_argument(
        "--allow-unevaluable",
        action="store_true",
        help=(
            "Keep examples the local evaluator cannot parse, if the model is "
            "highly confident (default: require programmatic eval)"
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    examples = load_math_csv(args.input)
    print(f"Validating {len(examples)} examples from {args.input}", flush=True)

    if args.audit:
        ensure_dir(args.audit.parent)
        # Truncate previous audit for this path.
        args.audit.write_text("", encoding="utf-8")

    kept = validate_examples(
        examples,
        model=args.model,
        model_path=args.model_path,
        require_programmatic=not args.allow_unevaluable,
        audit_path=args.audit,
    )
    write_math_csv(args.output, kept)
    print(f"Wrote {len(kept)} validated examples → {args.output}", flush=True)


if __name__ == "__main__":
    main()
