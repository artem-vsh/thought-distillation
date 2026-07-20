#!/usr/bin/env python3
"""Generate arithmetic data variations with a high-reasoning model.

Reads a seed CSV (operation, solution), samples a few examples, and asks
GPT-OSS @ high reasoning to invent similar problems with solutions.

Example::

    source .venv/bin/activate
    python -m generate_data \\
        --seed data/arithmetic_operations.csv \\
        --output output/autoresearch/run/iter_000/generated.csv \\
        --target 100
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Standalone project root (this directory).
_LOOP_ROOT = Path(__file__).resolve().parent.parent
if str(_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOOP_ROOT))

from common import (
    DEFAULT_GEN_TARGET,
    DEFAULT_GEN_TEMPERATURE,
    DEFAULT_MODEL,
    DEFAULT_SEED_DATA,
    DEFAULT_SEED_SAMPLES,
    DEFAULT_SEEDS_PER_BATCH,
    DEFAULT_VARIATIONS_PER_SEED,
    MathExample,
    canonical_operation_key,
    format_seed_block,
    load_math_csv,
    parse_generated_examples,
    sample_examples,
    write_math_csv,
)
from mathtask.math_integration import make_math_prompt
from mathtask.tinker_sample import SampleRequest, sample_many_sync

SUPPORTED_OPS_HINT = """
Supported expression surface syntax (same as test_arithmetic.evaluate_operation):
  - + - * / // % ** (or ^), unary ±, parentheses
  - postfix factorial: n! or (expr)!
  - functions: factorial/fact, comb/C, perm/P, gcd, lcm, abs, floor, ceil,
    round, sqrt, isqrt, fib, digit_sum, digit_product, reverse_digits,
    sigma, phi/totient, prime_count, nth_prime, pow, mod, max, min
Prefer integer-valued results when possible. Avoid division by zero.
Keep intermediate magnitudes reasonable (results |x| < 1e12 preferred).
Stay within evaluator limits or the example is discarded: factorial/fact n <= 20,
fib n <= 70, exponents <= 40, sigma/phi/totient n <= 1e12, comb/perm args <= 1e4,
prime_count n <= 2e5, nth_prime n <= 1e4, whole operation <= 200 characters.
These expressions are later asked with the same prompt as ask_arithmetic, e.g.
  "What is the result of <operation>? Give only the result."
""".strip()


def build_generation_prompt(
    seeds: list[MathExample],
    *,
    n_variations: int,
    batch_id: int,
) -> str:
    """Prompt the high-reasoning model to invent new problems."""
    seed_block = format_seed_block(seeds, limit=len(seeds))
    example_prompt = make_math_prompt("<operation>")
    return "\n".join(
        [
            "You are creating new training problems for an arithmetic / discrete-math dataset.",
            "Given the seed examples below, invent NEW problems that are variations:",
            "similar operators/structure, different numbers, comparable difficulty.",
            "Do NOT copy seeds. Each problem must be unique.",
            "",
            f"At train/eval time each operation is asked as: {example_prompt!r}",
            "",
            SUPPORTED_OPS_HINT,
            "",
            f"Seed examples (batch {batch_id}):",
            seed_block,
            "",
            f"Produce exactly {n_variations} new examples.",
            "For each example compute the correct numeric solution carefully.",
            "",
            "Respond with a JSON array only (no markdown fence), of the form:",
            '[',
            '  {"operation": "12*34+5", "solution": "413"},',
            '  {"operation": "gcd(48,18)", "solution": "6"}',
            "]",
            "Solutions must be bare numbers (integers preferred, no units/words).",
        ]
    )


def generate_variations(
    seeds: list[MathExample],
    *,
    target: int,
    variations_per_seed_batch: int,
    seed_batch_size: int,
    model: str,
    model_path: str | None,
    temperature: float,
    rng_seed: int,
) -> list[MathExample]:
    """Ask high reasoning for variations until we have ~target unique ops."""
    if not seeds:
        raise ValueError("No seed examples to vary")

    # Build batches of seed examples to condition generation.
    batches: list[list[MathExample]] = []
    # Roughly enough batches to hit target.
    n_batches = max(1, math.ceil(target / max(1, variations_per_seed_batch)))
    for b in range(n_batches):
        batch = sample_examples(
            seeds,
            min(seed_batch_size, len(seeds)),
            seed=rng_seed + b * 17,
        )
        batches.append(batch)

    requests = [
        SampleRequest(
            key=f"batch-{i}",
            user_content=build_generation_prompt(
                batch, n_variations=variations_per_seed_batch, batch_id=i
            ),
        )
        for i, batch in enumerate(batches)
    ]

    print(
        f"Generating with high reasoning: {len(requests)} batches, "
        f"~{variations_per_seed_batch} examples each (target {target})",
        flush=True,
    )
    results = sample_many_sync(
        requests,
        reasoning_effort="high",
        model=model,
        model_path=model_path,
        temperature=temperature,
    )

    seen: set[str] = set()
    seed_keys = {canonical_operation_key(s.operation) for s in seeds}
    generated: list[MathExample] = []
    for res in results:
        parsed = parse_generated_examples(res.text)
        print(
            f"  {res.key}: parsed {len(parsed)} examples "
            f"(raw chars={len(res.text)})",
            flush=True,
        )
        for ex in parsed:
            # Canonical dedup also catches commuted/whitespace variants of
            # seeds and of already-kept generations.
            key = canonical_operation_key(ex.operation)
            if not key or key in seen or key in seed_keys:
                continue
            seen.add(key)
            generated.append(ex)
            if len(generated) >= target:
                return generated
    return generated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed",
        type=Path,
        default=DEFAULT_SEED_DATA,
        help=f"Seed CSV with operation,solution (default: {DEFAULT_SEED_DATA})",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output CSV path for generated (unvalidated) examples",
    )
    p.add_argument(
        "--target",
        type=int,
        default=DEFAULT_GEN_TARGET,
        help=f"How many unique examples to aim for (default: {DEFAULT_GEN_TARGET})",
    )
    p.add_argument(
        "--seed-samples",
        type=int,
        default=DEFAULT_SEED_SAMPLES,
        help=(
            "How many seed rows to draw from overall when building batches "
            f"(default: {DEFAULT_SEED_SAMPLES})"
        ),
    )
    p.add_argument(
        "--variations-per-batch",
        type=int,
        default=DEFAULT_VARIATIONS_PER_SEED,
        help=f"Examples requested per generation call (default: {DEFAULT_VARIATIONS_PER_SEED})",
    )
    p.add_argument(
        "--seeds-per-batch",
        type=int,
        default=DEFAULT_SEEDS_PER_BATCH,
        help=(
            "Seed examples shown in each generation prompt "
            f"(default: {DEFAULT_SEEDS_PER_BATCH})"
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="Base model name")
    p.add_argument(
        "--model-path",
        default=None,
        help="Optional tinker:// sampler path for the generator",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_GEN_TEMPERATURE,
        help=f"Sampling temperature for generation (default: {DEFAULT_GEN_TEMPERATURE})",
    )
    p.add_argument("--rng-seed", type=int, default=0, help="RNG seed for seed sampling")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    all_seeds = load_math_csv(args.seed)
    # Prefer rows that already have solutions as few-shot exemplars.
    with_sol = [ex for ex in all_seeds if ex.solution]
    pool = with_sol or all_seeds
    seeds = sample_examples(pool, min(args.seed_samples, len(pool)), seed=args.rng_seed)
    print(f"Loaded {len(all_seeds)} seed rows; using {len(seeds)} as few-shot pool", flush=True)

    generated = generate_variations(
        seeds,
        target=args.target,
        variations_per_seed_batch=args.variations_per_batch,
        seed_batch_size=args.seeds_per_batch,
        model=args.model,
        model_path=args.model_path,
        temperature=args.temperature,
        rng_seed=args.rng_seed,
    )
    write_math_csv(args.output, generated)
    print(f"Wrote {len(generated)} generated examples → {args.output}", flush=True)


if __name__ == "__main__":
    main()
