"""Math dataset operations: CSV IO, sampling, splits, canonical keys, scoring."""

from __future__ import annotations

import ast
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from core.io import atomic_text_writer, ensure_dir, save_json
from core.metrics import EvalMetrics


@dataclass(frozen=True)
class MathExample:
    """One arithmetic problem with a reference solution."""

    operation: str
    solution: str

    def key(self) -> str:
        return self.operation.strip()


def load_math_csv(path: Path) -> list[MathExample]:
    """Load operation/solution rows (also tolerates operation-only files)."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    examples: list[MathExample] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        if "operation" not in reader.fieldnames:
            raise ValueError(f"CSV must have an 'operation' column: {path}")

        has_solution = "solution" in reader.fieldnames
        for line_number, row in enumerate(reader, start=2):
            operation = (row.get("operation") or "").strip()
            if not operation:
                raise ValueError(f"{path}:{line_number} empty operation")
            solution = (row.get("solution") or "").strip() if has_solution else ""
            examples.append(MathExample(operation=operation, solution=solution))
    return examples


def write_math_csv(path: Path, examples: Sequence[MathExample]) -> None:
    """Write operation,solution CSV (UTF-8, LF)."""
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["operation", "solution"])
        for ex in examples:
            writer.writerow([ex.operation, ex.solution])


def write_operations_csv(path: Path, operations: Sequence[str]) -> None:
    """Write operation-only CSV for ask_arithmetic input."""
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["operation"])
        for op in operations:
            writer.writerow([op])


def merge_unique_examples(
    base: Sequence[MathExample],
    extra: Sequence[MathExample],
) -> list[MathExample]:
    """Append extra examples not already in base (canonical comparison).

    Canonical keys keep near-duplicates (``"34+12"`` when ``"12+34"`` is
    present) out of the train pool, not just exact string matches.
    """
    seen = {canonical_operation_key(ex.operation) for ex in base}
    merged = list(base)
    for ex in extra:
        key = canonical_operation_key(ex.operation)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(ex)
    return merged


def sample_examples(
    examples: Sequence[MathExample],
    n: int,
    *,
    seed: int,
) -> list[MathExample]:
    """Deterministic sample without replacement (or full set if n >= len)."""
    if n <= 0:
        return []
    if n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    return rng.sample(list(examples), n)


def split_train_heldout(
    examples: Sequence[MathExample],
    *,
    heldout_fraction: float,
    seed: int,
) -> tuple[list[MathExample], list[MathExample]]:
    """Deterministic train / held-out split (held-out never used for training).

    Same shuffle style as ``train_math_llm_judge._split_problems`` so results
    are stable across runs with the same seed.
    """
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError(
            f"heldout_fraction must be in (0, 1), got {heldout_fraction}"
        )
    if len(examples) < 2:
        raise ValueError("Need at least 2 examples to form a held-out split")

    indexed = list(enumerate(examples))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    ordered = [ex for _, ex in indexed]

    n_heldout = max(1, int(round(len(ordered) * heldout_fraction)))
    n_heldout = min(n_heldout, len(ordered) - 1)
    heldout = ordered[:n_heldout]
    train = ordered[n_heldout:]
    return train, heldout


# ---------------------------------------------------------------------------
# Canonical operation keys (leak-safe duplicate / held-out detection)
# ---------------------------------------------------------------------------

# Calls whose argument order carries no meaning for the leak guard.
_COMMUTATIVE_CALLS = {"gcd", "lcm", "max", "min"}


def _canonical_ast_node(node: ast.AST) -> str:
    """Serialize an AST node in a commutative-normalized canonical form."""
    if isinstance(node, ast.Expression):
        return _canonical_ast_node(node.body)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return repr(value)
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.BinOp):
        left = _canonical_ast_node(node.left)
        right = _canonical_ast_node(node.right)
        if isinstance(node.op, (ast.Add, ast.Mult)) and right < left:
            left, right = right, left
        return f"({left}{type(node.op).__name__}{right})"
    if isinstance(node, ast.UnaryOp):
        return f"({type(node.op).__name__}{_canonical_ast_node(node.operand)})"
    if isinstance(node, ast.Call):
        if node.keywords:
            return ast.dump(node)
        name = _canonical_ast_node(node.func)
        args = [_canonical_ast_node(arg) for arg in node.args]
        if name.lower() in _COMMUTATIVE_CALLS:
            args.sort()
        return f"{name}({','.join(args)})"
    return ast.dump(node)


def canonical_operation_key(operation: str) -> str:
    """Canonical form of an operation for duplicate / held-out leak detection.

    Normalizes whitespace, parentheses, ``^`` vs ``**``, numeric formatting
    (``4.0`` vs ``4``), letter case, and operand order of commutative
    operators (``+``, ``*``, ``gcd``, ``lcm``, ``max``, ``min``). So
    ``"12 + 34"`` and ``"(34+12)"``-style variants collapse to one key.

    Falls back to whitespace-stripped lowercase text when the surface syntax
    is not Python-parseable (e.g. postfix factorial ``5!``).
    """
    text = operation.strip()
    if not text:
        return text
    candidate = text.replace("^", "**")
    try:
        tree = ast.parse(candidate, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return re.sub(r"\s+", "", text).lower()
    return _canonical_ast_node(tree.body)


def operation_keys(
    examples: Sequence[MathExample],
    *,
    key_fn: Callable[[str], str] = canonical_operation_key,
) -> set[str]:
    return {key_fn(ex.operation) for ex in examples}


def filter_out_operations(
    examples: Sequence[MathExample],
    banned: set[str],
    *,
    key_fn: Callable[[str], str] = canonical_operation_key,
) -> list[MathExample]:
    """Drop any example whose canonical operation is in ``banned``.

    Canonical matching catches held-out leaks in disguise: whitespace variants,
    commuted operands (``34+12`` vs ``12+34``), and ``^``/``**`` rewrites.
    """
    if not banned:
        return list(examples)
    return [ex for ex in examples if key_fn(ex.operation) not in banned]


def assert_disjoint_train_heldout(
    train: Sequence[MathExample],
    heldout: Sequence[MathExample],
) -> None:
    """Raise if train and held-out share any canonical operation forms."""
    overlap = operation_keys(train) & operation_keys(heldout)
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(
            f"Train/held-out leak: {len(overlap)} shared operations "
            f"(e.g. {sample})"
        )


def write_split_manifest(
    path: Path,
    *,
    seed_data: Path,
    train: Sequence[MathExample],
    heldout: Sequence[MathExample],
    eval_train_sample: Sequence[MathExample],
    eval_heldout_sample: Sequence[MathExample],
    heldout_fraction: float,
    rng_seed: int,
) -> None:
    """Record how the never-train test set was carved out (dashboard + audit)."""
    save_json(
        path,
        {
            "seed_data": str(seed_data),
            "heldout_fraction": heldout_fraction,
            "rng_seed": rng_seed,
            "n_seed": len(train) + len(heldout),
            "n_train_initial": len(train),
            "n_heldout": len(heldout),
            "n_eval_train_sample": len(eval_train_sample),
            "n_eval_heldout_sample": len(eval_heldout_sample),
            "heldout_operations": [ex.operation for ex in heldout],
            "eval_train_operations": [ex.operation for ex in eval_train_sample],
            "eval_heldout_operations": [ex.operation for ex in eval_heldout_sample],
            "note": (
                "heldout_operations must never appear in train_data or generation. "
                "eval_train_sample is drawn from the initial train split (in-domain). "
                "eval_heldout_sample is drawn from heldout only."
            ),
        },
    )


def canonicalize_solution_string(value: str) -> str:
    """Normalize a numeric solution string for storage/comparison."""
    text = value.strip().lstrip("+")
    if not text:
        return text
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    if abs(number - round(number)) <= 1e-9 and abs(number) <= 2**53:
        return str(int(round(number)))
    return text


# Longest seed operation is ~43 chars; anything past this is a degenerate
# generation (e.g. digit ops on thousand-digit literals) that could grind
# evaluate_operation for minutes.
MAX_PROGRAMMATIC_OP_LEN = 200


def programmatic_solution(operation: str) -> str | None:
    """Evaluate operation with test_arithmetic; return None if unsupported."""
    if len(operation) > MAX_PROGRAMMATIC_OP_LEN:
        return None
    # Import lazily so pure helpers stay importable without the full env.
    from mathtask.test_arithmetic import evaluate_operation

    try:
        value = evaluate_operation(operation)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError, SyntaxError, RecursionError):
        return None
    if not math.isfinite(value):
        return None
    if abs(value - round(value)) <= 1e-9 and abs(value) <= 2**53:
        return str(int(round(value)))
    # Keep a stable short float repr for non-integers.
    return canonicalize_solution_string(f"{value:.12g}")


def solutions_match(expected: str, actual: str) -> bool:
    """Numeric equality with the same tolerance as test_arithmetic scoring."""
    from mathtask.test_arithmetic import extract_numeric_answer, numbers_equal

    exp = extract_numeric_answer(expected)
    act = extract_numeric_answer(actual)
    if exp is None or act is None:
        return expected.strip() == actual.strip()
    return numbers_equal(exp, act)


def score_answers_csv(
    answers_csv: Path,
    scored_csv: Path,
    *,
    effort: str = "instant",
    use_cli: bool = True,
) -> EvalMetrics:
    """Score an ask_arithmetic output CSV.

    By default shells out to ``test_arithmetic.py`` (same as a manual run).
    Set ``use_cli=False`` to call ``is_correct`` in-process (used by unit tests).
    """
    ensure_dir(scored_csv.parent)

    if use_cli:
        from mathtask.math_integration import run_test_arithmetic

        run_test_arithmetic(
            effort=effort,
            input_csv=answers_csv,
            output_csv=scored_csv,
        )
    else:
        from mathtask.test_arithmetic import is_completed, is_correct

        with (
            answers_csv.open(newline="", encoding="utf-8-sig") as inp,
            scored_csv.open("w", newline="", encoding="utf-8") as out,
        ):
            reader = csv.DictReader(inp)
            if reader.fieldnames is None:
                raise ValueError(f"Empty answers CSV: {answers_csv}")
            required = {"operation", "output"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"{answers_csv} missing columns {sorted(missing)}")

            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(["operation", "output", "correct"])
            for row in reader:
                operation = (row.get("operation") or "").strip()
                output = row.get("output") or ""
                if not operation:
                    continue
                if not is_completed(output):
                    writer.writerow([operation, output, "incomplete"])
                    continue
                ok = is_correct(operation, output)
                writer.writerow([operation, output, "true" if ok else "false"])

    return metrics_from_scored_csv(scored_csv, answers_csv=answers_csv, effort=effort)


def metrics_from_scored_csv(
    scored_csv: Path,
    *,
    answers_csv: Path | None = None,
    effort: str = "",
) -> EvalMetrics:
    """Aggregate a test_arithmetic ``*_test.csv`` into EvalMetrics."""
    total = completed = correct = incomplete = 0
    with scored_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty scored CSV: {scored_csv}")
        for row in reader:
            operation = (row.get("operation") or "").strip()
            if not operation:
                continue
            total += 1
            flag = (row.get("correct") or "").strip().lower()
            if flag == "incomplete":
                incomplete += 1
                continue
            completed += 1
            if flag == "true":
                correct += 1

    # Every requested operation belongs in the denominator. Empty/invalid
    # responses are failures, not rows that can silently improve accuracy.
    accuracy = (correct / total) if total else 0.0
    return EvalMetrics(
        effort=effort,
        total=total,
        completed=completed,
        correct=correct,
        accuracy=accuracy,
        incomplete=incomplete,
        answers_csv=str(answers_csv) if answers_csv else None,
        scored_csv=str(scored_csv),
    )


def format_seed_block(examples: Iterable[MathExample], limit: int = 20) -> str:
    """Human-readable sample block for generation prompts."""
    lines = []
    for i, ex in enumerate(examples):
        if i >= limit:
            break
        lines.append(f"- operation: {ex.operation} | solution: {ex.solution}")
    return "\n".join(lines)
