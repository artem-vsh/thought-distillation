"""Shared helpers for the math autoresearch loop."""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# This directory is the standalone project root (``cd loop && python -m …``).
LOOP_ROOT = Path(__file__).resolve().parent
# All math scripts and data live here too (no parent-sandbox dependency).
MATH_ROOT = LOOP_ROOT
REPO_ROOT = LOOP_ROOT

_LOOP_SEED = LOOP_ROOT / "data" / "arithmetic_operations.csv"
DEFAULT_SEED_DATA = _LOOP_SEED
DEFAULT_RUNS_ROOT = LOOP_ROOT / "output" / "autoresearch"
DEFAULT_MODEL = "openai/gpt-oss-20b"

# Stop when absolute instant-accuracy improvement falls below this for N iters.
DEFAULT_MARGINAL_DELTA = 0.005  # 0.5 percentage points as fraction
DEFAULT_MARGINAL_STREAK = 2
DEFAULT_MAX_ITERS = 10

# How many problems to generate / eval per loop iteration.
DEFAULT_GEN_TARGET = 200
DEFAULT_SEED_SAMPLES = 40
DEFAULT_EVAL_SAMPLE_SIZE = 200
DEFAULT_VARIATIONS_PER_SEED = 5
DEFAULT_SEEDS_PER_BATCH = 8
DEFAULT_GEN_TEMPERATURE = 0.8
# Fraction of original seed carved out as never-train held-out test.
DEFAULT_HELDOUT_FRACTION = 0.10
# Below this many held-out eval rows the generalization track is too noisy
# to drive early stopping; init_run warns when the sample is smaller.
MIN_HELDOUT_EVAL_SAMPLE = 30

# Training defaults (match a modest train_math_llm_judge run).
DEFAULT_TRAIN_MAX_STEPS = 20
DEFAULT_TRAIN_GROUPS_PER_BATCH = 16
DEFAULT_TRAIN_GROUP_SIZE = 4
DEFAULT_TRAIN_SAVE_EVERY = 20
DEFAULT_TRAIN_EVAL_EVERY = 20
DEFAULT_TRAIN_LEARNING_RATE = 1e-5
DEFAULT_TRAIN_LORA_RANK = 32

# Early stopping: an instant-accuracy gain only counts as real progress when
# it exceeds max(marginal_delta, noise_z * binomial SE of the pre/post
# comparison). 1.0 ≈ a one-standard-deviation noise band.
DEFAULT_NOISE_Z = 1.0


@dataclass(frozen=True)
class MathExample:
    """One arithmetic problem with a reference solution."""

    operation: str
    solution: str

    def key(self) -> str:
        return self.operation.strip()


@dataclass
class EvalMetrics:
    """Accuracy summary for one policy on an eval sample."""

    effort: str
    total: int
    completed: int
    correct: int
    accuracy: float
    incomplete: int = 0
    model_path: str | None = None
    answers_csv: str | None = None
    scored_csv: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalMetrics:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DifferentialMetrics:
    """Instant vs high performance on the same eval sample."""

    sample_size: int
    instant: EvalMetrics
    high: EvalMetrics
    accuracy_gap: float | None  # high - instant; None when high eval is skipped
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "instant": self.instant.to_dict(),
            "high": self.high.to_dict(),
            "accuracy_gap": self.accuracy_gap,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DifferentialMetrics:
        return cls(
            sample_size=int(data["sample_size"]),
            instant=EvalMetrics.from_dict(data["instant"]),
            high=EvalMetrics.from_dict(data["high"]),
            accuracy_gap=data.get("accuracy_gap"),
            timestamp=str(data.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class LoopState:
    """Persisted state for resumable autoresearch runs."""

    run_dir: str
    seed_data: str
    train_data: str
    # Fixed eval sample drawn from the *train* split (in-distribution / can overfit).
    eval_sample: str
    # Full held-out test pool + fixed sample for evals (never used in training).
    heldout_test: str = ""
    eval_heldout_sample: str = ""
    heldout_fraction: float = 0.10
    iteration: int = 0
    policy_log_dirs: list[str] = field(default_factory=list)
    # Policy resume (weights) and sampler/judge (inference) from last train.
    last_policy_state_path: str | None = None
    last_policy_sampler_path: str | None = None
    # Explicit judge pointer (usually same as last_policy_sampler_path).
    last_judge_model_path: str | None = None
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    marginal_streak: int = 0
    stopped: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopState:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        # Tolerate older state.json missing newer fields.
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _atomic_text_writer(path: Path, *, newline: str | None = None):
    """Write a text file through a same-directory temp file and atomic replace."""
    ensure_dir(path.parent)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", newline=newline, encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


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
    with _atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["operation", "solution"])
        for ex in examples:
            writer.writerow([ex.operation, ex.solution])


def write_operations_csv(path: Path, operations: Sequence[str]) -> None:
    """Write operation-only CSV for ask_arithmetic input."""
    with _atomic_text_writer(path, newline="") as handle:
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
    # Import lazily so loop modules stay importable without full env for pure helpers.
    from test_arithmetic import evaluate_operation

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
    from test_arithmetic import extract_numeric_answer, numbers_equal

    exp = extract_numeric_answer(expected)
    act = extract_numeric_answer(actual)
    if exp is None or act is None:
        return expected.strip() == actual.strip()
    return numbers_equal(exp, act)


# ---------------------------------------------------------------------------
# Free-form model output parsing for data gen / validation
# ---------------------------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_EXAMPLE_BLOCK_RE = re.compile(
    r"<example>\s*operation:\s*(.+?)\s*solution:\s*(.+?)\s*</example>",
    re.IGNORECASE | re.DOTALL,
)
_OP_SOL_LINE_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])?\s*(.+?)\s*(?:=|,|->|=>)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def parse_generated_examples(text: str) -> list[MathExample]:
    """Extract (operation, solution) pairs from model generation text.

    Accepts, in order of preference:
      1. A JSON array of objects with operation/solution keys
      2. <example>operation: ... solution: ...</example> blocks
      3. Line-oriented ``expr = value`` or ``expr, value`` forms
    """
    text = text.strip()
    if not text:
        return []

    # 1) JSON array anywhere in the reply
    for match in _JSON_ARRAY_RE.finditer(text):
        blob = match.group(0)
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        examples: list[MathExample] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            op = str(item.get("operation") or item.get("expr") or "").strip()
            sol = str(item.get("solution") or item.get("answer") or "").strip()
            if op and sol:
                examples.append(
                    MathExample(operation=op, solution=canonicalize_solution_string(sol))
                )
        if examples:
            return examples

    # 2) Tagged blocks
    blocks = _EXAMPLE_BLOCK_RE.findall(text)
    if blocks:
        return [
            MathExample(
                operation=op.strip(),
                solution=canonicalize_solution_string(sol.strip()),
            )
            for op, sol in blocks
            if op.strip() and sol.strip()
        ]

    # 3) Line-oriented fallback
    examples = []
    for line in text.splitlines():
        m = _OP_SOL_LINE_RE.match(line)
        if not m:
            continue
        op, sol = m.group(1).strip(), m.group(2).strip()
        # Skip markdown tables / headers
        if op.lower() in {"operation", "expression", "expr"}:
            continue
        examples.append(
            MathExample(operation=op, solution=canonicalize_solution_string(sol))
        )
    return examples


_VERDICT_RE = re.compile(
    r"<verdict>\s*(keep|discard)\s*</verdict>",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r"<confidence>\s*(high|medium|low|\d+(?:\.\d+)?)\s*</confidence>",
    re.IGNORECASE,
)
_FIXED_SOLUTION_RE = re.compile(
    r"<solution>\s*([^\s<]+)\s*</solution>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationVerdict:
    keep: bool
    confidence: str
    solution: str | None
    raw: str


def parse_validation_verdict(text: str, fallback_solution: str = "") -> ValidationVerdict:
    """Parse keep/discard + confidence from a high-reasoning validator reply."""
    verdict_m = _VERDICT_RE.search(text)
    conf_m = _CONFIDENCE_RE.search(text)
    sol_m = _FIXED_SOLUTION_RE.search(text)

    keep = False
    if verdict_m is not None:
        keep = verdict_m.group(1).lower() == "keep"

    confidence = conf_m.group(1).lower() if conf_m else "low"
    solution = (
        canonicalize_solution_string(sol_m.group(1))
        if sol_m
        else (fallback_solution or None)
    )
    return ValidationVerdict(
        keep=keep, confidence=confidence, solution=solution, raw=text
    )


def confidence_is_high(confidence: str) -> bool:
    """True when the validator reported high confidence."""
    c = confidence.strip().lower()
    if c == "high":
        return True
    try:
        return float(c) >= 0.8
    except ValueError:
        return False


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
        from math_integration import run_test_arithmetic

        run_test_arithmetic(
            effort=effort,
            input_csv=answers_csv,
            output_csv=scored_csv,
        )
    else:
        from test_arithmetic import is_completed, is_correct

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


def run_python(args: Sequence[str], *, cwd: Path | None = None) -> None:
    """Run a python module/script; raise on non-zero exit."""
    cmd = [sys.executable, *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or LOOP_ROOT))


def save_json(path: Path, data: Any) -> None:
    payload = json.dumps(
        data,
        indent=2,
        sort_keys=False,
        allow_nan=False,
    ) + "\n"
    with _atomic_text_writer(path) as handle:
        handle.write(payload)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file with strict-JSON records."""
    lines = [json.dumps(record, allow_nan=False) for record in records]
    payload = "\n".join(lines) + ("\n" if lines else "")
    with _atomic_text_writer(path) as handle:
        handle.write(payload)


def load_state(path: Path) -> LoopState:
    return LoopState.from_dict(load_json(path))


def save_state(path: Path, state: LoopState) -> None:
    save_json(path, state.to_dict())


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


def is_marginal_improvement(
    history: Sequence[dict[str, Any]],
    *,
    min_delta: float,
    prefer_heldout: bool = True,
    noise_z: float = DEFAULT_NOISE_Z,
) -> bool:
    """True when the latest instant accuracy gain is not real progress.

    A gain counts as progress only when it exceeds
    ``max(min_delta, noise_z * SE)`` where SE is the binomial standard error
    of the accuracy difference, computed from the pre/post ``correct``/``total``
    counters when present (independent-proportions approximation, i.e. an
    upper bound for paired samples). Records without counters fall back to a
    plain ``min_delta`` comparison. With default settings (n=200, z=1.0) this
    keeps the stopping rule from being driven by eval sampling noise.

    Prefer **held-out** pre/post (``pre_heldout`` / ``post_heldout``) so early
    stopping is not driven by train-seed overfitting. Falls back to train-seed
    or legacy ``pre``/``post`` keys.
    """
    if len(history) < 1:
        return False
    last = history[-1]
    pre_blob, post_blob = _pick_pre_post_blobs(last, prefer_heldout=prefer_heldout)
    pre = _extract_instant_acc(pre_blob)
    post = _extract_instant_acc(post_blob)
    if pre is not None and post is not None:
        threshold = max(min_delta, _noise_band(pre_blob, post_blob, noise_z=noise_z))
        return (post - pre) < threshold

    # Else compare last two post-eval accuracies across iterations.
    if len(history) < 2:
        return False
    prev = history[-2]
    _, prev_post = _pick_pre_post_blobs(prev, prefer_heldout=prefer_heldout)
    a_blob = (
        prev_post
        if prev_post is not None
        else (prev.get("post") or prev.get("post_eval") or prev)
    )
    b_blob = (
        post_blob
        if post_blob is not None
        else (last.get("post") or last.get("post_eval") or last)
    )
    a = _extract_instant_acc(a_blob)
    b = _extract_instant_acc(b_blob)
    if a is None or b is None:
        return False
    threshold = max(min_delta, _noise_band(a_blob, b_blob, noise_z=noise_z))
    return (b - a) < threshold


def marginal_improvement_streak(
    history: Sequence[dict[str, Any]],
    *,
    min_delta: float,
    noise_z: float = DEFAULT_NOISE_Z,
) -> int:
    """Recompute the trailing marginal streak from durable metrics history."""
    streak = 0
    for end in range(len(history), 0, -1):
        if not is_marginal_improvement(
            history[:end], min_delta=min_delta, noise_z=noise_z
        ):
            break
        streak += 1
    return streak


def _pick_pre_post_blobs(
    record: dict[str, Any],
    *,
    prefer_heldout: bool,
) -> tuple[Any, Any]:
    """Pick the (pre, post) instant-metric blobs for a history record.

    A pair is only returned when **both** sides carry an accuracy, so a
    half-missing pair cleanly falls back to the next source instead of
    crashing the delta computation.
    """
    heldout = record.get("heldout") if isinstance(record.get("heldout"), dict) else {}
    train_seed = (
        record.get("train_seed") if isinstance(record.get("train_seed"), dict) else {}
    )

    candidates: list[tuple[Any, Any]] = []
    if prefer_heldout:
        candidates.append(
            (
                record.get("pre_heldout") or heldout.get("pre"),
                record.get("post_heldout") or heldout.get("post"),
            )
        )
    candidates.append(
        (
            record.get("pre_train")
            or record.get("pre")
            or record.get("pre_eval")
            or train_seed.get("pre"),
            record.get("post_train")
            or record.get("post")
            or record.get("post_eval")
            or train_seed.get("post"),
        )
    )
    for pre_blob, post_blob in candidates:
        if (
            _extract_instant_acc(pre_blob) is not None
            and _extract_instant_acc(post_blob) is not None
        ):
            return pre_blob, post_blob
    return None, None


def _extract_instant_counts(blob: Any) -> tuple[int, int] | None:
    """(correct, total) for an instant metrics blob, when recorded."""
    if not isinstance(blob, dict):
        return None
    instant = blob.get("instant")
    if isinstance(instant, dict):
        blob = instant
    correct = blob.get("correct")
    total = blob.get("total")
    if (
        isinstance(correct, (int, float))
        and isinstance(total, (int, float))
        and total > 0
    ):
        return int(correct), int(total)
    return None


def _noise_band(pre_blob: Any, post_blob: Any, *, noise_z: float) -> float:
    """Binomial standard error of the accuracy difference times ``noise_z``.

    Uses the recorded correct/total counters; returns 0.0 (no band) when
    either side lacks counters or ``noise_z`` is not positive.
    """
    if noise_z <= 0:
        return 0.0
    pre_counts = _extract_instant_counts(pre_blob)
    post_counts = _extract_instant_counts(post_blob)
    if pre_counts is None or post_counts is None:
        return 0.0
    c_pre, n_pre = pre_counts
    c_post, n_post = post_counts
    p_pre = c_pre / n_pre
    p_post = c_post / n_post
    se = math.sqrt(
        p_pre * (1.0 - p_pre) / n_pre + p_post * (1.0 - p_post) / n_post
    )
    return noise_z * se


def _extract_instant_acc(blob: Any) -> float | None:
    if blob is None:
        return None
    if isinstance(blob, (int, float)):
        return float(blob)
    if not isinstance(blob, dict):
        return None
    if "instant" in blob and isinstance(blob["instant"], dict):
        acc = blob["instant"].get("accuracy")
        return float(acc) if acc is not None else None
    if "accuracy" in blob:
        return float(blob["accuracy"])
    if "instant_accuracy" in blob:
        return float(blob["instant_accuracy"])
    return None


def format_seed_block(examples: Iterable[MathExample], limit: int = 20) -> str:
    """Human-readable sample block for generation prompts."""
    lines = []
    for i, ex in enumerate(examples):
        if i >= limit:
            break
        lines.append(f"- operation: {ex.operation} | solution: {ex.solution}")
    return "\n".join(lines)
