"""Integration anchors for local math pipeline scripts.

Shells out to (or imports from) vendored modules in this project:

  - ``ask_arithmetic.py``     — sampling (instant / high / …)
  - ``test_arithmetic.py``    — programmatic scoring of answers
  - ``train_math_llm_judge.py`` — Tinker RL (instant policy, high judge)
  - ``data/arithmetic_operations.csv`` — seed dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import (
    MathExample,
    REPO_ROOT,
    ensure_dir,
    run_python,
    save_json,
)

# ---------------------------------------------------------------------------
# Script paths (this project root)
# ---------------------------------------------------------------------------

ASK_ARITHMETIC_SCRIPT = REPO_ROOT / "ask_arithmetic.py"
TEST_ARITHMETIC_SCRIPT = REPO_ROOT / "test_arithmetic.py"
TRAIN_MATH_LLM_JUDGE_SCRIPT = REPO_ROOT / "train_math_llm_judge.py"

# Local seed (copy of parent data/arithmetic_operations.csv)
LOOP_DATA_DIR = Path(__file__).resolve().parent / "data"
LOOP_SEED_DATA = LOOP_DATA_DIR / "arithmetic_operations.csv"
REPO_SEED_DATA = REPO_ROOT / "data" / "arithmetic_operations.csv"


def default_seed_data() -> Path:
    """Prefer project data/ seed CSV."""
    if LOOP_SEED_DATA.is_file():
        return LOOP_SEED_DATA
    return REPO_SEED_DATA


# ---------------------------------------------------------------------------
# Shared constants from the math scripts (single source of truth)
# ---------------------------------------------------------------------------

def math_model_defaults() -> dict[str, Any]:
    """Pull model / renderer defaults from ask + train scripts."""
    from ask_arithmetic import (
        FINAL_CHANNEL_PREFILL,
        INPUT_CSV,
        MODEL,
        RENDERER_BY_EFFORT,
    )
    from train_math_llm_judge import (
        DEFAULT_JUDGE_MAX_TOKENS,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_JUDGE_RENDERER,
        DEFAULT_POLICY_MODEL,
        DEFAULT_POLICY_PREFILL,
        DEFAULT_POLICY_RENDERER,
        DEFAULT_POLICY_REASONING_EFFORT,
    )

    return {
        "ask_model": MODEL,
        "ask_input_csv": str(INPUT_CSV),
        "renderer_by_effort": dict(RENDERER_BY_EFFORT),
        "final_channel_prefill": FINAL_CHANNEL_PREFILL,
        "policy_model": DEFAULT_POLICY_MODEL,
        "policy_renderer": DEFAULT_POLICY_RENDERER,
        "policy_prefill": DEFAULT_POLICY_PREFILL,
        "policy_reasoning_effort": DEFAULT_POLICY_REASONING_EFFORT,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "judge_renderer": DEFAULT_JUDGE_RENDERER,
        "judge_max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
        "loop_seed_data": str(default_seed_data()),
        "repo_seed_data": str(REPO_SEED_DATA),
        "scripts": {
            "ask": str(ASK_ARITHMETIC_SCRIPT),
            "test": str(TEST_ARITHMETIC_SCRIPT),
            "train": str(TRAIN_MATH_LLM_JUDGE_SCRIPT),
        },
    }


def load_train_problems(path: Path) -> list[MathExample]:
    """Load train CSV via ``train_math_llm_judge.load_math_problems``.

    Ensures the same schema/validation the trainer uses (both columns required
    and non-empty).
    """
    from train_math_llm_judge import load_math_problems

    problems = load_math_problems(Path(path))
    return [MathExample(operation=p.operation, solution=p.solution) for p in problems]


def make_math_prompt(operation: str) -> str:
    """Same user prompt as ask_arithmetic / train_math_llm_judge."""
    from ask_arithmetic import make_prompt

    return make_prompt(operation)


def run_ask_arithmetic(
    *,
    effort: str,
    input_csv: Path,
    output_csv: Path,
    model: str,
    model_path: str | None = None,
) -> None:
    """Shell out to ``ask_arithmetic.py`` (same CLI as manual runs)."""
    flag = {
        "instant": "--instant",
        "low": "--low",
        "medium": "--medium",
        "high": "--high",
    }[effort]
    args = [
        str(ASK_ARITHMETIC_SCRIPT),
        flag,
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
        "--model",
        model,
    ]
    if model_path:
        args.extend(["--model-path", model_path])
    run_python(args)


def run_test_arithmetic(
    *,
    effort: str,
    input_csv: Path,
    output_csv: Path,
) -> None:
    """Shell out to ``test_arithmetic.py`` (same CLI as manual runs)."""
    flag = {
        "instant": "--instant",
        "low": "--low",
        "medium": "--medium",
        "high": "--high",
    }[effort]
    args = [
        str(TEST_ARITHMETIC_SCRIPT),
        flag,
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
    ]
    run_python(args)


def write_integration_manifest(path: Path) -> dict[str, Any]:
    """Write a small JSON describing how the loop is wired to math scripts."""
    manifest = math_model_defaults()
    ensure_dir(path.parent)
    save_json(path, manifest)
    return manifest


def check_model_consistency(model: str) -> list[str]:
    """Warn when the loop's ``--model`` diverges from the math scripts.

    The loop keeps its own model default (``common.DEFAULT_MODEL``) so it
    stays importable without the training stack; this check makes any drift
    from the scripts' constants loud at run start instead of silent.
    Best-effort: returns [] when the scripts cannot be imported.
    """
    try:
        defaults = math_model_defaults()
    except Exception:
        return []
    warnings: list[str] = []
    for key, label in (
        ("ask_model", "ask_arithmetic.MODEL"),
        ("policy_model", "train_math_llm_judge.DEFAULT_POLICY_MODEL"),
    ):
        script_value = defaults.get(key)
        if script_value and script_value != model:
            warnings.append(
                f"--model {model!r} differs from {label}={script_value!r}; "
                "evals/gen/train will use --model — verify this is intentional"
            )
    return warnings
