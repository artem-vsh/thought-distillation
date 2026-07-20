#!/usr/bin/env python3
"""Unit tests for math LLM-judge training helpers (no network)."""

from __future__ import annotations

from pathlib import Path

from mathtask.ask_arithmetic import FINAL_CHANNEL_PREFILL, RENDERER_BY_EFFORT, make_prompt
from mathtask.train_math_llm_judge import (
    DEFAULT_POLICY_PREFILL,
    DEFAULT_POLICY_RENDERER,
    MathProblem,
    _split_problems,
    extract_judge_score,
    load_math_problems,
    make_judge_messages,
)


def test_make_prompt_matches_processor() -> None:
    assert make_prompt("12+3") == "What is the result of 12+3? Give only the result."


def test_policy_defaults_match_ask_instant() -> None:
    """Trained policy should match ask_arithmetic --instant (no CoT)."""
    assert DEFAULT_POLICY_RENDERER == RENDERER_BY_EFFORT["instant"]
    assert DEFAULT_POLICY_RENDERER == "gpt_oss_no_sysprompt"
    assert DEFAULT_POLICY_PREFILL == FINAL_CHANNEL_PREFILL
    assert DEFAULT_POLICY_PREFILL == "<|channel|>final<|message|>"


def test_extract_judge_score_tagged() -> None:
    assert extract_judge_score("Reasoning...\n<score>1</score>") == 1.0
    assert extract_judge_score("<score>0</score>") == 0.0
    assert extract_judge_score("<score>0.5</score>") == 0.5


def test_extract_judge_score_yes_no_fallback() -> None:
    assert extract_judge_score("The final decision is yes.") == 1.0
    assert extract_judge_score("I conclude the answer is incorrect.") == 0.0


def test_extract_judge_score_empty() -> None:
    assert extract_judge_score("") == 0.0
    assert extract_judge_score("no clear decision") == 0.0


def test_make_judge_messages_has_no_reference_solution() -> None:
    # Judge sees only the problem and the policy's answer; a distinctive
    # wrong answer stands in for a value that must NOT be echoed as a
    # reference solution anywhere in the prompt.
    messages = make_judge_messages("2+2", "417")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert make_prompt("2+2") in content
    assert "Expression: 2+2" in content
    assert "Model answer: 417" in content
    # The answer appears exactly once (as the model answer, never as a
    # reference), and no reference-solution label is present.
    assert content.count("417") == 1
    assert "Reference solution" not in content
    assert "<score>" in content
    assert "independently" in content


def test_load_math_problems(tmp_path: Path) -> None:
    csv_path = tmp_path / "ops.csv"
    csv_path.write_text("operation,solution\n1+1,2\n3*4,12\n", encoding="utf-8")
    problems = load_math_problems(csv_path)
    assert problems == [
        MathProblem(operation="1+1", solution="2"),
        MathProblem(operation="3*4", solution="12"),
    ]


def test_split_problems_deterministic() -> None:
    problems = [MathProblem(operation=str(i), solution=str(i)) for i in range(20)]
    a_train, a_test = _split_problems(problems, test_fraction=0.2, seed=7)
    b_train, b_test = _split_problems(problems, test_fraction=0.2, seed=7)
    assert a_train == b_train
    assert a_test == b_test
    assert len(a_train) + len(a_test) == 20
    assert len(a_test) >= 1
    assert len(a_train) >= 1


def test_create_tinker_judge_accepts_model_path_kwarg() -> None:
    """Judge can take an optional checkpoint path (no network call here)."""
    import inspect

    from mathtask.train_math_llm_judge import create_tinker_judge

    sig = inspect.signature(create_tinker_judge)
    assert "judge_model_path" in sig.parameters
