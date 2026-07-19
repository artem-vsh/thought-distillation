#!/usr/bin/env python3
"""Unit tests for loop helpers (no network / Tinker)."""

from __future__ import annotations

from pathlib import Path

import pytest

from common import (
    MathExample,
    canonicalize_solution_string,
    confidence_is_high,
    is_marginal_improvement,
    marginal_improvement_streak,
    merge_unique_examples,
    parse_generated_examples,
    parse_validation_verdict,
    programmatic_solution,
    sample_examples,
    solutions_match,
    write_math_csv,
    load_math_csv,
)


def test_parse_generated_json_array() -> None:
    text = """
    Here are problems:
    [
      {"operation": "2+2", "solution": "4"},
      {"operation": "3*5", "solution": "15"}
    ]
    """
    examples = parse_generated_examples(text)
    assert examples == [
        MathExample("2+2", "4"),
        MathExample("3*5", "15"),
    ]


def test_parse_generated_tagged_blocks() -> None:
    text = """
    <example>
    operation: 10-3
    solution: 7
    </example>
    <example>
    operation: gcd(12,8)
    solution: 4
    </example>
    """
    examples = parse_generated_examples(text)
    assert len(examples) == 2
    assert examples[0].operation == "10-3"
    assert examples[0].solution == "7"


def test_parse_generated_line_oriented() -> None:
    text = "1. 6*7 = 42\n- fib(7) -> 13\n"
    examples = parse_generated_examples(text)
    assert MathExample("6*7", "42") in examples
    assert MathExample("fib(7)", "13") in examples


def test_parse_validation_verdict() -> None:
    text = (
        "Looks correct.\n"
        "<verdict>keep</verdict>\n"
        "<confidence>high</confidence>\n"
        "<solution>42</solution>\n"
    )
    v = parse_validation_verdict(text)
    assert v.keep is True
    assert confidence_is_high(v.confidence)
    assert v.solution == "42"


def test_parse_validation_discard() -> None:
    v = parse_validation_verdict(
        "<verdict>discard</verdict><confidence>low</confidence>"
    )
    assert v.keep is False
    assert not confidence_is_high(v.confidence)


def test_confidence_numeric() -> None:
    assert confidence_is_high("0.9")
    assert not confidence_is_high("0.5")
    assert not confidence_is_high("medium")


def test_programmatic_solution_basic() -> None:
    assert programmatic_solution("2+2") == "4"
    assert programmatic_solution("gcd(48,18)") == "6"
    assert programmatic_solution("not a valid $$$") is None


def test_solutions_match() -> None:
    assert solutions_match("4", "4.0")
    assert solutions_match("+42", "42")
    assert not solutions_match("4", "5")


def test_canonicalize_solution_string() -> None:
    assert canonicalize_solution_string("4.0") == "4"
    assert canonicalize_solution_string("+12") == "12"


def test_merge_unique_examples() -> None:
    base = [MathExample("1+1", "2")]
    extra = [MathExample("1+1", "2"), MathExample("2+2", "4")]
    merged = merge_unique_examples(base, extra)
    assert merged == [MathExample("1+1", "2"), MathExample("2+2", "4")]


def test_sample_examples_deterministic() -> None:
    pool = [MathExample(str(i), str(i)) for i in range(20)]
    a = sample_examples(pool, 5, seed=3)
    b = sample_examples(pool, 5, seed=3)
    assert a == b
    assert len(a) == 5


def test_write_load_math_csv(tmp_path: Path) -> None:
    path = tmp_path / "ops.csv"
    examples = [MathExample("1+1", "2"), MathExample("3*3", "9")]
    write_math_csv(path, examples)
    loaded = load_math_csv(path)
    assert loaded == examples


def test_is_marginal_improvement_within_iter() -> None:
    history = [
        {
            "pre": {"instant": {"accuracy": 0.10}},
            "post": {"instant": {"accuracy": 0.101}},
        }
    ]
    assert is_marginal_improvement(history, min_delta=0.005) is True
    history[0]["post"]["instant"]["accuracy"] = 0.20
    assert is_marginal_improvement(history, min_delta=0.005) is False


def test_is_marginal_improvement_across_iters() -> None:
    history = [
        {"post": {"instant": {"accuracy": 0.20}}},
        {"post": {"instant": {"accuracy": 0.201}}},
    ]
    assert is_marginal_improvement(history, min_delta=0.005) is True


def test_score_answers_csv_inprocess(tmp_path: Path) -> None:
    from common import score_answers_csv

    answers = tmp_path / "answers.csv"
    scored = tmp_path / "scored.csv"
    answers.write_text(
        "operation,output\n2+2,4\n3*3,8\n4*4,\n",
        encoding="utf-8",
    )
    metrics = score_answers_csv(answers, scored, effort="instant", use_cli=False)
    assert metrics.total == 3
    assert metrics.completed == 2
    assert metrics.incomplete == 1
    assert metrics.correct == 1
    assert abs(metrics.accuracy - (1 / 3)) < 1e-9


def test_save_json_rejects_nan_without_overwriting(tmp_path: Path) -> None:
    from common import load_json, save_json

    path = tmp_path / "strict.json"
    save_json(path, {"status": "good"})
    with pytest.raises(ValueError):
        save_json(path, {"gap": float("nan")})
    assert load_json(path) == {"status": "good"}


def test_default_seed_points_at_loop_data() -> None:
    from common import DEFAULT_SEED_DATA
    from math_integration import LOOP_SEED_DATA, default_seed_data

    assert DEFAULT_SEED_DATA.is_file()
    assert default_seed_data().is_file()
    assert LOOP_SEED_DATA.is_file()
    assert DEFAULT_SEED_DATA.resolve() == LOOP_SEED_DATA.resolve()


def test_math_integration_manifest_keys() -> None:
    from math_integration import math_model_defaults

    d = math_model_defaults()
    assert d["policy_renderer"] == "gpt_oss_no_sysprompt"
    assert d["judge_renderer"] == "gpt_oss_high_reasoning"
    assert "ask_arithmetic.py" in d["scripts"]["ask"]
    assert "test_arithmetic.py" in d["scripts"]["test"]
    assert "train_math_llm_judge.py" in d["scripts"]["train"]


def test_progress_tracker_writes_status(tmp_path: Path) -> None:
    from progress import ProgressTracker

    tracker = ProgressTracker(tmp_path)
    tracker.set_phase("generate", "working", iteration=0)
    tracker.snapshot_train_pool(10, source="test")
    assert (tmp_path / "status.json").is_file()
    assert (tmp_path / "events.jsonl").is_file()
    status = tracker.status
    assert status.phase == "generate"
    assert status.train_pool_size == 10


def test_progress_checkpoint_index_is_idempotent(tmp_path: Path) -> None:
    from progress import ProgressTracker

    tracker = ProgressTracker(tmp_path / "run")
    train_log = tmp_path / "train"
    train_log.mkdir()
    (train_log / "checkpoints.jsonl").write_text(
        '{"name":"000020","batch":20,"state_path":"state","sampler_path":"sampler"}\n',
        encoding="utf-8",
    )
    tracker.index_train_log(train_log, iteration=0)
    with tracker.checkpoints_path.open("a", encoding="utf-8") as handle:
        handle.write('{"torn":')
    tracker.index_train_log(train_log, iteration=0)

    records = tracker.checkpoints_path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1


def test_split_train_heldout_disjoint() -> None:
    from common import (
        assert_disjoint_train_heldout,
        filter_out_operations,
        operation_keys,
        split_train_heldout,
    )

    pool = [MathExample(f"op{i}", str(i)) for i in range(100)]
    train, heldout = split_train_heldout(pool, heldout_fraction=0.1, seed=0)
    assert len(train) + len(heldout) == 100
    assert len(heldout) == 10
    assert_disjoint_train_heldout(train, heldout)
    # Same seed → same split
    t2, h2 = split_train_heldout(pool, heldout_fraction=0.1, seed=0)
    assert [e.operation for e in heldout] == [e.operation for e in h2]
    # Guard filter
    leaked = train + heldout[:1]
    cleaned = filter_out_operations(leaked, operation_keys(heldout))
    assert len(cleaned) == len(train)
    assert operation_keys(cleaned).isdisjoint(operation_keys(heldout))


def test_canonical_operation_key() -> None:
    from common import canonical_operation_key

    equivalent = [
        ("12+34", "34+12"),  # commuted addition
        ("12 + 34", "(34+12)"),  # whitespace / parentheses
        ("2^10", "2**10"),  # alternate power syntax
        ("gcd(48,18)", "gcd(18,48)"),  # commutative function args
        ("3*4.0", "4*3"),  # numeric formatting + commuted multiply
        ("Fib(7)", "fib(7)"),  # case-insensitive function name
        ("5!", "5 !"),  # fallback: whitespace-insensitive
    ]
    for a, b in equivalent:
        assert canonical_operation_key(a) == canonical_operation_key(b), (a, b)

    distinct = [
        ("12-34", "34-12"),  # subtraction is not commutative
        ("12+34", "12+35"),
        ("comb(5,2)", "comb(2,5)"),  # non-commutative function args
        ("2^10", "10^2"),
    ]
    for a, b in distinct:
        assert canonical_operation_key(a) != canonical_operation_key(b), (a, b)


def test_heldout_guard_catches_canonical_leaks() -> None:
    from common import (
        assert_disjoint_train_heldout,
        filter_out_operations,
        merge_unique_examples,
        operation_keys,
    )

    heldout = [MathExample("12+34", "46"), MathExample("gcd(48,18)", "6")]
    banned = operation_keys(heldout)

    pool = [
        MathExample("34 + 12", "46"),  # canonical dup of held-out → dropped
        MathExample("gcd(48,20)", "4"),  # different args → kept
        MathExample("7*8", "56"),
    ]
    cleaned = filter_out_operations(pool, banned)
    assert [ex.operation for ex in cleaned] == ["gcd(48,20)", "7*8"]

    with pytest.raises(ValueError, match="leak"):
        assert_disjoint_train_heldout(
            [MathExample("2**10", "1024")],
            [MathExample("2^10", "1024")],
        )

    merged = merge_unique_examples(
        [MathExample("12+34", "46")],
        [MathExample("34+12", "46"), MathExample("5*5", "25")],
    )
    assert merged == [MathExample("12+34", "46"), MathExample("5*5", "25")]


def test_marginal_prefers_heldout() -> None:
    history = [
        {
            "pre_heldout": {"instant": {"accuracy": 0.10}},
            "post_heldout": {"instant": {"accuracy": 0.101}},
            "pre_train": {"instant": {"accuracy": 0.10}},
            "post_train": {"instant": {"accuracy": 0.50}},  # huge train gain
        }
    ]
    # Held-out gain is marginal even though train-seed jumped.
    assert is_marginal_improvement(history, min_delta=0.005, prefer_heldout=True)
    assert not is_marginal_improvement(history, min_delta=0.005, prefer_heldout=False)


def test_marginal_respects_noise_band() -> None:
    """With counters recorded, gains inside the noise band count as marginal."""
    history = [
        {
            # +1.5 points on n=200: above min_delta but well within binomial noise.
            "pre_heldout": {
                "instant": {"accuracy": 0.50, "correct": 100, "total": 200}
            },
            "post_heldout": {
                "instant": {"accuracy": 0.515, "correct": 103, "total": 200}
            },
        }
    ]
    assert is_marginal_improvement(history, min_delta=0.005, noise_z=1.0)
    # noise_z=0 disables the band: raw delta (0.015) > min_delta → progress.
    assert not is_marginal_improvement(history, min_delta=0.005, noise_z=0.0)
    # A +12-point jump clears the band and counts as real progress.
    history[0]["post_heldout"]["instant"] = {
        "accuracy": 0.62,
        "correct": 124,
        "total": 200,
    }
    assert not is_marginal_improvement(history, min_delta=0.005, noise_z=1.0)


def test_marginal_handles_half_missing_pair() -> None:
    """A pre-only record must not crash the delta computation."""
    history = [
        {
            "pre_heldout": {"instant": {"accuracy": 0.10}},
            "post_heldout": {"instant": {"accuracy": 0.20}},
        },
        {
            # Held-out pair half missing; only a legacy flat post available.
            "pre_heldout": {"instant": {"accuracy": 0.20}},
            "post": {"instant": {"accuracy": 0.201}},
        },
    ]
    # Falls back to cross-iteration post comparison; no TypeError.
    assert is_marginal_improvement(history, min_delta=0.005) is True


def test_marginal_streak_is_recomputed_from_history() -> None:
    history = [
        {
            "pre_heldout": {"instant": {"accuracy": 0.10}},
            "post_heldout": {"instant": {"accuracy": 0.20}},
        },
        {
            "pre_heldout": {"instant": {"accuracy": 0.20}},
            "post_heldout": {"instant": {"accuracy": 0.202}},
        },
        {
            "pre_heldout": {"instant": {"accuracy": 0.202}},
            "post_heldout": {"instant": {"accuracy": 0.203}},
        },
    ]
    assert marginal_improvement_streak(history, min_delta=0.005) == 2


def test_train_argv_auto_switch_base_vs_checkpoint() -> None:
    from pathlib import Path

    from train_step import build_train_argv

    base_argv = build_train_argv(
        data_path=Path("data.csv"),
        log_path=Path("log"),
        load_checkpoint=None,
        judge_model_path=None,
        model="openai/gpt-oss-20b",
        max_steps=2,
        groups_per_batch=4,
        group_size=2,
        save_every=2,
        eval_every=2,
        learning_rate=1e-5,
        lora_rank=32,
        seed=0,
        behavior_if_log_dir_exists="delete",
    )
    assert not any(a.startswith("load_checkpoint_path=") for a in base_argv)
    assert not any(a.startswith("judge_model_path=") for a in base_argv)

    ckpt_argv = build_train_argv(
        data_path=Path("data.csv"),
        log_path=Path("log"),
        load_checkpoint="tinker://run/weights/000020",
        judge_model_path="tinker://run/sampler_weights/000020",
        model="openai/gpt-oss-20b",
        max_steps=2,
        groups_per_batch=4,
        group_size=2,
        save_every=2,
        eval_every=2,
        learning_rate=1e-5,
        lora_rank=32,
        seed=0,
        behavior_if_log_dir_exists="delete",
    )
    assert "load_checkpoint_path=tinker://run/weights/000020" in ckpt_argv
    assert "judge_model_path=tinker://run/sampler_weights/000020" in ckpt_argv
    assert "judge_renderer_name=gpt_oss_high_reasoning" in ckpt_argv


def test_programmatic_solution_bounds_pathological_inputs() -> None:
    # Capped functions still work on normal arguments.
    assert programmatic_solution("sigma(12)") == "28"
    assert programmatic_solution("phi(10)") == "4"
    assert programmatic_solution("comb(10,3)") == "120"
    # Unbounded-cost arguments are rejected quickly instead of hanging
    # (sigma/phi trial-divide in O(√n); comb/perm grow combinatorially).
    big = "9" * 30
    assert programmatic_solution(f"sigma({big})") is None
    assert programmatic_solution(f"phi({big})") is None
    assert programmatic_solution(f"comb({big},2)") is None
    assert programmatic_solution(f"perm({big},2)") is None
    # Oversized operations (e.g. digit ops on thousand-digit literals) are
    # refused outright by the length guard.
    assert programmatic_solution("1+" * 150 + "1") is None
