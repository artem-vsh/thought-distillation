"""Offline regression tests for autoresearch run/resume safety."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import autoresearch
from common import (
    DifferentialMetrics,
    EvalMetrics,
    MathExample,
    ensure_dir,
    load_json,
    save_json,
    write_math_csv,
)


def _seed_csv(path: Path) -> Path:
    write_math_csv(
        path,
        [MathExample(f"{i}+1", str(i + 1)) for i in range(10)],
    )
    return path


def _diff(accuracy: float, model_path: str | None = None) -> DifferentialMetrics:
    total = 2
    correct = int(accuracy * total)
    return DifferentialMetrics(
        sample_size=total,
        instant=EvalMetrics(
            effort="instant",
            total=total,
            completed=total,
            correct=correct,
            accuracy=correct / total,
            model_path=model_path,
        ),
        high=EvalMetrics(
            effort="high",
            total=total,
            completed=0,
            correct=0,
            accuracy=0.0,
            incomplete=total,
            model_path=model_path,
        ),
        accuracy_gap=None,
    )


def test_new_run_refuses_existing_name(tmp_path: Path) -> None:
    seed = _seed_csv(tmp_path / "seed.csv")
    runs_root = tmp_path / "runs"
    autoresearch.init_run(
        runs_root=runs_root,
        run_name="same-name",
        seed_data=seed,
        resume_dir=None,
        eval_sample_size=2,
    )

    with pytest.raises(FileExistsError, match="Use --resume"):
        autoresearch.init_run(
            runs_root=runs_root,
            run_name="same-name",
            seed_data=seed,
            resume_dir=None,
            eval_sample_size=2,
        )


def test_resume_restores_saved_config_and_audits_explicit_override(
    tmp_path: Path,
) -> None:
    run_dir = ensure_dir(tmp_path / "run")
    original = autoresearch.parse_args(
        [
            "--model",
            "saved/model",
            "--max-iters",
            "7",
            "--train-max-steps",
            "9",
            "--skip-train",
        ]
    )
    original_payload = autoresearch._serialize_args(original)
    save_json(run_dir / "config.json", original_payload)

    resumed = autoresearch.parse_args(
        [
            "--resume",
            str(run_dir),
            "--max-iters",
            "2",
            "--no-skip-train",
        ]
    )
    effective, audit = autoresearch.resolve_resume_config(
        resumed,
        argv=[
            "--resume",
            str(run_dir),
            "--max-iters",
            "2",
            "--no-skip-train",
        ],
    )

    assert effective.model == "saved/model"
    assert effective.train_max_steps == 9
    assert effective.max_iters == 2
    assert effective.skip_train is False
    assert audit is not None
    assert audit["overrides"]["max_iters"] == {"previous": 7, "effective": 2}
    assert audit["overrides"]["skip_train"] == {
        "previous": True,
        "effective": False,
    }
    assert load_json(run_dir / "config.json") == original_payload

    incompatible = autoresearch.parse_args(
        ["--resume", str(run_dir), "--eval-sample-size", "3"]
    )
    with pytest.raises(ValueError, match="eval_sample_size is fixed"):
        autoresearch.resolve_resume_config(
            incompatible,
            argv=["--resume", str(run_dir), "--eval-sample-size", "3"],
        )


def test_resume_detects_abbreviated_and_ignored_options(tmp_path: Path) -> None:
    """Argparse abbreviations must register as explicit overrides on resume."""
    run_dir = ensure_dir(tmp_path / "run")
    original = autoresearch.parse_args(["--max-iters", "7"])
    save_json(run_dir / "config.json", autoresearch._serialize_args(original))

    argv = ["--resume", str(run_dir), "--max-it", "2", "--seed-data", "x.csv"]
    resumed = autoresearch.parse_args(argv)
    effective, audit = autoresearch.resolve_resume_config(resumed, argv=argv)

    # "--max-it" is an unambiguous abbreviation of --max-iters: it must win.
    assert effective.max_iters == 2
    assert audit is not None
    assert audit["overrides"]["max_iters"] == {"previous": 7, "effective": 2}
    # --seed-data has no effect on a resumed run; it is reported, not silent.
    assert "seed_data" in audit["ignored_fields"]


def test_stopped_run_requires_force_to_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _seed_csv(tmp_path / "seed.csv")
    run_dir, state, _progress = autoresearch.init_run(
        runs_root=tmp_path / "runs",
        run_name="stopped-run",
        seed_data=seed,
        resume_dir=None,
        eval_sample_size=2,
    )
    # Original config is written by main() on fresh runs; resume needs it.
    save_json(
        run_dir / "config.json",
        autoresearch._serialize_args(autoresearch.parse_args(["--max-iters", "2"])),
    )
    state.stopped = True
    state.stop_reason = "test stop"
    from common import save_state

    save_state(run_dir / "state.json", state)

    iterations: list[int] = []

    def fake_run_iteration(state, progress, **kwargs):
        iterations.append(state.iteration)
        state.iteration += 1
        return state

    monkeypatch.setattr(autoresearch, "run_iteration", fake_run_iteration)
    monkeypatch.setattr(autoresearch, "check_model_consistency", lambda model: [])

    # Without --force, resuming a stopped run is an explicit no-op.
    autoresearch.main(["--resume", str(run_dir)])
    assert iterations == []

    # With --force, the stop decision is cleared and iterations continue.
    autoresearch.main(["--resume", str(run_dir), "--force"])
    assert iterations == [0, 1]


def test_run_lock_rejects_concurrent_process(tmp_path: Path) -> None:
    import fcntl
    import os

    run_dir = ensure_dir(tmp_path / "locked-run")
    # Simulate another process holding the run lock (flock conflicts apply
    # across open file descriptions, even within one process).
    fd = os.open(run_dir / ".autoresearch.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="locked by another"):
            autoresearch._acquire_run_lock(run_dir)
    finally:
        os.close(fd)

    # Re-entrant acquisition in the same process is a no-op once held.
    autoresearch._acquire_run_lock(run_dir)
    autoresearch._acquire_run_lock(run_dir)


def test_completed_iteration_phases_are_reused_after_state_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _seed_csv(tmp_path / "seed.csv")
    run_dir, state, progress = autoresearch.init_run(
        runs_root=tmp_path / "runs",
        run_name="recoverable",
        seed_data=seed,
        resume_dir=None,
        eval_sample_size=2,
    )
    eval_calls: list[str] = []
    train_calls: list[str] = []

    def fake_dual_evals(**kwargs):
        tag = kwargs["tag"]
        iter_dir = kwargs["iter_dir"]
        eval_calls.append(tag)
        train_diff = _diff(0.5)
        heldout_diff = _diff(0.5)
        for split, diff in (
            ("train_seed", train_diff),
            ("heldout", heldout_diff),
        ):
            out_dir = ensure_dir(iter_dir / f"eval_{tag}" / split)
            save_json(out_dir / "differential.json", diff.to_dict())
        return train_diff, heldout_diff

    def fake_train_step(**kwargs):
        train_calls.append(kwargs["behavior_if_log_dir_exists"])
        meta = {
            "state_path": "tinker://test/weights/20",
            "sampler_path": "tinker://test/sampler/20",
            "policy_source": "base",
            "judge_source": "base",
        }
        save_json(kwargs["log_path"] / "train_step_meta.json", meta)
        return meta

    monkeypatch.setattr(autoresearch, "_run_dual_evals", fake_dual_evals)
    monkeypatch.setattr(autoresearch, "run_train_step", fake_train_step)

    kwargs = {
        "model": "openai/gpt-oss-20b",
        "gen_target": 2,
        "seed_samples": 2,
        "variations_per_batch": 1,
        "seeds_per_batch": 1,
        "eval_sample_size": 2,
        "train_max_steps": 2,
        "groups_per_batch": 1,
        "group_size": 1,
        "save_every": 1,
        "eval_every": 1,
        "learning_rate": 1e-5,
        "lora_rank": 1,
        "rng_seed": 0,
        "skip_generate": True,
        "skip_train": False,
        "skip_high_eval": True,
        "gen_temperature": 0.0,
    }
    state = autoresearch.run_iteration(state, progress, **kwargs)
    assert eval_calls == ["pre", "post"]
    assert train_calls == ["resume"]

    # Simulate a crash after artifacts/history were committed but before the
    # caller durably advanced state.json.
    state.iteration = 0
    with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"torn":')
    state = autoresearch.run_iteration(state, progress, **kwargs)

    assert state.iteration == 1
    assert eval_calls == ["pre", "post"]
    assert train_calls == ["resume"]
    history_lines = (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1
    metrics = load_json(run_dir / "iter_000" / "metrics.json")
    assert metrics["delta"]["heldout_accuracy_gap"] is None


def test_pre_eval_carries_forward_previous_post_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _seed_csv(tmp_path / "seed.csv")
    run_dir, state, progress = autoresearch.init_run(
        runs_root=tmp_path / "runs",
        run_name="carry",
        seed_data=seed,
        resume_dir=None,
        eval_sample_size=2,
    )
    eval_calls: list[tuple[str, str | None]] = []

    def fake_dual_evals(**kwargs):
        tag = kwargs["tag"]
        iter_dir = kwargs["iter_dir"]
        sampler = kwargs["sampler_path"]
        eval_calls.append((tag, sampler))
        train_diff = _diff(0.5, model_path=sampler)
        heldout_diff = _diff(0.5, model_path=sampler)
        for split, diff, sample in (
            ("train_seed", train_diff, kwargs["eval_train_sample"]),
            ("heldout", heldout_diff, kwargs["eval_heldout_sample"]),
        ):
            out_dir = ensure_dir(iter_dir / f"eval_{tag}" / split)
            save_json(out_dir / "differential.json", diff.to_dict())
            shutil.copy2(sample, out_dir / "eval_sample.csv")
        return train_diff, heldout_diff

    def fake_train_step(**kwargs):
        meta = {
            "state_path": "tinker://test/weights/20",
            "sampler_path": "tinker://test/sampler/20",
            "policy_source": "base",
            "judge_source": "base",
        }
        save_json(kwargs["log_path"] / "train_step_meta.json", meta)
        return meta

    monkeypatch.setattr(autoresearch, "_run_dual_evals", fake_dual_evals)
    monkeypatch.setattr(autoresearch, "run_train_step", fake_train_step)

    kwargs = {
        "model": "openai/gpt-oss-20b",
        "gen_target": 2,
        "seed_samples": 2,
        "variations_per_batch": 1,
        "seeds_per_batch": 1,
        "eval_sample_size": 2,
        "train_max_steps": 2,
        "groups_per_batch": 1,
        "group_size": 1,
        "save_every": 1,
        "eval_every": 1,
        "learning_rate": 1e-5,
        "lora_rank": 1,
        "rng_seed": 0,
        "skip_generate": True,
        "skip_train": False,
        "skip_high_eval": True,
        "gen_temperature": 0.0,
    }
    state = autoresearch.run_iteration(state, progress, **kwargs)
    state = autoresearch.run_iteration(state, progress, **kwargs)

    # Iteration 1's pre eval is carried forward from iteration 0's post eval
    # (same checkpoint, same fixed samples) instead of being re-run.
    assert [tag for tag, _ in eval_calls] == ["pre", "post", "post"]
    journal = load_json(run_dir / "iter_001" / "iteration_state.json")
    assert journal["phase_data"]["eval_pre"]["reused_from"].endswith(
        "iter_000/eval_post"
    )
    carried = load_json(
        run_dir / "iter_001" / "eval_pre" / "train_seed" / "differential.json"
    )
    assert carried["instant"]["model_path"] == "tinker://test/sampler/20"
    metrics = load_json(run_dir / "iter_001" / "metrics.json")
    assert metrics["delta"]["heldout_instant"] == 0.0


def test_pre_eval_does_not_carry_forward_on_checkpoint_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = ensure_dir(tmp_path / "run")
    sample = _seed_csv(run_dir / "eval_sample.csv")
    prev_post = run_dir / "iter_000" / "eval_post"
    for split in ("train_seed", "heldout"):
        out_dir = ensure_dir(prev_post / split)
        save_json(
            out_dir / "differential.json",
            _diff(0.5, model_path="tinker://old/sampler/20").to_dict(),
        )
        shutil.copy2(sample, out_dir / "eval_sample.csv")

    common = {
        "run_dir": run_dir,
        "iteration": 1,
        "need_high": False,
        "eval_train_sample": sample,
        "eval_heldout_sample": sample,
    }
    # Same checkpoint → reusable; different/absent checkpoint → not reusable.
    assert (
        autoresearch._reusable_prev_post(
            sampler_path="tinker://old/sampler/20", **common
        )
        == prev_post
    )
    assert (
        autoresearch._reusable_prev_post(
            sampler_path="tinker://new/sampler/40", **common
        )
        is None
    )
    assert autoresearch._reusable_prev_post(sampler_path=None, **common) is None


def test_run_train_step_raises_when_no_checkpoint_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mathtask import train_step

    data = _seed_csv(tmp_path / "train.csv")
    # Trainer "runs" but never writes a checkpoints.jsonl (e.g. save_every
    # larger than max_steps): the loop must fail fast instead of silently
    # evaluating the base model afterwards.
    monkeypatch.setattr(train_step, "run_python", lambda argv, **kwargs: None)
    with pytest.raises(RuntimeError, match="no usable checkpoint"):
        train_step.run_train_step(
            data_path=data,
            log_path=tmp_path / "log",
            load_checkpoint=None,
        )


def test_validate_args_rejects_uncheckpointable_train() -> None:
    args = autoresearch.parse_args(["--train-max-steps", "5", "--save-every", "20"])
    with pytest.raises(ValueError, match="save_every"):
        autoresearch.validate_args(args)
    skipped = autoresearch.parse_args(
        ["--train-max-steps", "5", "--save-every", "20", "--skip-train"]
    )
    autoresearch.validate_args(skipped)  # checkpointing irrelevant without train
