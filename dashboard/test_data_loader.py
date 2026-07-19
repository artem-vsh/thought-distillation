#!/usr/bin/env python3
"""Unit tests for dashboard data aggregation (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from dashboard.data_loader import build_snapshot, list_runs, _history_series


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_snapshot_empty(tmp_path: Path) -> None:
    snap = build_snapshot(None)
    assert snap["ok"] is False


def test_build_snapshot_with_dual_evals(tmp_path: Path) -> None:
    run = tmp_path / "math-demo"
    run.mkdir()
    _write(
        run / "status.json",
        json.dumps(
            {
                "run_dir": str(run),
                "phase": "eval_post",
                "iteration": 1,
                "message": "ok",
                "last_instant_accuracy": 0.4,
                "last_heldout_instant_accuracy": 0.3,
                "last_overfit_gap": 0.1,
                "train_pool_size": 100,
                "heldout_size": 20,
                "updated_at": "2026-07-19T12:00:00+00:00",
            }
        ),
    )
    history = [
        {
            "iteration": 0,
            "train_pool_size": 90,
            "new_examples": 10,
            "train_seed": {
                "pre": {"instant": {"accuracy": 0.1, "correct": 1, "completed": 10}},
                "post": {
                    "instant": {"accuracy": 0.2, "correct": 2, "completed": 10},
                    "high": {"accuracy": 0.5},
                    "accuracy_gap": 0.3,
                },
                "delta": {"instant_accuracy": 0.1},
            },
            "heldout": {
                "pre": {"instant": {"accuracy": 0.08}},
                "post": {
                    "instant": {"accuracy": 0.15},
                    "high": {"accuracy": 0.45},
                    "accuracy_gap": 0.3,
                },
                "delta": {"instant_accuracy": 0.07},
            },
            "delta": {"overfit_gap": 0.05},
            "train": {
                "state_path": "tinker://a/weights/000020",
                "sampler_path": "tinker://a/sampler_weights/000020",
            },
        }
    ]
    with (run / "history.jsonl").open("w", encoding="utf-8") as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")

    _write(
        run / "checkpoints.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "iteration": 0,
                        "name": "000010",
                        "batch": 10,
                        "state_path": "tinker://a/weights/000010",
                        "sampler_path": "tinker://a/sampler_weights/000010",
                    }
                ),
                json.dumps(
                    {
                        "iteration": 0,
                        "name": "000020",
                        "batch": 20,
                        "state_path": "tinker://a/weights/000020",
                        "sampler_path": "tinker://a/sampler_weights/000020",
                    }
                ),
            ]
        )
        + "\n",
    )
    _write(
        run / "events.jsonl",
        json.dumps({"ts": "2026-07-19T12:00:00+00:00", "kind": "phase", "message": "train"})
        + "\n",
    )
    (run / "iter_000").mkdir()
    _write(run / "iter_000" / "generated.csv", "operation,solution\n1+1,2\n")

    # Parent is runs root for list_runs
    snap = build_snapshot(run)
    assert snap["ok"] is True
    assert snap["headline"]["phase"] == "eval_post"
    assert snap["headline"]["train_seed_instant"] == 0.4
    assert snap["headline"]["heldout_instant"] == 0.3
    assert snap["evals"]["latest"]["train_seed"]["post"]["instant"] == 0.2
    assert snap["evals"]["latest"]["heldout"]["post"]["instant"] == 0.15
    assert snap["checkpoints"]["count_finalized"] == 1
    assert snap["checkpoints"]["count_intermediate"] == 1
    assert snap["checkpoints"]["last_finalized"]["name"] == "000020"
    assert snap["checkpoints"]["last_intermediate"]["name"] == "000010"
    assert snap["evals"]["series"]["train_seed_instant_post"] == [0.2]
    assert snap["evals"]["series"]["heldout_instant_post"] == [0.15]
    assert len(snap["events"]) == 1
    assert snap["iterations"][0]["n_generated"] == 1


def test_history_series_high() -> None:
    series = _history_series(
        [
            {
                "iteration": 0,
                "train_seed": {
                    "post": {
                        "instant": {"accuracy": 0.2},
                        "high": {"accuracy": 0.6},
                    }
                },
                "heldout": {
                    "post": {
                        "instant": {"accuracy": 0.1},
                        "high": {"accuracy": 0.5},
                    }
                },
                "delta": {"overfit_gap": 0.1},
            }
        ]
    )
    assert series["train_seed_high_post"] == [0.6]
    assert series["heldout_high_post"] == [0.5]
    assert series["overfit_gap"] == [0.1]


def test_in_progress_point_from_eval_progress(tmp_path: Path) -> None:
    from dashboard.data_loader import build_snapshot

    run = tmp_path / "live"
    run.mkdir()
    _write(
        run / "status.json",
        json.dumps(
            {
                "run_dir": str(run),
                "phase": "eval_post",
                "iteration": 2,
                "eval_in_progress": True,
                "eval_progress": {
                    "status": "in_progress",
                    "tag": "post_heldout",
                    "n": 80,
                    "max_n": 200,
                    "target_ci_pp": 2.0,
                    "p_value": 0.05,
                    "instant": {"accuracy": 0.35, "correct": 28, "total": 80},
                    "high": {"accuracy": 0.55, "correct": 44, "total": 80},
                    "instant_ci": {"half_width": 0.1},
                },
            }
        ),
    )
    snap = build_snapshot(run)
    ip = snap.get("in_progress")
    assert ip is not None
    assert "eval" in ip["kinds"]
    assert ip["heldout_instant"] == 0.35
    assert ip["heldout_high"] == 0.55
    assert ip["heldout_instant_ci_half"] == 0.1


def test_list_runs(tmp_path: Path) -> None:
    a = tmp_path / "run-a"
    b = tmp_path / "run-b"
    a.mkdir()
    b.mkdir()
    _write(a / "status.json", json.dumps({"phase": "init", "iteration": 0}))
    _write(b / "status.json", json.dumps({"phase": "train", "iteration": 2}))
    runs = list_runs(tmp_path)
    assert len(runs) == 2
    assert {r["id"] for r in runs} == {"run-a", "run-b"}
