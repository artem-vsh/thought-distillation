#!/usr/bin/env python3
"""Unit tests for CI helpers and config loading."""

from __future__ import annotations

from pathlib import Path

from loop_config import EvalCIConfig, load_loop_config
from stats_ci import diff_ci, proportion_ci, z_critical


def test_z_critical_p05() -> None:
    z = z_critical(0.05)
    assert 1.95 < z < 1.97


def test_proportion_ci_shrinks_with_n() -> None:
    small = proportion_ci(5, 10, p_value=0.05)
    large = proportion_ci(50, 100, p_value=0.05)
    assert large.half_width < small.half_width
    assert abs(small.p - 0.5) < 1e-9


def test_diff_ci_meets_target() -> None:
    # Very large n → narrow CI
    wide = diff_ci(5, 10, 4, 10, name="x", p_value=0.05, target_half_width=0.02)
    assert wide.meets_target is False
    narrow = diff_ci(500, 1000, 480, 1000, name="x", p_value=0.05, target_half_width=0.05)
    assert narrow.meets_target is True


def test_load_default_config() -> None:
    cfg = load_loop_config(Path(__file__).resolve().parent / "loop_config.toml")
    assert cfg.generation.step_size == 400
    assert cfg.eval.target_ci_pp == 2.0
    assert abs(cfg.eval.target_half_width - 0.02) < 1e-12
    assert cfg.eval.p_value == 0.05


def test_eval_ci_config_frozen() -> None:
    e = EvalCIConfig()
    assert e.compare_instant_vs_high is True
