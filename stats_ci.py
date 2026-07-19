"""Binomial / two-proportion confidence intervals for adaptive evals."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any


@dataclass(frozen=True)
class ProportionCI:
    """Wald CI for a single proportion (accuracy)."""

    n: int
    correct: int
    p: float
    half_width: float
    low: float
    high: float
    z: float
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiffCI:
    """Wald CI for p_a − p_b (independent two-proportion)."""

    name: str
    n_a: int
    correct_a: int
    p_a: float
    n_b: int
    correct_b: int
    p_b: float
    diff: float
    half_width: float
    low: float
    high: float
    z: float
    p_value: float
    meets_target: bool
    target_half_width: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def z_critical(p_value: float) -> float:
    """Two-sided normal critical value for the given p-value (α)."""
    if not 0.0 < p_value < 1.0:
        raise ValueError(f"p_value must be in (0,1), got {p_value}")
    # P(|Z| > z) = p_value  →  z = Φ^{-1}(1 - p/2)
    return float(NormalDist().inv_cdf(1.0 - p_value / 2.0))


def proportion_ci(
    correct: int,
    n: int,
    *,
    p_value: float = 0.05,
) -> ProportionCI:
    if n <= 0:
        z = z_critical(p_value)
        return ProportionCI(
            n=0, correct=0, p=0.0, half_width=1.0, low=0.0, high=1.0, z=z, p_value=p_value
        )
    correct = max(0, min(int(correct), int(n)))
    p = correct / n
    z = z_critical(p_value)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    half = z * se
    return ProportionCI(
        n=n,
        correct=correct,
        p=p,
        half_width=half,
        low=max(0.0, p - half),
        high=min(1.0, p + half),
        z=z,
        p_value=p_value,
    )


def diff_ci(
    correct_a: int,
    n_a: int,
    correct_b: int,
    n_b: int,
    *,
    name: str,
    p_value: float = 0.05,
    target_half_width: float = 0.02,
) -> DiffCI:
    """CI for p_a − p_b with independent-proportions SE (conservative for paired)."""
    z = z_critical(p_value)
    if n_a <= 0 or n_b <= 0:
        return DiffCI(
            name=name,
            n_a=max(0, n_a),
            correct_a=max(0, correct_a),
            p_a=0.0,
            n_b=max(0, n_b),
            correct_b=max(0, correct_b),
            p_b=0.0,
            diff=0.0,
            half_width=1.0,
            low=-1.0,
            high=1.0,
            z=z,
            p_value=p_value,
            meets_target=False,
            target_half_width=target_half_width,
        )
    correct_a = max(0, min(int(correct_a), int(n_a)))
    correct_b = max(0, min(int(correct_b), int(n_b)))
    p_a = correct_a / n_a
    p_b = correct_b / n_b
    diff = p_a - p_b
    se = math.sqrt(
        max(p_a * (1.0 - p_a), 0.0) / n_a + max(p_b * (1.0 - p_b), 0.0) / n_b
    )
    half = z * se
    return DiffCI(
        name=name,
        n_a=n_a,
        correct_a=correct_a,
        p_a=p_a,
        n_b=n_b,
        correct_b=correct_b,
        p_b=p_b,
        diff=diff,
        half_width=half,
        low=diff - half,
        high=diff + half,
        z=z,
        p_value=p_value,
        meets_target=half <= target_half_width + 1e-15,
        target_half_width=target_half_width,
    )


def required_n_rough(
    *,
    target_half_width: float,
    p_value: float = 0.05,
    p: float = 0.5,
) -> int:
    """Rough n for a single-proportion CI of given half-width (worst-case p=0.5)."""
    if target_half_width <= 0:
        return 10**9
    z = z_critical(p_value)
    # half = z * sqrt(p(1-p)/n)  →  n = z² p(1-p) / half²
    return int(math.ceil((z * z * p * (1.0 - p)) / (target_half_width**2)))
