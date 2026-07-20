"""Early-stopping rule: marginal-improvement detection with a noise band."""

from __future__ import annotations

import math
from typing import Any, Sequence

from core.defaults import DEFAULT_NOISE_Z


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
