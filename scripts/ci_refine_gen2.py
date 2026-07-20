#!/usr/bin/env python3
"""Background CI refine for gen 2 while later gens run.

Grows held-out (and train-seed) post-eval sample toward the pool / CI target,
then upserts history.jsonl generation 2 without touching the live loop's
current iteration directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOOP_ROOT = Path(__file__).resolve().parent.parent
if str(_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOOP_ROOT))

from common import append_jsonl, load_json, save_json
from core.loop_config import EvalCIConfig
from mathtask.sequential_eval import run_sequential_differential


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upsert_gen2_history(run_dir: Path, *, heldout_post: dict, train_seed_post: dict | None) -> None:
    path = run_dir / "history.jsonl"
    records: list[dict] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))

    gen2 = None
    for rec in records:
        if rec.get("iteration") == 0 or rec.get("generation") == 2:
            gen2 = rec
            break
    if gen2 is None:
        gen2 = {"iteration": 0, "generation": 2, "kind": "iteration_complete"}
        records.append(gen2)

    gen2["ts"] = _utc_now()
    gen2["kind"] = "iteration_complete"
    gen2["ci_refine_pending"] = False
    gen2["note"] = (
        "Gen2 post-eval CI refined in background (held-out grown toward pool/target)."
    )
    heldout = gen2.setdefault("heldout", {})
    heldout["post"] = heldout_post
    gen2["post_heldout"] = heldout_post
    if train_seed_post is not None:
        ts = gen2.setdefault("train_seed", {})
        ts["post"] = train_seed_post
        gen2["post_train"] = train_seed_post
        gen2["post"] = train_seed_post

    # Recompute simple deltas if pre present
    def _acc(side: dict | None) -> float | None:
        if not isinstance(side, dict):
            return None
        inst = side.get("instant")
        if isinstance(inst, dict) and inst.get("accuracy") is not None:
            return float(inst["accuracy"])
        return None

    pre_h = _acc(heldout.get("pre"))
    post_h = _acc(heldout_post)
    if pre_h is not None and post_h is not None:
        heldout.setdefault("delta", {})["instant_accuracy"] = post_h - pre_h
        gen2.setdefault("delta", {})["heldout_instant"] = post_h - pre_h
    pre_t = _acc((gen2.get("train_seed") or {}).get("pre"))
    post_t = _acc(train_seed_post) if train_seed_post else None
    if pre_t is not None and post_t is not None:
        gen2.setdefault("train_seed", {}).setdefault("delta", {})["instant_accuracy"] = (
            post_t - pre_t
        )
        gen2.setdefault("delta", {})["train_seed_instant"] = post_t - pre_t
    if post_t is not None and post_h is not None:
        gen2["overfit_gap"] = post_t - post_h
        gen2.setdefault("delta", {})["overfit_gap"] = post_t - post_h

    # Replace iter0/gen2 record
    out: list[dict] = []
    replaced = False
    for rec in records:
        if rec.get("iteration") == 0 or rec.get("generation") == 2:
            if not replaced:
                out.append(gen2)
                replaced = True
            continue
        out.append(rec)
    if not replaced:
        out.append(gen2)

    path.write_text(
        "".join(json.dumps(r) + "\n" for r in out), encoding="utf-8"
    )
    save_json(run_dir / "iter_000" / "metrics.json", gen2)

    # Patch status headline if still pointing at gen2 numbers (optional soft update)
    status_path = run_dir / "status.json"
    if status_path.is_file():
        st = load_json(status_path)
        # Only update last_* if we're not mid-eval for a later gen, or always
        # refresh gen2 CI note in message suffix.
        st["extra"] = dict(st.get("extra") or {})
        st["extra"]["gen2_ci_refine"] = {
            "ts": _utc_now(),
            "heldout_n": heldout_post.get("sample_size")
            or (heldout_post.get("instant") or {}).get("total"),
            "heldout_instant": _acc(heldout_post),
            "heldout_high": (
                (heldout_post.get("high") or {}).get("accuracy")
                if isinstance(heldout_post.get("high"), dict)
                else None
            ),
            "status": "complete",
        }
        # Do not clobber live phase; just bump updated_at so dashboard refreshes
        st["updated_at"] = _utc_now()
        save_json(status_path, st)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("output/autoresearch/run"))
    p.add_argument("--skip-train-seed", action="store_true")
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    job_path = run_dir / "iter_000" / "ci_refine_job.json"
    if not job_path.is_file():
        raise SystemExit(f"Missing job file: {job_path}")
    job = load_json(job_path)
    model = job.get("model") or "openai/gpt-oss-20b"
    sampler = job["sampler_path"]
    cfg = EvalCIConfig(
        target_ci_pp=float(job.get("target_ci_pp") or 2.0),
        p_value=float(job.get("p_value") or 0.05),
        batch_size=int(job.get("batch_size") or 40),
        min_sample_size=40,
        max_sample_size=int(job.get("target_n") or 200),
        compare_instant_vs_high=True,
        compare_instant_vs_previous=False,
        require_heldout_ci=True,
        require_train_seed_ci=False,
    )

    held_dir = Path(job["heldout_dir"])
    held_pool = Path(job["heldout_pool"])
    print(
        f"[ci_refine] gen2 heldout grow → n≤{cfg.max_sample_size} "
        f"±{cfg.target_ci_pp}pp @ p<{cfg.p_value} sampler={sampler}",
        flush=True,
    )
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "ts": _utc_now(),
            "kind": "ci_refine_start",
            "iteration": 0,
            "generation": 2,
            "message": "Background gen2 CI refine started",
            "target_n": cfg.max_sample_size,
        },
    )

    held_seq = run_sequential_differential(
        pool_csv=held_pool,
        out_dir=held_dir,
        model=model,
        instant_model_path=sampler,
        high_model_path=sampler,
        skip_high=False,
        cfg=cfg,
        tag="post_heldout_ci_refine",
        prev_instant=None,
        progress_path=held_dir / "eval_progress.json",
        on_progress=None,
    )
    held_post = held_seq.differential.to_dict()
    held_post["tag"] = "post_heldout"
    held_post["ci"] = {
        "p_value": cfg.p_value,
        "target_ci_pp": cfg.target_ci_pp,
        "target_half_width": cfg.target_half_width,
        "n_used": held_seq.n_used,
        "n_pool": held_seq.n_pool,
        "target_met": held_seq.target_met,
        "exhausted": held_seq.exhausted,
        "instant": held_seq.instant_ci,
        "high": held_seq.high_ci,
        "comparisons": held_seq.comparisons,
        "refined_in_background": True,
    }
    save_json(held_dir / "differential.json", held_post)
    print(
        f"[ci_refine] heldout done n={held_seq.n_used} "
        f"instant={held_seq.differential.instant.accuracy:.4f} "
        f"high={held_seq.differential.high.accuracy:.4f} "
        f"target_met={held_seq.target_met} exhausted={held_seq.exhausted}",
        flush=True,
    )

    train_post = None
    if not args.skip_train_seed:
        ts_dir = Path(job["train_seed_dir"])
        ts_pool = Path(job["train_seed_pool"])
        # Cap train-seed at min (require_train_seed_ci=false semantics)
        ts_cfg = EvalCIConfig(
            target_ci_pp=cfg.target_ci_pp,
            p_value=cfg.p_value,
            batch_size=cfg.batch_size,
            min_sample_size=40,
            max_sample_size=40,
            compare_instant_vs_high=True,
            compare_instant_vs_previous=False,
            require_heldout_ci=False,
            require_train_seed_ci=False,
        )
        print("[ci_refine] train-seed dual at n=40 (no CI growth)", flush=True)
        ts_seq = run_sequential_differential(
            pool_csv=ts_pool,
            out_dir=ts_dir,
            model=model,
            instant_model_path=sampler,
            high_model_path=sampler,
            skip_high=False,
            cfg=ts_cfg,
            tag="post_train_seed_ci_refine",
            prev_instant=None,
            progress_path=ts_dir / "eval_progress.json",
            on_progress=None,
        )
        train_post = ts_seq.differential.to_dict()
        train_post["tag"] = "post_train_seed"
        save_json(ts_dir / "differential.json", train_post)
        print(
            f"[ci_refine] train-seed done n={ts_seq.n_used} "
            f"instant={ts_seq.differential.instant.accuracy:.4f} "
            f"high={ts_seq.differential.high.accuracy:.4f}",
            flush=True,
        )

    _upsert_gen2_history(run_dir, heldout_post=held_post, train_seed_post=train_post)
    save_json(
        run_dir / "iter_000" / "eval_post" / "sequential_ci.json",
        {
            "heldout": held_seq.to_dict(),
            "train_seed": train_post,
            "refined_in_background": True,
            "ts": _utc_now(),
        },
    )
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "ts": _utc_now(),
            "kind": "ci_refine_done",
            "iteration": 0,
            "generation": 2,
            "message": (
                f"Gen2 CI refine done heldout n={held_seq.n_used} "
                f"instant={held_seq.differential.instant.accuracy:.4f} "
                f"high={held_seq.differential.high.accuracy:.4f}"
            ),
            "heldout_n": held_seq.n_used,
            "target_met": held_seq.target_met,
            "exhausted": held_seq.exhausted,
        },
    )
    job["completed_at"] = _utc_now()
    job["heldout_n_final"] = held_seq.n_used
    job["heldout_instant"] = held_seq.differential.instant.accuracy
    job["heldout_high"] = held_seq.differential.high.accuracy
    save_json(job_path, job)
    print("[ci_refine] history gen2 updated", flush=True)


if __name__ == "__main__":
    main()
