# Program

The standing instructions for this autoresearch loop — what it optimizes,
what may be varied between runs, and what stays frozen. Iterate on this file
as hypotheses evolve.

## Goal

Close the **instant vs high** reasoning gap on arithmetic: train the instant
(no-CoT) policy until its held-out accuracy stops improving, using a
high-reasoning LLM judge for RL reward and validated self-generated data for
coverage.

## Metric

**Held-out instant accuracy**, measured on the never-train split with an
adaptive sample grown to a target confidence interval
(`eval.target_ci_pp` @ `eval.p_value` in `loop_config.toml`). Progress must
exceed `max(marginal_delta, noise_z × SE)`; after `marginal_streak`
consecutive misses the run stops. Train-seed accuracy and the
`overfit_gap = train_seed − heldout` are diagnostics, never the objective.

## What may be varied

- `loop.py` — the recipe: phase order, what feeds what, stop policy.
- `loop_config.toml` — step size, eval CI targets, train knobs.
- This file.

## What is frozen during research

- `prepare.py` — the harness: run setup, held-out split, eval/train
  primitives, crash-safe journaling.
- `mathtask/` — the vendored pipeline (ask/test/train scripts, prompts,
  expression evaluator with its argument caps).
- The held-out split of an existing run, and anything under a run's
  `heldout_test.csv` / `split_manifest.json`.

## Current setup notes

- Judge after iteration 0 is the previous policy checkpoint at high
  reasoning — deliberately self-referential; measurement stays anchored to
  programmatic scoring on the held-out split.
- Pre-train evals carry forward the previous post-train artifacts when the
  checkpoint and samples are unchanged (temperature-0 evals repeat).
- Ideas not yet tried: accept/reject iterations against the held-out metric
  (never keep a regression) instead of always advancing the checkpoint.
