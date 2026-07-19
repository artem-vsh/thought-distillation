# Math autoresearch loop

**Standalone project.** Everything you need lives in this directory (math
scripts, data, dashboard, git). No parent-folder imports.

```bash
cd loop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Also put TINKER_API_KEY in the env or ~/.secrets/tinker.env

python -m autoresearch …
python -m dashboard --open
```

Closed loop that grows math training data, measures the **instant vs high**
gap, trains the instant policy with a high-reasoning LLM judge, then re-evals
until gains are marginal. Artifacts land in **`output/autoresearch/`**.

## Vendored math pipeline

| Local module | Role |
|--------------|------|
| `ask_arithmetic.py` | Eval sampling (`--instant` / `--high`, optional `--model-path`) |
| `test_arithmetic.py` | Scoring + programmatic expression eval |
| `train_math_llm_judge.py` | RL train (instant policy + high judge; optional `judge_model_path`) |
| `data/arithmetic_operations.csv` | Seed dataset |

Wiring helpers live in `math_integration.py`. Each run writes `integration.json`.

Train CSVs are validated with `train_math_llm_judge.load_math_problems` before
every train step (same schema: non-empty `operation` + `solution`).

## Auto-switch: base → checkpoints

| Role | Before any train | After ≥1 checkpoint |
|------|------------------|---------------------|
| **Policy train** | base model | resume `state_path` (weights) |
| **LLM judge** | base @ high reasoning | previous `sampler_path` @ high reasoning |
| **Eval instant** | base @ instant | last `sampler_path` @ instant |
| **Eval high** | base @ high | last `sampler_path` @ high |
| **Data gen / validate** | always base @ high (stable labels) | same |

The judge uses the **previous** sampler while the policy continues training, then
both pointers advance when the train step finishes (`state.json` /
`status.json`).

A train step that produces no checkpoint (e.g. `save_every > max_steps`) raises
instead of letting later evals silently measure the base model; the config is
also rejected up front unless `--skip-train` is set.

Generated data is validated with `test_arithmetic.evaluate_operation`, which
caps expensive arguments (`factorial` ≤ 20, `fib` ≤ 70, exponents ≤ 40,
`sigma`/`phi` ≤ 1e12, `comb`/`perm` ≤ 1e4, sieve limits on prime functions)
and the loop skips operations over 200 characters, so a single degenerate
generated expression cannot stall validation.

## Held-out test (anti-overfitting)

At run start the seed is split once:

| File | Role |
|------|------|
| `train_data.csv` | Initial train pool (grows with validated generations only) |
| `heldout_test.csv` | **Never** used for train, gen few-shot, or merges |
| `eval_train_sample.csv` | Fixed sample from the *train* split (in-domain track) |
| `eval_heldout_sample.csv` | Fixed sample from held-out (generalization track) |
| `split_manifest.json` | Full audit of operations in each bucket |

Every pre/post eval runs **both** tracks:

```
iter_XXX/eval_pre/train_seed/   # checkpoint-seeded / in-domain
iter_XXX/eval_pre/heldout/      # never-train original test
iter_XXX/eval_post/train_seed/
iter_XXX/eval_post/heldout/
```

Evals sample at temperature 0 on fixed CSVs, so iteration N's pre-train eval
would repeat iteration N−1's post-train measurement exactly. When the policy
checkpoint and both samples are verified unchanged, the pre eval **carries
forward** the previous `eval_post` artifacts (copied into `eval_pre/`, noted
as `reused_from` in `iteration_state.json` and an `eval` event) instead of
re-sampling — halving eval cost per iteration. Any mismatch falls back to a
fresh eval.

Metrics include `overfit_gap = train_seed_instant − heldout_instant`. Early
stopping uses **held-out** instant Δ (not train-seed). Default held-out
fraction: `--heldout-fraction 0.10`.

A gain counts as progress only when it exceeds
`max(--marginal-delta, --noise-z × binomial SE)` of the pre/post comparison
(SE from the recorded correct/total counters; independent-proportions
approximation). With the defaults (n=200, z=1.0) the noise band is ≈5 points,
so the stop rule is not driven by eval sampling noise. `--noise-z 0` disables
the band and restores the plain `--marginal-delta` comparison. Runs with a
held-out eval sample under 30 rows log a warning at init.

Held-out membership is enforced with **canonical operation keys**:
whitespace, parentheses, `^`/`**`, numeric formatting, case, and operand
order of commutative operators (`+`, `*`, `gcd`, `lcm`, `max`, `min`) are
normalized before comparison, so near-duplicates like `34+12` cannot leak
into training when `12+34` is held out. Canonical duplicates found inside the
seed itself are dropped from the train side at split time (with a warning).

Eval accuracy uses the full fixed sample as its denominator. Empty or invalid
model responses count as incorrect and remain visible through the
`completed`/`incomplete` counters; they cannot inflate accuracy by being
excluded.

The held-out set makes the **measurement** independent of training data. The
training judge is a separate concern: after iteration zero it is the previous
policy checkpoint rendered at high reasoning. This intentionally measures the
learning dynamics of that self-improvement setup, but it is not an independent
teacher.

## Pipeline (per iteration)

| Step | Module | What it does |
|------|--------|----------------|
| 1. Generate | `generate_data.py` | High-reasoning variations of seed/train examples |
| 2. Validate | `validate_data.py` | `test_arithmetic.evaluate_operation` + high keep/discard |
| 3. Eval pre | `run_evals.py` | `ask_arithmetic` + `test_arithmetic` → gap |
| 4. Train | `train_step.py` | `train_math_llm_judge` resume last ckpt |
| 5. Eval post | `run_evals.py` | Instant uses new `sampler_path`; save Δ |
| 6. Stop | `autoresearch.py` | Marginal instant Δ for N iters |

## Quick start

```bash
cd loop
source .venv/bin/activate   # after pip install -r requirements.txt

python -m autoresearch \
  --seed-data data/arithmetic_operations.csv \
  --max-iters 5 \
  --gen-target 100 \
  --eval-sample-size 100 \
  --train-max-steps 20 \
  --marginal-delta 0.005 \
  --marginal-streak 2

# Live visual dashboard (checkpoints + dual evals)
python -m dashboard --port 8765 --open
# → http://127.0.0.1:8765/
```

Run names are exclusive: an existing `--run-name` is never overwritten. Resume
an existing run explicitly:

```bash
python -m autoresearch --resume output/autoresearch/<run-name>
```

Resume restores the original `config.json`. Only options explicitly supplied
on the resume command override saved values (detected via a real second
argparse pass, so abbreviations like `--max-it` count), and those overrides
are appended to `resume_history.jsonl`. Options that cannot apply to an
existing run (`--seed-data`, `--runs-root`, `--run-name`) are warned about and
ignored. An in-progress iteration keeps an atomic `iteration_state.json`
phase journal, reuses completed generation/eval/train artifacts, and resumes
rather than deletes an interrupted trainer log.

Resume semantics:

- **Interrupted run** → continues the in-progress iteration.
- **Completed run** (max-iters reached, not stopped) → *extends* the run by
  another `--max-iters` iterations; the extension is printed at start.
- **Early-stopped run** → no-op unless `--force` is given, which clears the
  stop decision and resumes iterating (the normal stopping rule applies to
  new iterations).

Each run directory holds an advisory lock (`.autoresearch.lock`) for the
process lifetime, so a second autoresearch process cannot resume a run that
is already active.

## Dashboard-ready intermediate data

Poll these under `output/autoresearch/<run>/` (designed for a future live UI):

| File | Purpose |
|------|---------|
| **`status.json`** | Current phase, iteration, accuracies, checkpoint paths (overwrite) |
| **`events.jsonl`** | Append-only timeline (`phase`, `eval`, `train_done`, …) |
| **`checkpoints.jsonl`** | Every Tinker checkpoint across iterations |
| **`train_progress.json`** | Latest train metrics tail + checkpoint summary |
| **`history.jsonl`** | One full metrics record per completed iteration |
| **`data_snapshots.jsonl`** | Train pool size after generate/validate |
| **`integration.json`** | Script/model/renderer wiring snapshot |
| **`state.json`** | Resumable loop state |
| **`resume_history.jsonl`** | Resume invocations and explicit config overrides |

### Per-iteration tree

```
iter_000/
  iteration_state.json            # crash-safe phase journal
  train_data_before.csv            # immutable pre-iteration snapshot
  generation_seeds.csv
  generated.csv
  validated.csv
  validation_audit.jsonl
  train_data_snapshot.csv
  eval_sample.csv
  eval_pre/
    eval_sample.csv
    answers_instant.csv          # ask_arithmetic
    scored_instant.csv           # test_arithmetic
    answers_high.csv
    scored_high.csv
    differential.json
    eval_manifest.json
  train/                         # train_math_llm_judge log_path
    checkpoints.jsonl
    metrics.jsonl
    config.json
    iteration_*/…
    train_step_meta.json
    last_checkpoint.json
    train_step_argv.json
  train_artifacts/               # durable copies of ckpt/metrics/evals
  eval_post/                     # same layout as eval_pre
  metrics.json                   # pre/post + Δ + paths
```

Also:

```
checkpoint_index/iter_000/       # copied checkpoints.jsonl + metrics.jsonl
```

## Components alone

```bash
python -m generate_data --seed data/arithmetic_operations.csv -o /tmp/g.csv --target 50
python -m validate_data -i /tmp/g.csv -o /tmp/v.csv --audit /tmp/a.jsonl
python -m run_evals --data data/arithmetic_operations.csv --sample-size 50 --out-dir /tmp/e
python -m train_step --data data/arithmetic_operations.csv --log-path /tmp/t --max-steps 2
```
