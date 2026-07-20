# Agent instructions

Standalone project: `cd` here, use the venv, run `python loop.py`. Tests:
`python -m pytest` (pytest is pinned to `tests/` + `dashboard/` via
`pyproject.toml`).

## Structure logic

The root is deliberately three files (Karpathy-autoresearch style); keep it
that way:

| File | Role | May an agent edit it? |
|------|------|-----------------------|
| `loop.py` | The research loop: `run_iteration` is six phase calls + `main()` | Yes — this is the research surface |
| `prepare.py` | Frozen harness: `init_run`, eval/train primitives, `IterationContext`, phase bodies | Only for bugs, not for research ideas |
| `program.md` | Goal, metric, mutable vs frozen surfaces, hypothesis log | Yes — update as hypotheses evolve |

Support files (`loop_config.toml`, `README.md`, `conftest.py`,
`pyproject.toml`) stay in root. Everything else lives in a package. Do NOT
add new top-level `.py` files; if logic doesn't belong in `loop.py` or
`prepare.py`, it belongs in a package:

- `core/` — task-agnostic plumbing: `cli`, `io`, `defaults`, `metrics`,
  `runstate` (state + run lock), `journal` (crash-safe phase journal),
  `history`, `stopping`, `progress`, `loop_config`, `stats_ci`.
- `mathtask/` — everything arithmetic-specific: vendored `ask_arithmetic` /
  `test_arithmetic` / `train_math_llm_judge` scripts, `tinker_sample`,
  `dataset`, `parsing`, `generate_data`, `validate_data`, `run_evals`,
  `sequential_eval`, `train_step`, `math_integration`. This is the seam to
  swap for a different task.
- `tests/` — all tests. `scripts/` — one-off run utilities. `dashboard/` —
  reads run-dir JSON artifacts only; it must never import loop modules.

## Dependency rules

- `mathtask` may import `core`; `core` must NEVER import `mathtask`. Where
  core-level code needs task behavior, inject a callback (see
  `core/journal.py:ensure_before`).
- `prepare.py` composes both; `loop.py` imports `prepare` as a module
  (`prepare.X(...)`) so tests can monkeypatch `prepare._run_dual_evals`,
  `prepare.run_train_step`, `loop.run_iteration`, `loop.check_model_consistency`.
- Subprocesses are invoked as modules (`python -m mathtask.<mod>`, cwd =
  project root), never by file path — the vendored scripts import each other
  through the package.

## Invariants (do not break)

- Never create a root `__init__.py`: it turns this project into a package of
  its parent directory and pytest then imports stale parent-dir modules
  (this was a real, silent bug).
- Run-dir formats are a compatibility contract for resume and the dashboard:
  `state.json` fields, `iteration_state.json` journal config keys,
  `history.jsonl` record keys, `status.json` / `events.jsonl`. Changing any
  key set breaks `--resume` of existing runs; extend tolerantly (readers
  already ignore unknown fields).
- The held-out split is frozen per run: nothing may write `heldout_test.csv`
  rows into training data, generation seeds, or merges. Canonical-key
  filters and `assert_disjoint_train_heldout` enforce this; keep them on
  every path that grows the train pool.
- `mathtask/test_arithmetic.py` caps expensive arguments (factorial, fib,
  pow, sigma/phi, comb/perm, primes) and `programmatic_solution` rejects
  operations over 200 chars — model-generated expressions are untrusted
  input; keep bounds when adding functions.
- Evals run at temperature 0 on fixed samples; the pre-eval carry-forward in
  `prepare._reusable_prev_post` relies on checkpoint path + pinned sample
  content checks. Preserve those checks if you touch eval artifacts.

## Workflow

- Gate every change on `python -m pytest` (all green, currently 59); the
  safety tests in `tests/test_autoresearch_safety.py` encode crash/resume
  behavior — adapt their entry points if you move code, never delete the
  behaviors they cover.
- Prefer moving code verbatim over rewriting when restructuring; keep
  commits mechanical-vs-behavioral separated.
- `data/` holds the seed and committed reproducibility samples; `output/` is
  gitignored run artifacts.
