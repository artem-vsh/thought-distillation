"""Default knobs for the autoresearch loop (CLI/TOML override these)."""

from __future__ import annotations

from core.io import LOOP_ROOT

DEFAULT_SEED_DATA = LOOP_ROOT / "data" / "arithmetic_operations.csv"
DEFAULT_RUNS_ROOT = LOOP_ROOT / "output" / "autoresearch"
DEFAULT_MODEL = "openai/gpt-oss-20b"

# Stop when absolute instant-accuracy improvement falls below this for N iters.
DEFAULT_MARGINAL_DELTA = 0.005  # 0.5 percentage points as fraction
DEFAULT_MARGINAL_STREAK = 2
DEFAULT_MAX_ITERS = 10

# How many problems to generate / eval per loop iteration.
DEFAULT_GEN_TARGET = 400
DEFAULT_SEED_SAMPLES = 40
DEFAULT_EVAL_SAMPLE_SIZE = 200
DEFAULT_VARIATIONS_PER_SEED = 8
DEFAULT_SEEDS_PER_BATCH = 8
DEFAULT_GEN_TEMPERATURE = 0.8
# Fraction of original seed carved out as never-train held-out test.
DEFAULT_HELDOUT_FRACTION = 0.10
# Below this many held-out eval rows the generalization track is too noisy
# to drive early stopping; init_run warns when the sample is smaller.
MIN_HELDOUT_EVAL_SAMPLE = 30

# Training defaults (match a modest train_math_llm_judge run).
DEFAULT_TRAIN_MAX_STEPS = 20
DEFAULT_TRAIN_GROUPS_PER_BATCH = 16
DEFAULT_TRAIN_GROUP_SIZE = 4
DEFAULT_TRAIN_SAVE_EVERY = 20
DEFAULT_TRAIN_EVAL_EVERY = 20
DEFAULT_TRAIN_LEARNING_RATE = 1e-5
DEFAULT_TRAIN_LORA_RANK = 32

# Early stopping: an instant-accuracy gain only counts as real progress when
# it exceeds max(marginal_delta, noise_z * binomial SE of the pre/post
# comparison). 1.0 ≈ a one-standard-deviation noise band.
DEFAULT_NOISE_Z = 1.0
