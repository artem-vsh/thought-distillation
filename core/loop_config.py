"""Load ``loop_config.toml`` (standalone defaults for generation / eval CI / train)."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# Prefer stdlib tomllib (3.11+); fall back to tomli if present.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

_LOOP_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _LOOP_ROOT / "loop_config.toml"


@dataclass(frozen=True)
class GenerationConfig:
    step_size: int = 400
    seed_samples: int = 40
    variations_per_batch: int = 8
    seeds_per_batch: int = 8
    temperature: float = 0.8


@dataclass(frozen=True)
class EvalCIConfig:
    """Adaptive eval / confidence-interval settings.

    ``target_ci_pp`` is the half-width of the two-sided ``(1 - p_value)`` CI
    on an accuracy *difference*, in percentage points. Default 2.0 @ p=0.05
    means we grow the sample until every required comparison has
    ``half_width ≤ 0.02`` (absolute accuracy units) or the pool is exhausted.
    """

    target_ci_pp: float = 2.0
    p_value: float = 0.05
    batch_size: int = 40
    min_sample_size: int = 40
    max_sample_size: int = 200
    compare_instant_vs_high: bool = True
    compare_instant_vs_previous: bool = True
    require_heldout_ci: bool = True
    require_train_seed_ci: bool = False
    pool_size_heldout: int = 200
    pool_size_train_seed: int = 200

    @property
    def target_half_width(self) -> float:
        """Half-width in accuracy units (0–1), not percentage points."""
        return float(self.target_ci_pp) / 100.0

    @property
    def confidence(self) -> float:
        return 1.0 - float(self.p_value)


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 20
    groups_per_batch: int = 16
    group_size: int = 4
    save_every: int = 20
    eval_every: int = 20
    learning_rate: float = 1e-5
    lora_rank: int = 32


@dataclass(frozen=True)
class StopConfig:
    marginal_delta: float = 0.005
    marginal_streak: int = 2
    noise_z: float = 1.0
    max_iters: int = 10


@dataclass(frozen=True)
class DataConfig:
    heldout_fraction: float = 0.10


@dataclass(frozen=True)
class LoopConfig:
    generation: GenerationConfig = GenerationConfig()
    eval: EvalCIConfig = EvalCIConfig()
    train: TrainConfig = TrainConfig()
    stop: StopConfig = StopConfig()
    data: DataConfig = DataConfig()
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "generation": asdict(self.generation),
            "eval": asdict(self.eval),
            "train": asdict(self.train),
            "stop": asdict(self.stop),
            "data": asdict(self.data),
            "source_path": self.source_path,
        }
        # Convenience flat aliases used by older CLI wiring
        d["step_size"] = self.generation.step_size
        d["gen_target"] = self.generation.step_size
        d["target_ci_pp"] = self.eval.target_ci_pp
        d["p_value"] = self.eval.p_value
        return d


def _section(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_loop_config(path: Path | str | None = None) -> LoopConfig:
    """Load TOML config; missing file → built-in defaults."""
    if path is None:
        env = os.environ.get("LOOP_CONFIG")
        path = Path(env) if env else DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.is_file():
        return LoopConfig(source_path=None)

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a table: {path}")

    cfg = LoopConfig(
        generation=_section(GenerationConfig, raw.get("generation")),
        eval=_section(EvalCIConfig, raw.get("eval")),
        train=_section(TrainConfig, raw.get("train")),
        stop=_section(StopConfig, raw.get("stop")),
        data=_section(DataConfig, raw.get("data")),
        source_path=str(path.resolve()),
    )
    _validate(cfg, path)
    return cfg


def _validate(cfg: LoopConfig, path: Path) -> None:
    e = cfg.eval
    if not 0.0 < e.p_value < 1.0:
        raise ValueError(f"{path}: eval.p_value must be in (0,1), got {e.p_value}")
    if e.target_ci_pp <= 0:
        raise ValueError(f"{path}: eval.target_ci_pp must be > 0")
    if e.batch_size < 1:
        raise ValueError(f"{path}: eval.batch_size must be >= 1")
    if e.max_sample_size < 1:
        raise ValueError(f"{path}: eval.max_sample_size must be >= 1")
    if e.min_sample_size < 1:
        raise ValueError(f"{path}: eval.min_sample_size must be >= 1")
    if cfg.generation.step_size < 1:
        raise ValueError(f"{path}: generation.step_size must be >= 1")
    if not 0.0 < cfg.data.heldout_fraction < 1.0:
        raise ValueError(f"{path}: data.heldout_fraction must be in (0,1)")


def resolve_config_path(cli_path: Path | None = None) -> Path:
    if cli_path is not None:
        return Path(cli_path)
    env = os.environ.get("LOOP_CONFIG")
    if env:
        return Path(env)
    return DEFAULT_CONFIG_PATH


if __name__ == "__main__":
    cfg = load_loop_config()
    import json

    print(json.dumps(cfg.to_dict(), indent=2))
    print(f"source: {cfg.source_path or '(defaults)'}", file=sys.stderr)
