"""Eval metric containers shared by eval wiring, progress, and the loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EvalMetrics:
    """Accuracy summary for one policy on an eval sample."""

    effort: str
    total: int
    completed: int
    correct: int
    accuracy: float
    incomplete: int = 0
    model_path: str | None = None
    answers_csv: str | None = None
    scored_csv: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalMetrics:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DifferentialMetrics:
    """Instant vs high performance on the same eval sample."""

    sample_size: int
    instant: EvalMetrics
    high: EvalMetrics
    accuracy_gap: float | None  # high - instant; None when high eval is skipped
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "instant": self.instant.to_dict(),
            "high": self.high.to_dict(),
            "accuracy_gap": self.accuracy_gap,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DifferentialMetrics:
        return cls(
            sample_size=int(data["sample_size"]),
            instant=EvalMetrics.from_dict(data["instant"]),
            high=EvalMetrics.from_dict(data["high"]),
            accuracy_gap=data.get("accuracy_gap"),
            timestamp=str(data.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        )
