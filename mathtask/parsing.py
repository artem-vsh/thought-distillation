"""Parse free-form model output for data generation and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from mathtask.dataset import MathExample, canonicalize_solution_string

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_EXAMPLE_BLOCK_RE = re.compile(
    r"<example>\s*operation:\s*(.+?)\s*solution:\s*(.+?)\s*</example>",
    re.IGNORECASE | re.DOTALL,
)
_OP_SOL_LINE_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])?\s*(.+?)\s*(?:=|,|->|=>)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def parse_generated_examples(text: str) -> list[MathExample]:
    """Extract (operation, solution) pairs from model generation text.

    Accepts, in order of preference:
      1. A JSON array of objects with operation/solution keys
      2. <example>operation: ... solution: ...</example> blocks
      3. Line-oriented ``expr = value`` or ``expr, value`` forms
    """
    text = text.strip()
    if not text:
        return []

    # 1) JSON array anywhere in the reply
    for match in _JSON_ARRAY_RE.finditer(text):
        blob = match.group(0)
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        examples: list[MathExample] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            op = str(item.get("operation") or item.get("expr") or "").strip()
            sol = str(item.get("solution") or item.get("answer") or "").strip()
            if op and sol:
                examples.append(
                    MathExample(operation=op, solution=canonicalize_solution_string(sol))
                )
        if examples:
            return examples

    # 2) Tagged blocks
    blocks = _EXAMPLE_BLOCK_RE.findall(text)
    if blocks:
        return [
            MathExample(
                operation=op.strip(),
                solution=canonicalize_solution_string(sol.strip()),
            )
            for op, sol in blocks
            if op.strip() and sol.strip()
        ]

    # 3) Line-oriented fallback
    examples = []
    for line in text.splitlines():
        m = _OP_SOL_LINE_RE.match(line)
        if not m:
            continue
        op, sol = m.group(1).strip(), m.group(2).strip()
        # Skip markdown tables / headers
        if op.lower() in {"operation", "expression", "expr"}:
            continue
        examples.append(
            MathExample(operation=op, solution=canonicalize_solution_string(sol))
        )
    return examples


_VERDICT_RE = re.compile(
    r"<verdict>\s*(keep|discard)\s*</verdict>",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r"<confidence>\s*(high|medium|low|\d+(?:\.\d+)?)\s*</confidence>",
    re.IGNORECASE,
)
_FIXED_SOLUTION_RE = re.compile(
    r"<solution>\s*([^\s<]+)\s*</solution>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationVerdict:
    keep: bool
    confidence: str
    solution: str | None
    raw: str


def parse_validation_verdict(text: str, fallback_solution: str = "") -> ValidationVerdict:
    """Parse keep/discard + confidence from a high-reasoning validator reply."""
    verdict_m = _VERDICT_RE.search(text)
    conf_m = _CONFIDENCE_RE.search(text)
    sol_m = _FIXED_SOLUTION_RE.search(text)

    keep = False
    if verdict_m is not None:
        keep = verdict_m.group(1).lower() == "keep"

    confidence = conf_m.group(1).lower() if conf_m else "low"
    solution = (
        canonicalize_solution_string(sol_m.group(1))
        if sol_m
        else (fallback_solution or None)
    )
    return ValidationVerdict(
        keep=keep, confidence=confidence, solution=solution, raw=text
    )


def confidence_is_high(confidence: str) -> bool:
    """True when the validator reported high confidence."""
    c = confidence.strip().lower()
    if c == "high":
        return True
    try:
        return float(c) >= 0.8
    except ValueError:
        return False
