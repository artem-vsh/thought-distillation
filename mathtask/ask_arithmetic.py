#!/usr/bin/env python3
"""Ask GPT-OSS to solve every arithmetic operation in the input CSV."""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import re
from pathlib import Path


# Paths are relative to this project root (loop/) so the package is standalone.
_PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = _PROJECT_ROOT / "data" / "arithmetic_operations.csv"
TINKER_SECRET_FILE = Path.home() / ".secrets" / "tinker.env"
HF_SECRET_FILE = Path.home() / ".secrets" / "hf.env"

MODEL = "openai/gpt-oss-20b"
# High reasoning needs room for analysis + final; 128 truncates CoT mid-answer.
MAX_TOKENS = 16384
# Concurrent in-flight sample requests. Work is asyncio tasks that yield on
# Tinker I/O, not a 1:1 OS thread pool.
NUM_IN_FLIGHT = 50

# Default (low) keeps the original path; other efforts get separate files.
OUTPUT_BY_EFFORT = {
    "instant": _PROJECT_ROOT / "output" / "arithmetic_run_instant.csv",
    "low": _PROJECT_ROOT / "output" / "arithmetic_run.csv",
    "medium": _PROJECT_ROOT / "output" / "arithmetic_run_medium.csv",
    "high": _PROJECT_ROOT / "output" / "arithmetic_run_high.csv",
}
RENDERER_BY_EFFORT = {
    # No system Reasoning: line; combined with FINAL_CHANNEL_PREFILL to skip CoT.
    "instant": "gpt_oss_no_sysprompt",
    "low": "gpt_oss_low_reasoning",
    "medium": "gpt_oss_medium_reasoning",
    "high": "gpt_oss_high_reasoning",
}
# Prefill Harmony final channel so the model cannot open an analysis (CoT) turn.
# Official gpt-oss only documents low/medium/high; this is the practical "off".
FINAL_CHANNEL_PREFILL = "<|channel|>final<|message|>"

TINKER_SECRET_VARIABLES = (
    "TINKER_API_KEY",
    "THINKERMACHINES_TINKER_APIKEY",
)
HF_SECRET_VARIABLES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)

# Integer or decimal, optional leading sign.
_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.\d+|\d+|\.\d+)")


def _parse_env_value(raw_value: str) -> str:
    """Parse the simple quoted or unquoted value used in an env file."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _read_env_file_values(secret_file: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Parse selected KEY=VALUE entries from a simple env file."""
    try:
        lines = secret_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}

    values: dict[str, str] = {}
    wanted = set(names)
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if separator and name in wanted:
            values[name] = _parse_env_value(raw_value)
    return values


def load_tinker_api_key(secret_file: Path = TINKER_SECRET_FILE) -> None:
    """Set TINKER_API_KEY from the environment or the local Tinker env file."""
    for name in TINKER_SECRET_VARIABLES:
        if os.environ.get(name):
            os.environ["TINKER_API_KEY"] = os.environ[name]
            return

    values = _read_env_file_values(secret_file, TINKER_SECRET_VARIABLES)
    if not values and not secret_file.is_file():
        raise RuntimeError(
            f"Set TINKER_API_KEY or create the Tinker secret file: {secret_file}"
        )

    for name in TINKER_SECRET_VARIABLES:
        if values.get(name):
            os.environ["TINKER_API_KEY"] = values[name]
            return

    expected = " or ".join(TINKER_SECRET_VARIABLES)
    raise RuntimeError(f"{secret_file} does not define {expected}")


def load_hf_token(secret_file: Path = HF_SECRET_FILE) -> None:
    """Set HF_TOKEN from the environment or ~/.secrets/hf.env.

    Hugging Face uses this for authenticated Hub downloads (tokenizers, etc.).
    Also sets HUGGING_FACE_HUB_TOKEN, which some libraries read instead.
    """
    for name in HF_SECRET_VARIABLES:
        if os.environ.get(name):
            token = os.environ[name]
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            return

    values = _read_env_file_values(secret_file, HF_SECRET_VARIABLES)
    for name in HF_SECRET_VARIABLES:
        if values.get(name):
            token = values[name]
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            return

    # Optional: unauthenticated Hub access still works, with lower rate limits.
    if not secret_file.is_file():
        print(
            f"Note: no HF token in the environment or {secret_file}; "
            "Hugging Face downloads will be unauthenticated.",
            flush=True,
        )


def iter_operations(input_csv: Path = INPUT_CSV):
    """Yield each non-empty operation from the CSV's operation column."""
    with input_csv.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_csv}")
        if "operation" not in reader.fieldnames:
            raise ValueError(
                f"Input CSV must contain an 'operation' column: {input_csv}"
            )

        for line_number, row in enumerate(reader, start=2):
            operation = (row.get("operation") or "").strip()
            if not operation:
                raise ValueError(f"CSV row {line_number} has an empty operation")
            yield operation


def make_prompt(operation: str) -> str:
    """Create the question sent as the user message."""
    return f"What is the result of {operation}? Give only the result."


def escape_newlines(value: str) -> str:
    """Replace physical line breaks with the two characters backslash-n."""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", r"\n")


def unescape_newlines(value: str) -> str:
    """Turn escaped newline markers back into real newlines."""
    return value.replace(r"\n", "\n").replace(r"\r", "\r")


def is_number_only(value: str) -> bool:
    """Return True when value is exactly one finite number (no extra text)."""
    text = value.strip()
    if not text or _NUMBER_RE.fullmatch(text) is None:
        return False
    try:
        number = float(text)
    except ValueError:
        return False
    return math.isfinite(number)


def canonicalize_number(value: str) -> str:
    """Normalize a numeric string (integers without a trailing .0)."""
    text = value.strip()
    number = float(text)
    if number.is_integer():
        return str(int(number))
    return text.lstrip("+")


def to_number_only(output: str) -> str | None:
    """Coerce model output to a bare number string, or None if impossible."""
    text = unescape_newlines(output).strip()
    if not text:
        return None

    if is_number_only(text):
        return canonicalize_number(text)

    # Allow light markdown/quote wrapping around an otherwise bare number.
    cleaned = text.strip().strip("*`\"'").strip()
    if is_number_only(cleaned):
        return canonicalize_number(cleaned)

    # Fall back to the last number in free-form text.
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return canonicalize_number(matches[-1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for reasoning effort, paths, and model."""
    parser = argparse.ArgumentParser(
        description=(
            "Solve arithmetic operations with GPT-OSS. "
            "Reasoning effort defaults to low."
        )
    )
    effort = parser.add_mutually_exclusive_group()
    effort.add_argument(
        "--instant",
        "-ei",
        action="store_const",
        const="instant",
        dest="reasoning_effort",
        help=(
            "Disable reasoning (final-channel prefill, no Reasoning system line). "
            "Writes output/arithmetic_run_instant.csv"
        ),
    )
    effort.add_argument(
        "--low",
        "-el",
        action="store_const",
        const="low",
        dest="reasoning_effort",
        help="Low reasoning effort (default). Writes output/arithmetic_run.csv",
    )
    effort.add_argument(
        "--medium",
        "-em",
        action="store_const",
        const="medium",
        dest="reasoning_effort",
        help="Medium reasoning effort. Writes output/arithmetic_run_medium.csv",
    )
    effort.add_argument(
        "--high",
        "-eh",
        action="store_const",
        const="high",
        dest="reasoning_effort",
        help="High reasoning effort. Writes output/arithmetic_run_high.csv",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=INPUT_CSV,
        help=f"Input CSV with an 'operation' column (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default depends on reasoning effort: "
            "instant→output/arithmetic_run_instant.csv, "
            "low→output/arithmetic_run.csv, "
            "medium→output/arithmetic_run_medium.csv, "
            "high→output/arithmetic_run_high.csv"
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        default=MODEL,
        help=f"Base model name (default: {MODEL})",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Optional Tinker sampler/weights path (tinker://...). "
            "When set, samples from this checkpoint instead of the base model."
        ),
    )
    parser.set_defaults(reasoning_effort="low")
    return parser.parse_args(argv)


def load_existing_results(output_csv: Path) -> dict[str, str]:
    """Load already-finished answers that are valid number-only outputs."""
    if not output_csv.is_file():
        return {}

    results: dict[str, str] = {}
    with output_csv.open(newline="", encoding="utf-8-sig") as output_file:
        reader = csv.DictReader(output_file)
        if reader.fieldnames is None:
            return {}
        if "operation" not in reader.fieldnames or "output" not in reader.fieldnames:
            return {}

        for row in reader:
            operation = (row.get("operation") or "").strip()
            output = row.get("output") or ""
            if not operation:
                continue
            if is_number_only(output):
                results[operation] = canonicalize_number(output)

    return results


def read_output_operation_order(output_csv: Path) -> list[str]:
    """Return the operation column from the output CSV in file order."""
    if not output_csv.is_file():
        return []

    operations: list[str] = []
    with output_csv.open(newline="", encoding="utf-8-sig") as output_file:
        reader = csv.DictReader(output_file)
        if reader.fieldnames is None or "operation" not in reader.fieldnames:
            return []
        for row in reader:
            operation = (row.get("operation") or "").strip()
            if operation:
                operations.append(operation)
    return operations


def output_order_matches(operations: list[str], output_csv: Path) -> bool:
    """True when output rows are exactly the input operations in the same order."""
    return read_output_operation_order(output_csv) == operations


def write_results_in_order(
    operations: list[str],
    completed: dict[str, str],
    output_csv: Path,
) -> None:
    """Write one row per input operation, always in input order.

    Missing answers are written as an empty output so row order stays aligned
    with the source CSV even when some items are still pending.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(["operation", "output"])
        for operation in operations:
            writer.writerow([operation, completed.get(operation, "")])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reasoning_effort: str = args.reasoning_effort
    input_csv: Path = args.input
    output_csv: Path = (
        args.output if args.output is not None else OUTPUT_BY_EFFORT[reasoning_effort]
    )
    model: str = args.model
    model_path: str | None = args.model_path
    renderer_name = RENDERER_BY_EFFORT[reasoning_effort]
    # Force Harmony final channel for instant mode (skips analysis/CoT).
    generation_prefill = (
        FINAL_CHANNEL_PREFILL if reasoning_effort == "instant" else None
    )

    load_tinker_api_key()
    load_hf_token()

    import tinker
    from tinker_cookbook.renderers import get_renderer, get_text_content

    service_client = tinker.ServiceClient()
    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model)
    renderer = get_renderer(
        renderer_name,
        sampling_client.get_tokenizer(),
        model_name=model,
    )
    sampling_params = tinker.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )

    operations = list(iter_operations(input_csv))
    existing = load_existing_results(output_csv)
    # Only keep answers for operations still present in the input CSV.
    completed = {
        operation: existing[operation]
        for operation in operations
        if operation in existing
    }
    pending = [operation for operation in operations if operation not in completed]
    skipped = len(operations) - len(pending)

    prefill_note = (
        f"\nPrefill: {generation_prefill!r} (reasoning off)"
        if generation_prefill
        else ""
    )
    model_note = f"\nModel path: {model_path}" if model_path else ""
    print(
        f"Model: {model}{model_note}\n"
        f"Reasoning effort: {reasoning_effort} ({renderer_name})"
        f"{prefill_note}\n"
        f"Input: {input_csv}\n"
        f"Output: {output_csv}\n"
        f"Operations: {len(operations)} total, "
        f"{skipped} already done (number-only), "
        f"{len(pending)} to process "
        f"({NUM_IN_FLIGHT} in-flight async tasks)",
        flush=True,
    )

    def write_snapshot() -> None:
        """Rewrite output with every input row in source order.

        Answers are looked up by operation, so a scrambled existing file is
        reordered automatically. Called only from the main coroutine.
        """
        write_results_in_order(operations, completed, output_csv)

    # Always rewrite once up front so a disordered or partial output file is
    # normalized to input order before any new sampling starts.
    if output_csv.is_file() and not output_order_matches(operations, output_csv):
        print(
            f"Output order does not match {input_csv}; reordering {output_csv}",
            flush=True,
        )
    write_snapshot()

    if not pending:
        print("Nothing to do.", flush=True)
        return

    async def run_pending() -> None:
        # Cap concurrent Tinker requests; tasks yield while awaiting I/O.
        in_flight = asyncio.Semaphore(NUM_IN_FLIGHT)
        # Tokenizer/renderer encode & decode are not guaranteed concurrent-safe.
        renderer_lock = asyncio.Lock()

        async def solve_operation(operation: str) -> str:
            async with in_flight:
                async with renderer_lock:
                    model_input = renderer.build_generation_prompt(
                        [{"role": "user", "content": make_prompt(operation)}],
                        prefill=generation_prefill,
                    )
                # sample_async yields to the event loop while waiting on Tinker.
                result = await sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
                async with renderer_lock:
                    response_message, _ = renderer.parse_response(
                        result.sequences[0].tokens
                    )
                    return get_text_content(response_message)

        tasks = {
            operation: asyncio.create_task(solve_operation(operation))
            for operation in pending
        }

        # Drain in input order so progress logs and CSV order stay aligned.
        # Up to NUM_IN_FLIGHT tasks run ahead while we await earlier ones.
        for row_number, operation in enumerate(operations, start=1):
            if operation in existing:
                output = completed[operation]
                status = "skipped"
            else:
                raw_output = await tasks[operation]
                number_only = to_number_only(raw_output)
                if number_only is not None:
                    output = number_only
                    status = "completed"
                    completed[operation] = output
                    write_snapshot()
                else:
                    # Do not persist non-numeric output; next run will redo it.
                    output = escape_newlines(raw_output)
                    status = "invalid-format"

            print(
                f"{status.capitalize()} {row_number}: {operation} -> {output}",
                flush=True,
            )

    asyncio.run(run_pending())

    # Final ordered rewrite after the run.
    write_snapshot()
    if not output_order_matches(operations, output_csv):
        raise RuntimeError(
            f"Output order still does not match input after write: {output_csv}"
        )


if __name__ == "__main__":
    main()
