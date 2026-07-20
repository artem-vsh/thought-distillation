#!/usr/bin/env python3
"""Test model answers in arithmetic_run*.csv and write results to *_test.csv."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import operator as op
import re
from functools import reduce
from math import comb, factorial, gcd, isqrt, lcm, perm
from pathlib import Path


# Match ask_arithmetic.py: defaults under this project root (loop/).
_PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_BY_EFFORT = {
    "instant": _PROJECT_ROOT / "output" / "arithmetic_run_instant.csv",
    "low": _PROJECT_ROOT / "output" / "arithmetic_run.csv",
    "medium": _PROJECT_ROOT / "output" / "arithmetic_run_medium.csv",
    "high": _PROJECT_ROOT / "output" / "arithmetic_run_high.csv",
}
OUTPUT_BY_EFFORT = {
    "instant": _PROJECT_ROOT / "output" / "arithmetic_run_instant_test.csv",
    "low": _PROJECT_ROOT / "output" / "arithmetic_run_test.csv",
    "medium": _PROJECT_ROOT / "output" / "arithmetic_run_medium_test.csv",
    "high": _PROJECT_ROOT / "output" / "arithmetic_run_high_test.csv",
}

# Safe subset of Python arithmetic (no attributes / free names).
_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

# Integers or decimals, optional leading sign / scientific notation.
_NUMBER_RE = re.compile(
    r"[+-]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?"
)

# Cap factorial / fib inputs so results stay exact in float comparisons.
_MAX_FACTORIAL = 20
_MAX_FIB = 70
_MAX_POW_EXP = 40
_MAX_POW_BASE = 10**12
# Cap trial-division / combinatorial inputs so evaluation stays fast even on
# adversarial arguments (sigma/phi are O(√n); comb/perm grow combinatorially).
_MAX_SIGMA_PHI = 10**12
_MAX_COMB_N = 10**4


def _to_int(value: float | int, name: str) -> int:
    """Require a whole number argument for discrete functions."""
    if isinstance(value, bool):
        raise ValueError(f"{name} does not accept booleans")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{name} requires an integer argument, got {value!r}")


def _fib(n: int) -> int:
    if n < 0:
        raise ValueError("fib requires n >= 0")
    if n > _MAX_FIB:
        raise ValueError(f"fib n too large (max {_MAX_FIB})")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _digit_sum(n: int) -> int:
    n = abs(n)
    total = 0
    while n:
        total += n % 10
        n //= 10
    return total


def _digit_product(n: int) -> int:
    n = abs(n)
    if n == 0:
        return 0
    product = 1
    while n:
        product *= n % 10
        n //= 10
    return product


def _reverse_digits(n: int) -> int:
    sign = -1 if n < 0 else 1
    s = str(abs(n))[::-1].lstrip("0") or "0"
    return sign * int(s)


def _sigma(n: int) -> int:
    """Sum of positive divisors of n."""
    if n <= 0:
        raise ValueError("sigma requires n > 0")
    if n > _MAX_SIGMA_PHI:
        raise ValueError(f"sigma n too large (max {_MAX_SIGMA_PHI})")
    total = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            total += i
            other = n // i
            if other != i:
                total += other
        i += 1
    return total


def _phi(n: int) -> int:
    """Euler's totient φ(n)."""
    if n <= 0:
        raise ValueError("phi requires n > 0")
    if n > _MAX_SIGMA_PHI:
        raise ValueError(f"phi n too large (max {_MAX_SIGMA_PHI})")
    result = n
    p = 2
    x = n
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1
    if x > 1:
        result -= result // x
    return result


def _prime_count(n: int) -> int:
    """Number of primes ≤ n (π(n))."""
    if n < 2:
        return 0
    if n > 200_000:
        raise ValueError("prime_count n too large")
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return int(sum(sieve))


def _nth_prime(n: int) -> int:
    if n < 1:
        raise ValueError("nth_prime requires n >= 1")
    if n > 10_000:
        raise ValueError("nth_prime n too large")
    # Prime number theorem bound with slack.
    if n < 6:
        limit = 15
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n))) * 1.3) + 20
    while True:
        sieve = bytearray(b"\x01") * (limit + 1)
        sieve[0:2] = b"\x00\x00"
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                start = i * i
                sieve[start : limit + 1 : i] = b"\x00" * (
                    ((limit - start) // i) + 1
                )
        primes = [i for i in range(2, limit + 1) if sieve[i]]
        if len(primes) >= n:
            return primes[n - 1]
        limit *= 2


def _safe_factorial(n: int) -> int:
    if n < 0:
        raise ValueError("factorial requires n >= 0")
    if n > _MAX_FACTORIAL:
        raise ValueError(f"factorial n too large (max {_MAX_FACTORIAL})")
    return factorial(n)


def _safe_comb(n: int, k: int) -> int:
    if n > _MAX_COMB_N or k > _MAX_COMB_N:
        raise ValueError(f"comb arguments too large (max {_MAX_COMB_N})")
    return comb(n, k)


def _safe_perm(n: int, k: int) -> int:
    if n > _MAX_COMB_N or k > _MAX_COMB_N:
        raise ValueError(f"perm arguments too large (max {_MAX_COMB_N})")
    return perm(n, k)


def _safe_pow(base: float | int, exp: float | int) -> float:
    b = float(base)
    e = float(exp)
    if abs(b) > _MAX_POW_BASE:
        raise ValueError("pow base too large")
    if e != int(e) or int(e) < 0 or int(e) > _MAX_POW_EXP:
        # Allow modest real powers only when result is well-defined and finite.
        if e < 0 or e > 20:
            raise ValueError("pow exponent out of range")
    result = b**e
    if not math.isfinite(result):
        raise ValueError("pow overflow")
    if abs(result) > 1e16:
        raise ValueError("pow result too large")
    return result


def _safe_sqrt(x: float | int) -> float:
    value = float(x)
    if value < 0:
        raise ValueError("sqrt of negative")
    root = math.sqrt(value)
    # Prefer exact integers for perfect squares.
    rounded = round(root)
    if abs(root - rounded) <= 1e-12 * max(1.0, abs(root)):
        return float(rounded)
    return root


def _multi_gcd(*args: float | int) -> int:
    if not args:
        raise ValueError("gcd requires at least one argument")
    values = [_to_int(a, "gcd") for a in args]
    return abs(reduce(gcd, values))


def _multi_lcm(*args: float | int) -> int:
    if not args:
        raise ValueError("lcm requires at least one argument")
    values = [abs(_to_int(a, "lcm")) for a in args]
    if any(v == 0 for v in values):
        return 0
    return int(reduce(lcm, values))


# Whitelisted callables available inside operation expressions.
_FUNCTIONS: dict[str, object] = {
    "factorial": lambda n: _safe_factorial(_to_int(n, "factorial")),
    "fact": lambda n: _safe_factorial(_to_int(n, "fact")),
    "comb": lambda n, k: _safe_comb(_to_int(n, "comb"), _to_int(k, "comb")),
    "C": lambda n, k: _safe_comb(_to_int(n, "C"), _to_int(k, "C")),
    "perm": lambda n, k: _safe_perm(_to_int(n, "perm"), _to_int(k, "perm")),
    "P": lambda n, k: _safe_perm(_to_int(n, "P"), _to_int(k, "P")),
    "gcd": _multi_gcd,
    "lcm": _multi_lcm,
    "abs": lambda x: abs(x),
    "floor": lambda x: float(math.floor(float(x))),
    "ceil": lambda x: float(math.ceil(float(x))),
    "round": lambda x: float(round(float(x))),
    "sqrt": _safe_sqrt,
    "isqrt": lambda n: float(isqrt(_to_int(n, "isqrt"))),
    "fib": lambda n: _fib(_to_int(n, "fib")),
    "digit_sum": lambda n: _digit_sum(_to_int(n, "digit_sum")),
    "digit_product": lambda n: _digit_product(_to_int(n, "digit_product")),
    "reverse_digits": lambda n: _reverse_digits(_to_int(n, "reverse_digits")),
    "sigma": lambda n: _sigma(_to_int(n, "sigma")),
    "phi": lambda n: _phi(_to_int(n, "phi")),
    "totient": lambda n: _phi(_to_int(n, "totient")),
    "prime_count": lambda n: _prime_count(_to_int(n, "prime_count")),
    "nth_prime": lambda n: _nth_prime(_to_int(n, "nth_prime")),
    "pow": _safe_pow,
    "mod": lambda a, b: _to_int(a, "mod") % _to_int(b, "mod"),
    "max": lambda *args: max(float(a) for a in args),
    "min": lambda *args: min(float(a) for a in args),
}


def _rewrite_caret_powers(expression: str) -> str:
    """Rewrite a^b as a**b (bitwise XOR is not supported)."""
    return expression.replace("^", "**")


def _rewrite_factorials(expression: str) -> str:
    """Rewrite postfix factorial (e.g. 12! or (n+1)!) as factorial(...)."""
    text = expression
    # Repeat until stable so that 5!! is rejected / handled step by step.
    # Process from the left; each '!' applies to the complete primary before it.
    while True:
        idx = text.find("!")
        if idx < 0:
            return text
        if idx == 0:
            raise ValueError("leading factorial")
        # Skip if part of a comparison we don't support (!=); treat as error.
        if idx + 1 < len(text) and text[idx + 1] == "=":
            raise ValueError("!= is not supported")

        j = idx - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j < 0:
            raise ValueError("factorial without operand")

        if text[j].isdigit():
            start = j
            while start > 0 and text[start - 1].isdigit():
                start -= 1
            operand = text[start:j + 1]
            text = f"{text[:start]}factorial({operand}){text[idx + 1:]}"
            continue

        if text[j] == ")":
            depth = 0
            k = j
            while k >= 0:
                if text[k] == ")":
                    depth += 1
                elif text[k] == "(":
                    depth -= 1
                    if depth == 0:
                        break
                k -= 1
            if depth != 0 or k < 0:
                raise ValueError("unbalanced parentheses in factorial")
            # Include function name before '(' if present: fib(10)!
            start = k
            name_end = k
            p = k - 1
            while p >= 0 and text[p].isspace():
                p -= 1
            if p >= 0 and (text[p].isalnum() or text[p] == "_"):
                name_start = p
                while name_start > 0 and (
                    text[name_start - 1].isalnum() or text[name_start - 1] == "_"
                ):
                    name_start -= 1
                start = name_start
            operand = text[start : j + 1]
            text = f"{text[:start]}factorial({operand}){text[idx + 1:]}"
            continue

        raise ValueError(f"invalid factorial operand near {text[max(0, j - 5): idx + 1]!r}")


def _preprocess_operation(operation: str) -> str:
    """Normalize surface syntax into a Python expression subset."""
    text = operation.strip()
    if not text:
        raise ValueError("empty operation")
    text = _rewrite_caret_powers(text)
    text = _rewrite_factorials(text)
    return text


def evaluate_operation(operation: str) -> float:
    """Evaluate an arithmetic / discrete-math expression and return its result.

    Supported:
      - +, -, *, /, //, %, ** (and ^), unary ±, parentheses
      - postfix factorial: n! or (expr)!
      - functions: factorial/fact, comb/C, perm/P, gcd, lcm, abs, floor, ceil,
        round, sqrt, isqrt, fib, digit_sum, digit_product, reverse_digits,
        sigma, phi/totient, prime_count, nth_prime, pow, mod, max, min
    """

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ValueError("division by zero")
            if isinstance(node.op, ast.Pow):
                return float(_safe_pow(left, right))
            return float(_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only simple function calls are allowed")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
            name = node.func.id
            if name not in _FUNCTIONS:
                raise ValueError(f"unsupported function: {name}")
            fn = _FUNCTIONS[name]
            args = [_eval(arg) for arg in node.args]
            result = fn(*args)  # type: ignore[operator]
            return float(result)
        if isinstance(node, ast.Name):
            raise ValueError(f"unsupported name: {node.id}")
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    prepared = _preprocess_operation(operation)
    tree = ast.parse(prepared, mode="eval")
    return _eval(tree)


def unescape_output(value: str) -> str:
    """Turn escaped newline markers back into real newlines for parsing."""
    return value.replace(r"\n", "\n").replace(r"\r", "\r")


def extract_numeric_answer(output: str) -> float | None:
    """Pull a numeric answer from free-form model output.

    Prefers a bare whole-string number; otherwise uses the last number found
    (typical for chain-of-thought that ends with the final value).
    """
    text = unescape_output(output).strip()
    if not text:
        return None

    # Exact numeric string (optionally wrapped in simple markdown/punctuation).
    cleaned = text.strip().strip("*`\"'").strip()
    # Strip trailing period often left after "The answer is 42."
    if cleaned.endswith(".") and cleaned.count(".") == 1:
        cleaned = cleaned[:-1].strip()
    try:
        return float(cleaned)
    except ValueError:
        pass

    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def numbers_equal(expected: float, actual: float) -> bool:
    """Compare answers allowing tiny float error; treat near-integers as ints."""
    if not math.isfinite(expected) or not math.isfinite(actual):
        return False
    if abs(expected - actual) <= 1e-9:
        return True
    # Exact integer agreement within float mantissa when both are whole.
    if (
        abs(expected - round(expected)) <= 1e-9
        and abs(actual - round(actual)) <= 1e-9
        and abs(expected) <= 2**53
        and abs(actual) <= 2**53
        and int(round(expected)) == int(round(actual))
    ):
        return True
    # Relative tolerance for large values from division chains.
    scale = max(abs(expected), abs(actual), 1.0)
    return abs(expected - actual) <= 1e-9 * scale


def is_completed(output: str) -> bool:
    """True when the row has a non-empty model answer (task was finished).

    Incomplete rows (empty output) are placeholders left by resume runs and
    must not be counted in accuracy.
    """
    return bool(unescape_output(output).strip())


def is_correct(operation: str, output: str) -> bool:
    """Return True when the model output matches the operation's result."""
    try:
        expected = evaluate_operation(operation)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError, SyntaxError):
        return False
    actual = extract_numeric_answer(output)
    if actual is None:
        return False
    return numbers_equal(expected, actual)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for reasoning effort and I/O paths (same as ask_arithmetic)."""
    parser = argparse.ArgumentParser(
        description=(
            "Score model answers in an arithmetic run CSV. "
            "Reasoning effort defaults to low (same flags as ask_arithmetic)."
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
            "No-reasoning / instant run. "
            "Reads output/arithmetic_run_instant.csv, "
            "writes output/arithmetic_run_instant_test.csv"
        ),
    )
    effort.add_argument(
        "--low",
        "-el",
        action="store_const",
        const="low",
        dest="reasoning_effort",
        help=(
            "Low reasoning effort (default). "
            "Reads output/arithmetic_run.csv, "
            "writes output/arithmetic_run_test.csv"
        ),
    )
    effort.add_argument(
        "--medium",
        "-em",
        action="store_const",
        const="medium",
        dest="reasoning_effort",
        help=(
            "Medium reasoning effort. "
            "Reads output/arithmetic_run_medium.csv, "
            "writes output/arithmetic_run_medium_test.csv"
        ),
    )
    effort.add_argument(
        "--high",
        "-eh",
        action="store_const",
        const="high",
        dest="reasoning_effort",
        help=(
            "High reasoning effort. "
            "Reads output/arithmetic_run_high.csv, "
            "writes output/arithmetic_run_high_test.csv"
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help=(
            "Input CSV with operation/output columns. Default depends on "
            "reasoning effort: "
            "instant→output/arithmetic_run_instant.csv, "
            "low→output/arithmetic_run.csv, "
            "medium→output/arithmetic_run_medium.csv, "
            "high→output/arithmetic_run_high.csv"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default depends on reasoning effort: "
            "instant→output/arithmetic_run_instant_test.csv, "
            "low→output/arithmetic_run_test.csv, "
            "medium→output/arithmetic_run_medium_test.csv, "
            "high→output/arithmetic_run_high_test.csv"
        ),
    )
    parser.set_defaults(reasoning_effort="low")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reasoning_effort: str = args.reasoning_effort
    input_csv: Path = (
        args.input if args.input is not None else INPUT_BY_EFFORT[reasoning_effort]
    )
    output_csv: Path = (
        args.output if args.output is not None else OUTPUT_BY_EFFORT[reasoning_effort]
    )

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    passed = 0
    completed = 0
    incomplete = 0
    rows = 0

    with (
        input_csv.open(newline="", encoding="utf-8-sig") as input_file,
        output_csv.open("w", newline="", encoding="utf-8") as output_file,
    ):
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_csv}")
        required = {"operation", "output"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV must contain columns {sorted(required)}; "
                f"missing {sorted(missing)}: {input_csv}"
            )

        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(["operation", "output", "correct"])

        for line_number, row in enumerate(reader, start=2):
            operation = (row.get("operation") or "").strip()
            output = row.get("output") or ""
            if not operation:
                raise ValueError(f"CSV row {line_number} has an empty operation")

            rows += 1
            if not is_completed(output):
                # Resume placeholders: do not score, do not dilute accuracy.
                incomplete += 1
                writer.writerow([operation, output, "incomplete"])
                continue

            completed += 1
            correct = is_correct(operation, output)
            writer.writerow([operation, output, "true" if correct else "false"])
            if correct:
                passed += 1

    accuracy = (100.0 * passed / completed) if completed else 0.0
    print(
        f"Reasoning effort: {reasoning_effort}\n"
        f"Input: {input_csv}\n"
        f"Wrote {output_csv}: {passed}/{completed} correct "
        f"({accuracy:.1f}%) among completed"
        f"{f'; {incomplete} incomplete skipped' if incomplete else ''}"
        f" ({rows} rows total)",
        flush=True,
    )


if __name__ == "__main__":
    main()
