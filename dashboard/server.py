#!/usr/bin/env python3
"""Serve the math autoresearch live dashboard.

Example::

    source .venv/bin/activate
    python -m dashboard.server --port 8765

    # Point at a specific run
    python -m dashboard.server --run-dir output/autoresearch/math-20260719-120000
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# loop/ is project root; parent sandbox has math scripts.
_LOOP_ROOT = Path(__file__).resolve().parent.parent
_MATH_ROOT = _LOOP_ROOT.parent
for _p in (_LOOP_ROOT, _MATH_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dashboard.data_loader import (  # noqa: E402
    DEFAULT_RUNS_ROOT,
    build_snapshot,
    list_runs,
    resolve_run_dir,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class DashboardState:
    def __init__(
        self,
        *,
        runs_root: Path,
        default_run_dir: Path | None,
    ):
        self.runs_root = runs_root
        self.default_run_dir = default_run_dir


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            # Quieter logs
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: object) -> None:
            body = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path in {"/", "/index.html"}:
                return self._serve_static("index.html")
            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                return self._serve_static(rel)
            # Allow bare asset names for simplicity
            if path in {"/app.js", "/styles.css"}:
                return self._serve_static(path.lstrip("/"))

            if path == "/api/runs":
                return self._send_json(
                    200,
                    {
                        "runs": list_runs(state.runs_root),
                        "runs_root": str(state.runs_root),
                    },
                )

            if path in {"/api/status", "/api/snapshot"}:
                run_id = (qs.get("run") or [None])[0]
                run_dir = resolve_run_dir(
                    run_id,
                    runs_root=state.runs_root,
                    run_dir=state.default_run_dir if not run_id else None,
                )
                # If default_run_dir set and no run_id, prefer it
                if not run_id and state.default_run_dir and state.default_run_dir.is_dir():
                    run_dir = state.default_run_dir
                snap = build_snapshot(run_dir)
                return self._send_json(200, snap)

            if path == "/api/health":
                return self._send_json(200, {"ok": True})

            self._send_json(404, {"ok": False, "error": f"Not found: {path}"})

        def _serve_static(self, rel: str) -> None:
            # Prevent path traversal
            candidate = (STATIC_DIR / rel).resolve()
            if not str(candidate).startswith(str(STATIC_DIR.resolve())):
                return self._send_json(403, {"error": "forbidden"})
            if not candidate.is_file():
                return self._send_json(404, {"error": f"missing static: {rel}"})
            data = candidate.read_bytes()
            ctype, _ = mimetypes.guess_type(str(candidate))
            if ctype is None:
                if candidate.suffix == ".js":
                    ctype = "application/javascript"
                elif candidate.suffix == ".css":
                    ctype = "text/css"
                else:
                    ctype = "application/octet-stream"
            if ctype.startswith("text/") or "javascript" in ctype:
                ctype = f"{ctype}; charset=utf-8"
            self._send(200, data, ctype)

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    p.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Autoresearch runs root (default: {DEFAULT_RUNS_ROOT})",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional fixed run directory to feature",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open browser after start",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="If no runs exist, write a small demo run under runs-root",
    )
    return p.parse_args(argv)


def _ensure_demo_run(runs_root: Path) -> Path:
    """Create a synthetic run so the UI is non-empty for first-time viewing."""
    from datetime import datetime, timezone

    runs_root.mkdir(parents=True, exist_ok=True)
    run = runs_root / "demo-math-preview"
    if (run / "status.json").is_file():
        return run
    run.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    def wj(name: str, obj: object) -> None:
        (run / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    def wjl(name: str, rows: list) -> None:
        with (run / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    history = []
    for it in range(3):
        ts_acc = 0.12 + it * 0.08
        ho_acc = 0.10 + it * 0.05
        history.append(
            {
                "iteration": it,
                "train_pool_size": 4500 + it * 120,
                "new_examples": 80 + it * 10,
                "train_seed": {
                    "pre": {
                        "instant": {"accuracy": ts_acc - 0.03, "correct": 10, "completed": 100},
                        "high": {"accuracy": 0.55},
                        "accuracy_gap": 0.4,
                    },
                    "post": {
                        "instant": {"accuracy": ts_acc, "correct": 12 + it * 4, "completed": 100},
                        "high": {"accuracy": 0.58 + it * 0.02},
                        "accuracy_gap": 0.35,
                    },
                    "delta": {"instant_accuracy": 0.03},
                },
                "heldout": {
                    "pre": {
                        "instant": {"accuracy": ho_acc - 0.02, "correct": 8, "completed": 100},
                        "high": {"accuracy": 0.5},
                        "accuracy_gap": 0.4,
                    },
                    "post": {
                        "instant": {"accuracy": ho_acc, "correct": 10 + it * 3, "completed": 100},
                        "high": {"accuracy": 0.52 + it * 0.02},
                        "accuracy_gap": 0.38,
                    },
                    "delta": {"instant_accuracy": 0.02},
                },
                "delta": {
                    "overfit_gap": ts_acc - ho_acc,
                    "train_seed_instant": 0.03,
                    "heldout_instant": 0.02,
                },
                "train": {
                    "state_path": f"tinker://demo/weights/{it:03d}020",
                    "sampler_path": f"tinker://demo/sampler_weights/{it:03d}020",
                    "policy_source": "base" if it == 0 else "checkpoint",
                    "judge_source": "base" if it == 0 else "checkpoint",
                },
            }
        )
        idir = run / f"iter_{it:03d}"
        idir.mkdir(exist_ok=True)
        (idir / "generated.csv").write_text(
            "operation,solution\n1+1,2\n2+2,4\n", encoding="utf-8"
        )
        (idir / "validated.csv").write_text(
            "operation,solution\n1+1,2\n", encoding="utf-8"
        )
        (idir / "train").mkdir(exist_ok=True)
        (idir / "eval_pre" / "train_seed").mkdir(parents=True, exist_ok=True)
        (idir / "eval_pre" / "heldout").mkdir(parents=True, exist_ok=True)
        (idir / "eval_post" / "train_seed").mkdir(parents=True, exist_ok=True)
        (idir / "eval_post" / "heldout").mkdir(parents=True, exist_ok=True)
        wj_path = idir / "metrics.json"
        wj_path.write_text(json.dumps(history[-1], indent=2) + "\n", encoding="utf-8")

    wjl("history.jsonl", history)
    ckpts = []
    for it in range(3):
        ckpts.append(
            {
                "iteration": it,
                "name": f"{it:03d}010",
                "batch": 10,
                "state_path": f"tinker://demo/weights/{it:03d}010",
                "sampler_path": f"tinker://demo/sampler_weights/{it:03d}010",
            }
        )
        ckpts.append(
            {
                "iteration": it,
                "name": f"{it:03d}020",
                "batch": 20,
                "state_path": f"tinker://demo/weights/{it:03d}020",
                "sampler_path": f"tinker://demo/sampler_weights/{it:03d}020",
            }
        )
    wjl("checkpoints.jsonl", ckpts)
    wjl(
        "events.jsonl",
        [
            {"ts": now, "kind": "phase", "phase": "init", "message": "demo run", "iteration": 0},
            {"ts": now, "kind": "eval", "message": "pre heldout", "iteration": 2},
            {"ts": now, "kind": "train_done", "message": "train finished", "iteration": 2},
        ],
    )
    wjl(
        "data_snapshots.jsonl",
        [
            {"ts": now, "iteration": 0, "source": "bootstrap_train", "size": 4500},
            {"ts": now, "iteration": 1, "source": "post_validate", "size": 4620},
            {"ts": now, "iteration": 2, "source": "post_validate", "size": 4740},
        ],
    )
    last = history[-1]
    wj(
        "status.json",
        {
            "run_dir": str(run),
            "phase": "completed",
            "iteration": 3,
            "message": "Demo preview data",
            "stopped": True,
            "stop_reason": "demo",
            "started_at": now,
            "updated_at": now,
            "train_pool_size": 4740,
            "heldout_size": 500,
            "last_instant_accuracy": last["train_seed"]["post"]["instant"]["accuracy"],
            "last_heldout_instant_accuracy": last["heldout"]["post"]["instant"]["accuracy"],
            "last_overfit_gap": last["delta"]["overfit_gap"],
            "last_instant_delta": 0.03,
            "last_heldout_instant_delta": 0.02,
            "last_policy_sampler_path": "tinker://demo/sampler_weights/002020",
            "last_policy_state_path": "tinker://demo/weights/002020",
            "last_judge_model_path": "tinker://demo/sampler_weights/002020",
            "last_checkpoint_name": "002020",
            "policy_source": "checkpoint",
            "judge_source": "checkpoint",
        },
    )
    wj(
        "state.json",
        {
            "run_dir": str(run),
            "seed_data": "data/arithmetic_operations.csv",
            "train_data": str(run / "train_data.csv"),
            "eval_sample": str(run / "eval_sample.csv"),
            "heldout_test": str(run / "heldout_test.csv"),
            "eval_heldout_sample": str(run / "eval_heldout_sample.csv"),
            "heldout_fraction": 0.1,
            "iteration": 3,
            "last_policy_state_path": "tinker://demo/weights/002020",
            "last_policy_sampler_path": "tinker://demo/sampler_weights/002020",
            "last_judge_model_path": "tinker://demo/sampler_weights/002020",
        },
    )
    wj(
        "split_manifest.json",
        {
            "n_train_initial": 4500,
            "n_heldout": 500,
            "n_eval_train_sample": 100,
            "n_eval_heldout_sample": 100,
            "heldout_fraction": 0.1,
            "note": "Demo split",
        },
    )
    wj(
        "train_progress.json",
        {
            "iteration": 2,
            "checkpoints": ckpts[-2:],
            "metrics_tail": [{"step": 20, "reward_mean": 0.4}],
        },
    )
    return run


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs_root = args.runs_root.resolve()
    if args.demo or not list_runs(runs_root):
        if args.demo or not runs_root.is_dir() or not any(runs_root.iterdir()):
            demo = _ensure_demo_run(runs_root)
            print(f"Demo run ready: {demo}", flush=True)
    state = DashboardState(
        runs_root=runs_root,
        default_run_dir=args.run_dir.resolve() if args.run_dir else None,
    )
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(
        f"Math autoresearch dashboard\n"
        f"  URL:       {url}\n"
        f"  Runs root: {state.runs_root}\n"
        f"  Run dir:   {state.default_run_dir or '(latest)'}\n"
        f"  API:       {url}api/status\n",
        flush=True,
    )
    if args.open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
