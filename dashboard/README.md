# Math autoresearch dashboard

Live UI for `output/autoresearch/<run>/` artifacts.

Run from the **loop/** project root:

```bash
cd loop
source ../.venv/bin/activate
python -m dashboard --port 8765 --open

# Pin a run
python -m dashboard --run-dir output/autoresearch/math-YYYYMMDD-HHMMSS

# Seed demo data if empty
python -m dashboard --demo --open
```

Open **http://127.0.0.1:8765/**

API: `GET /api/runs`, `GET /api/status`, `GET /api/status?run=<id>`

Polls every 2s (toggle Auto in the UI). No extra pip deps beyond the parent
sandbox env (stdlib HTTP server + Chart.js CDN).
