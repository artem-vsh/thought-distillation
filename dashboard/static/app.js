/* Thought Distillation live dashboard */

const $ = (id) => document.getElementById(id);

const state = {
  auto: true,
  runId: null,
  timer: null,
  accChart: null,
  lastPayload: null,
};

const COLORS = {
  instant: "#c084fc",
  high: "#fbbf24",
  liveInstant: "#fb923c",
  liveHigh: "#fdba74",
};

/** Draw ±CI error bars only for the hovered point(s). */
const errorBarsOnHover = {
  id: "errorBarsOnHover",
  afterDatasetsDraw(chart) {
    const active = chart.getActiveElements();
    if (!active.length) return;
    const { ctx } = chart;
    const yScale = chart.scales.y;
    for (const el of active) {
      const ds = chart.data.datasets[el.datasetIndex];
      const raw = ds?.data?.[el.index];
      if (!raw || raw.yMin == null || raw.yMax == null) continue;
      const meta = chart.getDatasetMeta(el.datasetIndex);
      const point = meta.data[el.index];
      if (!point) continue;
      const x = point.x;
      const yTop = yScale.getPixelForValue(raw.yMax);
      const yBot = yScale.getPixelForValue(raw.yMin);
      const color = typeof ds.borderColor === "string" ? ds.borderColor : COLORS.instant;
      const cap = 6;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.75;
      ctx.lineCap = "round";
      ctx.globalAlpha = 0.95;
      ctx.beginPath();
      ctx.moveTo(x, yTop);
      ctx.lineTo(x, yBot);
      ctx.moveTo(x - cap, yTop);
      ctx.lineTo(x + cap, yTop);
      ctx.moveTo(x - cap, yBot);
      ctx.lineTo(x + cap, yBot);
      ctx.stroke();
      ctx.restore();
    }
  },
};

function timeAgo(iso) {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const sec = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return iso.slice(0, 19).replace("T", " ") + "Z";
}

async function fetchJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

function setLive(ok, text) {
  const pill = $("livePill");
  const label = $("liveText");
  pill.classList.toggle("offline", !ok);
  label.textContent = text;
}

function chartDefaults() {
  Chart.defaults.color = "#8b9bb8";
  Chart.defaults.borderColor = "rgba(148,163,184,0.12)";
  Chart.defaults.font.family = "Segoe UI, system-ui, sans-serif";
}

function pctLabel(y) {
  if (y == null || Number.isNaN(y)) return "—";
  return `${(100 * Number(y)).toFixed(1)}%`;
}

function ensureCharts() {
  if (typeof Chart === "undefined") return;
  chartDefaults();
  if (!state.accChart) {
    state.accChart = new Chart($("accChart"), {
      type: "line",
      data: { datasets: [] },
      plugins: [errorBarsOnHover],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: false, axis: "x" },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => {
                const x = items[0]?.parsed?.x;
                if (x === 0) return "Generation 0 · baseline";
                if (x == null) return "";
                return `Generation ${x}`;
              },
              label: (ctx) => {
                const raw = ctx.raw || {};
                const name = ctx.dataset.label || "";
                const base = `${name}: ${pctLabel(raw.y ?? ctx.parsed.y)}`;
                if (raw.ciHalf != null && !Number.isNaN(raw.ciHalf)) {
                  return `${base}  ±${(100 * raw.ciHalf).toFixed(1)} pp`;
                }
                return base;
              },
              afterLabel: (ctx) => {
                const raw = ctx.raw || {};
                if (raw.yMin == null || raw.yMax == null) return "";
                return `CI [${pctLabel(raw.yMin)}, ${pctLabel(raw.yMax)}]`;
              },
            },
          },
        },
        scales: {
          y: {
            min: 0,
            max: 1,
            ticks: {
              callback: (v) => `${Math.round(100 * v)}%`,
            },
            grid: { color: "rgba(148,163,184,0.08)" },
          },
          x: {
            type: "linear",
            title: { display: true, text: "Generation (0 = baseline)" },
            min: 0,
            suggestedMax: 2,
            ticks: {
              stepSize: 1,
              precision: 0,
              callback: (v) => (Number.isInteger(v) ? String(v) : ""),
            },
            grid: { color: "rgba(148,163,184,0.06)" },
          },
        },
      },
    });
  }
}

/** Points with optional CI half-width → {x, y, yMin, yMax, ciHalf}. */
function xyWithCi(arr, key, halfKey) {
  return arr
    .filter((p) => p[key] != null && !Number.isNaN(p[key]))
    .map((p) => {
      const y = Number(p[key]);
      const half = p[halfKey];
      const pt = { x: p.generation, y };
      if (half != null && !Number.isNaN(Number(half))) {
        const h = Number(half);
        pt.ciHalf = h;
        pt.yMin = Math.max(0, y - h);
        pt.yMax = Math.min(1, y + h);
      }
      return pt;
    });
}

function seriesLine(label, data, color, opts = {}) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color + "33",
    tension: 0.2,
    borderWidth: opts.borderWidth ?? 2.4,
    pointRadius: opts.pointRadius ?? 5,
    pointHoverRadius: 8,
    pointStyle: opts.pointStyle || "circle",
    spanGaps: true,
    borderDash: opts.borderDash || [],
    showLine: opts.showLine !== false,
    order: opts.order ?? 10,
    parsing: false,
  };
}

/**
 * Merge live eval into the generation timeline so each gen has at most
 * two values (instant + high). Never plot a separate live overlay on top
 * of an existing gen — that was causing four dots at one x.
 */
function mergeLiveIntoPoints(points, live) {
  const merged = points.map((p) => ({ ...p }));
  // Interim history gens are incomplete — style them as live even without overlay.
  const interimGens = new Set(
    merged.filter((p) => p.kind === "interim" || p.ci_refine_pending).map((p) => Number(p.generation))
  );

  if (!live || live.generation == null) {
    const only = interimGens.size ? [...interimGens][interimGens.size - 1] : null;
    return { points: merged, liveGen: only };
  }
  const gx = Number(live.generation);
  const idx = merged.findIndex((p) => Number(p.generation) === gx);
  const overlay = {
    generation: gx,
    kind: "live",
    label: live.label || `gen ${gx} · live`,
    heldout_instant: live.heldout_instant,
    heldout_high: live.heldout_high,
    heldout_instant_ci_half: live.heldout_instant_ci_half,
    heldout_high_ci_half: live.heldout_high_ci_half,
  };
  if (idx >= 0) {
    const prev = merged[idx];
    merged[idx] = {
      ...prev,
      kind: prev.kind === "interim" ? "interim" : "live",
      label: overlay.label,
      // Prefer live/refine metrics when present (updates interim in place).
      heldout_instant:
        overlay.heldout_instant != null ? overlay.heldout_instant : prev.heldout_instant,
      heldout_high:
        overlay.heldout_high != null ? overlay.heldout_high : prev.heldout_high,
      heldout_instant_ci_half:
        overlay.heldout_instant_ci_half != null
          ? overlay.heldout_instant_ci_half
          : prev.heldout_instant_ci_half,
      heldout_high_ci_half:
        overlay.heldout_high_ci_half != null
          ? overlay.heldout_high_ci_half
          : prev.heldout_high_ci_half,
    };
  } else {
    // Only add a new x-slot when live has its own metrics (not empty marker).
    if (overlay.heldout_instant != null || overlay.heldout_high != null) {
      merged.push(overlay);
    }
  }
  interimGens.add(gx);
  return { points: merged, liveGen: gx, interimGens };
}

function seriesWithLiveStyle(label, data, color, liveGen, interimGens, opts = {}) {
  const ds = seriesLine(label, data, color, opts);
  const isHot = (x) =>
    (liveGen != null && Number(x) === Number(liveGen)) ||
    (interimGens && interimGens.has(Number(x)));
  // Per-point diamond for interim / live gens (still only two series total).
  ds.pointStyle = data.map((p) => (isHot(p.x) ? "rectRot" : "circle"));
  ds.pointRadius = data.map((p) => (isHot(p.x) ? 9 : opts.pointRadius ?? 6));
  ds.pointHoverRadius = data.map((p) => (isHot(p.x) ? 11 : 8));
  ds.pointBackgroundColor = data.map((p) =>
    isHot(p.x)
      ? label.toLowerCase().includes("high")
        ? COLORS.liveHigh
        : COLORS.liveInstant
      : color
  );
  ds.pointBorderColor = ds.pointBackgroundColor;
  return ds;
}

function updateCharts(snap) {
  if (typeof Chart === "undefined") return;
  ensureCharts();
  const chart = snap.evals?.series?.chart || {};
  const live = chart.live || snap.in_progress || null;
  const { points, liveGen, interimGens } = mergeLiveIntoPoints(chart.points || [], live);
  const hotGens = interimGens || new Set(liveGen != null ? [liveGen] : []);

  // Exactly two series: held-out instant + high. Interim/live restyled in-place.
  const datasets = [
    seriesWithLiveStyle(
      "Instant",
      xyWithCi(points, "heldout_instant", "heldout_instant_ci_half"),
      COLORS.instant,
      liveGen,
      hotGens,
      { pointRadius: 6 }
    ),
    seriesWithLiveStyle(
      "High reasoning",
      xyWithCi(points, "heldout_high", "heldout_high_ci_half"),
      COLORS.high,
      liveGen,
      hotGens,
      { pointRadius: 6, borderDash: [5, 4] }
    ),
  ];

  const xMax = Math.max(
    chart.x_suggested_max ?? 2,
    chart.x_max ?? 0,
    liveGen ?? 0,
    2
  );
  state.accChart.options.scales.x.min = 0;
  state.accChart.options.scales.x.suggestedMax = xMax;
  state.accChart.options.scales.x.max = xMax;
  state.accChart.data.datasets = datasets.filter((d) => (d.data || []).length > 0);
  state.accChart.update("none");

  const hint = $("chartHint");
  if (hint) {
    const nFinal = points.filter((p) => p.kind === "finalized" || p.kind === "imported").length;
    const nInterim = points.filter((p) => p.kind === "interim").length;
    const parts = ["instant + high · hover for ±CI"];
    if (nFinal) parts.push(`${nFinal} finalized`);
    if (nInterim) parts.push(`${nInterim} interim`);
    if (live) parts.push(live.label || "live");
    hint.textContent = parts.join(" · ");
  }
}

function fillRunSelect(runs, currentId) {
  const sel = $("runSelect");
  const prev = currentId || state.runId;
  sel.innerHTML = "";
  if (!runs.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No runs in output/autoresearch";
    sel.appendChild(opt);
    return;
  }
  for (const r of runs) {
    const opt = document.createElement("option");
    opt.value = r.id;
    const phase = r.phase ? ` · ${r.phase}` : "";
    opt.textContent = `${r.id}${phase}`;
    sel.appendChild(opt);
  }
  if (prev && [...sel.options].some((o) => o.value === prev)) {
    sel.value = prev;
  }
  state.runId = sel.value || null;
}

async function refresh() {
  try {
    const q = state.runId ? `?run=${encodeURIComponent(state.runId)}` : "";
    const snap = await fetchJSON(`/api/status${q}`);
    state.lastPayload = snap;
    fillRunSelect(snap.runs || [], snap.run?.id);

    if (!snap.ok) {
      setLive(true, "no run");
      if (state.accChart) {
        state.accChart.data.datasets = [];
        state.accChart.update("none");
      }
      const hint = $("chartHint");
      if (hint) hint.textContent = snap.error || "No active run — start python -m autoresearch";
      $("footerLeft").textContent = `Updated ${new Date().toLocaleTimeString()} · ${snap.error || "no run"}`;
      return;
    }

    setLive(true, `live · ${timeAgo(snap.headline?.updated_at || snap.generated_at)}`);
    updateCharts(snap);

    $("footerLeft").textContent =
      `Run ${snap.run.id} · updated ${timeAgo(snap.headline?.updated_at)} · polled ${new Date().toLocaleTimeString()}`;
    $("footerRight").textContent = snap.run.path;
  } catch (err) {
    setLive(false, "offline");
    $("footerLeft").textContent = `Error: ${err.message}`;
  }
}

function schedule() {
  if (state.timer) clearInterval(state.timer);
  if (state.auto) {
    state.timer = setInterval(refresh, 2000);
  }
}

function init() {
  $("refreshBtn").addEventListener("click", () => refresh());
  $("autoBtn").addEventListener("click", () => {
    state.auto = !state.auto;
    $("autoBtn").textContent = state.auto ? "Auto: ON" : "Auto: OFF";
    $("autoBtn").classList.toggle("primary", state.auto);
    schedule();
  });
  $("runSelect").addEventListener("change", () => {
    state.runId = $("runSelect").value || null;
    refresh();
  });
  const start = () => {
    ensureCharts();
    refresh();
    schedule();
  };
  if (typeof Chart !== "undefined") start();
  else window.addEventListener("load", start);
}

init();
