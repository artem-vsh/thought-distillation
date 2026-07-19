/* Math autoresearch live dashboard */

const $ = (id) => document.getElementById(id);

const state = {
  auto: true,
  runId: null,
  timer: null,
  accChart: null,
  gapChart: null,
  dataChart: null,
  lastPayload: null,
};

function pct(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(100 * Number(x)).toFixed(1)}%`;
}

function num(x, digits = 4) {
  if (x == null || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

function deltaHtml(x) {
  if (x == null || Number.isNaN(x)) return `<span class="delta neu">—</span>`;
  const n = Number(x);
  const cls = n > 1e-6 ? "pos" : n < -1e-6 ? "neg" : "neu";
  const sign = n > 0 ? "+" : "";
  return `<span class="delta ${cls}">${sign}${(100 * n).toFixed(2)} pp</span>`;
}

function shortPath(p) {
  if (!p) return "—";
  const s = String(p);
  if (s.length <= 48) return s;
  return s.slice(0, 18) + "…" + s.slice(-26);
}

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

function phaseClass(phase) {
  if (!phase) return "";
  if (phase === "stopped" || phase === "completed") return "stopped";
  if (String(phase).includes("train")) return "train";
  return "";
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

function renderEvalLive(snap) {
  const card = $("evalLiveCard");
  const body = $("evalLiveBody");
  const hint = $("evalLiveHint");
  const fill = $("evalLiveBarFill");
  const live = snap.eval_live || {};
  const prog = live.eval_progress || snap.status?.eval_progress || {};
  const inProg = !!(live.eval_in_progress || snap.headline?.eval_in_progress);
  if (!inProg && !prog.n) {
    card.style.display = "none";
    return;
  }
  card.style.display = "block";
  const n = prog.n ?? 0;
  const maxN = prog.max_n ?? prog.n_pool ?? 1;
  const pctDone = Math.min(100, Math.round((100 * n) / maxN));
  fill.style.width = `${pctDone}%`;
  const status = prog.status || (inProg ? "in_progress" : "—");
  hint.textContent = `${status} · target ±${prog.target_ci_pp ?? snap.headline?.target_ci_pp ?? "?"}pp @ p<${prog.p_value ?? snap.headline?.p_value ?? "?"}`;
  const comps = (prog.comparisons || [])
    .map(
      (c) =>
        `<div class="metric-row"><span class="k">${escapeHtml(c.name)}</span>` +
        `<span class="v">Δ=${(100 * c.diff).toFixed(1)}pp ±${(100 * c.half_width).toFixed(1)}pp ` +
        `${c.meets_target ? "✓" : "…"}</span></div>`
    )
    .join("");
  body.innerHTML = `
    <div class="metric-row"><span class="k">Tag</span><span class="v">${escapeHtml(prog.tag || "—")}</span></div>
    <div class="metric-row"><span class="k">Sample n</span><span class="v">${n} / ${maxN} (pool ${prog.n_pool ?? "—"})</span></div>
    <div class="metric-row"><span class="k">Instant</span><span class="v">${pct(prog.instant?.accuracy)}</span></div>
    <div class="metric-row"><span class="k">High</span><span class="v">${pct(prog.high?.accuracy)}</span></div>
    ${comps || '<div class="empty">Growing sample…</div>'}
  `;
}

function renderKpis(snap) {
  const h = snap.headline || {};
  const ck = snap.checkpoints || {};
  const lastFin = ck.last_finalized;
  const lastInt = ck.last_intermediate;

  const items = [
    {
      label: "Phase",
      value: `<span class="phase-badge ${phaseClass(h.phase)}${h.eval_in_progress ? " train" : ""}">${
        h.eval_in_progress ? "eval…" : h.phase || "—"
      }</span>`,
      delta: h.message ? `<span class="delta neu">${escapeHtml(h.message).slice(0, 80)}</span>` : "",
    },
    {
      label: "Iteration",
      value: h.iteration != null ? String(h.iteration) : "—",
      delta: h.stopped ? `<span class="delta neg">${escapeHtml(h.stop_reason || "stopped")}</span>` : `<span class="delta neu">running</span>`,
    },
    {
      label: "Train-seed instant",
      value: pct(h.train_seed_instant),
      delta: deltaHtml(h.last_instant_delta),
    },
    {
      label: "Held-out test instant",
      value: pct(h.heldout_instant),
      delta: deltaHtml(h.last_heldout_instant_delta),
    },
    {
      label: "Overfit gap",
      value: h.overfit_gap == null ? "—" : `${(100 * h.overfit_gap).toFixed(1)} pp`,
      delta: `<span class="delta neu">train − heldout</span>`,
    },
    {
      label: "Train pool",
      value: h.train_pool_size != null ? String(h.train_pool_size) : "—",
      delta: `<span class="delta neu">heldout ${h.heldout_size ?? "—"}</span>`,
    },
    {
      label: "Finalized ckpts",
      value: String(ck.count_finalized ?? 0),
      delta: lastFin
        ? `<span class="delta neu">last ${escapeHtml(String(lastFin.name ?? lastFin.iteration ?? ""))}</span>`
        : "",
    },
    {
      label: "Last intermediate",
      value: lastInt?.name != null ? String(lastInt.name) : "—",
      delta: lastInt?.batch != null
        ? `<span class="delta neu">batch ${escapeHtml(String(lastInt.batch))}</span>`
        : `<span class="delta neu">${ck.count_intermediate ?? 0} mid</span>`,
    },
    {
      label: "Target CI",
      value:
        h.target_ci_pp != null
          ? `±${Number(h.target_ci_pp).toFixed(1)} pp`
          : "—",
      delta: `<span class="delta neu">p&lt;${h.p_value ?? "0.05"}</span>`,
    },
  ];

  $("kpiRow").innerHTML = items
    .map(
      (it) => `
    <div class="card kpi">
      <div class="value">${it.value}</div>
      <div class="label">${it.label}</div>
      ${it.delta || ""}
    </div>`
    )
    .join("");
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderCheckpoints(snap) {
  const ck = snap.checkpoints || {};
  const fin = ck.last_finalized;
  const mid = ck.last_intermediate;
  const parts = [];

  const card = (role, rec, cls) => {
    if (!rec) {
      return `<div class="ckpt ${cls}"><div class="role">${role}</div><div class="empty">None yet</div></div>`;
    }
    return `
      <div class="ckpt ${cls}">
        <div class="role">${role}</div>
        <div class="name">${escapeHtml(String(rec.name ?? `iter ${rec.iteration ?? "?"}`))}</div>
        <span class="tag">${cls === "finalized" ? "finalized" : "intermediate"}</span>
        <div class="meta">
          <div>iter ${escapeHtml(String(rec.iteration ?? "—"))}${rec.batch != null ? ` · batch ${escapeHtml(String(rec.batch))}` : ""}</div>
          <div title="${escapeHtml(rec.sampler_path || "")}">sampler: ${escapeHtml(shortPath(rec.sampler_path))}</div>
          <div title="${escapeHtml(rec.state_path || "")}">state: ${escapeHtml(shortPath(rec.state_path))}</div>
        </div>
      </div>`;
  };

  parts.push(card("Finalized checkpoint (last complete)", fin, "finalized"));
  parts.push(card("Last intermediate checkpoint", mid, "intermediate"));

  const policy = snap.state?.last_policy_sampler_path;
  const judge = snap.state?.last_judge_model_path;
  if (policy || judge) {
    parts.push(`
      <div class="ckpt">
        <div class="role">Active model pointers</div>
        <div class="meta">
          <div title="${escapeHtml(policy || "")}">policy sampler: ${escapeHtml(shortPath(policy))}</div>
          <div title="${escapeHtml(judge || "")}">judge path: ${escapeHtml(shortPath(judge))}</div>
          <div>policy source: ${escapeHtml(snap.headline?.policy_source || "—")} · judge: ${escapeHtml(snap.headline?.judge_source || "—")}</div>
        </div>
      </div>`);
  }

  $("ckptStack").innerHTML = parts.join("");

  const rows = (ck.finalized || []).slice().reverse();
  $("ckptTable").innerHTML = rows.length
    ? rows
        .map(
          (r) => `
      <tr>
        <td>${escapeHtml(String(r.iteration ?? "—"))}</td>
        <td>${escapeHtml(String(r.name ?? "—"))}</td>
        <td>${escapeHtml(String(r.batch ?? "—"))}</td>
        <td class="mono" title="${escapeHtml(r.sampler_path || "")}">${escapeHtml(shortPath(r.sampler_path))}</td>
      </tr>`
        )
        .join("")
    : `<tr><td colspan="4" class="empty">No finalized checkpoints</td></tr>`;
}

function renderEvals(snap) {
  const legend = snap.evals?.legend || {};
  $("evalLegend").textContent =
    (legend.train_seed || "") + " · " + (legend.heldout || "");

  const latest = snap.evals?.latest || {};
  const train = latest.train_seed || {};
  const held = latest.heldout || {};
  const postT = train.post || {};
  const postH = held.post || {};
  const preT = train.pre || {};
  const preH = held.pre || {};

  const panel = (cls, title, desc, post, pre, delta) => `
    <div class="eval-panel ${cls}">
      <h3>${title}</h3>
      <div class="desc">${escapeHtml(desc)}</div>
      <div class="metric-row"><span class="k">Instant (post)</span><span class="v ${cls}">${pct(post.instant)}</span></div>
      <div class="metric-row"><span class="k">High (post)</span><span class="v high">${pct(post.high)}</span></div>
      <div class="metric-row"><span class="k">Instant−High gap</span><span class="v">${post.gap == null || Number.isNaN(post.gap) ? "—" : num(post.gap)}</span></div>
      <div class="metric-row"><span class="k">Instant (pre)</span><span class="v">${pct(pre.instant)}</span></div>
      <div class="metric-row"><span class="k">Δ instant (pre→post)</span><span class="v">${delta == null ? "—" : ((100 * delta) >= 0 ? "+" : "") + (100 * delta).toFixed(2) + " pp"}</span></div>
      <div class="metric-row"><span class="k">Correct / completed</span><span class="v">${post.instant_correct ?? "—"} / ${post.instant_completed ?? "—"}</span></div>
    </div>`;

  $("evalCols").innerHTML =
    panel(
      "train",
      "Train-seed / run-generated track",
      legend.train_seed || "In-domain sample from train split",
      postT,
      preT,
      train.delta_instant
    ) +
    panel(
      "heldout",
      "Held-out original test",
      legend.heldout || "Never-train test set",
      postH,
      preH,
      held.delta_instant
    );
}

function chartDefaults() {
  Chart.defaults.color = "#8b9bb8";
  Chart.defaults.borderColor = "rgba(148,163,184,0.12)";
  Chart.defaults.font.family = "Segoe UI, system-ui, sans-serif";
}

function ensureCharts() {
  if (typeof Chart === "undefined") return;
  chartDefaults();
  if (!state.accChart) {
    state.accChart = new Chart($("accChart"), {
      type: "line",
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false } },
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
            title: { display: true, text: "Iteration" },
            grid: { display: false },
          },
        },
      },
    });
  }
  if (!state.gapChart) {
    state.gapChart = new Chart($("gapChart"), {
      type: "bar",
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            ticks: { callback: (v) => `${(100 * v).toFixed(0)}pp` },
            grid: { color: "rgba(148,163,184,0.08)" },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }
  if (!state.dataChart) {
    state.dataChart = new Chart($("dataChart"), {
      type: "line",
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: "Train pool size", font: { size: 10 } },
            grid: { color: "rgba(148,163,184,0.08)" },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }
}

function updateCharts(snap) {
  if (typeof Chart === "undefined") return;
  ensureCharts();
  const s = snap.evals?.series || {};
  const labels = (s.iteration || []).map((i) => (i == null ? "?" : String(i)));

  const line = (label, data, color, dash) => ({
    label,
    data: data || [],
    borderColor: color,
    backgroundColor: color + "33",
    tension: 0.25,
    borderWidth: 2.2,
    pointRadius: 3,
    pointHoverRadius: 5,
    spanGaps: true,
    borderDash: dash || [],
  });

  // Error bars: Chart.js needs yMin/yMax point annotations via floating bars
  // or chartjs-plugin-error-bars — use simple fill band via extra datasets.
  const withBand = (label, data, half, color) => {
    const base = line(label, data, color);
    if (!half || !half.some((x) => x != null)) return [base];
    const upper = data.map((y, i) =>
      y == null || half[i] == null ? null : Math.min(1, y + half[i])
    );
    const lower = data.map((y, i) =>
      y == null || half[i] == null ? null : Math.max(0, y - half[i])
    );
    return [
      base,
      {
        label: `${label} CI+`,
        data: upper,
        borderColor: color + "55",
        backgroundColor: "transparent",
        borderWidth: 1,
        borderDash: [2, 2],
        pointRadius: 0,
        tension: 0.2,
        spanGaps: true,
      },
      {
        label: `${label} CI-`,
        data: lower,
        borderColor: color + "55",
        backgroundColor: "transparent",
        borderWidth: 1,
        borderDash: [2, 2],
        pointRadius: 0,
        tension: 0.2,
        spanGaps: true,
      },
    ];
  };

  state.accChart.data.labels = labels;
  state.accChart.data.datasets = [
    ...withBand(
      "Train-seed instant",
      s.train_seed_instant_post,
      s.train_seed_instant_ci_half,
      "#38bdf8"
    ),
    ...withBand(
      "Held-out instant",
      s.heldout_instant_post,
      s.heldout_instant_ci_half,
      "#c084fc"
    ),
    line("Train-seed high", s.train_seed_high_post, "#f472b6", [5, 4]),
    line("Held-out high", s.heldout_high_post, "#fbbf24", [5, 4]),
  ];
  state.accChart.update("none");

  state.gapChart.data.labels = labels;
  state.gapChart.data.datasets = [
    {
      label: "Overfit gap",
      data: s.overfit_gap || [],
      backgroundColor: (s.overfit_gap || []).map((v) =>
        v == null ? "#5c6b86" : v > 0 ? "rgba(248,113,113,0.65)" : "rgba(52,211,153,0.65)"
      ),
      borderRadius: 6,
    },
  ];
  state.gapChart.update("none");

  const growth = snap.data_growth || [];
  state.dataChart.data.labels = growth.map(
    (g, i) => (g.iteration != null ? `i${g.iteration}` : `#${i}`)
  );
  state.dataChart.data.datasets = [
    {
      label: "Train pool",
      data: growth.map((g) => g.size),
      borderColor: "#34d399",
      backgroundColor: "rgba(52,211,153,0.15)",
      fill: true,
      tension: 0.3,
      pointRadius: 2,
      borderWidth: 2,
    },
  ];
  state.dataChart.update("none");
}

function renderTimeline(snap) {
  const events = (snap.events || []).slice().reverse();
  if (!events.length) {
    $("timeline").innerHTML = `<div class="empty">No events yet — start an autoresearch run</div>`;
    return;
  }
  $("timeline").innerHTML = events
    .map((e) => {
      const ts = e.ts ? e.ts.slice(11, 19) : "—";
      return `
      <div class="event">
        <div class="ts">${escapeHtml(ts)}</div>
        <div class="body">
          <div class="kind">${escapeHtml(e.kind || "event")}${e.iteration != null ? ` · iter ${e.iteration}` : ""}</div>
          <div class="msg">${escapeHtml(e.message || e.phase || "")}</div>
        </div>
      </div>`;
    })
    .join("");
}

function renderIters(snap) {
  const iters = snap.iterations || [];
  $("iterTable").innerHTML = iters.length
    ? iters
        .map(
          (it) => `
      <tr>
        <td>${escapeHtml(it.name)}</td>
        <td>${it.n_generated ?? "—"}</td>
        <td>${it.n_validated ?? "—"}</td>
        <td>${it.has_eval_pre ? "✓" : "—"}</td>
        <td>${it.has_eval_post ? "✓" : "—"}</td>
        <td>${it.has_train ? "✓" : "—"}</td>
        <td>${it.has_metrics ? "✓" : "—"}</td>
      </tr>`
        )
        .join("")
    : `<tr><td colspan="7" class="empty">No iterations yet</td></tr>`;
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

function renderEmpty(error) {
  $("kpiRow").innerHTML = `
    <div class="card kpi" style="grid-column: 1 / -1">
      <div class="value" style="font-size:1.2rem">No active run</div>
      <div class="label">${escapeHtml(error || "Start python -m autoresearch")}</div>
      <div class="delta neu">Watching output/autoresearch/</div>
    </div>`;
  $("ckptStack").innerHTML = `<div class="empty">Checkpoints appear after the first train step</div>`;
  $("evalCols").innerHTML = `<div class="empty">Dual evals (train-seed + held-out) appear after the first eval phase</div>`;
  $("timeline").innerHTML = `<div class="empty">Waiting for events.jsonl</div>`;
}

async function refresh() {
  try {
    const q = state.runId ? `?run=${encodeURIComponent(state.runId)}` : "";
    const snap = await fetchJSON(`/api/status${q}`);
    state.lastPayload = snap;
    fillRunSelect(snap.runs || [], snap.run?.id);

    if (!snap.ok) {
      setLive(true, "no run");
      renderEmpty(snap.error);
      $("footerLeft").textContent = `Updated ${new Date().toLocaleTimeString()} · ${snap.error || "no run"}`;
      return;
    }

    setLive(true, `live · ${timeAgo(snap.headline?.updated_at || snap.generated_at)}`);
    renderKpis(snap);
    renderEvalLive(snap);
    renderCheckpoints(snap);
    renderEvals(snap);
    renderTimeline(snap);
    renderIters(snap);
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
  // Wait for Chart.js if loaded with defer
  const start = () => {
    ensureCharts();
    refresh();
    schedule();
  };
  if (typeof Chart !== "undefined") start();
  else window.addEventListener("load", start);
}

init();
