"""
Compact Single-Page Web App: Sleep Spindle Detection Comparison (Custom vs. Bridged vs. YASA)
=============================================================================================
Generates a standalone `spindle_methods_webapp.html` file and optionally serves it
over HTTP.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

from spindle_eval import run_full_modular_comparison

_reports_dir = Path(__file__).resolve().parent.parent / "outputs" / "reports"
REPORT_PATH = (
    (_reports_dir / "spindle_comparison_report.json")
    if _reports_dir.exists()
    else (Path(__file__).resolve().parent / "spindle_comparison_report.json")
)
HTML_PATH = (
    (_reports_dir / "spindle_methods_webapp.html")
    if _reports_dir.exists()
    else (Path(__file__).resolve().parent / "spindle_methods_webapp.html")
)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Spindle Comparison: Custom vs. Bridged vs. YASA</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --custom: #f59e0b;
      --bridged: #10b981;
      --yasa: #38bdf8;
      --match: #a855f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 13px;
      line-height: 1.4;
    }
    header {
      max-width: 1340px;
      margin: 0 auto;
      padding: 0.75rem 1rem;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      border-bottom: 1px solid var(--border);
    }
    header h1 { margin: 0; font-size: 1.1rem; font-weight: 600; }
    header span { color: var(--muted); font-size: 0.85rem; }
    main {
      max-width: 1340px;
      margin: 0 auto;
      padding: 0.75rem 1rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 0.6rem;
    }
    @media (max-width: 1050px) { .kpi-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 650px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
    .kpi {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.55rem 0.75rem;
    }
    .kpi-label { color: var(--muted); font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.03em; }
    .kpi-val { font-size: 1.15rem; font-weight: 700; margin: 0.15rem 0; }
    .kpi-sub { color: var(--muted); font-size: 0.76rem; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.75rem;
      overflow-x: auto;
    }
    .card h2 {
      margin: 0 0 0.45rem;
      font-size: 0.9rem;
      font-weight: 600;
      color: #e2e8f0;
    }
    .row-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    @media (max-width: 900px) { .row-2 { grid-template-columns: 1fr; } }
    table { width: 100%; border-collapse: collapse; font-size: 0.81rem; }
    th, td {
      padding: 0.36rem 0.5rem;
      border-bottom: 1px solid var(--border);
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-weight: 600; }
    tr:last-child td { border-bottom: none; }
    tr.highlight td { background: rgba(16, 185, 129, 0.08); }
  </style>
</head>
<body>

<header>
  <h1>Spindle Detection Comparison: Custom vs. Bridged vs. YASA</h1>
  <span id="meta-line"></span>
</header>

<main>
  <section class="kpi-grid" id="kpi-grid"></section>

  <section class="card">
    <h2>Head-to-Head Benchmark on Clean N2 (vs. YASA Reference)</h2>
    <table id="method-table"></table>
  </section>

  <section class="row-2">
    <div class="card">
      <h2>Spindle Density by Sleep Stage (spindles / min)</h2>
      <div id="plot-stage-density" style="width:100%; height:250px;"></div>
    </div>
    <div class="card">
      <h2>Spindle Count by Sleep Stage</h2>
      <div id="plot-stage-count" style="width:100%; height:250px;"></div>
    </div>
  </section>

  <section class="card">
    <h2>Sleep Stage Breakdown (Custom vs. Bridged vs. YASA)</h2>
    <table id="stage-table"></table>
  </section>

  <section class="row-2">
    <div class="card">
      <h2>Threshold Sweep on Clean N2 (Custom vs. Bridged)</h2>
      <div id="plot-thresh" style="width:100%; height:270px;"></div>
    </div>
    <div class="card">
      <h2>Threshold Sweep Table (N2: Custom / Bridged)</h2>
      <table id="thresh-table"></table>
    </div>
  </section>
</main>

<script>
const REPORT = __DATA_JSON__;

const C = { custom: '#f59e0b', bridged: '#10b981', yasa: '#38bdf8', match: '#a855f7', muted: '#94a3b8' };
const BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(15,23,42,0.5)',
  font: { color: '#cbd5e1', family: '-apple-system, BlinkMacSystemFont, sans-serif', size: 11 },
  margin: { l: 48, r: 16, t: 28, b: 36 },
  xaxis: { gridcolor: '#273449', zerolinecolor: '#475569' },
  yaxis: { gridcolor: '#273449', zerolinecolor: '#475569' },
  legend: { orientation: 'h', y: 1.15, x: 0, font: { size: 11 } },
};
const CFG = { responsive: true, displayModeBar: false };
const pct = x => (x * 100).toFixed(1) + '%';

function renderTop() {
  const m = REPORT.metadata;
  const e = REPORT.event_level, eb = REPORT.bridged_event_level || e;
  const f = REPORT.feature_level, fb = REPORT.bridged_feature_level || f;

  document.getElementById('meta-line').textContent =
    `${m.edf_name} · ${m.channel} (${m.fs} Hz) · ${m.clean_n2_minutes} min clean N2`;

  const items = [
    ['Custom (N2)', e.n_custom, `${m.custom_density_per_min} / min`, C.custom],
    ['Bridged (N2)', eb.n_custom, `${m.bridged_density_per_min} / min`, C.bridged],
    ['YASA (N2)', e.n_yasa, `${m.yasa_density_per_min} / min`, C.yasa],
    ['Matched w/ YASA (IoU ≥ ' + e.iou_threshold + ')', `${e.tp} vs ${eb.tp}`, `Custom TP: ${e.tp} · Bridged TP: ${eb.tp}`, '#f1f5f9'],
    ['Event F1 (Custom vs Bridged)', `${e.f1.toFixed(3)} vs ${eb.f1.toFixed(3)}`, `IoU: ${e.mean_iou.toFixed(2)} vs ${eb.mean_iou.toFixed(2)}`, C.bridged],
    ['N2 Mean Dur / Freq (C · B · Y)', `${f.dur.custom_mean.toFixed(2)}s · ${fb.dur.custom_mean.toFixed(2)}s · ${f.dur.yasa_mean.toFixed(2)}s`, `${f.freq.custom_mean.toFixed(1)} · ${fb.freq.custom_mean.toFixed(1)} · ${f.freq.yasa_mean.toFixed(1)} Hz`, '#f1f5f9'],
  ];

  document.getElementById('kpi-grid').innerHTML = items.map(([lbl, val, sub, col]) => `
    <div class="kpi">
      <div class="kpi-label">${lbl}</div>
      <div class="kpi-val" style="color:${col}">${val}</div>
      <div class="kpi-sub">${sub}</div>
    </div>
  `).join('');

  // Render Head-to-Head comparison table
  const s = REPORT.sample_level, sb = REPORT.bridged_sample_level || s;
  const b = REPORT.boundary_level, bb = REPORT.bridged_boundary_level || b;

  document.getElementById('method-table').innerHTML = `
    <thead>
      <tr>
        <th>Detector Method</th>
        <th>N2 Count (/min)</th>
        <th>Matched (TP)</th>
        <th>Method-Only (FP)</th>
        <th>YASA-Only (FN)</th>
        <th>Event Prec</th>
        <th>Event Recall</th>
        <th>Event F1</th>
        <th>Mean IoU</th>
        <th>Sample Prec</th>
        <th>Sample Recall</th>
        <th>Sample F1</th>
        <th>Cohen's κ</th>
        <th>Onset Bias (MAE)</th>
        <th>Offset Bias (MAE)</th>
        <th>Dur Bias (MAE)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong style="color:${C.custom}">Custom (Standard 1.5 SD)</strong></td>
        <td>${e.n_custom} (${m.custom_density_per_min}/m)</td>
        <td>${e.tp}</td>
        <td>${e.fp}</td>
        <td>${e.fn}</td>
        <td>${pct(e.precision)}</td>
        <td>${pct(e.recall)}</td>
        <td><strong>${e.f1.toFixed(3)}</strong></td>
        <td>${e.mean_iou.toFixed(3)}</td>
        <td>${pct(s.precision)}</td>
        <td>${pct(s.recall)}</td>
        <td>${s.f1.toFixed(3)}</td>
        <td>${s.cohen_kappa.toFixed(3)}</td>
        <td>${b.onset_error.mean_ms > 0 ? '+' : ''}${b.onset_error.mean_ms} ms (${b.onset_error.mae_ms})</td>
        <td>${b.offset_error.mean_ms > 0 ? '+' : ''}${b.offset_error.mean_ms} ms (${b.offset_error.mae_ms})</td>
        <td>${b.duration_error.mean_ms > 0 ? '+' : ''}${b.duration_error.mean_ms} ms (${b.duration_error.mae_ms})</td>
      </tr>
      <tr class="highlight">
        <td><strong style="color:${C.bridged}">Custom Bridged (1.5 / 2.0 SD, ≤0.2s dip)</strong></td>
        <td>${eb.n_custom} (${m.bridged_density_per_min}/m)</td>
        <td><strong>${eb.tp}</strong></td>
        <td>${eb.fp}</td>
        <td><strong>${eb.fn}</strong></td>
        <td>${pct(eb.precision)}</td>
        <td><strong>${pct(eb.recall)}</strong></td>
        <td><strong style="color:${C.bridged}">${eb.f1.toFixed(3)}</strong></td>
        <td><strong>${eb.mean_iou.toFixed(3)}</strong></td>
        <td>${pct(sb.precision)}</td>
        <td><strong>${pct(sb.recall)}</strong></td>
        <td><strong style="color:${C.bridged}">${sb.f1.toFixed(3)}</strong></td>
        <td><strong>${sb.cohen_kappa.toFixed(3)}</strong></td>
        <td>${bb.onset_error.mean_ms > 0 ? '+' : ''}${bb.onset_error.mean_ms} ms (${bb.onset_error.mae_ms})</td>
        <td>${bb.offset_error.mean_ms > 0 ? '+' : ''}${bb.offset_error.mean_ms} ms (${bb.offset_error.mae_ms})</td>
        <td>${bb.duration_error.mean_ms > 0 ? '+' : ''}${bb.duration_error.mean_ms} ms (${bb.duration_error.mae_ms})</td>
      </tr>
    </tbody>
  `;
}

function renderStages() {
  const st = REPORT.stage_level;
  if (!st) return;
  const rows = st.stages;
  const names = rows.map(r => r.stage);

  Plotly.newPlot('plot-stage-density', [
    { type: 'bar', x: names, y: rows.map(r => r.custom_density), name: 'Custom', marker: { color: C.custom },
      text: rows.map(r => r.custom_density.toFixed(2)), textposition: 'outside' },
    { type: 'bar', x: names, y: rows.map(r => r.bridged_density), name: 'Bridged', marker: { color: C.bridged },
      text: rows.map(r => r.bridged_density.toFixed(2)), textposition: 'outside' },
    { type: 'bar', x: names, y: rows.map(r => r.yasa_density), name: 'YASA', marker: { color: C.yasa },
      text: rows.map(r => r.yasa_density.toFixed(2)), textposition: 'outside' },
  ], {
    ...BASE, barmode: 'group',
    yaxis: { ...BASE.yaxis, title: 'Spindles / min', range: [0, Math.max(...rows.map(r => Math.max(r.custom_density, r.bridged_density || 0, r.yasa_density))) * 1.25] },
  }, CFG);

  Plotly.newPlot('plot-stage-count', [
    { type: 'bar', x: names, y: rows.map(r => r.n_custom), name: 'Custom', marker: { color: C.custom },
      text: rows.map(r => r.n_custom), textposition: 'outside' },
    { type: 'bar', x: names, y: rows.map(r => r.n_bridged), name: 'Bridged', marker: { color: C.bridged },
      text: rows.map(r => r.n_bridged), textposition: 'outside' },
    { type: 'bar', x: names, y: rows.map(r => r.n_yasa), name: 'YASA', marker: { color: C.yasa },
      text: rows.map(r => r.n_yasa), textposition: 'outside' },
    { type: 'bar', x: names, y: rows.map(r => r.n_both_bridged), name: 'Matched (B∩Y)', marker: { color: C.match },
      text: rows.map(r => r.n_both_bridged), textposition: 'outside' },
  ], {
    ...BASE, barmode: 'group',
    yaxis: { ...BASE.yaxis, title: 'Spindle Count', range: [0, Math.max(...rows.map(r => Math.max(r.n_custom, r.n_bridged || 0, r.n_yasa))) * 1.25] },
  }, CFG);

  const fmt = (v, d = 2) => (v == null ? '–' : v.toFixed(d));
  const fmtPct = v => (v == null ? '–' : pct(v));
  document.getElementById('stage-table').innerHTML =
    '<thead><tr><th>Stage</th><th>Clean Time (min)</th>' +
    '<th>Custom (/min)</th><th>Bridged (/min)</th><th>YASA (/min)</th>' +
    '<th>Matched (C / B)</th><th>Precision (C / B)</th><th>Recall (C / B)</th>' +
    '<th>Mean Freq C / B / Y (Hz)</th><th>Mean Dur C / B / Y (s)</th></tr></thead><tbody>' +
    rows.map(r => `<tr>
      <td><strong>${r.stage}</strong></td>
      <td>${r.clean_minutes.toFixed(1)}</td>
      <td style="color:${C.custom}">${r.n_custom} (${r.custom_density.toFixed(2)})</td>
      <td style="color:${C.bridged}">${r.n_bridged} (${r.bridged_density.toFixed(2)})</td>
      <td style="color:${C.yasa}">${r.n_yasa} (${r.yasa_density.toFixed(2)})</td>
      <td>${r.n_both} / <strong style="color:${C.bridged}">${r.n_both_bridged}</strong></td>
      <td>${fmtPct(r.precision)} / ${fmtPct(r.bridged_precision)}</td>
      <td>${fmtPct(r.recall)} / <strong>${fmtPct(r.bridged_recall)}</strong></td>
      <td>${fmt(r.custom_freq, 1)} / ${fmt(r.bridged_freq, 1)} / ${fmt(r.yasa_freq, 1)}</td>
      <td>${fmt(r.custom_dur, 2)} / ${fmt(r.bridged_dur, 2)} / ${fmt(r.yasa_dur, 2)}</td>
    </tr>`).join('') + '</tbody>';
}

function renderThresholdSweep() {
  const ts = REPORT.sweeps.threshold_sweep;

  Plotly.newPlot('plot-thresh', [
    { x: ts.map(d => d.std_thresh), y: ts.map(d => d.precision), mode: 'lines+markers', name: 'Custom Prec', line: { color: C.custom, width: 1.8, dash: 'dot' }, marker: { size: 4 } },
    { x: ts.map(d => d.std_thresh), y: ts.map(d => d.recall), mode: 'lines+markers', name: 'Custom Rec', line: { color: C.yasa, width: 1.8, dash: 'dot' }, marker: { size: 4 } },
    { x: ts.map(d => d.std_thresh), y: ts.map(d => d.f1), mode: 'lines+markers', name: 'Custom F1', line: { color: C.custom, width: 2.5 }, marker: { size: 6 } },
    { x: ts.map(d => d.std_thresh), y: ts.map(d => d.bridged_precision), mode: 'lines+markers', name: 'Bridged Prec', line: { color: '#34d399', width: 1.8, dash: 'dash' }, marker: { size: 4, symbol: 'diamond' } },
    { x: ts.map(d => d.std_thresh), y: ts.map(d => d.bridged_recall), mode: 'lines+markers', name: 'Bridged Rec', line: { color: '#60a5fa', width: 1.8, dash: 'dash' }, marker: { size: 4, symbol: 'diamond' } },
    { x: ts.map(d => d.std_thresh), y: ts.map(d => d.bridged_f1), mode: 'lines+markers', name: 'Bridged F1', line: { color: C.bridged, width: 2.8 }, marker: { size: 7, symbol: 'diamond' } },
  ], {
    ...BASE,
    xaxis: { ...BASE.xaxis, title: 'Base Threshold (SD above N2 mean; Bridged peak = +0.5 SD)' },
    yaxis: { ...BASE.yaxis, title: 'Score (vs. YASA)', range: [0, 1] },
    shapes: [{ type: 'line', x0: 1.5, x1: 1.5, y0: 0, y1: 1, line: { color: '#64748b', width: 1, dash: 'dot' } }],
  }, CFG);

  document.getElementById('thresh-table').innerHTML =
    '<thead><tr><th>std_thresh</th><th>Cutoff (µV)</th><th>Count (C / B)</th><th>Precision (C / B)</th><th>Recall (C / B)</th><th>F1 (C / B)</th><th>IoU (C / B)</th></tr></thead><tbody>' +
    ts.map(d => `<tr class="${d.std_thresh === 1.5 ? 'highlight' : ''}">
      <td><strong>${d.std_thresh.toFixed(2)}</strong>${d.std_thresh === 1.5 ? ' (def)' : ''}</td>
      <td>${d.uV_thresh.toFixed(1)} / ${d.uV_high_thresh ? d.uV_high_thresh.toFixed(1) : '–'}</td>
      <td>${d.count} / <span style="color:${C.bridged}">${d.bridged_count}</span></td>
      <td>${pct(d.precision)} / ${pct(d.bridged_precision)}</td>
      <td>${pct(d.recall)} / ${pct(d.bridged_recall)}</td>
      <td>${d.f1.toFixed(3)} / <strong style="color:${C.bridged}">${d.bridged_f1.toFixed(3)}</strong></td>
      <td>${d.mean_iou.toFixed(2)} / ${d.bridged_mean_iou.toFixed(2)}</td>
    </tr>`).join('') + '</tbody>';
}

window.addEventListener('DOMContentLoaded', () => {
  renderTop();
  renderStages();
  renderThresholdSweep();
});
</script>
</body>
</html>
"""


def build_html(report_data: dict) -> str:
    return HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(report_data))


def generate_webapp_files() -> Path:
    report_data = None
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r") as f:
            report_data = json.load(f)
    if report_data is None or "bridged_event_level" not in report_data:
        edf = report_data["metadata"]["edf_name"] if report_data and "metadata" in report_data else "BB0008_SP2.edf"
        ch = report_data["metadata"]["channel"] if report_data and "metadata" in report_data else "C3"
        report_data = run_full_modular_comparison(edf, ch)
        with open(REPORT_PATH, "w") as f:
            json.dump(report_data, f, indent=2)

    html = build_html(report_data)
    with open(HTML_PATH, "w") as f:
        f.write(html)
    return HTML_PATH


class SpindleAppHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.path = "/spindle_methods_webapp.html"
        elif parsed.path == "/api/compare":
            qs = parse_qs(parsed.query)
            edf = qs.get("edf", ["BB0008_SP2.edf"])[0]
            ch = qs.get("ch", ["C3"])[0]
            iou = float(qs.get("iou", ["0.2"])[0])
            std = float(qs.get("std", ["1.5"])[0])
            data = run_full_modular_comparison(edf, ch, iou_thresh=iou, std_thresh=std)
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()


if __name__ == "__main__":
    html_file = generate_webapp_files()
    print(f"Generated standalone interactive web app: {html_file}")
    if "--serve" in sys.argv:
        port = 8765
        print(f"Serving interactive web app at http://localhost:{port} ...")
        server = HTTPServer(("127.0.0.1", port), lambda *args, **kwargs: SpindleAppHandler(*args, directory=str(HTML_PATH.parent), **kwargs))
        server.serve_forever()
