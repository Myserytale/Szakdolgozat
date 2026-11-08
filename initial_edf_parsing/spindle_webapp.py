"""
Compact Single-Page Web App: Sleep Spindle Detection Comparison (Custom vs. YASA)
=================================================================================
Generates a standalone `spindle_methods_webapp.html` file and optionally serves it
over HTTP.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

from spindle_eval import run_full_modular_comparison

REPORT_PATH = Path(__file__).resolve().parent / "spindle_comparison_report.json"
HTML_PATH = Path(__file__).resolve().parent / "spindle_methods_webapp.html"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Spindle Comparison: Custom vs. YASA</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --custom: #f59e0b;
      --yasa: #38bdf8;
      --match: #10b981;
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
      max-width: 1280px;
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
      max-width: 1280px;
      margin: 0 auto;
      padding: 0.75rem 1rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 0.6rem;
    }
    @media (max-width: 900px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
    .kpi {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.55rem 0.75rem;
    }
    .kpi-label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.03em; }
    .kpi-val { font-size: 1.2rem; font-weight: 700; margin: 0.15rem 0; }
    .kpi-sub { color: var(--muted); font-size: 0.78rem; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.75rem;
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
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th, td {
      padding: 0.38rem 0.5rem;
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
  <h1>Spindle Detection Comparison: Custom vs. YASA</h1>
  <span id="meta-line"></span>
</header>

<main>
  <section class="kpi-grid" id="kpi-grid"></section>

  <section class="row-2">
    <div class="card">
      <h2>Spindle Density by Sleep Stage (spindles / min)</h2>
      <div id="plot-stage-density" style="width:100%; height:240px;"></div>
    </div>
    <div class="card">
      <h2>Spindle Count by Sleep Stage</h2>
      <div id="plot-stage-count" style="width:100%; height:240px;"></div>
    </div>
  </section>

  <section class="card">
    <h2>Sleep Stage Breakdown</h2>
    <table id="stage-table"></table>
  </section>

  <section class="row-2">
    <div class="card">
      <h2>Custom Threshold Sweep on Clean N2</h2>
      <div id="plot-thresh" style="width:100%; height:250px;"></div>
    </div>
    <div class="card">
      <h2>Threshold Sweep Table (N2)</h2>
      <table id="thresh-table"></table>
    </div>
  </section>
</main>

<script>
const REPORT = __DATA_JSON__;

const C = { custom: '#f59e0b', yasa: '#38bdf8', match: '#10b981', muted: '#94a3b8' };
const BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(15,23,42,0.5)',
  font: { color: '#cbd5e1', family: '-apple-system, BlinkMacSystemFont, sans-serif', size: 11 },
  margin: { l: 48, r: 16, t: 24, b: 36 },
  xaxis: { gridcolor: '#273449', zerolinecolor: '#475569' },
  yaxis: { gridcolor: '#273449', zerolinecolor: '#475569' },
  legend: { orientation: 'h', y: 1.14, x: 0, font: { size: 11 } },
};
const CFG = { responsive: true, displayModeBar: false };
const pct = x => (x * 100).toFixed(1) + '%';

function renderTop() {
  const m = REPORT.metadata, e = REPORT.event_level, f = REPORT.feature_level;

  document.getElementById('meta-line').textContent =
    `${m.edf_name} · ${m.channel} (${m.fs} Hz) · ${m.clean_n2_minutes} min clean N2`;

  const items = [
    ['Custom (N2)', e.n_custom, `${m.custom_density_per_min} / min`, C.custom],
    ['YASA (N2)', e.n_yasa, `${m.yasa_density_per_min} / min`, C.yasa],
    ['Matched (IoU ≥ ' + e.iou_threshold + ')', e.tp, `Custom-only: ${e.fp} · YASA-only: ${e.fn}`, C.match],
    ['Precision / Recall / F1', `${pct(e.precision)} / ${pct(e.recall)}`, `F1 = ${e.f1.toFixed(2)} · Mean IoU = ${e.mean_iou.toFixed(2)}`, '#f1f5f9'],
    ['N2 Mean Features (C vs Y)', `${f.dur.custom_mean.toFixed(2)}s vs ${f.dur.yasa_mean.toFixed(2)}s`, `${f.freq.custom_mean.toFixed(1)} vs ${f.freq.yasa_mean.toFixed(1)} Hz · ${f.peak_amp.custom_mean.toFixed(0)} vs ${f.peak_amp.yasa_mean.toFixed(0)} µV`, '#f1f5f9'],
  ];

  document.getElementById('kpi-grid').innerHTML = items.map(([lbl, val, sub, col]) => `
    <div class="kpi">
      <div class="kpi-label">${lbl}</div>
      <div class="kpi-val" style="color:${col}">${val}</div>
      <div class="kpi-sub">${sub}</div>
    </div>
  `).join('');
}

function renderStages() {
  const st = REPORT.stage_level;
  if (!st) return;
  const rows = st.stages;
  const names = rows.map(r => r.stage);

  Plotly.newPlot('plot-stage-density', [
    { type: 'bar', x: names, y: rows.map(r => r.custom_density), name: 'Custom', marker: { color: C.custom },
      text: rows.map(r => r.custom_density.toFixed(2)), textposition: 'outside' },
    { type: 'bar', x: names, y: rows.map(r => r.yasa_density), name: 'YASA', marker: { color: C.yasa },
      text: rows.map(r => r.yasa_density.toFixed(2)), textposition: 'outside' },
  ], {
    ...BASE, barmode: 'group',
    yaxis: { ...BASE.yaxis, title: 'Spindles / min', range: [0, Math.max(...rows.map(r => Math.max(r.custom_density, r.yasa_density))) * 1.22] },
  }, CFG);

  Plotly.newPlot('plot-stage-count', [
    { type: 'bar', x: names, y: rows.map(r => r.n_custom), name: 'Custom', marker: { color: C.custom },
      text: rows.map(r => r.n_custom), textposition: 'outside' },
    { type: 'bar', x: names, y: rows.map(r => r.n_yasa), name: 'YASA', marker: { color: C.yasa },
      text: rows.map(r => r.n_yasa), textposition: 'outside' },
    { type: 'bar', x: names, y: rows.map(r => r.n_both), name: 'Matched', marker: { color: C.match },
      text: rows.map(r => r.n_both), textposition: 'outside' },
  ], {
    ...BASE, barmode: 'group',
    yaxis: { ...BASE.yaxis, title: 'Spindle Count', range: [0, Math.max(...rows.map(r => Math.max(r.n_custom, r.n_yasa))) * 1.22] },
  }, CFG);

  const fmt = (v, d = 2) => (v == null ? '–' : v.toFixed(d));
  document.getElementById('stage-table').innerHTML =
    '<thead><tr><th>Stage</th><th>Clean Time (min)</th><th>Custom</th><th>Custom /min</th><th>YASA</th><th>YASA /min</th>' +
    '<th>Matched</th><th>Precision</th><th>Recall</th><th>Mean Freq C / Y (Hz)</th><th>Mean Dur C / Y (s)</th></tr></thead><tbody>' +
    rows.map(r => `<tr>
      <td><strong>${r.stage}</strong></td>
      <td>${r.clean_minutes.toFixed(1)}</td>
      <td style="color:${C.custom}">${r.n_custom}</td>
      <td>${r.custom_density.toFixed(2)}</td>
      <td style="color:${C.yasa}">${r.n_yasa}</td>
      <td>${r.yasa_density.toFixed(2)}</td>
      <td style="color:${C.match}">${r.n_both}</td>
      <td>${r.precision == null ? '–' : pct(r.precision)}</td>
      <td>${r.recall == null ? '–' : pct(r.recall)}</td>
      <td>${fmt(r.custom_freq, 1)} / ${fmt(r.yasa_freq, 1)}</td>
      <td>${fmt(r.custom_dur, 2)} / ${fmt(r.yasa_dur, 2)}</td>
    </tr>`).join('') + '</tbody>';
}

function renderThresholdSweep() {
  const ts = REPORT.sweeps.threshold_sweep;

  Plotly.newPlot('plot-thresh', [
    { x: ts.map(d => d.std_thresh), y: ts.map(d => d.precision), mode: 'lines+markers', name: 'Precision', line: { color: C.custom, width: 2 }, marker: { size: 5 } },
    { x: ts.map(d => d.std_thresh), y: ts.map(d => d.recall), mode: 'lines+markers', name: 'Recall', line: { color: C.yasa, width: 2 }, marker: { size: 5 } },
    { x: ts.map(d => d.std_thresh), y: ts.map(d => d.f1), mode: 'lines+markers', name: 'F1 Score', line: { color: C.match, width: 2.5 }, marker: { size: 6 } },
  ], {
    ...BASE,
    xaxis: { ...BASE.xaxis, title: 'Custom Threshold (SD above N2 mean)' },
    yaxis: { ...BASE.yaxis, title: 'Score', range: [0, 1] },
    shapes: [{ type: 'line', x0: 1.5, x1: 1.5, y0: 0, y1: 1, line: { color: '#64748b', width: 1, dash: 'dot' } }],
  }, CFG);

  document.getElementById('thresh-table').innerHTML =
    '<thead><tr><th>std_thresh (SD)</th><th>Cutoff (µV)</th><th>Custom Count</th><th>Precision</th><th>Recall</th><th>F1</th><th>Mean IoU</th></tr></thead><tbody>' +
    ts.map(d => `<tr class="${d.std_thresh === 1.5 ? 'highlight' : ''}">
      <td><strong>${d.std_thresh.toFixed(2)}</strong>${d.std_thresh === 1.5 ? ' (default)' : ''}</td>
      <td>${d.uV_thresh.toFixed(2)}</td>
      <td>${d.count}</td>
      <td>${pct(d.precision)}</td>
      <td>${pct(d.recall)}</td>
      <td><strong>${d.f1.toFixed(3)}</strong></td>
      <td>${d.mean_iou.toFixed(2)}</td>
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
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r") as f:
            report_data = json.load(f)
    else:
        report_data = run_full_modular_comparison("BB0008_SP2.edf", "P3")
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
            ch = qs.get("ch", ["P3"])[0]
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
        server = HTTPServer(("127.0.0.1", port), SpindleAppHandler)
        server.serve_forever()
