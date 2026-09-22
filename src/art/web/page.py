"""The demo page: one self-contained HTML document.

No CDN, no build step, no framework. It is served by FastAPI and talks to
``/api/run``. Keeping it dependency-free means the demo works on a machine with
no internet, which is the same constraint the rest of the project holds itself
to.

The client highlights the figures that changed by walking text nodes -- not by
string-replacing the rendered HTML, which would corrupt tags and attributes.
"""

from __future__ import annotations

__all__ = ["INDEX_HTML"]

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annual Report Translator — verification demo</title>
<style>
  :root {
    --bg: #f6f7f9;
    --panel: #ffffff;
    --ink: #1c2024;
    --muted: #5f6b7a;
    --line: #dfe3e8;
    --accent: #1f6feb;
    --ok: #1a7f37;
    --warn: #9a6700;
    --bad: #b42318;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
  }
  a { color: var(--accent); }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 28px 20px 72px; }
  header h1 { font-size: 22px; margin: 0 0 6px; letter-spacing: -0.01em; }
  header p { margin: 0; color: var(--muted); max-width: 80ch; }
  .controls {
    margin: 22px 0 18px; padding: 16px 18px; background: var(--panel);
    border: 1px solid var(--line); border-radius: 10px;
    display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-end;
  }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field label { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  .field .hint { font-size: 11px; color: var(--muted); text-transform: none; letter-spacing: 0; }
  select, input[type=number], button {
    font: inherit; padding: 7px 10px; border: 1px solid var(--line);
    border-radius: 7px; background: #fff; color: var(--ink);
  }
  button.primary {
    background: var(--accent); border-color: var(--accent); color: #fff;
    font-weight: 600; cursor: pointer; padding: 8px 18px;
  }
  button.primary:disabled { opacity: .6; cursor: progress; }
  button.ghost { cursor: pointer; }
  .verdict {
    padding: 14px 18px; border-radius: 10px; border: 1px solid var(--line);
    background: var(--panel); font-weight: 600; display: none;
  }
  .verdict.show { display: block; }
  .verdict.ok { border-color: #a7d7b3; background: #eaf7ee; color: var(--ok); }
  .verdict.bad { border-color: #f0b4ae; background: #fdecea; color: var(--bad); }
  .verdict .sub { font-weight: 400; color: var(--muted); font-size: 13px; margin-top: 4px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0 8px; }
  @media (max-width: 900px) { .grid, .cols { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 16px 18px;
  }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin: 0 0 12px; }
  .stat { display: flex; gap: 26px; flex-wrap: wrap; }
  .stat div { min-width: 96px; }
  .stat .n { font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .stat .l { font-size: 12px; color: var(--muted); }
  .stat .n.ok { color: var(--ok); }
  .stat .n.bad { color: var(--bad); }
  h2.section { font-size: 16px; margin: 28px 0 10px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .doc {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 16px 18px; max-height: 620px; overflow: auto;
  }
  .doc h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin: 0 0 12px; position: sticky; top: -16px; background: var(--panel); padding: 6px 0; }
  .doc h1, .doc h2, .doc h3, .doc h4 { line-height: 1.3; }
  .doc h1 { font-size: 19px; } .doc h2 { font-size: 17px; } .doc h3 { font-size: 15px; }
  .doc p { margin: 8px 0; }
  .doc .page-no, .doc .chunk-id {
    display: inline-block; font: 11px/1 var(--mono); color: var(--muted);
    background: #f0f2f5; border-radius: 4px; padding: 3px 6px; margin-bottom: 8px;
  }
  .doc .units { font-size: 12px; color: var(--muted); margin: 4px 0; }
  .doc figure { margin: 12px 0; }
  .doc figcaption { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .doc table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 4px 0 8px; }
  .doc th, .doc td { border: 1px solid var(--line); padding: 5px 8px; text-align: left; vertical-align: top; }
  .doc th { background: #f3f5f8; font-weight: 600; }
  .doc td:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; }
  .doc .img { color: var(--muted); font-style: italic; }
  table.res { border-collapse: collapse; width: 100%; font-size: 13px; background: var(--panel); }
  table.res th, table.res td { border: 1px solid var(--line); padding: 7px 10px; text-align: left; vertical-align: top; }
  table.res th { background: #f3f5f8; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; color: var(--muted); }
  .pill { display: inline-block; font: 11px/1 var(--mono); padding: 3px 7px; border-radius: 999px; border: 1px solid var(--line); background: #f6f7f9; }
  .pill.auto { color: var(--ok); border-color: #a7d7b3; background: #eaf7ee; }
  .pill.watch { color: var(--warn); border-color: #e8d08a; background: #fdf6e3; }
  .pill.review { color: var(--bad); border-color: #f0b4ae; background: #fdecea; }
  .pill.blocked { color: #fff; background: var(--bad); border-color: var(--bad); }
  mark.bad { background: #ffe0dd; color: var(--bad); font-weight: 700; padding: 0 2px; border-radius: 3px; }
  .muted { color: var(--muted); }
  .mono { font-family: var(--mono); font-size: 12.5px; }
  .reasons { font-size: 12.5px; color: var(--muted); }
  .reviewbox { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
  .reviewbox .head { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
  .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 900px) { .split { grid-template-columns: 1fr; } }
  .split pre { margin: 0; white-space: pre-wrap; word-break: break-word; background: #f7f8fa; border: 1px solid var(--line); border-radius: 7px; padding: 10px; max-height: 260px; overflow: auto; }
  .empty { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Annual Report Translator — verification demo</h1>
    <p>
      The whole pipeline (parse → chunk → translate → verify → HITL) runs offline against a
      deterministic mock backend. The interesting control is <strong>corrupt N figures</strong>:
      the backend deliberately alters N figures, and the number guard must report exactly N.
      If it reports fewer, the guard has regressed.
    </p>
  </header>

  <div class="controls">
    <div class="field">
      <label for="fixture">Document</label>
      <select id="fixture"><option value="builtin">built-in demo page</option></select>
    </div>
    <div class="field">
      <label for="fault">Corrupt N figures</label>
      <input id="fault" type="number" min="0" max="9" value="2" style="width:90px">
      <span class="hint">0 = faithful run</span>
    </div>
    <div class="field">
      <label for="threshold">HITL threshold</label>
      <input id="threshold" type="number" min="0" max="1" step="0.05" value="0.45" style="width:96px">
      <span class="hint">financial chunks get −0.15</span>
    </div>
    <button class="primary" id="run">Run</button>
    <button class="ghost" id="download" disabled>Download target .md</button>
  </div>

  <div class="verdict" id="verdict"></div>

  <div class="grid">
    <div class="card">
      <h2>Contents of the table payload</h2>
      <div class="stat" id="audit"></div>
      <p class="muted" style="font-size:12.5px;margin:12px 0 0">
        Every table is rebuilt twice: once to send to the model (labels only) and once to check that
        rebuild for any numeric cell text. The audit is a measurement, not a claim.
      </p>
    </div>
    <div class="card">
      <h2>Number integrity</h2>
      <div class="stat" id="numbers"></div>
      <p class="muted mono" id="numbersum" style="margin:12px 0 0"></p>
    </div>
  </div>

  <h2 class="section">Risk &amp; human review</h2>
  <table class="res" id="risks"><tbody></tbody></table>

  <h2 class="section">Source vs target</h2>
  <div class="cols">
    <div class="doc"><h3>Source (as parsed)</h3><div id="source"></div></div>
    <div class="doc"><h3>Target (as produced)</h3><div id="target"></div></div>
  </div>

  <h2 class="section">Review queue</h2>
  <div id="review"><p class="empty">Nothing flagged.</p></div>
</div>

<script>
const $ = (id) => document.getElementById(id);

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Wrap occurrences of `needles` in the container's text nodes only. Walking text
// nodes keeps tags and attributes intact -- a naive innerHTML.replace() would
// happily rewrite a class name or break an tag.
function highlight(container, needles) {
  const targets = needles.filter(Boolean).filter((v) => /\\d/.test(v));
  if (!targets.length) return;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    let text = node.nodeValue;
    let hit = false;
    for (const needle of targets) {
      if (text.includes(needle)) { text = text.split(needle).join("\\u0000" + needle + "\\u0001"); hit = true; }
    }
    if (!hit) continue;
    const span = document.createElement("span");
    span.innerHTML = esc(text)
      .split("\\u0000").join('<mark class="bad">')
      .split("\\u0001").join("</mark>");
    node.parentNode.replaceChild(span, node);
  }
}

function stat(value, label, cls) {
  return '<div><div class="n ' + (cls || "") + '">' + esc(value) + '</div><div class="l">' + esc(label) + '</div></div>';
}

function render(data) {
  // --- verdict ---
  const v = data.verdict || {};
  const box = $("verdict");
  box.className = "verdict show " + (v.ok ? "ok" : "bad");
  box.innerHTML =
    (v.ok ? "PASS — " : "FAIL — ") + esc(v.message || "") +
    '<div class="sub">injected ' + esc(v.injected) + " · applied by the backend " + esc(v.mock_applied) +
    " · reported by the guard " + esc(v.caught) + "</div>";

  // --- payload audit ---
  const a = data.audit || {};
  $("audit").innerHTML =
    stat(a.label_cells_sent, "label cells sent") +
    stat(a.numeric_cells_copied, "figures copied by code") +
    stat((a.leaks || []).length, "figure leaks", (a.leaks || []).length ? "bad" : "ok");

  // --- number integrity ---
  const rep = data.number_report || {};
  const s = data.summary || {};
  $("numbers").innerHTML =
    stat(s.numbers_matched + " / " + s.numbers_source, "figures matched") +
    stat(s.numbers_mismatched, "changed", s.numbers_mismatched ? "bad" : "ok") +
    stat(s.numbers_missing, "missing", s.numbers_missing ? "bad" : "ok") +
    stat(((s.number_drift || 0) * 100).toFixed(2) + "%", "drift", s.number_drift ? "bad" : "ok");
  $("numbersum").textContent = data.number_summary || "";

  // --- risk table ---
  const rows = (data.risks || []).map((r) => (
    "<tr>" +
      '<td class="mono">' + esc(r.chunk_id) + "</td>" +
      "<td>" + esc(r.section) + "</td>" +
      '<td><span class="pill ' + esc(r.band) + '">' + esc(r.band) + "</span></td>" +
      '<td class="mono">' + r.value.toFixed(2) + " / " + r.threshold.toFixed(2) + "</td>" +
      '<td class="reasons">' + (r.reasons || []).map((x) => "+" + x.contribution.toFixed(2) + " " + esc(x.detail)).join("<br>") + "</td>" +
    "</tr>"
  )).join("");
  $("risks").querySelector("tbody").innerHTML =
    "<tr><th>chunk</th><th>section</th><th>band</th><th>risk / threshold</th><th>why</th></tr>" + rows;

  // --- documents ---
  $("source").innerHTML = data.source_html || "";
  $("target").innerHTML = data.target_html || "";
  const changed = (rep.mismatched || []).map((m) => (m.target || {}).raw);
  highlight($("target"), changed);

  // --- review queue ---
  const items = data.review || [];
  $("review").innerHTML = items.length ? items.map((it) => (
    '<div class="reviewbox">' +
      '<div class="head">' +
        '<span class="pill ' + esc(it.band) + '">' + esc(it.band) + " " + it.risk.toFixed(2) + "</span>" +
        '<span class="mono">' + esc(it.chunk_id) + "</span>" +
        '<span class="muted">' + esc(it.section) + "</span>" +
        '<span class="pill">' + esc(it.status) + "</span>" +
      "</div>" +
      '<div class="muted reasons">' + (it.reasons || []).map(esc).join("<br>") + "</div>" +
      '<div class="split" style="margin-top:10px">' +
        "<pre>" + esc(it.source_text) + "</pre>" +
        "<pre>" + esc(it.target_text) + "</pre>" +
      "</div>" +
    "</div>"
  )).join("") : '<p class="empty">Nothing flagged.</p>';

  const dl = $("download");
  dl.disabled = false;
  dl.onclick = () => {
    const blob = new Blob([data.target_markdown || ""], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "target.md";
    link.click();
    URL.revokeObjectURL(url);
  };
}

async function run() {
  const btn = $("run");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fixture: $("fixture").value,
        fault: Number($("fault").value || 0),
        threshold: Number($("threshold").value || 0.45),
      }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status + " " + (await res.text()));
    render(await res.json());
  } catch (err) {
    const box = $("verdict");
    box.className = "verdict show bad";
    box.textContent = "Request failed: " + err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run";
  }
}

$("run").addEventListener("click", run);
$("fixture").addEventListener("change", run);
fetch("/api/fixtures").then((r) => r.json()).then((d) => {
  const sel = $("fixture");
  sel.innerHTML = (d.fixtures || []).map((f) =>
    '<option value="' + esc(f.id) + '">' + esc(f.label) + "</option>"
  ).join("");
  run();
}).catch(() => run());
</script>
</body>
</html>
"""
