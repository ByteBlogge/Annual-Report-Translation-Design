"""Reviewer-facing exports.

The reviewer is the customer of this module, and they are a finance professional
under time pressure, not an engineer. So the output is:

* **ordered by risk**, not by document order -- the highest-risk item is the one
  that must be seen first, and there may be 40 of them;
* **self-contained** -- source and target side by side, in the same view, so no
  cross-referencing between windows;
* **specific** -- "1,234,567 -> 1,274,567 (digit slip)" rather than "number
  mismatch detected". A reviewer who has to hunt for the difference will stop
  reading the reports, and the whole mechanism becomes theatre.

Markdown is the primary format because it is readable in a browser, diffs in
git, and pastes into an email. HTML is generated for the side-by-side view where
Markdown's flatness gets in the way of a real table comparison.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..textutils import escape_html
from .queue import ReviewItem, ReviewQueue

__all__ = [
    "render_item_markdown",
    "render_review_sheet",
    "render_item_html",
    "render_review_html",
    "write_review_bundle",
]


def render_item_markdown(item: ReviewItem) -> str:
    risk = item.risk or {}
    lines = [
        f"### `{item.chunk_id}` — risk {item.risk_value:.2f} ({risk.get('band', '?')})",
        "",
        f"- **Section**: {' > '.join(item.section_path) or '(none)'}",
        f"- **Pages**: {item.pages[0]}–{item.pages[1]}",
        f"- **Financial summary**: {'yes — lowered threshold applies' if item.is_financial_summary else 'no'}",
        f"- **Status**: `{item.status}`"
        + (f" · reviewer `{item.reviewer}`" if item.reviewer else ""),
        "",
    ]
    reasons = risk.get("reasons", [])
    if reasons:
        lines.append("**Why it was flagged**")
        lines.append("")
        for reason in sorted(reasons, key=lambda r: -float(r.get("contribution", 0))):
            lines.append(
                f"- `+{float(reason.get('contribution', 0)):.2f}` **{reason.get('name')}** — {reason.get('detail')}"
            )
        lines.append("")

    detail = _number_findings(item)
    if detail:
        lines.append("**Figures that need attention**")
        lines.append("")
        lines.extend(detail)
        lines.append("")

    lines.append("**Source**")
    lines.append("")
    lines.append("> " + (item.source_text.strip().replace("\n", "\n> ") or "_(empty)_"))
    lines.append("")
    lines.append("**Target**")
    lines.append("")
    lines.append("> " + (item.target_text.strip().replace("\n", "\n> ") or "_(empty)_"))
    lines.append("")

    for index, table in enumerate(item.tables, start=1):
        lines.append(f"**Table {index}: {table.get('caption') or '(untitled)'}**")
        lines.append("")
        lines.append(f"- units note: `{table.get('units_note') or '(none)'}`")
        lines.append("")
        lines.append("```html")
        lines.append((table.get("target_html") or "").strip())
        lines.append("```")
        lines.append("")

    lines.append(f"<!-- item_id: {item.item_id} -->")
    lines.append("")
    return "\n".join(lines)


def _number_findings(item: ReviewItem) -> list[str]:
    features = item.features or {}
    out: list[str] = []
    if features.get("numbers_mismatched"):
        out.append(f"- {features['numbers_mismatched']} figure(s) changed value")
    if features.get("numbers_missing"):
        out.append(f"- {features['numbers_missing']} source figure(s) missing from the target")
    if features.get("number_structural"):
        out.append(f"- {features['number_structural']} structural issue(s) (e.g. dropped unit note)")
    if features.get("table_merge_conflicts"):
        out.append(f"- {features['table_merge_conflicts']} table merge conflict(s)")
    if features.get("table_holes"):
        out.append(f"- {features['table_holes']} uncovered table slot(s)")
    if features.get("untranslated_label_cells"):
        out.append(f"- {features['untranslated_label_cells']} untranslated label cell(s)")
    if features.get("charts_unresolved"):
        out.append(f"- {features['charts_unresolved']} chart(s) with no recovered data")
    for error in features.get("errors", []) or []:
        out.append(f"- ERROR: {error}")
    return out


def render_review_sheet(queue: ReviewQueue, *, title: str = "Human review sheet") -> str:
    stats = queue.stats()
    pending = queue.sorted_by_risk()
    lines = [
        f"# {title}",
        "",
        f"_{stats['total']} flagged chunk(s) · {stats['pending']} pending · "
        f"{stats['financial_pending']} touching financial statements · "
        f"highest risk {stats['max_risk']:.2f}_",
        "",
        "Items are ordered by risk, highest first. Figures were verified by a deterministic",
        "comparison of source and target, not by the model judging its own output.",
        "",
        "---",
        "",
    ]
    if not pending:
        lines.append("_Nothing requires review._")
        return "\n".join(lines)
    for item in pending:
        lines.append(render_item_markdown(item))
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def render_item_html(item: ReviewItem) -> str:
    risk = item.risk or {}
    band = risk.get("band", "?")
    reasons = "".join(
        f"<li><code>+{float(r.get('contribution', 0)):.2f}</code> "
        f"<strong>{escape_html(str(r.get('name')))}</strong> — {escape_html(str(r.get('detail')))}</li>"
        for r in sorted(risk.get("reasons", []), key=lambda r: -float(r.get("contribution", 0)))
    )
    tables = "".join(
        f"<details><summary>{escape_html(str(t.get('caption') or 'untitled'))} "
        f"<em>({escape_html(str(t.get('units_note') or 'no unit note'))})</em></summary>"
        f"<div class='tw'>{t.get('target_html', '')}</div></details>"
        for t in item.tables
    )
    findings = _number_findings(item)
    findings_html = (
        "<ul>" + "".join(f"<li>{escape_html(f.lstrip('- '))}</li>" for f in findings) + "</ul>"
        if findings
        else "<p class='muted'>No figure-level findings.</p>"
    )
    return f"""
<section class="item band-{escape_html(band)}">
  <header>
    <h2>{escape_html(item.chunk_id)}</h2>
    <span class="badge">{escape_html(band)}</span>
    <span class="risk">risk {item.risk_value:.2f} / threshold {float(risk.get('threshold', 0)):.2f}</span>
    {'<span class="fin">financial</span>' if item.is_financial_summary else ''}
  </header>
  <p class="meta">{escape_html(' > '.join(item.section_path) or '(no section)')}
     &middot; pages {item.pages[0]}–{item.pages[1]}</p>
  <h3>Why it was flagged</h3>
  <ul class="reasons">{reasons or '<li class="muted">no findings</li>'}</ul>
  <h3>Figures</h3>
  {findings_html}
  <div class="pair">
    <div><h3>Source</h3><pre>{escape_html(item.source_text)}</pre></div>
    <div><h3>Target</h3><pre>{escape_html(item.target_text)}</pre></div>
  </div>
  {tables}
  <footer class="muted">item_id {escape_html(item.item_id)} &middot; status {escape_html(item.status)}</footer>
</section>
"""


def render_review_html(queue: ReviewQueue, *, title: str = "Human review sheet") -> str:
    stats = queue.stats()
    body = "\n".join(render_item_html(item) for item in queue.sorted_by_risk())
    if not body.strip():
        body = "<p class='muted'>Nothing requires review.</p>"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{escape_html(title)}</title>
<style>
  :root {{ color-scheme: light dark; --fg:#1a1a1a; --bg:#ffffff; --muted:#6b7280;
           --line:#e5e7eb; --card:#f9fafb; --crit:#b91c1c; --warn:#b45309; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e8e8e8; --bg:#141414; --muted:#9ca3af; --line:#2e2e2e;
             --card:#1c1c1c; --crit:#f87171; --warn:#fbbf24; }}
  }}
  body {{ font: 15px/1.6 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
          color: var(--fg); background: var(--bg); margin: 0 auto; max-width: 1080px; padding: 32px 20px; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.05rem; margin: 0; display: inline; }}
  h3 {{ font-size: .82rem; text-transform: uppercase; letter-spacing: .06em;
        color: var(--muted); margin: 18px 0 6px; }}
  .item {{ border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px;
           margin-bottom: 20px; background: var(--card); }}
  .item header {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; }}
  .badge {{ font-size: .72rem; padding: 2px 8px; border-radius: 999px;
            background: var(--line); text-transform: uppercase; letter-spacing: .05em; }}
  .band-review .badge, .band-blocked .badge {{ background: var(--crit); color: #fff; }}
  .band-watch .badge {{ background: var(--warn); color: #fff; }}
  .risk, .meta, .muted, footer {{ color: var(--muted); font-size: .84rem; }}
  .fin {{ font-size: .72rem; border: 1px solid var(--crit); color: var(--crit);
          padding: 1px 7px; border-radius: 999px; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 760px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  pre {{ white-space: pre-wrap; word-break: break-word; background: var(--bg);
         border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
         font-size: .84rem; max-height: 340px; overflow: auto; margin: 0; }}
  ul {{ padding-left: 20px; }} ul.reasons li {{ margin-bottom: 4px; }}
  .tw {{ overflow-x: auto; }}
  table.art-table {{ border-collapse: collapse; width: 100%; font-size: .82rem; }}
  table.art-table th, table.art-table td {{ border: 1px solid var(--line);
        padding: 5px 8px; text-align: left; vertical-align: top; }}
  table.art-table th {{ background: var(--line); font-weight: 600; }}
  table.art-table .art-units {{ display: none; }}
  .summary {{ border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px; margin-bottom: 24px; }}
</style></head>
<body>
<h1>{escape_html(title)}</h1>
<div class="summary">
  <p><strong>{stats['total']}</strong> flagged chunk(s) &middot;
     <strong>{stats['pending']}</strong> pending &middot;
     <strong>{stats['financial_pending']}</strong> touching financial statements &middot;
     highest risk <strong>{stats['max_risk']:.2f}</strong></p>
  <p class="muted">Ordered by risk. Figures were verified by deterministic source/target
     comparison, not by the model assessing its own output.</p>
</div>
{body}
</body></html>
"""


def write_review_bundle(
    queue: ReviewQueue,
    directory: str | Path,
    *,
    title: str = "Human review sheet",
    stem: str = "review",
) -> dict[str, Path]:
    """Write the reviewer bundle: sheet, HTML, CSV and the raw queue."""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    md = base / f"{stem}.md"
    md.write_text(render_review_sheet(queue, title=title), encoding="utf-8")
    outputs["markdown"] = md

    html = base / f"{stem}.html"
    html.write_text(render_review_html(queue, title=title), encoding="utf-8")
    outputs["html"] = html

    csv_path = base / f"{stem}.csv"
    csv_path.write_text(queue.to_csv(), encoding="utf-8-sig")
    outputs["csv"] = csv_path

    jsonl = base / f"{stem}.jsonl"
    queue.save(jsonl)
    outputs["jsonl"] = jsonl
    return outputs


def summarise_findings(items: Sequence[ReviewItem]) -> dict[str, Any]:
    """Aggregate the flagged reasons, for the run report's headline."""
    counts: dict[str, int] = {}
    for item in items:
        for reason in (item.risk or {}).get("reasons", []):
            name = str(reason.get("name", "unknown"))
            counts[name] = counts.get(name, 0) + 1
    return {
        "items": len(items),
        "by_reason": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "financial_items": sum(1 for i in items if i.is_financial_summary),
    }


def dedupe_items(items: Iterable[ReviewItem]) -> list[ReviewItem]:
    """Drop duplicate flags for the same chunk, keeping the highest risk."""
    best: dict[str, ReviewItem] = {}
    for item in items:
        current = best.get(item.chunk_id)
        if current is None or item.risk_value > current.risk_value:
            best[item.chunk_id] = item
    return sorted(best.values(), key=lambda i: -i.risk_value)
