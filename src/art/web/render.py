"""Rendering for the web demo, plus the payload audit.

``audit_payloads`` is the interesting one. The claim "a table's figures are never
sent to the model" is easy to assert and hard to believe, so the demo page shows
it as a measurement: rebuild every table payload exactly as the translator would,
and check that not one numeric cell's text occurs anywhere inside it. It is a
cheap check and it is falsifiable -- if the enforcement point in
``agents.render_table_payload`` ever regresses, the demo page says so.
"""

from __future__ import annotations

from typing import Any

from ..parser.chart_reader import chart_to_narrative
from ..parser.table_builder import table_to_html
from ..schema import ChartBlock, ImageBlock, StructuredDocument, TableBlock, TextBlock
from ..textutils import escape_html
from ..translator.agents import is_numeric_cell, render_table_payload
from ..translator.pipeline import TranslationResult

__all__ = ["render_source_html", "render_target_html", "audit_payloads"]


# ---------------------------------------------------------------------------
# document -> HTML
# ---------------------------------------------------------------------------


def _text_block_html(block: TextBlock, *, bump: int = 0) -> str:
    if block.heading_level:
        level = max(1, min(6, block.heading_level + bump))
        return f"<h{level}>{escape_html(block.text)}</h{level}>"
    return f"<p>{escape_html(block.text)}</p>"


def _block_html(block: Any, *, bump: int = 0) -> str:
    if isinstance(block, TextBlock):
        return _text_block_html(block, bump=bump)
    if isinstance(block, TableBlock):
        parts = []
        if block.caption:
            parts.append(f"<figcaption>{escape_html(block.caption)}</figcaption>")
        if block.units_note:
            parts.append(f'<p class="units">{escape_html(block.units_note)}</p>')
        parts.append(table_to_html(block))
        return '<figure class="tbl">' + "".join(parts) + "</figure>"
    if isinstance(block, ChartBlock):
        return f'<figure class="chart"><figcaption>{escape_html(block.caption)}</figcaption>' + (
            f"<p>{escape_html(block.description)}</p>" if block.description else ""
        ) + "</figure>"
    if isinstance(block, ImageBlock):
        return f'<p class="img">[image] {escape_html(block.caption)}</p>'
    return ""


def render_source_html(document: StructuredDocument) -> str:
    """The parsed source document, with tables rendered as real HTML grids."""
    parts: list[str] = []
    for page in document.pages:
        parts.append(f'<div class="page"><span class="page-no">page {page.index}</span>')
        for block in page.blocks:
            parts.append(_block_html(block))
        parts.append("</div>")
    return "\n".join(parts)


def render_target_html(result: TranslationResult) -> str:
    """The translated document, assembled from each chunk's target blocks.

    Tables are emitted as HTML for the same reason ``to_markdown`` uses HTML:
    Markdown has no ``rowspan``, so a merged header would be duplicated across
    every column it spans.
    """
    parts: list[str] = []
    seen_tables: set[int] = set()
    for chunk in result.chunk_results:
        parts.append(f'<div class="chunk"><span class="chunk-id">{escape_html(chunk.chunk_id)}</span>')
        for block in chunk.target_blocks:
            if isinstance(block, TableBlock):
                if id(block) in seen_tables:
                    continue
                seen_tables.add(id(block))
                parts.append(_block_html(block))
            elif isinstance(block, ChartBlock):
                parts.append(f"<p>{escape_html(chart_to_narrative(block))}</p>")
            else:
                parts.append(_block_html(block))
        parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------


def audit_payloads(document: StructuredDocument) -> dict[str, Any]:
    """Verify that no numeric table cell can reach the model.

    Returns a report the demo page renders verbatim, including any offending
    cell -- a check that cannot fail loudly is not a check.
    """
    label_cells = 0
    numeric_cells = 0
    leaks: list[dict[str, Any]] = []

    for index, table in enumerate(document.tables()):
        payload, _, numeric_count = render_table_payload(table)
        blob = " ".join(str(v) for v in _flatten(payload))
        label_cells += len(table.cells) - numeric_count
        numeric_cells += numeric_count

        for cell in table.cells:
            if not is_numeric_cell(cell.text, unit_hint=table.units_note):
                continue
            if cell.text and cell.text in blob:
                leaks.append(
                    {
                        "table": index,
                        "cell": [cell.row, cell.col],
                        "text": cell.text,
                    }
                )

    return {
        "ok": not leaks,
        "tables_audited": len(document.tables()),
        "label_cells_sent": label_cells,
        "numeric_cells_copied": numeric_cells,
        "leaks": leaks,
    }


def _flatten(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    else:
        yield value
