"""Local OCR layout analyzer (PaddleOCR PP-Structure), no API key required.

Why keep an offline perception path at all
------------------------------------------
Two real reasons, both of which came up while building this:

* **Cost and confidentiality.** An annual report is 150-250 dense pages. At one
  VLM call per page that is 200 calls per document, with the full financial
  statements leaving the building. Regulators and audit committees care about
  that. PP-Structure runs locally and is free.
* **Fallback.** When the VLM endpoint rate-limits or a page returns malformed
  JSON, a local analyzer keeps the run moving.

Its weakness is the mirror image: its table structure model is good at plain
grids and struggles with heavy merged-cell layouts, and its chart reading is
essentially nil. Hence the split: use PP-Structure for text-dominant pages and
the VLM for table/chart pages, and let the mixed-analyzer policy choose per
page. The module is imported lazily so the core install stays dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..schema import BBox
from .analyzer import (
    LayoutAnalyzer,
    PageSource,
    RawPageLayout,
    RawRegion,
    register_analyzer,
)
from .table_builder import RawCell, table_from_html

__all__ = ["PaddleLayoutAnalyzer"]

_INSTALL_HINT = (
    "PaddleOCR backend requested but not installed. Install it with:\n"
    "    pip install -e \".[ocr]\"\n"
    "or use the offline backend (`--parser mock`) or the API backend (`--parser qwen-vl`)."
)


@register_analyzer("paddle")
class PaddleLayoutAnalyzer(LayoutAnalyzer):
    """Wraps PP-Structure's layout + table + OCR pipeline.

    Parameters
    ----------
    lang:
        PaddleOCR language pack. ``"ch"`` handles mixed Chinese/English, which
        is what a bilingual annual report needs.
    use_table:
        Enable the table structure model. Its HTML output is fed through
        :func:`art.parser.table_builder.table_from_html`, which normalises it
        onto the same grid representation the VLM path produces -- so both
        backends yield identical downstream types, not two parallel formats.
    """

    name = "paddle"
    is_remote = False

    def __init__(
        self,
        *,
        lang: str = "ch",
        use_table: bool = True,
        use_pdfplumber: bool = False,
        device: str = "cpu",
        show_log: bool = False,
        **_: Any,
    ) -> None:
        self.lang = lang
        self.use_table = use_table
        self.device = device
        self.show_log = show_log
        self._engine: Any = None

    # -- engine lifecycle ---------------------------------------------------

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from paddleocr import PPStructure  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(_INSTALL_HINT) from exc
        self._engine = PPStructure(
            layout=True,
            table=self.use_table,
            ocr=True,
            lang=self.lang,
            show_log=self.show_log,
            device=self.device,
        )
        return self._engine

    def close(self) -> None:
        self._engine = None

    # -- analysis -----------------------------------------------------------

    def analyze_page(self, page: PageSource) -> RawPageLayout:
        engine = self._ensure_engine()
        if not page.has_image:
            return RawPageLayout(
                page_index=page.page_index,
                width=page.width,
                height=page.height,
                analyzer=self.name,
                warnings=["page has no image; OCR skipped"],
            )

        import io

        import numpy as np
        from PIL import Image  # type: ignore[import-not-found]

        source = page.image
        if source is None:
            from pathlib import Path

            source = Path(page.image_path).read_bytes()
        array = np.array(Image.open(io.BytesIO(source)).convert("RGB"))

        raw = engine(array)
        regions, warnings = self._convert(raw, page)
        return RawPageLayout(
            page_index=page.page_index,
            width=page.width,
            height=page.height,
            regions=regions,
            analyzer=self.name,
            warnings=warnings,
        )

    # -- result conversion --------------------------------------------------

    def _convert(self, raw: Any, page: PageSource) -> tuple[list[RawRegion], list[str]]:
        regions: list[RawRegion] = []
        warnings: list[str] = []
        if not isinstance(raw, (list, tuple)):
            return regions, [f"unexpected PP-Structure payload: {type(raw).__name__}"]

        for item in raw:
            if not isinstance(item, dict):
                continue
            label = str(item.get("type", "text")).lower()
            bbox = _bbox_of(item)
            payload = item.get("res", item)

            if label == "table":
                html = _table_html(payload)
                if html:
                    table = table_from_html(html, caption=str(item.get("caption", "")), page=page.page_index)
                    regions.append(
                        RawRegion(
                            label="table",
                            bbox=bbox,
                            caption=item.get("caption", ""),
                            cells=[RawCell(text=c.text, bbox=_cell_bbox(table, c.row, c.col), row=c.row,
                                           col=c.col, row_span=c.row_span, col_span=c.col_span)
                                   for c in table.cells],
                            units_note=table.units_note,
                            confidence=float(payload.get("confidence", 0.0) or 0.0) if isinstance(payload, dict) else 0.0,
                            raw=dict(item),
                        )
                    )
                else:
                    warnings.append("table region had no recoverable HTML; kept as text")
                    regions.append(RawRegion(label="text", bbox=bbox, text=_join_lines(payload)))
                continue

            if label in ("figure", "chart", "image"):
                regions.append(
                    RawRegion(
                        label="chart" if label != "image" else "image",
                        bbox=bbox,
                        caption=str(item.get("caption", "")),
                        # PP-Structure does not read chart values. Recording the
                        # failure explicitly is better than emitting a chart
                        # block that silently claims zero data points.
                        chart={"extraction_method": "unresolved", "confidence": 0.0},
                        raw=dict(item),
                    )
                )
                continue

            text = _join_lines(payload)
            if text.strip():
                regions.append(RawRegion(label="text", bbox=bbox, text=text, raw=dict(item)))

        return regions, warnings


def _bbox_of(item: dict[str, Any]) -> BBox | None:
    box = item.get("bbox") or item.get("box") or item.get("coordinate")
    if box is None:
        return None
    try:
        return BBox.from_any(box)
    except (ValueError, TypeError):
        return None


def _table_html(payload: Any) -> str:
    """Pull table HTML out of the several shapes PP-Structure has used."""
    if isinstance(payload, str):
        return payload if "<table" in payload.lower() else ""
    if isinstance(payload, dict):
        for key in ("html", "structure", "table_html"):
            value = payload.get(key)
            if isinstance(value, str) and "<table" in value.lower():
                return value
        inner = payload.get("res")
        if isinstance(inner, dict):
            return _table_html(inner)
    if isinstance(payload, (list, tuple)):
        for entry in payload:
            html = _table_html(entry)
            if html:
                return html
    return ""


def _join_lines(payload: Any) -> str:
    """Flatten OCR line records into a paragraph, preserving line order."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "rec_text", "rec_texts"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)):
                return "\n".join(str(v) for v in value)
        return _join_lines(payload.get("res", ""))
    if isinstance(payload, (list, tuple)):
        lines: list[str] = []
        for entry in payload:
            if isinstance(entry, dict):
                text = entry.get("text") or entry.get("rec_text")
                if text:
                    lines.append(str(text))
            elif isinstance(entry, str):
                lines.append(entry)
        return "\n".join(lines)
    return ""


def _cell_bbox(table: Any, row: int, col: int) -> BBox | None:
    """Reconstruct a cell box from boundary lines, for the SDO audit trail."""
    if not table.row_lines or not table.col_lines:
        return None
    try:
        return BBox(
            table.col_lines[col],
            table.row_lines[row],
            table.col_lines[min(col + 1, len(table.col_lines) - 1)],
            table.row_lines[min(row + 1, len(table.row_lines) - 1)],
        )
    except IndexError:  # pragma: no cover - defensive
        return None


def analyze_sequence(analyzer: PaddleLayoutAnalyzer, pages: Sequence[PageSource]) -> list[RawPageLayout]:
    return [analyzer.analyze_page(p) for p in pages]
