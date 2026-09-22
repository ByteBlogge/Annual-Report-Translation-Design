"""Stage 1 orchestration: source file -> StructuredDocument.

The parser's job is to be boring and reproducible. Two properties are worth
calling out because they are what make a demo into a tool:

* **Record and replay.** ``art parse --record runs/rec`` saves the analyzer's
  raw output. ``MockLayoutAnalyzer(fixture=...)`` replays it. So a run against
  a paid VLM can be turned into a permanent, free, deterministic regression
  fixture -- which is how the test suite gets realistic data without shipping a
  proprietary PDF or depending on a live endpoint.

* **Degrade, never crash.** A page whose VLM call fails returns a warning-laden
  layout instead of raising. An 80-page parse that dies on page 61 and loses all
  prior work is useless in practice; one that finishes with 79 good pages and a
  flagged page 61 is a tool.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import PARSER_VERSION
from ..schema import ChartBlock, StructuredDocument, TableBlock, TextBlock
from ..textutils import slugify
from .analyzer import (
    LayoutAnalyzer,
    PageSource,
    RawPageLayout,
    build_document,
)
from .chart_reader import ChartReader
from .pdf_loader import PageLoadOptions, load_pages
from .table_builder import TableGeometryOptions

__all__ = ["ParserOptions", "DocumentParser", "ParseResult"]


@dataclass
class ParserOptions:
    dpi: int = 180
    include_text_layer: bool = True
    max_pages: int | None = None
    first_page: int = 1
    table_geometry: TableGeometryOptions = field(default_factory=TableGeometryOptions)
    #: Run a dedicated vision pass on chart regions that the page analyzer
    #: could not resolve. Costs one extra call per unresolved chart.
    read_charts: bool = True
    doc_id: str = ""


@dataclass
class ParseResult:
    document: StructuredDocument
    layouts: list[RawPageLayout]
    source_stats: dict[str, Any]

    @property
    def warnings(self) -> list[str]:
        out = list(self.document.warnings)
        for layout in self.layouts:
            out.extend(f"page {layout.page_index}: {w}" for w in layout.warnings)
        return out


class DocumentParser:
    """Turns a PDF / image / recording into a :class:`StructuredDocument`."""

    def __init__(
        self,
        analyzer: LayoutAnalyzer | str | None = None,
        *,
        options: ParserOptions | None = None,
        llm: Any | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.options = options or ParserOptions()
        self.llm = llm
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        if analyzer is None:
            from .mock_analyzer import MockLayoutAnalyzer

            analyzer = MockLayoutAnalyzer()
        if isinstance(analyzer, str):
            from .analyzer import make_analyzer

            analyzer = make_analyzer(analyzer, llm=llm) if analyzer in ("qwen-vl", "qwen") else make_analyzer(analyzer)
        self.analyzer: LayoutAnalyzer = analyzer
        self.chart_reader = ChartReader(llm) if llm is not None else None

    # -- entry points -------------------------------------------------------

    def parse(self, source: str | Path) -> ParseResult:
        """Parse a PDF, an image, a directory of images, or a recording."""
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")

        if path.suffix.lower() == ".json":
            return self.parse_recording(path)

        load_options = PageLoadOptions(
            dpi=self.options.dpi,
            include_text_layer=self.options.include_text_layer,
            max_pages=self.options.max_pages,
            first_page=self.options.first_page,
        )
        pages = load_pages(path, load_options)
        result = self.parse_pages(pages, source=str(path))
        self.record(result.layouts, filename="layouts.json")
        return result

    def parse_recording(self, path: str | Path) -> ParseResult:
        """Load a previously saved recording (layouts) or a finished SDO."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("pages") and _looks_like_sdo(data):
            document = StructuredDocument.from_dict(data)
            return ParseResult(document=document, layouts=[], source_stats={"replayed": "sdo"})

        from .mock_analyzer import MockLayoutAnalyzer

        analyzer = MockLayoutAnalyzer(fixture=data)
        layouts = [
            analyzer.analyze_page(PageSource(page_index=page.page_index))
            for page in analyzer.page_layouts
        ]
        document = build_document(
            layouts,
            doc_id=self.options.doc_id or slugify(Path(path).stem),
            source=str(path),
            metadata={"replayed_from": str(path)},
            options=self.options.table_geometry,
        )
        return ParseResult(document=document, layouts=layouts, source_stats={"replayed": "layouts"})

    def parse_pages(self, pages: Sequence[PageSource], *, source: str = "") -> ParseResult:
        """Analyze already-loaded pages."""
        layouts = self.analyzer.analyze(pages)
        if self.options.read_charts:
            layouts = self._enrich_charts(layouts, pages)

        doc_id = self.options.doc_id or slugify(Path(source).stem if source else "document")
        document = build_document(
            layouts,
            doc_id=doc_id,
            source=source,
            metadata={
                "parser_version": PARSER_VERSION,
                "analyzer": self.analyzer.name,
                "dpi": self.options.dpi,
                "page_count": len(pages),
            },
            options=self.options.table_geometry,
        )
        stats = {
            "analyzer": self.analyzer.name,
            "pages_analysed": len(layouts),
            "regions": sum(len(layout.regions) for layout in layouts),
            "degraded_pages": sum(1 for layout in layouts if "degraded" in layout.analyzer),
        }
        return ParseResult(document=document, layouts=layouts, source_stats=stats)

    # -- chart enrichment ---------------------------------------------------

    def _enrich_charts(
        self, layouts: list[RawPageLayout], pages: Sequence[PageSource]
    ) -> list[RawPageLayout]:
        """Fill in chart data for regions the page analyzer left unresolved."""
        by_index = {p.page_index: p for p in pages}
        for layout in layouts:
            page = by_index.get(layout.page_index)
            if page is None or not page.has_image:
                continue
            for region in layout.regions:
                if region.kind != "chart":
                    continue
                if region.chart.get("series"):
                    continue
                if self.chart_reader is None:
                    continue
                chart = self.chart_reader.read(
                    image_bytes=_crop(page, region.bbox),
                    caption=region.caption,
                    bbox=region.bbox,
                    page=layout.page_index,
                    precomputed=region.chart,
                    mime=page.metadata.get("mime", "image/png"),
                )
                region.chart = {
                    "chart_type": chart.chart_type,
                    "series": chart.series,
                    "axis_labels": chart.axis_labels,
                    "description": chart.description,
                    "extraction_method": chart.extraction_method,
                    "confidence": chart.confidence,
                }
        return layouts

    # -- artefacts ----------------------------------------------------------

    def record(self, layouts: Sequence[RawPageLayout], *, filename: str = "layouts.json") -> Path | None:
        """Persist analyzer output for replay in tests and offline demos."""
        if not self.run_dir:
            return None
        payload = {"pages": [layout.to_dict() for layout in layouts]}
        target = self.run_dir / filename
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _looks_like_sdo(data: dict[str, Any]) -> bool:
    """A finished SDO has pages whose entries carry ``blocks``, not ``regions``."""
    pages = data.get("pages")
    if not isinstance(pages, list) or not pages:
        return False
    first = pages[0]
    return isinstance(first, dict) and "blocks" in first


def _crop(page: PageSource, bbox: Any, *, padding: float = 8.0) -> bytes:
    """Crop a region out of the page image, if Pillow is available.

    Cropping materially improves chart reading accuracy: a VLM shown a 1000px
    page and told "look at this rectangle" still attends to the whole page.
    Without Pillow we pass the full page and accept the accuracy hit -- a
    graceful degradation rather than a hard dependency.
    """
    if bbox is None or page.image is None:
        return page.image or b""
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return page.image
    try:
        image = Image.open(io.BytesIO(page.image))
        x0 = max(0.0, bbox.x0 - padding)
        y0 = max(0.0, bbox.y0 - padding)
        x1 = min(float(image.width), bbox.x1 + padding)
        y1 = min(float(image.height), bbox.y1 + padding)
        if x1 <= x0 or y1 <= y0:
            return page.image
        buffer = io.BytesIO()
        image.crop((int(x0), int(y0), int(x1), int(y1))).save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:  # pragma: no cover - never fail a run over a crop
        return page.image


def summarise_document(document: StructuredDocument) -> dict[str, Any]:
    """Stats used by the CLI, the report, and the README's screenshot section."""
    stats = document.stats()
    tables: list[TableBlock] = document.tables()
    charts: list[ChartBlock] = document.charts()
    texts: list[TextBlock] = document.texts()
    units_notes = {t.units_note for t in tables if t.units_note}
    stats.update(
        {
            "text_chars": sum(len(t.text) for t in texts),
            "tables_with_merged_cells": sum(1 for t in tables if any(c.is_merged for c in t.cells)),
            "tables_with_units_note": sum(1 for t in tables if t.units_note),
            "distinct_units_notes": sorted(units_notes),
            "charts_with_data": sum(1 for c in charts if c.series),
            "charts_unresolved": sum(1 for c in charts if c.extraction_method == "unresolved"),
            "table_warnings": sum(len(t.warnings) for t in tables),
            "table_holes": sum(len(t.holes()) for t in tables),
        }
    )
    return stats


def document_to_markdown(document: StructuredDocument, *, include_tables: bool = True) -> str:
    """Render the parsed source document as Markdown (the translation input view)."""
    from .table_builder import table_to_markdown

    parts: list[str] = []
    for page in document.pages:
        parts.append(f"<!-- page {page.index} -->")
        for block in page.blocks:
            if isinstance(block, TextBlock):
                if block.is_heading:
                    parts.append("#" * min(6, block.heading_level or 1) + " " + block.text)
                else:
                    parts.append(block.text)
            elif isinstance(block, TableBlock) and include_tables:
                if block.caption:
                    parts.append(f"**{block.caption}**")
                if block.units_note:
                    parts.append(f"*{block.units_note}*")
                parts.append(table_to_markdown(block))
            elif isinstance(block, ChartBlock):
                parts.append(f"**[chart] {block.caption}**")
                if block.description:
                    parts.append(block.description)
            else:
                parts.append(f"*[image] {block.caption}*")
            parts.append("")
    return "\n".join(parts)
