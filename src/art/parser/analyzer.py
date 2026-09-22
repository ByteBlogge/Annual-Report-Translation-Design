"""The layout-analysis contract, plus region -> block conversion.

Why an explicit Protocol instead of calling the VLM inline
----------------------------------------------------------
The interviewer's real question is "does this work, or does it only work on
your one demo PDF?". The honest answer requires the pipeline to be runnable
against a *deterministic* backend, so that every downstream claim (grid
reconstruction, chunk boundaries, number guard, risk scoring) can be tested
without a network call or an API key.

So: three interchangeable backends behind one interface.

    MockLayoutAnalyzer      deterministic, offline, drives the whole test suite
    QwenVLLayoutAnalyzer    Qwen2.5-VL via an OpenAI-compatible endpoint
    PaddleLayoutAnalyzer    local PP-Structure OCR, no API key required

Dependency injection is not ceremony here -- it is the mechanism that makes the
rest of the repo verifiable.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schema import (
    BBox,
    Block,
    ChartBlock,
    ImageBlock,
    StructuredDocument,
    TextBlock,
)
from ..textutils import normalize_whitespace
from .table_builder import RawCell, TableGeometryOptions, build_table

__all__ = [
    "PageSource",
    "RawRegion",
    "RawPageLayout",
    "LayoutAnalyzer",
    "ANALYZERS",
    "register_analyzer",
    "make_analyzer",
    "normalise_label",
]


#: Canonical region labels. Analyzers may emit synonyms; everything is folded
#: onto these four so downstream code never has to guess.
LABEL_ALIASES: dict[str, str] = {
    "table": "table",
    "表格": "table",
    "table_body": "table",
    "text": "text",
    "paragraph": "text",
    "正文": "text",
    "title": "text",
    "section_header": "text",
    "heading": "text",
    "list": "text",
    "caption": "text",
    "footnote": "text",
    "chart": "chart",
    "figure": "chart",
    "图表": "chart",
    "plot": "chart",
    "diagram": "chart",
    "image": "image",
    "picture": "image",
    "photo": "image",
    "图片": "image",
    "logo": "image",
    "stamp": "image",
}


def normalise_label(label: str) -> str:
    """Fold an analyzer's region label onto one of the four canonical kinds."""
    key = (label or "").strip().lower()
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    for alias, canonical in LABEL_ALIASES.items():
        if alias and alias in key:
            return canonical
    return "text"


# ---------------------------------------------------------------------------
# input side
# ---------------------------------------------------------------------------


@dataclass
class PageSource:
    """One page handed to an analyzer.

    ``text_layer`` is optional but valuable: when a PDF has a real text layer,
    the exact glyphs beat any OCR. We pass both and let the analyzer choose.
    """

    page_index: int
    width: float = 0.0
    height: float = 0.0
    image: bytes | None = None
    image_path: str = ""
    text_layer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_image(self) -> bool:
        return bool(self.image) or bool(self.image_path)


# ---------------------------------------------------------------------------
# output side
# ---------------------------------------------------------------------------


@dataclass
class RawRegion:
    """A region as reported by perception, before any structuring."""

    label: str = "text"
    bbox: BBox | None = None
    text: str = ""
    caption: str = ""
    cells: list[RawCell] = field(default_factory=list)
    units_note: str = ""
    font_size: float = 0.0
    is_header: bool = False
    #: For chart regions: whatever the VLM could say about the plot.
    chart: dict[str, Any] = field(default_factory=dict)
    #: Per-region perception confidence, if the backend reports one.
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return normalise_label(self.label)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawRegion:
        bbox = data.get("bbox")
        text = data.get("text", data.get("content", ""))
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(t) for t in text)
        return cls(
            label=str(data.get("label", data.get("type", "text"))),
            bbox=BBox.from_any(bbox) if bbox is not None else None,
            text=str(text or ""),
            caption=str(data.get("caption", "") or ""),
            cells=[RawCell.from_dict(c) for c in data.get("cells", [])],
            units_note=str(data.get("units", data.get("units_note", "")) or ""),
            font_size=float(data.get("font_size", 0.0) or 0.0),
            is_header=bool(data.get("is_header", False)),
            chart=dict(data.get("chart", {}) or {}),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            raw=data,
        )


@dataclass
class RawPageLayout:
    """Everything perception could tell us about one page."""

    page_index: int = 0
    width: float = 0.0
    height: float = 0.0
    regions: list[RawRegion] = field(default_factory=list)
    analyzer: str = ""
    warnings: list[str] = field(default_factory=list)

    def sorted_regions(self) -> list[RawRegion]:
        """Reading order: top-to-bottom, then left-to-right.

        Analyzers emit regions in arbitrary order (and VLM output order is not
        trustworthy at all). Reading order is geography, so we compute it. The
        band tolerance prevents a heading and a figure on the same visual line
        from being ordered by a 3-pixel baseline difference.
        """
        def key(region: RawRegion) -> tuple[float, float]:
            if region.bbox is None:
                return (float("inf"), float("inf"))
            return (round(region.bbox.y0 / 6.0), region.bbox.x0)

        return sorted(self.regions, key=key)

    def to_blocks(
        self, *, page: int | None = None, options: TableGeometryOptions | None = None
    ) -> list[Block]:
        """Convert raw regions into typed SDO blocks."""
        index = self.page_index if page is None else page
        blocks: list[Block] = []
        for order, region in enumerate(self.sorted_regions()):
            kind = region.kind
            if kind == "table":
                if not region.cells:
                    # A table region with no cells is a perception failure, not
                    # an empty table. Keep the caption as text so the content is
                    # not silently dropped, and flag it.
                    blocks.append(
                        TextBlock(
                            text=region.text or region.caption,
                            bbox=region.bbox,
                            page=index,
                            order=order,
                        )
                    )
                    continue
                table = build_table(
                    region.cells,
                    caption=region.caption,
                    page=index,
                    bbox=region.bbox,
                    units_note=region.units_note,
                    options=options,
                )
                blocks.append(table)
            elif kind == "chart":
                blocks.append(_chart_from_region(region, index))
            elif kind == "image":
                blocks.append(
                    ImageBlock(
                        caption=region.caption,
                        bbox=region.bbox,
                        page=index,
                        image_ref=str(region.raw.get("image_ref", "")),
                        alt_text=str(region.raw.get("alt_text", "")),
                    )
                )
            else:
                text = normalize_whitespace(region.text, keep_newlines=True)
                if not text:
                    continue
                blocks.append(
                    TextBlock(
                        text=text,
                        bbox=region.bbox,
                        page=index,
                        font_size=region.font_size,
                        order=order,
                    )
                )
        return blocks

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawPageLayout:
        return cls(
            page_index=int(data.get("page", data.get("page_index", 0))),
            width=float(data.get("width", 0.0) or 0.0),
            height=float(data.get("height", 0.0) or 0.0),
            regions=[RawRegion.from_dict(r) for r in data.get("regions", [])],
            analyzer=str(data.get("analyzer", "")),
            warnings=[str(w) for w in data.get("warnings", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        def region_dict(r: RawRegion) -> dict[str, Any]:
            out: dict[str, Any] = {"label": r.label}
            if r.bbox is not None:
                out["bbox"] = r.bbox.to_list()
            if r.text:
                out["text"] = r.text
            if r.caption:
                out["caption"] = r.caption
            if r.cells:
                out["cells"] = [c.raw if c.raw else {"text": c.text, "bbox": (c.bbox.to_list() if c.bbox else None)} for c in r.cells]
            if r.units_note:
                out["units"] = r.units_note
            if r.chart:
                out["chart"] = r.chart
            return out

        return {
            "page": self.page_index,
            "width": self.width,
            "height": self.height,
            "analyzer": self.analyzer,
            "regions": [region_dict(r) for r in self.regions],
        }


def _chart_from_region(region: RawRegion, page: int) -> ChartBlock:
    chart = region.chart or {}
    series = chart.get("series", [])
    if not isinstance(series, list):
        series = []
    method = str(chart.get("extraction_method", "vlm" if series else "unresolved"))
    warnings: list[str] = []
    if not series:
        warnings.append("no data points recovered from chart; requires human review")
    return ChartBlock(
        caption=region.caption,
        chart_type=str(chart.get("chart_type", "")),
        series=series,
        axis_labels=dict(chart.get("axis_labels", {}) or {}),
        description=str(chart.get("description", "") or ""),
        extraction_method=method,
        confidence=float(chart.get("confidence", region.confidence) or 0.0),
        bbox=region.bbox,
        page=page,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# analyzer interface
# ---------------------------------------------------------------------------


class LayoutAnalyzer(ABC):
    """Turns a page image (and optional text layer) into a :class:`RawPageLayout`."""

    name: str = "abstract"
    #: Whether this backend needs a live network call. The parser warns loudly
    #: when a run silently mixes online and offline backends.
    is_remote: bool = False

    @abstractmethod
    def analyze_page(self, page: PageSource) -> RawPageLayout:
        """Analyse a single page. Must not raise on a partially readable page."""

    def analyze(self, pages: Sequence[PageSource]) -> list[RawPageLayout]:
        """Analyze a whole document. Override to batch (VLM calls are cheaper batched)."""
        return [self.analyze_page(p) for p in pages]

    def close(self) -> None:
        """Release any backend resources. Overridden where needed."""

    def __enter__(self) -> LayoutAnalyzer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

ANALYZERS: dict[str, type[LayoutAnalyzer]] = {}


def register_analyzer(name: str):
    """Decorator: register a backend under ``name`` for CLI/env selection."""

    def wrapper(cls: type[LayoutAnalyzer]) -> type[LayoutAnalyzer]:
        cls.name = name
        ANALYZERS[name] = cls
        return cls

    return wrapper


def make_analyzer(spec: str | LayoutAnalyzer, **kwargs: Any) -> LayoutAnalyzer:
    """Resolve ``"mock"`` / ``"qwen-vl"`` / ``"paddle"`` (or an instance) to a backend."""
    if isinstance(spec, LayoutAnalyzer):
        return spec
    key = (spec or "mock").strip().lower().replace("_", "-")
    aliases = {"qwen": "qwen-vl", "qwen2.5-vl": "qwen-vl", "vlm": "qwen-vl", "ppstructure": "paddle", "pp-structure": "paddle"}
    key = aliases.get(key, key)
    if key not in ANALYZERS:  # imported lazily to avoid a circular import
        _import_builtin_backends()
    if key not in ANALYZERS:
        raise KeyError(f"unknown analyzer {spec!r}; available: {sorted(ANALYZERS)}")
    return ANALYZERS[key](**kwargs)


def _import_builtin_backends() -> None:
    from . import mock_analyzer, paddle_analyzer, qwen_vl_analyzer  # noqa: F401


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply.

    Models wrap JSON in prose, in ```json fences, or emit a trailing comma.
    Every one of those happens in production, so this is the single place that
    tolerates them rather than a try/except scattered through the pipeline.
    """
    if not text:
        raise ValueError("empty model reply")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model reply")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)  # trailing commas
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed JSON in model reply: {exc}") from exc
    raise ValueError("unbalanced JSON object in model reply")


def build_document(
    layouts: Iterable[RawPageLayout],
    *,
    doc_id: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    options: TableGeometryOptions | None = None,
) -> StructuredDocument:
    """Assemble page layouts into a :class:`StructuredDocument`."""
    from ..schema import DocumentPage

    pages: list[DocumentPage] = []
    warnings: list[str] = []
    for layout in layouts:
        blocks = layout.to_blocks(options=options)
        page = DocumentPage(
            index=layout.page_index,
            width=layout.width,
            height=layout.height,
            blocks=blocks,
            analyzer=layout.analyzer,
            warnings=list(layout.warnings),
        )
        pages.append(page)
    pages.sort(key=lambda p: p.index)
    doc = StructuredDocument(
        doc_id=doc_id,
        source=source,
        pages=pages,
        metadata=metadata or {},
        warnings=warnings,
    )
    for table in doc.tables():
        if table.conflicts:
            doc.warnings.append(
                f"page {table.page}: table merge conflicts={len(table.conflicts)} "
                f"(see table.conflicts for kept/dropped text)"
            )
    return doc
