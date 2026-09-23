"""The Structured Document Object (SDO).

Every stage of the pipeline speaks this one contract:

    analyzer  ->  Block / TableBlock / ChartBlock  ->  DocumentPage  ->  StructuredDocument
                                                              |
                                    chunker <-------+---------+
                                    translator <----+
                                    hitl <----------+

Why a frozen schema instead of passing raw dicts around
-------------------------------------------------------
The whole design rests on one claim: *layout survives the round trip*. If the
parser hands the translator an HTML blob and the translator hands it back,
you cannot mechanically prove that no row was dropped and no number changed.
A typed SDO gives us three things we can actually test and demo:

1. **Auditability** -- every cell carries its source ``bbox`` and the raw
   analyzer payload, so any output can be traced back to a region on a page.
2. **Diffability** -- source and translated tables are both ``TableBlock``s
   with identical ``(row, col)`` indexing, so the number guard can compare
   them cell-by-cell instead of regex-ing two blobs of HTML.
3. **A hard boundary for the number guard** -- numbers are compared as
   domain objects (value, unit, dimension), never as strings.

Serialisation is hand-rolled (not ``dataclasses.asdict``) so that ``BBox``
becomes a compact ``[x0, y0, x1, y1]`` list and enums become their values.
The wire format is intentionally boring and diff-friendly, because the JSON is
a deliverable a human reviewer will read.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dc_fields
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from . import PARSER_VERSION

__all__ = [
    "BlockKind",
    "BBox",
    "TableCell",
    "TableBlock",
    "TextBlock",
    "ChartBlock",
    "ImageBlock",
    "Block",
    "DocumentPage",
    "StructuredDocument",
    "MergeConflict",
]


# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------


class BlockKind(str, Enum):
    """Region label produced by the layout analyzer."""

    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in page coordinates.

    Origin is top-left, units are the analyzer's native page units (points for
    PDF-derived pages, pixels for rasterised pages). We never convert between
    the two inside the schema -- a ``BBox`` is only ever comparable to another
    ``BBox`` from the same page.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    # -- derived geometry --------------------------------------------------

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def intersection_area(self, other: BBox) -> float:
        dx = min(self.x1, other.x1) - max(self.x0, other.x0)
        dy = min(self.y1, other.y1) - max(self.y0, other.y0)
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy

    def iou(self, other: BBox) -> float:
        inter = self.intersection_area(other)
        if inter <= 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: BBox, tol: float = 0.0) -> bool:
        return (
            self.x0 - tol <= other.x0
            and self.y0 - tol <= other.y0
            and self.x1 + tol >= other.x1
            and self.y1 + tol >= other.y1
        )

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    # -- permissive construction ------------------------------------------

    @classmethod
    def from_any(cls, value: Any) -> BBox:
        """Coerce the many shapes a VLM/OCR returns into a ``BBox``.

        Accepts ``[x0, y0, x1, y1]``, ``{"x0":..}``, ``{"left":..}``,
        ``{"x":..,"y":..,"w":..,"h":..}`` and Qwen-VL's normalised
        ``{"bbox_2d": [x0, y0, x1, y1]}`` / 0-1000 scale boxes.
        """
        if isinstance(value, BBox):
            return value
        if isinstance(value, dict):
            if "bbox_2d" in value:
                return cls.from_any(value["bbox_2d"])
            if "bbox" in value:
                return cls.from_any(value["bbox"])
            if {"x0", "y0", "x1", "y1"} <= value.keys():
                return cls(float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"]))
            if {"left", "top", "right", "bottom"} <= value.keys():
                return cls(
                    float(value["left"]),
                    float(value["top"]),
                    float(value["right"]),
                    float(value["bottom"]),
                )
            if {"x", "y", "w", "h"} <= value.keys():
                x, y, w, h = (float(value[k]) for k in ("x", "y", "w", "h"))
                return cls(x, y, x + w, y + h)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return cls(*(float(v) for v in value))
        raise ValueError(f"cannot interpret {value!r} as a BBox")

    # -- JSON ---------------------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BBox:
        return cls.from_any(data)


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


@dataclass
class MergeConflict:
    """A slot that two cells claimed while rebuilding the grid.

    We never silently drop content. Conflicts are recorded, resolved by a
    deterministic rule (see ``table_builder``), and surfaced in the review
    queue -- a merged-cell misread is exactly the kind of parser bug that
    silently corrupts a financial statement.
    """

    row: int
    col: int
    kept: str
    dropped: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"row": self.row, "col": self.col, "kept": self.kept, "dropped": self.dropped, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeConflict:
        return cls(
            row=int(data["row"]),
            col=int(data["col"]),
            kept=str(data["kept"]),
            dropped=str(data["dropped"]),
            reason=str(data["reason"]),
        )


@dataclass
class TableCell:
    """One logical cell after merge resolution.

    ``row``/``col`` address the *span origin* (top-left slot). ``row_span`` and
    ``col_span`` are 1-based and always >= 1. Slots covered by a span but not
    owned by it are represented as ``None`` in ``TableBlock.grid()``.
    """

    text: str = ""
    row: int = -1
    col: int = -1
    row_span: int = 1
    col_span: int = 1
    bbox: BBox | None = None
    is_header: bool = False
    #: Original analyzer payload for this cell -- kept for forensic diffing.
    raw: dict[str, Any] = field(default_factory=dict)

    def covered_slots(self) -> Iterator[tuple[int, int]]:
        """Yield every ``(row, col)`` slot this cell occupies."""
        for r in range(self.row, self.row + max(1, self.row_span)):
            for c in range(self.col, self.col + max(1, self.col_span)):
                yield r, c

    @property
    def is_merged(self) -> bool:
        return self.row_span > 1 or self.col_span > 1

    @property
    def span_area(self) -> int:
        return max(1, self.row_span) * max(1, self.col_span)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "text": self.text,
            "row": self.row,
            "col": self.col,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "is_header": self.is_header,
        }
        if self.bbox is not None:
            out["bbox"] = self.bbox.to_list()
        if self.raw:
            out["raw"] = self.raw
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableCell:
        bbox = data.get("bbox")
        return cls(
            text=str(data.get("text", "")),
            row=int(data.get("row", -1)),
            col=int(data.get("col", -1)),
            row_span=int(data.get("row_span", 1)),
            col_span=int(data.get("col_span", 1)),
            bbox=BBox.from_any(bbox) if bbox is not None else None,
            is_header=bool(data.get("is_header", False)),
            raw=dict(data.get("raw", {})),
        )


@dataclass
class TableBlock:
    """A table with its reconstructed grid.

    ``row_lines`` / ``col_lines`` are the boundary coordinates used during grid
    reconstruction. They are kept on the object because they are the actual
    evidence that ``row_span``/``col_span`` are right: you can plot the cells
    and the lines and see whether they line up. They also make the
    reconstruction reproducible: re-running the builder with the same lines
    must produce the same grid (see ``tests/test_table_builder.py``).
    """

    cells: list[TableCell] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    caption: str = ""
    #: e.g. ``"单位：人民币千元"`` / ``"RMB'000"``. Critically important: a table
    #: header declares a unit, and a translator that drops or mistranslates it
    #: silently multiplies every figure by 1000.
    units_note: str = ""
    bbox: BBox | None = None
    page: int = 0
    row_lines: list[float] = field(default_factory=list)
    col_lines: list[float] = field(default_factory=list)
    conflicts: list[MergeConflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    kind: ClassVar[BlockKind] = BlockKind.TABLE

    # -- grid views ---------------------------------------------------------

    def grid(self) -> list[list[TableCell | None]]:
        """Materialise the ``n_rows x n_cols`` grid.

        Every slot *covered* by a cell holds that cell, so
        ``grid()[r][c]`` answers "which cell covers this position?" -- the
        question the number guard and the HTML emitter both actually ask.
        Use :meth:`is_origin` to tell a span's anchor slot from the slots it
        swallows (only anchors become ``<td>`` elements).
        """
        g: list[list[TableCell | None]] = [[None] * self.n_cols for _ in range(self.n_rows)]
        for cell in self.cells:
            if cell.row < 0 or cell.col < 0:
                continue
            for r, c in cell.covered_slots():
                if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                    g[r][c] = cell
        return g

    def is_origin(self, row: int, col: int) -> bool:
        """True if ``(row, col)`` anchors a cell rather than being swallowed."""
        return any(cell.row == row and cell.col == col for cell in self.cells)

    def holes(self) -> list[tuple[int, int]]:
        """Slots covered by no cell at all -- a symptom of a bad grid read."""
        g = self.grid()
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols) if g[r][c] is None]

    def cell_at(self, row: int, col: int) -> TableCell | None:
        """Return the cell whose span covers ``(row, col)``, or ``None``."""
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        for cell in self.cells:
            for r, c in cell.covered_slots():
                if (r, c) == (row, col):
                    return cell
        return None

    def by_position(self) -> dict[tuple[int, int], TableCell]:
        """Map span-origin ``(row, col)`` -> cell. Used for source/target diffing."""
        return {(c.row, c.col): c for c in self.cells if c.row >= 0 and c.col >= 0}

    def text_cells(self) -> list[TableCell]:
        return [c for c in self.cells if c.text.strip()]

    def numeric_cells(self) -> list[TableCell]:
        from .translator.number_guard import extract_numbers

        return [c for c in self.cells if extract_numbers(c.text)]

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        """Structural self-check. Returns human-readable problems."""
        problems: list[str] = []
        problems.extend(self.warnings)
        if self.n_rows <= 0 or self.n_cols <= 0:
            problems.append(f"degenerate table shape {self.n_rows}x{self.n_cols}")
            return problems
        seen: dict[tuple[int, int], TableCell] = {}
        for cell in self.cells:
            if cell.row_span < 1 or cell.col_span < 1:
                problems.append(f"cell at {cell.row},{cell.col} has non-positive span")
            if cell.row < 0 or cell.col < 0:
                problems.append(f"cell {cell.text[:20]!r} has no grid position")
                continue
            if cell.row + cell.row_span > self.n_rows or cell.col + cell.col_span > self.n_cols:
                problems.append(
                    f"cell at {cell.row},{cell.col} span "
                    f"{cell.row_span}x{cell.col_span} overflows {self.n_rows}x{self.n_cols}"
                )
            for slot in cell.covered_slots():
                if slot in seen and seen[slot] is not cell and slot == (cell.row, cell.col):
                    problems.append(f"duplicate origin at {slot}: {seen[slot].text[:20]!r} vs {cell.text[:20]!r}")
                seen.setdefault(slot, cell)
        return problems

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind.value,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "caption": self.caption,
            "units_note": self.units_note,
            "page": self.page,
            "cells": [c.to_dict() for c in self.cells],
        }
        if self.row_lines:
            out["row_lines"] = list(self.row_lines)
        if self.col_lines:
            out["col_lines"] = list(self.col_lines)
        if self.bbox is not None:
            out["bbox"] = self.bbox.to_list()
        if self.conflicts:
            out["conflicts"] = [c.to_dict() for c in self.conflicts]
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableBlock:
        bbox = data.get("bbox")
        return cls(
            cells=[TableCell.from_dict(c) for c in data.get("cells", [])],
            n_rows=int(data.get("n_rows", 0)),
            n_cols=int(data.get("n_cols", 0)),
            caption=str(data.get("caption", "")),
            units_note=str(data.get("units_note", "")),
            bbox=BBox.from_any(bbox) if bbox is not None else None,
            page=int(data.get("page", 0)),
            row_lines=[float(v) for v in data.get("row_lines", [])],
            col_lines=[float(v) for v in data.get("col_lines", [])],
            conflicts=[MergeConflict.from_dict(c) for c in data.get("conflicts", [])],
            warnings=[str(w) for w in data.get("warnings", [])],
        )


# ---------------------------------------------------------------------------
# text / chart / image
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    """A paragraph, list item, heading or footnote."""

    text: str = ""
    #: ``None`` for body text; otherwise heading depth (1 = top level).
    heading_level: int | None = None
    bbox: BBox | None = None
    page: int = 0
    lang: str = ""
    #: Median glyph height reported by the analyzer -- the cheapest reliable
    #: heading signal when numbering is absent.
    font_size: float = 0.0
    #: Page-relative reading order index assigned by the parser.
    order: int = 0

    kind: ClassVar[BlockKind] = BlockKind.TEXT

    @property
    def is_heading(self) -> bool:
        return self.heading_level is not None

    @property
    def is_footnote(self) -> bool:
        stripped = self.text.lstrip()
        return stripped.startswith(("注：", "注:", "Note:", "Notes:", "(*", "（*"))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind.value,
            "text": self.text,
            "heading_level": self.heading_level,
            "page": self.page,
            "order": self.order,
        }
        if self.bbox is not None:
            out["bbox"] = self.bbox.to_list()
        if self.lang:
            out["lang"] = self.lang
        if self.font_size:
            out["font_size"] = self.font_size
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextBlock:
        bbox = data.get("bbox")
        level = data.get("heading_level")
        return cls(
            text=str(data.get("text", "")),
            heading_level=int(level) if level is not None else None,
            bbox=BBox.from_any(bbox) if bbox is not None else None,
            page=int(data.get("page", 0)),
            lang=str(data.get("lang", "")),
            font_size=float(data.get("font_size", 0.0)),
            order=int(data.get("order", 0)),
        )


@dataclass
class ChartBlock:
    """A chart region plus the *numbers extracted from it*.

    This is the part of the design that is easiest to hand-wave and hardest to
    build. A chart is not translatable text; it is a data carrier. We make the
    extraction explicit and auditable:

    * ``series`` keeps ``(label, value)`` pairs so the values can be fed to the
      number guard exactly like table cells.
    * ``axis_labels`` keeps the untranslatable-vs-translatable split visible.
    * ``description`` is the generated natural-language rendering that goes
      into the translated document, so the reader of the target language gets
      the same information the reader of the source had.
    * ``extraction_method`` records whether values came from a real VLM read,
      a data-label OCR pass, or were unresolved (in which case the chart is a
      HITL item rather than a silent omission).
    """

    caption: str = ""
    chart_type: str = ""  # bar | line | pie | waterfall | unknown
    series: list[dict[str, Any]] = field(default_factory=list)
    axis_labels: dict[str, list[str]] = field(default_factory=dict)
    description: str = ""
    extraction_method: str = "unresolved"  # vlm | ocr | fixture | unresolved
    confidence: float = 0.0
    bbox: BBox | None = None
    page: int = 0
    warnings: list[str] = field(default_factory=list)

    kind: ClassVar[BlockKind] = BlockKind.CHART

    @property
    def points(self) -> list[tuple[str, float]]:
        """Flatten every series into ``(label, value)`` pairs.

        Accepts the two shapes a VLM realistically returns:
        ``{"name": "Revenue", "points": {"2023": 1234.5}}`` and
        ``{"label": "Revenue", "value": 1234.5}``.
        """
        out: list[tuple[str, float]] = []
        for s in self.series:
            label = str(s.get("label", s.get("name", "")))
            points = s.get("points")
            if isinstance(points, dict):
                for name, value in points.items():
                    try:
                        out.append((f"{label}/{name}" if label else str(name), float(value)))
                    except (TypeError, ValueError):
                        continue
            if "value" in s:
                try:
                    out.append((label, float(s["value"])))
                except (TypeError, ValueError):
                    continue
        return out

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind.value,
            "caption": self.caption,
            "chart_type": self.chart_type,
            "series": self.series,
            "axis_labels": self.axis_labels,
            "description": self.description,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "page": self.page,
        }
        if self.bbox is not None:
            out["bbox"] = self.bbox.to_list()
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChartBlock:
        bbox = data.get("bbox")
        return cls(
            caption=str(data.get("caption", "")),
            chart_type=str(data.get("chart_type", "")),
            series=list(data.get("series", [])),
            axis_labels=dict(data.get("axis_labels", {})),
            description=str(data.get("description", "")),
            extraction_method=str(data.get("extraction_method", "unresolved")),
            confidence=float(data.get("confidence", 0.0)),
            bbox=BBox.from_any(bbox) if bbox is not None else None,
            page=int(data.get("page", 0)),
            warnings=[str(w) for w in data.get("warnings", [])],
        )


@dataclass
class ImageBlock:
    """A non-chart figure (photo, logo, signature, stamp)."""

    caption: str = ""
    bbox: BBox | None = None
    page: int = 0
    #: Relative path inside the run directory, so exports stay self-contained.
    image_ref: str = ""
    alt_text: str = ""

    kind: ClassVar[BlockKind] = BlockKind.IMAGE

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind.value,
            "caption": self.caption,
            "page": self.page,
            "image_ref": self.image_ref,
            "alt_text": self.alt_text,
        }
        if self.bbox is not None:
            out["bbox"] = self.bbox.to_list()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageBlock:
        bbox = data.get("bbox")
        return cls(
            caption=str(data.get("caption", "")),
            bbox=BBox.from_any(bbox) if bbox is not None else None,
            page=int(data.get("page", 0)),
            image_ref=str(data.get("image_ref", "")),
            alt_text=str(data.get("alt_text", "")),
        )


Block = TextBlock | TableBlock | ChartBlock | ImageBlock

_BLOCK_TYPES: dict[str, type] = {
    BlockKind.TEXT.value: TextBlock,
    BlockKind.TABLE.value: TableBlock,
    BlockKind.CHART.value: ChartBlock,
    BlockKind.IMAGE.value: ImageBlock,
}


def block_from_dict(data: dict[str, Any]) -> Block:
    """Dispatch a serialised block back to its concrete class."""
    kind = str(data.get("kind", "text"))
    try:
        cls = _BLOCK_TYPES[kind]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown block kind {kind!r}") from exc
    return cls.from_dict(data)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------


@dataclass
class DocumentPage:
    index: int = 0
    width: float = 0.0
    height: float = 0.0
    blocks: list[Block] = field(default_factory=list)
    #: Rasterised page image path, relative to the run directory (if any).
    image_ref: str = ""
    #: Which analyzer produced this page -- useful when mixing backends
    #: (e.g. VLM for table pages, fast OCR for text-only pages).
    analyzer: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
        }
        if self.image_ref:
            out["image_ref"] = self.image_ref
        if self.analyzer:
            out["analyzer"] = self.analyzer
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentPage:
        return cls(
            index=int(data.get("index", 0)),
            width=float(data.get("width", 0.0)),
            height=float(data.get("height", 0.0)),
            blocks=[block_from_dict(b) for b in data.get("blocks", [])],
            image_ref=str(data.get("image_ref", "")),
            analyzer=str(data.get("analyzer", "")),
            warnings=[str(w) for w in data.get("warnings", [])],
        )


@dataclass
class StructuredDocument:
    """The SDO: one annual report, parsed, structured, and self-describing."""

    doc_id: str = ""
    source: str = ""
    pages: list[DocumentPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_version: str = PARSER_VERSION
    warnings: list[str] = field(default_factory=list)

    # -- traversal ----------------------------------------------------------

    def iter_blocks(self) -> Iterator[tuple[int, Block]]:
        """Yield ``(page_index, block)`` in reading order across the document."""
        for page in self.pages:
            for block in page.blocks:
                yield page.index, block

    def blocks(self) -> list[Block]:
        return [b for _, b in self.iter_blocks()]

    def tables(self) -> list[TableBlock]:
        return [b for b in self.blocks() if isinstance(b, TableBlock)]

    def charts(self) -> list[ChartBlock]:
        return [b for b in self.blocks() if isinstance(b, ChartBlock)]

    def texts(self) -> list[TextBlock]:
        return [b for b in self.blocks() if isinstance(b, TextBlock)]

    def find_blocks(self, cls: type) -> list[Any]:
        return [b for b in self.blocks() if isinstance(b, cls)]

    # -- stats --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        counts = {k.value: 0 for k in BlockKind}
        for b in self.blocks():
            counts[b.kind.value] += 1
        tables = self.tables()
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "pages": len(self.pages),
            **{f"{k}_blocks": v for k, v in counts.items()},
            "table_cells": sum(len(t.cells) for t in tables),
            "merged_cells": sum(1 for t in tables for c in t.cells if c.is_merged),
            "merge_conflicts": sum(len(t.conflicts) for t in tables),
            "charts": len(self.charts()),
            "warnings": len(self.warnings) + sum(len(p.warnings) for p in self.pages),
        }

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "doc_id": self.doc_id,
            "source": self.source,
            "parser_version": self.parser_version,
            "pages": [p.to_dict() for p in self.pages],
        }
        if self.metadata:
            out["metadata"] = self.metadata
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredDocument:
        return cls(
            doc_id=str(data.get("doc_id", "")),
            source=str(data.get("source", "")),
            pages=[DocumentPage.from_dict(p) for p in data.get("pages", [])],
            metadata=dict(data.get("metadata", {})),
            parser_version=str(data.get("parser_version", PARSER_VERSION)),
            warnings=[str(w) for w in data.get("warnings", [])],
        )

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str | os.PathLike[str], indent: int | None = 2) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> StructuredDocument:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def jsonable(obj: Any) -> Any:
    """Generic fallback converter for ad-hoc dataclasses in later stages."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BBox):
        return obj.to_list()
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dc_fields(obj)}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
