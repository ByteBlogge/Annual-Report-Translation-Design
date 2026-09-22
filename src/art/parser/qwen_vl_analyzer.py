"""Qwen2.5-VL layout analysis over an OpenAI-compatible endpoint.

The prompt is the product here, so it is written out in full and commented.
Three constraints in it are load-bearing and each fixes a failure mode observed
with vision models on financial pages:

1. **"one object per visually distinct cell, never repeat merged content"** --
   models love to copy a merged label into every column it spans. That produces
   a grid that looks fine and duplicates a caption across four cells.

2. **"transcribe digits exactly, never normalise or round"** -- asked to
   "clean up" a table, models will happily turn ``1,234`` into ``1234`` or
   ``1.2m``. The number guard would then flag the model's own preprocessing as
   hallucination, which is a false positive we can remove at the source.

3. **"bbox in pixels of the supplied image, [x0, y0, x1, y1]"** -- without this
   the model returns normalised 0-1 floats or 0-1000 boxes. We accept those in
   ``BBox.from_any`` as a safety net, but asking precisely avoids silently
   mis-scaled geometry, which would corrupt every rowspan.

Note what the prompt does **not** ask for: rowspan, colspan, or row/column
indices. Those are computed geometrically in :mod:`art.parser.table_builder`.
Asking a language model to count grid spans is asking it to do the one thing it
is reliably bad at; asking it to read text and boxes is asking it to do the one
thing it is reliably good at.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..translator.llm import LLMClient, LLMError
from .analyzer import (
    LayoutAnalyzer,
    PageSource,
    RawPageLayout,
    RawRegion,
    extract_json_object,
    register_analyzer,
)

__all__ = ["QwenVLLayoutAnalyzer", "LAYOUT_SYSTEM_PROMPT", "build_layout_prompt"]


LAYOUT_SYSTEM_PROMPT = """You are a document layout analysis engine for financial reports.

You receive one page image and return a single JSON object. Return JSON only --
no prose, no markdown fences, no explanation.

Schema:
{
  "regions": [
    {
      "label": "text" | "table" | "chart" | "image",
      "bbox": [x0, y0, x1, y1],
      "text": "<for label=text: the full text of the block, line breaks as \\n>",
      "caption": "<for table/chart/image: the caption or title, if present>",
      "font_size": <for label=text: approximate glyph height in pixels>,
      "cells": [
        {"text": "<cell text>", "bbox": [x0, y0, x1, y1]}
      ],
      "units": "<for label=table: the unit declaration, e.g. 单位：人民币千元 or RMB'000>",
      "chart": {
        "chart_type": "bar" | "line" | "pie" | "area" | "waterfall" | "unknown",
        "series": [{"name": "<series name>", "points": {"<category>": <number>}}],
        "axis_labels": {"x": ["..."], "y": ["..."]},
        "description": "<2-4 sentences stating what the chart shows and its key figures>",
        "confidence": <0.0-1.0>
      }
    }
  ]
}

Rules you must follow exactly:

1. Coordinates are PIXELS in the supplied image, origin top-left, format
   [x0, y0, x1, y1] where x1>x0 and y1>y0. Never emit normalised 0-1 values.
2. Do NOT output rowspan, colspan, or row/column indices. Geometry is computed
   downstream. Your job is text plus boxes only.
3. For a table, emit ONE object per visually distinct cell. When a cell is
   merged across several columns, emit it ONCE with its true full bounding box.
   Never repeat a merged cell's text in the cells it spans.
4. Transcribe digits EXACTLY as printed. Do not add, remove, or reposition
   thousands separators, do not convert 1.2m to 1200000, do not round. Preserve
   the printed string character-for-character.
5. Transcribe text in the page's original language. Do not translate anything.
6. For every chart, extract the data points into "series" even when they are
   only readable from axis labels and bar heights. If a value cannot be
   determined, omit that point rather than guessing, and lower "confidence".
7. Every region must have a bbox. Omit decorative rules, page numbers, headers
   and footers rather than inventing a text region for them.
8. If the page is blank or unreadable, return {"regions": []}.
"""


def build_layout_prompt(page: PageSource, *, prior_pages: str = "", target_language: str = "") -> str:
    """Assemble the user turn for one page.

    ``prior_pages`` carries a short digest of already-analysed pages so the
    model keeps field names and units consistent across a long document (a
    table on page 40 should use the same unit note as page 12).
    """
    parts = [
        f"page_index: {page.page_index}",
        f"page_size_px: width={page.width:.0f} height={page.height:.0f}",
    ]
    if target_language:
        parts.append(f"document_language: {target_language}")
    if prior_pages:
        parts.append(
            "consistency_context (already analysed, keep terminology and units consistent):\n"
            + prior_pages.strip()[:1200]
        )
    if page.text_layer.strip():
        parts.append(
            "embedded_text_layer (authoritative for wording; use it to fix OCR errors, "
            "but still report the boxes you see):\n" + page.text_layer.strip()[:3000]
        )
    parts.append("Analyse the attached page image and return the JSON object.")
    return "\n\n".join(parts)


@register_analyzer("qwen-vl")
class QwenVLLayoutAnalyzer(LayoutAnalyzer):
    """Vision-language layout analyzer.

    ``max_retries_per_page`` exists because a single malformed JSON reply should
    not abort an 80-page report. On repeated failure the page degrades to the
    text layer with a warning, which downstream stages treat as a HITL item --
    a partial page is recoverable, a crashed run is not.
    """

    name = "qwen-vl"
    is_remote = True

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_retries_per_page: int = 2,
        max_tokens: int = 8192,
        target_language: str = "",
        **_: Any,
    ) -> None:
        self.llm = llm
        self.max_retries_per_page = max(1, int(max_retries_per_page))
        self.max_tokens = max_tokens
        self.target_language = target_language
        self._digest: list[str] = []

    # -- LayoutAnalyzer ----------------------------------------------------

    def analyze_page(self, page: PageSource) -> RawPageLayout:
        if not page.has_image:
            return RawPageLayout(
                page_index=page.page_index,
                width=page.width,
                height=page.height,
                analyzer=self.name,
                warnings=["page has no image; vision analysis skipped"],
            )

        image = page.image
        if image is None:
            from pathlib import Path

            image = Path(page.image_path).read_bytes()

        prompt = build_layout_prompt(page, prior_pages="\n".join(self._digest), target_language=self.target_language)
        last_error: Exception | None = None

        for attempt in range(self.max_retries_per_page):
            try:
                reply = self.llm.complete_vision(
                    system=LAYOUT_SYSTEM_PROMPT,
                    user=prompt if attempt == 0 else prompt + "\n\nReturn valid JSON only. Previous attempt failed to parse.",
                    image_bytes=image,
                    mime=page.metadata.get("mime", "image/png"),
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
                data = extract_json_object(reply)
                layout = self._to_layout(data, page)
                self._remember(layout)
                return layout
            except (ValueError, LLMError) as exc:
                last_error = exc

        return self._degrade(page, last_error)

    # -- conversion --------------------------------------------------------

    def _to_layout(self, data: dict[str, Any], page: PageSource) -> RawPageLayout:
        raw_regions = data.get("regions")
        if not isinstance(raw_regions, list):
            raise ValueError("reply has no 'regions' array")
        regions: list[RawRegion] = []
        warnings: list[str] = []
        for entry in raw_regions:
            if not isinstance(entry, dict):
                continue
            region = RawRegion.from_dict(entry)
            if region.bbox is None:
                warnings.append(f"region {region.label!r} had no usable bbox; dropped")
                continue
            if region.kind == "table" and not region.cells:
                warnings.append(f"table at {region.bbox.to_list()} returned no cells")
            regions.append(region)
        return RawPageLayout(
            page_index=page.page_index,
            width=page.width,
            height=page.height,
            regions=regions,
            analyzer=self.name,
            warnings=warnings,
        )

    def _degrade(self, page: PageSource, error: Exception | None) -> RawPageLayout:
        """Fall back to the embedded text layer rather than losing the page."""
        warnings = [f"vision analysis failed after {self.max_retries_per_page} attempts: {error}"]
        regions: list[RawRegion] = []
        if page.text_layer.strip():
            regions.append(RawRegion(label="text", text=page.text_layer, bbox=None))
            warnings.append("degraded to embedded text layer; no table structure recovered")
        return RawPageLayout(
            page_index=page.page_index,
            width=page.width,
            height=page.height,
            regions=regions,
            analyzer=f"{self.name}:degraded",
            warnings=warnings,
        )

    def _remember(self, layout: RawPageLayout) -> None:
        """Keep a compact digest for cross-page consistency."""
        bits: list[str] = []
        for region in layout.regions[:8]:
            if region.kind == "table" and region.units_note:
                bits.append(f"units={region.units_note}")
            if region.caption:
                bits.append(f"caption={region.caption[:60]}")
        if bits:
            self._digest.append(f"page {layout.page_index}: " + "; ".join(dict.fromkeys(bits)))
            self._digest = self._digest[-6:]


def describe_layout(layout: RawPageLayout) -> str:
    """Compact human/log rendering used by the CLI and the run report."""
    lines = [f"page {layout.page_index} ({layout.analyzer}) {layout.width:.0f}x{layout.height:.0f}"]
    for region in layout.sorted_regions():
        box = "-" if region.bbox is None else ",".join(f"{v:.0f}" for v in region.bbox.to_list())
        detail = ""
        if region.kind == "table":
            detail = f" cells={len(region.cells)} units={region.units_note!r}"
        elif region.kind == "chart":
            detail = f" type={region.chart.get('chart_type')} points={len(region.chart.get('series', []))}"
        elif region.kind == "text":
            detail = f" {len(region.text)} chars"
        lines.append(f"  [{region.kind:5s}] [{box}] {region.caption or region.text[:48]!r}{detail}")
    return "\n".join(lines)


def dump_layouts(layouts: Sequence[RawPageLayout], path: str) -> None:
    """Persist recordings so a live run can be replayed offline later.

    This is the workflow that makes a remote backend testable: record once
    against the real VLM, then commit the recording and replay it forever.
    """
    from pathlib import Path

    payload = {"pages": [layout.to_dict() for layout in layouts]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
