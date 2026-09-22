"""Deterministic offline analyzer.

This backend is the reason the repository is *verifiable* rather than
*plausible*. It replays a recorded page layout, so:

* the whole pipeline runs with no API key, no model download, no network;
* CI is fast and has no flaky external dependency;
* geometry bugs become reproducible (fix a seed, get the same page forever).

It also supports **injectable noise** (``jitter_px``, ``drop_indices``,
``shuffle_regions``). Those are not decoration: they are how the test suite
demonstrates that boundary clustering tolerates a VLM whose boxes are a few
pixels off, and that reading order is recovered from geometry rather than
trusted from the model's output order.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..schema import BBox
from .analyzer import (
    LayoutAnalyzer,
    PageSource,
    RawPageLayout,
    RawRegion,
    register_analyzer,
)
from .table_builder import RawCell

__all__ = ["MockLayoutAnalyzer", "BUILTIN_DEMO_PAGE"]


#: A small but structurally realistic page: a title, a two-line prose block, a
#: financial table with a merged header spanning two rows and a merged
#: "Total" row, and a chart. Used when no fixture is supplied so that
#: ``art demo`` works out of the box.
BUILTIN_DEMO_PAGE: dict[str, Any] = {
    "page": 1,
    "width": 1000.0,
    "height": 1400.0,
    "regions": [
        {
            "label": "text",
            "bbox": [60, 60, 940, 110],
            "text": "Annual Report 2023",
            "font_size": 26.0,
        },
        {
            "label": "text",
            "bbox": [60, 130, 940, 230],
            "text": (
                "The board presents the consolidated results of the Group for the year ended "
                "31 December 2023. Revenue for the year was 1,234,567 千元, an increase of "
                "12.4% year on year, and operating profit rose to 234,567 千元."
            ),
            "font_size": 12.0,
        },
        {
            "label": "table",
            "bbox": [60, 260, 940, 620],
            "caption": "Financial Summary",
            "units": "单位：人民币千元",
            "cells": [
                # --- header band (rows 0-1) ---
                # "Item" spans BOTH header rows (rowspan=2) and the period label
                # spans both value columns (colspan=2). Two different merge
                # directions in one band -- the case that breaks naive parsers.
                {"text": "Item", "bbox": [60, 260, 400, 340]},
                {"text": "Year ended 31 December", "bbox": [400, 260, 940, 300]},
                {"text": "2023", "bbox": [400, 300, 670, 340]},
                {"text": "2022", "bbox": [670, 300, 940, 340]},
                # --- body ---
                {"text": "Revenue", "bbox": [60, 340, 400, 380]},
                {"text": "1,234,567", "bbox": [400, 340, 670, 380]},
                {"text": "1,098,765", "bbox": [670, 340, 940, 380]},
                {"text": "Operating profit", "bbox": [60, 380, 400, 420]},
                {"text": "234,567", "bbox": [400, 380, 670, 420]},
                {"text": "198,765", "bbox": [670, 380, 940, 420]},
                {"text": "Profit before tax", "bbox": [60, 420, 400, 460]},
                {"text": "210,000", "bbox": [400, 420, 670, 460]},
                {"text": "180,000", "bbox": [670, 420, 940, 460]},
                {"text": "Depreciation and amortisation", "bbox": [60, 460, 400, 500]},
                {"text": "56,789", "bbox": [400, 460, 670, 500]},
                {"text": "51,234", "bbox": [670, 460, 940, 500]},
                {"text": "Basic earnings per share", "bbox": [60, 500, 400, 540]},
                {"text": "1.23", "bbox": [400, 500, 670, 540]},
                {"text": "1.05", "bbox": [670, 500, 940, 540]},
                # --- total row: fully merged label, values in the two value cols ---
                {"text": "Total assets", "bbox": [60, 540, 400, 580]},
                {"text": "8,765,432", "bbox": [400, 540, 670, 580]},
                {"text": "7,654,321", "bbox": [670, 540, 940, 580]},
                {
                    "text": "Note: figures are audited and prepared under IFRS.",
                    "bbox": [60, 580, 940, 620],
                },
            ],
        },
        {
            "label": "chart",
            "bbox": [60, 660, 940, 1040],
            "caption": "Revenue by segment (RMB'000)",
            "chart": {
                "chart_type": "bar",
                "series": [
                    {"name": "Cloud", "points": {"2023": 512345.0, "2022": 431098.0}},
                    {"name": "Hardware", "points": {"2023": 402222.0, "2022": 441667.0}},
                    {"name": "Services", "points": {"2023": 320000.0, "2022": 226000.0}},
                ],
                "axis_labels": {
                    "x": ["Cloud", "Hardware", "Services"],
                    "y": ["0", "200,000", "400,000", "600,000"],
                },
                "description": (
                    "Bar chart comparing 2023 and 2022 revenue across three segments: "
                    "Cloud grew to RMB512,345 thousand, Hardware declined to RMB402,222 thousand, "
                    "and Services grew to RMB320,000 thousand."
                ),
                "extraction_method": "fixture",
                "confidence": 0.86,
            },
        },
        {
            "label": "text",
            "bbox": [60, 1080, 940, 1180],
            "text": (
                "Cash and cash equivalents increased by 12.4% year on year, driven by improved "
                "working capital management and disciplined capital expenditure."
            ),
            "font_size": 12.0,
        },
        {
            "label": "image",
            "bbox": [60, 1220, 340, 1330],
            "caption": "",
            "image_ref": "",
            "alt_text": "Company seal",
        },
    ],
}


@register_analyzer("mock")
class MockLayoutAnalyzer(LayoutAnalyzer):
    """Replays a recorded layout.

    Parameters
    ----------
    fixture:
        ``None`` (use the built-in page), a path to a JSON file, a list of page
        dicts, a list of :class:`RawPageLayout`, or a single such dict.
    jitter_px:
        Randomly displace every cell/region box by up to this many page units.
        Simulates the small overhang real VLM boxes have.
    drop_indices:
        Strip the ``row``/``col`` hints from cells, forcing the geometry path.
        The VLM backends never supply hints, so this is the realistic default.
    shuffle_regions:
        Emit regions in shuffled order, proving reading order is computed.
    drop_cell_probability:
        Randomly drop cells, simulating a perception miss. Used to show the
        grid validator noticing holes instead of silently misaligning rows.
    """

    name = "mock"
    is_remote = False

    def __init__(
        self,
        fixture: Any = None,
        *,
        jitter_px: float = 0.0,
        drop_indices: bool = True,
        shuffle_regions: bool = False,
        drop_cell_probability: float = 0.0,
        seed: int = 7,
        **_: Any,
    ) -> None:
        self.jitter_px = float(jitter_px)
        self.drop_indices = drop_indices
        self.shuffle_regions = shuffle_regions
        self.drop_cell_probability = float(drop_cell_probability)
        self.seed = seed
        self._rng = random.Random(seed)
        self._layouts = self._load_fixture(fixture)

    # -- fixture loading ----------------------------------------------------

    def _load_fixture(self, fixture: Any) -> list[RawPageLayout]:
        if fixture is None:
            return [RawPageLayout.from_dict(BUILTIN_DEMO_PAGE)]
        if isinstance(fixture, (str, Path)):
            data = json.loads(Path(fixture).read_text(encoding="utf-8"))
            return self._load_fixture(data)
        if isinstance(fixture, RawPageLayout):
            return [fixture]
        if isinstance(fixture, dict):
            if "pages" in fixture:
                return [RawPageLayout.from_dict(p) for p in fixture["pages"]]
            return [RawPageLayout.from_dict(fixture)]
        if isinstance(fixture, (list, tuple)):
            out: list[RawPageLayout] = []
            for item in fixture:
                if isinstance(item, RawPageLayout):
                    out.append(item)
                elif isinstance(item, dict):
                    out.append(RawPageLayout.from_dict(item))
                else:
                    raise TypeError(f"unsupported fixture item {type(item)!r}")
            return out
        raise TypeError(f"unsupported fixture type {type(fixture)!r}")

    # -- analysis -----------------------------------------------------------

    @property
    def page_layouts(self) -> list[RawPageLayout]:
        """The recorded layouts, unperturbed.

        Used by replay paths that need the fixture as recorded rather than as
        perturbed -- e.g. ``DocumentParser.parse_recording``.
        """
        return list(self._layouts)

    def analyze_page(self, page: PageSource) -> RawPageLayout:
        index = page.page_index
        layout = self._layout_for(index)
        return self._perturb(layout, page)

    def analyze(self, pages: Sequence[PageSource]) -> list[RawPageLayout]:
        return [self.analyze_page(p) for p in pages]

    def _layout_for(self, index: int) -> RawPageLayout:
        for layout in self._layouts:
            if layout.page_index == index:
                return layout
        # Fall back to positional lookup (fixtures are often 0-indexed while
        # PDF page numbers start at 1).
        if 0 <= index < len(self._layouts):
            return self._layouts[index]
        return RawPageLayout(page_index=index, analyzer=self.name, warnings=["no fixture for this page"])

    def _perturb(self, layout: RawPageLayout, page: PageSource) -> RawPageLayout:
        rng = random.Random(f"{self.seed}:{layout.page_index}")
        regions: list[RawRegion] = []
        for region in layout.regions:
            bbox = region.bbox
            cells = region.cells

            if self.drop_cell_probability > 0 and cells:
                kept = [c for c in cells if rng.random() > self.drop_cell_probability]
                cells = kept or cells

            if self.jitter_px > 0:
                bbox = _jitter(bbox, self.jitter_px, rng)
                cells = [_jitter_cell(c, self.jitter_px, rng) for c in cells]

            regions.append(
                RawRegion(
                    label=region.label,
                    bbox=bbox,
                    text=region.text,
                    caption=region.caption,
                    cells=cells,
                    units_note=region.units_note,
                    font_size=region.font_size,
                    is_header=region.is_header,
                    chart=region.chart,
                    confidence=region.confidence or 0.9,
                    raw=region.raw,
                )
            )

        if self.shuffle_regions:
            rng.shuffle(regions)

        return RawPageLayout(
            page_index=layout.page_index,
            width=page.width or layout.width,
            height=page.height or layout.height,
            regions=regions,
            analyzer=self.name,
            warnings=list(layout.warnings),
        )


def _jitter(bbox: BBox | None, amount: float, rng: random.Random) -> BBox | None:
    if bbox is None:
        return None
    return BBox(
        bbox.x0 + rng.uniform(-amount, amount),
        bbox.y0 + rng.uniform(-amount, amount),
        bbox.x1 + rng.uniform(-amount, amount),
        bbox.y1 + rng.uniform(-amount, amount),
    )


def _jitter_cell(cell: RawCell, amount: float, rng: random.Random) -> RawCell:
    bbox = _jitter(cell.bbox, amount, rng)
    return RawCell(
        text=cell.text,
        bbox=bbox,
        row=-1,
        col=-1,
        is_header=cell.is_header,
        raw=cell.raw,
    )
