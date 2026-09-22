"""Shared fixtures and table-synthesis helpers.

The helpers here exist for one reason: most of the interesting behaviour in this
project is *geometric*, so the tests need to state a table as a grid of
rectangles rather than as a finished structure. A test that says "these boxes are
one row apart" is checking the reconstruction; a test that hands in a
prettified ``TableBlock`` would only be checking the dataclass.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from art.schema import BBox

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_FIXTURE = REPO_ROOT / "examples" / "demo_annual_report.json"


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def box(x0: float, y0: float, x1: float, y1: float) -> BBox:
    return BBox(x0, y0, x1, y1)


def make_cells(
    matrix,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    col_w: float = 120.0,
    row_h: float = 40.0,
    jitter: float = 0.0,
    seed: int = 0,
    hints: bool = False,
):
    """Turn a text matrix into ``RawCell``s laid out on a regular grid.

    ``None`` entries are skipped, which is how a *hole* (a slot no box covers)
    is expressed. ``jitter`` displaces every edge independently, simulating a
    VLM whose boxes are a few pixels off -- the exact input that used to shatter
    the grid before the tolerance was rescaled to the cell rather than the edge.
    """
    from art.parser.table_builder import RawCell

    rng = random.Random(seed)
    cells = []
    for r, row in enumerate(matrix):
        for c, text in enumerate(row):
            if text is None:
                continue
            bx0 = x0 + c * col_w
            by0 = y0 + r * row_h
            bx1 = bx0 + col_w
            by1 = by0 + row_h
            if jitter:
                bx0 += rng.uniform(-jitter, jitter)
                by0 += rng.uniform(-jitter, jitter)
                bx1 += rng.uniform(-jitter, jitter)
                by1 += rng.uniform(-jitter, jitter)
            cells.append(
                RawCell(
                    text=text,
                    bbox=box(bx0, by0, bx1, by1),
                    row=r if hints else -1,
                    col=c if hints else -1,
                    is_header=(r == 0),
                )
            )
    return cells


def to_positions(table):
    """``{text: (row, col, row_span, col_span)}`` -- the reconstruction, grep-able.

    Comparing this dict between two runs is how the jitter tests assert "exact
    same grid" without depending on cell ordering.
    """
    return {
        cell.text: (cell.row, cell.col, cell.row_span, cell.col_span) for cell in table.cells
    }


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def demo_fixture_path() -> Path:
    assert DEMO_FIXTURE.exists(), f"missing fixture: {DEMO_FIXTURE}"
    return DEMO_FIXTURE


@pytest.fixture(scope="session")
def demo_fixture(demo_fixture_path: Path) -> dict:
    return json.loads(demo_fixture_path.read_text(encoding="utf-8"))


@pytest.fixture()
def demo_document(demo_fixture_path: Path):
    """The parsed SDO for the recorded 4-page demo report."""
    from art.parser.analyzer import PageSource, build_document
    from art.parser.mock_analyzer import MockLayoutAnalyzer

    analyzer = MockLayoutAnalyzer(fixture=str(demo_fixture_path))
    layouts = [
        analyzer.analyze_page(PageSource(page_index=p.page_index)) for p in analyzer.page_layouts
    ]
    return build_document(
        layouts, doc_id="demo-annual-report", source=str(demo_fixture_path)
    )


@pytest.fixture()
def demo_chunking(demo_document):
    from art.chunker.pipeline import ChunkingPipeline

    return ChunkingPipeline().run(demo_document)


# ---------------------------------------------------------------------------
# a merged-header table stated purely as rectangles
# ---------------------------------------------------------------------------


@pytest.fixture()
def merged_header_cells():
    """A statement header with both a rowspan and a colspan.

        +--------+---------------------------+
        |        | Year ended 31 December    |   <- colspan 2
        | Item   +-------------+-------------+   <- rowspan 2 on "Item"
        |        | 2023        | 2022        |
        +--------+-------------+-------------+
        | Revenue| 1,234,567   | 1,111,111   |
        +--------+-------------+-------------+
    """
    from art.parser.table_builder import RawCell

    cw, rh = 120.0, 40.0

    def cell(text, c0, c1, r0, r1, header=False):
        return RawCell(
            text=text,
            bbox=box(c0 * cw, r0 * rh, (c1 + 1) * cw, (r1 + 1) * rh),
            is_header=header,
        )

    return [
        cell("Item", 0, 0, 0, 1, header=True),
        cell("Year ended 31 December", 1, 2, 0, 0, header=True),
        cell("2023", 1, 1, 1, 1, header=True),
        cell("2022", 2, 2, 1, 1, header=True),
        cell("Revenue", 0, 0, 2, 2),
        cell("1,234,567", 1, 1, 2, 2),
        cell("1,111,111", 2, 2, 2, 2),
    ]
