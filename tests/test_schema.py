"""The Structured Document Object -- the contract every stage speaks.

If this contract is stable and losslessly serialisable, then a layout bug can
only ever be a *parsing* bug, never a lossy-hand-off bug. That is what these
tests defend.
"""

from __future__ import annotations

import json

import pytest

from art.parser.table_builder import build_table
from art.schema import (
    BBox,
    ChartBlock,
    DocumentPage,
    ImageBlock,
    StructuredDocument,
    TableBlock,
    TableCell,
    TextBlock,
)
from art.textutils import (
    cjk_ratio,
    escape_html,
    estimate_tokens,
    looks_numeric,
    normalize_whitespace,
    normalize_width,
    slugify,
    truncate,
)
from conftest import make_cells

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


class TestBBox:
    def test_derived_measurements(self):
        b = BBox(10, 20, 40, 60)
        assert b.width == 30
        assert b.height == 40
        assert b.area == 1200
        assert b.center_x == 25 and b.center_y == 40

    def test_union_and_intersection(self):
        a, b = BBox(0, 0, 10, 10), BBox(5, 5, 20, 20)
        assert a.union(b) == BBox(0, 0, 20, 20)
        assert a.intersection_area(b) == 25

    def test_iou(self):
        a, b = BBox(0, 0, 10, 10), BBox(5, 5, 20, 20)
        # inter 25, union 100 + 225 - 25 = 300
        assert a.iou(b) == pytest.approx(25 / 300)

    def test_from_any_parses_the_shapes_a_vlm_returns(self):
        assert BBox.from_any([1, 2, 3, 4]) == BBox(1, 2, 3, 4)
        assert BBox.from_any({"x0": 1, "y0": 2, "x1": 3, "y1": 4}) == BBox(1, 2, 3, 4)
        assert BBox.from_any({"x": 1, "y": 2, "w": 2, "h": 2}) == BBox(1, 2, 3, 4)

    def test_round_trip(self):
        assert BBox.from_dict(BBox(1, 2, 3, 4).to_dict()) == BBox(1, 2, 3, 4)


# ---------------------------------------------------------------------------
# cells and grids
# ---------------------------------------------------------------------------


class TestTableGrid:
    def test_covered_slots_of_a_span(self):
        cell = TableCell(text="hdr", row=0, col=1, row_span=2, col_span=2)
        assert set(cell.covered_slots()) == {(0, 1), (0, 2), (1, 1), (1, 2)}
        assert cell.is_merged is True
        assert cell.span_area == 4

    def test_grid_answers_which_cell_covers_a_slot(self):
        table = TableBlock(
            cells=[
                TableCell(text="hdr", row=0, col=0, col_span=2),
                TableCell(text="a", row=1, col=0),
                TableCell(text="b", row=1, col=1),
            ],
            n_rows=2,
            n_cols=2,
        )
        grid = table.grid()
        # The spanning header covers both columns of row 0 ...
        assert grid[0][0].text == "hdr" and grid[0][1].text == "hdr"
        # ... but only its origin slot becomes an element.
        assert table.is_origin(0, 0) is True
        assert table.is_origin(0, 1) is False

    def test_holes_are_visible(self):
        table = build_table(make_cells([["A", "B"], ["C", None]]))
        assert table.holes() == [(1, 1)]

    def test_validate_reports_an_overfull_grid(self):
        table = TableBlock(
            cells=[TableCell(text="x", row=5, col=5)], n_rows=2, n_cols=2
        )
        assert table.validate(), "a cell outside the declared grid must be reported"


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


class FancyDocument:
    """A tiny SDO exercising every block type and a merged table."""

    @staticmethod
    def build() -> StructuredDocument:
        table = build_table(
            make_cells([["Item", "2023"], ["Revenue", "1,234,567"]]),
            caption="Income statement",
            units_note="单位：人民币千元",
        )
        return StructuredDocument(
            doc_id="unit-test",
            source="synthetic",
            pages=[
                DocumentPage(
                    index=1,
                    width=595,
                    height=842,
                    analyzer="mock",
                    blocks=[
                        TextBlock(text="Annual Report 2023", heading_level=1),
                        TextBlock(text="Some prose with a figure 1,234."),
                        table,
                        ChartBlock(
                            caption="Revenue by segment",
                            chart_type="bar",
                            series=[{"name": "Revenue", "points": {"2023": 1234.5}}],
                            axis_labels={"y": ["千元"]},
                            description="Revenue was 1,234.5 千元 in 2023.",
                            extraction_method="vlm",
                            confidence=0.9,
                        ),
                        ImageBlock(caption="Company logo"),
                    ],
                    warnings=["page-level warning"],
                )
            ],
            metadata={"analyst": "unit-test"},
            warnings=["document-level warning"],
        )


class TestStructuredDocument:
    def test_iter_blocks_yields_page_index_and_block(self):
        doc = FancyDocument.build()
        pairs = list(doc.iter_blocks())
        assert all(page == 1 for page, _ in pairs)
        assert len(pairs) == 5

    def test_typed_accessors(self):
        doc = FancyDocument.build()
        assert len(doc.tables()) == 1
        assert len(doc.charts()) == 1
        assert len(doc.texts()) == 2

    def test_stats_count_everything(self):
        stats = FancyDocument.build().stats()
        assert stats["pages"] == 1
        assert stats["table_blocks"] == 1
        assert stats["chart_blocks"] == 1
        assert stats["text_blocks"] == 2
        # 2 for the document + 1 for the page.
        assert stats["warnings"] == 2

    def test_json_round_trip_is_lossless(self):
        """The whole point of a typed contract: nothing drifts on the way through."""
        original = FancyDocument.build()
        restored = StructuredDocument.from_dict(json.loads(original.to_json()))

        assert restored.doc_id == original.doc_id
        assert restored.metadata == original.metadata

        orig_table = original.tables()[0]
        new_table = restored.tables()[0]
        assert (new_table.n_rows, new_table.n_cols) == (orig_table.n_rows, orig_table.n_cols)
        assert new_table.units_note == orig_table.units_note
        assert {c.text: (c.row, c.col, c.row_span, c.col_span) for c in new_table.cells} == {
            c.text: (c.row, c.col, c.row_span, c.col_span) for c in orig_table.cells
        }

        orig_chart = original.charts()[0]
        new_chart = restored.charts()[0]
        assert new_chart.series == orig_chart.series
        assert new_chart.points == orig_chart.points

    def test_save_and_load(self, tmp_path):
        original = FancyDocument.build()
        path = original.save(tmp_path / "nested" / "doc.json")
        assert path.exists()
        restored = StructuredDocument.load(path)
        assert restored.to_dict() == original.to_dict()

    def test_block_kinds_survive_serialisation(self):
        """``block_from_dict`` must dispatch on ``kind``, not guess by keys."""
        doc = FancyDocument.build()
        restored = StructuredDocument.from_dict(json.loads(doc.to_json()))
        kinds = [type(b).__name__ for b in restored.pages[0].blocks]
        assert kinds == ["TextBlock", "TextBlock", "TableBlock", "ChartBlock", "ImageBlock"]


class TestChartPoints:
    def test_flattens_name_points_shape(self):
        chart = ChartBlock(series=[{"name": "Revenue", "points": {"2023": 1.5, "2022": 1.2}}])
        assert chart.points == [("Revenue/2023", 1.5), ("Revenue/2022", 1.2)]

    def test_flattens_label_value_shape(self):
        chart = ChartBlock(series=[{"label": "Cost", "value": 3.0}])
        assert chart.points == [("Cost", 3.0)]

    def test_skips_non_numeric_points(self):
        chart = ChartBlock(series=[{"name": "S", "points": {"a": 1.0, "b": "n/a"}}])
        assert chart.points == [("S/a", 1.0)]


# ---------------------------------------------------------------------------
# text utilities
# ---------------------------------------------------------------------------


class TestTextUtils:
    def test_normalize_width_folds_fullwidth_digits(self):
        """Without this, "１２３" and "123" are different numbers."""
        assert normalize_width("１２３") == "123"

    def test_cjk_ratio(self):
        assert cjk_ratio("营业收入") == pytest.approx(1.0)
        assert cjk_ratio("Revenue") == pytest.approx(0.0)
        assert 0.0 < cjk_ratio("营业收入 Revenue") < 1.0

    def test_looks_numeric(self):
        assert looks_numeric("1,234.56") is True
        assert looks_numeric("Revenue") is False
        # Regression: the suffix char class used to omit ")", so a
        # parenthesised negative -- the standard way an annual report prints a
        # loss -- was not recognised as a figure, contradicting the docstring.
        assert looks_numeric("(456)") is True
        assert looks_numeric("(456,789)") is True
        assert looks_numeric("  (1,234.5) ") is True

    def test_looks_numeric_treats_nil_markers_as_numeric(self):
        for marker in ("-", "—", "N/A", "不适用", "无"):
            assert looks_numeric(marker) is True, marker

    def test_escape_html_closes_xss_and_markup(self):
        assert escape_html("<script>") == "&lt;script&gt;"

    def test_slugify_keeps_cjk_and_dashes_spaces(self):
        slug = slugify("第一章 财务摘要")
        assert "第一章" in slug
        assert " " not in slug

    def test_truncate_keeps_total_length_within_limit(self):
        # ``limit`` is the total width including the suffix.
        out = truncate("abcdefghij", limit=5)
        assert out == "ab..."
        assert len(out) == 5
        assert truncate("ab", limit=5) == "ab"

    def test_estimate_tokens_is_monotonic(self):
        assert estimate_tokens("x" * 400) > estimate_tokens("x" * 40)

    def test_normalize_whitespace_can_flatten_newlines(self):
        assert normalize_whitespace("a\n\n b", keep_newlines=False) == "a b"
        # Regression: flattening replaced "\n" with " " without re-collapsing,
        # producing "a   b" -- three spaces that then become a table record key.
        assert normalize_whitespace("a\n\n\n\n   b", keep_newlines=False) == "a b"
        assert "\n" in normalize_whitespace("a\n\n\n b", keep_newlines=True)
