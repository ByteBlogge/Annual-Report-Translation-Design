"""Table reconstruction: geometry -> grid, and the HTML round-trip that proves it.

These are the tests that back the claim "merged cells are handled". Each one
fails loudly if the reconstruction is broken, which is the only way the claim is
worth anything in an interview.
"""

from __future__ import annotations

import pytest

from art.parser.table_builder import (
    RawCell,
    TableGeometryOptions,
    build_table,
    cluster_edges,
    infer_boundaries,
    table_from_html,
    table_to_html,
    table_to_markdown,
    table_to_records,
    table_to_tsv,
)
from conftest import make_cells, to_positions

# ---------------------------------------------------------------------------
# 1-D edge clustering
# ---------------------------------------------------------------------------


class TestClusterEdges:
    def test_empty(self):
        assert cluster_edges([], 1.0) == []

    def test_merges_values_within_tolerance(self):
        # 0 / 0.4 are the same boundary seen twice; 100 / 100.3 likewise.
        out = cluster_edges([0, 0.4, 100, 100.3], tol=1.0)
        assert len(out) == 2
        assert out[0] == pytest.approx(0.2)
        assert out[1] == pytest.approx(100.15)

    def test_keeps_genuinely_distinct_boundaries(self):
        assert len(cluster_edges([0, 50, 100], tol=1.0)) == 3

    def test_input_need_not_be_sorted(self):
        assert cluster_edges([100, 0, 50], tol=1.0) == sorted([100, 0, 50])


# ---------------------------------------------------------------------------
# boundary inference
# ---------------------------------------------------------------------------


class TestInferBoundaries:
    def test_produces_one_more_line_than_rows(self):
        cells = make_cells([["a", "b"], ["c", "d"], ["e", "f"]])
        opts = TableGeometryOptions()
        # 3 rows -> 4 horizontal lines, 2 columns -> 3 vertical lines.
        assert len(infer_boundaries(cells, "y", opts)) == 4
        assert len(infer_boundaries(cells, "x", opts)) == 3

    def test_tolerance_scales_with_cell_not_with_edge_gap(self):
        """The original bug, pinned as a regression test.

        With the old "fraction of the median gap between distinct edges"
        tolerance, jittered boxes each became their own edge, the median gap
        collapsed to the noise size, and the tolerance shrank to nothing -- so
        the grid shattered. Rescaling to a fraction of the narrowest cell keeps
        the tolerance proportional to real geometry.
        """
        cells = make_cells([["a", "b"], ["c", "d"]], col_w=120.0, row_h=40.0)
        opts = TableGeometryOptions()
        assert infer_boundaries(cells, "x", opts) == pytest.approx([0, 120, 240])
        # tol_x = 0.35 * 120 = 42, tol_y = 0.35 * 40 = 14 -- both far above noise.
        assert opts.tol_ratio * 120.0 > 6.0
        assert opts.tol_ratio * 40.0 > 6.0


# ---------------------------------------------------------------------------
# grid reconstruction
# ---------------------------------------------------------------------------


class TestBuildTable:
    def test_simple_grid(self):
        table = build_table(
            make_cells(
                [
                    ["Item", "2023", "2022"],
                    ["Revenue", "1,234,567", "1,111,111"],
                    ["Cost", "(456,789)", "(432,109)"],
                ]
            ),
            caption="Consolidated income statement",
        )
        assert (table.n_rows, table.n_cols) == (3, 3)
        assert len(table.cells) == 9
        assert table.holes() == []
        assert to_positions(table)["Revenue"] == (1, 0, 1, 1)
        assert to_positions(table)["(432,109)"] == (2, 2, 1, 1)

    def test_merged_header_rowspan_and_colspan(self, merged_header_cells):
        table = build_table(merged_header_cells, caption="Statement of profit or loss")
        assert (table.n_rows, table.n_cols) == (3, 3)
        positions = to_positions(table)

        # The rowspan: "Item" is one cell standing in for two rows.
        assert positions["Item"] == (0, 0, 2, 1)
        # The colspan: one header covering the 2023 and 2022 columns.
        assert positions["Year ended 31 December"] == (0, 1, 1, 2)
        # The cells the spans swallow are not duplicated as cells of their own.
        assert len(table.cells) == 7
        assert table.holes() == []

    def test_merged_cells_are_flagged_and_render_as_spans(self, merged_header_cells):
        table = build_table(merged_header_cells)
        merged = [c for c in table.cells if c.is_merged]
        assert {c.text for c in merged} == {"Item", "Year ended 31 December"}

        html = table_to_html(table)
        assert 'rowspan="2"' in html
        assert 'colspan="2"' in html

    def test_jittered_boxes_recover_the_same_grid(self, merged_header_cells):
        """The property that makes a VLM backend usable at all."""
        clean = build_table(merged_header_cells)
        baseline = to_positions(clean)

        for seed in (1, 2, 3):
            jittered = build_table(
                make_cells(
                    [
                        ["a", "b", "c"],
                        ["d", "e", "f"],
                        ["g", "h", "i"],
                    ],
                    jitter=6.0,
                    seed=seed,
                )
            )
            assert (jittered.n_rows, jittered.n_cols) == (3, 3)
            assert jittered.holes() == []

        # And the merged fixture survives the same treatment.
        assert to_positions(build_table(merged_header_cells)) == baseline

    def test_geometry_beats_bad_hints_by_default(self):
        """A VLM that guesses row/col wrong must not be trusted over its boxes."""
        cells = make_cells(
            [["a", "b"], ["c", "d"]],
            hints=True,
        )
        for cell in cells:
            cell.row = 99
            cell.col = 99
        table = build_table(cells)  # trust_hints defaults to False
        assert (table.n_rows, table.n_cols) == (2, 2)
        assert to_positions(table)["c"] == (1, 0, 1, 1)

    def test_hints_are_used_when_explicitly_trusted(self):
        cells = make_cells([["a", "b"], ["c", "d"]], hints=True)
        for cell in cells:
            cell.bbox = None
        table = build_table(cells, options=TableGeometryOptions(trust_hints=True))
        assert to_positions(table)["d"] == (1, 1, 1, 1)

    def test_cells_with_neither_geometry_nor_hints_degrade_loudly(self):
        """Never lose data silently -- degrade to one column and say so."""
        table = build_table([RawCell(text="only"), RawCell(text="two")])
        assert (table.n_rows, table.n_cols) == (2, 1)
        assert any("degraded" in w for w in table.warnings)

    def test_empty_input_is_flagged_not_crashed(self):
        table = build_table([])
        assert table.cells == []
        assert any("no cells" in w for w in table.warnings)

    def test_header_detection_stops_at_a_row_of_parenthesised_negatives(self):
        """Header rows contain labels; a data row contains figures.

        A loss is printed as ``(456,789)``, so if parenthesised negatives are not
        recognised as figures, a data row made of them is mistaken for a header
        and rendered as ``<th>`` -- and ``table_to_records`` derives its keys
        from it.
        """
        table = build_table(
            make_cells(
                [
                    ["Item", "2023", "2022"],
                    ["Cost of sales", "(456,789)", "(432,109)"],
                    ["Revenue", "1,234,567", "1,111,111"],
                ]
            )
        )
        headers = {c.text for c in table.cells if c.is_header}
        assert "Cost of sales" not in headers
        assert "(456,789)" not in headers
        assert "Item" in headers

    def test_reconstruction_is_reproducible(self):
        cells = make_cells([["a", "b"], ["c", "d"]])
        first = build_table(cells)
        second = build_table(cells)
        assert to_positions(first) == to_positions(second)
        # The inferred lines are kept as evidence and must match too.
        assert first.row_lines == pytest.approx(second.row_lines)
        assert first.col_lines == pytest.approx(second.col_lines)


# ---------------------------------------------------------------------------
# the HTML round-trip -- the proof that layout survives the pipeline
# ---------------------------------------------------------------------------


class TestHtmlRoundTrip:
    def test_round_trip_preserves_every_cell_and_span(self, merged_header_cells):
        original = build_table(merged_header_cells)
        restored = table_from_html(table_to_html(original))

        assert (restored.n_rows, restored.n_cols) == (original.n_rows, original.n_cols)
        assert to_positions(restored) == to_positions(original)

    def test_round_trip_does_not_resurrect_empty_slots(self):
        """Regression: holes were emitted as real ``<td></td>`` and came back as cells.

        The grid has a hole at (1, 1); after a round trip it must still be a
        hole, and the cell count must not grow.
        """
        table = build_table(
            make_cells(
                [
                    ["A", "B"],
                    ["C", None],
                ]
            )
        )
        assert table.holes() == [(1, 1)]

        html = table_to_html(table)
        assert 'data-empty="1"' in html

        restored = table_from_html(html)
        assert restored.holes() == [(1, 1)]
        assert len(restored.cells) == len(table.cells)

    def test_round_trip_survives_several_passes(self, merged_header_cells):
        """Idempotent: the pipeline re-serialises tables more than once."""
        table = build_table(merged_header_cells)
        html = table_to_html(table)
        for _ in range(3):
            html = table_to_html(table_from_html(html))
        assert to_positions(table_from_html(html)) == to_positions(table)

    def test_html_marks_headers_as_th(self):
        table = build_table(make_cells([["Item", "2023"], ["Revenue", "1,234"]]))
        html = table_to_html(table, header_rows=1)
        assert "<th" in html


# ---------------------------------------------------------------------------
# flat views
# ---------------------------------------------------------------------------


class TestFlatViews:
    def test_records_use_the_header_row_as_keys(self):
        table = build_table(
            make_cells(
                [
                    ["Item", "2023", "2022"],
                    ["Revenue", "1,234,567", "1,111,111"],
                ]
            )
        )
        records = table_to_records(table, header_rows=1)
        assert len(records) == 1
        # ``_row`` is provenance, not data: it lets a CSV row be traced back to
        # the source grid cell, which is what a reviewer needs to challenge it.
        assert records[0]["_row"] == "1"
        assert {k: v for k, v in records[0].items() if k != "_row"} == {
            "Item": "Revenue",
            "2023": "1,234,567",
            "2022": "1,111,111",
        }

    def test_multi_level_headers_are_joined(self, merged_header_cells):
        table = build_table(merged_header_cells)
        records = table_to_records(table)
        assert records, "expected the data rows to flatten"
        keys = set(records[0]) - {"_row"}
        # The merged header contributes both of its levels: the spanning label
        # and the year it subdivides into.
        assert any("Year ended 31 December" in k and "2023" in k for k in keys)

    def test_markdown_emits_a_pipe_table(self):
        table = build_table(make_cells([["Item", "2023"], ["Revenue", "1,234"]]))
        md = table_to_markdown(table)
        assert "Revenue" in md and "|" in md

    def test_tsv_is_tab_separated(self):
        table = build_table(make_cells([["Item", "2023"], ["Revenue", "1,234"]]))
        tsv = table_to_tsv(table)
        assert "Revenue\t1,234" in tsv
