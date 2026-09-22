"""The number guard: proof that a figure cannot change under translation.

The interesting cases are all *negative space* -- things that a naive
``str(n) in text`` check would wave through. Each test here names one of them.
"""

from __future__ import annotations

import pytest

from art.parser.table_builder import build_table
from art.translator.number_guard import (
    COUNT,
    MONEY,
    RATIO,
    NumberDiffReport,
    NumberGuard,
    cn_to_int,
    extract_numbers,
    fingerprint_numbers,
    parse_unit,
    unit_from_note,
)
from conftest import make_cells

# ---------------------------------------------------------------------------
# Chinese numerals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一", 1),
        ("三", 3),
        ("十", 10),
        ("十二", 12),
        ("一百", 100),
        ("一千二百三十四", 1234),
        ("一亿", 100_000_000),
        ("两", 2),
    ],
)
def test_cn_to_int(text, expected):
    assert cn_to_int(text) == expected


def test_cn_to_int_rejects_non_numerals():
    assert cn_to_int("abc") is None
    assert cn_to_int("") is None


# ---------------------------------------------------------------------------
# unit parsing
# ---------------------------------------------------------------------------


class TestUnitParsing:
    @pytest.mark.parametrize(
        ("note", "dimension", "multiplier"),
        [
            ("单位：人民币千元", MONEY, 1e3),
            ("单位：人民币百万元", MONEY, 1e6),
            ("单位：人民币元", MONEY, 1.0),
            ("RMB'000", MONEY, 1e3),
            ("RMB million", MONEY, 1e6),
            ("in thousands", MONEY, 1e3),
        ],
    )
    def test_unit_from_note(self, note, dimension, multiplier):
        parsed = unit_from_note(note)
        assert parsed is not None
        assert parsed[0] == dimension
        assert parsed[1] == multiplier

    def test_unit_from_note_rejects_empty(self):
        assert unit_from_note("") is None

    def test_parse_unit_longest_match_wins(self):
        # "百万元" must not be read as "万元" -- a 100x error.
        assert parse_unit("百万元")[1] == 1e6
        assert parse_unit("万元")[1] == 1e4


# ---------------------------------------------------------------------------
# extraction and normalisation
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_plain_figure(self):
        found = extract_numbers("Revenue of 1,234,567 was recorded.")
        assert len(found) == 1
        assert found[0].value == 1_234_567
        assert found[0].digits == "1234567"

    def test_parenthesised_figure_is_negative(self):
        found = extract_numbers("Net loss (456,789)")
        assert found[0].value == -456_789

    def test_percentage_is_a_ratio(self):
        found = extract_numbers("grew 12.4%")
        assert found[0].dimension == RATIO
        assert found[0].base_value == pytest.approx(0.124)

    def test_scale_word_gives_money_with_multiplier(self):
        found = extract_numbers("5 million")
        assert found[0].dimension == MONEY
        assert found[0].base_value == pytest.approx(5e6)

    def test_chinese_scale_unit(self):
        found = extract_numbers("1,234 千元")
        assert found[0].dimension == MONEY
        assert found[0].base_value == pytest.approx(1_234_000)

    def test_table_unit_note_is_inherited_and_flagged(self):
        """The mechanism that catches a dropped unit header."""
        found = extract_numbers("1,234", unit_hint="单位：人民币千元")
        assert found[0].multiplier == 1e3
        assert found[0].unit_inherited is True

    def test_thousands_scale_marker_is_not_a_figure(self):
        """"RMB'000" contains digits that are a unit, not a figure.

        Without the apostrophe rule this extracts a phantom value of 0 (or
        000), and the report fills with false findings on every statement.
        """
        found = extract_numbers("RMB'000 1,234")
        assert len(found) == 1
        assert found[0].value == 1_234

    def test_chinese_numeral_merging_into_a_unit_is_not_a_figure(self):
        """"百万元" contains 百万, which must not be read as 1,000,000."""
        found = extract_numbers("合计 1.234 百万元")
        values = sorted(o.value for o in found)
        assert 1_000_000 not in values
        assert 1.234 in values

    def test_chinese_numeral_with_currency_is_extracted(self):
        found = extract_numbers("注册资本为人民币一亿元")
        assert any(o.base_value == pytest.approx(1e8) for o in found)

    def test_list_markers_are_not_figures(self):
        """一、二、 are section markers. Extracting them is pure noise."""
        assert extract_numbers("一、经营回顾") == []

    def test_count_dimension(self):
        found = extract_numbers("3,000 股")
        assert found[0].dimension == COUNT

    def test_unlabelled_decimal_is_treated_as_ratio(self):
        found = extract_numbers("earnings per share of 1.23")
        assert found[0].dimension == RATIO

    def test_fingerprint_is_order_insensitive(self):
        assert fingerprint_numbers("1,234 then 5,678") == fingerprint_numbers("5,678 then 1,234")


# ---------------------------------------------------------------------------
# the comparison itself
# ---------------------------------------------------------------------------


class TestCheckText:
    def setup_method(self):
        self.guard = NumberGuard()

    def test_identical_text_passes(self):
        report = self.guard.check_text("Revenue 1,234,567", "营业收入 1,234,567")
        assert report.ok
        assert report.drift == 0.0
        assert report.matched == 1

    def test_unit_conversion_is_not_a_difference(self):
        """1,234 千元 and 1.234 百万元 are the same amount."""
        report = self.guard.check_text("1,234 千元", "1.234 百万元")
        assert report.ok, report.summary()
        assert report.matched == 1

    def test_digit_slip_is_a_mismatch_not_missing_plus_added(self):
        report = self.guard.check_text("1,234,567", "1,274,567")
        assert not report.ok
        assert report.mismatched
        assert report.missing == [] and report.added == []
        source, target, reason = report.mismatched[0]
        assert source.raw == "1,234,567"
        assert target.raw == "1,274,567"
        assert "digit slip" in reason

    def test_large_single_digit_change_is_still_caught(self):
        """Regression: pairing by *value* distance misses this entirely.

        One changed digit moves the value by 57%, so a 2%-value test would
        report the two numbers as unrelated -- and, worse, would let them sit in
        the missing/added buckets where nobody looks.
        """
        report = self.guard.check_text("1,234,567", "1,934,567")
        assert report.mismatched, report.summary()
        assert "digit slip" in report.mismatched[0][2]

    def test_missing_figure(self):
        report = self.guard.check_text("Revenue 1,234 and cost 5,678", "Revenue 1,234")
        assert len(report.missing) == 1
        assert report.missing[0].raw == "5,678"
        assert report.drift == 0.5

    def test_added_figure_is_a_finding_but_not_a_failure(self):
        """A figure in the target only is reported, but does not fail ``ok``.

        This asymmetry is deliberate and worth stating, because the naive
        reading ("an extra figure is obviously a hallucination") is wrong often
        enough to be harmful. Rendering "Year ended 31 December" as
        ``截至12月31日止年度`` *adds* a figure -- the month -- that the source
        never printed. If additions failed the report, every document containing
        a date would be flagged and the real findings would be lost in the noise.

        So additions are recorded (``has_findings``, and a low-weight risk
        factor) while ``ok`` and ``drift`` stay source-centric: drift measures
        the fraction of *source* figures that did not survive, and an addition
        cannot make a source figure fail to survive.
        """
        report = self.guard.check_text("Revenue 1,234", "Revenue 1,234 and 5,678")
        assert len(report.added) == 1
        assert report.added[0].raw == "5,678"
        assert report.has_findings is True
        assert report.ok is True
        assert report.drift == 0.0

    def test_dated_translation_adds_a_benign_figure(self):
        """The concrete benign case, pinned so nobody 'fixes' it into a failure."""
        report = self.guard.check_text("Year ended 31 December", "截至12月31日止年度")
        assert [o.raw for o in report.added] == ["12"]
        assert report.ok is True, "a reformatted date must not flag a document"

    def test_dimension_change_is_flagged(self):
        """"100 shares" becoming "100 million" must never pass silently."""
        report = self.guard.check_text("100 股", "100 百万元")
        assert not report.ok

    def test_dropped_unit_note_is_structural_even_when_digits_match(self):
        """The 1000x error that looks like a perfect match.

        Every figure agrees; the only difference is that the target lost the
        header saying the figures are in thousands. A digit-level guard sees
        nothing. This is why the unit note is threaded through extraction.
        """
        report = self.guard.check_text(
            "1,234,567",
            "1,234,567",
            unit_hint="单位：人民币千元",
        )
        assert report.structural, "a dropped unit note must be a structural finding"
        assert not report.ok

    def test_unit_note_preserved_is_fine(self):
        report = self.guard.check_text(
            "1,234,567 千元",
            "1,234,567 千元",
            unit_hint="单位：人民币千元",
        )
        assert report.ok, report.summary()

    def test_drift_is_bounded(self):
        report = NumberDiffReport()
        assert report.drift == 0.0
        report.source_total = 2
        report.missing = extract_numbers("1,234")
        report.mismatched = [(extract_numbers("5,678")[0], extract_numbers("5,679")[0], "x")]
        assert report.drift == 1.0


# ---------------------------------------------------------------------------
# table-vs-table, positionally
# ---------------------------------------------------------------------------


class TestCheckTablePair:
    def setup_method(self):
        self.guard = NumberGuard()
        self.raw = make_cells(
            [
                ["Item", "2023", "2022"],
                ["Revenue", "1,234,567", "1,111,111"],
                ["Cost of sales", "(456,789)", "(432,109)"],
            ]
        )
        self.source = build_table(self.raw, units_note="单位：人民币千元")

    @staticmethod
    def _copy(table):
        """A structurally identical table with fresh (mutable) cells."""
        from art.schema import TableBlock, TableCell

        return TableBlock(
            cells=[TableCell.from_dict(c.to_dict()) for c in table.cells],
            n_rows=table.n_rows,
            n_cols=table.n_cols,
            units_note=table.units_note,
        )

    def test_identical_tables_pass(self):
        report = self.guard.check_table_pair(self.source, self._copy(self.source))
        assert report.ok, report.summary()

    def test_target_with_one_changed_figure_is_located_by_position(self):
        target = self._copy(self.source)
        # Corrupt exactly one cell: the 2023 revenue.
        for cell in target.cells:
            if cell.text == "1,234,567":
                cell.text = "1,274,567"

        report = self.guard.check_table_pair(self.source, target)
        assert len(report.mismatched) == 1
        _, _, reason = report.mismatched[0]
        assert "(1, 1)" in reason, f"the finding must name the offending cell, got {reason!r}"

    def test_shape_change_is_structural(self):
        from art.schema import TableBlock, TableCell

        target = TableBlock(
            cells=[TableCell(text="1", row=0, col=0)], n_rows=1, n_cols=1,
            units_note=self.source.units_note,
        )
        report = self.guard.check_table_pair(self.source, target)
        assert any("shape changed" in s for s in report.structural)

    def test_dropped_units_note_is_structural(self):
        target = self._copy(self.source)
        target.units_note = ""
        report = self.guard.check_table_pair(self.source, target)
        assert any("unit note" in s for s in report.structural)
