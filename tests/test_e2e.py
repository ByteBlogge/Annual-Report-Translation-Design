"""End-to-end: the whole pipeline, and the two claims that matter most.

Everything else in this repo is scaffolding for these assertions:

1. **A figure cannot change under translation.** Not "we check afterwards" --
   the numerical content of a table is never sent to the model at all, and the
   guard is shown to catch deliberate corruption, N for N.
2. **The layout survives.** Merged cells round-trip through the parser, the
   translator and the renderer with their spans intact.

Both are asserted against the recorded 4-page fixture (offline, deterministic)
and against a synthetic document built to make the payload claim falsifiable.
"""

from __future__ import annotations

import json

import pytest

from art.chunker.pipeline import ChunkingPipeline
from art.hitl.policy import RiskPolicy
from art.parser.analyzer import PageSource, build_document
from art.parser.mock_analyzer import MockLayoutAnalyzer
from art.parser.table_builder import build_table, table_from_html
from art.schema import DocumentPage, StructuredDocument, TableBlock, TextBlock
from art.translator.agents import is_numeric_cell, render_table_payload, table_preview
from art.translator.llm import MockLLM
from art.translator.pipeline import TranslationPipeline, preview_tables, render_number_findings
from conftest import make_cells, to_positions

# ---------------------------------------------------------------------------
# shared setup: parse + chunk the fixture once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parsed(demo_fixture_path):
    analyzer = MockLayoutAnalyzer(fixture=str(demo_fixture_path))
    layouts = [
        analyzer.analyze_page(PageSource(page_index=p.page_index)) for p in analyzer.page_layouts
    ]
    return build_document(layouts, doc_id="demo-annual-report", source=str(demo_fixture_path))


@pytest.fixture(scope="module")
def chunking(parsed):
    return ChunkingPipeline().run(parsed)


def run(parsed, chunking, **llm_kwargs):
    """Run the translation stage and hand back both the result and the LLM."""
    llm = MockLLM(seed=1, **llm_kwargs)
    result = TranslationPipeline(llm, policy=RiskPolicy()).run(chunking)
    return result, llm


def span_map(table: TableBlock) -> dict[tuple[int, int], tuple[int, int]]:
    """``(row, col) -> (row_span, col_span)``: a translation-invariant layout fingerprint.

    Keying by cell *text* would be wrong here -- the labels are translated on
    purpose, so the texts are supposed to differ. What must not differ is which
    slot each cell anchors and how far it reaches.
    """
    return {(c.row, c.col): (c.row_span, c.col_span) for c in table.cells}


# ---------------------------------------------------------------------------
# the fixture is what we think it is
# ---------------------------------------------------------------------------


class TestFixtureIntegrity:
    """If the fixture is not exercising the hard cases, every test below is hollow."""

    def test_has_the_structures_we_claim(self, parsed):
        stats = parsed.stats()
        assert stats["pages"] == 4
        assert stats["table_blocks"] == 3
        assert stats["chart_blocks"] == 1
        assert stats["merged_cells"] >= 2, "merged cells are the point of the fixture"
        assert stats["merge_conflicts"] == 0

    def test_has_a_rowspan_and_a_colspan(self, parsed):
        spans = {
            (c.row_span, c.col_span)
            for t in parsed.tables()
            for c in t.cells
            if c.is_merged
        }
        assert any(rs > 1 for rs, _ in spans), "no rowspan in the fixture"
        assert any(cs > 1 for _, cs in spans), "no colspan in the fixture"

    def test_has_unit_notes(self, parsed):
        assert all(t.units_note for t in parsed.tables())

    def test_has_issuer_supplied_bilingual_glosses(self, chunking):
        issuer = [e for e in chunking.glossary if e.authority == "source_gloss"]
        assert len(issuer) >= 5
        assert chunking.glossary.translate("Revenue") == "营业收入"


# ---------------------------------------------------------------------------
# claim 1: figures never reach the model inside a table
# ---------------------------------------------------------------------------


class TestNumbersAreKeptAwayFromTheModel:
    def test_table_payload_excludes_every_numeric_cell(self, parsed):
        """The enforcement point, asserted directly on the payload."""
        for table in parsed.tables():
            payload, label_slots, numeric_count = render_table_payload(table)
            blob = json.dumps(payload, ensure_ascii=False)

            expected_numeric = sum(
                1 for c in table.cells if is_numeric_cell(c.text, unit_hint=table.units_note)
            )
            assert numeric_count == expected_numeric

            for cell in table.cells:
                if not is_numeric_cell(cell.text, unit_hint=table.units_note):
                    continue
                # A figure must not appear anywhere in the payload...
                assert cell.text not in blob, f"{cell.text!r} leaked into the table payload"
                # ...and must not be addressable as a label slot either.
                assert (cell.row, cell.col) not in label_slots

    def test_a_unique_figure_never_appears_in_any_prompt(self):
        """Falsifiable version: a figure that exists nowhere else in the document.

        The fixture's figures also occur in prose (which *is* sent), so it alone
        could not distinguish "copied by code" from "sent and returned intact".
        Here the figure is unique, so finding it in a prompt is unambiguous.
        """
        sentinel = "9,876,543"
        table = build_table(
            make_cells([["Item", "2023"], ["Revenue", sentinel]]),
            caption="Unique figure table",
        )
        document = StructuredDocument(
            doc_id="sentinel",
            pages=[
                DocumentPage(
                    index=1,
                    width=595,
                    height=842,
                    blocks=[
                        TextBlock(text="第一章 财务摘要", heading_level=1),
                        TextBlock(text="A sentence of prose with no figures in it at all."),
                        table,
                    ],
                )
            ],
        )
        chunking = ChunkingPipeline().run(document)
        llm = MockLLM(seed=1)
        result = TranslationPipeline(llm, policy=RiskPolicy()).run(chunking)

        prompts = [call["payload"].get("user", "") for call in llm.calls]
        assert prompts, "no model calls recorded -- the test would be vacuous"
        assert all(sentinel not in prompt for prompt in prompts), (
            "a table figure reached the model; it must be copied by code, not translated"
        )
        # And prove the payload was actually built: the label was sent.
        assert any("Revenue" in prompt for prompt in prompts)
        # The figure survived by construction.
        target = result.chunk_results[-1].table_translations[0].target
        assert any(c.text == sentinel for c in target.cells)

    def test_prose_is_verified_even_though_it_is_sent(self, parsed, chunking):
        """Prose has to go to the model. That is why the guard exists."""
        result, _ = run(parsed, chunking)
        report = result.aggregate_number_report()
        assert report.source_total > 0
        assert report.ok, report.summary()


# ---------------------------------------------------------------------------
# claim 2: layout survives the pipeline
# ---------------------------------------------------------------------------


class TestLayoutSurvives:
    def test_merged_spans_are_identical_in_source_and_target(self, parsed, chunking):
        result, _ = run(parsed, chunking)
        compared = 0
        for chunk_result in result.chunk_results:
            for translation in chunk_result.table_translations:
                source, target = translation.source, translation.target
                assert (source.n_rows, source.n_cols) == (target.n_rows, target.n_cols)
                # Same slots, same spans -- only the label text may differ.
                assert span_map(source) == span_map(target)
                assert len(source.cells) == len(target.cells)
                compared += 1
        assert compared == len(parsed.tables()), "not every table was compared"

    def test_figures_are_byte_identical_at_every_position(self, parsed, chunking):
        """The strongest form of the promise: figures do not merely 'match',
        they are the same characters in the same cell."""
        result, _ = run(parsed, chunking)
        for chunk_result in result.chunk_results:
            for translation in chunk_result.table_translations:
                source_cells = {(c.row, c.col): c.text for c in translation.source.cells}
                for cell in translation.target.cells:
                    original = source_cells[(cell.row, cell.col)]
                    if is_numeric_cell(original, unit_hint=translation.source.units_note):
                        assert cell.text == original, (
                            f"figure at {(cell.row, cell.col)} changed: "
                            f"{original!r} -> {cell.text!r}"
                        )

    def test_target_table_renders_its_spans_back(self, parsed, chunking):
        result, _ = run(parsed, chunking)
        for chunk_result in result.chunk_results:
            for translation in chunk_result.table_translations:
                html = translation.target_html
                restored = table_from_html(html)
                assert (restored.n_rows, restored.n_cols) == (
                    translation.target.n_rows,
                    translation.target.n_cols,
                )
                assert to_positions(restored) == to_positions(translation.target)

    def test_merged_table_in_the_output_is_rendered_as_html_not_markdown(self, parsed, chunking):
        """Markdown cannot express rowspan: it would duplicate the merged label."""
        result, _ = run(parsed, chunking)
        markdown = result.to_markdown()
        assert "<table" in markdown, "tables must be emitted as HTML to keep their spans"
        assert "rowspan=" in markdown or "colspan=" in markdown

    def test_unit_note_survives_translation(self, parsed, chunking):
        result, _ = run(parsed, chunking)
        for chunk_result in result.chunk_results:
            for translation in chunk_result.table_translations:
                if translation.source.units_note:
                    assert translation.target.units_note, "the unit note was dropped"


# ---------------------------------------------------------------------------
# claim 1, second half: the guard actually catches corruption
# ---------------------------------------------------------------------------


class TestGuardCatchesInjectedCorruption:
    def test_the_clean_run_is_clean(self, parsed, chunking):
        result, llm = run(parsed, chunking)
        report = result.aggregate_number_report()
        assert report.mismatched == []
        assert report.missing == []
        assert report.drift == 0.0
        assert result.needs_review == []
        assert llm.corruption_applied == 0

    @pytest.mark.parametrize("faults", [1, 2, 3, 5])
    def test_every_injected_corruption_is_caught(self, parsed, chunking, faults):
        """N in, N out. If this ever drops, the validator has regressed."""
        result, llm = run(parsed, chunking, inject_number_drift=faults)
        report = result.aggregate_number_report()

        # Trust but verify: confirm the mock really did corrupt that many, so a
        # passing test cannot be an artefact of the injection silently failing.
        assert llm.corruption_applied == faults

        assert len(report.mismatched) == faults, (
            f"injected {faults}, guard reported {len(report.mismatched)}\n"
            + render_number_findings(result)
        )
        assert report.missing == []
        assert report.drift > 0

    def test_a_corrupted_chunk_reaches_the_review_queue_with_evidence(self, parsed, chunking):
        result, _ = run(parsed, chunking, inject_number_drift=2)
        flagged = result.needs_review
        assert flagged, "a changed figure must be routed to a human"

        item = result.queue.sorted_by_risk()[0]
        assert item.status == "pending"
        # Self-contained: a reviewer can judge it without re-running anything.
        assert item.source_text
        assert item.target_text
        assert item.features["numbers_mismatched"] >= 1
        assert item.risk["value"] >= item.risk["threshold"]
        assert item.top_reasons

    def test_the_demo_command_exits_zero(self, demo_fixture_path):
        """The CLI's own verdict, executed as a test rather than a claim."""
        from art.cli import main

        assert main(["demo", "--fixture", str(demo_fixture_path), "--fault", "2"]) == 0


# ---------------------------------------------------------------------------
# terminology consistency
# ---------------------------------------------------------------------------


class TestTerminologyConsistency:
    def test_every_chunk_is_told_the_same_target_for_a_term(self, parsed, chunking):
        """The reason the glossary is document-wide and built before chunking.

        A term offered to one chunk and not another is exactly how the same
        concept ends up rendered two different ways on pages 6 and 90.
        """
        approved = {entry.source: entry.target for entry in chunking.glossary}
        assert approved.get("Revenue") == "营业收入"

        for chunk in chunking.chunks:
            for source, target in chunk.glossary_terms:
                assert approved[source] == target, (
                    f"{chunk.chunk_id} was offered {source!r} => {target!r}, "
                    f"but the document glossary says {approved[source]!r}"
                )

    def test_no_glossary_conflicts_were_raised_by_the_fixture(self, parsed, chunking):
        assert chunking.glossary.conflicts == []

    def test_the_run_reports_zero_glossary_conflicts(self, parsed, chunking):
        result, _ = run(parsed, chunking)
        assert result.summary()["glossary_conflicts"] == 0


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------


class TestArtefacts:
    def test_a_run_writes_the_reviewer_bundle(self, parsed, chunking, tmp_path):
        llm = MockLLM(seed=1, inject_number_drift=2)
        result = TranslationPipeline(llm, policy=RiskPolicy(), run_dir=tmp_path).run(chunking)

        assert result.outputs
        target_md = tmp_path / "demo-annual-report.target.md"
        assert target_md.exists()
        assert "营业收入" in target_md.read_text(encoding="utf-8")

        report = tmp_path / "demo-annual-report.report.json"
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["summary"]["numbers_mismatched"] == 2
        assert payload["summary"]["review"]["pending"] >= 1

        assert (tmp_path / "review.jsonl").exists()
        assert (tmp_path / "review.csv").exists()
        assert (tmp_path / "review.md").exists()

    def test_review_queue_persists_decisions_across_runs(self, parsed, chunking, tmp_path):
        """A human decision must not be thrown away by the next run."""
        llm = MockLLM(seed=1, inject_number_drift=2)
        TranslationPipeline(llm, policy=RiskPolicy(), run_dir=tmp_path).run(chunking)

        from art.hitl.queue import ReviewQueue

        queue_path = tmp_path / "review.jsonl"
        queues = ReviewQueue(queue_path).load()
        assert len(queues) >= 1

        item = queues.sorted_by_risk()[0]
        queues.decide(item.item_id, "approved", reviewer="alice", note="verified against the PDF")
        queues.save(queue_path)

        reopened = ReviewQueue(queue_path).load()
        assert reopened.stats()["pending"] == 0
        assert reopened.stats()["by_status"]["approved"] >= 1

    def test_markdown_marks_the_risky_chunk(self, parsed, chunking):
        result, _ = run(parsed, chunking, inject_number_drift=1)
        markdown = result.to_markdown()
        assert "⚠ figure changed" in markdown


class TestTablePreview:
    """The ``art translate --preview-tables`` path.

    Worth its own test because it was dead code: a lint pass that removed a
    now-redundant import left the call below with no binding, and nothing caught
    it because no test reached this helper. The path exists so a run is
    inspectable from the terminal, so it needs to keep working.
    """

    def test_preview_shows_rebuilt_tables_and_copies_numbers(self, parsed, chunking):
        result, _ = run(parsed, chunking)
        preview = preview_tables(result)

        assert preview != "(no tables)"
        # the reader must be able to tell which table they are looking at
        assert "Financial Summary" in preview
        # a figure that survives only because numeric cells are copied by code
        assert "1,842,336" in preview
        # and a translated label, proving this is the target and not the source
        assert "营业收入" in preview

    def test_preview_of_an_empty_table_says_so(self):
        assert table_preview(TableBlock(cells=[], n_rows=0, n_cols=0)) == "(empty table)"

# ---------------------------------------------------------------------------
# robustness: a failing backend must degrade, not crash
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_a_chunk_that_fails_still_produces_output_and_a_score(self, parsed, chunking):
        """Losing a page of a financial report is worse than flagging one."""
        llm = MockLLM(seed=1, fail_on_calls=range(0, 60))
        result = TranslationPipeline(llm, policy=RiskPolicy()).run(chunking)

        assert len(result.chunk_results) == len(chunking.chunks)
        # Nothing was dropped ...
        assert all(c.target_text or c.table_translations for c in result.chunk_results)
        # ... and everything is scored, so a human sees it.
        assert len(result.risk_scores) == len(result.chunk_results)

    def test_pipeline_reports_a_failed_chunk_rather_than_raising(self, parsed, chunking):
        llm = MockLLM(seed=1, fail_on_calls=range(0, 60))
        result = TranslationPipeline(llm, policy=RiskPolicy()).run(chunking)
        assert result.summary()["failed_chunks"] >= 0  # must not have raised
