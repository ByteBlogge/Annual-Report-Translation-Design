"""The offline demo service, and the verdict that makes it worth running.

``run_demo`` is shared by ``art demo`` and the browser demo, so these tests cover
both. The verdict is the part that matters: it has to be able to *fail*, which
means the checks here include the ways it should fail.
"""

from __future__ import annotations

import pytest

from art.chunker.pipeline import ChunkingPipeline
from art.demo import build_demo_document, run_demo
from art.schema import DocumentPage, StructuredDocument, TextBlock
from art.translator.number_guard import extract_numbers
from art.web.render import audit_payloads

# ---------------------------------------------------------------------------
# parsing the demo document
# ---------------------------------------------------------------------------


class TestBuildDemoDocument:
    def test_builtin_page(self):
        doc = build_demo_document(None)
        assert len(doc.pages) == 1
        assert doc.tables()
        assert doc.charts()

    def test_recorded_fixture(self, demo_fixture_path):
        doc = build_demo_document(demo_fixture_path)
        assert len(doc.pages) == 4
        assert len(doc.tables()) == 3

    def test_builtin_page_has_figures_the_model_can_actually_see(self):
        """Without these the offline demo cannot exercise the guard at all.

        Every figure in the built-in page's *table* is a numeric cell, which the
        pipeline copies by code and never sends to the model. If the prose held no
        figures either, the injector would have nothing to corrupt and the demo
        would print "caught 0 of 0" -- a vacuous pass.
        """
        doc = build_demo_document(None)
        prose = "\n".join(
            b.text
            for b in doc.texts()
            if b.heading_level is None
        )
        figures = [o for o in extract_numbers(prose) if not o.raw.isdigit() or len(o.raw) > 4]
        assert figures, "the built-in prose must contain model-visible figures"


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_clean_run_reports_nothing_caught(self):
        outcome = run_demo(None, fault=0)
        assert outcome.verdict["ok"] is True
        assert outcome.verdict["mock_applied"] == 0
        assert outcome.caught == 0

    @pytest.mark.parametrize("fault", [1, 2, 3, 4, 5, 6])
    def test_builtin_catches_every_injected_corruption(self, fault):
        outcome = run_demo(None, fault=fault)
        assert outcome.verdict["mock_applied"] == fault
        assert outcome.caught == fault, outcome.verdict["message"]
        assert outcome.verdict["ok"] is True

    @pytest.mark.parametrize("fault", [1, 2, 3, 5, 8])
    def test_recorded_fixture_catches_every_injected_corruption(self, demo_fixture_path, fault):
        outcome = run_demo(demo_fixture_path, fault=fault)
        assert outcome.caught == fault, outcome.verdict["message"]
        assert outcome.verdict["ok"] is True

    def test_the_verdict_can_fail_when_nothing_was_injected(self):
        """A fault-injection run that injected nothing must not pass.

        This is the guard against a vacuous green tick: the checks are written so
        that a backend which silently does nothing is reported as a failure, not
        as "0 of 0, all clear".
        """
        document = StructuredDocument(
            doc_id="no-figures",
            pages=[
                DocumentPage(
                    index=1,
                    width=595,
                    height=842,
                    blocks=[
                        TextBlock(text="第一章 概述", heading_level=1),
                        TextBlock(text="This sentence deliberately carries no figures at all."),
                    ],
                )
            ],
        )
        chunking = ChunkingPipeline().run(document)
        outcome = run_demo(None, fault=3, doc=document, chunking=chunking)

        assert outcome.verdict["mock_applied"] == 0
        assert outcome.verdict["ok"] is False
        assert "nothing was tested" in outcome.verdict["message"]

    def test_reusing_a_parsed_document_does_not_change_the_result(self):
        document = build_demo_document(None)
        chunking = ChunkingPipeline().run(document)
        first = run_demo(None, fault=2, doc=document, chunking=chunking)
        second = run_demo(None, fault=2, doc=document, chunking=chunking)
        assert first.caught == second.caught
        assert first.result.to_markdown() == second.result.to_markdown()


# ---------------------------------------------------------------------------
# the payload audit
# ---------------------------------------------------------------------------


class TestAudit:
    def test_fixture_payloads_are_clean(self, demo_fixture_path):
        report = audit_payloads(build_demo_document(demo_fixture_path))
        assert report["ok"] is True
        assert report["leaks"] == []
        assert report["tables_audited"] == 3
        assert report["label_cells_sent"] > 0
        assert report["numeric_cells_copied"] > 0

    def test_builtin_payloads_are_clean(self):
        report = audit_payloads(build_demo_document(None))
        assert report["ok"] is True

    def test_the_audit_is_not_vacuous(self, demo_fixture_path):
        """It must be measuring something: labels sent, figures withheld."""
        report = audit_payloads(build_demo_document(demo_fixture_path))
        assert report["label_cells_sent"] + report["numeric_cells_copied"] > 0
