"""Human-in-the-loop: risk scoring from measurements, and a queue that closes.

The design claim is narrow and testable: the score is computed from *measured
artefacts only* (number drift, grid holes, unresolved charts) and never from a
model's self-reported confidence -- because a model's confidence is not evidence.
The second claim is that the queue distinguishes "flagged" from "reviewed":
"12 items flagged, 0 pending" is a different statement from "12 items flagged".
"""

from __future__ import annotations

import json

import pytest

from art.hitl.exporters import (
    render_item_html,
    render_item_markdown,
    render_review_html,
    render_review_sheet,
    summarise_findings,
    write_review_bundle,
)
from art.hitl.policy import (
    DEFAULT_WEIGHTS,
    RiskFeatures,
    RiskPolicy,
    RiskScore,
    batch_summary,
)
from art.hitl.queue import STATUSES, ReviewItem, ReviewQueue

# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


class TestRiskPolicy:
    def setup_method(self):
        self.policy = RiskPolicy()

    def test_a_clean_chunk_scores_zero_and_is_auto(self):
        score = self.policy.score(RiskFeatures(chunk_id="c1", number_drift=0.0))
        assert score.value == 0.0
        assert score.band == "auto"
        assert score.needs_review is False
        assert score.reasons == []

    def test_number_drift_dominates(self):
        """Total drift is the worst case: every figure failed to survive."""
        score = self.policy.score(RiskFeatures(number_drift=1.0))
        assert score.value == pytest.approx(1.0)
        assert score.needs_review is True
        assert score.band == "review"
        assert any(r.severity == "critical" for r in score.reasons)

    def test_drift_below_a_tenth_is_a_warning_not_critical(self):
        score = self.policy.score(RiskFeatures(number_drift=0.05))
        factor = next(r for r in score.reasons if r.name == "number_drift")
        assert factor.severity == "warn"

    def test_mismatch_contribution_saturates_at_two(self):
        """Two changed figures already earn the full weight -- this is a
        financial document, not a style exercise."""
        one = self.policy.score(RiskFeatures(numbers_mismatched=1)).value
        two = self.policy.score(RiskFeatures(numbers_mismatched=2)).value
        many = self.policy.score(RiskFeatures(numbers_mismatched=20)).value
        assert one == pytest.approx(DEFAULT_WEIGHTS["number_mismatch"] * 0.5)
        assert two == pytest.approx(DEFAULT_WEIGHTS["number_mismatch"])
        assert many == pytest.approx(DEFAULT_WEIGHTS["number_mismatch"])

    def test_a_single_added_figure_does_not_flag_a_chunk(self):
        """The benign case (a reformatted date) must not drown the real ones."""
        score = self.policy.score(RiskFeatures(numbers_added=1))
        assert score.needs_review is False
        assert score.band == "auto"

    def test_repeated_added_figures_do_raise_risk(self):
        """A fabricated figure must not be invisible either."""
        score = self.policy.score(RiskFeatures(numbers_added=2))
        assert score.value == pytest.approx(DEFAULT_WEIGHTS["number_added"])
        assert score.reasons[0].name == "number_added"

    def test_structural_number_issue_is_scored(self):
        score = self.policy.score(RiskFeatures(number_structural=3))
        assert score.value == pytest.approx(DEFAULT_WEIGHTS["number_structural"])

    def test_grid_holes_and_merge_conflicts_are_scored(self):
        holes = self.policy.score(RiskFeatures(table_holes=4)).value
        conflicts = self.policy.score(RiskFeatures(table_merge_conflicts=2)).value
        assert holes == pytest.approx(DEFAULT_WEIGHTS["grid_hole"])
        assert conflicts == pytest.approx(DEFAULT_WEIGHTS["merge_conflict"])

    def test_unresolved_charts_are_scored_by_proportion(self):
        score = self.policy.score(RiskFeatures(chart_count=2, charts_unresolved=2))
        assert score.value == pytest.approx(DEFAULT_WEIGHTS["chart_unresolved"])

    def test_terminology_leftovers_are_scored(self):
        score = self.policy.score(RiskFeatures(terminology_remaining=3))
        assert score.value == pytest.approx(DEFAULT_WEIGHTS["terminology_remaining"])

    def test_dense_numeric_sections_are_flagged_as_risky_source(self):
        low = self.policy.score(RiskFeatures(numeric_density=0.10)).value
        high = self.policy.score(RiskFeatures(numeric_density=0.30)).value
        assert low == 0.0
        assert high > 0.0

    def test_errors_are_a_hard_failure(self):
        score = self.policy.score(RiskFeatures(errors=("model returned nothing",)))
        assert score.hard_fail is True
        assert score.value == pytest.approx(1.0)
        assert score.band == "blocked"
        assert score.needs_review is True

    def test_unknown_factor_weights_are_ignored(self):
        """A feature with no weight configured must not crash the scorer."""
        assert self.policy.score(RiskFeatures(numeric_density=0.5)).value >= 0.0


class TestThresholds:
    def test_financial_summary_lowers_the_threshold(self):
        """A restatement risk is not the same as a style issue."""
        policy = RiskPolicy()
        plain = policy.effective_threshold(RiskFeatures(is_financial_summary=False))
        financial = policy.effective_threshold(RiskFeatures(is_financial_summary=True))
        assert plain == 0.45
        assert financial == pytest.approx(0.30)

    def test_financial_relief_is_floored(self):
        policy = RiskPolicy(threshold=0.12, financial_relief=0.5, min_threshold=0.10)
        assert policy.effective_threshold(RiskFeatures(is_financial_summary=True)) == 0.10

    def test_band_boundaries(self):
        assert RiskScore(value=0.0, threshold=0.5).band == "auto"
        assert RiskScore(value=0.30, threshold=0.5).band == "watch"  # >= 0.6 * 0.5
        assert RiskScore(value=0.50, threshold=0.5).band == "review"
        assert RiskScore(value=0.0, threshold=0.5, hard_fail=True).band == "blocked"

    def test_a_financial_chunk_can_flag_where_a_plain_one_would_not(self):
        """Same evidence, different threshold -- that is the whole point.

        0.275 (one mismatch) + 0.10 (one structural issue) = 0.375, which sits
        between the financial threshold (0.30) and the default one (0.45).
        """
        policy = RiskPolicy()
        evidence = {"numbers_mismatched": 1, "number_structural": 1}
        plain = policy.score(RiskFeatures(is_financial_summary=False, **evidence))
        financial = policy.score(RiskFeatures(is_financial_summary=True, **evidence))
        assert plain.value == pytest.approx(financial.value)
        assert plain.needs_review is False
        assert financial.needs_review is True
        assert financial.band == "review"

    def test_batch_summary_counts_bands(self):
        scores = [
            RiskScore(value=0.0, threshold=0.5),
            RiskScore(value=0.4, threshold=0.5),
            RiskScore(value=0.9, threshold=0.5),
        ]
        summary = batch_summary(scores)
        assert summary["bands"] == {"auto": 1, "watch": 1, "review": 1}
        assert summary["chunks"] == 3


class TestExplainability:
    def test_explain_names_the_factors(self):
        score = RiskPolicy().score(RiskFeatures(number_drift=0.5, chart_count=1, charts_unresolved=1))
        text = RiskPolicy.explain(score)
        assert "band=review" in text
        assert "number_drift" in text

    def test_explain_handles_a_clean_score(self):
        text = RiskPolicy.explain(RiskPolicy().score(RiskFeatures()))
        assert "no findings" in text

    def test_features_serialise_without_model_claims(self):
        """No field may carry a model-reported confidence."""
        payload = RiskFeatures(number_drift=0.1).to_dict()
        assert "confidence" not in payload
        assert payload["number_drift"] == 0.1


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------


def make_item(chunk_id="c1", risk=0.9, **kw) -> ReviewItem:
    # Overridable defaults, so a test can vary the evidence (target text,
    # created_at, ...) without rebuilding the whole item.
    fields = {"source_text": "Revenue 1,234,567", "target_text": "营业收入 1,234,567"}
    fields.update(kw)
    return ReviewItem(
        chunk_id=chunk_id,
        section_path=list(fields.pop("section_path", ["第一章 财务摘要"])),
        pages=[1, 2],
        risk={"value": risk, "band": "review", "reasons": [{"name": "number_drift", "detail": "x"}]},
        features={"number_drift": 0.5},
        **fields,
    )


class TestReviewQueue:
    def test_add_and_container_protocol(self):
        queue = ReviewQueue()
        queue.add(make_item("c1"))
        queue.add(make_item("c2"))
        assert len(queue) == 2
        assert [i.chunk_id for i in queue] == ["c1", "c2"]
        assert queue.get(queue.sorted_by_risk()[0].item_id) is not None

    def test_items_start_pending(self):
        queue = ReviewQueue()
        queue.add(make_item())
        assert len(queue.pending()) == 1
        assert queue.stats()["pending"] == 1

    def test_deciding_closes_an_item(self):
        """The distinction the whole layer exists for: flagged != reviewed."""
        queue = ReviewQueue()
        item = queue.add(make_item())
        queue.decide(item.item_id, "approved", reviewer="alice", note="checked against source")
        assert queue.pending() == []
        assert queue.stats()["total"] == 1
        assert queue.stats()["by_status"]["approved"] == 1
        assert item.reviewer == "alice"

    def test_invalid_status_is_rejected(self):
        with pytest.raises(ValueError):
            make_item().decide("maybe")

    def test_all_documented_statuses_are_accepted(self):
        for status in STATUSES:
            assert make_item().decide(status).status == status

    def test_sorted_by_risk_is_descending_by_default(self):
        queue = ReviewQueue()
        queue.add(make_item("low", risk=0.2))
        queue.add(make_item("high", risk=0.95))
        assert [i.chunk_id for i in queue.sorted_by_risk()] == ["high", "low"]

    def test_stats_track_financial_items_separately(self):
        queue = ReviewQueue()
        queue.add(make_item("fin", risk=0.9, is_financial_summary=True))
        queue.add(make_item("plain", risk=0.8, is_financial_summary=False))
        stats = queue.stats()
        assert stats["financial_pending"] == 1
        assert stats["total"] == 2

    def test_jsonl_round_trip(self, tmp_path):
        queue = ReviewQueue()
        queue.add(make_item("c1"))
        queue.add(make_item("c2", risk=0.5))
        queue.decide(queue.sorted_by_risk()[0].item_id, "rejected", reviewer="bob")

        path = queue.save(tmp_path / "review.jsonl")
        # One JSON object per line -- append-safe, and diffable in review.
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2
        assert all(json.loads(line)["item_id"] for line in lines)

        reloaded = ReviewQueue(tmp_path / "review.jsonl").load()
        assert len(reloaded) == 2
        assert reloaded.stats()["by_status"]["rejected"] == 1

    def test_append_does_not_rewrite_the_file(self, tmp_path):
        path = tmp_path / "review.jsonl"
        queue = ReviewQueue(path)
        queue.append(make_item("c1"))
        queue.append(make_item("c2"))
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_load_of_a_missing_file_is_a_no_op(self, tmp_path):
        assert len(ReviewQueue(tmp_path / "nope.jsonl").load()) == 0

    def test_csv_export_has_a_header_and_one_row_per_item(self):
        queue = ReviewQueue()
        queue.add(make_item("c1"))
        queue.add(make_item("c2", risk=0.5))
        csv_text = queue.to_csv()
        lines = [line for line in csv_text.splitlines() if line.strip()]
        assert lines[0].startswith("item_id,chunk_id,status")
        assert len(lines) == 3
        assert "c1" in csv_text

    def test_csv_can_include_the_text_for_a_reviewer_offline(self):
        queue = ReviewQueue()
        queue.add(make_item("c1"))
        assert "营业收入" in queue.to_csv(include_text=True)

    def test_save_without_a_path_is_an_error(self):
        with pytest.raises(ValueError):
            ReviewQueue().save()


class TestRerunningIntoTheSameQueue:
    """Translating a document twice must not double the reviewer's work.

    Run directories are reused (`art translate ... --out runs/demo`), so the
    second run has to refresh the entries the reviewer already saw. It also has
    to be honest about what a decision covers: an approval is of one specific
    output, not of a chunk forever.
    """

    def test_a_second_run_refreshes_rather_than_duplicates(self):
        queue = ReviewQueue()
        queue.add(make_item("c1"))
        queue.add(make_item("c1"))
        assert len(queue) == 1
        assert queue.stats()["pending"] == 1

    def test_a_decision_survives_a_rerun_with_unchanged_evidence(self):
        queue = ReviewQueue()
        item = queue.add(make_item("c1"))
        queue.decide(item.item_id, "approved", reviewer="alice", note="checked")

        queue.add(make_item("c1"))
        kept = queue.get(item.item_id)
        assert len(queue) == 1
        assert kept.status == "approved"
        assert kept.reviewer == "alice"
        assert kept.note == "checked"

    def test_changed_risk_reopens_a_decided_item_and_keeps_the_audit_trail(self):
        queue = ReviewQueue()
        item = queue.add(make_item("c1", risk=0.9))
        queue.decide(item.item_id, "approved", reviewer="alice", note="looks right")

        queue.add(make_item("c1", risk=0.4))
        reopened = queue.get(item.item_id)
        assert len(queue) == 1
        assert reopened.status == "pending"
        assert "reopened on re-run" in reopened.note
        assert "approved by alice" in reopened.note

    def test_changed_text_reopens_even_when_the_risk_is_identical(self):
        queue = ReviewQueue()
        item = queue.add(make_item("c1"))
        queue.decide(item.item_id, "rejected", reviewer="bob")

        queue.add(make_item("c1", target_text="营业收入 1,234,568"))
        assert queue.get(item.item_id).status == "pending"

    def test_created_at_records_when_the_work_first_appeared(self):
        queue = ReviewQueue()
        item = make_item("c1", created_at="2020-01-01T00:00:00")
        queue.add(item)
        queue.add(make_item("c1"))
        assert queue.get(item.item_id).created_at == "2020-01-01T00:00:00"

    def test_loading_a_file_with_a_duplicate_line_collapses_it(self, tmp_path):
        """Files written by the old append-per-run behaviour must still load."""
        queue = ReviewQueue()
        queue.add(make_item("c1"))
        queue.add(make_item("c2"))
        path = queue.save(tmp_path / "review.jsonl")

        text = path.read_text(encoding="utf-8")
        path.write_text(text + text.splitlines()[0] + "\n", encoding="utf-8")

        assert len(ReviewQueue(path).load()) == 2


class TestReviewItem:
    def test_id_is_derived_from_the_chunk(self):
        """Stable identity, so a re-run refreshes the entry instead of adding one.

        A random id would append a near-duplicate on every run and put a stored
        decision permanently out of reach -- and it would break the diffable
        queue the module exists to provide.
        """
        assert make_item("c1").item_id == make_item("c1").item_id
        assert make_item("c1").item_id != make_item("c2").item_id

    def test_id_is_stable_across_serialisation(self):
        item = make_item()
        assert ReviewItem.from_dict(item.to_dict()).item_id == item.item_id

    def test_top_reasons_are_rendered_for_a_reviewer(self):
        assert make_item().top_reasons == ["number_drift: x"]

    def test_risk_value_is_robust_to_a_malformed_payload(self):
        assert ReviewItem(risk={"value": "oops"}).risk_value == 0.0

    def test_decide_keeps_the_previous_reviewer_when_omitted(self):
        item = make_item().decide("approved", reviewer="alice")
        item.decide("rejected")
        assert item.reviewer == "alice"

    def test_serialise_round_trip(self):
        item = make_item()
        restored = ReviewItem.from_dict(json.loads(json.dumps(item.to_dict())))
        assert restored.item_id == item.item_id
        assert restored.chunk_id == item.chunk_id
        assert restored.risk == item.risk


# ---------------------------------------------------------------------------
# reviewer-facing exports
# ---------------------------------------------------------------------------


class TestExports:
    def setup_method(self):
        self.queue = ReviewQueue()
        self.queue.add(make_item("c1", risk=0.9))
        self.queue.add(make_item("c2", risk=0.5, is_financial_summary=True))

    def test_markdown_item_shows_both_sides(self):
        text = render_item_markdown(self.queue.sorted_by_risk()[0])
        assert "Revenue 1,234,567" in text
        assert "营业收入 1,234,567" in text

    def test_html_item_is_escaped(self):
        item = make_item()
        item.source_text = "<script>alert(1)</script>"
        assert "<script>" not in render_item_html(item)

    def test_review_sheet_lists_every_item(self):
        sheet = render_review_sheet(self.queue)
        assert "c1" in sheet and "c2" in sheet

    def test_review_html_contains_both_chunks(self):
        html = render_review_html(self.queue)
        assert "c1" in html and "c2" in html

    def test_summarise_findings_aggregates(self):
        summary = summarise_findings(list(self.queue))
        assert summary["items"] == 2
        assert summary["by_reason"] == {"number_drift": 2}
        assert summary["financial_items"] == 1

    def test_write_bundle_creates_reviewer_artefacts(self, tmp_path):
        outputs = write_review_bundle(self.queue, tmp_path, stem="review")
        assert outputs, "the bundle must report what it wrote"
        for path in outputs.values():
            assert path.exists()


class TestEvidenceIsSelfContained:
    def test_from_evidence_captures_everything_a_reviewer_needs(self):
        """A reviewer must not need to re-run the pipeline to judge an item.

        That is the difference between a queue that gets worked and a queue that
        gets ignored.
        """
        from art.hitl.policy import RiskPolicy

        features = RiskFeatures(chunk_id="c1", section_path=("第一章",), pages=(3, 3), number_drift=1.0)
        score = RiskPolicy().score(features)
        item = ReviewItem.from_evidence(
            chunk_id="c1",
            section_path=("第一章",),
            pages=(3, 3),
            score=score,
            features=features,
            source_text="Revenue 1,234,567",
            target_text="营业收入 1,234,000",
            tables=[{"caption": "T", "units_note": "千元", "source_html": "<table></table>",
                     "target_html": "<table></table>"}],
        )
        assert item.source_text and item.target_text
        assert item.tables
        assert item.risk["value"] == pytest.approx(1.0)
        assert item.pages == [3, 3]
        assert item.status == "pending"
        assert item.created_at
