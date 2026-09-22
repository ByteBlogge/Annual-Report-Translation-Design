"""Glossary: the mechanism behind "terminology is consistent across the document".

The two claims under test are (a) the report's own bilingual glosses are
harvested for free and treated as authoritative, and (b) a weaker source can
never overwrite a stronger one -- not even by agreeing, and especially not by
disagreeing.
"""

from __future__ import annotations

import pytest

from art.chunker.glossary import (
    AUTHORITY_RANK,
    SEED_TERMS,
    GlossaryEntry,
    GlossaryStore,
    build_glossary_from_document,
    extract_bilingual_pairs,
    extract_candidates,
)

# ---------------------------------------------------------------------------
# authority ordering
# ---------------------------------------------------------------------------


class TestAuthority:
    def test_rank_order_is_the_documented_one(self):
        # Seed must out-rank model: the glossary exists to constrain the model.
        assert AUTHORITY_RANK["source_gloss"] < AUTHORITY_RANK["curated"]
        assert AUTHORITY_RANK["curated"] < AUTHORITY_RANK["seed"]
        assert AUTHORITY_RANK["seed"] < AUTHORITY_RANK["model"]

    def test_seed_terms_are_loaded_by_default(self):
        store = GlossaryStore()
        assert len(store) == len(SEED_TERMS)
        assert store.get("Revenue") is not None

    def test_seed_can_be_suppressed(self):
        assert len(GlossaryStore(with_seed=False)) == 0

    def test_model_cannot_override_the_seed_lexicon(self):
        """Otherwise every run silently re-decides the terms it was given."""
        store = GlossaryStore()
        original = store.translate("Revenue")
        store.upsert(GlossaryEntry(source="Revenue", target="营业额", authority="model"))
        assert store.translate("Revenue") == original
        assert store.conflicts, "the rejected override must be recorded, not dropped"

    def test_higher_authority_replaces_lower(self):
        store = GlossaryStore(with_seed=False)
        store.upsert(GlossaryEntry(source="Foo", target="弱", authority="model"))
        store.upsert(GlossaryEntry(source="Foo", target="强", authority="source_gloss"))
        assert store.translate("Foo") == "强"
        assert store.get("Foo").authority == "source_gloss"

    def test_agreement_promotes_authority(self):
        """Regression: an issuer gloss agreeing with the seed was downgraded.

        "Revenue" is in the seed lexicon *and* may be supplied by the issuer.
        Agreement is not a conflict -- and the surviving entry must carry the
        stronger authority, or the issuer's own wording could later be
        overwritten by a model proposal.
        """
        store = GlossaryStore()
        canonical = store.translate("Revenue")
        store.upsert(GlossaryEntry(source="Revenue", target=canonical, authority="source_gloss"))
        entry = store.get("Revenue")
        assert entry.authority == "source_gloss"
        assert store.conflicts == []

    def test_agreement_sums_frequency_and_unions_aliases(self):
        store = GlossaryStore(with_seed=False)
        store.upsert(GlossaryEntry(source="Foo", target="甲", authority="seed", aliases=["A"]))
        store.upsert(GlossaryEntry(source="Foo", target="甲", authority="curated", aliases=["B"]))
        entry = store.get("Foo")
        assert entry.frequency == 2
        assert set(entry.aliases) == {"A", "B"}

    def test_equal_authority_disagreement_keeps_the_first_and_reports_it(self):
        """Silently picking a winner is how a glossary becomes untrustworthy."""
        store = GlossaryStore(with_seed=False)
        store.upsert(GlossaryEntry(source="Foo", target="甲", authority="curated"))
        store.upsert(GlossaryEntry(source="Foo", target="乙", authority="curated"))
        assert store.translate("Foo") == "甲"
        assert len(store.conflicts) == 1
        assert "equal-authority" in store.conflicts[0].reason

    def test_lookup_is_case_insensitive(self):
        store = GlossaryStore()
        assert store.translate("revenue") == store.translate("Revenue")
        assert "REVENUE" in store

    def test_translate_passes_unknown_terms_through(self):
        assert GlossaryStore().translate("Nonexistent Term") == "Nonexistent Term"


# ---------------------------------------------------------------------------
# harvesting the issuer's own bilingual glosses
# ---------------------------------------------------------------------------


class TestBilingualPairExtraction:
    @pytest.mark.parametrize(
        ("text", "source", "target"),
        [
            ("EBITDA (息税折旧摊销前利润)", "EBITDA", "息税折旧摊销前利润"),
            ("EBITDA（息税折旧摊销前利润）", "EBITDA", "息税折旧摊销前利润"),
            ("息税折旧摊销前利润（EBITDA）", "EBITDA", "息税折旧摊销前利润"),
            ("Revenue (营业收入)", "Revenue", "营业收入"),
        ],
    )
    def test_extracts_pairs_in_both_directions(self, text, source, target):
        pairs = dict(extract_bilingual_pairs(text))
        assert pairs.get(source) == target

    def test_direction_is_normalised_to_english_key(self):
        """Otherwise the same term lands under two keys depending on the page."""
        assert dict(extract_bilingual_pairs("营业收入（Revenue）"))["Revenue"] == "营业收入"

    def test_rejects_cross_references_and_units(self):
        """"(注5)" and "(单位：千元)" are not translations."""
        assert extract_bilingual_pairs("Revenue (Note 5)") == []
        assert extract_bilingual_pairs("Revenue (RMB'000)") == []

    def test_rejects_same_script_pairs(self):
        assert extract_bilingual_pairs("Revenue (Income)") == []
        assert extract_bilingual_pairs("收入（营业收入）") == []

    def test_deduplicates_preserving_order(self):
        pairs = extract_bilingual_pairs("Revenue (营业收入) and Revenue (营业收入)")
        assert len(pairs) == 1
        assert pairs[0] == ("Revenue", "营业收入")

    def test_leading_conjunction_is_stripped_from_a_harvested_key(self):
        """Regression: a repeated gloss produced the key "and Segment information"."""
        pairs = extract_bilingual_pairs(
            "Segment information (分部资料) and Segment information (分部资料)"
        )
        assert pairs == [("Segment information", "分部资料")]
        assert all(not s.lower().startswith(("and ", "or ", "nor ")) for s, _ in pairs)

    def test_a_term_may_start_with_an_article(self):
        """"The …" is a legitimate gloss, not a stray conjunction."""
        pairs = extract_bilingual_pairs(
            "The board recommends a final dividend per share (每股股息)"
        )
        assert any(s.startswith("The board") for s, _ in pairs)

    def test_mid_sentence_gloss_over_captures_to_the_left(self):
        """A *known* limitation, pinned so it is not mistaken for a guarantee.

        The left-hand pattern must span words for multi-word terms to survive,
        which means a gloss placed mid-sentence captures everything back to the
        start of the run of words. Real bilingual reports put these glosses in
        headings and definition lists, where this cannot happen -- and the
        harvested entry is only ever a *candidate* that a human can correct.
        """
        pairs = extract_bilingual_pairs("Revenue rose, and Segment information (分部资料) followed")
        assert pairs == [("Revenue rose, and Segment information", "分部资料")]

    def test_empty_input(self):
        assert extract_bilingual_pairs("") == []


class TestBuildGlossaryFromDocument:
    def test_issuer_glosses_become_the_top_authority_layer(self):
        store = build_glossary_from_document(["EBITDA (息税折旧摊销前利润)"])
        entry = store.get("EBITDA")
        assert entry.authority == "source_gloss"
        assert entry.target == "息税折旧摊销前利润"

    def test_issuer_gloss_beats_the_seed_lexicon(self):
        """The report's own wording wins over the built-in list."""
        store = build_glossary_from_document(["Capital expenditure (资本性支出)"])
        assert store.translate("Capital expenditure") == "资本性支出"

    def test_seed_frequency_is_bumped_for_terms_the_report_uses(self):
        store = build_glossary_from_document(["Revenue grew and revenue fell"])
        assert store.get("Revenue").frequency > 1

    def test_prune_drops_low_frequency_entries(self):
        store = build_glossary_from_document(
            ["Revenue (营业收入)"], min_frequency=5
        )
        assert store.get("Revenue") is None


# ---------------------------------------------------------------------------
# candidate terminology
# ---------------------------------------------------------------------------


class TestExtractCandidates:
    def test_finds_acronyms(self):
        # CCASS is not in PRESERVED_ACRONYMS, so it is a genuine "no approved
        # translation yet" candidate -- unlike EBITDA, which is tested below.
        assert "CCASS" in extract_candidates("Settled through CCASS")

    def test_keeps_preserved_acronyms_out_of_the_unknown_list(self):
        """IFRS and RMB are *correct* in Latin script, not untranslated terms.

        Counting them would fire the "no approved translation" signal on every
        page and train reviewers to ignore the flags.
        """
        found = extract_candidates("Prepared under IFRS in RMB and HKD; EPS of 0.5")
        assert "IFRS" not in found
        assert "RMB" not in found
        assert "HKD" not in found
        assert "EPS" not in found

    def test_stoplist_acronyms_are_dropped(self):
        assert "THE" not in extract_candidates("THE COMPANY")

    def test_finds_capitalised_phrases(self):
        found = extract_candidates("The Deferred Tax Asset was recognised")
        assert any("Deferred Tax Asset" in f for f in found)

    def test_glossary_hits_are_included(self):
        store = GlossaryStore()
        assert "Revenue" in extract_candidates("Revenue increased", glossary=store)

    def test_deduplicates_and_respects_limit(self):
        found = extract_candidates("EBITDA EBITDA EBITDA", limit=1)
        assert len(found) <= 1

    def test_empty_input(self):
        assert extract_candidates("") == []


# ---------------------------------------------------------------------------
# persistence -- the HITL loop
# ---------------------------------------------------------------------------


class TestGlossaryPersistence:
    def test_save_load_round_trip(self, tmp_path):
        store = GlossaryStore()
        store.upsert(GlossaryEntry(source="Custom Term", target="自定义术语", authority="curated"))
        path = store.save(tmp_path / "glossary.json")

        restored = GlossaryStore.load(path)
        assert restored.translate("Custom Term") == "自定义术语"
        assert restored.get("Custom Term").authority == "curated"

    def test_a_hand_edited_glossary_changes_the_run(self, tmp_path):
        """The workflow this enables: fix a term in JSON, not in code."""
        path = tmp_path / "glossary.json"
        store = GlossaryStore(with_seed=False)
        store.upsert(GlossaryEntry(source="Revenue", target="营业额", authority="curated"))
        store.save(path)

        reloaded = GlossaryStore.load(path, with_seed=False)
        assert reloaded.translate("Revenue") == "营业额"

    def test_conflicts_survive_serialisation(self, tmp_path):
        store = GlossaryStore(with_seed=False)
        store.upsert(GlossaryEntry(source="Foo", target="甲", authority="curated"))
        store.upsert(GlossaryEntry(source="Foo", target="乙", authority="curated"))
        path = store.save(tmp_path / "g.json")

        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        assert payload["conflicts"], "a conflict must be persisted for the reviewer"
