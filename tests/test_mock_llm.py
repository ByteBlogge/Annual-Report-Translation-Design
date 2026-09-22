"""The mock backend, and the discipline of its fault injector.

The injector is load-bearing: every "the guard caught N of N" claim in the repo
and on the demo page is only as meaningful as the injector's aim. Two rules keep
it honest, and both are tested here:

* it corrupts only what the guard actually tracks as a *figure*; and
* it corrupts a figure in a way that shows up as a **change**, not as an
  addition the guard would treat as benign.
"""

from __future__ import annotations

import pytest

from art.translator.llm import (
    LLMError,
    LLMTransientError,
    MockLLM,
    OpenAICompatClient,
    looks_like_a_date,
    make_llm,
)
from art.translator.number_guard import extract_numbers

# ---------------------------------------------------------------------------
# date detection -- the exclusion rule
# ---------------------------------------------------------------------------


class TestLooksLikeADate:
    @pytest.mark.parametrize(
        "text",
        [
            "Year ended 31 December 2023",
            "for the year ended 31 December",
            "截至12月31日止年度",
            "3月15日",
            "31 December",
        ],
    )
    def test_recognises_date_figures(self, text):
        digits = extract_numbers(text)
        assert digits, text
        assert all(
            looks_like_a_date(text, o.span[0], o.span[1]) for o in digits
        ), f"a figure in {text!r} was not recognised as part of a date"

    @pytest.mark.parametrize(
        "text",
        [
            "Revenue was 1,234,567 千元",
            "an increase of 12.4% year on year",
            "operating profit rose to 234,567 千元",
            "net loss of (456,789)",
        ],
    )
    def test_does_not_mistake_values_for_dates(self, text):
        digits = extract_numbers(text)
        assert digits, text
        assert not any(
            looks_like_a_date(text, o.span[0], o.span[1]) for o in digits
        ), f"a real figure in {text!r} was wrongly treated as a date"


class TestInjectorAim:
    def setup_method(self):
        self.llm = MockLLM(inject_number_drift=1, seed=3)

    def test_corrupts_a_plain_figure(self):
        out = self.llm._corrupt_numbers("Revenue was 1,234,567 千元")
        assert out != "Revenue was 1,234,567 千元"

    def test_leaves_a_date_figure_alone(self):
        """A corrupted date is indistinguishable from a benign reformat.

        ``Year ended 31 December`` becomes ``截至12月31日止年度`` -- which
        *legitimately* adds a figure (the month) that the source never printed.
        Corrupting the day leaves the original ``31`` in place, so the guard would
        report an addition rather than a change and the demo's verdict would
        understate what was caught. The injector therefore has nothing to do here.
        """
        text = "Year ended 31 December"
        assert self.llm._corrupt_numbers(text) == text
        assert self.llm.corruption_applied == 0

    def test_leaves_a_scale_marker_alone(self):
        """``RMB'000`` is a unit, not a figure, and the guard reads it that way.

        Corrupting it to ``RMB'002`` produced a change the guard is designed not
        to see -- the injector and the verifier disagreed about what a figure was.
        """
        text = "Revenue by segment (RMB'000)"
        assert self.llm._corrupt_numbers(text) == text
        assert self.llm.corruption_applied == 0

    def test_only_corrupts_figures_the_guard_tracks(self):
        """The invariant, stated directly: injector eligibility == guard extraction."""
        text = "At 31 December 2023 revenue was 1,234,567 千元 (RMB'000)."
        before = self.llm._corrupt_numbers(text)
        corrupted_spans = [o for o in extract_numbers(before) if o.raw and o.raw not in text]
        assert corrupted_spans, "expected exactly one figure to have been changed"
        # The pristine figures (the two dates and the scale marker) survived.
        assert "31 December 2023" in before
        assert "RMB'000" in before
        assert self.llm.corruption_applied == 1

    def test_digit_swap_preserves_length_and_commas(self):
        llm = MockLLM(inject_number_drift=1, seed=5)
        text = "Total 8,765,432 千元"
        out = llm._corrupt_numbers(text)
        original = extract_numbers(text)[0]
        changed = extract_numbers(out)[0]
        assert len(changed.digits) == len(original.digits)
        assert changed.digits.count("") == original.digits.count("")
        # Comma grouping is rebuilt, not dropped.
        assert "," in out

    def test_no_budget_means_no_change(self):
        llm = MockLLM(inject_number_drift=0)
        assert llm._corrupt_numbers("1,234,567") == "1,234,567"
        assert llm.corruption_applied == 0

    def test_text_without_figures_is_untouched(self):
        assert self.llm._corrupt_numbers("no figures at all") == "no figures at all"


class TestBudget:
    def test_budget_is_consumed_and_reported(self):
        llm = MockLLM(inject_number_drift=3)
        for text in ("a 1,111", "b 2,222", "c 3,333", "d 4,444"):
            llm._corrupt_numbers(text)
        assert llm.corruption_applied == 3, "the budget must not be over-spent"

    def test_budget_is_reusable_after_reset(self):
        """A harness that reuses one instance must re-arm, or the second pass
        silently injects nothing and looks like a perfect run."""
        llm = MockLLM(inject_number_drift=2)
        llm._corrupt_numbers("1,111 2,222")
        assert llm.corruption_applied == 2

        llm.reset_run()
        assert llm.corruption_applied == 0
        assert llm.calls == []
        llm._corrupt_numbers("3,333 4,444")
        assert llm.corruption_applied == 2


# ---------------------------------------------------------------------------
# other injection modes
# ---------------------------------------------------------------------------


class TestOtherModes:
    def test_unit_flip_breaks_a_unit_note(self):
        llm = MockLLM(inject_unit_flip=True)
        out = llm._prose_reply("SOURCE:\n收益为 1,234 千元\n---")
        assert "元" in out and "千元" not in out

    def test_truncate_shortens_the_reply(self):
        llm = MockLLM(truncate=True)
        long_text = "SOURCE:\n" + ("Revenue grew substantially. " * 20) + "\n---"
        assert len(llm._prose_reply(long_text)) < len(long_text) / 2

    def test_transient_failure_is_raised_on_the_named_call(self):
        llm = MockLLM(fail_on_calls=[1])
        llm.complete(system="s", user="u")
        with pytest.raises(LLMTransientError):
            llm.complete(system="s", user="u")
        llm.complete(system="s", user="u")  # recovers afterwards

    def test_prose_reply_marks_text_it_could_not_translate(self):
        llm = MockLLM()
        out = llm._prose_reply("SOURCE:\nUnmappedSentence here\n---")
        assert out.startswith("[译]")

    def test_calls_are_recorded_for_forensics(self):
        llm = MockLLM()
        llm.complete(system="sys", user="usr")
        assert len(llm.calls) == 1
        assert llm.calls[0]["payload"]["user"] == "usr"
        assert llm.call_count == 1


class TestFactory:
    def test_mock_is_the_default(self):
        assert isinstance(make_llm("mock"), MockLLM)
        assert isinstance(make_llm(""), MockLLM)

    def test_auto_falls_back_without_credentials(self):
        assert isinstance(make_llm("auto"), MockLLM)

    def test_auto_uses_a_real_backend_when_configured(self):
        client = make_llm("auto", base_url="https://example.invalid/v1", api_key="k", model="m")
        assert isinstance(client, OpenAICompatClient)

    def test_missing_key_is_an_error_not_a_silent_fallback(self):
        """Silently mocking a production run would be the worst possible failure."""
        with pytest.raises(LLMError):
            make_llm("openai-compat", base_url="https://example.invalid/v1", model="m")

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(KeyError):
            make_llm("definitely-not-a-backend")
