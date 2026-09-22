"""Controlled multi-agent translation.

The single most important design decision in this file
------------------------------------------------------
**Numbers are never sent to the model to be regenerated.**

The naive pipeline translates the table as text and then checks the numbers
afterwards, hoping to catch damage. That is backwards: it makes corruption the
default and detection the safety net, and the safety net has to be perfect.

Instead, a table is decomposed:

* **Label cells** (``Revenue``, ``Item``, ``Year ended 31 December``) are sent
  for translation, addressed by ``(row, col)``.
* **Numeric cells** (``1,234,567``, ``12.4%``) are *never sent*. They are copied
  from source to target by code.

The model therefore has no opportunity to alter a figure, and it cannot disturb
the grid either, because span and index metadata is copied across untouched.
The number guard still runs -- it now catches the *residual* risk (unit notes,
prose figures, a model that adds a figure to a label cell) rather than doing
all the work.

This is why the parser insists on ``(row, col)`` indices: they are the handle
that makes cell-level substitution possible. A pipeline that carried tables as
HTML strings could not do this.

Agent topology
--------------
The design document proposed "a main agent with sub-agents". Kept, but the
sub-agents are **deterministic functions with a model inside them**, not free
agents with their own loops. That matters for a document with audit
implications: you need a fixed, inspectable sequence of steps, not emergent
behaviour. Every step is recorded in an :class:`AgentStep` trace.

    OrchestratorAgent
      |-- BodyTranslatorAgent     prose  -> target, numbers copied verbatim
      |-- TableTranslatorAgent    label cells only, positional payload
      |-- TerminologyAgent        detect + repair glossary violations
      `-- ValidatorAgent          number guard over the assembled target
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..chunker.chunker import Chunk, block_to_text
from ..schema import Block, ChartBlock, ImageBlock, TableBlock, TableCell, TextBlock
from ..textutils import normalize_whitespace
from ..translator.llm import LLMClient, LLMError
from ..translator.number_guard import NumberDiffReport, NumberGuard, extract_numbers

__all__ = [
    "AgentStep",
    "TableTranslation",
    "TerminologyViolation",
    "TranslatorOptions",
    "render_table_payload",
    "apply_table_translation",
    "BodyTranslatorAgent",
    "TableTranslatorAgent",
    "TerminologyAgent",
    "ValidatorAgent",
    "OrchestratorAgent",
]


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

BODY_SYSTEM_PROMPT = """You are a professional financial translator working on an annual report.

Translate the source into {target_language}. Output ONLY the translation.

Absolute rules:
1. Every number must appear in the output character for character as printed.
   Do not translate, reformat, round, add or remove thousands separators, or
   change a decimal point. "1,234,567" stays "1,234,567".
2. Percentage signs, currency symbols, unit words and footnote markers are part
   of the figure. Carry them across unchanged in meaning.
3. Use the approved terminology exactly as given. Do not substitute synonyms.
4. Do not add, explain, summarise, or omit content. Translate what is there.
5. Preserve the [TABLE] / [CHART] / [IMAGE] markers if any appear.
6. Keep paragraph breaks. Keep headings as headings.
"""

TABLE_SYSTEM_PROMPT = """You translate the LABELS of a financial table into {target_language}.

You will receive JSON of the form:
  {{"cells": [{{"r": 0, "c": 1, "text": "Year ended 31 December"}}], "glossary": {{"Revenue": "营业收入"}}}}

Return JSON of the form:
  {{"translations": [{{"r": 0, "c": 1, "text": "<translated label>"}}]}}

Absolute rules:
1. Return one entry for every cell you were given, with the same "r" and "c".
2. Translate ONLY what is given. You are given label cells only. If a supplied
   cell happens to contain a number, copy that number character for character.
3. Never invent or add cells. Never merge or split cells.
4. Use the glossary translations exactly where the source term appears.
5. Keep translations short: these are table labels, not sentences.
"""

TERMINOLOGY_SYSTEM_PROMPT = """You are a terminology reviewer for a translated annual report.

You will receive a source excerpt, its translation, and a list of approved term
pairs that must appear in the translation. Return JSON:
  {{"fixed": "<the translation with the approved terms applied>",
   "applied": ["term1", "term2"],
   "unresolved": ["term3"]}}

Rules:
1. Change ONLY the wording needed to apply the approved terms. Do not rewrite,
   shorten, lengthen, reorder or restructure anything else.
2. Do not touch any number, currency symbol, percentage or unit.
3. If an approved term does not appear in the source excerpt, do nothing with it.
"""

CHART_SYSTEM_PROMPT = """You translate the caption and description of charts in an annual report
into {target_language}.

You will receive JSON:
  {{"charts": [{{"i": 0, "caption": "...", "description": "..."}}]}}

Return JSON:
  {{"charts": [{{"i": 0, "caption": "<translated>", "description": "<translated>"}}]}}

Absolute rules:
1. Return one entry per chart, with the same "i".
2. Every number must appear character for character as printed. Chart descriptions
   are full of figures; not one of them may change.
3. Legend labels and category labels that are proper nouns or product names stay
   as printed. Translate descriptive words.
4. Do not add interpretation, trend commentary or conclusions that are not in
   the source. Translate what is there.
5. Keep the description to the same number of sentences.
"""


@dataclass
class TranslatorOptions:
    target_language: str = "Simplified Chinese"
    source_language: str = "English"
    temperature: float = 0.0
    max_tokens: int = 4096
    #: Retry a failed sub-step once with a repair instruction before giving up.
    max_retries: int = 1
    enforce_terminology: bool = True
    translate_charts: bool = True
    #: Ratio of label cells the model must return before we accept the answer.
    table_coverage_floor: float = 0.6
    glossary_limit: int = 40


@dataclass
class AgentStep:
    agent: str
    action: str
    ok: bool
    detail: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action,
            "ok": self.ok,
            "detail": self.detail[:300],
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class TerminologyViolation:
    source_term: str
    expected: str
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"source_term": self.source_term, "expected": self.expected, "context": self.context}


@dataclass
class TableTranslation:
    """A translated table plus the evidence about how it was produced."""

    source: TableBlock
    target: TableBlock
    number_report: NumberDiffReport
    translated_cells: int = 0
    copied_numeric_cells: int = 0
    missing_labels: list[tuple[int, int]] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)

    @property
    def source_html(self) -> str:
        from ..parser.table_builder import table_to_html

        return table_to_html(self.source)

    @property
    def target_html(self) -> str:
        from ..parser.table_builder import table_to_html

        return table_to_html(self.target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement": [self.source.page, self.source.caption],
            "shape": [self.source.n_rows, self.source.n_cols],
            "units_note": self.source.units_note,
            "translated_label_cells": self.translated_cells,
            "copied_numeric_cells": self.copied_numeric_cells,
            "missing_label_cells": [list(slot) for slot in self.missing_labels],
            "number_report": self.number_report.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# table decomposition helpers
# ---------------------------------------------------------------------------


def is_numeric_cell(text: str, *, unit_hint: str = "") -> bool:
    """True when a cell holds only figures (plus harmless punctuation).

    Such a cell is copied through and never sent to the model.
    """
    if not text or not text.strip():
        return False
    numbers = extract_numbers(text, unit_hint=unit_hint)
    if not numbers:
        return False
    residue = text
    for occurrence in numbers:
        residue = residue.replace(occurrence.raw, " ")
    residue = normalize_whitespace(residue, keep_newlines=False)
    residue = residue.strip(" .,;:()[]%+-–—/\\|'\"")
    # A short alphanumeric residue ("FY", "RMB", "Note") still means the cell is
    # label-bearing; a long one certainly does.
    return len(residue) <= 3


def render_table_payload(
    table: TableBlock,
    glossary_terms: Sequence[tuple[str, str]] = (),
) -> tuple[dict[str, Any], list[tuple[int, int]], int]:
    """Build the translation payload for a table.

    Returns ``(payload, label_slots, numeric_count)``. Only label cells appear in
    the payload -- this function is the enforcement point for "numbers are never
    sent to the model".
    """
    labels: list[dict[str, Any]] = []
    numeric = 0
    for cell in table.cells:
        if not cell.text.strip():
            continue
        if is_numeric_cell(cell.text, unit_hint=table.units_note):
            numeric += 1
            continue
        labels.append({"r": cell.row, "c": cell.col, "text": cell.text})

    payload: dict[str, Any] = {
        "table_caption": table.caption,
        "units_note": table.units_note,
        "shape": [table.n_rows, table.n_cols],
        "cells": labels,
    }
    if glossary_terms:
        payload["glossary"] = dict(glossary_terms)
    return payload, [(c["r"], c["c"]) for c in labels], numeric


def apply_table_translation(
    table: TableBlock,
    translations: dict[tuple[int, int], str],
    *,
    fallback_prefix: str = "",
) -> TableBlock:
    """Rebuild the table with translated labels, everything else copied.

    Spans, indices, geometry and unit note are carried across untouched. This is
    what makes structural corruption of a table impossible rather than merely
    detectable.
    """
    cells: list[TableCell] = []
    for cell in table.cells:
        slot = (cell.row, cell.col)
        translated = translations.get(slot)
        if translated is None:
            text = cell.text
        elif translated.strip():
            text = normalize_whitespace(translated, keep_newlines=True)
        else:
            text = f"{fallback_prefix}{cell.text}" if fallback_prefix else cell.text
        cells.append(
            TableCell(
                text=text,
                row=cell.row,
                col=cell.col,
                row_span=cell.row_span,
                col_span=cell.col_span,
                bbox=cell.bbox,
                is_header=cell.is_header,
                raw=cell.raw,
            )
        )
    return TableBlock(
        cells=cells,
        n_rows=table.n_rows,
        n_cols=table.n_cols,
        caption=table.caption,
        units_note=table.units_note,
        bbox=table.bbox,
        page=table.page,
        row_lines=list(table.row_lines),
        col_lines=list(table.col_lines),
        conflicts=list(table.conflicts),
        warnings=list(table.warnings),
    )


def render_table_payload_text(payload: dict[str, Any]) -> str:
    """Compact, unambiguous JSON for the prompt (keys shortened, no indent churn)."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# sub-agents
# ---------------------------------------------------------------------------


class _BaseAgent:
    name = "agent"

    def __init__(self, llm: LLMClient, options: TranslatorOptions) -> None:
        self.llm = llm
        self.options = options

    def _usage(self) -> tuple[int, int]:
        usage = getattr(self.llm, "last_usage", {}) or {}
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    def _step(self, action: str, ok: bool, detail: str, started: float) -> AgentStep:
        prompt_tokens, completion_tokens = self._usage()
        return AgentStep(
            agent=self.name,
            action=action,
            ok=ok,
            detail=detail,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class BodyTranslatorAgent(_BaseAgent):
    """Translates prose. Explicitly tells the model numbers are not its business."""

    name = "body-translator"

    def translate(self, chunk: Chunk) -> tuple[str, list[AgentStep]]:
        steps: list[AgentStep] = []
        prose, skipped = _chunk_prose(chunk)
        if not prose.strip():
            steps.append(AgentStep(self.name, "skip", True, "no prose in chunk"))
            return "", steps

        system = BODY_SYSTEM_PROMPT.format(target_language=self.options.target_language)
        parts: list[str] = []
        if chunk.section_path:
            parts.append("Section: " + " > ".join(chunk.section_path))
        if chunk.prefix_context:
            # Context only, deliberately marked so the model does not translate it.
            parts.append(
                "PRECEDING CONTEXT (for continuity only -- do NOT translate or repeat this):\n"
                + chunk.prefix_context
            )
        if chunk.glossary_terms:
            pairs = chunk.glossary_terms[: self.options.glossary_limit]
            block = "\n".join(f"- {s} => {t}" for s, t in pairs)
            parts.append("Approved terminology (must be used verbatim):\n" + block)
        parts.append("SOURCE:\n" + prose + "\n---")

        user = "\n\n".join(parts)
        started = time.perf_counter()
        try:
            target = self.llm.complete(
                system=system,
                user=user,
                temperature=self.options.temperature,
                max_tokens=self.options.max_tokens,
            )
        except LLMError as exc:
            steps.append(self._step("translate", False, f"failed: {exc}", started))
            return prose, steps  # fall back to the source so nothing is lost

        steps.append(
            self._step("translate", True, f"{len(prose)} -> {len(target)} chars, {skipped} markers", started)
        )
        return target, steps


class TableTranslatorAgent(_BaseAgent):
    """Translates label cells positionally; numbers are copied by code."""

    name = "table-translator"

    def translate(
        self, table: TableBlock, glossary_terms: Sequence[tuple[str, str]] = ()
    ) -> tuple[TableTranslation, list[AgentStep]]:
        steps: list[AgentStep] = []
        guard = NumberGuard()
        payload, label_slots, numeric_cells = render_table_payload(table, glossary_terms)

        if not label_slots:
            # Pure numeric table: nothing to translate, nothing to ask for.
            steps.append(AgentStep(self.name, "skip", True, "no label cells; table copied verbatim"))
            target = apply_table_translation(table, {})
            return (
                TableTranslation(
                    source=table,
                    target=target,
                    number_report=guard.check_table_pair(table, target),
                    copied_numeric_cells=numeric_cells,
                    steps=steps,
                ),
                steps,
            )

        system = TABLE_SYSTEM_PROMPT.format(target_language=self.options.target_language)
        started = time.perf_counter()
        translations: dict[tuple[int, int], str] = {}
        try:
            reply = self.llm.complete_json(
                system=system,
                user="Translate these table labels.\n" + render_table_payload_text(payload),
                temperature=self.options.temperature,
                max_tokens=self.options.max_tokens,
            )
            translations = _parse_table_translations(reply)
            steps.append(
                self._step(
                    "translate-labels",
                    True,
                    f"{len(translations)}/{len(label_slots)} labels returned",
                    started,
                )
            )
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            steps.append(self._step("translate-labels", False, f"failed: {exc}", started))

        # A partial or empty answer is repaired once before we accept it: a
        # single malformed reply should not cost the whole table its translation.
        missing = [slot for slot in label_slots if slot not in translations]
        if missing and self.options.max_retries > 0:
            retry_payload = dict(payload)
            retry_payload["cells"] = [
                c for c in payload["cells"] if (c["r"], c["c"]) in missing
            ]
            started = time.perf_counter()
            try:
                reply = self.llm.complete_json(
                    system=system,
                    user="Some labels were missing. Return ONLY these.\n"
                    + render_table_payload_text(retry_payload),
                    temperature=self.options.temperature,
                    max_tokens=self.options.max_tokens,
                )
                recovered = _parse_table_translations(reply)
                translations.update(recovered)
                steps.append(
                    self._step("retry-missing-labels", bool(recovered), f"recovered {len(recovered)}", started)
                )
            except (LLMError, ValueError, json.JSONDecodeError) as exc:
                steps.append(self._step("retry-missing-labels", False, f"failed: {exc}", started))
            missing = [slot for slot in label_slots if slot not in translations]

        target = apply_table_translation(table, translations)
        report = guard.check_table_pair(table, target)
        coverage = 1.0 - (len(missing) / len(label_slots) if label_slots else 0.0)
        if coverage < self.options.table_coverage_floor:
            report.structural.append(
                f"only {coverage:.0%} of label cells were translated; table may be partially untranslated"
            )

        return (
            TableTranslation(
                source=table,
                target=target,
                number_report=report,
                translated_cells=len(translations),
                copied_numeric_cells=numeric_cells,
                missing_labels=missing,
                steps=steps,
            ),
            steps,
        )


def _parse_table_translations(reply: dict[str, Any]) -> dict[tuple[int, int], str]:
    """Read the model's label translations into a ``(row, col) -> text`` map."""
    entries = reply.get("translations", reply.get("cells", []))
    out: dict[tuple[int, int], str] = {}
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            row = int(entry.get("r", entry.get("row", -1)))
            col = int(entry.get("c", entry.get("col", -1)))
        except (TypeError, ValueError):
            continue
        text = entry.get("text", entry.get("translation", ""))
        if row < 0 or col < 0 or not isinstance(text, str):
            continue
        out[(row, col)] = text
    return out


class ChartTranslatorAgent(_BaseAgent):
    """Translates chart captions and descriptions.

    Charts are excluded from the prose payload and handled here, for the same
    reason tables are: if a chart's description travels inside the prose blob, it
    gets translated twice -- once as prose and once (unchanged) as a block -- and
    the target document ends up with the chart narrative duplicated in two
    languages. Batching every chart in a chunk into one call keeps the cost at
    one request per chunk rather than one per chart.
    """

    name = "chart-translator"

    def translate(
        self, charts: Sequence[ChartBlock], glossary_terms: Sequence[tuple[str, str]] = ()
    ) -> tuple[list[ChartBlock], list[AgentStep]]:
        steps: list[AgentStep] = []
        if not charts:
            return [], steps

        payload: dict[str, Any] = {
            "charts": [
                {"i": i, "caption": c.caption, "description": c.description}
                for i, c in enumerate(charts)
            ]
        }
        if glossary_terms:
            payload["glossary"] = dict(glossary_terms)

        # Axis ticks and series names are data, not prose; they ride along so
        # the model has context but are never asked to change them.
        system = CHART_SYSTEM_PROMPT.format(target_language=self.options.target_language)
        started = time.perf_counter()
        translated: dict[int, dict[str, str]] = {}
        try:
            reply = self.llm.complete_json(
                system=system,
                user="Translate these chart captions and descriptions.\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                temperature=self.options.temperature,
                max_tokens=self.options.max_tokens,
            )
            translated = _parse_chart_translations(reply)
            steps.append(
                self._step("translate-charts", True, f"{len(translated)}/{len(charts)} charts", started)
            )
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            steps.append(self._step("translate-charts", False, f"failed: {exc}", started))

        out: list[ChartBlock] = []
        for index, chart in enumerate(charts):
            entry = translated.get(index, {})
            out.append(
                ChartBlock(
                    caption=normalize_whitespace(entry.get("caption", chart.caption), keep_newlines=False),
                    chart_type=chart.chart_type,
                    series=[dict(s) for s in chart.series],
                    axis_labels={k: list(v) for k, v in chart.axis_labels.items()},
                    description=normalize_whitespace(entry.get("description", chart.description), keep_newlines=False),
                    extraction_method=chart.extraction_method,
                    confidence=chart.confidence,
                    bbox=chart.bbox,
                    page=chart.page,
                    warnings=list(chart.warnings),
                )
            )
        return out, steps


def _parse_chart_translations(reply: dict[str, Any]) -> dict[int, dict[str, str]]:
    entries = reply.get("charts", [])
    out: dict[int, dict[str, str]] = {}
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("i", entry.get("index", -1)))
        except (TypeError, ValueError):
            continue
        if index < 0:
            continue
        out[index] = {
            "caption": str(entry.get("caption", "")),
            "description": str(entry.get("description", "")),
        }
    return out


class TerminologyAgent(_BaseAgent):
    """Detects and repairs approved-term violations.

    Detection is deterministic (a term present in the source whose approved
    translation is absent from the target). Only the *repair* uses the model, and
    the repair prompt is constrained to touch nothing but wording -- because the
    obvious alternative, a free "improve this translation" pass, is the fastest
    way to lose a number.
    """

    name = "terminology"

    def detect(
        self, source_text: str, target_text: str, terms: Sequence[tuple[str, str]]
    ) -> list[TerminologyViolation]:
        violations: list[TerminologyViolation] = []
        lowered_source = source_text.lower()
        for source_term, target_term in terms:
            if not source_term or not target_term:
                continue
            if source_term.lower() not in lowered_source:
                continue
            if target_term in target_text:
                continue
            violations.append(
                TerminologyViolation(
                    source_term=source_term,
                    expected=target_term,
                    context=_context_around(source_text, source_term),
                )
            )
        return violations

    def repair(
        self, source_text: str, target_text: str, violations: Sequence[TerminologyViolation]
    ) -> tuple[str, list[AgentStep]]:
        steps: list[AgentStep] = []
        if not violations:
            return target_text, steps
        system = TERMINOLOGY_SYSTEM_PROMPT
        pairs = "\n".join(f"- {v.source_term} => {v.expected}" for v in violations)
        user = (
            "Approved terms that are missing from the translation:\n"
            f"{pairs}\n\nSOURCE:\n{source_text[:2000]}\n\nCURRENT TRANSLATION:\n{target_text[:4000]}\n"
        )
        started = time.perf_counter()
        try:
            reply = self.llm.complete_json(
                system=system,
                user=user,
                temperature=self.options.temperature,
                max_tokens=self.options.max_tokens,
            )
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            steps.append(self._step("repair", False, f"failed: {exc}", started))
            return target_text, steps

        fixed = reply.get("fixed")
        if not isinstance(fixed, str) or not fixed.strip():
            steps.append(self._step("repair", False, "repair returned no usable text", started))
            return target_text, steps

        # Guard the repair: a repair that changes figures is rejected outright.
        guard = NumberGuard()
        before = guard.check_text(source_text, target_text)
        after = guard.check_text(source_text, fixed)
        if after.drift > before.drift:
            steps.append(
                self._step(
                    "repair",
                    False,
                    f"rejected: repair worsened number drift {before.drift:.3f} -> {after.drift:.3f}",
                    started,
                )
            )
            return target_text, steps

        resolved = [v for v in violations if v.expected in fixed]
        steps.append(
            self._step("repair", True, f"applied {len(resolved)}/{len(violations)} terms", started)
        )
        return fixed, steps


def _context_around(text: str, needle: str, window: int = 40) -> str:
    index = text.lower().find(needle.lower())
    if index < 0:
        return ""
    return normalize_whitespace(
        text[max(0, index - window) : index + len(needle) + window], keep_newlines=False
    )


class ValidatorAgent(_BaseAgent):
    """Runs the deterministic number guard over the assembled target."""

    name = "validator"

    def __init__(self, llm: LLMClient, options: TranslatorOptions, guard: NumberGuard | None = None) -> None:
        super().__init__(llm, options)
        self.guard = guard or NumberGuard()

    def validate_text(self, source_text: str, target_text: str, *, unit_hint: str = "") -> NumberDiffReport:
        return self.guard.check_text(source_text, target_text, unit_hint=unit_hint)

    def validate_table(self, source: TableBlock, target: TableBlock) -> NumberDiffReport:
        return self.guard.check_table_pair(source, target)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


@dataclass
class ChunkTranslation:
    chunk_id: str
    section_path: tuple[str, ...]
    source_text: str
    target_text: str
    target_blocks: list[Block] = field(default_factory=list)
    table_translations: list[TableTranslation] = field(default_factory=list)
    number_report: NumberDiffReport = field(default_factory=NumberDiffReport)
    terminology_before: list[TerminologyViolation] = field(default_factory=list)
    terminology_after: list[TerminologyViolation] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    glossary_terms: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.number_report.ok

    @property
    def translated_tables(self) -> list[TableBlock]:
        return [t.target for t in self.table_translations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_path": list(self.section_path),
            "source_chars": len(self.source_text),
            "target_chars": len(self.target_text),
            "ok": self.ok,
            "errors": list(self.errors),
            "number_report": self.number_report.to_dict(),
            "terminology": {
                "violations_before": [v.to_dict() for v in self.terminology_before],
                "violations_after": [v.to_dict() for v in self.terminology_after],
            },
            "tables": [t.to_dict() for t in self.table_translations],
            "charts": [
                {
                    "caption": c.caption,
                    "description": c.description,
                    "points": len(c.points),
                    "method": c.extraction_method,
                }
                for c in self.chart_translations
            ],
            "agents": [s.to_dict() for s in self.steps],
        }


class OrchestratorAgent(_BaseAgent):
    """Sequences the sub-agents per chunk and aggregates their evidence."""

    name = "orchestrator"

    def __init__(self, llm: LLMClient, options: TranslatorOptions | None = None) -> None:
        options = options or TranslatorOptions()
        super().__init__(llm, options)
        self.body = BodyTranslatorAgent(llm, options)
        self.tables = TableTranslatorAgent(llm, options)
        self.charts = ChartTranslatorAgent(llm, options)
        self.terminology = TerminologyAgent(llm, options)
        self.validator = ValidatorAgent(llm, options)

    # -- per chunk ----------------------------------------------------------

    def translate_chunk(self, chunk: Chunk) -> ChunkTranslation:
        result = ChunkTranslation(
            chunk_id=chunk.chunk_id,
            section_path=chunk.section_path,
            source_text=chunk.source_text,
            target_text="",
            glossary_terms=list(chunk.glossary_terms),
        )

        # 1. Tables first: they are structural, and their translation is needed
        #    to assemble the target blocks in the right order.
        for block in chunk.blocks:
            if isinstance(block, TableBlock):
                translation, steps = self.tables.translate(block, chunk.glossary_terms)
                result.table_translations.append(translation)
                result.steps.extend(steps)

        # 2. Charts, batched into a single call per chunk.
        if self.options.translate_charts:
            source_charts = [b for b in chunk.blocks if isinstance(b, ChartBlock)]
            translated_charts, steps = self.charts.translate(source_charts, chunk.glossary_terms)
            result.chart_translations = translated_charts
            result.steps.extend(steps)

        # 3. Prose.
        target_prose, steps = self.body.translate(chunk)
        result.target_text = target_prose
        result.steps.extend(steps)

        # 4. Terminology enforcement (deterministic detection, constrained repair).
        #
        # Compare *like with like*: the prose source against the prose target.
        # Using chunk.source_text here is a subtle and expensive bug -- the source
        # includes the table, whose labels are translated on the positional path,
        # so every table term would look like a missing term in the prose and the
        # chunk would be flagged for review on every single run.
        if self.options.enforce_terminology and chunk.glossary_terms and target_prose.strip():
            prose_source, _ = _chunk_prose(chunk)
            violations = self.terminology.detect(prose_source, target_prose, chunk.glossary_terms)
            result.terminology_before = violations
            if violations:
                repaired, steps = self.terminology.repair(prose_source, target_prose, violations)
                result.target_text = repaired
                result.steps.extend(steps)
            result.terminology_after = self.terminology.detect(
                prose_source, result.target_text, chunk.glossary_terms
            )

        # 5. Numbers. Prose, every table, and every chart narrative are all
        #    compared -- a chart description is full of figures and is exactly as
        #    load-bearing as a table cell.
        prose_source, _ = _chunk_prose(chunk)
        result.number_report = self.validator.validate_text(
            prose_source,
            result.target_text,
            unit_hint=_chunk_units_hint(chunk),
        )
        for translation in result.table_translations:
            _merge_reports(result.number_report, translation.number_report)
        source_charts = [b for b in chunk.blocks if isinstance(b, ChartBlock)]
        for source_chart, target_chart in zip(source_charts, result.chart_translations, strict=False):
            chart_source = "\n".join(p for p in (source_chart.caption, source_chart.description) if p)
            chart_target = "\n".join(p for p in (target_chart.caption, target_chart.description) if p)
            if chart_source.strip():
                _merge_reports(
                    result.number_report, self.validator.validate_text(chart_source, chart_target)
                )

        # 6. Reassemble blocks in source order.
        result.target_blocks = self._assemble_blocks(chunk, result)
        if not result.target_text.strip() and not result.table_translations:
            result.errors.append("chunk produced no output")
        return result

    def _assemble_blocks(self, chunk: Chunk, result: ChunkTranslation) -> list[Block]:
        """Rebuild the chunk's block list with translations substituted in place."""
        translated_prose = _split_prose_units(result.target_text, chunk)
        queue = list(translated_prose)
        table_iter = iter(result.table_translations)
        chart_iter = iter(result.chart_translations)
        blocks: list[Block] = []
        for block in chunk.blocks:
            if isinstance(block, TableBlock):
                translation = next(table_iter, None)
                blocks.append(translation.target if translation is not None else block)
            elif isinstance(block, ChartBlock):
                # Fall back to the source chart when the chart agent produced
                # nothing, so the figure and its data still reach the reader.
                blocks.append(next(chart_iter, block))
            elif isinstance(block, TextBlock):
                text = queue.pop(0) if queue else block.text
                blocks.append(
                    TextBlock(
                        text=text,
                        heading_level=block.heading_level,
                        bbox=block.bbox,
                        page=block.page,
                        lang=block.lang,
                        font_size=block.font_size,
                        order=block.order,
                    )
                )
            else:
                blocks.append(block)
        return blocks

    # -- document -----------------------------------------------------------

    def translate_chunks(self, chunks: Sequence[Chunk]) -> list[ChunkTranslation]:
        return [self.translate_chunk(chunk) for chunk in chunks]


def _chunk_prose(chunk: Chunk) -> tuple[str, int]:
    """Render only the TextBlocks; return ``(text, tables_suppressed)``.

    Tables *and* charts are excluded. Both have their own specialised path, and
    including them here would have two consequences, both bad: the model gets a
    second chance to touch a figure, and the block gets emitted twice in the
    target document -- once translated as prose, once unchanged as a block.
    """
    parts: list[str] = []
    suppressed = 0
    for block in chunk.blocks:
        if isinstance(block, (TableBlock, ChartBlock, ImageBlock)):
            suppressed += 1
            continue
        text = block_to_text(block)
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts), suppressed


def _chunk_units_hint(chunk: Chunk) -> str:
    for block in chunk.blocks:
        if isinstance(block, TableBlock) and block.units_note:
            return block.units_note
    return ""


def _split_prose_units(target_text: str, chunk: Chunk) -> list[str]:
    """Split translated prose back onto its source TextBlocks.

    Paragraph-level round-tripping is best-effort: models re-flow paragraphs.
    The count is what matters most, and when the model returns a different number
    of paragraphs we degrade gracefully by putting the whole target on the first
    block rather than dropping text.
    """
    text_blocks = [b for b in chunk.blocks if isinstance(b, TextBlock)]
    if not text_blocks:
        return []
    paragraphs = [p.strip() for p in normalize_whitespace(target_text, keep_newlines=True).split("\n\n") if p.strip()]
    if len(paragraphs) == len(text_blocks):
        return paragraphs
    if len(paragraphs) < len(text_blocks):
        if not paragraphs:
            return []
        # Fewer paragraphs than blocks: keep the first N-1 aligned and give the
        # remainder to the last block so nothing is silently lost.
        return [*paragraphs[: len(text_blocks) - 1], "\n\n".join(paragraphs[len(text_blocks) - 1 :])]
    # More paragraphs than blocks: fold the overflow into the last block for the
    # same reason -- dropping text is never the right failure mode.
    return [*paragraphs[: len(text_blocks) - 1], "\n\n".join(paragraphs[len(text_blocks) - 1 :])]


def _merge_reports(base: NumberDiffReport, extra: NumberDiffReport) -> NumberDiffReport:
    """Fold a table report into the chunk-level report, in place."""
    base.matched += extra.matched
    base.source_total += extra.source_total
    base.target_total += extra.target_total
    base.missing.extend(extra.missing)
    base.added.extend(extra.added)
    base.mismatched.extend(extra.mismatched)
    base.structural.extend(extra.structural)
    base.notes.extend(extra.notes)
    return base


def table_preview(table: TableBlock, *, max_rows: int = 5, max_cols: int = 5) -> str:
    """Small ASCII preview used by the CLI so a run is inspectable without files.

    Renders the slot grid rather than a records view. A records view derives its
    keys from the header row, and a header merged across columns collapses to a
    single key -- which hides exactly the columns a reviewer wants to eyeball on
    a statement. The grid shows every slot, with a merged cell appearing once at
    its origin and the slots it covers left blank.
    """
    if not table.cells:
        return "(empty table)"

    grid = table.grid()
    cols = min(table.n_cols or (len(grid[0]) if grid else 0), max_cols)

    def cell_text(cell: TableCell | None) -> str:
        # ``None`` is either a slot covered by a span or a hole; both render blank.
        return normalize_whitespace(cell.text)[:20] if cell is not None else ""

    head = grid[:max_rows]
    widths = [
        max((len(cell_text(row[c])) for row in head if c < len(row)), default=0)
        for c in range(cols)
    ]

    lines = []
    for row in head:
        cells = [cell_text(row[c]).ljust(widths[c]) for c in range(min(cols, len(row)))]
        lines.append(" | ".join(cells).rstrip())
    if len(grid) > max_rows:
        lines.append(f"... {len(grid) - max_rows} more row(s)")
    return "\n".join(lines)


def summarize_steps(steps: Iterable[AgentStep]) -> dict[str, int]:
    out: dict[str, int] = {}
    for step in steps:
        out[step.agent] = out.get(step.agent, 0) + 1
    return out


def chart_blocks(chunk: Chunk) -> list[ChartBlock]:
    return [b for b in chunk.blocks if isinstance(b, ChartBlock)]


def image_blocks(chunk: Chunk) -> list[ImageBlock]:
    return [b for b in chunk.blocks if isinstance(b, ImageBlock)]
