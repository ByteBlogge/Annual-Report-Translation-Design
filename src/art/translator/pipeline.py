"""Stage 3 orchestration: chunks -> translations + risk verdicts.

This is the top-level entry point of the pipeline, and the place where the
design's central promise is cashed in: *every produced chunk carries its own
evidence*. A :class:`TranslationResult` is not just text -- it is the target
text, the translated tables, the per-chunk number report, the risk score with
its reasons, and the agent trace that produced it. That is what makes the HITL
queue meaningful rather than decorative, and it is what lets the run report
answer "which page am I not allowed to trust?".

Failure policy
--------------
A chunk that fails still produces output and still produces a risk score. It is
never dropped. Silently losing a page of a financial report is a far worse
outcome than emitting an untranslated one that is loudly flagged, and the queue
exists precisely so the latter is recoverable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..chunker.chunker import Chunk
from ..chunker.pipeline import ChunkingResult
from ..hitl.exporters import write_review_bundle
from ..hitl.policy import RiskFeatures, RiskPolicy, RiskScore, batch_summary
from ..hitl.queue import ReviewItem, ReviewQueue
from ..schema import ChartBlock, TableBlock, TextBlock
from ..textutils import normalize_whitespace, slugify
from .agents import ChunkTranslation, OrchestratorAgent, TranslatorOptions, table_preview
from .llm import LLMClient
from .number_guard import NumberDiffReport, NumberGuard

__all__ = ["TranslationResult", "TranslationPipeline", "build_features"]


@dataclass
class TranslationResult:
    doc_id: str
    chunk_results: list[ChunkTranslation] = field(default_factory=list)
    risk_scores: list[RiskScore] = field(default_factory=list)
    features: list[RiskFeatures] = field(default_factory=list)
    glossary_conflicts: list[dict[str, Any]] = field(default_factory=list)
    queue: ReviewQueue = field(default_factory=ReviewQueue)
    stats: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    outlines: list[str] = field(default_factory=list)
    outputs: dict[str, Path] = field(default_factory=dict)

    # -- views --------------------------------------------------------------

    @property
    def ok(self) -> bool:
        return all(chunk.ok for chunk in self.chunk_results)

    @property
    def needs_review(self) -> list[tuple[ChunkTranslation, RiskScore]]:
        return [
            (chunk, score)
            for chunk, score in zip(self.chunk_results, self.risk_scores, strict=False)
            if score.needs_review
        ]

    def aggregate_number_report(self) -> NumberDiffReport:
        total = NumberDiffReport()
        for chunk in self.chunk_results:
            report = chunk.number_report
            total.matched += report.matched
            total.source_total += report.source_total
            total.target_total += report.target_total
            total.missing.extend(report.missing)
            total.added.extend(report.added)
            total.mismatched.extend(report.mismatched)
            total.structural.extend(report.structural)
        return total

    def to_markdown(self, *, include_review_notes: bool = True) -> str:
        """Assemble the translated document.

        Tables are emitted as **HTML**, not Markdown, and that is a deliberate
        choice rather than a shortcut. Markdown has no ``rowspan``/``colspan``,
        so rendering a merged-header statement as a Markdown pipe table
        duplicates the merged label across every slot it spans -- a reader of the
        translated document would see ``项目 | 项目 | 项目`` where the source had
        one cell. HTML inside Markdown is valid CommonMark, renders on GitHub and
        in the browser, and preserves the grid exactly as it was verified.

        Charts are rendered from their *extracted data*, not copied as pictures,
        so the numbers in a chart remain searchable and checkable in the target.
        """
        from ..parser.chart_reader import chart_to_narrative
        from ..parser.table_builder import table_to_html

        risk_by_chunk = {id(c): s.value for c, s in zip(self.chunk_results, self.risk_scores, strict=False)}
        parts: list[str] = []
        seen: set[int] = set()
        for chunk in self.chunk_results:
            if chunk.section_path and not parts:
                parts.append("# " + " > ".join(chunk.section_path))
            parts.append(f"<!-- {chunk.chunk_id} risk={risk_by_chunk.get(id(chunk), 0.0):.2f} -->")
            for block in chunk.target_blocks:
                if isinstance(block, TextBlock):
                    if block.heading_level is not None:
                        parts.append("#" * min(6, block.heading_level + 1) + " " + block.text)
                    else:
                        parts.append(block.text)
                elif isinstance(block, TableBlock):
                    if id(block) in seen:
                        continue
                    seen.add(id(block))
                    if block.caption:
                        parts.append(f"**{block.caption}**")
                    parts.append(table_to_html(block))
                elif isinstance(block, ChartBlock):
                    parts.append(chart_to_narrative(block))
                else:
                    caption = getattr(block, "caption", "")
                    parts.append(f"*[image] {caption}*" if caption else "*[image]*")
            parts.append("")
            if include_review_notes:
                parts.extend(self._chunk_footnotes(chunk))
        return "\n".join(parts).strip() + "\n"

    def _chunk_footnotes(self, chunk: ChunkTranslation) -> list[str]:
        notes: list[str] = []
        for source, target, reason in chunk.number_report.mismatched:
            notes.append(f"> ⚠ figure changed: `{source.raw}` → `{target.raw}` ({reason})")
        for occurrence in chunk.number_report.missing:
            notes.append(f"> ⚠ figure missing in target: `{occurrence.raw}`")
        for issue in chunk.number_report.structural:
            notes.append(f"> ⚠ {issue}")
        return ["", *notes, ""] if notes else []

    def summary(self) -> dict[str, Any]:
        report = self.aggregate_number_report()
        scores = self.risk_scores
        return {
            "doc_id": self.doc_id,
            "chunks": len(self.chunk_results),
            "ok_chunks": sum(1 for c in self.chunk_results if c.ok),
            "failed_chunks": sum(1 for c in self.chunk_results if c.errors),
            "tables_translated": sum(len(c.table_translations) for c in self.chunk_results),
            "label_cells_translated": sum(
                t.translated_cells for c in self.chunk_results for t in c.table_translations
            ),
            "numeric_cells_copied": sum(
                t.copied_numeric_cells for c in self.chunk_results for t in c.table_translations
            ),
            "numbers_source": report.source_total,
            "numbers_matched": report.matched,
            "numbers_mismatched": len(report.mismatched),
            "numbers_missing": len(report.missing),
            "number_drift": round(report.drift, 4),
            "risk": batch_summary(scores),
            "review": self.queue.stats(),
            "usage": dict(self.usage),
            "glossary_conflicts": len(self.glossary_conflicts),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "summary": self.summary(),
            "outline": list(self.outlines),
            "chunks": [c.to_dict() for c in self.chunk_results],
            "risk_scores": [s.to_dict() for s in self.risk_scores],
            "features": [f.to_dict() for f in self.features],
            "review_queue": [item.to_dict() for item in self.queue],
            "glossary_conflicts": list(self.glossary_conflicts),
        }

    def save(self, directory: str | Path, *, stem: str = "run") -> dict[str, Path]:
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}

        target_md = base / f"{stem}.target.md"
        target_md.write_text(self.to_markdown(), encoding="utf-8")
        outputs["target_markdown"] = target_md

        report = base / f"{stem}.report.json"
        report.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["report"] = report

        outputs.update(write_review_bundle(self.queue, base, stem="review"))
        self.outputs = outputs
        return outputs


def build_features(
    chunk: Chunk,
    translation: ChunkTranslation,
) -> RiskFeatures:
    """Convert a chunk's artefacts into the policy's input.

    Deliberately the only place that knows how to read a
    :class:`ChunkTranslation`. Keeping the mapping here means the policy stays a
    pure function of measurements, which is what makes it testable without
    running a model.
    """
    report = translation.number_report
    table_conflicts = 0
    table_holes = 0
    table_warnings = 0
    charts_unresolved = 0
    chart_count = 0
    for block in chunk.blocks:
        if isinstance(block, TableBlock):
            table_conflicts += len(block.conflicts)
            table_holes += len(block.holes())
            table_warnings += len(block.warnings)
        elif isinstance(block, ChartBlock):
            chart_count += 1
            if block.extraction_method == "unresolved" or not block.series:
                charts_unresolved += 1

    return RiskFeatures(
        chunk_id=chunk.chunk_id,
        section_path=chunk.section_path,
        pages=(chunk.page_start, chunk.page_end),
        is_financial_summary=chunk.is_financial_summary,
        number_drift=report.drift,
        numbers_missing=len(report.missing),
        numbers_mismatched=len(report.mismatched),
        numbers_added=len(report.added),
        number_structural=len(report.structural),
        table_count=len(chunk.tables),
        table_merge_conflicts=table_conflicts,
        table_holes=table_holes,
        table_warnings=table_warnings,
        untranslated_label_cells=sum(len(t.missing_labels) for t in translation.table_translations),
        chart_count=chart_count,
        charts_unresolved=charts_unresolved,
        numeric_density=chunk.numeric_density,
        glossary_terms=len(chunk.glossary_terms),
        unknown_terms=len(chunk.unknown_terms),
        terminology_remaining=len(translation.terminology_after),
        errors=tuple(translation.errors),
    )


class TranslationPipeline:
    """Stage 3 end to end: translate, verify, score, queue."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        options: TranslatorOptions | None = None,
        policy: RiskPolicy | None = None,
        guard: NumberGuard | None = None,
        run_dir: str | Path | None = None,
        queue_path: str | Path | None = None,
    ) -> None:
        self.llm = llm
        self.options = options or TranslatorOptions()
        self.policy = policy or RiskPolicy()
        self.guard = guard or NumberGuard()
        self.orchestrator = OrchestratorAgent(llm, self.options)
        self.run_dir = Path(run_dir) if run_dir else None
        self.queue_path = Path(queue_path) if queue_path else (self.run_dir / "review.jsonl" if self.run_dir else None)
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # -- run ----------------------------------------------------------------

    def run(self, chunking: ChunkingResult, *, doc_id: str | None = None) -> TranslationResult:
        doc_id = doc_id or chunking.document.doc_id or "document"
        result = TranslationResult(
            doc_id=doc_id,
            glossary_conflicts=[c.to_dict() for c in chunking.glossary.conflicts],
            outlines=list(chunking.outline),
            queue=ReviewQueue(self.queue_path),
        )

        for chunk in chunking.chunks:
            translation = self.orchestrator.translate_chunk(chunk)
            features = build_features(chunk, translation)
            score = self.policy.score(features)

            result.chunk_results.append(translation)
            result.features.append(features)
            result.risk_scores.append(score)

            if score.needs_review:
                self._enqueue(result.queue, chunk, translation, features, score)

        result.usage = self._collect_usage(chunking.chunks)
        result.stats = result.summary()
        if self.run_dir:
            result.save(self.run_dir, stem=slugify(doc_id, max_len=40))
        return result

    # -- queue --------------------------------------------------------------

    def _enqueue(
        self,
        queue: ReviewQueue,
        chunk: Chunk,
        translation: ChunkTranslation,
        features: RiskFeatures,
        score: RiskScore,
    ) -> ReviewItem:
        tables = [
            {
                "caption": t.source.caption,
                "units_note": t.source.units_note,
                "source_html": t.source_html,
                "target_html": t.target_html,
            }
            for t in translation.table_translations
        ]
        item = ReviewItem.from_evidence(
            chunk_id=chunk.chunk_id,
            section_path=chunk.section_path,
            pages=(chunk.page_start, chunk.page_end),
            score=score,
            features=features,
            source_text=chunk.source_text,
            target_text=translation.target_text,
            tables=tables,
        )
        queue.add(item)
        if self.queue_path:
            queue.save(self.queue_path)
        return item

    # -- usage --------------------------------------------------------------

    def _collect_usage(self, chunks: Sequence[Chunk]) -> dict[str, int]:
        usage = getattr(self.llm, "last_usage", {}) or {}
        calls = getattr(self.llm, "call_count", None)
        out = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "chunks": len(chunks),
        }
        if calls is not None:
            out["model_calls"] = int(calls)
        return out


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------


def render_run_summary(result: TranslationResult, *, max_items: int = 8) -> str:
    """Console/report summary: what happened, and what a human still owes."""
    summary = result.summary()
    lines = [
        "=" * 78,
        f"TRANSLATION RUN — {result.doc_id}",
        "=" * 78,
        f"chunks                 {summary['chunks']} ({summary['ok_chunks']} clean, {summary['failed_chunks']} failed)",
        f"tables translated      {summary['tables_translated']}",
        f"  label cells sent     {summary['label_cells_translated']}",
        f"  numeric cells copied {summary['numeric_cells_copied']}  (never sent to the model)",
        f"figures compared       {summary['numbers_source']} source, "
        f"{summary['numbers_matched']} matched, "
        f"{summary['numbers_mismatched']} changed, {summary['numbers_missing']} missing",
        f"number drift           {summary['number_drift']:.2%}",
        f"risk bands             {summary['risk']['bands']}",
        f"review queue           {summary['review']['pending']} pending of {summary['review']['total']} flagged "
        f"({summary['review']['financial_pending']} financial)",
        f"glossary conflicts     {summary['glossary_conflicts']}",
        f"model calls            {summary['usage'].get('model_calls', 'n/a')}",
        "=" * 78,
    ]

    flagged = result.needs_review
    if flagged:
        lines.append("")
        lines.append("FLAGGED FOR HUMAN REVIEW (highest risk first)")
        lines.append("")
        for chunk, score in sorted(flagged, key=lambda pair: -pair[1].value)[:max_items]:
            section = " > ".join(chunk.section_path)[:52] or "(no section)"
            lines.append(f"  [{score.band:7s}] risk {score.value:.2f}  {chunk.chunk_id}")
            lines.append(f"            {section}")
            for factor in sorted(score.reasons, key=lambda f: -f.contribution)[:3]:
                lines.append(f"            +{factor.contribution:.2f} {factor.detail}")
        if len(flagged) > max_items:
            lines.append(f"  ... {len(flagged) - max_items} more in the review sheet")
    else:
        lines.append("")
        lines.append("Nothing flagged: every figure matched and every table rebuilt cleanly.")

    lines.append("")
    return "\n".join(lines)


def render_number_findings(result: TranslationResult, *, limit: int = 20) -> str:
    report = result.aggregate_number_report()
    lines = [f"number integrity: {report.summary()}"]
    if report.mismatched:
        lines.append("")
        lines.append("figures that changed:")
        for source, target, reason in report.mismatched[:limit]:
            lines.append(f"  {source.raw:>16} -> {target.raw:<16} {reason}")
    if report.missing:
        lines.append("")
        lines.append("figures missing from the target:")
        for occurrence in report.missing[:limit]:
            lines.append(f"  {occurrence.raw:>16}  …{occurrence.context[:56]}")
    if report.structural:
        lines.append("")
        lines.append("structural findings:")
        for issue in report.structural[:limit]:
            lines.append(f"  - {issue}")
    return "\n".join(lines)


def save_target_markdown(result: TranslationResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.to_markdown(), encoding="utf-8")
    return p


def preview_tables(result: TranslationResult, *, limit: int = 3) -> str:
    blocks: list[str] = []
    for chunk in result.chunk_results:
        for translation in chunk.table_translations[:limit]:
            blocks.append(f"[{chunk.chunk_id}] {translation.source.caption or '(table)'} "
                          f"{translation.source.n_rows}x{translation.source.n_cols}")
            blocks.append(table_preview(translation.target))
            blocks.append("")
    return "\n".join(blocks) if blocks else "(no tables)"


def normalize_target_text(text: str) -> str:
    return normalize_whitespace(text, keep_newlines=True)
