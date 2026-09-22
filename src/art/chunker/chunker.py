"""Semantic chunking with table atomicity.

The rule that matters
---------------------
**A table is never split.** Every other boundary in this module is a tuning
knob; this one is a correctness constraint. Split a financial statement across
two chunks and the column header travels with the first half only -- so the
second half is a grid of numbers with no way to know what they measure, and the
translator will confidently invent headings to fill the gap.

Everything else is a judgement call:

* Start a new chunk at a major heading (level <= ``split_at_level``), because a
  section boundary is also a discourse boundary.
* Pack greedily to ``max_tokens``. Not a fixed window: an annual report mixes
  dense prose with wide tables, and a fixed window cuts both badly.
* Carry the tail of the previous chunk as **prefix context**, kept separate
  from the chunk body. It is shown to the translator but is never translated
  again -- otherwise every boundary would be translated twice and the overlap
  would drift.
* Merge chunks below ``min_tokens`` back into their neighbour, so a stray
  heading does not become its own paid API call.
* Give a large table its own chunk, with the section path attached. A 300-cell
  statement deserves undivided attention and its own number-guard report.

``numeric_density`` is computed here rather than in the policy module because it
is a property of the slice, and because it is the single best predictor of
where a translation is expensive to verify.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schema import Block, ChartBlock, ImageBlock, StructuredDocument, TableBlock, TextBlock
from ..textutils import estimate_tokens, normalize_whitespace, slugify
from .glossary import GlossaryStore, extract_candidates
from .headings import SectionSpan, assign_sections, is_financial_section
from .retriever import TfidfIndex

__all__ = ["ChunkOptions", "Chunk", "block_to_text", "chunk_document", "Chunker"]

_DIGIT_RE = re.compile(r"\d")


@dataclass
class ChunkOptions:
    """Slice sizing. Defaults tuned for a 150-250 page bilingual annual report."""

    #: ~1400 tokens is roughly two dense paragraphs plus a modest table: big
    #: enough for discourse, small enough that a failure costs one retry.
    max_tokens: int = 1400
    #: Chunks below this are merged forward; below this today, one extra API
    #: call per stray heading tomorrow.
    min_tokens: int = 60
    #: A heading at or above this level forces a boundary.
    split_at_level: int = 2
    keep_tables_intact: bool = True
    #: A table consuming more than this fraction of the budget gets its own chunk.
    large_table_ratio: float = 0.5
    prefix_context_tokens: int = 120
    #: Overlap the glossary/context window with the previous chunk's terms.
    inherit_glossary: bool = True
    token_counter: Callable[[str], int] = estimate_tokens


@dataclass
class Chunk:
    """One translatable unit, with everything the translator needs attached."""

    chunk_id: str
    section_path: tuple[str, ...] = ()
    page_start: int = 0
    page_end: int = 0
    blocks: list[Block] = field(default_factory=list)
    source_text: str = ""
    token_estimate: int = 0
    glossary_terms: list[tuple[str, str]] = field(default_factory=list)
    #: Candidate terminology with no approved translation yet -- a HITL signal.
    unknown_terms: list[str] = field(default_factory=list)
    numeric_density: float = 0.0
    is_financial_summary: bool = False
    prefix_context: str = ""
    #: Indexes into ``blocks`` that are tables, for position-preserving translation.
    table_positions: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def section_title(self) -> str:
        return " > ".join(self.section_path)

    @property
    def tables(self) -> list[TableBlock]:
        return [b for b in self.blocks if isinstance(b, TableBlock)]

    @property
    def heading(self) -> str:
        for block in self.blocks:
            if isinstance(block, TextBlock) and block.heading_level is not None:
                return normalize_whitespace(block.text, keep_newlines=False)
        return self.section_path[-1] if self.section_path else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_path": list(self.section_path),
            "pages": [self.page_start, self.page_end],
            "token_estimate": self.token_estimate,
            "numeric_density": round(self.numeric_density, 4),
            "is_financial_summary": self.is_financial_summary,
            "table_positions": list(self.table_positions),
            "glossary_terms": [{"source": s, "target": t} for s, t in self.glossary_terms],
            "unknown_terms": list(self.unknown_terms),
            "block_kinds": [b.kind.value for b in self.blocks],
            "source_text_preview": self.source_text[:300],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# block rendering
# ---------------------------------------------------------------------------


def block_to_text(block: Block) -> str:
    """Render a block as readable text.

    Tables are rendered as grid-expanded Markdown, which repeats merged cell
    content across the slots it spans. That is the *honest* reading view -- it
    is what a person sees -- but it is deliberately **not** what gets sent to
    the translator, because repetition invites the model to translate the same
    header four times with four different results. See
    ``translator.agents.render_table_payload`` for the translation view.
    """
    if isinstance(block, TextBlock):
        prefix = ""
        if block.heading_level is not None:
            prefix = "#" * min(6, block.heading_level) + " "
        return prefix + block.text
    if isinstance(block, TableBlock):
        from ..parser.table_builder import table_to_markdown

        parts: list[str] = []
        if block.caption:
            parts.append(f"[TABLE] {block.caption}")
        if block.units_note:
            parts.append(block.units_note)
        parts.append(table_to_markdown(block))
        return "\n".join(parts)
    if isinstance(block, ChartBlock):
        parts = [f"[CHART] {block.caption}".strip()]
        if block.description:
            parts.append(block.description)
        return "\n".join(parts)
    if isinstance(block, ImageBlock):
        return f"[IMAGE] {block.caption}".strip()
    return ""  # pragma: no cover - exhaustive above


def numeric_density(text: str) -> float:
    """Fraction of characters that are digits.

    A crude but effective proxy: prose sits near 0.01, a results table near
    0.25. It is the signal the risk policy uses to decide how much verification
    a chunk needs.
    """
    if not text:
        return 0.0
    digits = len(_DIGIT_RE.findall(text))
    return digits / len(text)


# ---------------------------------------------------------------------------
# the chunker
# ---------------------------------------------------------------------------


class Chunker:
    def __init__(
        self,
        options: ChunkOptions | None = None,
        *,
        glossary: GlossaryStore | None = None,
        retriever: TfidfIndex | None = None,
    ) -> None:
        self.options = options or ChunkOptions()
        self.glossary = glossary or GlossaryStore()
        self.retriever = retriever

    # -- public API ---------------------------------------------------------

    def chunk(self, document: StructuredDocument) -> list[Chunk]:
        items = list(document.iter_blocks())
        spans = assign_sections(items)
        chunks: list[Chunk] = []
        for span_index, span in enumerate(spans):
            chunks.extend(self._chunk_span(span, document, span_index))
        chunks = self._merge_tiny(chunks)
        self._attach_sequence_context(chunks)
        return chunks

    # -- span -> chunks -----------------------------------------------------

    def _chunk_span(self, span: SectionSpan, document: StructuredDocument, span_index: int) -> list[Chunk]:
        opts = self.options
        is_financial = is_financial_section(span.title_path)

        heading_text = next(
            (block_to_text(b) for b in span.blocks if isinstance(b, TextBlock) and b.heading_level is not None),
            "",
        )
        # The heading is context every chunk in this span carries, not content
        # to be repeated: charging for it once and prefacing each chunk with it
        # keeps the model oriented for free.
        heading_tokens = opts.token_counter(heading_text) if heading_text else 0

        chunks: list[Chunk] = []
        current: list[Block] = []
        used = heading_tokens

        def flush() -> None:
            nonlocal current, used
            if current:
                chunks.append(self._make_chunk(span, current, len(chunks), span_index, is_financial))
            current = []
            used = heading_tokens

        for block in span.blocks:
            text = block_to_text(block)
            cost = opts.token_counter(text)

            if isinstance(block, TableBlock) and opts.keep_tables_intact:
                # Atomic: never split, and prefer a dedicated chunk when big.
                if current and cost > opts.max_tokens * opts.large_table_ratio:
                    flush()
                if not current and cost > opts.max_tokens:
                    current.append(block)
                    chunks.append(self._make_chunk(span, current, len(chunks), span_index, is_financial))
                    current = []
                    used = heading_tokens
                    continue
                if current and used + cost > opts.max_tokens:
                    flush()
                current.append(block)
                used += cost
                continue

            is_major_heading = (
                isinstance(block, TextBlock)
                and block.heading_level is not None
                and block.heading_level <= opts.split_at_level
            )
            if is_major_heading and current:
                flush()

            if current and used + cost > opts.max_tokens:
                flush()

            current.append(block)
            used += cost

        flush()
        return chunks

    def _make_chunk(
        self,
        span: SectionSpan,
        blocks: list[Block],
        index: int,
        span_index: int,
        is_financial: bool,
    ) -> Chunk:
        source_text = "\n\n".join(block_to_text(b) for b in blocks)
        title_slug = slugify(span.title_path[-1] if span.title_path else "section", max_len=24)
        chunk = Chunk(
            chunk_id=f"c{span_index:03d}-{index:02d}-{title_slug}",
            section_path=span.title_path,
            page_start=span.page_start,
            page_end=span.page_end,
            blocks=list(blocks),
            source_text=source_text,
            token_estimate=self.options.token_counter(source_text),
            numeric_density=numeric_density(source_text),
            is_financial_summary=is_financial,
            table_positions=[i for i, b in enumerate(blocks) if isinstance(b, TableBlock)],
        )
        self._attach_glossary(chunk)
        for block in blocks:
            if isinstance(block, TableBlock):
                chunk.warnings.extend(f"table page {block.page}: {w}" for w in block.warnings)
        if not source_text.strip():
            chunk.warnings.append("chunk has no renderable text")
        return chunk

    def _attach_glossary(self, chunk: Chunk) -> None:
        """Attach the terms this chunk actually uses.

        Only relevant terms are attached. Injecting all ~600 seed entries into
        every prompt is expensive and counterproductive: long term lists dilute
        adherence to the ones that matter.
        """
        text = chunk.source_text
        hits = self.glossary.hits(text)
        if self.retriever is not None and self.retriever:
            # Retrieval adds terms that are relevant to this section even when
            # the exact string is absent (e.g. a notes section that discusses
            # "goodwill" using the pronoun-heavy style of footnotes).
            query = " ".join([*chunk.section_path, text[:400]])
            for doc in self.retriever.search(query, k=8):
                entry = self.glossary.get(doc.doc_id)
                if entry is not None and (entry.source, entry.target) not in hits:
                    hits.append((entry.source, entry.target))
        chunk.glossary_terms = hits[:40]
        covered = {source for source, _ in hits}
        chunk.unknown_terms = [
            term for term in extract_candidates(text, glossary=self.glossary)[:30] if term not in covered
        ]

    # -- post-processing ----------------------------------------------------

    def _merge_tiny(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fold undersized chunks into the previous one when they share a section."""
        if not chunks:
            return []
        merged: list[Chunk] = [chunks[0]]
        for chunk in chunks[1:]:
            previous = merged[-1]
            if (
                chunk.token_estimate < self.options.min_tokens
                and previous.section_path == chunk.section_path
                and not chunk.tables
                and previous.token_estimate + chunk.token_estimate <= self.options.max_tokens
            ):
                previous.blocks.extend(chunk.blocks)
                previous.source_text = "\n\n".join(
                    part for part in (previous.source_text, chunk.source_text) if part
                )
                previous.token_estimate = self.options.token_counter(previous.source_text)
                previous.numeric_density = numeric_density(previous.source_text)
                previous.page_end = max(previous.page_end, chunk.page_end)
                previous.table_positions = [
                    i for i, b in enumerate(previous.blocks) if isinstance(b, TableBlock)
                ]
                self._attach_glossary(previous)
                continue
            merged.append(chunk)
        return merged

    def _attach_sequence_context(self, chunks: list[Chunk]) -> None:
        """Give each chunk the tail of the previous one as untranslated context."""
        for previous, chunk in zip(chunks, chunks[1:], strict=False):
            tail = previous.source_text[-self.options.prefix_context_tokens * 4 :]
            chunk.prefix_context = normalize_whitespace(tail, keep_newlines=True)
            if (
                self.options.inherit_glossary
                and not chunk.glossary_terms
                and previous.glossary_terms
            ):
                chunk.glossary_terms = list(previous.glossary_terms)


# ---------------------------------------------------------------------------
# functional API
# ---------------------------------------------------------------------------


def chunk_document(
    document: StructuredDocument,
    options: ChunkOptions | None = None,
    *,
    glossary: GlossaryStore | None = None,
    retriever: TfidfIndex | None = None,
) -> list[Chunk]:
    return Chunker(options, glossary=glossary, retriever=retriever).chunk(document)


def chunking_stats(chunks: Sequence[Chunk]) -> dict[str, Any]:
    if not chunks:
        return {"chunks": 0}
    tokens = [c.token_estimate for c in chunks]
    return {
        "chunks": len(chunks),
        "token_min": min(tokens),
        "token_max": max(tokens),
        "token_mean": round(sum(tokens) / len(tokens), 1),
        "with_tables": sum(1 for c in chunks if c.tables),
        "financial_summary_chunks": sum(1 for c in chunks if c.is_financial_summary),
        "mean_numeric_density": round(sum(c.numeric_density for c in chunks) / len(chunks), 4),
        "sections": len({c.section_path for c in chunks}),
        "unknown_terms": sum(len(c.unknown_terms) for c in chunks),
    }
