"""Stage 2 orchestration: StructuredDocument -> glossary + chunks.

The ordering here is the whole point, and it is easy to get wrong:

    1. harvest the report's own bilingual glosses  (document-wide, first)
    2. build the glossary store                    (authority-ranked merge)
    3. index the glossary                          (retrieval)
    4. THEN chunk                                  (so each chunk can carry
                                                    the terms it actually uses)

Chunking before the glossary exists is the common shortcut, and it produces
exactly the failure the design set out to avoid: the first chunk of the document
gets an empty glossary, so the terms established in the opening section are
translated one way on page 6 and another way on page 90.

The glossary is built from the *whole document* before any translating happens,
which is the only way terminology can be consistent across the whole document.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import StructuredDocument
from .chunker import Chunk, Chunker, ChunkOptions, chunking_stats
from .glossary import (
    GlossaryEntry,
    GlossaryStore,
    build_glossary_from_document,
)
from .headings import assign_sections, heading_digest
from .retriever import TfidfIndex

__all__ = ["ChunkingResult", "ChunkingPipeline"]


@dataclass
class ChunkingResult:
    document: StructuredDocument
    chunks: list[Chunk]
    glossary: GlossaryStore
    retriever: TfidfIndex
    outline: list[str]
    stats: dict[str, Any]

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        for chunk in self.chunks:
            out.extend(f"{chunk.chunk_id}: {w}" for w in chunk.warnings)
        out.extend(
            f"glossary conflict on {c.source!r}: {c.reason}" for c in self.glossary.conflicts
        )
        return out


class ChunkingPipeline:
    """Builds the document glossary and slices the document into chunks."""

    def __init__(
        self,
        options: ChunkOptions | None = None,
        *,
        glossary: GlossaryStore | None = None,
        index_glossary: bool = True,
    ) -> None:
        self.options = options or ChunkOptions()
        self.base_glossary = glossary
        self.index_glossary = index_glossary

    def run(self, document: StructuredDocument) -> ChunkingResult:
        glossary = self.build_glossary(document)
        retriever = self.build_index(glossary)
        chunks = Chunker(self.options, glossary=glossary, retriever=retriever).chunk(document)
        outline = heading_digest(assign_sections(list(document.iter_blocks())))
        return ChunkingResult(
            document=document,
            chunks=chunks,
            glossary=glossary,
            retriever=retriever,
            outline=outline,
            stats=chunking_stats(chunks),
        )

    # -- glossary -----------------------------------------------------------

    def build_glossary(self, document: StructuredDocument) -> GlossaryStore:
        """Harvest the report's bilingual glosses, then layer them on the seed."""
        texts = _document_text_views(document)
        return build_glossary_from_document(texts, base=self.base_glossary)

    def build_index(self, glossary: GlossaryStore) -> TfidfIndex:
        if not self.index_glossary:
            return TfidfIndex()
        documents = [
            (entry.source, f"{entry.source} {entry.target} {' '.join(entry.aliases)}")
            for entry in glossary
        ]
        return TfidfIndex(documents)

    # -- extras -------------------------------------------------------------

    def glossary_of(self, entries: Iterable[tuple[str, str]], *, authority: str = "model") -> GlossaryStore:
        store = self.base_glossary or GlossaryStore()
        for source, target in entries:
            store.upsert(GlossaryEntry(source=source, target=target, authority=authority))
        return store


def _document_text_views(document: StructuredDocument) -> list[str]:
    """Per-page text, one string per page.

    Page granularity (not one giant string) so that ``source_gloss`` frequency
    counting is not inflated by a term that happens to appear on every page --
    and so memory stays bounded on a 250-page report.
    """
    views: list[str] = []
    for page in document.pages:
        parts: list[str] = []
        for block in page.blocks:
            if hasattr(block, "text"):
                parts.append(block.text)
            if hasattr(block, "caption"):
                parts.append(str(block.caption))
            if hasattr(block, "units_note"):
                parts.append(str(block.units_note))
            for cell in getattr(block, "cells", []) or []:
                parts.append(cell.text)
            for series in getattr(block, "series", []) or []:
                parts.append(str(series.get("name", "")))
        views.append("\n".join(p for p in parts if p))
    return views


def glossary_report(glossary: GlossaryStore, *, limit: int = 25) -> str:
    """Human-readable glossary summary for the CLI and the run report."""
    by_authority: dict[str, int] = {}
    for entry in glossary:
        by_authority[entry.authority] = by_authority.get(entry.authority, 0) + 1
    lines = [
        f"glossary: {len(glossary)} entries "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_authority.items())),
    ]
    if glossary.conflicts:
        lines.append(f"  {len(glossary.conflicts)} conflict(s) need a human decision")
    issuer = sorted(
        (e for e in glossary if e.authority == "source_gloss"),
        key=lambda e: e.source,
    )
    if issuer:
        lines.append("  issuer-supplied pairs (authoritative):")
        for entry in issuer[:limit]:
            lines.append(f"    {entry.source} => {entry.target}")
    return "\n".join(lines)


def save_glossary(glossary: GlossaryStore, path: str | Path) -> Path:
    return glossary.save(path)


def load_or_build(
    document: StructuredDocument, path: str | Path, **kwargs: Any
) -> tuple[GlossaryStore, ChunkingResult]:
    """Reuse a curated glossary from disk when one exists, else build it.

    The workflow this enables: run once, hand-edit ``glossary.json`` to fix the
    terms the model got wrong, re-run, and get a different translation without
    touching code. That is what makes the HITL loop real rather than a diagram.
    """
    p = Path(path)
    glossary = GlossaryStore.load(p) if p.exists() else None
    pipeline = ChunkingPipeline(glossary=glossary, **kwargs)
    result = pipeline.run(document)
    return result.glossary, result
