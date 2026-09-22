"""Stage 2: semantic chunking and terminology management.

    StructuredDocument
        -> assign_sections        heading detection, section paths
        -> build_glossary         issuer glosses + seed lexicon, authority-ranked
        -> TfidfIndex             retrievable term table
        -> Chunker.chunk           table-atomic, heading-aware slices
        -> list[Chunk]             each carrying its own glossary + context
"""

from __future__ import annotations

from .chunker import (
    Chunk,
    Chunker,
    ChunkOptions,
    block_to_text,
    chunk_document,
    chunking_stats,
    numeric_density,
)
from .glossary import (
    SEED_TERMS,
    GlossaryConflict,
    GlossaryEntry,
    GlossaryStore,
    build_glossary_from_document,
    extract_bilingual_pairs,
    extract_candidates,
)
from .headings import (
    HEADING_RULES,
    SECTION_KEYWORDS,
    HeadingHit,
    SectionSpan,
    assign_sections,
    detect_heading,
    heading_digest,
    infer_body_font_size,
    is_financial_section,
)
from .pipeline import ChunkingPipeline, ChunkingResult, glossary_report
from .retriever import ScoredDoc, TfidfIndex, VectorRetriever, tokenize

__all__ = [
    # chunking
    "Chunk",
    "Chunker",
    "ChunkOptions",
    "chunk_document",
    "chunking_stats",
    "block_to_text",
    "numeric_density",
    # glossary
    "GlossaryStore",
    "GlossaryEntry",
    "GlossaryConflict",
    "SEED_TERMS",
    "extract_bilingual_pairs",
    "extract_candidates",
    "build_glossary_from_document",
    # headings
    "assign_sections",
    "detect_heading",
    "heading_digest",
    "infer_body_font_size",
    "is_financial_section",
    "HeadingHit",
    "SectionSpan",
    "HEADING_RULES",
    "SECTION_KEYWORDS",
    # retrieval
    "TfidfIndex",
    "VectorRetriever",
    "ScoredDoc",
    "tokenize",
    # orchestration
    "ChunkingPipeline",
    "ChunkingResult",
    "glossary_report",
]
