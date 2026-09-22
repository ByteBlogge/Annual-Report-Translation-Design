"""Stage 3: controlled multi-agent translation with a deterministic number guard.

    list[Chunk]
        -> OrchestratorAgent
             |-- TableTranslatorAgent   label cells only, positional payload
             |-- BodyTranslatorAgent    prose, numbers copied verbatim
             |-- TerminologyAgent       detect + constrained repair
             `-- ValidatorAgent         NumberGuard over the assembled target
        -> ChunkTranslation            target text + tables + evidence + trace
        -> RiskFeatures -> RiskPolicy  -> ReviewQueue (stage 4)
"""

from __future__ import annotations

from .agents import (
    AgentStep,
    BodyTranslatorAgent,
    ChunkTranslation,
    OrchestratorAgent,
    TableTranslation,
    TableTranslatorAgent,
    TerminologyAgent,
    TerminologyViolation,
    TranslatorOptions,
    ValidatorAgent,
    apply_table_translation,
    is_numeric_cell,
    render_table_payload,
)
from .llm import (
    Completion,
    LLMClient,
    LLMError,
    LLMTransientError,
    MockLLM,
    OpenAICompatClient,
    make_llm,
)
from .number_guard import (
    COUNT,
    MONEY,
    RATIO,
    UNKNOWN,
    NumberDiffReport,
    NumberGuard,
    NumberOccurrence,
    cn_to_int,
    extract_numbers,
    fingerprint_numbers,
    parse_unit,
    unit_from_note,
)
from .pipeline import (
    TranslationPipeline,
    TranslationResult,
    render_number_findings,
    render_run_summary,
)

__all__ = [
    # llm
    "LLMClient",
    "LLMError",
    "LLMTransientError",
    "Completion",
    "OpenAICompatClient",
    "MockLLM",
    "make_llm",
    # number guard
    "NumberGuard",
    "NumberDiffReport",
    "NumberOccurrence",
    "extract_numbers",
    "unit_from_note",
    "parse_unit",
    "cn_to_int",
    "fingerprint_numbers",
    "MONEY",
    "COUNT",
    "RATIO",
    "UNKNOWN",
    # agents
    "OrchestratorAgent",
    "BodyTranslatorAgent",
    "TableTranslatorAgent",
    "TerminologyAgent",
    "ValidatorAgent",
    "TranslatorOptions",
    "ChunkTranslation",
    "TableTranslation",
    "TerminologyViolation",
    "AgentStep",
    "render_table_payload",
    "apply_table_translation",
    "is_numeric_cell",
    # pipeline
    "TranslationPipeline",
    "TranslationResult",
    "render_run_summary",
    "render_number_findings",
]
