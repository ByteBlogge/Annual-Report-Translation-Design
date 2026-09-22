"""The offline demo run, shared by the CLI and the web demo.

This module exists so that ``art demo`` and the browser demo cannot drift apart:
both call :func:`run_demo` and both get the same parse -> chunk -> translate ->
verify sequence. It runs entirely offline on the standard library, using
``MockLayoutAnalyzer`` (replaying a recorded layout) and ``MockLLM`` (a
deterministic stand-in that can be told to corrupt figures on purpose).

The verdict is the point. A demo that merely *shows* a translation proves
nothing; a demo that injects a known number of corruptions and then reports
"caught 3 of 3" is a verification surface the reviewer can operate themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunker.pipeline import ChunkingPipeline, ChunkingResult
from .hitl.policy import RiskPolicy
from .parser.analyzer import PageSource, build_document
from .parser.mock_analyzer import MockLayoutAnalyzer
from .schema import StructuredDocument
from .translator.agents import TranslatorOptions
from .translator.llm import MockLLM
from .translator.pipeline import TranslationPipeline, TranslationResult

__all__ = ["DemoOutcome", "build_demo_document", "run_demo"]


def build_demo_document(
    fixture: str | Path | None = None, *, jitter: float = 2.5
) -> StructuredDocument:
    """Parse the recorded fixture (or the built-in demo page) offline.

    ``jitter`` displaces the analyzer's boxes by a few page units, which is what
    a real VLM returns; the table builder has to recover the grid anyway. Pass
    ``jitter=0`` for the clean geometry.
    """
    if fixture:
        analyzer = MockLayoutAnalyzer(fixture=str(fixture))
        layouts = [
            analyzer.analyze_page(PageSource(page_index=p.page_index))
            for p in analyzer.page_layouts
        ]
        source = str(fixture)
    else:
        analyzer = MockLayoutAnalyzer(jitter=jitter)
        layouts = [
            analyzer.analyze_page(
                PageSource(page_index=p.page_index, width=p.width, height=p.height)
            )
            for p in analyzer.page_layouts
        ]
        source = "<builtin>"

    return build_document(layouts, doc_id="demo-annual-report", source=source)


@dataclass
class DemoOutcome:
    """Everything a caller (CLI line, HTTP response, test) needs from one run."""

    document: StructuredDocument
    chunking: ChunkingResult
    result: TranslationResult
    llm: MockLLM
    injected: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def caught(self) -> int:
        """How many injected corruptions the guard actually reported."""
        return len(self.result.aggregate_number_report().mismatched)

    @property
    def verdict(self) -> dict[str, Any]:
        """Did the guard catch everything the mock actually corrupted?

        The comparison is against ``mock_applied``, **not** the requested count.
        A short fixture can hold fewer figures than the caller asked to corrupt,
        and comparing against the request turns that into a spurious failure
        ("caught 1 of 3!") when in fact nothing was missed. Trust but verify in
        both directions: a silent no-op on the mock's side must not be able to
        masquerade as a pass, so ``mock_applied == 0`` is itself a failure when
        faults were requested.
        """
        applied = self.llm.corruption_applied
        caught = self.caught
        if self.injected == 0:
            return {
                "injected": 0,
                "caught": 0,
                "mock_applied": 0,
                "ok": True,
                "message": "no corruption injected; this is the clean pass",
            }
        if applied == 0:
            message = (
                f"{self.injected} corruption(s) were requested but the mock could inject none "
                f"-- this fixture has too few figures, so nothing was tested"
            )
        elif applied < self.injected:
            message = (
                f"guard caught {caught} of {applied} injected corruption(s); only {applied} of "
                f"{self.injected} could be injected (the fixture has too few figures)"
            )
        else:
            message = f"guard caught {caught} of {applied} injected corruption(s)"
        return {
            "injected": self.injected,
            "caught": caught,
            "mock_applied": applied,
            "ok": applied > 0 and caught >= applied,
            "message": message,
        }


def run_demo(
    fixture: str | Path | None = None,
    *,
    fault: int = 0,
    threshold: float = 0.45,
    jitter: float = 2.5,
    target_language: str = "Simplified Chinese",
    doc: StructuredDocument | None = None,
    chunking: ChunkingResult | None = None,
) -> DemoOutcome:
    """Run the pipeline offline with a deterministic mock backend.

    Pass ``doc``/``chunking`` to reuse an already-parsed document across several
    faults -- the parse is the expensive part and it does not depend on the LLM.
    """
    document = doc if doc is not None else build_demo_document(fixture, jitter=jitter)
    chunking = chunking if chunking is not None else ChunkingPipeline().run(document)

    llm = MockLLM(seed=1 if fault == 0 else 3, inject_number_drift=fault)
    pipeline = TranslationPipeline(
        llm,
        options=TranslatorOptions(target_language=target_language),
        policy=RiskPolicy(threshold=threshold),
    )
    result = pipeline.run(chunking)
    return DemoOutcome(document=document, chunking=chunking, result=result, llm=llm, injected=fault)
