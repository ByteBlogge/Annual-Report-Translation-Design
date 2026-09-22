"""``art`` -- verifiable multi-agent translation pipeline for annual reports.

The package is organised as the four pipeline stages plus a cross-cutting
HITL layer, mirroring the design document:

    parser/     page layout analysis -> Structured Document Object (SDO)
    chunker/    semantic slicing + glossary management
    translator/ main agent + sub agents, with number-hallucination guard
    hitl/       risk scoring, review queue, reviewer-facing exports

Core is stdlib-only. Heavy backends (VLM API, PaddleOCR, PyMuPDF, FastAPI)
live behind optional extras and are imported lazily inside the modules that
need them, so importing ``art`` never fails on a bare interpreter.
"""

from __future__ import annotations

__version__ = "0.1.0"
PARSER_VERSION = "0.1.0"

__all__ = ["__version__", "PARSER_VERSION"]
