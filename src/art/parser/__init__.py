"""Stage 1: multimodal document parsing.

    source file
        -> pdf_loader.load_pages          (PDF/PNG -> PageSource with dimensions)
        -> LayoutAnalyzer.analyze         (mock | qwen-vl | paddle)
        -> RawPageLayout.to_blocks        (regions -> typed blocks)
        -> table_builder.build_table      (geometry -> rowspan/colspan grid)
        -> StructuredDocument             (the SDO every later stage consumes)

Import cost is deliberately low: the three backends are imported lazily so that
``import art.parser`` stays dependency-free.
"""

from __future__ import annotations

from .analyzer import (
    ANALYZERS,
    LayoutAnalyzer,
    PageSource,
    RawPageLayout,
    RawRegion,
    build_document,
    make_analyzer,
    normalise_label,
    register_analyzer,
)
from .chart_reader import ChartReader, chart_to_narrative
from .pdf_loader import image_size, load_pages, rasterise_pdf
from .pipeline import (
    DocumentParser,
    ParseResult,
    ParserOptions,
    document_to_markdown,
    summarise_document,
)
from .table_builder import (
    RawCell,
    TableGeometryOptions,
    build_table,
    cluster_edges,
    infer_boundaries,
    table_from_html,
    table_to_html,
    table_to_markdown,
    table_to_records,
    table_to_tsv,
)

__all__ = [
    # analyzer contract
    "LayoutAnalyzer",
    "PageSource",
    "RawPageLayout",
    "RawRegion",
    "ANALYZERS",
    "make_analyzer",
    "register_analyzer",
    "normalise_label",
    "build_document",
    # tables
    "RawCell",
    "TableGeometryOptions",
    "build_table",
    "cluster_edges",
    "infer_boundaries",
    "table_to_html",
    "table_from_html",
    "table_to_markdown",
    "table_to_tsv",
    "table_to_records",
    # io
    "load_pages",
    "image_size",
    "rasterise_pdf",
    # charts
    "ChartReader",
    "chart_to_narrative",
    # orchestration
    "DocumentParser",
    "ParserOptions",
    "ParseResult",
    "summarise_document",
    "document_to_markdown",
]
