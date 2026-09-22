"""Stage 4: human-in-the-loop.

    ChunkTranslation  ->  RiskFeatures      (measurements only, no model claims)
                      ->  RiskScore         (weighted, explained, thresholded)
                      ->  ReviewItem        (self-contained evidence)
                      ->  ReviewQueue       (JSONL, persists across runs)
                      ->  exports           (Markdown / HTML / CSV for the reviewer)

The policy decides *what* needs a human; the queue records *that* it was
reviewed, by whom, and why. Keeping those separate is what lets the run report
say "12 items flagged, 0 pending" — i.e. that a human actually closed the loop —
rather than merely "12 items flagged".
"""

from __future__ import annotations

from .exporters import (
    dedupe_items,
    render_item_html,
    render_item_markdown,
    render_review_html,
    render_review_sheet,
    summarise_findings,
    write_review_bundle,
)
from .policy import (
    DEFAULT_WEIGHTS,
    RiskFactor,
    RiskFeatures,
    RiskPolicy,
    RiskScore,
    batch_summary,
)
from .queue import STATUSES, ReviewItem, ReviewQueue

__all__ = [
    "RiskFeatures",
    "RiskFactor",
    "RiskScore",
    "RiskPolicy",
    "DEFAULT_WEIGHTS",
    "batch_summary",
    "ReviewItem",
    "ReviewQueue",
    "STATUSES",
    "render_review_sheet",
    "render_review_html",
    "render_item_markdown",
    "render_item_html",
    "write_review_bundle",
    "summarise_findings",
    "dedupe_items",
]
