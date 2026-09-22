"""Heading detection and section-path assignment.

Why this is not ``split("\\n\\n")``
----------------------------------
The design claim is "semantic slicing, not character slicing". For that to mean
anything, every chunk has to know *where it sits in the document* -- because
that path is what gets injected into the translation prompt, and it is what
makes a translated term consistent between the summary table on page 5 and the
detailed note on page 87.

An annual report offers several independent heading signals, and they disagree
often enough that using only one is unreliable:

* **Numbering** -- ``第一章``, ``1.2.3``, ``（一）``. Precise when present, but
  appendix pages and front matter frequently have none.
* **Font size** -- reliable on typeset PDFs, unavailable from a VLM that only
  reports a bounding box.
* **Capitalisation and brevity** -- ``CONSOLIDATED STATEMENT OF PROFIT OR LOSS``.

So the detector tries them in order of reliability and returns the first hit,
recording which rule fired. That provenance is kept on the block because when a
heading is misdetected the reviewer needs to know *why* the boundary was drawn
there, and "the all-caps rule fired on a footnote" is a different bug from "the
numbering rule misread 2023 as a section number".
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

from ..schema import Block, TableBlock, TextBlock
from ..textutils import normalize_whitespace

__all__ = [
    "HEADING_RULES",
    "HeadingHit",
    "detect_heading",
    "infer_body_font_size",
    "SectionSpan",
    "assign_sections",
    "SECTION_KEYWORDS",
]


@dataclass(frozen=True)
class HeadingRule:
    """One heading pattern and the depth it implies."""

    name: str
    pattern: re.Pattern[str]
    #: Fixed level, or ``None`` to derive the level from the numbering depth.
    level: int | None
    #: Guard against absurd matches (a whole paragraph starting with "1.").
    max_length: int = 80


#: Ordered most-reliable first. A rule only fires if no earlier rule matched.
HEADING_RULES: tuple[HeadingRule, ...] = (
    HeadingRule("cn-part", re.compile(r"^第[一二三四五六七八九十百]+[篇部分编]"), 1),
    HeadingRule("cn-chapter", re.compile(r"^第[一二三四五六七八九十百]+章"), 1),
    HeadingRule("en-part", re.compile(r"^(?:PART|Part)\s+[IVXLC\d]+\b"), 1),
    HeadingRule("cn-section", re.compile(r"^第[一二三四五六七八九十百]+节"), 2),
    HeadingRule(
        "appendix",
        re.compile(r"^(?:附录|附注|Appendix|Annex|Schedule|Notes?\s+to\s+the)\b", re.IGNORECASE),
        2,
    ),
    HeadingRule("numbered", re.compile(r"^(\d+(?:\.\d+)*)[\s、．.]\s*\S"), None),
    HeadingRule("cn-paren", re.compile(r"^[（(]\s*[一二三四五六七八九十]+\s*[）)]"), 3),
    HeadingRule("cn-dun", re.compile(r"^[一二三四五六七八九十]+\s*[、．]\s*\S"), 2),
    HeadingRule("roman", re.compile(r"^[IVXLC]{1,5}[.、]\s+\S"), 2),
    HeadingRule(
        "allcaps",
        re.compile(r"^[A-Z][A-Z0-9 ,&/'\-()]{4,70}$"),
        2,
    ),
)

#: Phrases that mark a chunk as a financial-summary block. These are the chunks
#: whose numbers a reader will quote, so they get a lower HITL threshold.
SECTION_KEYWORDS: tuple[str, ...] = (
    "财务摘要", "主要财务数据", "主要会计数据", "财务概要", "五年财务", "经营业绩",
    "综合损益表", "合并损益表", "合并资产负债表", "合并现金流量表", "财务状况表",
    "每股收益", "股息", "分部资料", "管理层讨论", "经营讨论与分析",
    "financial summary", "financial highlights", "key financial",
    "consolidated statement", "statement of financial position",
    "statement of profit or loss", "statement of cash flows",
    "management discussion and analysis", "earnings per share",
    "segment information", "five-year summary",
)


@dataclass
class HeadingHit:
    level: int
    title: str
    rule: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {"level": self.level, "title": self.title, "rule": self.rule, "confidence": self.confidence}


def infer_body_font_size(blocks: Sequence[TextBlock]) -> float:
    """Median font size of the long multi-line blocks -- i.e. body text.

    Using the median over blocks (rather than a configured constant) keeps this
    working when a different report uses a different base size.
    """
    sizes = sorted(b.font_size for b in blocks if b.font_size > 0 and len(b.text) > 60)
    if not sizes:
        sizes = sorted(b.font_size for b in blocks if b.font_size > 0)
    if not sizes:
        return 0.0
    return sizes[len(sizes) // 2]


def detect_heading(
    block: TextBlock,
    *,
    body_font_size: float = 0.0,
    allow_font_rule: bool = True,
) -> HeadingHit | None:
    """Classify a text block as a heading, or return ``None``.

    Rules are tried in order and the first match wins, so a numbered heading is
    never downgraded by the weaker all-caps heuristic.
    """
    if not isinstance(block, TextBlock):
        return None
    text = normalize_whitespace(block.text, keep_newlines=False)
    if not text or len(text) > 140:
        return None
    # A heading does not end in a full stop; a sentence does. This single check
    # removes most false positives from the numbering and all-caps rules.
    if text.endswith(("。", ".", "；", ";", "：", ":")) and len(text) > 24:
        return None

    for rule in HEADING_RULES:
        if len(text) > rule.max_length:
            continue
        if not rule.pattern.match(text):
            continue
        level = rule.level
        if level is None:
            match = rule.pattern.match(text)
            token = match.group(1) if match and match.groups() else "1"
            level = min(6, token.count(".") + 1)
        # A "numbered" match inside a long sentence with a trailing comma is a
        # list item, not a heading.
        if rule.name == "numbered" and "," in text[:12] and "." not in text[:12]:
            continue
        confidence = 1.0 if rule.name not in ("allcaps", "numbered") else 0.75
        return HeadingHit(level=level, title=text, rule=rule.name, confidence=confidence)

    if (
        allow_font_rule
        and body_font_size > 0
        and block.font_size > 0
        and block.font_size >= body_font_size * 1.25
        and len(text) <= 60
    ):
        return HeadingHit(level=2, title=text, rule="font-size", confidence=0.6)
    return None


@dataclass
class SectionSpan:
    """A run of blocks sharing one section path."""

    title_path: tuple[str, ...] = ()
    page_start: int = 0
    page_end: int = 0
    blocks: list[Block] = field(default_factory=list)
    headings: list[HeadingHit] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.title_path[-1] if self.title_path else ""

    def text(self) -> str:
        from .chunker import block_to_text

        return "\n\n".join(block_to_text(b) for b in self.blocks)


def assign_sections(items: Sequence[tuple[int, Block]]) -> list[SectionSpan]:
    """Split a document's ``(page, block)`` stream into heading-delimited spans.

    The stack discipline matters: a level-3 heading inside a level-2 section
    must produce path ``(L2 title, L3 title)``, and a following level-2 heading
    must pop back to ``(L2 title,)``. Getting this wrong silently corrupts every
    context string injected downstream, so it is tested directly.
    """
    spans: list[SectionSpan] = []
    stack: list[HeadingHit] = []
    current = SectionSpan()
    body_size = infer_body_font_size([b for _, b in items if isinstance(b, TextBlock)])

    def flush() -> None:
        nonlocal current
        if current.blocks:
            current.page_end = current.page_end or current.page_start
            spans.append(current)
        current = SectionSpan(title_path=tuple(h.title for h in stack))

    for page, block in items:
        hit: HeadingHit | None = None
        if isinstance(block, TextBlock):
            if block.heading_level is not None:
                # The parser already labelled this block a heading; trust it and
                # keep the label visible in the provenance trail.
                hit = HeadingHit(
                    level=block.heading_level,
                    title=normalize_whitespace(block.text, keep_newlines=False),
                    rule="prelabelled",
                )
            else:
                hit = detect_heading(block, body_font_size=body_size)

        if hit is not None:
            flush()
            while stack and stack[-1].level >= hit.level:
                stack.pop()
            stack.append(hit)
            current = SectionSpan(title_path=tuple(h.title for h in stack), page_start=page, page_end=page)
            current.headings.append(hit)
            current.blocks.append(block)
            continue

        if not current.blocks:
            current.page_start = page
        current.page_end = page
        current.blocks.append(block)

    flush()
    return spans


def is_financial_section(title_path: Iterable[str], *, extra_keywords: Sequence[str] = ()) -> bool:
    """True when any level of the section path names a financial-summary area."""
    haystack = " ".join(title_path).lower()
    if not haystack:
        return False
    keywords = tuple(SECTION_KEYWORDS) + tuple(extra_keywords)
    return any(keyword.lower() in haystack for keyword in keywords)


def heading_digest(spans: Sequence[SectionSpan], *, max_depth: int = 2) -> list[str]:
    """Compact outline, used in the run report and the README screenshot."""
    seen: list[str] = []
    for span in spans:
        path = span.title_path[:max_depth]
        line = " > ".join(path)
        if line and line not in seen:
            seen.append(line)
    return seen


def iter_blocks_with_pages(document: object) -> Iterator[tuple[int, Block]]:
    """Small adapter so callers do not need the SDO type for traversal."""
    yield from document.iter_blocks()  # type: ignore[attr-defined]


def table_page_map(spans: Sequence[SectionSpan]) -> dict[int, tuple[str, ...]]:
    """Map ``id(TableBlock)`` -> section path, for the number guard's context."""
    out: dict[int, tuple[str, ...]] = {}
    for span in spans:
        for block in span.blocks:
            if isinstance(block, TableBlock):
                out[id(block)] = span.title_path
    return out
