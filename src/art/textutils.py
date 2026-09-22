"""Small, dependency-free text helpers shared across all stages.

Deliberately in one place: full-width/half-width normalisation and token
estimation are used by the parser (cell text), the chunker (budgets) and the
number guard (figure comparison). If they diverge, numbers silently stop
matching -- so there is exactly one implementation.
"""

from __future__ import annotations

import html
import re
import unicodedata

__all__ = [
    "normalize_width",
    "normalize_whitespace",
    "escape_html",
    "unescape_html",
    "cjk_ratio",
    "estimate_tokens",
    "slugify",
    "truncate",
    "looks_numeric",
]

# Full-width ASCII (U+FF01..U+FF5E) maps onto ASCII by subtracting 0xFEE0.
# Ideographic space U+3000 becomes a normal space.
_FW_OFFSET = 0xFEE0
_FW_START = 0xFF01
_FW_END = 0xFF5E

_WS_RE = re.compile(r"[ \t\u00a0\u3000]+")
#: Any whitespace run, used when newlines are being flattened. ``_WS_RE``
#: deliberately excludes ``\n`` (so the keep-newlines branch can see line
#: structure), which means flattening must re-collapse or "a\n\n b" comes out as
#: "a   b" -- a spacing artefact that then becomes a dict key in
#: ``table_to_records`` and a record lookup that silently misses.
_ALL_WS_RE = re.compile(r"\s+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff]+")


def normalize_width(text: str) -> str:
    """Fold full-width forms to half-width.

    Annual reports produced by Chinese typesetting tools routinely mix
    ``１２３`` / ``９．５％`` with ASCII ``123`` / ``9.5%``. Without this fold the
    number guard would treat the same figure as two different values.
    """
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if _FW_START <= code <= _FW_END:
            out.append(chr(code - _FW_OFFSET))
        elif code == 0x3000:  # ideographic space
            out.append(" ")
        elif ch == "\u00a5":  # ¥
            out.append("¥")
        else:
            out.append(ch)
    return "".join(out)


def normalize_whitespace(text: str, keep_newlines: bool = True) -> str:
    """Collapse runs of spaces; optionally collapse 3+ newlines to a blank line."""
    if not text:
        return ""
    text = normalize_width(text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _WS_RE.sub(" ", text)
    if keep_newlines:
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _MULTI_NL_RE.sub("\n\n", text)
    else:
        text = _ALL_WS_RE.sub(" ", text)
    return text.strip()


def escape_html(text: str) -> str:
    """Escape text for HTML emission (quotes included, so attributes are safe)."""
    return html.escape(text or "", quote=True)


def unescape_html(text: str) -> str:
    return html.unescape(text or "")


def cjk_ratio(text: str) -> float:
    """Fraction of characters that are CJK ideographs (0.0 for empty input)."""
    if not text:
        return 0.0
    total = 0
    cjk = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        code = ord(ch)
        if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
            cjk += 1
    return cjk / total if total else 0.0


def estimate_tokens(text: str) -> int:
    """Rough token estimate, good enough for budgeting.

    Heuristic, and *documented as a heuristic*: CJK is roughly 1 token per
    1.5 characters for current BPE tokenizers, Latin roughly 1 token per 4
    characters. It is used only to decide where to break a chunk, so being
    off by 20% is harmless -- and unlike a real tokenizer it costs nothing
    and never fails at import time. Swap in a real tokenizer by overriding
    ``chunker.ChunkOptions.token_counter``.
    """
    if not text:
        return 0
    text = normalize_width(text)
    cjk = sum(
        1
        for ch in text
        if 0x3400 <= ord(ch) <= 0x4DBF or 0x4E00 <= ord(ch) <= 0x9FFF or 0xF900 <= ord(ch) <= 0xFAFF
    )
    other = len(text) - cjk
    digits = sum(1 for ch in text if ch.isdigit() or ch in ",.%")
    # Digits tokenize badly: a 12-digit figure can be 4-6 tokens on its own.
    return int(cjk / 1.5 + other / 4.0 + digits * 0.35) + 1


def slugify(text: str, max_len: int = 60) -> str:
    """Filesystem/URL-safe slug that keeps CJK (readable in Chinese repos)."""
    text = normalize_whitespace(text or "", keep_newlines=False)
    slug = _SLUG_RE.sub("-", text).strip("-").lower()
    return slug[:max_len] or "unnamed"


def truncate(text: str, limit: int = 80, suffix: str = "...") -> str:
    text = normalize_whitespace(text or "", keep_newlines=False)
    return text if len(text) <= limit else text[: limit - len(suffix)] + suffix


#: A cell that holds nothing but a figure. The prefix class handles a sign or an
#: opening paren; the suffix class must *also* contain the paren, because
#: ``(1,234)`` has one on each side and a prefix-only class cannot match the
#: trailing one. Getting this wrong means a row of parenthesised negatives is
#: not recognised as data, so it gets labelled as a header (see
#: ``table_builder._finalise_labels``).
_NUMERIC_RE = re.compile(r"^[\s\-+()]*[\d,.\u2014\u2013%-]+[\s%()]*$")


def looks_numeric(text: str) -> bool:
    """True if a cell holds only a figure (possibly parenthesised negative).

    Used for header detection: an annual-report header row contains labels,
    a data row contains at least one figure. Parenthesised negatives
    ``(1,234)`` and dashes meaning nil are counted as numeric on purpose.
    """
    t = normalize_width(text or "").strip()
    if not t:
        return False
    if t in {"-", "—", "–", "/", "N/A", "n/a", "不适用", "无"}:
        return True
    if not any(ch.isdigit() for ch in t):
        return False
    return bool(_NUMERIC_RE.match(t))


def display_width(text: str) -> int:
    """Terminal display width (CJK counts as 2 columns)."""
    width = 0
    for ch in normalize_width(text or ""):
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width
