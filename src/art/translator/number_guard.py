"""Number-hallucination guard.

The claim under test
--------------------
"Translation cannot change a financial figure." Every other quality property of
a machine-translated annual report is a matter of taste; this one is a legal
and audit matter. So it gets the strictest treatment in the codebase, and the
whole check is deterministic -- no model judges its own output.

How it works, and why not the obvious way
-----------------------------------------
The obvious implementation is ``assert str(number) in translated_text``. It
fails on the three things that actually happen:

1. **Unit conversion.** ``1,234 千元`` and ``1.234 百万元`` are the same amount.
   A string comparison calls one of them wrong. So every figure is normalised to
   a base unit (yuan / count / unit-ratio) and compared as a *number*.
2. **Scale loss.** A table declares ``单位：人民币千元`` once in its header. If the
   translation drops that header, every bare figure in the table changes
   meaning by 1000x while looking identical. So the unit hint is parsed and
   threaded into extraction, and a missing hint is itself a finding.
3. **Near-miss corruption.** The realistic hallucination is a digit swap:
   ``1,234,567`` -> ``1,274,567``. A guard that only checks presence never sees
   it. So unmatched figures are paired against each other by digit distance and
   reported as mismatches, not as one missing plus one added.

Output is a :class:`NumberDiffReport` with matched / missing / added /
mismatched buckets and a ``drift`` ratio. That ratio is what the HITL policy
scores on -- the guard does not decide, it produces evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..schema import TableBlock
from ..textutils import normalize_whitespace, normalize_width

__all__ = [
    "NumberOccurrence",
    "NumberDiffReport",
    "NumberGuard",
    "extract_numbers",
    "cn_to_int",
    "parse_unit",
    "UNIT_TABLE",
]


# ---------------------------------------------------------------------------
# dimensions and units
# ---------------------------------------------------------------------------

MONEY = "money"
COUNT = "count"
RATIO = "ratio"
UNKNOWN = "unknown"

#: Canonical unit -> (dimension, multiplier to base unit).
#: Base units: money -> yuan, count -> 1, ratio -> fraction (0.15 = 15%).
UNIT_TABLE: dict[str, tuple[str, float]] = {
    # --- Chinese currency scales ---
    "元": (MONEY, 1.0),
    "人民币元": (MONEY, 1.0),
    "千元": (MONEY, 1e3),
    "万元": (MONEY, 1e4),
    "百万元": (MONEY, 1e6),
    "亿元": (MONEY, 1e8),
    "万": (MONEY, 1e4),
    "亿": (MONEY, 1e8),
    # --- English scales ---
    "thousand": (MONEY, 1e3),
    "thousands": (MONEY, 1e3),
    "k": (MONEY, 1e3),
    "'000": (MONEY, 1e3),
    "mn": (MONEY, 1e6),
    "million": (MONEY, 1e6),
    "millions": (MONEY, 1e6),
    "bn": (MONEY, 1e9),
    "billion": (MONEY, 1e9),
    "trillion": (MONEY, 1e12),
    # --- ratios ---
    "%": (RATIO, 0.01),
    "percent": (RATIO, 0.01),
    "percentage point": (RATIO, 0.01),
    "percentage points": (RATIO, 0.01),
    "个百分点": (RATIO, 0.01),
    "bp": (RATIO, 1e-4),
    "bps": (RATIO, 1e-4),
    "基点": (RATIO, 1e-4),
    # --- counts ---
    "股": (COUNT, 1.0),
    "shares": (COUNT, 1.0),
    "share": (COUNT, 1.0),
    "个": (COUNT, 1.0),
    "名": (COUNT, 1.0),
    "人": (COUNT, 1.0),
    "家": (COUNT, 1.0),
}

#: Currency markers that carry a dimension but no multiplier of their own.
#: Checked longest-first so ``人民币`` wins over ``元``.
CURRENCY_MARKERS = (
    "人民币", "港元", "美元", "欧元",
    "rmb", "cny", "hkd", "usd", "hk$", "us$", "eur",
    "¥", "￥", "$", "€", "£", "元",
)

#: Longest-first so "百万元" is tried before "万元" and "万元" before "万".
_UNIT_KEYS = sorted(UNIT_TABLE, key=len, reverse=True)

#: Single ASCII letters used as scale words ("5k"). They need a word boundary,
#: otherwise "5 km" would be read as five thousand of something.
_SINGLE_LETTER_UNITS = frozenset("k")

_NUMBER_RE = re.compile(
    r"""
    (?P<paren>\()?                       # opening paren: (1,234) means -1234
    (?P<sign>[-+\u2212])?                # ASCII minus, or U+2212 minus sign
    (?P<value>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?     # 1,234,567.89
      | \d+\.\d+                         # 12.34
      | \.\d+                            # .5
      | \d+                              # 1234
    )
    \s*(?P<close>\))?
    """,
    re.VERBOSE,
)

_CN_NUMERAL_RE = re.compile(r"[零一二三四五六七八九十百千万亿两]+")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_SMALL = {"十": 10, "百": 100, "千": 1000}
_CN_BIG = {"万": 10000, "亿": 100000000}

#: A bare Chinese numeral only counts as a figure when it carries a scale word
#: or is followed by a unit. Otherwise section markers like "一、" and "（二）"
#: would be extracted as the number 1 and 2, producing pure noise findings.
_CN_UNIT_FOLLOWERS = ("元", "万", "亿", "千", "百", "%", "％", "个", "股", "人", "家", "名", "倍")

_WINDOW_BEFORE = 10
_WINDOW_AFTER = 12


def cn_to_int(text: str) -> int | None:
    """Convert a Chinese numeral (``一千二百三十四``) to an int."""
    text = text.strip()
    if not text or any(ch not in _CN_DIGITS and ch not in _CN_SMALL and ch not in _CN_BIG for ch in text):
        return None
    total = 0
    section = 0
    number = 0
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_SMALL:
            section += (number or 1) * _CN_SMALL[ch]
            number = 0
        elif ch in _CN_BIG:
            unit = _CN_BIG[ch]
            section = (section + number) * unit
            if unit == _CN_BIG["亿"]:
                total += section
                section = 0
            number = 0
    result = total + section + number
    return result if result else None


# ---------------------------------------------------------------------------
# occurrence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberOccurrence:
    """One figure found in text, normalised into a comparable form."""

    raw: str
    value: float
    #: money | count | ratio | unknown
    dimension: str
    #: Canonical unit label, e.g. ``"千元"``, ``"million"``, ``"%"``, ``""``.
    unit: str
    #: Multiplier applied to get from ``value`` to ``base_value``.
    multiplier: float
    span: tuple[int, int]
    context: str = ""
    source: str = ""
    #: True when the multiplier came from the table's unit note rather than
    #: from the figure's own suffix -- i.e. this number's meaning depends on a
    #: header that the translation could have dropped.
    unit_inherited: bool = False

    @property
    def base_value(self) -> float:
        """Value in base units (yuan / shares / unit-ratio)."""
        return self.value * self.multiplier

    @property
    def is_integer(self) -> bool:
        return abs(self.base_value - round(self.base_value)) < 1e-9

    @property
    def digits(self) -> str:
        """Digit characters only, for near-miss comparison."""
        return re.sub(r"\D", "", self.raw)

    def key(self, precision: int = 6) -> tuple[str, float]:
        """Comparison key: dimension plus the base value at ``precision``."""
        return (self.dimension, round(self.base_value, precision) if self.dimension == RATIO else round(self.base_value, 4))

    def describe(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.raw}{unit} -> {self.base_value:,.4f} ({self.dimension})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "value": self.value,
            "dimension": self.dimension,
            "unit": self.unit,
            "multiplier": self.multiplier,
            "base_value": self.base_value,
            "context": self.context,
            "unit_inherited": self.unit_inherited,
        }


def parse_unit(text: str) -> tuple[str, float, str] | None:
    """Longest-match a unit token, returning ``(dimension, multiplier, label)``."""
    lowered = normalize_width(text).lower()
    for key in _UNIT_KEYS:
        if key.lower() in lowered:
            dimension, multiplier = UNIT_TABLE[key]
            return dimension, multiplier, key
    for marker in CURRENCY_MARKERS:
        if marker in lowered:
            return MONEY, 1.0, marker
    return None


def unit_from_note(note: str) -> tuple[str, float, str] | None:
    """Parse a table's unit declaration (``单位：人民币千元`` / ``RMB'000``)."""
    if not note:
        return None
    cleaned = normalize_width(note)
    # "RMB'000" has no unit word -- the "'000" is the scale.
    if re.search(r"'\s*000", cleaned) or re.search(r"\b000\b", cleaned):
        return MONEY, 1e3, "thousand"
    if re.search(r"\bmillion\b", cleaned, re.I):
        return MONEY, 1e6, "million"
    return parse_unit(cleaned)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def extract_numbers(text: str, *, unit_hint: str = "", source: str = "") -> list[NumberOccurrence]:
    """Extract every figure from ``text``, normalising units.

    ``unit_hint`` is the enclosing table's unit note. Figures with no unit of
    their own inherit it (see ``NumberOccurrence.unit_inherited``), which is the
    mechanism that catches a dropped unit header.
    """
    if not text:
        return []
    normalized = normalize_width(text)
    hint = unit_from_note(unit_hint) if unit_hint else None

    occurrences: list[NumberOccurrence] = []
    consumed: list[tuple[int, int]] = []

    for match in _NUMBER_RE.finditer(normalized):
        raw_value = match.group("value")

        # Skip thousands-scale markers: in "RMB'000" the 000 is a unit, not a
        # figure. The apostrophe is the tell, and it is the only reliable one --
        # there is no way to distinguish it from the digits alone.
        if match.start() > 0 and normalized[match.start() - 1] in ("'", "\u2019", "\u2032"):
            continue

        sign = match.group("sign")
        negative = bool(sign == "-" or sign == "\u2212")
        if match.group("paren") and match.group("close"):
            negative = True

        try:
            value = float(raw_value.replace(",", ""))
        except ValueError:
            continue
        if negative:
            value = -value

        after = normalized[match.end() : match.end() + _WINDOW_AFTER]
        before = normalized[max(0, match.start() - _WINDOW_BEFORE) : match.start()]
        merged = f"{before}|{after}"

        dimension, multiplier, label, inherited = _resolve_unit(
            after, merged, hint, value, is_decimal="." in raw_value
        )

        occurrences.append(
            NumberOccurrence(
                raw=match.group(0).strip(),
                value=value,
                dimension=dimension,
                unit=label,
                multiplier=multiplier,
                span=(match.start(), match.end()),
                context=normalize_whitespace(
                    normalized[max(0, match.start() - 14) : match.end() + 14], keep_newlines=False
                ),
                source=source,
                unit_inherited=inherited,
            )
        )
        consumed.append((match.start(), match.end()))

    occurrences.extend(_extract_chinese_numerals(normalized, consumed, hint, source))
    occurrences.sort(key=lambda o: o.span)
    return occurrences


def _match_leading_unit(text: str) -> tuple[str, float, str] | None:
    """Longest unit token at the very start of ``text``.

    Uses the unit table uniformly, so a Chinese count unit (``股``) and an
    English scale word (``million``) are handled by the same code path.
    """
    trimmed = text.lstrip()
    if not trimmed:
        return None
    for key in _UNIT_KEYS:
        if not trimmed.startswith(key):
            continue
        if key in _SINGLE_LETTER_UNITS and len(trimmed) > len(key) and trimmed[len(key)].isalnum():
            continue
        dimension, multiplier = UNIT_TABLE[key]
        return dimension, multiplier, key
    return None


def _resolve_unit(
    after: str,
    merged: str,
    hint: tuple[str, float, str] | None,
    value: float,
    is_decimal: bool = False,
) -> tuple[str, float, str, bool]:
    """Decide a figure's dimension/multiplier from its suffix, then the hint."""
    # 1. Percentage is the strongest signal and is often glued on: "12.4%".
    if re.match(r"\s*(%|％|percent|per cent)", after, re.I):
        return RATIO, 0.01, "%", False
    if re.match(r"\s*(bp|bps|基点|个基点)\b", after, re.I):
        return RATIO, 1e-4, "bp", False
    # "百分之12.4" -- the marker sits *before* the figure.
    if re.search(r"百分之\s*$", merged.split("|", 1)[0][-8:]):
        return RATIO, 0.01, "%", False

    # 2. An explicit scale/unit word right after the figure:
    #    "1,234 千元", "5 million", "3,000 股".
    parsed = _match_leading_unit(after)
    if parsed is not None:
        dimension, multiplier, label = parsed
        return dimension, multiplier, label, False

    # 3. A currency marker before the figure carries the dimension at scale 1,
    #    upgraded to the table's scale when the table declares one.
    tail = merged.split("|", 1)[0][-_WINDOW_BEFORE:]
    if any(marker in tail.lower() for marker in CURRENCY_MARKERS):
        if hint and hint[0] == MONEY and hint[1] > 1:
            return MONEY, hint[1], hint[2], True
        return MONEY, 1.0, "", False

    # 4. No local signal: inherit the table's unit note. This is the link that
    #    a dropped unit header breaks, so the flag matters.
    if hint is not None:
        return hint[0], hint[1], hint[2], True

    # 5. A decimal with no unit is most likely a ratio (EPS, per-share, index).
    #    Tested on the printed token, not on the float: ``float(5)`` formats as
    #    "5.0" and would mislabel every small integer as a ratio.
    dimension = RATIO if (is_decimal and abs(value) < 1000) else UNKNOWN
    return dimension, 1.0, "", False


def _extract_chinese_numerals(
    text: str,
    consumed: Sequence[tuple[int, int]],
    hint: tuple[str, float, str] | None,
    source: str,
) -> list[NumberOccurrence]:
    """Extract Chinese numerals that a digit regex would miss.

    Two filters keep this from becoming a noise generator:

    * A numeral only counts as a figure when it carries a scale word
      (十/百/千/万/亿) or is followed by a unit. Without that, every list marker
      ("一、", "（二）") becomes a spurious finding.
    * A numeral that merges with the following characters to form a *unit* is
      skipped. ``1.234 百万元`` contains the numeral-looking ``百万``; treating
      it as a figure invents a second, nonexistent value of 1,000,000. The same
      trap hides in ``百分之``.
    """
    out: list[NumberOccurrence] = []
    for match in _CN_NUMERAL_RE.finditer(text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        literal = match.group(0)
        after = text[match.end() : match.end() + 6]

        # "百万" + "元" == the unit 百万元, and "百" + "分之" is a percentage
        # marker. Neither is a figure.
        if (literal + after[:1]) in UNIT_TABLE or any(
            (literal + after[:n]) in UNIT_TABLE for n in range(1, 4)
        ):
            continue
        if after.startswith(("分之", "分")):
            continue

        has_scale = any(ch in _CN_SMALL or ch in _CN_BIG for ch in literal)
        has_follower = after.startswith(_CN_UNIT_FOLLOWERS) or bool(
            re.match(r"\s*(元|万|亿|个|股|人|家|名|倍|%)", after)
        )
        if not (has_scale or has_follower):
            continue

        value = cn_to_int(literal)
        if value is None:
            continue
        dimension, multiplier, label, inherited = _resolve_unit(after, f"|{after}", hint, float(value))
        out.append(
            NumberOccurrence(
                raw=literal,
                value=float(value),
                dimension=dimension,
                unit=label,
                multiplier=multiplier,
                span=(match.start(), match.end()),
                context=normalize_whitespace(
                    text[max(0, match.start() - 10) : match.end() + 10], keep_newlines=False
                ),
                source=source,
                unit_inherited=inherited,
            )
        )
    return out


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


@dataclass
class NumberDiffReport:
    """Evidence, not a verdict. The HITL policy turns this into a decision."""

    matched: int = 0
    missing: list[NumberOccurrence] = field(default_factory=list)
    added: list[NumberOccurrence] = field(default_factory=list)
    mismatched: list[tuple[NumberOccurrence, NumberOccurrence, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_total: int = 0
    target_total: int = 0
    structural: list[str] = field(default_factory=list)

    @property
    def drift(self) -> float:
        """Fraction of source figures that did not survive intact, in [0, 1]."""
        if self.source_total == 0:
            return 0.0
        bad = len(self.missing) + len(self.mismatched)
        return min(1.0, bad / self.source_total)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.mismatched or self.structural)

    @property
    def has_findings(self) -> bool:
        return bool(self.missing or self.added or self.mismatched or self.structural)

    def summary(self) -> str:
        bits = [
            f"source={self.source_total}",
            f"target={self.target_total}",
            f"matched={self.matched}",
        ]
        if self.missing:
            bits.append(f"missing={len(self.missing)}")
        if self.added:
            bits.append(f"added={len(self.added)}")
        if self.mismatched:
            bits.append(f"mismatched={len(self.mismatched)}")
        if self.structural:
            bits.append(f"structural={len(self.structural)}")
        bits.append(f"drift={self.drift:.3f}")
        return " ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "drift": round(self.drift, 4),
            "source_total": self.source_total,
            "target_total": self.target_total,
            "matched": self.matched,
            "missing": [o.to_dict() for o in self.missing],
            "added": [o.to_dict() for o in self.added],
            "mismatched": [
                {"source": s.to_dict(), "target": t.to_dict(), "reason": reason}
                for s, t, reason in self.mismatched
            ],
            "structural": list(self.structural),
            "notes": list(self.notes),
            "summary": self.summary(),
        }


#: A digit-level difference at or below this many characters is treated as a
#: transcription slip rather than an independent figure.
_DIGIT_EDIT_LIMIT = 2


def _digit_distance(a: str, b: str) -> int:
    """Levenshtein distance over digit strings, with an early exit."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > _DIGIT_EDIT_LIMIT:
        return _DIGIT_EDIT_LIMIT + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


class NumberGuard:
    """Deterministic number-integrity checker.

    Parameters
    ----------
    rel_tol:
        Relative tolerance for calling two figures equal. Default ``0.0`` --
        annual-report figures are printed exactly, and a tolerance here would
        be a hole through which real corruption escapes.
    near_tol:
        Relative distance within which an unmatched pair is reported as a
        *mismatch* (digit slip) rather than as separate missing/added entries.
        2% covers the observed hallucination mode while staying far below any
        genuine restatement.
    """

    def __init__(self, *, rel_tol: float = 0.0, near_tol: float = 0.02, precision: int = 6) -> None:
        self.rel_tol = rel_tol
        self.near_tol = near_tol
        self.precision = precision

    # -- text vs text -------------------------------------------------------

    def check_text(
        self, source_text: str, target_text: str, *, unit_hint: str = "", label: str = ""
    ) -> NumberDiffReport:
        source = extract_numbers(source_text, unit_hint=unit_hint, source="source")
        target = extract_numbers(target_text, unit_hint=unit_hint, source="target")
        report = _diff(source, target, self.rel_tol, self.near_tol)
        if label:
            report.notes.append(f"compared: {label}")
        # A unit note present in the source but absent from the target silently
        # rescales every bare figure, so it is a structural finding in its own
        # right, independent of whether the digits match.
        if (
            unit_hint
            and unit_hint.strip()
            and unit_from_note(unit_hint)
            and not unit_from_note(target_text[:200])
            and not _mentions_unit_note(target_text, unit_hint)
        ):
            report.structural.append(
                f"unit note {unit_hint!r} is present in the source but not identifiable in the target; "
                f"all bare figures in this block may be rescaled"
            )
        return report

    # -- table vs table -----------------------------------------------------

    def check_table_pair(self, source: TableBlock, target: TableBlock) -> NumberDiffReport:
        """Compare two tables position by position.

        Positional comparison is the reason the parser keeps ``(row, col)``
        indices: a whole-table text diff would not tell you *which* cell moved,
        and would produce a wall of false findings the moment the translator
        legitimately reorders nothing but rewraps a label.
        """
        report = NumberDiffReport()
        source_map = source.by_position()
        target_map = target.by_position()

        if (source.n_rows, source.n_cols) != (target.n_rows, target.n_cols):
            report.structural.append(
                f"shape changed {source.n_rows}x{source.n_cols} -> {target.n_rows}x{target.n_cols}"
            )
        source_origins = set(source_map)
        target_origins = set(target_map)
        for missing_slot in sorted(source_origins - target_origins):
            report.structural.append(f"cell {missing_slot} exists in the source but not the target")
        for extra_slot in sorted(target_origins - source_origins):
            report.structural.append(f"cell {extra_slot} exists in the target but not the source")
        if (source.units_note or "").strip() and not (target.units_note or "").strip():
            report.structural.append(
                f"table unit note {source.units_note!r} was dropped in the target"
            )

        for slot in sorted(source_origins & target_origins):
            src_cell = source_map[slot]
            tgt_cell = target_map[slot]
            src_nums = extract_numbers(src_cell.text, unit_hint=source.units_note, source=f"table:{slot}")
            tgt_nums = extract_numbers(tgt_cell.text, unit_hint=target.units_note or source.units_note,
                                       source=f"table:{slot}")
            cell_report = _diff(src_nums, tgt_nums, self.rel_tol, self.near_tol)
            report.matched += cell_report.matched
            report.source_total += cell_report.source_total
            report.target_total += cell_report.target_total
            for occurrence in cell_report.missing:
                report.missing.append(occurrence)
            for occurrence in cell_report.added:
                report.added.append(occurrence)
            for pair in cell_report.mismatched:
                report.mismatched.append((pair[0], pair[1], f"{pair[2]} at {slot}"))

        if source.units_note:
            report.notes.append(f"units: {source.units_note}")
        return report


def _mentions_unit_note(target_text: str, note: str) -> bool:
    """Loose check that a unit declaration survived translation."""
    if not note:
        return True
    if note.strip() in target_text:
        return True
    parsed = unit_from_note(note)
    if parsed is None:
        return True
    _, multiplier, _ = parsed
    # Accept an equivalent declaration in the target language.
    equivalents = {
        1e3: ("千元", "千", "thousand", "'000", "000"),
        1e4: ("万元", "万",),
        1e6: ("百万元", "million", "mn"),
        1e8: ("亿元", "亿",),
        1.0: ("元", "yuan", "rmb"),
    }.get(multiplier, ())
    return any(eq in target_text for eq in equivalents)


def _diff(
    source: list[NumberOccurrence],
    target: list[NumberOccurrence],
    rel_tol: float,
    near_tol: float,
) -> NumberDiffReport:
    """Multiset comparison with near-miss pairing."""
    report = NumberDiffReport(source_total=len(source), target_total=len(target))

    used_target = [False] * len(target)
    remaining_source: list[NumberOccurrence] = []

    # Pass 1: exact matches (by normalised base value).
    for occurrence in source:
        found = -1
        for index, candidate in enumerate(target):
            if used_target[index]:
                continue
            if _equal(occurrence, candidate, rel_tol):
                found = index
                break
        if found >= 0:
            used_target[found] = True
            report.matched += 1
        else:
            remaining_source.append(occurrence)

    # Pass 2: pair leftovers into near-miss mismatches, closest first.
    #
    # The pairing criterion must be **digit distance, not value distance**.
    # Changing one digit of ``1,234,567`` to ``1,934,567`` moves the value by
    # 57%, so a "within 2%" value test -- the intuitive choice -- misses the
    # single most common hallucination entirely. Digit edit distance sees it
    # immediately. The value test is kept as a second channel for cases where
    # the notation changed without the digits doing so.
    #
    # Reporting these as one *mismatch* rather than one missing + one added is
    # what makes the report actionable: the reviewer is shown exactly which
    # figure became what.
    candidates: list[tuple[int, float, int, int]] = []
    for si, src in enumerate(remaining_source):
        for ti, tgt in enumerate(target):
            if used_target[ti] or src.dimension != tgt.dimension:
                continue
            if len(src.digits) != len(tgt.digits):
                continue
            distance = _digit_distance(src.digits, tgt.digits)
            if src.base_value == 0:
                relative = float("inf") if tgt.base_value else 0.0
            else:
                relative = abs(tgt.base_value - src.base_value) / abs(src.base_value)
            if distance <= _DIGIT_EDIT_LIMIT or relative <= near_tol:
                candidates.append((distance, relative, si, ti))
    candidates.sort()

    paired_source: set[int] = set()
    for distance, relative, si, ti in candidates:
        if si in paired_source or used_target[ti]:
            continue
        paired_source.add(si)
        used_target[ti] = True
        reason = (
            f"digit slip (edit distance {distance})"
            if distance <= _DIGIT_EDIT_LIMIT
            else f"value changed by {relative * 100:.3f}%"
        )
        report.mismatched.append((remaining_source[si], target[ti], reason))

    for si, src in enumerate(remaining_source):
        if si not in paired_source:
            report.missing.append(src)
    for ti, tgt in enumerate(target):
        if not used_target[ti]:
            report.added.append(tgt)

    # A pure reordering or a duplicate is worth calling out separately: the
    # counts match but the multiplicities do not.
    if not report.missing and not report.mismatched and report.added:
        report.notes.append("counts differ only by extra figures in the target; check for duplicated values")

    return report


def _equal(a: NumberOccurrence, b: NumberOccurrence, rel_tol: float) -> bool:
    if a.dimension != b.dimension and UNKNOWN not in (a.dimension, b.dimension):
        # An unlabelled figure on either side may legitimately be the same
        # number in a different dimension -- compare on value alone, and let the
        # dimension mismatch show up as a note rather than a hard finding.
        return False
    if a.base_value == b.base_value:
        return True
    if rel_tol <= 0:
        return False
    scale = max(abs(a.base_value), abs(b.base_value))
    return scale > 0 and abs(a.base_value - b.base_value) / scale <= rel_tol


# ---------------------------------------------------------------------------
# document-level convenience
# ---------------------------------------------------------------------------


def fingerprint_numbers(text: str) -> tuple[tuple[str, float], ...]:
    """Order-insensitive fingerprint of every figure in ``text``."""
    return tuple(sorted(o.key() for o in extract_numbers(text)))


def compare_documents(
    guard: NumberGuard, source_text: str, target_text: str, *, unit_hint: str = ""
) -> NumberDiffReport:
    return guard.check_text(source_text, target_text, unit_hint=unit_hint)


def collect_chart_numbers(charts: Iterable[Any]) -> list[NumberOccurrence]:
    """Extract figures from chart data points so charts join the same guard."""
    out: list[NumberOccurrence] = []
    for chart in charts:
        for label, value in getattr(chart, "points", []):
            out.append(
                NumberOccurrence(
                    raw=f"{value:g}",
                    value=float(value),
                    dimension=UNKNOWN,
                    unit="",
                    multiplier=1.0,
                    span=(0, 0),
                    context=str(label),
                    source="chart",
                )
            )
    return out
