"""Terminology extraction and glossary management.

The consistency problem
-----------------------
``归母净利润`` must render the same way on page 6 and page 141. Left to a
language model per-chunk, it will not: you get "net profit attributable to
shareholders", "net profit attributable to the parent", and "profit
attributable to equity holders of the parent" within one document. In a legal
document that is a defect.

The extraction trick
--------------------
Annual reports gloss themselves. A bilingual report writes
``EBITDA（息税折旧摊销前利润）`` or ``归属于母公司股东的净利润（"归母净利润"）``.
Those parenthetical pairs are *authoritative* human translations supplied by
the issuer, and they are free -- no model call needed. Extracting them is the
highest-value, lowest-cost step in the whole glossary pipeline, so it runs
first and its pairs are marked ``authority="source_gloss"``.

Layered authority
-----------------
Terms carry where they came from, because conflicts must be resolved by
precedence rather than by whoever wrote last:

    1. ``source_gloss``  -- the issuer's own bilingual pair. Never overridden.
    2. ``curated``       -- a human-edited entry in terms.json.
    3. ``seed``          -- the built-in financial lexicon.
    4. ``model``         -- proposed by an LLM pass.

Note the order of 3 and 4: the model is *below* the built-in lexicon on purpose.
The seed is a curated human term list; the model is what the glossary exists to
constrain. Letting a model proposal override the seed would make the glossary
constrain nothing -- every run could silently re-decide ``Revenue`` -- which
would defeat the consistency guarantee the glossary is here to provide.

A conflict between two entries of the same precedence is surfaced as a
:class:`GlossaryConflict` for human decision rather than silently resolved.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..textutils import cjk_ratio, normalize_whitespace

__all__ = [
    "SEED_TERMS",
    "GlossaryEntry",
    "GlossaryConflict",
    "GlossaryStore",
    "extract_bilingual_pairs",
    "extract_candidates",
    "ACRONYM_RE",
]

#: Authority ordering, highest first.
#:
#: The model sits *below* the seed lexicon on purpose. The seed is a curated
#: human term list; the model is what the glossary exists to constrain. If a
#: model proposal could overwrite the seed, the glossary would provide no
#: constraint at all -- every run would silently re-decide `Revenue` and the
#: "consistency" guarantee would be self-defeating. Model entries can still
#: *fill gaps* (new terms absent from the seed), which is what they are for.
AUTHORITY_RANK: dict[str, int] = {
    "source_gloss": 0,
    "curated": 1,
    "seed": 2,
    "model": 3,
}


#: A compact but genuinely useful EN<->ZH lexicon for financial reporting. It is
#: the *lowest* authority layer: any pair the report itself supplies wins.
SEED_TERMS: dict[str, str] = {
    # income statement
    "Revenue": "营业收入",
    "Turnover": "营业额",
    "Cost of sales": "销售成本",
    "Gross profit": "毛利",
    "Operating expenses": "经营费用",
    "Operating profit": "营业利润",
    "Finance costs": "财务费用",
    "Profit before tax": "税前利润",
    "Income tax expense": "所得税费用",
    "Profit for the year": "年度利润",
    "Profit for the period": "期内利润",
    "Net profit": "净利润",
    "EBITDA": "息税折旧摊销前利润",
    "EBIT": "息税前利润",
    "Earnings per share": "每股收益",
    "Basic earnings per share": "基本每股收益",
    "Diluted earnings per share": "稀释每股收益",
    "Dividend per share": "每股股息",
    # balance sheet
    "Total assets": "资产总额",
    "Current assets": "流动资产",
    "Non-current assets": "非流动资产",
    "Total liabilities": "负债总额",
    "Current liabilities": "流动负债",
    "Non-current liabilities": "非流动负债",
    "Total equity": "所有者权益总额",
    "Share capital": "股本",
    "Retained earnings": "留存收益",
    "Goodwill": "商誉",
    "Intangible assets": "无形资产",
    "Property, plant and equipment": "物业、厂房及设备",
    "Right-of-use assets": "使用权资产",
    "Inventories": "存货",
    "Trade receivables": "应收账款",
    "Trade payables": "应付账款",
    "Borrowings": "借款",
    "Lease liabilities": "租赁负债",
    # cash flow
    "Cash and cash equivalents": "现金及现金等价物",
    "Net cash generated from operating activities": "经营活动产生的现金净额",
    "Net cash used in investing activities": "投资活动使用的现金净额",
    "Net cash from financing activities": "筹资活动产生的现金净额",
    "Capital expenditure": "资本开支",
    "Depreciation and amortisation": "折旧及摊销",
    "Working capital": "营运资金",
    # ratios and metrics
    "Gross margin": "毛利率",
    "Net margin": "净利率",
    "Return on equity": "净资产收益率",
    "Return on assets": "资产收益率",
    "Gearing ratio": "资产负债率",
    "Current ratio": "流动比率",
    "Year-on-year": "同比",
    "Segment": "分部",
    "Reporting segment": "报告分部",
    "Related party": "关联方",
    "Going concern": "持续经营",
    # corporate
    "Board of Directors": "董事会",
    "Supervisory Board": "监事会",
    "Audit Committee": "审计委员会",
    "Remuneration Committee": "薪酬委员会",
    "Chief Executive Officer": "首席执行官",
    "Chief Financial Officer": "首席财务官",
    "Subsidiary": "子公司",
    "Joint venture": "合营企业",
    "Associate": "联营企业",
    "Shareholder": "股东",
    "Controlling shareholder": "控股股东",
    # audit / reporting
    "Independent auditor's report": "独立核数师报告",
    "Auditor's report": "审计报告",
    "Key audit matter": "关键审计事项",
    "Accounting policy": "会计政策",
    "Contingent liability": "或有负债",
    "Provision": "拨备",
    "Impairment": "减值",
    "Fair value": "公允价值",
    "Amortised cost": "摊余成本",
    "Functional currency": "功能货币",
    "Reporting period": "报告期间",
    "Financial year": "财政年度",
    "Notes to the financial statements": "财务报表附注",
}

#: Standalone acronyms of interest in a financial report.
ACRONYM_RE = re.compile(r"\b([A-Z]{2,8}(?:/[A-Z]{2,4})?)\b")

#: Acronyms that are noise rather than terminology.
_ACRONYM_STOPLIST = frozenset(
    {
        "THE", "AND", "FOR", "NOT", "ALL", "ANY", "OUR", "ITS", "YOU", "MAY", "CAN",
        "HK", "US", "UK", "EU", "PRC", "LTD", "PLC", "INC", "CO", "NO", "VS", "ESG",
        "AGM", "EGM", "PDF", "HTML", "URL", "API", "ID", "OK", "N", "A",
    }
)

#: Acronyms and short forms that a Chinese financial report deliberately keeps in
#: Latin script. They are not untranslated terms -- they are the *correct*
#: rendering. Counting them as "no approved translation" would flood the risk
#: score with pure noise and teach reviewers to ignore the flags, which is worse
#: than not flagging at all.
PRESERVED_ACRONYMS = frozenset(
    {
        "IFRS", "HKFRS", "IFRIC", "GAAP", "RMB", "CNY", "HKD", "USD", "EUR", "JPY",
        "IPO", "REIT", "ETF", "SPV", "JV", "MOU", "NDA", "SLA", "KPI", "ROI",
        "CEO", "CFO", "COO", "CTO", "CIO", "MD", "VP", "HR", "IT", "R&D",
        "EBIT", "EBITDA", "EPS", "ROE", "ROA", "ROIC", "NAV", "NTA", "PE", "PB",
        "CAGR", "TSR", "WACC", "DCF", "NPV", "IRR", "PPE", "VAT", "CIT", "EIT",
        "Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY", "YTD", "YOY", "MOM", "QOQ",
        "AI", "ML", "SaaS", "IaaS", "PaaS", "CRM", "ERP", "IoT", "API",
        "A", "H", "B", "X", "N", "T", "M", "U", "DRA",
    }
)

#: English<->Chinese bracketed gloss patterns.
_BILINGUAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # EBITDA（息税折旧摊销前利润）  /  EBITDA (息税折旧摊销前利润)
    re.compile(r"([A-Za-z][A-Za-z0-9 ,&/'\-]{1,60}?)\s*[（(]\s*([\u4e00-\u9fff][\u4e00-\u9fff0-9A-Za-z、，,\.\-]{0,40}?)\s*[）)]"),
    # 归属于母公司股东的净利润（"归母净利润"）
    re.compile(r"([\u4e00-\u9fff][\u4e00-\u9fff0-9、，,\.\-]{1,30}?)\s*[（(]\s*[\"'“”]([^)\"'“”]{1,30})[\"'“”]\s*[）)]"),
    # 息税折旧摊销前利润（EBITDA）
    re.compile(r"([\u4e00-\u9fff][\u4e00-\u9fff0-9、，,\.\-]{1,30}?)\s*[（(]\s*([A-Z][A-Za-z0-9 ,&/'\-]{1,60}?)\s*[）)]"),
    # 每股收益 ("EPS")
    re.compile(r"([\u4e00-\u9fff]{2,30}?)\s*[（(]\s*([A-Z]{2,8})\s*[）)]"),
)

#: Bracketed content that is a cross-reference or a unit, never a translation.
_GLOSS_REJECT_RE = re.compile(r"^(?:注|note|see|第|共|单位|元|%|rmb|in\s|as\s|以下|上述|\d)", re.I)


@dataclass
class GlossaryEntry:
    source: str
    target: str
    authority: str = "model"
    #: How many times the term appears; drives which entries are worth injecting.
    frequency: int = 1
    note: str = ""
    aliases: list[str] = field(default_factory=list)

    @property
    def rank(self) -> int:
        return AUTHORITY_RANK.get(self.authority, 99)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "authority": self.authority,
            "frequency": self.frequency,
        }
        if self.note:
            out["note"] = self.note
        if self.aliases:
            out["aliases"] = list(self.aliases)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlossaryEntry:
        return cls(
            source=str(data.get("source", data.get("term", ""))).strip(),
            target=str(data.get("target", data.get("translation", ""))).strip(),
            authority=str(data.get("authority", "curated")),
            frequency=int(data.get("frequency", 1)),
            note=str(data.get("note", "")),
            aliases=[str(a) for a in data.get("aliases", [])],
        )


@dataclass
class GlossaryConflict:
    source: str
    existing: GlossaryEntry
    incoming: GlossaryEntry
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "existing": self.existing.to_dict(),
            "incoming": self.incoming.to_dict(),
            "reason": self.reason,
        }


class GlossaryStore:
    """Term table with authority-ranked merging and conflict tracking."""

    def __init__(self, entries: Iterable[GlossaryEntry] | None = None, *, with_seed: bool = True) -> None:
        self._entries: dict[str, GlossaryEntry] = {}
        self.conflicts: list[GlossaryConflict] = []
        if with_seed:
            for source, target in SEED_TERMS.items():
                self._entries[source.lower()] = GlossaryEntry(source=source, target=target, authority="seed")
        for entry in entries or []:
            self.upsert(entry)

    # -- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[GlossaryEntry]:
        return iter(self._entries.values())

    def __contains__(self, term: object) -> bool:
        return str(term).lower() in self._entries

    def get(self, term: str) -> GlossaryEntry | None:
        return self._entries.get(term.lower())

    def translate(self, term: str) -> str:
        entry = self.get(term)
        return entry.target if entry else term

    @property
    def entries(self) -> list[GlossaryEntry]:
        return list(self._entries.values())

    # -- mutation -----------------------------------------------------------

    def upsert(self, entry: GlossaryEntry) -> GlossaryEntry | None:
        """Insert or merge an entry, respecting authority. Returns the winner.

        Rules, in order:

        * Same target -> merge (bump frequency, union aliases).
        * Higher authority -> replace outright.
        * Lower authority -> rejected, recorded as ``superseded``.
        * Equal authority, different target -> **conflict**: the first entry
          stands and the clash is recorded for a human. Silently picking one is
          how a glossary ends up inconsistent in a way nobody notices.
        """
        key = entry.source.lower()
        existing = self._entries.get(key)
        if existing is None:
            self._entries[key] = entry
            return entry

        if existing.target == entry.target:
            # Agreement. Frequency sums, aliases union -- and, critically, the
            # **higher authority is promoted**. This is not a detail: the seed
            # lexicon happens to agree with the issuer on common terms like
            # "Revenue", and without promotion an issuer-supplied gloss would
            # stay labelled `seed` and could later be overwritten by a model
            # proposal. Agreement between an authoritative and a weak source is
            # still authoritative.
            if entry.rank < existing.rank:
                existing.authority = entry.authority
                existing.note = entry.note or existing.note
            existing.frequency += entry.frequency
            for alias in entry.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            return existing

        if entry.rank < existing.rank:
            self.conflicts.append(
                GlossaryConflict(
                    source=entry.source,
                    existing=existing,
                    incoming=entry,
                    reason=f"replaced {existing.authority} with higher-authority {entry.authority}",
                )
            )
            self._entries[key] = entry
            return entry

        reason = (
            "superseded by higher authority"
            if entry.rank > existing.rank
            else "equal-authority disagreement; kept first entry"
        )
        self.conflicts.append(
            GlossaryConflict(source=entry.source, existing=existing, incoming=entry, reason=reason)
        )
        return existing

    def merge(self, entries: Iterable[GlossaryEntry]) -> None:
        for entry in entries:
            self.upsert(entry)

    def prune(self, *, min_frequency: int = 1, drop_seed: bool = False) -> GlossaryStore:
        """Drop low-value entries so the prompt block stays small."""
        self._entries = {
            key: entry
            for key, entry in self._entries.items()
            if entry.frequency >= min_frequency and not (drop_seed and entry.authority == "seed")
        }
        return self

    # -- queries ------------------------------------------------------------

    def hits(self, text: str) -> list[tuple[str, str]]:
        """Terms from this glossary that occur in ``text`` (longest match first)."""
        lowered = text.lower()
        found: list[tuple[str, str]] = []
        for entry in sorted(self._entries.values(), key=lambda e: len(e.source), reverse=True):
            if entry.source.lower() in lowered or any(a.lower() in lowered for a in entry.aliases):
                found.append((entry.source, entry.target))
        return found

    def prompt_block(self, terms: Sequence[tuple[str, str]] | None = None, *, limit: int = 60) -> str:
        """Render a glossary block for prompt injection.

        Only the terms relevant to the current chunk are injected. Sending all
        600 seed entries every call is both expensive and actively harmful --
        long irrelevant term lists measurably degrade adherence.
        """
        pairs = list(terms) if terms is not None else [(e.source, e.target) for e in self._entries.values()]
        if not pairs:
            return ""
        pairs = pairs[:limit]
        lines = [f"- {source} => {target}" for source, target in pairs]
        return "Approved terminology (use these exact translations):\n" + "\n".join(lines)

    def coverage(self, text: str) -> float:
        """Fraction of glossary-relevant tokens in ``text`` that are covered.

        Used as a HITL signal: a financial block with many unknown terms is
        exactly where a model improvises.
        """
        candidates = extract_candidates(text, glossary=self)
        if not candidates:
            return 1.0
        covered = sum(1 for c in candidates if self.get(c) is not None)
        return covered / len(candidates)

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in sorted(self._entries.values(), key=lambda e: e.source)],
            "conflicts": [c.to_dict() for c in self.conflicts],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str | Path, indent: int | None = 2) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path, *, with_seed: bool = True) -> GlossaryStore:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = data.get("entries", data if isinstance(data, list) else [])
        store = cls([GlossaryEntry.from_dict(e) for e in raw], with_seed=with_seed)
        for conflict in data.get("conflicts", []) if isinstance(data, dict) else []:
            store.conflicts.append(
                GlossaryConflict(
                    source=str(conflict.get("source", "")),
                    existing=GlossaryEntry.from_dict(conflict.get("existing", {})),
                    incoming=GlossaryEntry.from_dict(conflict.get("incoming", {})),
                    reason=str(conflict.get("reason", "")),
                )
            )
        return store


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def extract_bilingual_pairs(text: str, *, min_len: int = 2) -> list[tuple[str, str]]:
    """Pull ``term（译文）`` pairs out of the report's own bilingual glosses.

    Returns ``(source_term, target_term)`` in textual order. The direction is
    normalised so the English/Chinese side is always the key and the other side
    the value, regardless of which language the document is written in.

    Known limitation: the left-hand pattern spans words so that multi-word terms
    (``Deferred tax asset``) survive, which means a gloss appearing mid-sentence
    captures the words before it too. Bilingual reports put these glosses in
    headings and definition lists, and a harvested entry is a *candidate* a
    human can edit, so this is tolerated rather than papered over.
    """
    if not text:
        return []
    pairs: list[tuple[str, str]] = []
    for pattern in _BILINGUAL_PATTERNS:
        for match in pattern.finditer(text):
            left = normalize_whitespace(match.group(1), keep_newlines=False).strip(" -–—:：")
            # The left-hand pattern is allowed to span words so that multi-word
            # terms ("Profit before tax") survive. That latitude means a gloss
            # appearing mid-sentence drags the preceding conjunction in with it:
            # "…and Revenue (营业收入)" yields the key "and Revenue", which is not
            # a term and would sit in the glossary forever. Articles are left
            # alone because a real glossary entry may legitimately begin with
            # one ("The board recommends a final dividend per share").
            left = re.sub(r"^(?:and|or|nor)\s+", "", left, flags=re.I)
            right = normalize_whitespace(match.group(2), keep_newlines=False).strip(" -–—:：“”\"'")
            if len(left) < min_len or len(right) < min_len:
                continue
            if _GLOSS_REJECT_RE.match(right) or _GLOSS_REJECT_RE.match(left):
                continue
            left_cjk = cjk_ratio(left) > 0.5
            right_cjk = cjk_ratio(right) > 0.5
            if left_cjk == right_cjk:
                # Both Chinese or both Latin: not a translation pair.
                continue
            source, target = (right, left) if left_cjk else (left, right)
            if source.lower() == target.lower():
                continue
            pairs.append((source, target))
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for source, target in pairs:
        key = source.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, target))
    return unique


def extract_candidates(text: str, *, glossary: GlossaryStore | None = None, limit: int = 400) -> list[str]:
    """Candidate terminology in ``text``: acronyms, seed hits, capitalised phrases.

    Acronyms a financial report keeps in Latin script (``IFRS``, ``RMB``) and
    section-numbering noise are filtered out. This list feeds the "terms with no
    approved translation" risk signal, and a signal that fires on every page is
    a signal nobody reads.
    """
    if not text:
        return []
    found: list[str] = []

    for match in ACRONYM_RE.finditer(text):
        token = match.group(1)
        if token in _ACRONYM_STOPLIST or len(token) < 2:
            continue
        if token in PRESERVED_ACRONYMS:
            continue
        found.append(token)

    if glossary is not None:
        lowered = text.lower()
        for entry in glossary:
            if entry.source.lower() in lowered:
                found.append(entry.source)

    # Capitalised multi-word phrases, e.g. "Deferred Tax Asset".
    for match in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+(?:of|and|the|for)?\s*[A-Z][a-z]{2,}){1,3})\b", text):
        phrase = match.group(1)
        if 6 <= len(phrase) <= 60:
            found.append(phrase)

    seen: set[str] = set()
    out: list[str] = []
    for term in found:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def build_glossary_from_document(
    texts: Iterable[str],
    *,
    base: GlossaryStore | None = None,
    min_frequency: int = 1,
) -> GlossaryStore:
    """Build a glossary: issuer glosses first, then seed-confirmed candidates."""
    store = base if base is not None else GlossaryStore()
    for text in texts:
        for source, target in extract_bilingual_pairs(text):
            store.upsert(GlossaryEntry(source=source, target=target, authority="source_gloss"))
        # Seed terms found in the text get their frequency bumped, which pushes
        # the terms this report actually uses to the top of the prompt block.
        for entry in list(store):
            if entry.authority == "seed" and entry.source.lower() in text.lower():
                store.upsert(GlossaryEntry(source=entry.source, target=entry.target, authority="seed"))
    if min_frequency > 1:
        store.prune(min_frequency=min_frequency)
    return store
