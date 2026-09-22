"""A dependency-free TF-IDF retriever for glossary injection.

Why not a vector database
-------------------------
The design document said "store terms in a vector database". That is a
reasonable production choice and a poor default, because it makes the project
unrunnable without an extra service -- and the retrieval it replaces is
*lexical*, not semantic. Terminology consistency is an exact-match problem:
``EBITDA`` must map to ``息税折旧摊销前利润``, and an embedding model will
happily retrieve "operating profit" as semantically nearby.

So the default is deterministic TF-IDF over character n-grams, which:

* needs no network, no model download, no service;
* is fully reproducible, so a retrieval regression shows up in a test;
* handles Chinese without a segmentation library, by using character bigrams
  (``息税``, ``税折``, ``折旧``) -- which is both simpler and more robust than
  shipping a dictionary-based segmenter.

:class:`VectorRetriever` is the adapter for when you do want embeddings; it
takes any callable that maps text to a vector, so a real backend drops in
without changing the pipeline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from ..textutils import normalize_width

__all__ = ["tokenize", "TfidfIndex", "ScoredDoc", "VectorRetriever"]

_LATIN_RE = re.compile(r"[a-z][a-z0-9'\-\.]{1,}")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenise mixed Chinese/English text without a segmentation model.

    Latin runs become lowercased words; CJK runs become overlapping character
    bigrams plus the individual characters (so both ``EBITDA`` and ``息税前利润``
    match). Digits are kept as tokens because financial terminology frequently
    contains them (``IFRS 16``, ``HKFRS 9``).
    """
    if not text:
        return []
    normalized = normalize_width(text).lower()
    tokens: list[str] = _LATIN_RE.findall(normalized)
    tokens.extend(re.findall(r"\d+(?:\.\d+)?", normalized))

    for run in _CJK_RE.findall(normalized):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass
class ScoredDoc:
    doc_id: str
    score: float
    text: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class TfidfIndex:
    """In-memory TF-IDF over a small document set (hundreds to a few thousand)."""

    def __init__(self, documents: Sequence[tuple[str, str]] | None = None) -> None:
        self._ids: list[str] = []
        self._texts: dict[str, str] = {}
        self._vectors: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        if documents:
            self.add_many(documents)

    def add(self, doc_id: str, text: str, **metadata: object) -> None:
        tokens = tokenize(text)
        vector: Counter[str] = Counter(tokens)
        self._ids.append(doc_id)
        self._texts[doc_id] = text
        self._vectors.append(vector)
        for token in set(tokens):
            self._df[token] += 1
        self._idf = {}

    def add_many(self, documents: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in documents:
            self.add(doc_id, text)

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def idf(self) -> dict[str, float]:
        if not self._idf:
            total = max(1, len(self._ids))
            self._idf = {
                token: math.log((1 + total) / (1 + df)) + 1.0
                for token, df in self._df.items()
            }
        return self._idf

    def search(self, query: str, k: int = 5, *, min_score: float = 0.0) -> list[ScoredDoc]:
        if not self._ids:
            return []
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return []
        idf = self.idf
        query_norm = math.sqrt(sum((count * idf.get(t, 1.0)) ** 2 for t, count in query_tokens.items())) or 1.0
        results: list[ScoredDoc] = []
        for doc_id, vector in zip(self._ids, self._vectors, strict=False):
            if not vector:
                continue
            dot = 0.0
            for token, q_count in query_tokens.items():
                d_count = vector.get(token)
                if d_count:
                    weight = idf.get(token, 1.0)
                    dot += q_count * d_count * weight * weight
            if dot <= 0:
                continue
            doc_norm = math.sqrt(sum((count * idf.get(t, 1.0)) ** 2 for t, count in vector.items())) or 1.0
            score = dot / (query_norm * doc_norm)
            if score > min_score:
                results.append(ScoredDoc(doc_id=doc_id, score=score, text=self._texts.get(doc_id, "")))
        results.sort(key=lambda r: (-r.score, r.doc_id))
        return results[:k]


class VectorRetriever:
    """Adapter that makes any embedding function usable as a retriever.

    Deliberately tiny. The point is that swapping lexical retrieval for
    embedding retrieval is a constructor argument, not a rewrite -- so the
    claim "we used a vector store" can be evaluated on its merits without
    forcing every user of the repo to stand up a vector service.
    """

    def __init__(self, embed: Callable[[list[str]], list[list[float]]]) -> None:
        self.embed = embed
        self._ids: list[str] = []
        self._texts: dict[str, str] = {}
        self._vectors: list[list[float]] = []

    def add(self, doc_id: str, text: str) -> None:
        vector = self.embed([text])[0]
        self._ids.append(doc_id)
        self._texts[doc_id] = text
        self._vectors.append(list(vector))

    def add_many(self, documents: Iterable[tuple[str, str]]) -> None:
        items = list(documents)
        if not items:
            return
        vectors = self.embed([text for _, text in items])
        for (doc_id, text), vector in zip(items, vectors, strict=False):
            self._ids.append(doc_id)
            self._texts[doc_id] = text
            self._vectors.append(list(vector))

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: str, k: int = 5, *, min_score: float = 0.0) -> list[ScoredDoc]:
        if not self._ids:
            return []
        query_vector = self.embed([query])[0]
        scored: list[ScoredDoc] = []
        for doc_id, vector in zip(self._ids, self._vectors, strict=False):
            score = _cosine(query_vector, vector)
            if score > min_score:
                scored.append(ScoredDoc(doc_id=doc_id, score=score, text=self._texts.get(doc_id, "")))
        scored.sort(key=lambda r: (-r.score, r.doc_id))
        return scored[:k]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
