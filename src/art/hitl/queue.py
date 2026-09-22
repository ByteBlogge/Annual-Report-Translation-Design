"""The human review queue.

Why a file, not a database
--------------------------
The queue is an artefact of a run, not a service. Writing it as JSONL means:

* it survives the process, so a reviewer can pick it up tomorrow;
* it diffs in git, so you can see that last week's model left 12 items and this
  week's leaves 3 -- a real, trackable quality signal;
* it needs no infrastructure, so the reviewer does not have to be a developer.

Each item is append-only in spirit: ``pending`` -> ``approved`` / ``rejected``,
with the decision, the reviewer and the note recorded alongside the original
evidence. A rejected item retains the model's output *and* the reason it was
rejected, which is what makes the corpus reusable for evaluation later.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .policy import RiskFeatures, RiskScore

__all__ = ["ReviewItem", "ReviewQueue", "STATUSES"]

STATUSES = ("pending", "approved", "rejected", "skipped")


@dataclass
class ReviewItem:
    item_id: str = ""
    chunk_id: str = ""
    section_path: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=lambda: [0, 0])
    is_financial_summary: bool = False
    risk: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    target_text: str = ""
    tables: list[dict[str, str]] = field(default_factory=list)
    status: str = "pending"
    created_at: str = ""
    decided_at: str = ""
    reviewer: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.item_id:
            # Deterministic on purpose. ``chunk_id`` names the unit of work, so
            # translating a document again must land on the same queue entry --
            # otherwise every re-run appends a near-duplicate, the queue stops
            # being diffable (which is the whole reason it is a file), and a
            # decision recorded last week can never be found again.
            seed = self.chunk_id or f"{list(self.section_path)}:{list(self.pages)}"
            self.item_id = f"rv-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:10]}"
        if not self.created_at:
            self.created_at = _now()

    @property
    def evidence_fingerprint(self) -> tuple[Any, ...]:
        """What a reviewer's decision actually applied to.

        The target text is part of it, not just the score: a translation can
        change while the risk stays identical, and approving one is not
        approving the other.
        """
        return (self.risk_value, self.target_text)

    # -- views --------------------------------------------------------------

    @property
    def top_reasons(self) -> list[str]:
        reasons = self.risk.get("reasons", []) if isinstance(self.risk, dict) else []
        return [f"{r.get('name')}: {r.get('detail')}" for r in reasons[:4]]

    @property
    def risk_value(self) -> float:
        try:
            return float(self.risk.get("value", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def decide(self, status: str, *, reviewer: str = "", note: str = "") -> ReviewItem:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        self.status = status
        self.reviewer = reviewer or self.reviewer
        self.note = note or self.note
        self.decided_at = _now()
        return self

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "chunk_id": self.chunk_id,
            "section_path": list(self.section_path),
            "pages": list(self.pages),
            "is_financial_summary": self.is_financial_summary,
            "risk": self.risk,
            "features": self.features,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "tables": self.tables,
            "status": self.status,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "reviewer": self.reviewer,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewItem:
        return cls(
            item_id=str(data.get("item_id", "")),
            chunk_id=str(data.get("chunk_id", "")),
            section_path=[str(p) for p in data.get("section_path", [])],
            pages=[int(p) for p in data.get("pages", [0, 0])],
            is_financial_summary=bool(data.get("is_financial_summary", False)),
            risk=dict(data.get("risk", {})),
            features=dict(data.get("features", {})),
            source_text=str(data.get("source_text", "")),
            target_text=str(data.get("target_text", "")),
            tables=list(data.get("tables", [])),
            status=str(data.get("status", "pending")),
            created_at=str(data.get("created_at", "")),
            decided_at=str(data.get("decided_at", "")),
            reviewer=str(data.get("reviewer", "")),
            note=str(data.get("note", "")),
        )

    @classmethod
    def from_evidence(
        cls,
        *,
        chunk_id: str,
        section_path: Sequence[str],
        pages: tuple[int, int],
        score: RiskScore,
        features: RiskFeatures,
        source_text: str,
        target_text: str,
        tables: Sequence[dict[str, str]] = (),
    ) -> ReviewItem:
        return cls(
            chunk_id=chunk_id,
            section_path=list(section_path),
            pages=list(pages),
            is_financial_summary=features.is_financial_summary,
            risk=score.to_dict(),
            features=features.to_dict(),
            source_text=source_text,
            target_text=target_text,
            tables=list(tables),
        )


class ReviewQueue:
    """A JSONL-backed set of review items."""

    def __init__(self, path: str | Path | None = None, items: Iterable[ReviewItem] | None = None) -> None:
        self.path = Path(path) if path else None
        self._items: dict[str, ReviewItem] = {}
        self._order: list[str] = []
        if self.path and self.path.exists():
            self.load()
        for item in items or []:
            self.add(item)

    # -- container ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ReviewItem]:
        return (self._items[i] for i in self._order)

    def _put(self, item: ReviewItem) -> ReviewItem:
        """Last-wins insert, with no decision handling.

        Used when reading a file: on load the file is the truth, so a repeated
        id must simply overwrite rather than be reinterpreted as a refresh.
        """
        if item.item_id not in self._items:
            self._order.append(item.item_id)
        self._items[item.item_id] = item
        return item

    def add(self, item: ReviewItem) -> ReviewItem:
        """Upsert an item, keeping a decision that still applies.

        Because ``item_id`` is derived from the chunk, translating the same
        document into the same run directory again *refreshes* the entry the
        reviewer already saw instead of appending a near-duplicate.

        A decision survives only while the evidence it was made against is
        unchanged. If the target text or the risk moved, the output the reviewer
        approved no longer exists, so the item reopens as ``pending`` and the old
        decision is preserved in ``note`` as an audit trail. Silently keeping an
        approval for output that has been replaced is the one outcome worse than
        losing it.
        """
        existing = self._items.get(item.item_id)
        if existing is None:
            return self._put(item)

        if existing.status != "pending":
            if existing.evidence_fingerprint == item.evidence_fingerprint:
                item.status = existing.status
                item.reviewer = existing.reviewer
                item.note = existing.note
                item.decided_at = existing.decided_at
            else:
                item.note = _reopen_note(existing)
        # The item's history starts when the work first appeared, not on re-run.
        item.created_at = existing.created_at or item.created_at
        return self._put(item)

    def get(self, item_id: str) -> ReviewItem | None:
        return self._items.get(item_id)

    # -- queries ------------------------------------------------------------

    def pending(self) -> list[ReviewItem]:
        return [i for i in self if i.status == "pending"]

    def by_status(self, status: str) -> list[ReviewItem]:
        return [i for i in self if i.status == status]

    def sorted_by_risk(self, *, descending: bool = True) -> list[ReviewItem]:
        return sorted(self, key=lambda i: i.risk_value, reverse=descending)

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {status: 0 for status in STATUSES}
        for item in self:
            counts[item.status] = counts.get(item.status, 0) + 1
        pending = [i for i in self if i.status == "pending"]
        return {
            "total": len(self),
            "by_status": counts,
            "pending": len(pending),
            "financial_pending": sum(1 for i in pending if i.is_financial_summary),
            "max_risk": round(max((i.risk_value for i in self), default=0.0), 4),
        }

    # -- decisions ----------------------------------------------------------

    def decide(self, item_id: str, status: str, *, reviewer: str = "", note: str = "") -> ReviewItem:
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"no review item {item_id!r}")
        return item.decide(status, reviewer=reviewer, note=note)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path | None = None, *, fmt: str = "jsonl") -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path given and no default path configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            target.write_text(
                json.dumps([i.to_dict() for i in self], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return target
        with target.open("w", encoding="utf-8") as handle:
            for item in self:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        return target

    def load(self, path: str | Path | None = None) -> ReviewQueue:
        target = Path(path) if path else self.path
        if target is None or not target.exists():
            return self
        text = target.read_text(encoding="utf-8").strip()
        if not text:
            return self
        if text.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for record in records:
            self._put(ReviewItem.from_dict(record))
        return self

    def append(self, item: ReviewItem, path: str | Path | None = None) -> None:
        """Append one item without rewriting the file (safe for parallel runs)."""
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    # -- export -------------------------------------------------------------

    def to_csv(self, *, include_text: bool = False) -> str:
        buffer = io.StringIO()
        columns = [
            "item_id", "chunk_id", "status", "risk", "band", "financial",
            "pages", "section", "reasons", "reviewer", "note",
        ]
        if include_text:
            columns += ["source_text", "target_text"]
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in self.sorted_by_risk():
            writer.writerow(
                {
                    "item_id": item.item_id,
                    "chunk_id": item.chunk_id,
                    "status": item.status,
                    "risk": f"{item.risk_value:.3f}",
                    "band": item.risk.get("band", ""),
                    "financial": "yes" if item.is_financial_summary else "",
                    "pages": f"{item.pages[0]}-{item.pages[1]}" if item.pages else "",
                    "section": " > ".join(item.section_path),
                    "reasons": " | ".join(item.top_reasons),
                    "reviewer": item.reviewer,
                    "note": item.note,
                    "source_text": item.source_text[:500],
                    "target_text": item.target_text[:500],
                }
            )
        return buffer.getvalue()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _reopen_note(existing: ReviewItem) -> str:
    """Audit line carried onto an item whose evidence changed under a decision."""
    prior = existing.status + (f" by {existing.reviewer}" if existing.reviewer else "")
    return (
        f"reopened on re-run: evidence changed "
        f"(risk was {existing.risk_value:.2f}, previously {prior})"
    )
