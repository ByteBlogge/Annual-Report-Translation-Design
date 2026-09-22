"""Risk scoring: deciding what a human must look at.

Design stance
-------------
HITL is usually implemented as "flag the low-confidence outputs". That does not
work, because the confidence number a model reports is itself a model output --
it is uncorrelated with being right, and it is highest exactly when the model is
most fluent and most wrong.

So there is no model-reported confidence in this policy. Every factor is a
**measurement taken from the artefacts**:

* number drift, from the deterministic guard
* grid holes and merge conflicts, from geometry
* untranslated label cells, counted by the pipeline
* unresolved charts, recorded by the analyzer
* numeric density, a property of the source slice
* remaining terminology violations, counted by string comparison

The policy is a scoring function, not a classifier, and every contribution is
reported with its reason. A reviewer who disagrees with a flag can see which
signal produced it and adjust one weight -- which is the difference between a
tunable system and a black box that occasionally demands attention.

The threshold is **lower for financial-summary sections**. A dropped figure in a
chairman's letter is a quality issue; the same drop in the consolidated
statement of profit or loss is a restatement waiting to happen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["RiskFeatures", "RiskFactor", "RiskScore", "RiskPolicy"]


@dataclass
class RiskFeatures:
    """Everything the policy is allowed to look at. No model-reported values."""

    chunk_id: str = ""
    section_path: tuple[str, ...] = ()
    pages: tuple[int, int] = (0, 0)
    is_financial_summary: bool = False

    # --- from the number guard ---
    number_drift: float = 0.0
    numbers_missing: int = 0
    numbers_mismatched: int = 0
    #: Figures present in the target with no counterpart in the source. Kept
    #: separate from ``numbers_mismatched`` and scored at a deliberately lower
    #: weight, because the benign case is common: an English "31 December"
    #: becomes 12月31日, which adds a legitimate figure (the month) that the
    #: source never printed. Flagging every date would drown the real findings.
    #: But a *fabricated* figure lands here too, so it must not be invisible.
    numbers_added: int = 0
    number_structural: int = 0

    # --- from the parser / table path ---
    table_count: int = 0
    table_merge_conflicts: int = 0
    table_holes: int = 0
    table_warnings: int = 0
    untranslated_label_cells: int = 0

    # --- from the chart path ---
    chart_count: int = 0
    charts_unresolved: int = 0

    # --- from the chunker ---
    numeric_density: float = 0.0
    glossary_terms: int = 0
    unknown_terms: int = 0

    # --- from the terminology agent ---
    terminology_remaining: int = 0

    # --- hard failures ---
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_path": list(self.section_path),
            "pages": list(self.pages),
            "is_financial_summary": self.is_financial_summary,
            "number_drift": round(self.number_drift, 4),
            "numbers_missing": self.numbers_missing,
            "numbers_mismatched": self.numbers_mismatched,
            "numbers_added": self.numbers_added,
            "number_structural": self.number_structural,
            "table_count": self.table_count,
            "table_merge_conflicts": self.table_merge_conflicts,
            "table_holes": self.table_holes,
            "table_warnings": self.table_warnings,
            "untranslated_label_cells": self.untranslated_label_cells,
            "chart_count": self.chart_count,
            "charts_unresolved": self.charts_unresolved,
            "numeric_density": round(self.numeric_density, 4),
            "glossary_terms": self.glossary_terms,
            "unknown_terms": self.unknown_terms,
            "terminology_remaining": self.terminology_remaining,
            "errors": list(self.errors),
        }


@dataclass
class RiskFactor:
    name: str
    contribution: float
    detail: str
    severity: str = "info"  # info | warn | critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contribution": round(self.contribution, 4),
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass
class RiskScore:
    value: float
    threshold: float
    reasons: list[RiskFactor] = field(default_factory=list)
    hard_fail: bool = False

    @property
    def needs_review(self) -> bool:
        return self.hard_fail or self.value >= self.threshold

    @property
    def band(self) -> str:
        if self.hard_fail:
            return "blocked"
        if self.value >= self.threshold:
            return "review"
        if self.value >= self.threshold * 0.6:
            return "watch"
        return "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
            "band": self.band,
            "needs_review": self.needs_review,
            "hard_fail": self.hard_fail,
            "reasons": [r.to_dict() for r in self.reasons],
        }


#: Default weights. Rationale for each is documented at the scoring site.
DEFAULT_WEIGHTS: dict[str, float] = {
    "number_drift": 1.0,
    "number_mismatch": 0.55,
    #: Deliberately half a mismatch: real, but the benign date case is common
    #: enough that these must accumulate before anyone is asked to look.
    "number_added": 0.25,
    "number_structural": 0.30,
    "merge_conflict": 0.40,
    "grid_hole": 0.45,
    "table_warning": 0.20,
    "untranslated_labels": 0.30,
    "chart_unresolved": 0.55,
    "numeric_density": 0.35,
    "unknown_terms": 0.30,
    "terminology_remaining": 0.40,
    "error": 1.0,
}


@dataclass
class RiskPolicy:
    """Scores a chunk in ``[0, 1]`` from measured artefacts."""

    threshold: float = 0.45
    #: Financial-summary chunks get the threshold lowered by this much, floored
    #: at ``min_threshold``. A restatement risk is not the same as a style issue.
    financial_relief: float = 0.15
    min_threshold: float = 0.10
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    #: Normalisation points: the count at which a factor reaches full weight.
    saturation: dict[str, float] = field(
        default_factory=lambda: {
            "number_mismatch": 2.0,
            "number_added": 2.0,
            "number_structural": 3.0,
            "merge_conflict": 2.0,
            "grid_hole": 4.0,
            "table_warning": 5.0,
            "untranslated_labels": 5.0,
            "unknown_terms": 15.0,
            "terminology_remaining": 3.0,
        }
    )

    # -- API ---------------------------------------------------------------

    def effective_threshold(self, features: RiskFeatures) -> float:
        if features.is_financial_summary:
            return max(self.min_threshold, self.threshold - self.financial_relief)
        return self.threshold

    def score(self, features: RiskFeatures) -> RiskScore:
        reasons: list[RiskFactor] = []
        total = 0.0

        def add(name: str, raw: float, detail: str, severity: str = "info", cap: float = 1.0) -> None:
            nonlocal total
            weight = self.weights.get(name, 0.0)
            if weight <= 0 or raw <= 0:
                return
            contribution = min(cap, raw) * weight
            total += contribution
            reasons.append(RiskFactor(name=name, contribution=contribution, detail=detail, severity=severity))

        def norm(name: str, count: float) -> float:
            pivot = self.saturation.get(name, 1.0)
            return count / pivot if pivot > 0 else float(count)

        # 1. The direct evidence: figures that did not survive.
        if features.number_drift > 0:
            add(
                "number_drift",
                features.number_drift,
                f"{features.number_drift:.0%} of source figures did not match the target",
                severity="critical" if features.number_drift >= 0.1 else "warn",
            )
        if features.numbers_mismatched:
            add(
                "number_mismatch",
                norm("number_mismatch", features.numbers_mismatched),
                f"{features.numbers_mismatched} figure(s) changed value (digit slip or restatement)",
                severity="critical",
            )
        if features.numbers_added:
            add(
                "number_added",
                norm("number_added", features.numbers_added),
                f"{features.numbers_added} figure(s) appear in the target only "
                f"(fabrication, or a reformatted date / cross-reference)",
                severity="warn",
            )
        if features.number_structural:
            add(
                "number_structural",
                norm("number_structural", features.number_structural),
                f"{features.number_structural} structural number issue(s), e.g. a dropped unit note",
                severity="critical",
            )

        # 2. Structural integrity of the tables.
        if features.table_merge_conflicts:
            add(
                "merge_conflict",
                norm("merge_conflict", features.table_merge_conflicts),
                f"{features.table_merge_conflicts} merge conflict(s): two cells claimed the same slot",
                severity="critical",
            )
        if features.table_holes:
            add(
                "grid_hole",
                norm("grid_hole", features.table_holes),
                f"{features.table_holes} uncovered grid slot(s): a cell may be missing",
                severity="warn",
            )
        if features.table_warnings:
            add(
                "table_warning",
                norm("table_warning", features.table_warnings),
                f"{features.table_warnings} parser warning(s) on tables",
            )
        if features.untranslated_label_cells:
            add(
                "untranslated_labels",
                norm("untranslated_labels", features.untranslated_label_cells),
                f"{features.untranslated_label_cells} table label cell(s) left untranslated",
                severity="warn",
            )

        # 3. Chart data that could not be recovered.
        if features.charts_unresolved:
            add(
                "chart_unresolved",
                features.charts_unresolved / max(1.0, features.chart_count),
                f"{features.charts_unresolved}/{features.chart_count} chart(s) yielded no data points",
                severity="warn",
            )

        # 4. Source properties that predict trouble.
        if features.numeric_density > 0.12:
            excess = (features.numeric_density - 0.12) / 0.25
            add(
                "numeric_density",
                excess,
                f"numeric density {features.numeric_density:.1%} (dense figure section)",
            )
        if features.unknown_terms:
            add(
                "unknown_terms",
                norm("unknown_terms", features.unknown_terms),
                f"{features.unknown_terms} term(s) with no approved translation",
            )
        if features.terminology_remaining:
            add(
                "terminology_remaining",
                norm("terminology_remaining", features.terminology_remaining),
                f"{features.terminology_remaining} approved term(s) still not applied after repair",
                severity="warn",
            )

        # 5. Hard failures: the chunk produced nothing usable.
        hard_fail = bool(features.errors)
        for error in features.errors:
            reasons.append(
                RiskFactor(name="error", contribution=self.weights.get("error", 1.0), detail=error, severity="critical")
            )
        if hard_fail:
            total = max(total, 1.0)

        return RiskScore(
            value=min(1.0, total),
            threshold=self.effective_threshold(features),
            reasons=reasons,
            hard_fail=hard_fail,
        )

    # -- reporting ---------------------------------------------------------

    @staticmethod
    def explain(score: RiskScore) -> str:
        head = f"risk={score.value:.2f} threshold={score.threshold:.2f} band={score.band}"
        if not score.reasons:
            return head + " (no findings)"
        lines = [head]
        for factor in sorted(score.reasons, key=lambda f: -f.contribution):
            lines.append(f"  +{factor.contribution:.2f} {factor.name}: {factor.detail}")
        return "\n".join(lines)


def batch_summary(scores: Sequence[RiskScore]) -> dict[str, Any]:
    bands: dict[str, int] = {}
    for score in scores:
        bands[score.band] = bands.get(score.band, 0) + 1
    return {
        "chunks": len(scores),
        "bands": bands,
        "needs_review": sum(1 for s in scores if s.needs_review),
        "max_risk": round(max((s.value for s in scores), default=0.0), 4),
    }
