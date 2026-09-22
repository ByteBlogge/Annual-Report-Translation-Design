"""Chart region reading: a chart is data, not decoration.

The failure mode this module exists to prevent
----------------------------------------------
A naive translation pipeline treats a chart as an image: it copies the bitmap
across, translates the caption, and moves on. The reader of the translated
document then sees a chart whose axis labels and legend are still in the source
language, with the numbers embedded in pixels they cannot search, copy, or
verify. In a financial report that is not a cosmetic issue -- it is data loss.

So a chart is handled as a data structure:

* ``series`` holds the extracted ``(label, value)`` pairs, which flow into the
  number guard exactly like table cells do.
* ``axis_labels`` separates translatable category labels from non-translatable
  numeric ticks.
* ``description`` is a generated natural-language rendering, which is what
  actually goes into the translated document.
* ``extraction_method`` and ``confidence`` make the quality of the read
  explicit. A chart whose values could not be recovered is flagged for human
  review rather than silently emitted as an empty block.

The honest limitation: reading exact values off a chart is genuinely hard, and
models are unreliable at it. Rather than pretend otherwise, low-confidence
reads are routed to HITL -- which is the same policy the tables get.
"""

from __future__ import annotations

import re
from typing import Any

from ..schema import BBox, ChartBlock
from ..textutils import normalize_whitespace

__all__ = ["ChartReader", "CHART_SYSTEM_PROMPT", "chart_to_narrative", "ChartTextOptions"]


CHART_SYSTEM_PROMPT = """You extract data from a single chart in a financial report.

Return JSON only -- no prose, no markdown fences.

{
  "chart_type": "bar" | "line" | "pie" | "area" | "waterfall" | "stacked_bar" | "unknown",
  "title": "<the chart title if printed>",
  "series": [{"name": "<legend label>", "points": {"<category>": <number>}}],
  "axis_labels": {"x": ["..."], "y": ["..."]},
  "description": "<3-5 sentences: what the chart shows, the trend, and the key figures>",
  "confidence": <0.0-1.0>
}

Rules:
1. Read values in the chart's own units. Do NOT rescale. If the axis says
   'RMB thousand', report the axis numbers as printed.
2. Transcribe digits exactly. Do not round or add separators that are not there.
3. Only report a data point you can actually read. Omit unreadable points and
   lower "confidence" instead of interpolating.
4. Keep legend and category labels in the original language; do not translate.
5. If the image is not a chart, return {"chart_type": "unknown", "series": [],
   "description": "", "confidence": 0.0}.
"""


class ChartTextOptions:
    """Controls how a chart is rendered into the translated document."""

    def __init__(
        self,
        *,
        include_description: bool = True,
        include_data_table: bool = True,
        max_points: int = 40,
        label_translations: dict[str, str] | None = None,
    ) -> None:
        self.include_description = include_description
        self.include_data_table = include_data_table
        self.max_points = max_points
        self.label_translations = label_translations or {}


class ChartReader:
    """Recovers chart data via the VLM, falling back to whatever is on hand."""

    def __init__(self, llm: Any | None = None, *, max_tokens: int = 2048) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def read(
        self,
        *,
        image_bytes: bytes,
        caption: str = "",
        bbox: BBox | None = None,
        page: int = 0,
        precomputed: dict[str, Any] | None = None,
        mime: str = "image/png",
    ) -> ChartBlock:
        """Read one chart region.

        ``precomputed`` is the analyzer's own chart payload, when the page-level
        VLM already extracted it. Reusing it avoids paying for a second vision
        call on the same crop -- a real cost consideration at 200 pages.
        """
        if precomputed and precomputed.get("series"):
            return _chart_from_payload(precomputed, caption=caption, bbox=bbox, page=page, method="vlm")

        if self.llm is None or not image_bytes:
            return ChartBlock(
                caption=caption,
                bbox=bbox,
                page=page,
                extraction_method="unresolved",
                warnings=["no vision backend or no image available; chart data not recovered"],
            )

        from ..parser.analyzer import extract_json_object
        from ..translator.llm import LLMError

        prompt = f"Chart caption: {caption or '(none printed)'}\nExtract the chart data as JSON."
        try:
            reply = self.llm.complete_vision(
                system=CHART_SYSTEM_PROMPT,
                user=prompt,
                image_bytes=image_bytes,
                mime=mime,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            payload = extract_json_object(reply)
        except (ValueError, LLMError) as exc:
            return ChartBlock(
                caption=caption,
                bbox=bbox,
                page=page,
                extraction_method="unresolved",
                warnings=[f"chart extraction failed: {exc}"],
            )

        chart = _chart_from_payload(payload, caption=caption, bbox=bbox, page=page, method="vlm")
        if chart.confidence < 0.5 and chart.series:
            chart.warnings.append(f"low extraction confidence ({chart.confidence:.2f}); values need review")
        return chart

    def attach_translations(self, chart: ChartBlock, translations: dict[str, str]) -> ChartBlock:
        """Map translated legend/category labels onto the chart's axis labels."""
        if not translations:
            return chart
        axis = dict(chart.axis_labels)
        for key, values in axis.items():
            axis[key] = [translations.get(v, v) for v in values]
        chart.axis_labels = axis
        for series in chart.series:
            name = series.get("name")
            if name and name in translations:
                series["name"] = translations[name]
            points = series.get("points")
            if isinstance(points, dict):
                series["points"] = {translations.get(k, k): v for k, v in points.items()}
        return chart


def _chart_from_payload(
    payload: dict[str, Any], *, caption: str, bbox: BBox | None, page: int, method: str
) -> ChartBlock:
    series = payload.get("series")
    if not isinstance(series, list):
        series = []
    series = [s for s in series if isinstance(s, dict)]
    axis = payload.get("axis_labels")
    if not isinstance(axis, dict):
        axis = {}
    warnings: list[str] = []
    if not series:
        warnings.append("no numeric series recovered")
    return ChartBlock(
        caption=normalize_whitespace(caption or str(payload.get("title", "")), keep_newlines=False),
        chart_type=str(payload.get("chart_type", "unknown")),
        series=series,
        axis_labels={str(k): list(v) if isinstance(v, (list, tuple)) else [str(v)] for k, v in axis.items()},
        description=normalize_whitespace(str(payload.get("description", "")), keep_newlines=False),
        extraction_method=method,
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        bbox=bbox,
        page=page,
        warnings=warnings,
    )


_TICK_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?%?$")


def chart_to_narrative(chart: ChartBlock, options: ChartTextOptions | None = None) -> str:
    """Render a chart as text for the translated document.

    Numeric axis ticks are kept verbatim; only the description is expected to be
    in the target language. That split is deliberate -- a translated tick label
    is a number that no longer matches the plotted value.
    """
    opts = options or ChartTextOptions()
    lines: list[str] = []
    if chart.caption:
        lines.append(f"**{chart.caption}**")
    if opts.include_description and chart.description:
        lines.append(chart.description)
    if not chart.series:
        lines.append("[chart data could not be extracted; see review queue]")
        return "\n\n".join(lines)

    if opts.include_data_table:
        rows: list[tuple[str, str, float]] = []
        for s in chart.series:
            name = str(s.get("name", s.get("label", "")))
            points = s.get("points")
            if isinstance(points, dict):
                for category, value in points.items():
                    try:
                        rows.append((category, name, float(value)))
                    except (TypeError, ValueError):
                        continue
            elif "value" in s:
                try:
                    rows.append(("", name, float(s["value"])))
                except (TypeError, ValueError):
                    continue
        if rows:
            categories = list(dict.fromkeys(r[0] for r in rows))
            names = list(dict.fromkeys(r[1] for r in rows))
            lookup = {(r[0], r[1]): r[2] for r in rows}
            # Join the Markdown table with single newlines: a blank line between
            # rows terminates the table and it renders as loose paragraphs of
            # pipes instead of a grid.
            table_lines = [
                "| " + " | ".join(["", *names]) + " |",
                "| " + " | ".join("---" for _ in range(len(names) + 1)) + " |",
            ]
            for category in categories[: opts.max_points]:
                cells = [_format_number(lookup.get((category, n))) for n in names]
                table_lines.append("| " + " | ".join([category, *cells]) + " |")
            lines.append("\n".join(table_lines))

    if chart.axis_labels:
        ticks: list[str] = []
        for axis, values in chart.axis_labels.items():
            kept = [v for v in values if _TICK_RE.match(str(v).strip())]
            if kept:
                # Ticks are numbers; they are never translated. Keeping them as
                # printed is what lets a reader verify the chart against the text.
                ticks.append(f"{axis}-axis: {', '.join(kept)}")
        if ticks:
            lines.append("\n".join(ticks))
    return "\n\n".join(lines)


def _format_number(value: float | None) -> str:
    if value is None:
        return "-"
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")
