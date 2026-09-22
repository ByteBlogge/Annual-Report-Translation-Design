"""FastAPI app for the browser demo.

Design notes worth stating, because they are the difference between a demo and a
screenshot:

* **It runs offline.** The default backend is the deterministic mock, so the page
  works with no API key. The point of the page is *verification*, not translation
  quality -- a visitor can press "corrupt 3 figures" and watch the guard catch
  three.
* **It is the same code path as the CLI.** Both call :func:`art.demo.run_demo`.
* **It does not trust a path from the client.** ``fixture`` is resolved against
  the repo's ``examples/`` directory only, so the endpoint cannot be used to read
  arbitrary files off the host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..chunker.pipeline import ChunkingPipeline
from ..demo import build_demo_document, run_demo
from .page import INDEX_HTML
from .render import audit_payloads, render_source_html, render_target_html

__all__ = ["create_app", "EXAMPLES_DIR"]

#: Fixtures the demo page is allowed to load. A client-supplied path is resolved
#: inside this directory and must stay inside it.
EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"

_BUILTIN = {"id": "builtin", "label": "Built-in demo page (1 page, merged header + chart)"}


def _available_fixtures() -> list[dict[str, str]]:
    fixtures = [_BUILTIN]
    if EXAMPLES_DIR.is_dir():
        for path in sorted(EXAMPLES_DIR.glob("*.json")):
            fixtures.append({"id": path.name, "label": path.name})
    return fixtures


def resolve_fixture(name: str | None) -> Path | None:
    """Map a client-supplied fixture name to a path inside ``examples/``.

    Raises ``ValueError`` for anything that escapes the directory. A demo that
    will happily read ``/etc/passwd`` because it forwarded a query parameter into
    an open() call is a bad look, and this costs three lines.
    """
    if not name or name in ("builtin", "<builtin>"):
        return None
    candidate = (EXAMPLES_DIR / name).resolve()
    if EXAMPLES_DIR.resolve() not in candidate.parents:
        raise ValueError(f"fixture {name!r} is outside {EXAMPLES_DIR}")
    if not candidate.is_file():
        raise ValueError(f"no such fixture: {name!r}")
    return candidate


def build_payload(
    *,
    fixture: str | None = None,
    fault: int = 2,
    threshold: float = 0.45,
    jitter: float = 2.5,
    target_language: str = "Simplified Chinese",
) -> dict[str, Any]:
    """Run the demo and shape the response. Pure data, no FastAPI involved."""
    path = resolve_fixture(fixture)
    document = build_demo_document(path, jitter=jitter)
    chunking = ChunkingPipeline().run(document)
    outcome = run_demo(
        path,
        fault=max(0, int(fault)),
        threshold=float(threshold),
        jitter=jitter,
        target_language=target_language,
        doc=document,
        chunking=chunking,
    )
    result = outcome.result

    risks = []
    for chunk, score, features in zip(
        result.chunk_results, result.risk_scores, result.features, strict=False
    ):
        risks.append(
            {
                "chunk_id": chunk.chunk_id,
                "section": " > ".join(chunk.section_path) or "(no section)",
                "pages": list(features.pages),
                "band": score.band,
                "value": round(score.value, 3),
                "threshold": round(score.threshold, 3),
                "needs_review": score.needs_review,
                "financial": features.is_financial_summary,
                "reasons": [
                    {"detail": f.detail, "contribution": round(f.contribution, 3), "severity": f.severity}
                    for f in sorted(score.reasons, key=lambda f: -f.contribution)
                ],
            }
        )

    review = []
    for item in result.queue.sorted_by_risk():
        review.append(
            {
                "item_id": item.item_id,
                "chunk_id": item.chunk_id,
                "section": " > ".join(item.section_path) or "(no section)",
                "risk": item.risk_value,
                "band": item.risk.get("band", ""),
                "status": item.status,
                "financial": item.is_financial_summary,
                "source_text": item.source_text[:1200],
                "target_text": item.target_text[:1200],
                "reasons": item.top_reasons,
            }
        )

    report = result.aggregate_number_report()
    return {
        "verdict": outcome.verdict,
        "audit": audit_payloads(document),
        "summary": result.summary(),
        "number_report": report.to_dict(),
        "number_summary": report.summary(),
        "risks": risks,
        "review": review,
        "source_html": render_source_html(document),
        "target_html": render_target_html(result),
        "target_markdown": result.to_markdown(),
        "outline": list(result.outlines),
    }


def create_app():
    """Build the FastAPI app. Imported lazily so ``art`` stays dependency-free."""
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised by the CLI's error path
        raise ImportError(
            "the web demo needs the optional extra:  pip install -e '.[web]'"
        ) from exc

    app = FastAPI(
        title="Annual Report Translator — verification demo",
        description=(
            "Offline demo of the parser / chunker / translator / HITL pipeline, "
            "including deliberate number corruption and the guard that catches it."
        ),
        version="0.1.0",
    )

    @app.get("/", response_class=HTMLResponse)
    def index():
        # No return annotation on purpose: ``from __future__ import annotations``
        # turns it into a string, and FastAPI resolves it against module globals
        # -- where HTMLResponse does not exist, because FastAPI is imported
        # lazily inside this function so that ``art`` stays dependency-free.
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/fixtures")
    def fixtures() -> dict[str, Any]:
        return {"fixtures": _available_fixtures()}

    @app.post("/api/run")
    def run(payload: dict[str, Any] = Body(default={})):
        # Annotation omitted for the same reason as ``index`` above.
        try:
            body = build_payload(
                fixture=payload.get("fixture"),
                fault=int(payload.get("fault", 2)),
                threshold=float(payload.get("threshold", 0.45)),
                jitter=float(payload.get("jitter", 2.5)),
                target_language=str(payload.get("target_language", "Simplified Chinese")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(body)

    return app
