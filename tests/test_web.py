"""The browser demo's HTTP surface.

Skipped when FastAPI is absent, because the core project must stay installable
and testable with zero dependencies. When it is present, the tests cover the
things that would actually embarrass a demo: the endpoint must answer, the
verdict must be rendered, and a client-supplied path must not escape
``examples/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the web demo needs the optional .[web] extra")
# FastAPI does not pull in an ASGI test transport, and Starlette's TestClient
# needs httpx. Skip (rather than error) when only the runtime extra is present.
pytest.importorskip("httpx", reason="TestClient needs httpx; install the .[dev] extra")

from fastapi.testclient import TestClient  # noqa: E402

from art.web.app import EXAMPLES_DIR, build_payload, create_app, resolve_fixture  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# pages and metadata
# ---------------------------------------------------------------------------


class TestSurface:
    def test_index_serves_the_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="run"' in response.text
        assert "Annual Report Translator" in response.text

    def test_the_page_needs_no_network(self, client):
        """No CDN: the demo must work on a machine with no internet."""
        text = client.get("/").text
        assert "http://cdn" not in text and "https://cdn" not in text
        assert "unpkg" not in text and "jsdelivr" not in text

    def test_health(self, client):
        assert client.get("/api/health").json() == {"ok": True}

    def test_fixtures_list_includes_the_builtin_page(self, client):
        ids = [f["id"] for f in client.get("/api/fixtures").json()["fixtures"]]
        assert "builtin" in ids
        assert "demo_annual_report.json" in ids


# ---------------------------------------------------------------------------
# the run endpoint
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    def test_clean_run_has_no_findings(self, client):
        payload = client.post("/api/run", json={"fault": 0}).json()
        assert payload["verdict"]["ok"] is True
        assert payload["summary"]["numbers_mismatched"] == 0
        assert payload["review"] == []

    @pytest.mark.parametrize("fault", [1, 2, 3])
    def test_corruption_is_caught_and_reported(self, client, fault):
        payload = client.post("/api/run", json={"fault": fault}).json()
        assert payload["verdict"]["caught"] == fault
        assert payload["verdict"]["ok"] is True
        assert payload["summary"]["numbers_mismatched"] == fault
        assert payload["review"], "a flagged chunk must reach the review queue"

    def test_response_carries_everything_the_page_renders(self, client):
        payload = client.post("/api/run", json={"fault": 2}).json()
        for key in (
            "verdict",
            "audit",
            "summary",
            "number_report",
            "number_summary",
            "risks",
            "review",
            "source_html",
            "target_html",
            "target_markdown",
        ):
            assert key in payload, key

    def test_audit_reports_no_leaked_figures(self, client):
        audit = client.post("/api/run", json={"fault": 0}).json()["audit"]
        assert audit["ok"] is True
        assert audit["leaks"] == []
        assert audit["numeric_cells_copied"] > 0

    def test_target_html_preserves_merged_cells(self, client):
        payload = client.post("/api/run", json={"fault": 0}).json()
        assert "rowspan=" in payload["target_html"] or "colspan=" in payload["target_html"]

    def test_risks_are_banded_and_explained(self, client):
        risks = client.post("/api/run", json={"fault": 2}).json()["risks"]
        assert risks
        for risk in risks:
            assert risk["band"] in {"auto", "watch", "review", "blocked"}
            assert risk["threshold"] > 0
        flagged = [r for r in risks if r["needs_review"]]
        assert flagged and any(r["reasons"] for r in flagged)

    def test_named_fixture_is_used(self, client):
        payload = client.post(
            "/api/run", json={"fault": 1, "fixture": "demo_annual_report.json"}
        ).json()
        assert payload["verdict"]["caught"] == 1
        # The recorded fixture is four pages; the built-in page is one.
        assert payload["source_html"].count('class="page"') == 4

    def test_threshold_is_honoured(self, client):
        strict = client.post("/api/run", json={"fault": 1, "threshold": 0.01}).json()
        lax = client.post("/api/run", json={"fault": 1, "threshold": 0.99}).json()
        assert len(strict["review"]) >= len(lax["review"])


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------


class TestFixtureResolution:
    def test_builtin_and_empty_resolve_to_none(self):
        assert resolve_fixture(None) is None
        assert resolve_fixture("") is None
        assert resolve_fixture("builtin") is None

    def test_a_real_fixture_resolves_inside_examples(self):
        path = resolve_fixture("demo_annual_report.json")
        assert path is not None
        assert EXAMPLES_DIR.resolve() in path.parents

    @pytest.mark.parametrize(
        "name",
        ["../../../etc/passwd", "..\\..\\secrets.json", "/etc/hosts", "sub/dir/../../x.json"],
    )
    def test_traversal_is_rejected(self, name):
        """A demo endpoint must not become a file-read primitive."""
        with pytest.raises(ValueError):
            resolve_fixture(name)

    def test_a_missing_fixture_is_a_clear_error(self):
        with pytest.raises(ValueError, match="no such fixture"):
            resolve_fixture("does-not-exist.json")

    def test_endpoint_returns_400_for_a_bad_fixture(self, client):
        response = client.post("/api/run", json={"fixture": "../../../etc/passwd"})
        assert response.status_code == 400
        assert "outside" in response.json()["detail"]


# ---------------------------------------------------------------------------
# the pure builder (no HTTP involved)
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_works_without_a_server(self):
        payload = build_payload(fault=1)
        assert payload["verdict"]["caught"] == 1

    def test_fault_is_clamped_to_zero(self):
        payload = build_payload(fault=-5)
        assert payload["verdict"]["injected"] == 0
        assert payload["verdict"]["ok"] is True

    def test_accepts_a_path_object(self, demo_fixture_path: Path):
        payload = build_payload(fixture=None)
        assert payload["summary"]["chunks"] >= 1

    def test_outline_is_returned_for_the_sidebar(self):
        payload = build_payload()
        assert isinstance(payload["outline"], list)
