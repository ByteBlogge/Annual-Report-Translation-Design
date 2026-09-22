"""The ``art`` command's own logic.

The CLI is the surface all six requirements are demonstrated through, and it has
behaviour nothing else in the suite can see:

* whether a flag actually reaches the pipeline (``--inject-fault`` once set an
  attribute that ``__init__`` had already snapshotted, so it silently did
  nothing -- a flag that lies is worse than one that is missing);
* how a review decision is addressed (the listing showed the chunk id while the
  flags demanded the queue id, so the documented workflow could not be followed
  from the terminal);
* what happens when an id is wrong (it used to be a raw ``KeyError`` traceback).
"""

from __future__ import annotations

import json

import pytest

from art.cli import main

pytestmark = pytest.mark.usefixtures("demo_fixture_path")


@pytest.fixture
def run_dir(tmp_path, demo_fixture_path):
    """A real run directory produced by the real command."""
    code = main(
        [
            "translate",
            str(demo_fixture_path),
            "--parser", "mock",
            "--llm", "mock",
            "--inject-fault", "2",
            "--out", str(tmp_path),
        ]
    )
    assert code == 0
    return tmp_path


def records(run_dir):
    text = (run_dir / "review.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def summary(run_dir):
    path = next(run_dir.glob("*.report.json"))
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


class TestTranslate:
    def test_it_writes_a_pending_queue(self, run_dir):
        items = records(run_dir)
        assert items
        assert all(item["status"] == "pending" for item in items)

    def test_inject_fault_really_injects(self, run_dir):
        """Regression: the flag set an attribute the constructor had snapshotted.

        It therefore injected nothing while still reporting success, which is the
        failure mode that would make the guard look trustworthy for the wrong
        reason.
        """
        assert summary(run_dir)["numbers_mismatched"] == 2

    def test_a_clean_run_mismatches_nothing(self, tmp_path, demo_fixture_path):
        assert main(["translate", str(demo_fixture_path), "--parser", "mock",
                     "--llm", "mock", "--out", str(tmp_path)]) == 0
        assert summary(tmp_path)["numbers_mismatched"] == 0

    def test_preview_tables_prints_the_rebuilt_grid(self, tmp_path, demo_fixture_path, capsys):
        assert main(["translate", str(demo_fixture_path), "--parser", "mock",
                     "--llm", "mock", "--preview-tables", "--out", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "translated tables (as rebuilt)" in out
        # a label translated by the model, and a figure copied by code
        assert "营业收入" in out
        assert "1,842,336" in out

    def test_the_sdo_can_be_written_alone(self, tmp_path, demo_fixture_path):
        out = tmp_path / "doc.json"
        assert main(["parse", str(demo_fixture_path), "-o", str(out)]) == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["doc_id"]
        assert payload["pages"]


class TestReviewDecisions:
    def test_the_listing_prints_the_id_the_flags_want(self, run_dir, capsys):
        assert main(["review", str(run_dir / "review.jsonl")]) == 0
        item = records(run_dir)[0]
        assert item["item_id"] in capsys.readouterr().out

    def test_a_decision_can_be_addressed_by_item_id(self, run_dir):
        item_id = records(run_dir)[0]["item_id"]
        assert main(["review", str(run_dir / "review.jsonl"),
                     "--approve", item_id, "--reviewer", "alice"]) == 0
        assert records(run_dir)[0]["status"] == "approved"

    def test_a_decision_can_be_addressed_by_chunk_id(self, run_dir):
        chunk_id = records(run_dir)[0]["chunk_id"]
        assert main(["review", str(run_dir / "review.jsonl"),
                     "--approve", chunk_id, "--reviewer", "alice"]) == 0
        assert records(run_dir)[0]["status"] == "approved"

    def test_an_unknown_id_is_refused_and_nothing_is_applied(self, run_dir):
        """All-or-nothing: a batch that half-applies on a typo is worse than one
        that refuses."""
        item_id = records(run_dir)[0]["item_id"]
        code = main(["review", str(run_dir / "review.jsonl"),
                     "--approve", item_id, "--reject", "not-an-id"])
        assert code == 2
        assert records(run_dir)[0]["status"] == "pending"

    def test_a_decision_survives_a_rerun(self, run_dir, demo_fixture_path):
        item_id = records(run_dir)[0]["item_id"]
        assert main(["review", str(run_dir / "review.jsonl"), "--approve", item_id,
                     "--reviewer", "alice", "--note", "checked"]) == 0

        # same document, same output -> the decision still applies
        assert main(["translate", str(demo_fixture_path), "--parser", "mock",
                     "--llm", "mock", "--inject-fault", "2", "--out", str(run_dir)]) == 0
        kept = records(run_dir)[0]
        assert len(records(run_dir)) == 1
        assert kept["status"] == "approved"
        assert kept["reviewer"] == "alice"
        assert kept["note"] == "checked"

    def test_reviewing_an_empty_queue_is_not_an_error(self, tmp_path):
        empty = tmp_path / "review.jsonl"
        empty.write_text("", encoding="utf-8")
        assert main(["review", str(empty)]) == 0
