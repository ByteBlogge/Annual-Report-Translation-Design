"""Command line interface.

    art parse     <pdf|json>          stage 1: structure -> SDO
    art chunk     <pdf|json>          stage 1+2: SDO -> glossary + chunks
    art translate <pdf|json>          stages 1-4: full run + review queue
    art review    <queue.jsonl>       inspect and decide review items
    art inspect   <pdf|json>          quick look at structure or chunks
    art demo                          offline end-to-end, no credentials

Everything works with no API key (``--llm mock``) and no third-party package.
With ``--llm openai-compat`` plus credentials it runs against a real model
without any code change -- which is the whole point of the backend seam.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .hitl.queue import STATUSES

# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal ``.env`` loader.

    Implemented here rather than pulling in ``python-dotenv`` so that the CLI has
    no install step. Only ``KEY=VALUE`` lines and ``#`` comments are supported,
    which is all ``.env.example`` uses.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def make_parser(backend: str, *, llm: Any = None, record: str | None = None, **kwargs: Any):
    """Build a DocumentParser for the requested backend.

    The VLM backend needs an LLM client, which is why this factory takes one --
    and why the mock backend must stay usable without one. That asymmetry is the
    seam that makes offline runs possible.
    """
    from .parser.pipeline import DocumentParser, ParserOptions

    key = (backend or env("ART_PARSER_BACKEND", "mock")).lower()
    options = ParserOptions(
        dpi=int(kwargs.pop("dpi", env("ART_PDF_DPI", "180"))),
        max_pages=kwargs.pop("max_pages", None),
        read_charts=kwargs.pop("read_charts", True),
    )

    if key in ("qwen-vl", "qwen", "vlm"):
        if llm is None:
            raise SystemExit(
                "--parser qwen-vl needs an LLM client. Pass --llm openai-compat with "
                "ART_LLM_API_KEY set, or use --parser mock."
            )
        from .parser.qwen_vl_analyzer import QwenVLLayoutAnalyzer

        analyzer = QwenVLLayoutAnalyzer(llm)
    elif key in ("paddle", "ppstructure", "pp-structure"):
        from .parser.paddle_analyzer import PaddleLayoutAnalyzer

        analyzer = PaddleLayoutAnalyzer(lang=env("ART_OCR_LANG", "ch"))
    else:
        from .parser.mock_analyzer import MockLayoutAnalyzer

        analyzer = MockLayoutAnalyzer(
            fixture=kwargs.pop("fixture", None),
            jitter_px=kwargs.pop("jitter", 0.0),
            shuffle_regions=kwargs.pop("shuffle", False),
        )
    return DocumentParser(analyzer, options=options, llm=llm, run_dir=record)


def make_llm_client(llm_name: str, **overrides: Any):
    from .translator.llm import make_llm

    key = (llm_name or env("ART_LLM_BACKEND", "mock")).lower()
    params: dict[str, Any] = {
        "base_url": overrides.pop("base_url", env("ART_LLM_BASE_URL", "")),
        "api_key": overrides.pop("api_key", env("ART_LLM_API_KEY", "")),
        "model": overrides.pop("model", env("ART_LLM_MODEL", "")),
        "vlm_model": overrides.pop("vlm_model", env("ART_VLM_MODEL", "")),
    }
    if key in ("mock", "offline", "stub") or key == "auto":
        return make_llm("mock", **overrides)
    return make_llm(key, **params, **overrides)


def _fmt_table(table: Any, *, max_rows: int = 12) -> str:
    """Plain-text rendering of a table for the terminal."""
    from .parser.table_builder import table_to_tsv

    tsv = table_to_tsv(table)
    lines = tsv.splitlines()
    head = lines[:max_rows]
    out = "\n".join("    " + line for line in head)
    if len(lines) > max_rows:
        out += f"\n    ... {len(lines) - max_rows} more row(s)"
    return out


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_parse(args: argparse.Namespace) -> int:
    from .parser.pipeline import summarise_document

    llm = make_llm_client(args.llm) if args.parser in ("qwen-vl", "qwen", "vlm") else None
    parser = make_parser(args.parser, llm=llm, record=args.record, dpi=args.dpi)
    result = parser.parse(args.input)
    document = result.document

    if args.json or args.out:
        target = Path(args.out) if args.out else None
        if target:
            document.save(target)
            print(f"wrote {target}")
        else:
            print(document.to_json())

    print(f"\nparsed {document.doc_id}  ({len(document.pages)} page(s), analyzer={parser.analyzer.name})")
    stats = summarise_document(document)
    for key, value in stats.items():
        print(f"  {key:28} {value}")

    if args.tables:
        for index, table in enumerate(document.tables(), start=1):
            merged = sum(1 for c in table.cells if c.is_merged)
            print(
                f"\n  table {index}  page {table.page}  {table.n_rows}x{table.n_cols}  "
                f"cells={len(table.cells)} merged={merged} units={table.units_note!r}"
                f"  caption={table.caption!r}"
            )
            if table.conflicts:
                for conflict in table.conflicts:
                    print(f"    ! merge conflict at ({conflict.row},{conflict.col}): {conflict.reason}")
            for warning in table.warnings:
                print(f"    ! {warning}")
            print(_fmt_table(table))

    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    from .chunker.chunker import ChunkOptions
    from .chunker.pipeline import ChunkingPipeline, glossary_report

    llm = make_llm_client(args.llm) if args.parser in ("qwen-vl", "qwen", "vlm") else None
    parser = make_parser(args.parser, llm=llm, dpi=args.dpi)
    document = parser.parse(args.input).document

    options = ChunkOptions(
        max_tokens=args.max_tokens,
        split_at_level=args.split_level,
        keep_tables_intact=not args.allow_split_tables,
    )
    result = ChunkingPipeline(options).run(document)

    print(f"document {document.doc_id}: {len(document.pages)} page(s)")
    print("\noutline (section paths, first two levels):")
    for line in result.outline:
        print(f"  {line}")

    print("\nchunks:")
    for chunk in result.chunks:
        print(
            f"  {chunk.chunk_id:34} {chunk.section_title[:40]:42} "
            f"tok={chunk.token_estimate:5} nd={chunk.numeric_density:.3f} "
            f"tables={len(chunk.tables)} terms={len(chunk.glossary_terms)}"
        )
    print()
    _print_json(result.stats)
    print()
    print(glossary_report(result.glossary))

    if args.glossary:
        result.glossary.save(args.glossary)
        print(f"\nwrote glossary -> {args.glossary}")
    if args.out:
        payload = {
            "doc_id": document.doc_id,
            "outline": result.outline,
            "stats": result.stats,
            "chunks": [c.to_dict() for c in result.chunks],
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    from .chunker.chunker import ChunkOptions
    from .chunker.pipeline import ChunkingPipeline, glossary_report
    from .hitl.policy import RiskPolicy
    from .translator.pipeline import (
        TranslationPipeline,
        preview_tables,
        render_number_findings,
        render_run_summary,
    )

    llm = make_llm_client(
        args.llm,
        base_url=args.base_url or None,
        api_key=args.api_key or None,
        model=args.model or None,
    )
    if args.inject_fault:
        # Only the offline backend can be told to misbehave; that is its purpose.
        from .translator.llm import MockLLM

        if not isinstance(llm, MockLLM):
            print("--inject-fault only works with --llm mock", file=sys.stderr)
            return 2
        llm.inject_number_drift = args.inject_fault
        # ``make_llm_client`` built the instance with the default budget of 0, so
        # the attribute above is not enough: ``_drift_budget`` is snapshotted in
        # ``__init__``. Without re-arming it the flag would silently inject
        # nothing and the guard would look trustworthy for the wrong reason.
        llm.reset_run()

    parser = make_parser(args.parser, llm=llm if args.parser in ("qwen-vl", "qwen", "vlm") else None, dpi=args.dpi)
    document = parser.parse(args.input).document

    chunking = ChunkingPipeline(ChunkOptions(max_tokens=args.max_tokens)).run(document)
    policy = RiskPolicy(threshold=args.threshold)
    run_dir = Path(args.out) if args.out else None
    pipeline = TranslationPipeline(
        llm,
        policy=policy,
        run_dir=run_dir,
        queue_path=(run_dir / "review.jsonl") if run_dir else None,
    )
    result = pipeline.run(chunking)

    print(f"document {document.doc_id}: {len(document.pages)} page(s) -> {len(chunking.chunks)} chunk(s)")
    print()
    print(render_run_summary(result))
    print(render_number_findings(result))
    print()
    print(glossary_report(chunking.glossary))

    if args.preview_tables:
        # Shows the table as *rebuilt* by the pipeline, not as the model returned
        # it -- i.e. the positional copy-through of every numeric cell, which is
        # the property the whole design rests on.
        print()
        print("translated tables (as rebuilt):")
        print(preview_tables(result))

    if run_dir:
        print(f"\nartefacts in {run_dir}/:")
        for name, path in sorted(result.outputs.items()):
            print(f"  {name:18} {path.name}")
        target = run_dir / "target.md"
        target.write_text(result.to_markdown(), encoding="utf-8")
        print(f"  {'target_document':18} {target.name}")
    if args.print_target:
        print("\n" + result.to_markdown())
    return 0


def _resolve_review_item(queue, ident: str):
    """Find a queue item by its queue id, or by the chunk id shown in the listing.

    The listing prints ``chunk_id`` because that is what a reviewer reasons
    about, and the chunk id is unique within a queue, so accepting it removes a
    pointless copy-and-paste step (or a wrong paste, since both ids are opaque).
    """
    item = queue.get(ident)
    if item is not None:
        return item
    matches = [candidate for candidate in queue if candidate.chunk_id == ident]
    return matches[0] if len(matches) == 1 else None


def cmd_review(args: argparse.Namespace) -> int:
    from .hitl.exporters import write_review_bundle
    from .hitl.queue import ReviewQueue

    queue = ReviewQueue(args.queue)
    if not len(queue):
        print(f"no items in {args.queue}")
        return 0

    # Resolve every requested id before applying any of them: a batch decision
    # that half-applies because one id was mistyped is worse than one that
    # refuses outright.
    requested: list[tuple[Any, str]] = []
    unknown: list[str] = []
    for status, identifiers in (
        ("approved", args.approve),
        ("rejected", args.reject),
        ("skipped", args.skip),
    ):
        for ident in identifiers or []:
            item = _resolve_review_item(queue, ident)
            if item is None:
                unknown.append(ident)
            else:
                requested.append((item, status))
    if unknown:
        print(f"error: no such review item: {', '.join(unknown)}", file=sys.stderr)
        return 2

    for item, status in requested:
        item.decide(status, reviewer=args.reviewer, note=args.note or "")
    if requested:
        queue.save(args.queue)
        print(f"updated {args.queue}")

    stats = queue.stats()
    print(f"\nreview queue {args.queue}")
    print(f"  total {stats['total']} · pending {stats['pending']} · "
          f"financial pending {stats['financial_pending']} · max risk {stats['max_risk']:.2f}")
    print(f"  by status {stats['by_status']}")

    if args.export:
        outputs = write_review_bundle(queue, args.export)
        print("\nexported:")
        for name, path in sorted(outputs.items()):
            print(f"  {name:10} {path}")

    items = queue.sorted_by_risk()
    if args.status:
        items = [i for i in items if i.status == args.status]
    limit = args.limit if args.limit is not None else 20
    print(f"\nitems (highest risk first, showing {min(limit, len(items))} of {len(items)}):")
    for item in items[:limit]:
        band = (item.risk or {}).get("band", "?")
        print(f"\n  [{band:7s}] {item.risk_value:.2f}  {item.chunk_id}  ({item.status})")
        # Printed because it is what --approve/--reject/--skip consume; without
        # it the documented flags are unusable from the terminal.
        print(f"           id {item.item_id}")
        print(f"           {' > '.join(item.section_path)}")
        if item.reviewer:
            print(f"           reviewer {item.reviewer}: {item.note}")
        for reason in item.top_reasons:
            print(f"           - {reason}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    llm = make_llm_client(args.llm) if args.parser in ("qwen-vl", "qwen", "vlm") else None
    parser = make_parser(args.parser, llm=llm, dpi=args.dpi)
    result = parser.parse(args.input)
    document = result.document

    print(f"== source layout ({parser.analyzer.name}) ==")
    for layout in result.layouts:
        from .parser.qwen_vl_analyzer import describe_layout

        print(describe_layout(layout))

    print(f"\n== blocks ({len(document.blocks())} total) ==")
    for page_index, block in document.iter_blocks():
        detail = ""
        if block.kind.value == "table":
            detail = f" {block.n_rows}x{block.n_cols} cells={len(block.cells)}"
        elif block.kind.value == "chart":
            detail = f" points={len(block.points)} method={block.extraction_method}"
        text = getattr(block, "text", None) or getattr(block, "caption", "")
        print(f"  p{page_index} [{block.kind.value:5s}]{detail}  {text[:70]!r}")

    if args.outline:
        from .chunker import heading_digest

        print("\n== outline ==")
        from .chunker.headings import assign_sections

        for line in heading_digest(assign_sections(list(document.iter_blocks()))):
            print(f"  {line}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Offline end-to-end run, including a deliberate corruption.

    The second half is the point: a deterministic mock is told to alter figures,
    and the guard must catch every one. If that ever stops happening, this
    command fails -- which makes "we validate the numbers" a test, not a claim.
    """
    from .chunker.pipeline import ChunkingPipeline
    from .demo import build_demo_document
    from .hitl.policy import RiskPolicy
    from .parser.pipeline import summarise_document
    from .translator.llm import MockLLM
    from .translator.pipeline import TranslationPipeline, render_number_findings, render_run_summary

    # Shared with the web demo (see art.demo) so the two cannot drift apart.
    fixture_path = args.fixture
    document = build_demo_document(fixture_path, jitter=args.jitter)
    print("=" * 78)
    print("STAGE 1 — PARSER")
    print("=" * 78)
    stats = summarise_document(document)
    for key, value in stats.items():
        print(f"  {key:30} {value}")
    for table in document.tables():
        merged = [c for c in table.cells if c.is_merged]
        print(
            f"\n  table on page {table.page}: {table.n_rows}x{table.n_cols}, "
            f"{len(table.cells)} cells, {len(merged)} merged, units={table.units_note!r}"
        )
        for cell in merged:
            print(
                f"    merged ({cell.row},{cell.col}) rowspan={cell.row_span} colspan={cell.col_span} "
                f"{cell.text[:40]!r}"
            )
        if table.conflicts:
            for conflict in table.conflicts:
                print(f"    conflict: {conflict.reason}")

    print()
    print("=" * 78)
    print("STAGE 2 — CHUNKER + GLOSSARY")
    print("=" * 78)
    chunking = ChunkingPipeline().run(document)
    for chunk in chunking.chunks:
        print(
            f"  {chunk.chunk_id:34} tok={chunk.token_estimate:5} "
            f"tables={len(chunk.tables)} terms={len(chunk.glossary_terms)}"
        )
    print()
    from .chunker.pipeline import glossary_report

    print(glossary_report(chunking.glossary, limit=6))

    # Two passes: the faithful mock, then the same run with N figures corrupted
    # on purpose. The second pass is the one that matters -- it turns "we check
    # the numbers" into an assertion. ``--fault 0`` means there is no second
    # pass, so it is skipped rather than run as a duplicate clean pass.
    passes: list[tuple[str, MockLLM, bool]] = [
        ("STAGE 3+4 — CLEAN RUN (faithful mock)", MockLLM(seed=1), False)
    ]
    if args.fault > 0:
        passes.append(
            (
                f"STAGE 3+4 — FAULT INJECTION ({args.fault} figure(s) corrupted deliberately)",
                MockLLM(inject_number_drift=args.fault, seed=3),
                True,
            )
        )

    for label, llm, injects in passes:
        print()
        print("=" * 78)
        print(label)
        print("=" * 78)
        run_dir = Path(args.out) / ("faulty" if injects else "clean") if args.out else None
        pipeline = TranslationPipeline(llm, policy=RiskPolicy(), run_dir=run_dir)
        result = pipeline.run(chunking)
        print(render_run_summary(result))
        print(render_number_findings(result))
        if run_dir:
            print(f"\n  artefacts written to {run_dir}/")
        if not injects:
            continue

        caught = len(result.aggregate_number_report().mismatched)
        applied = llm.corruption_applied
        # Compare against what the mock actually corrupted, not against the
        # requested count: a short fixture may hold fewer figures than --fault
        # asked for, and treating that as a miss would be a false alarm that
        # trains people to ignore the verdict line. A no-op on the mock's side,
        # on the other hand, is a real failure -- nothing was tested.
        print(f"\n  VERDICT: guard caught {caught} of {applied} injected corruption(s)")
        if applied < args.fault:
            print(
                f"  NOTE: only {applied} of {args.fault} corruption(s) could be injected "
                f"-- this document does not hold that many injectable figures"
            )
        if applied == 0:
            print("  FAILURE: nothing could be injected, so the guard was not exercised")
            return 1
        if caught < applied:
            print("  FAILURE: the number guard missed an injected corruption")
            return 1
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    """Serve the browser demo.

    Binds to 127.0.0.1 by default: this is a local demo, and a verification tool
    that quietly exposes a file-reading endpoint on a LAN interface is a bad
    trade for convenience.
    """
    try:
        import uvicorn
    except ImportError:
        print(
            "error: the web demo needs the optional extra --  pip install -e '.[web]'",
            file=sys.stderr,
        )
        return 2

    from .web.app import create_app

    target = f"http://{args.host}:{args.port}/"
    print(f"serving the demo at {target}  (Ctrl+C to stop)")
    if args.reload:
        uvicorn.run(
            "art.web.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="art",
        description="Verifiable multi-agent pipeline for translating layout-heavy annual reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  art demo                                  # offline, no credentials\n"
            "  art parse report.pdf --tables             # structure only\n"
            "  art translate examples/demo_annual_report.json --out runs/demo\n"
            "  art translate report.pdf --parser qwen-vl --llm openai-compat --out runs/live\n"
            "  art review runs/demo/review.jsonl --export runs/demo/sheet\n"
            "  art web                                   # browser demo, runs offline\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"art {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, *, need_input: bool = True) -> None:
        if need_input:
            p.add_argument("input", help="PDF, image, directory of images, or a recorded layout JSON")
        p.add_argument("--parser", default=env("ART_PARSER_BACKEND", "mock"),
                       choices=["mock", "qwen-vl", "paddle"], help="layout analysis backend")
        p.add_argument("--llm", default=env("ART_LLM_BACKEND", "mock"),
                       help="LLM backend: mock | openai-compat | auto")
        p.add_argument("--dpi", type=int, default=int(env("ART_PDF_DPI", "180")),
                       help="PDF rasterisation DPI (180 is the practical floor for dense tables)")

    p_parse = sub.add_parser("parse", help="stage 1: parse a document into the SDO")
    add_common(p_parse)
    p_parse.add_argument("-o", "--out", help="write the SDO JSON here")
    p_parse.add_argument("--record", help="record analyzer output here for offline replay")
    p_parse.add_argument("--json", action="store_true", help="print the SDO JSON to stdout")
    p_parse.add_argument("--tables", action="store_true", help="show every table with its grid")
    p_parse.set_defaults(func=cmd_parse)

    p_chunk = sub.add_parser("chunk", help="stage 1+2: parse and slice")
    add_common(p_chunk)
    p_chunk.add_argument("-o", "--out", help="write chunk metadata here")
    p_chunk.add_argument("--glossary", help="write the glossary here (re-runnable and hand-editable)")
    p_chunk.add_argument("--max-tokens", type=int, default=1400)
    p_chunk.add_argument("--split-level", type=int, default=2, help="heading level that forces a chunk boundary")
    p_chunk.add_argument("--allow-split-tables", action="store_true",
                         help="DANGEROUS: permit splitting a table across chunks")
    p_chunk.set_defaults(func=cmd_chunk)

    p_tr = sub.add_parser("translate", help="stages 1-4: full run")
    add_common(p_tr)
    p_tr.add_argument("-o", "--out", help="run directory for artefacts (SDO, target, review sheet)")
    p_tr.add_argument("--max-tokens", type=int, default=1400)
    p_tr.add_argument("--threshold", type=float, default=float(env("ART_HITL_THRESHOLD", "0.45")),
                      help="HITL risk threshold in [0,1]")
    p_tr.add_argument("--base-url", default=env("ART_LLM_BASE_URL", ""))
    p_tr.add_argument("--api-key", default=env("ART_LLM_API_KEY", ""))
    p_tr.add_argument("--model", default=env("ART_LLM_MODEL", ""))
    p_tr.add_argument("--inject-fault", type=int, default=0, metavar="N",
                      help="mock backend only: deliberately corrupt N figures, to prove the guard catches them")
    p_tr.add_argument("--print-target", action="store_true", help="print the translated document")
    p_tr.add_argument("--preview-tables", action="store_true",
                      help="print each translated table as rebuilt by the pipeline (numeric cells copied)")
    p_tr.set_defaults(func=cmd_translate)

    p_rev = sub.add_parser("review", help="inspect and decide review-queue items")
    p_rev.add_argument("queue", help="review.jsonl produced by `art translate`")
    p_rev.add_argument("--approve", action="append", metavar="ITEM_ID",
                       help="approve an item (queue id, or the chunk id shown in the listing)")
    p_rev.add_argument("--reject", action="append", metavar="ITEM_ID",
                       help="reject an item (queue id or chunk id)")
    p_rev.add_argument("--skip", action="append", metavar="ITEM_ID",
                       help="skip an item (queue id or chunk id)")
    p_rev.add_argument("--reviewer", default=env("ART_REVIEWER", ""))
    p_rev.add_argument("--note", default="")
    p_rev.add_argument("--status", choices=list(STATUSES), help="only show items with this status")
    p_rev.add_argument("--limit", type=int, default=20)
    p_rev.add_argument("--export", help="write the reviewer bundle (md/html/csv/jsonl) here")
    p_rev.set_defaults(func=cmd_review)

    p_ins = sub.add_parser("inspect", help="show detected layout, blocks and outline")
    add_common(p_ins)
    p_ins.add_argument("--outline", action="store_true", help="also print the section outline")
    p_ins.set_defaults(func=cmd_inspect)

    p_demo = sub.add_parser("demo", help="offline end-to-end run, no credentials required")
    p_demo.add_argument("--fixture", help="layout recording JSON (defaults to the built-in page)")
    p_demo.add_argument("-o", "--out", help="write run artefacts here")
    p_demo.add_argument("--fault", type=int, default=2, help="how many figures the faulty run corrupts")
    p_demo.add_argument("--jitter", type=float, default=2.5, help="simulate VLM box noise, in page units")
    p_demo.set_defaults(func=cmd_demo)

    p_web = sub.add_parser("web", help="serve the browser demo (needs the optional .[web] extra)")
    p_web.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    p_web.add_argument("--reload", action="store_true", help="development only: reload on change")
    p_web.set_defaults(func=cmd_web)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
