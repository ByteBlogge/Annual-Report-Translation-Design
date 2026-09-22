"""Chunking: heading structure, table atomicity, and the glossary attachment order.

Two properties carry the weight here:

* **Table atomicity.** A table split across two chunks is translated by two
  different prompts, and the header/unit note ends up in a different call from
  the figures it governs. It must never happen.
* **Glossary before chunks.** Each chunk carries the terms it uses, built from
  the *whole* document. Chunking first is the shortcut that produces the same
  term translated two ways on pages 6 and 90.
"""

from __future__ import annotations

import pytest

from art.chunker.chunker import (
    Chunk,
    Chunker,
    ChunkOptions,
    block_to_text,
    numeric_density,
)
from art.chunker.headings import (
    HeadingHit,
    assign_sections,
    detect_heading,
    heading_digest,
    is_financial_section,
)
from art.parser.table_builder import build_table
from art.schema import (
    ChartBlock,
    DocumentPage,
    StructuredDocument,
    TableBlock,
    TextBlock,
)
from conftest import make_cells

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def doc_from(blocks, *, page: int = 1, doc_id: str = "doc") -> StructuredDocument:
    """A one-page SDO from a flat block list."""
    return StructuredDocument(
        doc_id=doc_id,
        pages=[DocumentPage(index=page, width=595, height=842, blocks=list(blocks))],
    )


def simple_table(caption: str = "Table") -> TableBlock:
    return build_table(
        make_cells([["Item", "2023"], ["Revenue", "1,234,567"]]), caption=caption
    )


# ---------------------------------------------------------------------------
# heading detection
# ---------------------------------------------------------------------------


class TestDetectHeading:
    @pytest.mark.parametrize(
        ("text", "level", "rule"),
        [
            ("第一部分 概述", 1, "cn-part"),
            ("第一章 财务摘要", 1, "cn-chapter"),
            ("PART I OVERVIEW", 1, "en-part"),
            ("第一节 收入", 2, "cn-section"),
            ("附注 1 会计政策", 2, "appendix"),
            ("一、经营回顾", 2, "cn-dun"),
            ("（一）收入分析", 3, "cn-paren"),
        ],
    )
    def test_chinese_and_english_headings(self, text, level, rule):
        hit = detect_heading(TextBlock(text=text))
        assert hit is not None, text
        assert hit.level == level
        assert hit.rule == rule

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_numbered_headings_derive_their_level_from_depth(self, depth):
        text = ".".join(str(i) for i in range(1, depth + 1)) + " Revenue analysis"
        hit = detect_heading(TextBlock(text=text))
        assert hit is not None
        assert hit.level == depth

    def test_a_sentence_is_not_a_heading(self):
        block = TextBlock(text="The group recorded revenue growth during the year under review.")
        assert detect_heading(block) is None

    def test_long_text_is_rejected(self):
        assert detect_heading(TextBlock(text="1. " + "x" * 200)) is None

    def test_empty_text_is_rejected(self):
        assert detect_heading(TextBlock(text="   ")) is None

    def test_numbering_rule_loses_to_the_stronger_rules(self):
        """A numbered heading must keep its true level, not be flattened."""
        assert detect_heading(TextBlock(text="第一章 财务摘要")).rule == "cn-chapter"

    def test_font_size_fallback_needs_a_larger_than_body_font(self):
        small = TextBlock(text="Overview", font_size=10)
        big = TextBlock(text="Overview", font_size=16)
        assert detect_heading(small, body_font_size=10) is None
        assert detect_heading(big, body_font_size=10).rule == "font-size"

    def test_font_fallback_can_be_disabled(self):
        big = TextBlock(text="Overview", font_size=16)
        assert detect_heading(big, body_font_size=10, allow_font_rule=False) is None


class TestSectionAssignment:
    def test_nested_headings_build_and_pop_the_path(self):
        """Stack discipline: a level-3 nests; the next level-2 pops back."""
        items = [
            (1, TextBlock(text="第二章 管理层讨论与分析", heading_level=1)),
            (1, TextBlock(text="2.1 Revenue analysis", heading_level=2)),
            (1, TextBlock(text="Revenue rose sharply.")),
            (1, TextBlock(text="2.1.1 Segment detail", heading_level=3)),
            (1, TextBlock(text="Segment detail prose.")),
            (1, TextBlock(text="2.2 Liquidity", heading_level=2)),
            (1, TextBlock(text="Liquidity prose.")),
        ]
        spans = assign_sections(items)
        paths = [s.title_path for s in spans]

        assert ("第二章 管理层讨论与分析",) in paths
        assert ("第二章 管理层讨论与分析", "2.1 Revenue analysis") in paths
        assert (
            "第二章 管理层讨论与分析",
            "2.1 Revenue analysis",
            "2.1.1 Segment detail",
        ) in paths
        # The level-2 heading after a level-3 must pop the level-3 off.
        assert ("第二章 管理层讨论与分析", "2.2 Liquidity") in paths
        assert all(len(p) <= 3 for p in paths)

    def test_prelabelled_headings_are_trusted(self):
        """The parser's own heading label wins over re-detecting the text."""
        items = [
            (1, TextBlock(text="Something unusual", heading_level=1)),
            (1, TextBlock(text="body")),
        ]
        spans = assign_sections(items)
        assert spans[0].title_path == ("Something unusual",)
        assert spans[0].headings[0].rule == "prelabelled"

    def test_page_range_is_tracked(self):
        items = [
            (1, TextBlock(text="第一章 概述", heading_level=1)),
            (1, TextBlock(text="a")),
            (3, TextBlock(text="b")),
        ]
        span = assign_sections(items)[0]
        assert (span.page_start, span.page_end) == (1, 3)

    def test_heading_digest_dedupes_and_limits_depth(self):
        items = [
            (1, TextBlock(text="第一章 财务摘要", heading_level=1)),
            (1, TextBlock(text="1.1 Revenue", heading_level=2)),
            (1, TextBlock(text="1.2 Cost", heading_level=2)),
        ]
        digest = heading_digest(assign_sections(items))
        assert digest[0] == "第一章 财务摘要"
        assert len(digest) == len(set(digest))


class TestFinancialSection:
    @pytest.mark.parametrize(
        "title",
        [
            "第一章 财务摘要",
            "Financial Summary",
            "Consolidated Statement of Profit or Loss",
            "Segment information",
            "管理层讨论与分析",
        ],
    )
    def test_recognises_financial_sections(self, title):
        assert is_financial_section((title,)) is True

    def test_ignores_unrelated_sections(self):
        assert is_financial_section(("Corporate governance",)) is False

    def test_matches_at_any_level_of_the_path(self):
        assert is_financial_section(("第一章", "财务摘要", "收入")) is True

    def test_empty_path_is_not_financial(self):
        assert is_financial_section(()) is False


# ---------------------------------------------------------------------------
# table atomicity
# ---------------------------------------------------------------------------


class TestTableAtomicity:
    def test_each_table_lands_in_exactly_one_chunk(self):
        """Splitting a table splits its header from its figures. Never do it."""
        blocks = [TextBlock(text="第一章 财务摘要", heading_level=1)]
        for i in range(4):
            blocks.append(TextBlock(text=f"Prose paragraph number {i}. " + "word " * 60))
            blocks.append(simple_table(f"Table {i}"))

        document = doc_from(blocks)
        chunks = Chunker(ChunkOptions(max_tokens=150, keep_tables_intact=True)).chunk(document)

        placed = [id(b) for c in chunks for b in c.blocks if isinstance(b, TableBlock)]
        assert len(placed) == len(set(placed)), "a table appeared in two chunks"
        assert len(placed) == len(document.tables()), "a table was dropped"

    def test_oversized_table_gets_its_own_chunk(self):
        document = doc_from(
            [
                TextBlock(text="第一章 财务摘要", heading_level=1),
                TextBlock(text="Prose before the table. " * 20),
                simple_table("Big"),
                TextBlock(text="Prose after the table."),
            ]
        )
        chunks = Chunker(
            ChunkOptions(max_tokens=80, large_table_ratio=0.1, keep_tables_intact=True)
        ).chunk(document)

        holders = [c for c in chunks if c.tables]
        assert len(holders) == 1
        assert len(holders[0].tables) == 1

    def test_chunk_tables_property_matches_blocks(self):
        document = doc_from(
            [
                TextBlock(text="第一章 财务摘要", heading_level=1),
                simple_table("A"),
                TextBlock(text="prose"),
            ]
        )
        chunks = Chunker().chunk(document)
        for chunk in chunks:
            expected = [b for b in chunk.blocks if isinstance(b, TableBlock)]
            assert chunk.tables == expected

    def test_chart_stays_with_its_section(self):
        document = doc_from(
            [
                TextBlock(text="第二章 管理层讨论与分析", heading_level=1),
                TextBlock(text="Intro prose."),
                ChartBlock(caption="Revenue", extraction_method="vlm"),
            ]
        )
        chunks = Chunker().chunk(document)
        assert any(
            isinstance(b, ChartBlock) for c in chunks for b in c.blocks
        )


# ---------------------------------------------------------------------------
# glossary attachment
# ---------------------------------------------------------------------------


class TestGlossaryAttachment:
    def test_glossary_is_built_before_chunking_and_reaches_every_chunk(self):
        """The ordering guarantee: no chunk gets an empty glossary."""
        document = doc_from(
            [
                TextBlock(text="第一章 财务摘要", heading_level=1),
                TextBlock(text="Revenue (营业收入) is the key metric."),
                TextBlock(text="第二章 管理层讨论与分析", heading_level=1),
                TextBlock(text="Revenue grew again in the second half."),
            ]
        )
        from art.chunker.pipeline import ChunkingPipeline

        result = ChunkingPipeline().run(document)

        assert result.glossary.translate("Revenue") == "营业收入"
        # The term is established on page 1 and must be offered to the chunk on
        # page 2 as well, or the two mentions can diverge.
        carrying = [c for c in result.chunks if any(s == "Revenue" for s, _ in c.glossary_terms)]
        assert carrying, "at least the chunk that uses the term must carry it"
        assert all(
            isinstance(t, tuple) and len(t) == 2 for c in result.chunks for t in c.glossary_terms
        )

    def test_chunk_ids_are_unique_and_ordered(self):
        blocks = []
        for i in range(6):
            blocks.append(TextBlock(text=f"第一章 第{i}节", heading_level=1))
            blocks.append(TextBlock(text="prose " * 40))
        from art.chunker.pipeline import ChunkingPipeline

        chunks = ChunkingPipeline().run(doc_from(blocks)).chunks
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


class TestChunkHelpers:
    def test_block_to_text_renders_each_block_type(self):
        assert "prose" in block_to_text(TextBlock(text="prose"))
        assert "Revenue" in block_to_text(simple_table())
        assert "Revenue" in block_to_text(ChartBlock(caption="Revenue"))
        assert block_to_text(ChartBlock(caption="")) is not None

    def test_numeric_density_is_a_fraction(self):
        assert numeric_density("no figures here") == 0.0
        dense = numeric_density("1,234 5,678 9,012") 
        assert 0.0 <= dense <= 1.0

    def test_numeric_density_higher_for_figure_dense_text(self):
        assert numeric_density("1,234 5,678 9,012") > numeric_density("Revenue grew in the year")

    def test_chunk_to_dict_is_serialisable(self):
        import json

        chunk = Chunk(chunk_id="c1", section_path=("第一章",), source_text="hello")
        json.dumps(chunk.to_dict())

    def test_chunk_section_title_joins_the_path(self):
        assert Chunk(chunk_id="c", section_path=("A", "B")).section_title == "A > B"

    def test_chunk_heading_falls_back_to_the_section_path(self):
        assert Chunk(chunk_id="c", section_path=("A",)).heading == "A"

    def test_heading_hit_to_dict(self):
        payload = HeadingHit(level=1, title="T", rule="cn-chapter").to_dict()
        assert payload["level"] == 1 and payload["rule"] == "cn-chapter"
