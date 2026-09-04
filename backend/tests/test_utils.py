from research_agent.utils import (
    canonicalize_url,
    deduplicate_sources,
    deduplicate_sources_by_id,
    format_dimension_results,
    format_sources_for_research,
    locate_evidence_quote,
    normalize_search_score,
    render_source_citations,
    tavily_results_to_sources,
)


def test_tavily_results_are_normalized_and_invalid_rows_are_skipped():
    response = {
        "results": [
            {
                "title": "Example",
                "url": "https://example.com/article",
                "content": "A relevant fact.",
                "score": 0.9,
            },
            {"title": "Missing content", "url": "https://example.com/empty"},
        ]
    }

    sources = tavily_results_to_sources(response, "example query", "run-2-1", "run")

    assert sources == [
        {
            "research_run_id": "run",
            "source_id": "Srun-2-1-0",
            "query": "example query",
            "title": "Example",
            "url": "https://example.com/article",
            "canonical_url": "https://example.com/article",
            "domain": "example.com",
            "content": "A relevant fact.",
            "score": 0.9,
            "published_date": None,
        }
    ]


def test_evidence_quote_is_recovered_from_original_source_with_locator():
    content = "Header\nThe   authoritative value is 42 percent.\nFooter"

    result = locate_evidence_quote(content, "the authoritative value is 42 percent.")

    assert result == (
        "The   authoritative value is 42 percent.",
        "chars:7-47",
    )


def test_evidence_quote_rejects_missing_or_trivial_text():
    assert (
        locate_evidence_quote("A sufficiently long source body.", "not present") is None
    )
    assert locate_evidence_quote("A sufficiently long source body.", "source") is None


def test_sources_are_deduplicated_by_url():
    first = {
        "research_run_id": "run",
        "source_id": "S0-0-0",
        "query": "q1",
        "title": "First",
        "url": "https://example.com",
        "content": "one",
    }
    duplicate = {**first, "source_id": "S1-0-0", "query": "q2"}

    result = deduplicate_sources([first, duplicate])
    assert len(result) == 1
    assert result[0]["source_id"] == "S0-0-0"
    assert result[0]["canonical_url"] == "https://example.com/"


def test_deduplication_prefers_the_higher_quality_duplicate():
    weaker = {
        "research_run_id": "run",
        "source_id": "Sweak",
        "query": "q1",
        "title": "A sufficiently distinctive duplicated research title",
        "url": "https://example.com/report?utm_source=test",
        "content": "short",
        "score": 0.2,
    }
    stronger = {
        **weaker,
        "source_id": "Sstrong",
        "query": "q2",
        "url": "https://example.com/report",
        "content": "more complete evidence",
        "score": 0.9,
    }

    result = deduplicate_sources([weaker, stronger])

    assert len(result) == 1
    assert result[0]["source_id"] == "Sstrong"


def test_deduplication_preserves_gap_requirements_across_queries():
    first = {
        "research_run_id": "run",
        "source_id": "Sfirst",
        "query": "official query",
        "title": "A sufficiently distinctive primary source title",
        "url": "https://example.com/report",
        "content": "strong evidence",
        "score": 0.9,
        "gap_id": "gap-one",
        "gap_ids": ["gap-one"],
        "requested_source_types": ["government"],
        "expected_evidence": "Official statistic",
    }
    duplicate = {
        **first,
        "source_id": "Ssecond",
        "query": "independent query",
        "score": 0.8,
        "gap_id": "gap-two",
        "gap_ids": ["gap-two"],
        "requested_source_types": ["academic"],
        "expected_evidence": "Independent comparison",
    }

    result = deduplicate_sources([first, duplicate])

    assert len(result) == 1
    assert result[0]["gap_ids"] == ["gap-one", "gap-two"]
    assert result[0]["requested_source_types"] == ["academic", "government"]
    assert result[0]["expected_evidence"] == (
        "Official statistic | Independent comparison"
    )


def test_long_unicode_titles_are_deduplicated():
    first = {
        "research_run_id": "run",
        "source_id": "S1",
        "query": "q",
        "title": "中国新能源汽车产业发展年度研究报告完整版",
        "url": "https://one.example/report",
        "content": "one",
    }
    repost = {
        **first,
        "source_id": "S2",
        "url": "https://two.example/repost",
    }

    assert len(deduplicate_sources([first, repost])) == 1


def test_citation_deduplication_preserves_same_url_with_different_ids():
    first = {
        "research_run_id": "run",
        "source_id": "Sdimension-0",
        "query": "q1",
        "title": "Shared source",
        "url": "https://example.com/shared",
        "content": "evidence",
    }
    second = {**first, "source_id": "Sdimension-1", "query": "q2"}

    result = deduplicate_sources_by_id([first, second, first])

    assert [source["source_id"] for source in result] == [
        "Sdimension-0",
        "Sdimension-1",
    ]


def test_url_canonicalization_removes_tracking_and_fragments():
    assert (
        canonicalize_url(
            "HTTPS://Example.com/report/?utm_source=test&year=2026#details"
        )
        == "https://example.com/report?year=2026"
    )


def test_invalid_urls_and_scores_are_safely_normalized():
    assert canonicalize_url("javascript:alert(1)") == ""
    assert canonicalize_url("https://example.com:invalid/report") == ""
    assert normalize_search_score("not-a-number") == 0.5
    assert normalize_search_score(float("nan")) == 0.5
    assert normalize_search_score(2.5) == 1.0


def test_only_known_source_markers_are_rendered():
    source = {
        "research_run_id": "run",
        "source_id": "S0-0-0",
        "query": "q",
        "title": "Example [Site]",
        "url": "https://example.com",
        "content": "fact",
    }

    answer, used = render_source_citations(
        "Supported [S0-0-0]. Unknown [S9-9-9].", [source]
    )

    assert answer == "Supported [Example Site](https://example.com). Unknown ."
    assert used == [source]


def test_sources_are_formatted_with_stable_ids():
    source = {
        "research_run_id": "run",
        "source_id": "S0-0-0",
        "query": "q",
        "title": "Example",
        "url": "https://example.com",
        "content": "fact",
        "published_date": "2026-07-30",
    }

    rendered = format_sources_for_research([source])

    assert "[S0-0-0]" in rendered
    assert "Published: 2026-07-30" in rendered
    assert "Content: fact" in rendered


def test_dimension_results_are_grouped_in_dimension_order():
    results = [
        {
            "research_run_id": "run",
            "dimension": {"id": "1", "title": "Technology", "scope": "Tech"},
            "research_content": "second",
            "sources": [],
            "research_loop_count": 2,
            "is_sufficient": False,
        },
        {
            "research_run_id": "run",
            "dimension": {"id": "0", "title": "Market", "scope": "Market"},
            "research_content": "first",
            "sources": [],
            "research_loop_count": 1,
            "is_sufficient": True,
        },
    ]

    rendered = format_dimension_results(results)

    assert rendered.index("Dimension 0: Market") < rendered.index(
        "Dimension 1: Technology"
    )
    assert "loop_limit_reached after 2 loop(s)" in rendered


def test_dimension_results_include_claim_excerpt_but_not_full_source_content():
    results = [
        {
            "research_run_id": "run",
            "dimension": {"id": "0", "title": "Market", "scope": "Market"},
            "research_content": "summary",
            "sources": [
                {
                    "source_id": "S1",
                    "title": "Source",
                    "url": "https://example.com",
                    "content": "FULL SOURCE CONTENT MUST NOT BE REPEATED",
                }
            ],
            "research_loop_count": 1,
            "is_sufficient": True,
            "claims": [
                {
                    "claim": "Supported claim",
                    "supporting_source_ids": ["S1"],
                    "supporting_evidence": "Compact evidence excerpt.",
                    "contradicting_source_ids": [],
                    "confidence": 0.5,
                    "uncertainty_reason": "",
                }
            ],
        }
    ]

    rendered = format_dimension_results(results)

    assert "Compact evidence excerpt." in rendered
    assert "FULL SOURCE CONTENT MUST NOT BE REPEATED" not in rendered


def test_dimension_results_render_a_human_readable_stop_reason():
    results = [
        {
            "research_run_id": "run",
            "dimension": {"id": "0", "title": "Policy", "scope": "Policy"},
            "research_content": "",
            "sources": [],
            "research_loop_count": 4,
            "is_sufficient": False,
            "completion_status": "completed_with_limitations",
            "termination_reason": "no_progress",
            "claims": [],
        }
    ]

    rendered = format_dimension_results(results)

    assert "completed with limitations" in rendered
    assert "stop reason: no progress" in rendered
    assert "budget_exhausted" not in rendered
