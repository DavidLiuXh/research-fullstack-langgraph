from research_agent.utils import (
    canonicalize_url,
    deduplicate_sources,
    deduplicate_sources_by_id,
    format_dimension_results,
    format_sources_for_research,
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
