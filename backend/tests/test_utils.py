from agent.utils import (
    deduplicate_sources,
    format_dimension_results,
    format_sources_for_research,
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

    sources = tavily_results_to_sources(response, "example query", "2-1")

    assert sources == [
        {
            "source_id": "S2-1-0",
            "query": "example query",
            "title": "Example",
            "url": "https://example.com/article",
            "content": "A relevant fact.",
            "score": 0.9,
            "published_date": None,
        }
    ]


def test_sources_are_deduplicated_by_url():
    first = {
        "source_id": "S0-0-0",
        "query": "q1",
        "title": "First",
        "url": "https://example.com",
        "content": "one",
    }
    duplicate = {**first, "source_id": "S1-0-0", "query": "q2"}

    assert deduplicate_sources([first, duplicate]) == [first]


def test_only_known_source_markers_are_rendered():
    source = {
        "source_id": "S0-0-0",
        "query": "q",
        "title": "Example [Site]",
        "url": "https://example.com",
        "content": "fact",
    }

    answer, used = render_source_citations(
        "Supported [S0-0-0]. Unknown [S9-9-9].", [source]
    )

    assert answer == (
        "Supported [Example Site](https://example.com). Unknown [S9-9-9]."
    )
    assert used == [source]


def test_sources_are_formatted_with_stable_ids():
    source = {
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
            "dimension": {"id": "1", "title": "Technology", "scope": "Tech"},
            "research_content": "second",
            "sources": [],
            "research_loop_count": 2,
            "is_sufficient": False,
        },
        {
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
    assert "loop limit reached after 2 loop(s)" in rendered
