import importlib
import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from research_agent.graph import (
    dispatch_research_dimensions,
    finalize_answer,
    graph,
    review_research_dimensions,
    route_dimension_review,
    route_dimension_research,
    web_research,
)
from research_agent.tools_and_schemas import (
    Reflection,
    ResearchDimension,
    ResearchDimensionList,
    SearchQueryList,
)


def _dimension_state(**overrides):
    state = {
        "research_topic": "topic",
        "research_run_id": "run",
        "dimension": {"id": "0", "title": "Market", "scope": "scope"},
        "current_knowledge_gap": "missing evidence",
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "initial_search_query_count": 2,
        "max_research_loops": 3,
        "research_loop_count": 1,
        "is_sufficient": False,
    }
    state.update(overrides)
    return state


def test_dimension_gap_routes_back_to_query_generation():
    assert route_dimension_research(_dimension_state()) == "generate_query"


def test_sufficient_dimension_routes_to_end():
    assert route_dimension_research(_dimension_state(is_sufficient=True)) == END


def test_dimension_stops_at_loop_limit():
    assert route_dimension_research(_dimension_state(research_loop_count=3)) == END


def test_parent_dispatches_isolated_dimension_inputs():
    state = {
        "messages": [],
        "research_run_id": "run",
        "research_dimensions": [
            {"id": "0", "title": "Market", "scope": "market scope"},
            {"id": "1", "title": "Technology", "scope": "tech scope"},
        ],
        "initial_search_query_count": 2,
        "max_research_loops": 3,
    }

    sends = dispatch_research_dimensions(state)

    assert len(sends) == 2
    assert sends[0].arg["dimension"]["id"] == "0"
    assert sends[1].arg["dimension"]["id"] == "1"
    assert sends[0].arg is not sends[1].arg


def test_dimension_review_rejection_requires_regeneration(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda value: {"approved": False, "feedback": "Add regulation"},
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)

    result = review_research_dimensions(
        {
            "research_run_id": "run",
            "research_dimensions": [
                {"id": "0", "title": "Market", "scope": "market scope"}
            ],
        }
    )

    assert result == {
        "dimension_approved": False,
        "dimension_feedback": "Add regulation",
    }
    assert (
        route_dimension_review({"dimension_approved": False})
        == "generate_research_dimensions"
    )


def test_dimension_review_approval_dispatches_research(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda value: {"approved": True, "feedback": ""},
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    state = {
        "messages": [],
        "research_run_id": "run",
        "research_dimensions": [
            {"id": "0", "title": "Market", "scope": "market scope"}
        ],
        "initial_search_query_count": 1,
        "max_research_loops": 1,
    }

    result = review_research_dimensions(state)
    routed = route_dimension_review({**state, **result})

    assert result["dimension_approved"] is True
    assert len(routed) == 1
    assert routed[0].node == "research_dimension"


def test_compiled_parent_graph_has_dimension_pipeline():
    assert graph.name == "deepseek-tavily-multidimensional-research-agent"
    assert {
        "generate_research_dimensions",
        "review_research_dimensions",
        "research_dimension",
        "finalize_answer",
    }.issubset(graph.nodes)


def test_parent_graph_runs_parallel_dimension_subgraphs(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeModel:
        schema = None

        def with_structured_output(self, schema, method):
            assert method == "json_mode"
            self.schema = schema
            return self

        def invoke(self, prompt):
            if self.schema is ResearchDimensionList:
                return ResearchDimensionList(
                    dimensions=[
                        ResearchDimension(title="Market", scope="market scope"),
                        ResearchDimension(title="Technology", scope="tech scope"),
                    ]
                )
            if self.schema is SearchQueryList:
                query = "market query" if "Market" in prompt else "technology query"
                return SearchQueryList(query=[query], rationale="test")
            if self.schema is Reflection:
                return Reflection(is_sufficient=True, knowledge_gap="")
            source_ids = list(
                dict.fromkeys(re.findall(r"\[(S[A-Za-z0-9-]+)\]", prompt))
            )
            return AIMessage(
                content=" ".join(f"Evidence [{item}]." for item in source_ids)
            )

    class FakeTavilyClient:
        def __init__(self, api_key):
            assert api_key == "test-tavily-key"

        def search(self, query, **kwargs):
            slug = query.replace(" ", "-")
            return {
                "results": [
                    {
                        "title": query.title(),
                        "url": f"https://example.com/{slug}",
                        "content": f"Evidence for {query}",
                        "score": 0.9,
                    }
                ]
            }

    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FakeModel()
    )
    monkeypatch.setattr(graph_module, "TavilyClient", FakeTavilyClient)
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda value: {"approved": True, "feedback": ""},
    )

    graph_input = {
        "messages": [HumanMessage(content="Research this topic")],
        "initial_search_query_count": 1,
        "max_research_loops": 1,
        "reasoning_model": "deepseek-v4-pro",
    }
    result = graph.invoke(graph_input)

    assert len(result["dimension_results"]) == 2
    assert {item["dimension"]["title"] for item in result["dimension_results"]} == {
        "Market",
        "Technology",
    }
    assert "https://example.com/market-query" in result["messages"][-1].content
    assert "https://example.com/technology-query" in result["messages"][-1].content

    custom_events = list(graph.stream(graph_input, stream_mode="custom"))
    event_types = {event["type"] for event in custom_events}
    assert {
        "planning_dimensions",
        "dimensions_created",
        "dimensions_reviewed",
        "queries_generated",
        "search_started",
        "search_completed",
        "reflection_completed",
        "dimension_completed",
        "finalizing_answer",
    }.issubset(event_types)


def test_tavily_failure_degrades_to_empty_evidence(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FailingTavilyClient:
        def __init__(self, api_key):
            pass

        def search(self, query, **kwargs):
            raise TimeoutError("Tavily timed out")

    events = []
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(graph_module, "TavilyClient", FailingTavilyClient)
    monkeypatch.setattr(
        graph_module,
        "emit_research_event",
        lambda event_type, **data: events.append({"type": event_type, **data}),
    )

    result = web_research(
        {
            "research_run_id": "run",
            "search_query": "unavailable query",
            "search_id": "run-0-0-0",
        },
        {"configurable": {"tavily_max_retries": 0}},
    )

    assert result["sources_gathered"] == []
    assert "Search failed" in result["web_research_result"][0]
    assert events[-1]["type"] == "search_failed"


def test_final_answer_uses_only_current_research_run(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    captured_prompts = []

    class FakeFinalModel:
        def invoke(self, prompt):
            captured_prompts.append(prompt)
            return AIMessage(content="Current fact [Snew-0-0-0-0].")

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FakeFinalModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)

    def dimension_result(run_id, content):
        return {
            "research_run_id": run_id,
            "dimension": {"id": "0", "title": "Market", "scope": "scope"},
            "research_content": content,
            "sources": [],
            "research_loop_count": 1,
            "is_sufficient": True,
        }

    def source(run_id, source_id, content):
        return {
            "research_run_id": run_id,
            "source_id": source_id,
            "query": "query",
            "title": f"{run_id} source",
            "url": f"https://example.com/{run_id}",
            "content": content,
        }

    result = finalize_answer(
        {
            "messages": [HumanMessage(content="Current question")],
            "research_run_id": "new",
            "dimension_results": [
                dimension_result("old", "OLD DIMENSION CONTENT"),
                dimension_result("new", "NEW DIMENSION CONTENT"),
            ],
            "sources_gathered": [
                source("old", "Sold-0-0-0-0", "OLD SOURCE CONTENT"),
                source("new", "Snew-0-0-0-0", "NEW SOURCE CONTENT"),
            ],
            "reasoning_model": "deepseek-v4-pro",
        },
        {},
    )

    assert "NEW DIMENSION CONTENT" in captured_prompts[0]
    assert "OLD DIMENSION CONTENT" not in captured_prompts[0]
    assert "https://example.com/new" in result["messages"][0].content
