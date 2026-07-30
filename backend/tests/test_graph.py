import importlib

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from agent.graph import (
    dispatch_research_dimensions,
    graph,
    route_dimension_research,
)
from agent.tools_and_schemas import (
    Reflection,
    ResearchDimension,
    ResearchDimensionList,
    SearchQueryList,
)


def _dimension_state(**overrides):
    state = {
        "research_topic": "topic",
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


def test_compiled_parent_graph_has_dimension_pipeline():
    assert graph.name == "deepseek-tavily-multidimensional-research-agent"
    assert {
        "generate_research_dimensions",
        "research_dimension",
        "finalize_answer",
    }.issubset(graph.nodes)


def test_parent_graph_runs_parallel_dimension_subgraphs(monkeypatch):
    graph_module = importlib.import_module("agent.graph")

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
            return AIMessage(content="Market [S0-0-0-0]. Technology [S1-0-0-0].")

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

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Research this topic")],
            "initial_search_query_count": 1,
            "max_research_loops": 1,
            "reasoning_model": "deepseek-v4-pro",
        }
    )

    assert len(result["dimension_results"]) == 2
    assert {item["dimension"]["title"] for item in result["dimension_results"]} == {
        "Market",
        "Technology",
    }
    assert "https://example.com/market-query" in result["messages"][-1].content
    assert "https://example.com/technology-query" in result["messages"][-1].content
