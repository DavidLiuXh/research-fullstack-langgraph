# ruff: noqa: E402

import os
import sys
import time
from pathlib import Path
from uuid import uuid4

# The CLI executes this file by path, so the src-layout package root is not
# guaranteed to be importable unless the project has already been installed.
project_src = str(Path(__file__).resolve().parents[1])
if project_src not in sys.path:
    sys.path.insert(0, project_src)

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from tavily import TavilyClient

from research_agent.configuration import Configuration
from research_agent.llm import create_deepseek_model
from research_agent.prompts import (
    answer_instructions,
    dimension_instructions,
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
)
from research_agent.state import (
    DimensionInput,
    DimensionState,
    OverallState,
    QueryGenerationState,
    WebSearchState,
)
from research_agent.tools_and_schemas import Reflection, ResearchDimensionList, SearchQueryList
from research_agent.utils import (
    deduplicate_sources,
    format_dimension_results,
    format_sources_for_research,
    get_research_topic,
    render_source_citations,
    tavily_results_to_sources,
)

load_dotenv()


def emit_research_event(event_type: str, **data):
    """Emit progress that remains visible while nested subgraphs are running."""
    get_stream_writer()({"type": event_type, **data})


def generate_research_dimensions(state: OverallState, config: RunnableConfig):
    """Decompose the main topic into independent, complementary dimensions."""
    configurable = Configuration.from_runnable_config(config)
    topic = get_research_topic(state["messages"])
    research_run_id = uuid4().hex[:12]
    emit_research_event("planning_dimensions", message="Planning research dimensions")
    prompt = dimension_instructions.format(
        current_date=get_current_date(),
        number_dimensions=configurable.number_of_research_dimensions,
        research_topic=topic,
        previous_dimensions="\n".join(
            f"- {item['title']}: {item['scope']}"
            for item in state.get("research_dimensions", [])
        )
        or "None; this is the first proposal.",
        human_feedback=state.get("dimension_feedback")
        or "None; this is the first proposal.",
    )
    llm = create_deepseek_model(configurable.query_generator_model)
    result = llm.with_structured_output(
        ResearchDimensionList, method="json_mode"
    ).invoke(prompt)
    dimensions = [
        {"id": str(index), "title": item.title, "scope": item.scope}
        for index, item in enumerate(
            result.dimensions[: configurable.number_of_research_dimensions]
        )
    ]
    if not dimensions:
        raise ValueError("DeepSeek did not generate any research dimensions")
    emit_research_event("dimensions_created", dimensions=dimensions)
    return {
        "research_run_id": research_run_id,
        "research_dimensions": dimensions,
        "dimension_approved": False,
    }


def review_research_dimensions(state: OverallState):
    """Pause until a human approves the dimensions or supplies revision feedback."""
    decision = interrupt(
        {
            "type": "research_dimension_review",
            "research_run_id": state["research_run_id"],
            "dimensions": state["research_dimensions"],
            "message": "Review the proposed research dimensions before research begins.",
        }
    )
    if not isinstance(decision, dict) or not isinstance(
        decision.get("approved"), bool
    ):
        raise ValueError("Dimension review must include a boolean 'approved' value")

    approved = decision["approved"]
    feedback = str(decision.get("feedback", "")).strip()
    if not approved and not feedback:
        raise ValueError("Revision feedback is required when dimensions are rejected")

    emit_research_event(
        "dimensions_reviewed",
        research_run_id=state["research_run_id"],
        approved=approved,
        feedback=feedback,
    )
    return {"dimension_approved": approved, "dimension_feedback": feedback}


def route_dimension_review(state: OverallState):
    """Regenerate rejected dimensions; dispatch approved dimensions for research."""
    if not state["dimension_approved"]:
        return "generate_research_dimensions"
    return dispatch_research_dimensions(state)


def dispatch_research_dimensions(state: OverallState):
    """Run one isolated research subgraph for each dimension in parallel."""
    topic = get_research_topic(state["messages"])
    return [
        Send(
            "research_dimension",
            {
                "research_topic": topic,
                "research_run_id": state["research_run_id"],
                "dimension": dimension,
                "initial_search_query_count": state.get(
                    "initial_search_query_count", 3
                ),
                "max_research_loops": state.get("max_research_loops", 2),
            },
        )
        for dimension in state["research_dimensions"]
    ]


def generate_query(
    state: DimensionState, config: RunnableConfig
) -> QueryGenerationState:
    """Generate searches for a dimension and its latest knowledge gap."""
    configurable = Configuration.from_runnable_config(config)
    query_count = (
        state.get("initial_search_query_count")
        or configurable.number_of_initial_queries
    )
    prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=state["research_topic"],
        dimension_title=state["dimension"]["title"],
        dimension_scope=state["dimension"]["scope"],
        knowledge_gap=state.get("current_knowledge_gap")
        or "None; this is the first pass.",
        number_queries=query_count,
    )
    llm = create_deepseek_model(configurable.query_generator_model)
    result = llm.with_structured_output(SearchQueryList, method="json_mode").invoke(
        prompt
    )
    queries = result.query[:query_count]
    emit_research_event(
        "queries_generated",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        queries=queries,
        loop=state.get("research_loop_count", 0),
    )
    return {
        "research_run_id": state["research_run_id"],
        "research_topic": state["research_topic"],
        "dimension": state["dimension"],
        "search_query": queries,
        "research_loop_count": state.get("research_loop_count", 0),
    }


def dispatch_search_queries(state: QueryGenerationState):
    """Fan out the current dimension's search queries to Tavily."""
    return [
        Send(
            "web_research",
            {
                "search_query": query,
                "research_run_id": state["research_run_id"],
                "search_id": (
                    f"{state['research_run_id']}-{state['dimension']['id']}-"
                    f"{state['research_loop_count']}-{index}"
                ),
            },
        )
        for index, query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> DimensionState:
    """Search Tavily and normalize evidence under stable source IDs."""
    configurable = Configuration.from_runnable_config(config)
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set")
    emit_research_event(
        "search_started",
        research_run_id=state["research_run_id"],
        query=state["search_query"],
    )
    response = None
    last_error: Exception | None = None
    for attempt in range(configurable.tavily_max_retries + 1):
        try:
            response = TavilyClient(api_key=api_key).search(
                query=state["search_query"],
                search_depth=configurable.tavily_search_depth,
                max_results=configurable.tavily_max_results,
                chunks_per_source=2,
                include_answer=False,
                include_raw_content=False,
                include_usage=True,
            )
            break
        except Exception as error:  # Tavily exposes multiple transport exceptions.
            last_error = error
            if attempt < configurable.tavily_max_retries:
                emit_research_event(
                    "search_retrying",
                    query=state["search_query"],
                    attempt=attempt + 1,
                )
                time.sleep(min(2**attempt, 4))

    if response is None:
        emit_research_event(
            "search_failed",
            query=state["search_query"],
            error=str(last_error) if last_error else "Unknown Tavily error",
        )
        return {
            "sources_gathered": [],
            "web_research_result": [
                f"Search failed for query: {state['search_query']}. No evidence was added."
            ],
        }

    sources = tavily_results_to_sources(
        response,
        state["search_query"],
        state["search_id"],
        state["research_run_id"],
    )
    emit_research_event(
        "search_completed",
        query=state["search_query"],
        source_count=len(sources),
        sources=sources,
    )
    return {
        "sources_gathered": sources,
        "web_research_result": [format_sources_for_research(sources)],
    }


def reflection(state: DimensionState, config: RunnableConfig):
    """Evaluate a dimension and return only the next knowledge gap."""
    configurable = Configuration.from_runnable_config(config)
    loop_count = state.get("research_loop_count", 0) + 1
    prompt = reflection_instructions.format(
        research_topic=state["research_topic"],
        dimension_title=state["dimension"]["title"],
        dimension_scope=state["dimension"]["scope"],
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    llm = create_deepseek_model(configurable.reflection_model)
    result = llm.with_structured_output(Reflection, method="json_mode").invoke(prompt)
    emit_research_event(
        "reflection_completed",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        is_sufficient=result.is_sufficient,
        knowledge_gap=result.knowledge_gap,
        loop=loop_count,
    )
    return {
        "is_sufficient": result.is_sufficient,
        "current_knowledge_gap": result.knowledge_gap,
        "research_loop_count": loop_count,
    }


def route_dimension_research(state: DimensionState):
    """Return gaps to query generation until the dimension is done."""
    if (
        state["is_sufficient"]
        or state["research_loop_count"] >= state["max_research_loops"]
    ):
        return END
    return "generate_query"


dimension_builder = StateGraph(DimensionState, input_schema=DimensionInput)
dimension_builder.add_node("generate_query", generate_query)
dimension_builder.add_node("web_research", web_research)
dimension_builder.add_node("reflection", reflection)
dimension_builder.add_edge(START, "generate_query")
dimension_builder.add_conditional_edges(
    "generate_query", dispatch_search_queries, ["web_research"]
)
dimension_builder.add_edge("web_research", "reflection")
dimension_builder.add_conditional_edges(
    "reflection", route_dimension_research, ["generate_query", END]
)
dimension_subgraph = dimension_builder.compile(name="dimension-research-subgraph")


def research_dimension(state: DimensionInput, config: RunnableConfig):
    """Adapt parent state into the dimension subgraph and collect its output."""
    result = None
    for stream_mode, chunk in dimension_subgraph.stream(
        state, config, stream_mode=["custom", "values"]
    ):
        if stream_mode == "custom":
            get_stream_writer()(chunk)
        elif stream_mode == "values":
            result = chunk
    if result is None:
        raise RuntimeError("Dimension subgraph completed without a final state")
    dimension_result = {
        "research_run_id": result["research_run_id"],
        "dimension": result["dimension"],
        "research_content": "\n\n---\n\n".join(result["web_research_result"]),
        "sources": result["sources_gathered"],
        "research_loop_count": result["research_loop_count"],
        "is_sufficient": result["is_sufficient"],
    }
    emit_research_event(
        "dimension_completed",
        research_run_id=result["research_run_id"],
        dimension=result["dimension"],
        is_sufficient=result["is_sufficient"],
        loops=result["research_loop_count"],
    )
    return {
        "dimension_results": [dimension_result],
        "sources_gathered": result["sources_gathered"],
    }


def finalize_answer(state: OverallState, config: RunnableConfig):
    """Synthesize all completed dimensions into one cited report."""
    configurable = Configuration.from_runnable_config(config)
    model = state.get("reasoning_model") or configurable.answer_model
    current_results = [
        result
        for result in state["dimension_results"]
        if result["research_run_id"] == state["research_run_id"]
    ]
    current_sources = [
        source
        for source in state["sources_gathered"]
        if source["research_run_id"] == state["research_run_id"]
    ]
    sources = deduplicate_sources(current_sources)
    emit_research_event("finalizing_answer", research_run_id=state["research_run_id"])
    prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        dimension_research=format_dimension_results(current_results),
    )
    result = create_deepseek_model(model, thinking=True).invoke(prompt)
    answer, _ = render_source_citations(str(result.content), sources)
    return {"messages": [AIMessage(content=answer)]}


builder = StateGraph(OverallState, config_schema=Configuration)
builder.add_node("generate_research_dimensions", generate_research_dimensions)
builder.add_node("review_research_dimensions", review_research_dimensions)
builder.add_node("research_dimension", research_dimension)
builder.add_node("finalize_answer", finalize_answer)
builder.add_edge(START, "generate_research_dimensions")
builder.add_edge("generate_research_dimensions", "review_research_dimensions")
builder.add_conditional_edges(
    "review_research_dimensions",
    route_dimension_review,
    ["generate_research_dimensions", "research_dimension"],
)
builder.add_edge("research_dimension", "finalize_answer")
builder.add_edge("finalize_answer", END)
graph = builder.compile(name="deepseek-tavily-multidimensional-research-agent")
