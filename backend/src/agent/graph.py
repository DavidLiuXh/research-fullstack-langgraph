import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from tavily import TavilyClient

from agent.configuration import Configuration
from agent.llm import create_deepseek_model
from agent.prompts import (
    answer_instructions,
    dimension_instructions,
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
)
from agent.state import (
    DimensionInput,
    DimensionState,
    OverallState,
    QueryGenerationState,
    WebSearchState,
)
from agent.tools_and_schemas import Reflection, ResearchDimensionList, SearchQueryList
from agent.utils import (
    deduplicate_sources,
    format_dimension_results,
    format_sources_for_research,
    get_research_topic,
    render_source_citations,
    tavily_results_to_sources,
)

load_dotenv()


def generate_research_dimensions(state: OverallState, config: RunnableConfig):
    """Decompose the main topic into independent, complementary dimensions."""
    configurable = Configuration.from_runnable_config(config)
    topic = get_research_topic(state["messages"])
    prompt = dimension_instructions.format(
        current_date=get_current_date(),
        number_dimensions=configurable.number_of_research_dimensions,
        research_topic=topic,
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
    return {"research_dimensions": dimensions}


def dispatch_research_dimensions(state: OverallState):
    """Run one isolated research subgraph for each dimension in parallel."""
    topic = get_research_topic(state["messages"])
    return [
        Send(
            "research_dimension",
            {
                "research_topic": topic,
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
    return {
        "research_topic": state["research_topic"],
        "dimension": state["dimension"],
        "search_query": result.query[:query_count],
        "research_loop_count": state.get("research_loop_count", 0),
    }


def dispatch_search_queries(state: QueryGenerationState):
    """Fan out the current dimension's search queries to Tavily."""
    return [
        Send(
            "web_research",
            {
                "search_query": query,
                "search_id": (
                    f"{state['dimension']['id']}-{state['research_loop_count']}-{index}"
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
    response = TavilyClient(api_key=api_key).search(
        query=state["search_query"],
        search_depth=configurable.tavily_search_depth,
        max_results=configurable.tavily_max_results,
        chunks_per_source=2,
        include_answer=False,
        include_raw_content=False,
        include_usage=True,
    )
    sources = tavily_results_to_sources(
        response, state["search_query"], state["search_id"]
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
    result = dimension_subgraph.invoke(state, config)
    dimension_result = {
        "dimension": result["dimension"],
        "research_content": "\n\n---\n\n".join(result["web_research_result"]),
        "sources": result["sources_gathered"],
        "research_loop_count": result["research_loop_count"],
        "is_sufficient": result["is_sufficient"],
    }
    return {
        "dimension_results": [dimension_result],
        "sources_gathered": result["sources_gathered"],
    }


def finalize_answer(state: OverallState, config: RunnableConfig):
    """Synthesize all completed dimensions into one cited report."""
    configurable = Configuration.from_runnable_config(config)
    model = state.get("reasoning_model") or configurable.answer_model
    sources = deduplicate_sources(state["sources_gathered"])
    prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        dimension_research=format_dimension_results(state["dimension_results"]),
    )
    result = create_deepseek_model(model, thinking=True).invoke(prompt)
    answer, _ = render_source_citations(str(result.content), sources)
    return {"messages": [AIMessage(content=answer)]}


builder = StateGraph(OverallState, config_schema=Configuration)
builder.add_node("generate_research_dimensions", generate_research_dimensions)
builder.add_node("research_dimension", research_dimension)
builder.add_node("finalize_answer", finalize_answer)
builder.add_edge(START, "generate_research_dimensions")
builder.add_conditional_edges(
    "generate_research_dimensions",
    dispatch_research_dimensions,
    ["research_dimension"],
)
builder.add_edge("research_dimension", "finalize_answer")
builder.add_edge("finalize_answer", END)
graph = builder.compile(name="deepseek-tavily-multidimensional-research-agent")
