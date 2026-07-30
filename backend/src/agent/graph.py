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
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
)
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.tools_and_schemas import Reflection, SearchQueryList
from agent.utils import (
    deduplicate_sources,
    format_sources_for_research,
    get_research_topic,
    render_source_citations,
    tavily_results_to_sources,
)

load_dotenv()


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """Generate a bounded list of web search queries with DeepSeek."""
    configurable = Configuration.from_runnable_config(config)
    query_count = (
        state.get("initial_search_query_count")
        or configurable.number_of_initial_queries
    )
    prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        number_queries=query_count,
    )
    llm = create_deepseek_model(configurable.query_generator_model)
    result = llm.with_structured_output(SearchQueryList, method="json_mode").invoke(
        prompt
    )
    return {"search_query": result.query[:query_count]}


def continue_to_web_research(state: QueryGenerationState):
    """Fan out initial queries to independent Tavily searches."""
    return [
        Send("web_research", {"search_query": query, "id": f"0-{index}"})
        for index, query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Search Tavily and normalize its results into stable source records."""
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
    sources = tavily_results_to_sources(response, state["search_query"], state["id"])
    return {
        "sources_gathered": sources,
        "web_research_result": [format_sources_for_research(sources)],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """Judge research sufficiency and produce bounded follow-up queries."""
    configurable = Configuration.from_runnable_config(config)
    loop_count = state.get("research_loop_count", 0) + 1
    model = state.get("reasoning_model") or configurable.reflection_model
    prompt = reflection_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    llm = create_deepseek_model(model)
    result = llm.with_structured_output(Reflection, method="json_mode").invoke(prompt)
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries[
            : configurable.max_follow_up_queries
        ],
        "research_loop_count": loop_count,
    }


def evaluate_research(state: ReflectionState, config: RunnableConfig):
    """Route to finalization or fan out a bounded follow-up search round."""
    configurable = Configuration.from_runnable_config(config)
    max_loops = state.get("max_research_loops") or configurable.max_research_loops
    if (
        state["is_sufficient"]
        or state["research_loop_count"] >= max_loops
        or not state["follow_up_queries"]
    ):
        return "finalize_answer"
    return [
        Send(
            "web_research",
            {"search_query": query, "id": f"{state['research_loop_count']}-{index}"},
        )
        for index, query in enumerate(state["follow_up_queries"])
    ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """Synthesize the report and render only citations backed by Tavily sources."""
    configurable = Configuration.from_runnable_config(config)
    model = state.get("reasoning_model") or configurable.answer_model
    sources = deduplicate_sources(state["sources_gathered"])
    prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        summaries=format_sources_for_research(sources),
    )
    result = create_deepseek_model(model, thinking=True).invoke(prompt)
    answer, used_sources = render_source_citations(str(result.content), sources)
    return {"messages": [AIMessage(content=answer)], "sources_gathered": used_sources}


builder = StateGraph(OverallState, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
builder.add_edge("finalize_answer", END)
graph = builder.compile(name="deepseek-tavily-research-agent")
