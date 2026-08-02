from __future__ import annotations

import operator
from typing import NotRequired, TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


class ResearchDimension(TypedDict):
    id: str
    title: str
    scope: str


class ResearchSource(TypedDict):
    research_run_id: str
    source_id: str
    query: str
    title: str
    url: str
    content: str
    score: NotRequired[float | None]
    published_date: NotRequired[str | None]


class DimensionResult(TypedDict):
    research_run_id: str
    dimension: ResearchDimension
    research_content: str
    sources: list[ResearchSource]
    research_loop_count: int
    is_sufficient: bool


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    research_dimensions: list[ResearchDimension]
    research_run_id: str
    dimension_approved: bool
    dimension_feedback: str
    dimension_results: Annotated[list[DimensionResult], operator.add]
    sources_gathered: Annotated[list[ResearchSource], operator.add]
    initial_search_query_count: int
    max_research_loops: int
    reasoning_model: str


class DimensionState(TypedDict):
    research_run_id: str
    research_topic: str
    dimension: ResearchDimension
    current_knowledge_gap: str
    search_query: list[str]
    web_research_result: Annotated[list[str], operator.add]
    sources_gathered: Annotated[list[ResearchSource], operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    is_sufficient: bool


class DimensionInput(TypedDict):
    research_run_id: str
    research_topic: str
    dimension: ResearchDimension
    initial_search_query_count: int
    max_research_loops: int


class QueryGenerationState(TypedDict):
    research_run_id: str
    research_topic: str
    dimension: ResearchDimension
    search_query: list[str]
    research_loop_count: int


class WebSearchState(TypedDict):
    research_run_id: str
    search_query: str
    search_id: str
