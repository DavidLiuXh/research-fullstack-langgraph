from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int


class QueryGenerationState(TypedDict):
    search_query: list[str]


class WebSearchState(TypedDict):
    search_query: str
    id: str


class ResearchSource(TypedDict):
    source_id: str
    query: str
    title: str
    url: str
    content: str
    score: NotRequired[float | None]
    published_date: NotRequired[str | None]


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
