from __future__ import annotations

import operator
from typing import NotRequired, TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


class ResearchDimension(TypedDict):
    id: str
    title: str
    scope: str


class ClarificationTurn(TypedDict):
    questions: list[str]
    response: str


class ResearchSource(TypedDict):
    research_run_id: str
    source_id: str
    query: str
    title: str
    url: str
    content: str
    score: NotRequired[float | None]
    published_date: NotRequired[str | None]
    canonical_url: NotRequired[str]
    domain: NotRequired[str]
    source_type: NotRequired[str]
    authority_score: NotRequired[float]
    relevance_score: NotRequired[float]
    recency_score: NotRequired[float]
    evidence_score: NotRequired[float]
    is_primary_source: NotRequired[bool]
    is_likely_repost: NotRequired[bool]
    supported_topics: NotRequired[list[str]]
    rejection_reasons: NotRequired[list[str]]
    quality_status: NotRequired[str]


class EvidenceClaim(TypedDict):
    claim: str
    supporting_source_ids: list[str]
    supporting_evidence: str
    contradicting_source_ids: list[str]
    confidence: float
    uncertainty_reason: str


class DimensionResult(TypedDict):
    research_run_id: str
    dimension: ResearchDimension
    research_content: str
    sources: list[ResearchSource]
    research_loop_count: int
    is_sufficient: bool
    completion_status: str
    covered_questions: list[str]
    unresolved_gaps: list[dict]
    contradictions: list[dict]
    source_quality_issues: list[str]
    confidence: float
    claims: list[EvidenceClaim]


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    original_research_topic: str
    normalized_research_topic: str
    topic_needs_clarification: bool
    topic_ambiguities: list[str]
    topic_clarification_questions: list[str]
    topic_assumptions: list[str]
    topic_clarification_reason: str
    topic_clarification_history: list[ClarificationTurn]
    topic_clarification_action: str
    research_dimensions: list[ResearchDimension]
    research_run_id: str
    dimension_approved: bool
    dimension_feedback: str
    dimension_results: Annotated[list[DimensionResult], operator.add]
    sources_gathered: Annotated[list[ResearchSource], operator.add]
    initial_search_query_count: int
    max_research_loops: int
    reasoning_model: str
    report_draft: str
    report_audit: dict
    report_revision_count: int
    max_report_revisions: int


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
    query_history: Annotated[list[str], operator.add]
    evaluated_sources: list[ResearchSource]
    selected_sources: list[ResearchSource]
    rejected_sources: list[ResearchSource]
    reflection_assessment: dict
    reflection_history: Annotated[list[dict], operator.add]
    evidence_source_count_history: Annotated[list[int], operator.add]
    completion_status: str
    claims: list[EvidenceClaim]
    dimension_summary: str


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
    query_history: Annotated[list[str], operator.add]


class WebSearchState(TypedDict):
    research_run_id: str
    search_query: str
    search_id: str
