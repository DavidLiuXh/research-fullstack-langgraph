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
    gap_id: NotRequired[str]
    gap_ids: NotRequired[list[str]]
    protected_gap_ids: NotRequired[list[str]]
    requested_source_types: NotRequired[list[str]]
    expected_evidence: NotRequired[str]
    matches_requested_source_type: NotRequired[bool]
    content_completeness_score: NotRequired[float]
    is_authoritative_source: NotRequired[bool]


class EvidenceQuote(TypedDict):
    source_id: str
    quote: str
    locator: str


class EvidenceClaim(TypedDict):
    claim_id: NotRequired[str]
    gap_ids: NotRequired[list[str]]
    claim: str
    supporting_source_ids: list[str]
    supporting_evidence: list[EvidenceQuote]
    contradicting_source_ids: list[str]
    contradicting_evidence: list[EvidenceQuote]
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
    known_gap_count: int
    resolved_gap_count: int
    high_priority_gap_count: int
    resolved_high_priority_gap_count: int
    high_priority_gap_source_coverage_count: int
    evidence_gain_history: list[dict]
    search_failure_count: int
    closed_gap_count: int
    unresolvable_gap_count: int
    gap_status_counts: dict[str, int]
    direct_evidence_gap_count: int
    supported_claim_gap_count: int
    requested_type_gap_count: int
    independent_source_gap_count: int
    gap_assessment_failure_count: int
    no_gain_loop_count: int
    gap_diagnostics: list[dict]
    final_gap_audit: dict


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
    report_generation_mode: str
    report_overview: str
    report_sections: list[dict]
    report_audit: dict
    report_revision_count: int
    max_report_revisions: int
    report_dimension_results: list[DimensionResult]
    report_sources: list[ResearchSource]
    report_evidence_ledger: dict
    claim_conflicts: list[dict]
    consistency_analysis_complete: bool
    report_consistency_audit: dict
    report_safe_fallback_used: bool


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
    search_tasks: list[dict]
    search_failures: Annotated[list[str], operator.add]
    search_success_count: Annotated[int, operator.add]
    evaluated_sources: list[ResearchSource]
    selected_sources: list[ResearchSource]
    rejected_sources: list[ResearchSource]
    reflection_assessment: dict
    reflection_history: Annotated[list[dict], operator.add]
    evidence_source_count_history: Annotated[list[int], operator.add]
    evidence_source_id_history: Annotated[list[list[str]], operator.add]
    evidence_gain_history: Annotated[list[dict], operator.add]
    gap_registry: dict[str, dict]
    active_gap_id: str
    active_gap: dict
    gap_processing_complete: bool
    gap_evidence_assessment: dict
    pending_reflection_gaps: list[dict]
    gap_route: str
    dimension_reflection_count: int
    resolved_gap_ids: list[str]
    gap_source_coverage_ids: list[str]
    completion_status: str
    claims: list[EvidenceClaim]
    gap_claim_ledger: NotRequired[list[EvidenceClaim]]
    dimension_summary: str
    gap_assessment_failure_count: Annotated[int, operator.add]
    final_gap_audit: NotRequired[dict]


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
    search_tasks: list[dict]
    research_loop_count: int
    query_history: Annotated[list[str], operator.add]
    active_gap_id: str


class WebSearchState(TypedDict):
    research_run_id: str
    search_query: str
    search_id: str
    gap_id: str
    requested_source_types: list[str]
    expected_evidence: str
    exclude_domains: list[str]
