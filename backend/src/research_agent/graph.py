"""LangGraph workflow for clarification, dimension research, and reporting."""

# ruff: noqa: E402

import json
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

# The CLI executes this file by path, so the src-layout package root is not
# guaranteed to be importable unless the project has already been installed.
project_src = str(Path(__file__).resolve().parents[1])
if project_src not in sys.path:
    sys.path.insert(0, project_src)

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from openai import LengthFinishReasonError
from tavily import TavilyClient

from research_agent.configuration import Configuration
from research_agent.llm import create_deepseek_model
from research_agent.prompts import (
    answer_instructions,
    claim_conflict_instructions,
    claim_extraction_instructions,
    dimension_instructions,
    gap_evidence_assessment_instructions,
    get_current_date,
    initial_gap_planning_instructions,
    query_writer_instructions,
    reflection_instructions,
    report_audit_instructions,
    report_consistency_audit_instructions,
    report_overview_instructions,
    report_overview_revision_instructions,
    report_revision_instructions,
    report_section_instructions,
    report_section_revision_instructions,
    source_evaluation_instructions,
    topic_clarification_instructions,
)
from research_agent.state import (
    DimensionInput,
    DimensionResult,
    DimensionState,
    OverallState,
    QueryGenerationState,
    ResearchSource,
    WebSearchState,
)
from research_agent.tools_and_schemas import (
    ClaimConflictAnalysis,
    ClaimExtraction,
    GapEvidenceAssessment,
    Reflection,
    ReportAudit,
    ReportConsistencyAudit,
    ResearchDimensionList,
    ResearchGap,
    ResearchGapPlan,
    SearchQueryList,
    SourceAssessment,
    SourceAssessmentList,
    TopicClarificationAssessment,
)
from research_agent.utils import (
    deduplicate_sources,
    deduplicate_sources_by_id,
    format_dimension_results,
    format_rejected_source_summary,
    format_source_candidates,
    format_sources_for_research,
    get_research_topic,
    locate_evidence_quote,
    normalize_search_score,
    render_source_citations,
    tavily_results_to_sources,
)

load_dotenv()

AUTHORITATIVE_SOURCE_TYPES = {
    "government",
    "academic",
    "official_company",
    "standards_body",
    "international_organization",
    "industry_association",
    "research_institute",
}


def _default_source_types(
    research_topic: str, dimension: Mapping[str, str]
) -> list[str]:
    """Infer a compact authoritative-source strategy for an initial search pass."""
    context = f"{research_topic} {dimension['title']} {dimension['scope']}".casefold()
    if any(term in context for term in ("law", "regulation", "policy", "法规", "政策")):
        return ["government", "international_organization", "standards_body"]
    if any(
        term in context
        for term in ("technology", "technical", "software", "技术", "架构")
    ):
        return ["official_company", "standards_body", "academic"]
    if any(term in context for term in ("market", "industry", "市场", "行业")):
        return ["government", "industry_association", "research_institute"]
    return ["government", "academic", "official_company"]


def _source_rank_key(source: Mapping[str, Any]) -> tuple:
    """Rank sources by requested fit, acceptance, primacy, authority, and score."""
    return (
        bool(source.get("matches_requested_source_type")),
        source.get("quality_status") == "accepted",
        bool(source.get("is_primary_source")),
        bool(source.get("is_authoritative_source")),
        float(source.get("evidence_score", 0)),
    )


def emit_research_event(event_type: str, **data):
    """Emit progress that remains visible while nested subgraphs are running."""
    get_stream_writer()({"type": event_type, **data})


def initialize_research_topic(state: OverallState):
    """Reset clarification state for the latest research request."""
    topic = get_research_topic(state["messages"])
    return {
        "original_research_topic": topic,
        "normalized_research_topic": topic,
        "topic_needs_clarification": False,
        "topic_ambiguities": [],
        "topic_clarification_questions": [],
        "topic_assumptions": [],
        "topic_clarification_reason": "",
        "topic_clarification_history": [],
        "topic_clarification_action": "",
    }


def analyze_research_topic(state: OverallState, config: RunnableConfig):
    """Decide whether material ambiguity requires human clarification."""
    configurable = Configuration.from_runnable_config(config)
    history = state.get("topic_clarification_history", [])
    history_text = (
        "\n\n".join(
            "Questions:\n- "
            + "\n- ".join(turn["questions"])
            + f"\nUser response: {turn['response']}"
            for turn in history
        )
        or "None; this is the first assessment."
    )
    prompt = topic_clarification_instructions.format(
        original_topic=state["original_research_topic"],
        clarification_history=history_text,
    )
    result = (
        create_deepseek_model(configurable.query_generator_model)
        .with_structured_output(TopicClarificationAssessment, method="json_mode")
        .invoke(prompt)
    )
    questions = result.clarification_questions[:3]
    needs_clarification = result.needs_clarification and bool(questions)
    emit_research_event(
        "topic_analyzed",
        needs_clarification=needs_clarification,
        ambiguities=result.ambiguities,
        questions=questions,
        assumptions=result.assumptions,
    )
    return {
        "normalized_research_topic": result.normalized_topic.strip()
        or state["original_research_topic"],
        "topic_needs_clarification": needs_clarification,
        "topic_ambiguities": result.ambiguities,
        "topic_clarification_questions": questions,
        "topic_assumptions": result.assumptions,
        "topic_clarification_reason": result.reason,
        "topic_clarification_action": "",
    }


def route_topic_analysis(state: OverallState):
    """Request clarification only when the topic has material ambiguity."""
    if state["topic_needs_clarification"]:
        return "request_topic_clarification"
    return "generate_research_dimensions"


def request_topic_clarification(state: OverallState):
    """Pause for clarification or explicit acceptance of proposed assumptions."""
    decision = interrupt(
        {
            "type": "research_topic_clarification",
            "message": "Clarify the research topic before planning begins.",
            "ambiguities": state["topic_ambiguities"],
            "questions": state["topic_clarification_questions"],
            "assumptions": state["topic_assumptions"],
            "reason": state["topic_clarification_reason"],
        }
    )
    if not isinstance(decision, dict):
        raise ValueError("Topic clarification must be an object")
    action = str(decision.get("action", "")).strip()
    if action not in {"clarify", "accept_assumptions"}:
        raise ValueError("Topic clarification action is invalid")

    if action == "accept_assumptions":
        response = "Accepted the proposed assumptions."
        needs_clarification = False
    else:
        response = str(decision.get("response", "")).strip()
        if not response:
            raise ValueError("A clarification response is required")
        needs_clarification = True

    history = [
        *state.get("topic_clarification_history", []),
        {
            "questions": state["topic_clarification_questions"],
            "response": response,
        },
    ]
    emit_research_event(
        "topic_clarification_received", action=action, response=response
    )
    return {
        "topic_clarification_action": action,
        "topic_needs_clarification": needs_clarification,
        "topic_clarification_history": history,
    }


def route_topic_clarification(state: OverallState):
    """Reassess user input, or continue immediately with accepted assumptions."""
    if state["topic_clarification_action"] == "accept_assumptions":
        return "generate_research_dimensions"
    return "analyze_research_topic"


def generate_research_dimensions(state: OverallState, config: RunnableConfig):
    """Decompose the main topic into independent, complementary dimensions."""
    configurable = Configuration.from_runnable_config(config)
    topic = state["normalized_research_topic"]
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
    if not isinstance(decision, dict) or not isinstance(decision.get("approved"), bool):
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
    topic = state["normalized_research_topic"]
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
                "max_research_loops": state.get("max_research_loops", 3),
            },
        )
        for dimension in state["research_dimensions"]
    ]


def _operational_gap(gap: Mapping[str, Any], *, origin: str) -> dict[str, Any]:
    """Return a normalized gap record with deterministic lifecycle fields."""
    normalized = ResearchGap.model_validate(gap).model_dump()
    normalized.update(
        {
            "origin": origin,
            "status": "open",
            "attempt_count": 0,
            "no_progress_count": 0,
            "strategy_level": 0,
            "matched_source_ids": [],
            "supported_claims": [],
            "closure_reason": "",
            "remaining_evidence": "",
            "closure_blockers": [],
            "assessment_status": "not_assessed",
        }
    )
    return normalized


def plan_initial_gaps(state: DimensionState, config: RunnableConfig):
    """Plan concrete evidence gaps before selecting the first research target."""
    configurable = Configuration.from_runnable_config(config)
    prompt = initial_gap_planning_instructions.format(
        number_gaps=configurable.max_initial_gaps_per_dimension,
        research_topic=state["research_topic"],
        dimension_title=state["dimension"]["title"],
        dimension_scope=state["dimension"]["scope"],
    )
    try:
        result = (
            create_deepseek_model(configurable.reflection_model)
            .with_structured_output(ResearchGapPlan, method="json_mode")
            .invoke(prompt)
        )
        planned = result.gaps[: configurable.max_initial_gaps_per_dimension]
    except (
        AttributeError,
        LengthFinishReasonError,
        OutputParserException,
        TypeError,
        ValueError,
    ) as error:
        emit_research_event(
            "gap_planning_fallback",
            research_run_id=state["research_run_id"],
            dimension=state["dimension"],
            error=str(error),
        )
        planned = [
            ResearchGap(
                gap_id=f"{state['dimension']['id']}-initial-scope",
                question=state["dimension"]["scope"],
                reason="Structured initial gap planning failed; the full dimension scope remains open.",
                priority="high",
                required_source_types=_default_source_types(
                    state["research_topic"], state["dimension"]
                ),
                expected_evidence=state["dimension"]["scope"],
                suggested_query_focus=state["dimension"]["scope"],
            )
        ]

    registry: dict[str, dict] = {}
    default_types = _default_source_types(state["research_topic"], state["dimension"])
    for item in planned:
        gap = _operational_gap(item.model_dump(), origin="planned")
        gap["required_source_types"] = gap["required_source_types"] or default_types
        registry.setdefault(gap["gap_id"], gap)

    emit_research_event(
        "initial_gaps_planned",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        gaps=list(registry.values()),
    )
    return {
        "gap_registry": registry,
        "active_gap_id": "",
        "active_gap": {},
        "gap_processing_complete": False,
        "pending_reflection_gaps": [],
        "dimension_reflection_count": 0,
        "research_loop_count": 0,
        "completion_status": "researching",
        "is_sufficient": False,
        "resolved_gap_ids": [],
        "gap_source_coverage_ids": [],
    }


def select_next_gap(state: DimensionState, config: RunnableConfig):
    """Select exactly one actionable gap by priority and prior progress."""
    configurable = Configuration.from_runnable_config(config)
    registry = {
        key: dict(value) for key, value in state.get("gap_registry", {}).items()
    }
    actionable = []
    for gap in registry.values():
        status = gap.get("status", "open")
        attempts = int(gap.get("attempt_count", 0))
        no_progress = int(gap.get("no_progress_count", 0))
        if status == "active":
            status = "partial"
            gap["status"] = status
        if (
            status in {"open", "partial", "reopened"}
            and attempts < state["max_research_loops"]
            and no_progress < configurable.max_gap_no_progress_attempts
        ):
            actionable.append(gap)

    priority = {"high": 0, "medium": 1, "low": 2}
    actionable.sort(
        key=lambda gap: (
            priority.get(gap.get("priority"), 3),
            -int(gap.get("no_progress_count", 0)),
            int(gap.get("attempt_count", 0)),
            gap.get("gap_id", ""),
        )
    )
    if not actionable:
        emit_research_event(
            "all_gaps_processed",
            research_run_id=state["research_run_id"],
            dimension=state["dimension"],
            gap_statuses={
                key: value.get("status", "open") for key, value in registry.items()
            },
        )
        return {
            "gap_registry": registry,
            "active_gap_id": "",
            "active_gap": {},
            "gap_processing_complete": True,
        }

    active = dict(actionable[0])
    active["status"] = "active"
    registry[active["gap_id"]] = active
    emit_research_event(
        "gap_selected",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        gap=active,
    )
    return {
        "gap_registry": registry,
        "active_gap_id": active["gap_id"],
        "active_gap": active,
        "gap_processing_complete": False,
        "current_knowledge_gap": (
            f"[{active['gap_id']}] {active['question']}. "
            f"Expected evidence: {active['expected_evidence']}"
        ),
    }


def route_gap_selection(state: DimensionState):
    """Research the active gap or audit the dimension when none remain."""
    return (
        "dimension_reflection"
        if state.get("gap_processing_complete")
        else "generate_query"
    )


def generate_query(
    state: DimensionState, config: RunnableConfig
) -> QueryGenerationState:
    """Generate searches for exactly one selected evidence gap."""
    configurable = Configuration.from_runnable_config(config)
    query_count = (
        state.get("initial_search_query_count")
        or configurable.number_of_initial_queries
    )
    gap = dict(state.get("active_gap") or {})
    if not gap:
        raise ValueError("No active evidence gap was selected")
    required_source_types = gap.get("required_source_types", [])
    strategies = gap.get("search_strategy", [])
    do_not_repeat = gap.get("do_not_repeat", [])
    query_history = state.get("query_history", [])
    accepted_sources = [
        source
        for source in state.get("selected_sources", [])
        if source.get("quality_status") == "accepted"
    ]
    domain_counts: dict[str, int] = {}
    for source in accepted_sources:
        domain = str(source.get("domain", "")).strip()
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    covered_domains = sorted(domain_counts)
    accepted_by_id = {source["source_id"]: source for source in accepted_sources}
    matched_domains = {
        accepted_by_id[source_id].get("domain")
        for source_id in gap.get("matched_source_ids", [])
        if source_id in accepted_by_id and accepted_by_id[source_id].get("domain")
    }
    if gap.get("gap_id") == "quality-accepted-sources":
        required_independent = configurable.min_accepted_sources_per_dimension
    elif gap.get("priority") == "high":
        required_independent = configurable.min_independent_sources_per_high_gap
    else:
        required_independent = 1
    gap_domains_to_avoid = (
        matched_domains if len(matched_domains) < required_independent else set()
    )
    saturated_domains = sorted(
        {
            *gap.get("excluded_domains", []),
            *gap_domains_to_avoid,
            *(
                domain
                for domain, count in domain_counts.items()
                if count >= configurable.max_sources_per_domain
            ),
        }
    )
    covered_source_types = sorted(
        {
            str(source.get("source_type"))
            for source in accepted_sources
            if source.get("source_type") and source.get("source_type") != "unknown"
        }
    )
    missing_source_types = sorted(
        set(required_source_types) - set(covered_source_types)
    )
    prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=state["research_topic"],
        dimension_title=state["dimension"]["title"],
        dimension_scope=state["dimension"]["scope"],
        knowledge_gap=state.get("current_knowledge_gap")
        or "None; this is the first pass.",
        active_gap=json.dumps(gap, ensure_ascii=False),
        required_source_types=required_source_types or "No special requirement.",
        recommended_search_strategy=strategies or "No special strategy.",
        strategy_level=gap.get("strategy_level", 0),
        covered_source_types=covered_source_types or "None.",
        missing_source_types=missing_source_types or "None.",
        covered_domains=covered_domains or "None.",
        saturated_domains=saturated_domains or "None.",
        query_history=[*query_history, *do_not_repeat]
        or "None; this is the first pass.",
        number_queries=query_count,
    )
    llm = create_deepseek_model(configurable.query_generator_model)
    try:
        result = llm.with_structured_output(SearchQueryList, method="json_mode").invoke(
            prompt
        )
        model_queries = result.query
    except (
        AttributeError,
        LengthFinishReasonError,
        OutputParserException,
        TypeError,
        ValueError,
    ) as error:
        emit_research_event(
            "query_generation_fallback",
            research_run_id=state["research_run_id"],
            dimension=state["dimension"],
            gap_id=gap["gap_id"],
            error=str(error),
        )
        model_queries = []
    normalized_history = {query.casefold().strip() for query in query_history}
    queries = []
    for query in model_queries:
        cleaned = query.strip()
        if (
            cleaned
            and cleaned.casefold() not in normalized_history
            and cleaned not in queries
        ):
            queries.append(cleaned)
        if len(queries) >= query_count:
            break
    if not queries:
        fallback_parts = [
            gap.get("suggested_query_focus") or gap.get("question", ""),
            gap.get("expected_evidence", ""),
            " ".join(required_source_types),
            " ".join(strategies),
        ]
        fallback = " ".join(part.strip() for part in fallback_parts if part).strip()
        if not fallback:
            fallback = state["dimension"]["scope"]
        if fallback.casefold() in normalized_history:
            fallback = (
                f"{fallback} original source strategy level "
                f"{int(gap.get('strategy_level', 0)) + 1}"
            )
        queries = [fallback]
    search_tasks = []
    for query in queries:
        search_tasks.append(
            {
                "query": query,
                "gap_id": gap["gap_id"],
                "requested_source_types": gap.get("required_source_types", [])
                or _default_source_types(state["research_topic"], state["dimension"]),
                "expected_evidence": gap.get("expected_evidence")
                or gap.get("reason", ""),
                "exclude_domains": saturated_domains,
            }
        )
    emit_research_event(
        "queries_generated",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        queries=queries,
        search_tasks=search_tasks,
        loop=state.get("research_loop_count", 0),
    )
    return {
        "research_run_id": state["research_run_id"],
        "research_topic": state["research_topic"],
        "dimension": state["dimension"],
        "search_query": queries,
        "search_tasks": search_tasks,
        "query_history": queries,
        "research_loop_count": state.get("research_loop_count", 0),
        "active_gap_id": gap["gap_id"],
    }


def dispatch_search_queries(state: QueryGenerationState):
    """Fan out the current dimension's search queries to Tavily."""
    return [
        Send(
            "web_research",
            {
                "search_query": task["query"],
                "research_run_id": state["research_run_id"],
                "search_id": (
                    f"{state['research_run_id']}-{state['dimension']['id']}-"
                    f"{state['research_loop_count']}-{index}"
                ),
                "gap_id": task["gap_id"],
                "requested_source_types": task["requested_source_types"],
                "expected_evidence": task["expected_evidence"],
                "exclude_domains": task.get("exclude_domains", []),
            },
        )
        for index, task in enumerate(state["search_tasks"])
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
                exclude_domains=state.get("exclude_domains", []) or None,
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
            "search_failures": [
                f"{state['search_query']}: "
                + (str(last_error) if last_error else "Unknown Tavily error")
            ],
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
    sources = [
        {
            **source,
            "gap_id": state.get("gap_id", ""),
            "gap_ids": [state["gap_id"]] if state.get("gap_id") else [],
            "requested_source_types": state.get("requested_source_types", []),
            "expected_evidence": state.get("expected_evidence", ""),
        }
        for source in sources
    ]
    emit_research_event(
        "search_completed",
        query=state["search_query"],
        source_count=len(sources),
        sources=sources,
    )
    return {
        "sources_gathered": sources,
        "search_success_count": 1,
        "web_research_result": [format_sources_for_research(sources)],
    }


def evaluate_sources(state: DimensionState, config: RunnableConfig):
    """Normalize, assess, and select evidence before reflection."""
    configurable = Configuration.from_runnable_config(config)
    all_candidates = deduplicate_sources(state.get("sources_gathered", []))
    candidates = sorted(
        all_candidates,
        key=lambda source: normalize_search_score(source.get("score"), default=0),
        reverse=True,
    )[: configurable.max_source_candidates_per_dimension]
    if not candidates:
        emit_research_event(
            "sources_evaluated",
            research_run_id=state["research_run_id"],
            dimension=state["dimension"],
            candidate_count=0,
            assessed_count=0,
            accepted_count=0,
            supplementary_count=0,
            rejected_count=0,
        )
        return {
            "evaluated_sources": [],
            "selected_sources": [],
            "rejected_sources": [],
        }

    prompt = source_evaluation_instructions.format(
        research_topic=state["research_topic"],
        dimension_title=state["dimension"]["title"],
        dimension_scope=state["dimension"]["scope"],
        candidate_sources=format_source_candidates(candidates),
    )
    try:
        result = (
            create_deepseek_model(configurable.reflection_model)
            .with_structured_output(SourceAssessmentList, method="json_mode")
            .invoke(prompt)
        )
    except OutputParserException as error:
        emit_research_event(
            "source_evaluation_fallback",
            research_run_id=state["research_run_id"],
            dimension=state["dimension"],
            error=str(error),
        )
        result = SourceAssessmentList(
            assessments=[
                SourceAssessment(
                    source_id=source["source_id"],
                    source_type="unknown",
                    authority_score=0.25,
                    relevance_score=normalize_search_score(source.get("score")),
                    recency_score=0.5,
                    is_primary_source=False,
                    is_likely_repost=False,
                    supported_topics=[],
                    rejection_reasons=[
                        "Structured source evaluation failed; conservative "
                        "search-score fallback was used."
                    ],
                )
                for source in candidates
            ]
        )
    assessment_map = {item.source_id: item for item in result.assessments}
    evaluated = []
    for source in candidates:
        assessment = assessment_map.get(source["source_id"])
        if assessment is None:
            authority = 0.25
            relevance = 0.4
            recency = 0.4
            primary = False
            repost = False
            source_type = "unknown"
            supported_topics: list[str] = []
            rejection_reasons = ["The source assessment was missing."]
        else:
            authority = assessment.authority_score
            relevance = assessment.relevance_score
            recency = assessment.recency_score
            primary = assessment.is_primary_source
            repost = assessment.is_likely_repost
            source_type = assessment.source_type
            supported_topics = assessment.supported_topics
            rejection_reasons = assessment.rejection_reasons
        if source_type == "unknown":
            authority = min(authority, 0.4)
            primary = False
            rejection_reasons = [
                *rejection_reasons,
                "The provider source type was unrecognized; authority was capped conservatively.",
            ]
        requested_source_types = source.get("requested_source_types", [])
        matches_requested_source_type = (
            not requested_source_types or source_type in requested_source_types
        )
        content_completeness_score = max(
            0.25, min(len(source.get("content", "")) / 1200, 1)
        )
        is_authoritative_source = (
            source_type in AUTHORITATIVE_SOURCE_TYPES and authority >= 0.7
        )
        evidence_score = (
            relevance * 0.35
            + authority * 0.30
            + (0.15 if primary else 0.0)
            + recency * 0.10
            + content_completeness_score * 0.10
        )
        if assessment is None:
            quality_status = "rejected"
        elif repost or relevance < 0.35:
            quality_status = "rejected"
        elif evidence_score >= configurable.source_acceptance_threshold:
            quality_status = "accepted"
        elif evidence_score >= configurable.source_supplementary_threshold:
            quality_status = "supplementary"
        else:
            quality_status = "rejected"
        evaluated.append(
            {
                **source,
                "source_type": source_type,
                "authority_score": authority,
                "relevance_score": relevance,
                "recency_score": recency,
                "evidence_score": evidence_score,
                "is_primary_source": primary,
                "is_likely_repost": repost,
                "supported_topics": supported_topics,
                "rejection_reasons": rejection_reasons,
                "quality_status": quality_status,
                "matches_requested_source_type": matches_requested_source_type,
                "content_completeness_score": content_completeness_score,
                "is_authoritative_source": is_authoritative_source,
            }
        )
    domain_counts: dict[str, int] = {}
    for source in sorted(evaluated, key=_source_rank_key, reverse=True):
        if source["quality_status"] == "rejected":
            continue
        domain = source.get("domain", "")
        if domain_counts.get(domain, 0) >= configurable.max_sources_per_domain:
            source["quality_status"] = "rejected"
            source["rejection_reasons"] = [
                *source["rejection_reasons"],
                "The per-domain evidence limit was reached.",
            ]
            continue
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    quality_ranked = sorted(
        (
            source
            for source in evaluated
            if source["quality_status"] in {"accepted", "supplementary"}
        ),
        key=_source_rank_key,
        reverse=True,
    )
    for source in quality_ranked[configurable.max_selected_sources_per_dimension :]:
        source["quality_status"] = "rejected"
        source["rejection_reasons"] = [
            *source["rejection_reasons"],
            "The per-dimension evidence budget was reached.",
        ]
    selected = sorted(
        (
            source
            for source in evaluated
            if source["quality_status"] in {"accepted", "supplementary"}
        ),
        key=_source_rank_key,
        reverse=True,
    )
    rejected = [
        source for source in evaluated if source["quality_status"] == "rejected"
    ]
    emit_research_event(
        "sources_evaluated",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        candidate_count=len(candidates),
        assessed_count=len(candidates),
        candidate_count_total=len(all_candidates),
        accepted_count=sum(
            source["quality_status"] == "accepted" for source in evaluated
        ),
        supplementary_count=sum(
            source["quality_status"] == "supplementary" for source in evaluated
        ),
        rejected_count=len(rejected),
    )
    return {
        "evaluated_sources": evaluated,
        "selected_sources": selected,
        "rejected_sources": rejected,
    }


def assess_gap_evidence(state: DimensionState, config: RunnableConfig):
    """Map accepted evidence to the active gap and measure only direct gain."""
    configurable = Configuration.from_runnable_config(config)
    gap = dict(state.get("active_gap") or {})
    if not gap:
        raise ValueError("Cannot assess evidence without an active gap")
    all_accepted = [
        source
        for source in state.get("selected_sources", [])
        if source.get("quality_status") == "accepted"
    ]
    prior_source_ids = set(gap.get("matched_source_ids", []))
    has_provenance = any(
        source.get("gap_id") or source.get("gap_ids") for source in all_accepted
    )
    accepted = [
        source
        for source in all_accepted
        if not has_provenance
        or source["source_id"] in prior_source_ids
        or gap["gap_id"] == source.get("gap_id")
        or gap["gap_id"] in source.get("gap_ids", [])
    ]
    assessment_status = "completed"
    assessment_failure_count = 0
    if gap["gap_id"] in {
        "quality-accepted-sources",
        "quality-authoritative-source",
        "quality-primary-source",
    }:
        if gap["gap_id"] == "quality-authoritative-source":
            accepted = [
                source
                for source in all_accepted
                if source.get("is_authoritative_source")
            ]
        elif gap["gap_id"] == "quality-primary-source":
            accepted = [
                source for source in all_accepted if source.get("is_primary_source")
            ]
        else:
            accepted = all_accepted
        result = GapEvidenceAssessment(
            gap_id=gap["gap_id"],
            directly_answers_gap=bool(accepted),
            matched_source_ids=[source["source_id"] for source in accepted],
            supported_claims=[
                f"The dimension has {len(accepted)} qualifying accepted source(s)."
            ]
            if accepted
            else [],
            remaining_evidence="" if accepted else gap.get("expected_evidence", ""),
        )
        assessment_status = "deterministic_quality_check"
    elif accepted:
        prompt = gap_evidence_assessment_instructions.format(
            research_topic=state["research_topic"],
            dimension_title=state["dimension"]["title"],
            active_gap=json.dumps(gap, ensure_ascii=False),
            output_schema=json.dumps(
                ClaimExtraction.model_json_schema(), ensure_ascii=False
            ),
            accepted_evidence=format_sources_for_research(accepted),
        )
        structured_model = create_deepseek_model(
            configurable.reflection_model
        ).with_structured_output(ClaimExtraction, method="json_mode")
        candidate_by_id = {source["source_id"]: source for source in accepted}

        def invoke_assessment(retry: bool = False):
            retry_instruction = (
                "\nThe previous response was invalid. Return only claims with exact "
                f"verbatim quotes that directly answer gap_id {gap['gap_id']}."
                if retry
                else ""
            )
            assessment_result = structured_model.invoke(prompt + retry_instruction)
            if not isinstance(assessment_result, ClaimExtraction):
                raise TypeError("Gap evidence assessment returned an unexpected type")
            matched_source_ids = []
            contradictory_source_ids = []
            supported_claims = []
            for claim in assessment_result.claims[:6]:
                if claim.gap_ids and gap["gap_id"] not in claim.gap_ids:
                    continue
                verified_supporting_ids = []
                for evidence in claim.evidence:
                    source = candidate_by_id.get(evidence.source_id)
                    if source and locate_evidence_quote(
                        source.get("content", ""),
                        evidence.quote,
                        min_chars=configurable.min_evidence_quote_chars,
                    ):
                        verified_supporting_ids.append(evidence.source_id)
                if not verified_supporting_ids:
                    continue
                matched_source_ids.extend(verified_supporting_ids)
                supported_claims.append(claim.claim)
                for evidence in claim.counter_evidence:
                    source = candidate_by_id.get(evidence.source_id)
                    if source and locate_evidence_quote(
                        source.get("content", ""),
                        evidence.quote,
                        min_chars=configurable.min_evidence_quote_chars,
                    ):
                        contradictory_source_ids.append(evidence.source_id)
            matched_source_ids = list(dict.fromkeys(matched_source_ids))
            return GapEvidenceAssessment(
                gap_id=gap["gap_id"],
                directly_answers_gap=bool(matched_source_ids),
                matched_source_ids=matched_source_ids,
                supported_claims=list(dict.fromkeys(supported_claims)),
                contradictory_source_ids=list(dict.fromkeys(contradictory_source_ids)),
                remaining_evidence=(
                    "" if matched_source_ids else assessment_result.summary
                ),
            )

        try:
            result = invoke_assessment()
        except (
            AttributeError,
            LengthFinishReasonError,
            OutputParserException,
            TypeError,
            ValueError,
        ) as error:
            try:
                result = invoke_assessment(retry=True)
                assessment_status = "completed_after_retry"
            except (
                AttributeError,
                LengthFinishReasonError,
                OutputParserException,
                TypeError,
                ValueError,
            ) as retry_error:
                emit_research_event(
                    "gap_evidence_assessment_fallback",
                    research_run_id=state["research_run_id"],
                    dimension=state["dimension"],
                    gap_id=gap["gap_id"],
                    error=f"{type(error).__name__}: {error}; retry: {retry_error}",
                )
                result = GapEvidenceAssessment(
                    gap_id=gap["gap_id"],
                    remaining_evidence=gap.get("expected_evidence", ""),
                )
                assessment_status = "structured_output_failure"
                assessment_failure_count = 1
    else:
        result = GapEvidenceAssessment(
            gap_id=gap["gap_id"],
            remaining_evidence=gap.get("expected_evidence", ""),
        )
        assessment_status = "no_gap_candidate_evidence"

    accepted_by_id = {source["source_id"]: source for source in accepted}
    matched_ids = list(
        dict.fromkeys(
            source_id
            for source_id in result.matched_source_ids
            if source_id in accepted_by_id
        )
    )
    contradictory_ids = list(
        dict.fromkeys(
            source_id
            for source_id in result.contradictory_source_ids
            if source_id in accepted_by_id
        )
    )
    directly_answers = bool(result.directly_answers_gap and matched_ids)
    if not directly_answers:
        matched_ids = []
    new_source_ids = [item for item in matched_ids if item not in prior_source_ids]
    prior_claims = {
        re.sub(r"\s+", " ", claim).strip().casefold()
        for claim in gap.get("supported_claims", [])
    }
    supported_claims = list(
        dict.fromkeys(
            claim.strip() for claim in result.supported_claims if claim.strip()
        )
    )
    if not directly_answers:
        supported_claims = []
    new_claims = [
        claim
        for claim in supported_claims
        if re.sub(r"\s+", " ", claim).strip().casefold() not in prior_claims
    ]
    matched_sources = [accepted_by_id[source_id] for source_id in matched_ids]
    required_types = set(gap.get("required_source_types", []))
    requested_type_satisfied = bool(
        matched_sources
        and (
            not required_types
            or any(
                source.get("source_type") in required_types
                for source in matched_sources
            )
        )
    )
    independent_domains = {
        source.get("domain") or source.get("canonical_url") or source.get("url")
        for source in matched_sources
    }
    evidence_strength = (
        sum(float(source.get("evidence_score", 0)) for source in matched_sources)
        / len(matched_sources)
        if matched_sources
        else 0.0
    )
    # New evidence, rather than a differently worded model claim over evidence
    # already seen, is the reliable signal for another search pass.
    has_progress = bool(directly_answers and new_source_ids)
    assessment = {
        **result.model_dump(),
        "gap_id": gap["gap_id"],
        "directly_answers_gap": directly_answers,
        "matched_source_ids": matched_ids,
        "new_matched_source_ids": new_source_ids,
        "supported_claims": supported_claims,
        "new_supported_claims": new_claims,
        "contradictory_source_ids": contradictory_ids,
        "requested_source_type_satisfied": requested_type_satisfied,
        "independent_source_count": len(independent_domains),
        "evidence_strength": evidence_strength,
        "has_progress": has_progress,
        "assessment_status": assessment_status,
        "candidate_source_count": len(accepted),
    }
    emit_research_event(
        "gap_evidence_assessed",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        assessment=assessment,
    )
    return {
        "gap_evidence_assessment": assessment,
        "gap_assessment_failure_count": assessment_failure_count,
    }


def _gap_closure_snapshot(
    gap: dict,
    *,
    matched_ids: list[str],
    supported_claims: list[str],
    contradictory_ids: list[str],
    selected_by_id: dict[str, dict],
    configurable: Configuration,
    direct_evidence_confirmed: bool,
) -> dict[str, Any]:
    """Calculate deterministic gap closure requirements from cumulative evidence."""
    matched_sources = [
        selected_by_id[source_id]
        for source_id in matched_ids
        if source_id in selected_by_id
    ]
    required_types = set(gap.get("required_source_types", []))
    requested_type_satisfied = bool(
        matched_sources
        and (
            not required_types
            or any(
                source.get("source_type") in required_types
                for source in matched_sources
            )
        )
    )
    independent_domains = {
        source.get("domain") or source.get("canonical_url") or source.get("url")
        for source in matched_sources
    }
    gap_id = gap["gap_id"]
    if gap_id == "quality-primary-source":
        requested_type_satisfied = any(
            bool(source.get("is_primary_source")) for source in matched_sources
        )
        required_independent = 1
    elif gap_id == "quality-authoritative-source":
        requested_type_satisfied = any(
            bool(source.get("is_authoritative_source")) for source in matched_sources
        )
        required_independent = 1
    elif gap_id == "quality-accepted-sources":
        required_independent = configurable.min_accepted_sources_per_dimension
    else:
        required_independent = (
            configurable.min_independent_sources_per_high_gap
            if gap.get("priority") == "high"
            else 1
        )
    closure_blockers = []
    if not direct_evidence_confirmed:
        closure_blockers.append("missing_direct_evidence")
    if not supported_claims:
        closure_blockers.append("missing_supported_claim")
    if not requested_type_satisfied:
        closure_blockers.append("missing_requested_source_type")
    if len(independent_domains) < required_independent:
        closure_blockers.append("insufficient_independent_sources")
    if contradictory_ids:
        closure_blockers.append("unresolved_contradiction")
    return {
        "matched_sources": matched_sources,
        "requested_type_satisfied": requested_type_satisfied,
        "independent_domains": independent_domains,
        "required_independent": required_independent,
        "direct_evidence_confirmed": direct_evidence_confirmed,
        "closure_blockers": closure_blockers,
    }


def update_gap_status(state: DimensionState, config: RunnableConfig):
    """Update the active gap with deterministic closure and stall rules."""
    configurable = Configuration.from_runnable_config(config)
    registry = {
        key: dict(value) for key, value in state.get("gap_registry", {}).items()
    }
    gap_id = state.get("active_gap_id", "")
    if not gap_id or gap_id not in registry:
        raise ValueError("The active gap is missing from the gap registry")
    gap = dict(registry[gap_id])
    assessment = dict(state.get("gap_evidence_assessment") or {})
    selected_by_id = {
        source["source_id"]: source
        for source in state.get("selected_sources", [])
        if source.get("quality_status") == "accepted"
    }

    matched_ids = list(
        dict.fromkeys(
            [
                *gap.get("matched_source_ids", []),
                *assessment.get("matched_source_ids", []),
            ]
        )
    )
    supported_claims = list(
        dict.fromkeys(
            [
                *gap.get("supported_claims", []),
                *assessment.get("supported_claims", []),
            ]
        )
    )
    # Every assessment sees the complete provenance-constrained evidence set for
    # this gap, so the latest result supersedes earlier contradiction candidates.
    contradictory_ids = list(
        dict.fromkeys(assessment.get("contradictory_source_ids", []))
    )
    snapshot = _gap_closure_snapshot(
        gap,
        matched_ids=matched_ids,
        supported_claims=supported_claims,
        contradictory_ids=contradictory_ids,
        selected_by_id=selected_by_id,
        configurable=configurable,
        direct_evidence_confirmed=bool(
            gap.get("direct_evidence_confirmed")
            or assessment.get("directly_answers_gap")
        ),
    )
    requested_type_satisfied = snapshot["requested_type_satisfied"]
    independent_domains = snapshot["independent_domains"]
    required_independent = snapshot["required_independent"]
    direct_evidence_confirmed = snapshot["direct_evidence_confirmed"]
    closure_blockers = snapshot["closure_blockers"]
    prior_blockers = set(gap.get("closure_blockers", []))
    blockers_resolved = sorted(prior_blockers - set(closure_blockers))
    has_progress = bool(assessment.get("has_progress") or blockers_resolved)
    attempt_count = int(gap.get("attempt_count", 0)) + 1
    no_progress_count = 0 if has_progress else int(gap.get("no_progress_count", 0)) + 1
    can_close = bool(not closure_blockers)

    if can_close:
        status = "closed"
        gap_route = "closed"
        closure_reason = (
            f"Direct accepted evidence from {len(independent_domains)} independent "
            "source domain(s) satisfied the gap requirements."
        )
    elif assessment.get("assessment_status") == "structured_output_failure":
        status = "unresolvable"
        gap_route = "unresolvable"
        closure_reason = (
            "Gap evidence could not be assessed after the bounded structured-output "
            "retry; repeating the same web search would not repair the assessment."
        )
    elif (
        attempt_count >= state["max_research_loops"]
        or no_progress_count >= configurable.max_gap_no_progress_attempts
    ):
        status = "unresolvable"
        gap_route = "unresolvable"
        closure_reason = (
            "The bounded search strategy was exhausted without evidence that met "
            "the deterministic closure requirements."
        )
    elif has_progress:
        status = "partial"
        gap_route = "progressing"
        closure_reason = ""
    else:
        status = "partial"
        gap_route = "stalled"
        closure_reason = ""

    gap.update(
        {
            "status": status,
            "attempt_count": attempt_count,
            "no_progress_count": no_progress_count,
            "matched_source_ids": matched_ids,
            "supported_claims": supported_claims,
            "contradictory_source_ids": contradictory_ids,
            "direct_evidence_confirmed": direct_evidence_confirmed,
            "requested_source_type_satisfied": requested_type_satisfied,
            "independent_source_count": len(independent_domains),
            "closure_reason": closure_reason,
            "remaining_evidence": assessment.get("remaining_evidence", ""),
            "closure_blockers": closure_blockers,
            "required_independent_source_count": required_independent,
            "assessment_status": assessment.get("assessment_status", "unknown"),
        }
    )
    registry[gap_id] = gap
    resolved_ids = set(state.get("resolved_gap_ids", []))
    if status == "closed":
        resolved_ids.add(gap_id)
    else:
        resolved_ids.discard(gap_id)
    coverage_ids = set(state.get("gap_source_coverage_ids", []))
    if direct_evidence_confirmed and requested_type_satisfied:
        coverage_ids.add(gap_id)
    new_source_count = len(assessment.get("new_matched_source_ids", []))
    gain = {
        "loop": state.get("research_loop_count", 0) + 1,
        "gap_id": gap_id,
        "new_source_count": new_source_count,
        "new_accepted_source_count": new_source_count,
        # A paraphrase of an earlier LLM claim is not independent evidence gain.
        # Claim-only progress is represented by a resolved closure blocker below.
        "new_supported_claim_count": (
            len(assessment.get("new_supported_claims", [])) if new_source_count else 0
        ),
        "resolved_gap_count": int(status == "closed"),
        "resolved_blocker_count": len(blockers_resolved),
    }
    gain["total_gain"] = (
        gain["new_accepted_source_count"]
        + gain["new_supported_claim_count"]
        + gain["resolved_gap_count"]
        + gain["resolved_blocker_count"]
    )
    emit_research_event(
        "gap_status_updated",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        gap=gap,
        route=gap_route,
        evidence_gain=gain,
        closure_blockers=closure_blockers,
    )
    return {
        "gap_registry": registry,
        "active_gap": gap,
        "gap_route": gap_route,
        "research_loop_count": state.get("research_loop_count", 0) + 1,
        "resolved_gap_ids": sorted(resolved_ids),
        "gap_source_coverage_ids": sorted(coverage_ids),
        "evidence_gain_history": [gain],
    }


def route_gap_progress(state: DimensionState):
    """Route deterministic gap status to retry, replan, or next selection."""
    route = state.get("gap_route")
    if route == "progressing":
        return "generate_query"
    if route == "stalled":
        return "replan_search"
    return "select_next_gap"


def replan_search(state: DimensionState, config: RunnableConfig):
    """Escalate the active gap's search strategy after a no-progress pass."""
    configurable = Configuration.from_runnable_config(config)
    registry = {
        key: dict(value) for key, value in state.get("gap_registry", {}).items()
    }
    gap_id = state.get("active_gap_id", "")
    gap = dict(registry[gap_id])
    level = int(gap.get("strategy_level", 0)) + 1
    blockers = gap.get("closure_blockers", [])
    blocker_guidance = {
        "missing_direct_evidence": "Search the exact expected fact, metric, date, unit, or quoted statement instead of broad topic coverage.",
        "missing_supported_claim": "Target documents containing extractable factual statements and data, not landing pages or summaries.",
        "missing_requested_source_type": "Target the still-missing requested source type and name likely official institutions explicitly.",
        "insufficient_independent_sources": "Exclude already saturated publisher domains and find an independent organization confirming the evidence.",
        "unresolved_contradiction": "Search primary documents that define scope, date, unit, and methodology needed to reconcile the contradiction.",
    }
    generic_strategies = {
        1: "Target named authoritative institutions and restrict queries to their domains.",
        2: "Search for the original document, dataset, publication title, author, and date.",
        3: "Split the evidence requirement into narrower factual subquestions and use source-specific terminology.",
    }
    guidance_parts = [
        blocker_guidance[blocker] for blocker in blockers if blocker in blocker_guidance
    ]
    if not guidance_parts:
        guidance_parts = [
            generic_strategies.get(
                level,
                "Use exact phrases, multilingual terminology, and archival or bibliographic discovery queries.",
            )
        ]
    domain_counts: dict[str, int] = {}
    for source in state.get("selected_sources", []):
        if source.get("quality_status") != "accepted":
            continue
        domain = str(source.get("domain", "")).strip()
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    saturated_domains = sorted(
        domain
        for domain, count in domain_counts.items()
        if count >= configurable.max_sources_per_domain
    )
    if saturated_domains:
        guidance_parts.append(
            "Do not target these saturated domains: " + ", ".join(saturated_domains)
        )
    guidance = " ".join(dict.fromkeys(guidance_parts))
    gap["strategy_level"] = level
    gap["search_strategy"] = [guidance]
    gap["excluded_domains"] = saturated_domains
    gap["status"] = "active"
    registry[gap_id] = gap
    emit_research_event(
        "search_replanned",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        gap_id=gap_id,
        strategy_level=level,
        strategy=guidance,
    )
    return {"gap_registry": registry, "active_gap": gap}


def _quality_requirement_gaps(
    state: DimensionState, configurable: Configuration
) -> list[ResearchGap]:
    """Return deterministic source-quality gaps that remain for a dimension."""
    selected = state.get("selected_sources", [])
    accepted = [
        source for source in selected if source.get("quality_status") == "accepted"
    ]
    gaps = []
    if len(accepted) < configurable.min_accepted_sources_per_dimension:
        gaps.append(
            ResearchGap(
                gap_id="quality-accepted-sources",
                question="Which additional independent sources directly support this dimension?",
                reason="The minimum accepted-source requirement has not been met.",
                priority="high",
                required_source_types=sorted(AUTHORITATIVE_SOURCE_TYPES),
                expected_evidence="Direct, independent evidence from an accepted source.",
                suggested_query_focus="Find independent authoritative evidence for this dimension.",
            )
        )
    if (
        sum(bool(source.get("is_authoritative_source")) for source in accepted)
        < configurable.min_authoritative_sources_per_dimension
    ):
        gaps.append(
            ResearchGap(
                gap_id="quality-authoritative-source",
                question="What authoritative source directly supports this dimension?",
                reason="The authoritative-source requirement has not been met.",
                priority="high",
                required_source_types=sorted(AUTHORITATIVE_SOURCE_TYPES),
                expected_evidence="A direct statement or data point from an authoritative source.",
                suggested_query_focus="Search official, academic, standards, or institutional sources.",
            )
        )
    if (
        sum(bool(source.get("is_primary_source")) for source in accepted)
        < configurable.min_primary_sources_per_dimension
    ):
        gaps.append(
            ResearchGap(
                gap_id="quality-primary-source",
                question="What primary source directly supports this dimension?",
                reason="The primary-source requirement has not been met.",
                priority="high",
                required_source_types=[
                    "government",
                    "official_company",
                    "standards_body",
                    "academic",
                ],
                expected_evidence="First-party data, documentation, regulation, or original research.",
                suggested_query_focus="Find the original official document, dataset, or publication.",
            )
        )
    return gaps


def dimension_reflection(state: DimensionState, config: RunnableConfig):
    """Audit complete dimension coverage and discover only actionable new gaps."""
    configurable = Configuration.from_runnable_config(config)
    reflection_count = state.get("dimension_reflection_count", 0) + 1
    registry = {
        key: dict(value) for key, value in state.get("gap_registry", {}).items()
    }
    selected = state.get("selected_sources", [])
    accepted = [
        source for source in selected if source.get("quality_status") == "accepted"
    ]
    minimum_source_requirements = {
        "accepted": configurable.min_accepted_sources_per_dimension,
        "authoritative": configurable.min_authoritative_sources_per_dimension,
        "primary": configurable.min_primary_sources_per_dimension,
        "current": {
            "accepted": len(accepted),
            "authoritative": sum(
                bool(source.get("is_authoritative_source")) for source in accepted
            ),
            "primary": sum(
                bool(source.get("is_primary_source")) for source in accepted
            ),
        },
    }
    nonaccepted = deduplicate_sources_by_id(
        [
            *[
                source
                for source in selected
                if source.get("quality_status") != "accepted"
            ],
            *state.get("rejected_sources", []),
        ]
    )
    prompt = reflection_instructions.format(
        research_topic=state["research_topic"],
        dimension_title=state["dimension"]["title"],
        dimension_scope=state["dimension"]["scope"],
        gap_registry=json.dumps(registry, ensure_ascii=False),
        summaries=format_sources_for_research(accepted),
        rejected_source_summary=format_rejected_source_summary(nonaccepted),
        previous_reflection=json.dumps(
            state.get("reflection_history", [])[-1:] or [], ensure_ascii=False
        ),
        query_history=state.get("query_history", []) or "None.",
        evidence_gain_history=json.dumps(
            state.get("evidence_gain_history", [])[-5:], ensure_ascii=False
        ),
        minimum_source_requirements=json.dumps(
            minimum_source_requirements, ensure_ascii=False
        ),
    )
    try:
        result = (
            create_deepseek_model(configurable.reflection_model)
            .with_structured_output(Reflection, method="json_mode")
            .invoke(prompt)
        )
    except (
        AttributeError,
        LengthFinishReasonError,
        OutputParserException,
        TypeError,
        ValueError,
    ) as error:
        emit_research_event(
            "reflection_fallback",
            research_run_id=state["research_run_id"],
            dimension=state["dimension"],
            error=str(error),
        )
        result = Reflection(
            is_sufficient=False,
            covered_questions=[],
            missing_questions=[
                ResearchGap(
                    gap_id="reflection-unresolved-evidence",
                    question=f"What material evidence remains missing for {state['dimension']['title']}?",
                    reason="The structured whole-dimension audit could not be parsed safely.",
                    priority="high",
                    required_source_types=_default_source_types(
                        state["research_topic"], state["dimension"]
                    ),
                    expected_evidence=state["dimension"]["scope"],
                    suggested_query_focus=state["dimension"]["scope"],
                )
            ],
            unsupported_claims=[],
            contradictions=[],
            source_quality_issues=["Structured dimension reflection failed."],
            recommended_search_strategy=[],
            do_not_repeat=state.get("query_history", []),
            completion_reason="The audit could not safely declare the dimension sufficient.",
            confidence=0,
        )

    candidates = [
        _operational_gap(item.model_dump(), origin="discovered")
        for item in result.missing_questions
    ]
    candidates.extend(
        _operational_gap(item.model_dump(), origin="quality")
        for item in _quality_requirement_gaps(state, configurable)
    )
    unresolved_terminal = [
        gap for gap in registry.values() if gap.get("status") == "unresolvable"
    ]
    if not result.is_sufficient and not candidates and not unresolved_terminal:
        candidates.append(
            _operational_gap(
                ResearchGap(
                    gap_id="reflection-unresolved-evidence",
                    question=f"What material evidence remains missing for {state['dimension']['title']}?",
                    reason=result.completion_reason
                    or "The dimension audit did not declare the evidence sufficient.",
                    priority="high",
                    required_source_types=_default_source_types(
                        state["research_topic"], state["dimension"]
                    ),
                    expected_evidence="Direct evidence that resolves the remaining dimension-level uncertainty.",
                    suggested_query_focus=(
                        result.recommended_search_strategy[0]
                        if result.recommended_search_strategy
                        else state["dimension"]["scope"]
                    ),
                ).model_dump(),
                origin="discovered",
            )
        )
    pending: list[dict] = []
    for candidate in candidates:
        existing = registry.get(candidate["gap_id"])
        if existing and existing.get("status") == "unresolvable":
            prior_requirement = (
                existing.get("question"),
                existing.get("expected_evidence"),
                tuple(existing.get("required_source_types", [])),
            )
            new_requirement = (
                candidate.get("question"),
                candidate.get("expected_evidence"),
                tuple(candidate.get("required_source_types", [])),
            )
            if prior_requirement == new_requirement:
                continue
        if existing and existing.get("status") == "closed":
            candidate["origin"] = "reopened"
            candidate["status"] = "reopened"
        pending.append(candidate)

    quality_gaps = _quality_requirement_gaps(state, configurable)
    unresolved_conflict = any(
        conflict.requires_follow_up for conflict in result.contradictions
    )
    deterministic_sufficient = bool(
        result.is_sufficient
        and not pending
        and not quality_gaps
        and not unresolved_terminal
        and not unresolved_conflict
    )
    search_unavailable = bool(
        not accepted
        and state.get("search_failures")
        and state.get("search_success_count", 0) == 0
    )
    if search_unavailable:
        completion_status = "search_unavailable"
    elif deterministic_sufficient:
        completion_status = "sufficient"
    elif pending and reflection_count < configurable.max_dimension_reflections:
        completion_status = "discovering_gaps"
    elif reflection_count >= configurable.max_dimension_reflections:
        completion_status = "budget_exhausted"
    else:
        completion_status = "partial"

    unresolved = [gap for gap in registry.values() if gap.get("status") != "closed"]
    missing_by_id = {gap["gap_id"]: gap for gap in unresolved}
    missing_by_id.update({gap["gap_id"]: gap for gap in pending})
    assessment = result.model_dump()
    assessment["is_sufficient"] = deterministic_sufficient
    assessment["missing_questions"] = list(missing_by_id.values())
    if quality_gaps:
        assessment["source_quality_issues"] = list(
            dict.fromkeys(
                [
                    *assessment.get("source_quality_issues", []),
                    "Deterministic minimum source requirements are not met.",
                ]
            )
        )
    knowledge_gap = "\n".join(
        f"[{gap['gap_id']}] {gap['question']}: {gap.get('remaining_evidence') or gap.get('expected_evidence', '')}"
        for gap in assessment["missing_questions"]
    )
    emit_research_event(
        "reflection_completed",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        is_sufficient=deterministic_sufficient,
        missing_questions=assessment["missing_questions"],
        contradictions=[item.model_dump() for item in result.contradictions],
        confidence=result.confidence,
        reflection=reflection_count,
        completion_status=completion_status,
        knowledge_gap=knowledge_gap,
    )
    return {
        "is_sufficient": deterministic_sufficient,
        "completion_status": completion_status,
        "reflection_assessment": assessment,
        "reflection_history": [assessment],
        "pending_reflection_gaps": pending,
        "dimension_reflection_count": reflection_count,
        "current_knowledge_gap": knowledge_gap,
    }


def route_dimension_reflection(state: DimensionState):
    """Merge newly discovered gaps or finish with sufficient/partial evidence."""
    if state.get("completion_status") == "discovering_gaps":
        return "merge_gap_registry"
    return "extract_claims"


def merge_gap_registry(state: DimensionState):
    """Merge newly discovered or reopened gaps into the lifecycle registry."""
    registry = {
        key: dict(value) for key, value in state.get("gap_registry", {}).items()
    }
    resolved_ids = set(state.get("resolved_gap_ids", []))
    merged_ids = []
    for candidate_value in state.get("pending_reflection_gaps", []):
        candidate = dict(candidate_value)
        gap_id = candidate["gap_id"]
        existing = registry.get(gap_id)
        if existing is None:
            candidate["origin"] = candidate.get("origin") or "discovered"
            candidate["status"] = "open"
            registry[gap_id] = candidate
            resolved_ids.discard(gap_id)
            merged_ids.append(gap_id)
            continue
        if existing.get("status") == "closed":
            candidate.update(
                {
                    "origin": "reopened",
                    "status": "reopened",
                    "attempt_count": 0,
                    "no_progress_count": 0,
                    "strategy_level": int(existing.get("strategy_level", 0)) + 1,
                    "matched_source_ids": existing.get("matched_source_ids", []),
                    "supported_claims": existing.get("supported_claims", []),
                }
            )
            registry[gap_id] = candidate
            resolved_ids.discard(gap_id)
            merged_ids.append(gap_id)
            continue
        if existing.get("status") == "unresolvable":
            prior_requirement = (
                existing.get("question"),
                existing.get("expected_evidence"),
                tuple(existing.get("required_source_types", [])),
            )
            new_requirement = (
                candidate.get("question"),
                candidate.get("expected_evidence"),
                tuple(candidate.get("required_source_types", [])),
            )
            if prior_requirement == new_requirement:
                continue
            candidate.update(
                {
                    "origin": "reopened",
                    "status": "reopened",
                    "attempt_count": 0,
                    "no_progress_count": 0,
                }
            )
            registry[gap_id] = candidate
            resolved_ids.discard(gap_id)
            merged_ids.append(gap_id)

    emit_research_event(
        "gap_registry_merged",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        merged_gap_ids=merged_ids,
        gap_count=len(registry),
    )
    return {
        "gap_registry": registry,
        "resolved_gap_ids": sorted(resolved_ids),
        "pending_reflection_gaps": [],
        "gap_processing_complete": False,
        "completion_status": "researching",
    }


def _reconcile_extracted_claims(
    state: DimensionState,
    claims: list[dict],
    configurable: Configuration,
) -> dict[str, Any]:
    """Backfill verified claim evidence into its provenance-constrained gaps."""
    registry = {
        key: dict(value) for key, value in state.get("gap_registry", {}).items()
    }
    selected_by_id = {
        source["source_id"]: source
        for source in state.get("selected_sources", [])
        if source.get("quality_status") == "accepted"
    }
    resolved_ids = set(state.get("resolved_gap_ids", []))
    coverage_ids = set(state.get("gap_source_coverage_ids", []))
    reconciled_claim_count = 0
    for claim in claims:
        supporting_ids = claim.get("supporting_source_ids", [])
        for gap_id in claim.get("gap_ids", []):
            gap = registry.get(gap_id)
            if gap is None:
                continue
            provenance_ids = [
                source_id
                for source_id in supporting_ids
                if source_id in selected_by_id
                and (
                    gap_id == selected_by_id[source_id].get("gap_id")
                    or gap_id in selected_by_id[source_id].get("gap_ids", [])
                )
            ]
            if not provenance_ids:
                continue
            matched_ids = list(
                dict.fromkeys([*gap.get("matched_source_ids", []), *provenance_ids])
            )
            supported_claims = list(
                dict.fromkeys([*gap.get("supported_claims", []), claim["claim"]])
            )
            contradictory_ids = list(
                dict.fromkeys(gap.get("contradictory_source_ids", []))
            )
            snapshot = _gap_closure_snapshot(
                gap,
                matched_ids=matched_ids,
                supported_claims=supported_claims,
                contradictory_ids=contradictory_ids,
                selected_by_id=selected_by_id,
                configurable=configurable,
                direct_evidence_confirmed=True,
            )
            gap.update(
                {
                    "matched_source_ids": matched_ids,
                    "supported_claims": supported_claims,
                    "direct_evidence_confirmed": True,
                    "requested_source_type_satisfied": snapshot[
                        "requested_type_satisfied"
                    ],
                    "independent_source_count": len(snapshot["independent_domains"]),
                    "required_independent_source_count": snapshot[
                        "required_independent"
                    ],
                    "closure_blockers": snapshot["closure_blockers"],
                    "assessment_status": "claim_reconciled",
                }
            )
            if not snapshot["closure_blockers"]:
                gap["status"] = "closed"
                gap["closure_reason"] = (
                    "Verified extracted claims and provenance-constrained accepted "
                    "evidence satisfied every closure requirement."
                )
                resolved_ids.add(gap_id)
            if (
                snapshot["direct_evidence_confirmed"]
                and snapshot["requested_type_satisfied"]
            ):
                coverage_ids.add(gap_id)
            registry[gap_id] = gap
            reconciled_claim_count += 1
    return {
        "gap_registry": registry,
        "resolved_gap_ids": sorted(resolved_ids),
        "gap_source_coverage_ids": sorted(coverage_ids),
        "claim_reconciled_count": reconciled_claim_count,
    }


def extract_claims(state: DimensionState, config: RunnableConfig):
    """Convert selected evidence into an auditable claim set."""
    configurable = Configuration.from_runnable_config(config)
    selected = [
        source
        for source in state.get("selected_sources", [])
        if source.get("quality_status") == "accepted"
    ]
    if not selected:
        emit_research_event(
            "claims_extracted",
            research_run_id=state["research_run_id"],
            dimension=state["dimension"],
            claim_count=0,
            invalid_evidence_count=0,
        )
        return {
            "claims": [],
            "dimension_summary": "No claim was extracted because no quality-screened evidence was available.",
        }

    def build_prompt(sources, content_chars, max_claims):
        return claim_extraction_instructions.format(
            research_topic=state["research_topic"],
            dimension_title=state["dimension"]["title"],
            dimension_scope=state["dimension"]["scope"],
            max_claims=max_claims,
            output_schema=json.dumps(
                ClaimExtraction.model_json_schema(), ensure_ascii=False
            ),
            reflection_assessment=json.dumps(
                state.get("reflection_assessment", {}), ensure_ascii=False
            ),
            gap_registry=json.dumps(state.get("gap_registry", {}), ensure_ascii=False),
            selected_evidence=format_sources_for_research(
                sources, max_content_chars=content_chars
            ),
        )

    structured_model = create_deepseek_model(
        configurable.reflection_model
    ).with_structured_output(ClaimExtraction, method="json_mode")

    def compact_retry(reason: str):
        retry_source_limit = min(len(selected), 6)
        retry_claim_limit = min(configurable.max_claims_per_dimension, 8)
        emit_research_event(
            "claim_extraction_retry",
            research_run_id=state["research_run_id"],
            dimension=state["dimension"],
            reason=reason,
            source_count=retry_source_limit,
            max_claims=retry_claim_limit,
        )
        return structured_model.invoke(
            build_prompt(
                selected[:retry_source_limit],
                min(configurable.max_claim_source_chars, 1000),
                retry_claim_limit,
            )
            + "\nReturn only claims with verbatim quote text copied from Content."
        )

    try:
        result = structured_model.invoke(
            build_prompt(
                selected,
                configurable.max_claim_source_chars,
                configurable.max_claims_per_dimension,
            )
        )
    except (LengthFinishReasonError, OutputParserException) as error:
        reason = (
            "length_limit"
            if isinstance(error, LengthFinishReasonError)
            else "structured_output_failure"
        )
        try:
            result = compact_retry(reason)
        except (LengthFinishReasonError, OutputParserException) as retry_error:
            emit_research_event(
                "claim_extraction_failed",
                research_run_id=state["research_run_id"],
                dimension=state["dimension"],
                error=f"{type(retry_error).__name__}: {retry_error}",
            )
            return {
                "claims": [],
                "dimension_summary": "No claim passed structured evidence extraction.",
            }
    source_by_id = {source["source_id"]: source for source in selected}
    valid_gap_ids = set(state.get("gap_registry", {}))

    def validate_extraction(extraction):
        validated_claims = []
        invalid_count = 0
        for claim in extraction.claims[: configurable.max_claims_per_dimension]:
            evidence_groups = []
            for evidence_items in (claim.evidence, claim.counter_evidence):
                verified = []
                for evidence in evidence_items:
                    source = source_by_id.get(evidence.source_id)
                    located = (
                        locate_evidence_quote(
                            source["content"],
                            evidence.quote,
                            min_chars=configurable.min_evidence_quote_chars,
                        )
                        if source
                        else None
                    )
                    if located is None:
                        invalid_count += 1
                        continue
                    quote, locator = located
                    verified.append(
                        {
                            "source_id": evidence.source_id,
                            "quote": quote,
                            "locator": locator,
                        }
                    )
                evidence_groups.append(verified)
            supporting_evidence, contradicting_evidence = evidence_groups
            if not supporting_evidence:
                continue
            supporting_ids = {item["source_id"] for item in supporting_evidence}
            mapped_gap_ids = []
            for gap_id in claim.gap_ids:
                if gap_id not in valid_gap_ids:
                    continue
                if any(
                    gap_id == source_by_id[source_id].get("gap_id")
                    or gap_id in source_by_id[source_id].get("gap_ids", [])
                    for source_id in supporting_ids
                    if source_id in source_by_id
                ):
                    mapped_gap_ids.append(gap_id)
            validated_claims.append(
                {
                    "claim": claim.claim,
                    "gap_ids": list(dict.fromkeys(mapped_gap_ids)),
                    "supporting_source_ids": list(
                        dict.fromkeys(item["source_id"] for item in supporting_evidence)
                    ),
                    "supporting_evidence": supporting_evidence,
                    "contradicting_source_ids": list(
                        dict.fromkeys(
                            item["source_id"] for item in contradicting_evidence
                        )
                    ),
                    "contradicting_evidence": contradicting_evidence,
                    "confidence": claim.confidence,
                    "uncertainty_reason": claim.uncertainty,
                }
            )
        return validated_claims, invalid_count

    claims, invalid_evidence_count = validate_extraction(result)
    dimension_summary = result.summary
    minimum_expected_claims = max(1, (len(result.claims) + 1) // 2)
    if selected and invalid_evidence_count and len(claims) < minimum_expected_claims:
        try:
            repaired_result = compact_retry("invalid_evidence_quotes")
        except (LengthFinishReasonError, OutputParserException) as repair_error:
            emit_research_event(
                "claim_evidence_repair_failed",
                research_run_id=state["research_run_id"],
                dimension=state["dimension"],
                error=f"{type(repair_error).__name__}: {repair_error}",
            )
        else:
            repaired_claims, repaired_invalid_count = validate_extraction(
                repaired_result
            )
            if repaired_claims:
                dimension_summary = repaired_result.summary
            invalid_evidence_count += repaired_invalid_count
            claims_by_text = {claim["claim"].casefold(): claim for claim in claims}
            claims_by_text.update(
                {claim["claim"].casefold(): claim for claim in repaired_claims}
            )
            claims = list(claims_by_text.values())[
                : configurable.max_claims_per_dimension
            ]
    emit_research_event(
        "claims_extracted",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        claim_count=len(claims),
        invalid_evidence_count=invalid_evidence_count,
    )
    reconciliation = _reconcile_extracted_claims(state, claims, configurable)
    emit_research_event(
        "claims_reconciled_to_gaps",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        reconciled_claim_count=reconciliation["claim_reconciled_count"],
        resolved_gap_count=len(reconciliation["resolved_gap_ids"]),
    )
    return {
        "claims": claims,
        "dimension_summary": dimension_summary,
        "gap_registry": reconciliation["gap_registry"],
        "resolved_gap_ids": reconciliation["resolved_gap_ids"],
        "gap_source_coverage_ids": reconciliation["gap_source_coverage_ids"],
    }


dimension_builder = StateGraph(DimensionState, input_schema=DimensionInput)
dimension_builder.add_node("plan_initial_gaps", plan_initial_gaps)
dimension_builder.add_node("select_next_gap", select_next_gap)
dimension_builder.add_node("generate_query", generate_query)
dimension_builder.add_node("web_research", web_research)
dimension_builder.add_node("evaluate_sources", evaluate_sources)
dimension_builder.add_node("assess_gap_evidence", assess_gap_evidence)
dimension_builder.add_node("update_gap_status", update_gap_status)
dimension_builder.add_node("replan_search", replan_search)
dimension_builder.add_node("dimension_reflection", dimension_reflection)
dimension_builder.add_node("merge_gap_registry", merge_gap_registry)
dimension_builder.add_node("extract_claims", extract_claims)
dimension_builder.add_edge(START, "plan_initial_gaps")
dimension_builder.add_edge("plan_initial_gaps", "select_next_gap")
dimension_builder.add_conditional_edges(
    "select_next_gap",
    route_gap_selection,
    ["generate_query", "dimension_reflection"],
)
dimension_builder.add_conditional_edges(
    "generate_query", dispatch_search_queries, ["web_research"]
)
dimension_builder.add_edge("web_research", "evaluate_sources")
dimension_builder.add_edge("evaluate_sources", "assess_gap_evidence")
dimension_builder.add_edge("assess_gap_evidence", "update_gap_status")
dimension_builder.add_conditional_edges(
    "update_gap_status",
    route_gap_progress,
    ["generate_query", "replan_search", "select_next_gap"],
)
dimension_builder.add_edge("replan_search", "generate_query")
dimension_builder.add_conditional_edges(
    "dimension_reflection",
    route_dimension_reflection,
    ["merge_gap_registry", "extract_claims"],
)
dimension_builder.add_edge("merge_gap_registry", "select_next_gap")
dimension_builder.add_edge("extract_claims", END)
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
    gap_registry = result.get("gap_registry", {})
    status_counts: dict[str, int] = {}
    for gap in gap_registry.values():
        status = gap.get("status", "open")
        status_counts[status] = status_counts.get(status, 0) + 1
    accepted_sources = [
        source
        for source in result.get("selected_sources", [])
        if source.get("quality_status") == "accepted"
    ]
    accepted_by_id = {source["source_id"]: source for source in accepted_sources}
    gap_diagnostics = []
    for gap_id, gap in sorted(gap_registry.items()):
        matched_source_ids = list(dict.fromkeys(gap.get("matched_source_ids", [])))
        matched_sources = [
            {
                "source_id": source_id,
                "title": accepted_by_id[source_id].get("title", ""),
                "url": accepted_by_id[source_id].get("url", ""),
                "domain": accepted_by_id[source_id].get("domain", ""),
                "source_type": accepted_by_id[source_id].get("source_type", "unknown"),
                "is_primary_source": bool(
                    accepted_by_id[source_id].get("is_primary_source")
                ),
                "is_authoritative_source": bool(
                    accepted_by_id[source_id].get("is_authoritative_source")
                ),
            }
            for source_id in matched_source_ids
            if source_id in accepted_by_id
        ]
        matched_source_types = sorted(
            {
                source["source_type"]
                for source in matched_sources
                if source["source_type"] != "unknown"
            }
        )
        closure_blockers = list(gap.get("closure_blockers", []))
        blocker_details = []
        for blocker in closure_blockers:
            if blocker == "missing_requested_source_type":
                blocker_details.append(
                    "Required source types "
                    f"{gap.get('required_source_types', [])} were not found among "
                    f"matched types {matched_source_types}."
                )
            elif blocker == "insufficient_independent_sources":
                blocker_details.append(
                    "Independent publisher coverage is "
                    f"{int(gap.get('independent_source_count', 0))}/"
                    f"{int(gap.get('required_independent_source_count', 1))}."
                )
            elif blocker == "missing_direct_evidence":
                blocker_details.append(
                    "No accepted, quote-verified evidence directly answers the gap."
                )
            elif blocker == "missing_supported_claim":
                blocker_details.append(
                    "No quote-verified factual claim currently supports the gap."
                )
            elif blocker == "unresolved_contradiction":
                blocker_details.append(
                    "Contradictory evidence remains unresolved: "
                    + ", ".join(gap.get("contradictory_source_ids", []))
                )
        gap_diagnostics.append(
            {
                "dimension_id": str(result["dimension"]["id"]),
                "dimension_title": result["dimension"]["title"],
                "gap_id": gap_id,
                "origin": gap.get("origin", "unknown"),
                "priority": gap.get("priority", "unknown"),
                "question": gap.get("question", ""),
                "expected_evidence": gap.get("expected_evidence", ""),
                "required_source_types": gap.get("required_source_types", []),
                "status": gap.get("status", "open"),
                "closure_blockers": closure_blockers,
                "blocker_details": blocker_details,
                "closure_reason": gap.get("closure_reason", ""),
                "remaining_evidence": gap.get("remaining_evidence", ""),
                "assessment_status": gap.get("assessment_status", "not_assessed"),
                "attempt_count": int(gap.get("attempt_count", 0)),
                "no_progress_count": int(gap.get("no_progress_count", 0)),
                "strategy_level": int(gap.get("strategy_level", 0)),
                "direct_evidence_confirmed": bool(gap.get("direct_evidence_confirmed")),
                "supported_claim_count": len(gap.get("supported_claims", [])),
                "supported_claims": gap.get("supported_claims", []),
                "contradictory_source_ids": gap.get("contradictory_source_ids", []),
                "requested_source_type_satisfied": bool(
                    gap.get("requested_source_type_satisfied")
                ),
                "independent_source_count": int(gap.get("independent_source_count", 0)),
                "required_independent_source_count": int(
                    gap.get("required_independent_source_count", 1)
                ),
                "matched_source_ids": matched_source_ids,
                "matched_source_types": matched_source_types,
                "matched_sources": matched_sources,
                "search_strategy": gap.get("search_strategy", []),
                "excluded_domains": gap.get("excluded_domains", []),
                "missing_matched_source_ids": [
                    source_id
                    for source_id in matched_source_ids
                    if source_id not in accepted_by_id
                ],
            }
        )
    dimension_result = {
        "research_run_id": result["research_run_id"],
        "dimension": result["dimension"],
        "research_content": result.get("dimension_summary", ""),
        "sources": accepted_sources,
        "research_loop_count": result["research_loop_count"],
        "is_sufficient": result["is_sufficient"],
        "completion_status": result["completion_status"],
        "covered_questions": result["reflection_assessment"].get(
            "covered_questions", []
        ),
        "unresolved_gaps": result["reflection_assessment"].get("missing_questions", []),
        "contradictions": result["reflection_assessment"].get("contradictions", []),
        "source_quality_issues": result["reflection_assessment"].get(
            "source_quality_issues", []
        ),
        "confidence": result["reflection_assessment"].get("confidence", 0),
        "claims": result.get("claims", []),
        "known_gap_count": len(gap_registry),
        "resolved_gap_count": len(
            set(result.get("resolved_gap_ids", []))
            & set(result.get("gap_registry", {}))
        ),
        "high_priority_gap_count": sum(
            gap.get("priority") == "high"
            for gap in result.get("gap_registry", {}).values()
        ),
        "resolved_high_priority_gap_count": sum(
            gap_id in set(result.get("resolved_gap_ids", []))
            and gap.get("priority") == "high"
            for gap_id, gap in result.get("gap_registry", {}).items()
        ),
        "high_priority_gap_source_coverage_count": sum(
            gap_id in set(result.get("gap_source_coverage_ids", []))
            and gap.get("priority") == "high"
            for gap_id, gap in result.get("gap_registry", {}).items()
        ),
        "evidence_gain_history": result.get("evidence_gain_history", []),
        "search_failure_count": len(result.get("search_failures", [])),
        "closed_gap_count": status_counts.get("closed", 0),
        "unresolvable_gap_count": status_counts.get("unresolvable", 0),
        "gap_status_counts": status_counts,
        "direct_evidence_gap_count": sum(
            bool(gap.get("direct_evidence_confirmed")) for gap in gap_registry.values()
        ),
        "supported_claim_gap_count": sum(
            bool(gap.get("supported_claims")) for gap in gap_registry.values()
        ),
        "requested_type_gap_count": sum(
            bool(gap.get("requested_source_type_satisfied"))
            for gap in gap_registry.values()
        ),
        "independent_source_gap_count": sum(
            int(gap.get("independent_source_count", 0))
            >= int(gap.get("required_independent_source_count", 1))
            for gap in gap_registry.values()
        ),
        "gap_assessment_failure_count": int(
            result.get("gap_assessment_failure_count", 0)
        ),
        "no_gain_loop_count": sum(
            int(gain.get("total_gain", 0)) <= 0
            for gain in result.get("evidence_gain_history", [])
        ),
        "gap_diagnostics": gap_diagnostics,
    }
    emit_research_event(
        "dimension_completed",
        research_run_id=result["research_run_id"],
        dimension=result["dimension"],
        is_sufficient=result["is_sufficient"],
        loops=result["research_loop_count"],
        completion_status=result["completion_status"],
    )
    return {
        "dimension_results": [dimension_result],
        "sources_gathered": accepted_sources,
    }


def _current_research_material(state: OverallState):
    """Return current-run results, sources, and formatted claim material."""
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
    return (
        current_results,
        deduplicate_sources_by_id(current_sources),
        format_dimension_results(current_results),
    )


def _build_report_evidence(
    state: OverallState, configurable: Configuration
) -> tuple[list[DimensionResult], list[ResearchSource], dict[str, Any]]:
    """Build a fail-closed report ledger from final accepted evidence only."""
    current_results, current_sources, _ = _current_research_material(state)
    source_candidates = deduplicate_sources_by_id(
        [
            *current_sources,
            *[
                source
                for result in current_results
                for source in result.get("sources", [])
            ],
        ]
    )
    accepted_sources = [
        source
        for source in source_candidates
        if source.get("quality_status") == "accepted"
    ]
    accepted_by_id = {source["source_id"]: source for source in accepted_sources}
    rejected_source_ids = sorted(
        {
            source["source_id"]
            for source in source_candidates
            if source.get("quality_status") != "accepted"
        }
    )
    invalid_evidence_count = 0
    rejected_claim_count = 0
    report_results: list[DimensionResult] = []

    def verified_evidence(items: object) -> list[dict]:
        nonlocal invalid_evidence_count
        verified: list[dict[str, str]] = []
        if not isinstance(items, list):
            return verified
        for item in items:
            if not isinstance(item, dict):
                invalid_evidence_count += 1
                continue
            source_id = str(item.get("source_id", ""))
            source = accepted_by_id.get(source_id)
            located = (
                locate_evidence_quote(
                    str(source.get("content", "")),
                    str(item.get("quote", "")),
                    min_chars=configurable.min_evidence_quote_chars,
                )
                if source
                else None
            )
            if located is None:
                invalid_evidence_count += 1
                continue
            quote, locator = located
            verified.append(
                {"source_id": source_id, "quote": quote, "locator": locator}
            )
        return verified

    for result in current_results:
        dimension_id = str(result["dimension"]["id"])
        claims = []
        for claim_index, claim in enumerate(result.get("claims", [])):
            supporting = verified_evidence(claim.get("supporting_evidence", []))
            counter = verified_evidence(claim.get("contradicting_evidence", []))
            if not supporting:
                rejected_claim_count += 1
                continue
            claims.append(
                {
                    **claim,
                    "claim_id": claim.get("claim_id")
                    or f"C-{dimension_id}-{claim_index + 1}",
                    "supporting_source_ids": list(
                        dict.fromkeys(item["source_id"] for item in supporting)
                    ),
                    "supporting_evidence": supporting,
                    "contradicting_source_ids": list(
                        dict.fromkeys(item["source_id"] for item in counter)
                    ),
                    "contradicting_evidence": counter,
                }
            )
        contradictions = []
        for conflict in result.get("contradictions", []):
            if not isinstance(conflict, dict):
                continue
            source_ids = [
                source_id
                for source_id in conflict.get("source_ids", [])
                if source_id in accepted_by_id
            ]
            if source_ids:
                contradictions.append({**conflict, "source_ids": source_ids})
        report_results.append(
            cast(
                DimensionResult,
                {
                    **result,
                    "sources": [
                        source
                        for source in result.get("sources", [])
                        if source.get("quality_status") == "accepted"
                        and source.get("source_id") in accepted_by_id
                    ],
                    "claims": claims,
                    "contradictions": contradictions,
                },
            )
        )

    used_source_ids = {
        source_id
        for result in report_results
        for claim in result.get("claims", [])
        for source_id in [
            *claim.get("supporting_source_ids", []),
            *claim.get("contradicting_source_ids", []),
        ]
    }
    report_sources = [
        source for source in accepted_sources if source["source_id"] in used_source_ids
    ]
    ledger = {
        "accepted_source_ids": sorted(accepted_by_id),
        "report_source_ids": sorted(used_source_ids),
        "rejected_source_ids": rejected_source_ids,
        "accepted_claim_ids": [
            claim["claim_id"]
            for result in report_results
            for claim in result.get("claims", [])
        ],
        "invalid_evidence_count": invalid_evidence_count,
        "rejected_claim_count": rejected_claim_count,
    }
    return report_results, report_sources, ledger


def prepare_report_evidence(state: OverallState, config: RunnableConfig):
    """Create the immutable evidence boundary used by every report node."""
    configurable = Configuration.from_runnable_config(config)
    results, sources, ledger = _build_report_evidence(state, configurable)
    emit_research_event(
        "report_evidence_prepared",
        research_run_id=state["research_run_id"],
        accepted_source_count=len(ledger["accepted_source_ids"]),
        report_source_count=len(sources),
        claim_count=len(ledger["accepted_claim_ids"]),
        rejected_claim_count=ledger["rejected_claim_count"],
        invalid_evidence_count=ledger["invalid_evidence_count"],
    )
    return {
        "report_dimension_results": results,
        "report_sources": sources,
        "report_evidence_ledger": ledger,
    }


def _report_research_material(
    state: OverallState, configurable: Configuration | None = None
) -> tuple[list[DimensionResult], list[ResearchSource], str]:
    """Return only the prepared report ledger, building it for direct callers."""
    if "report_dimension_results" in state and "report_sources" in state:
        results = state.get("report_dimension_results", [])
        sources = state.get("report_sources", [])
    else:
        results, sources, _ = _build_report_evidence(
            state, configurable or Configuration()
        )
    return results, sources, format_dimension_results(results)


def _claim_index(results: list[DimensionResult]) -> dict[str, dict]:
    """Index prepared claims with their dimension context."""
    return {
        claim["claim_id"]: {
            **claim,
            "dimension_id": result["dimension"]["id"],
            "dimension_title": result["dimension"]["title"],
        }
        for result in results
        for claim in result.get("claims", [])
        if claim.get("claim_id")
    }


def _format_conflict_ledger(conflicts: list[dict]) -> str:
    """Render compact, identifier-stable conflicts for drafting and audit."""
    if not conflicts:
        return "No material cross-claim conflicts were detected."
    return json.dumps(conflicts, ensure_ascii=False, indent=2)


def detect_claim_conflicts(state: OverallState, config: RunnableConfig):
    """Detect and validate cross-dimension claim conflicts before drafting."""
    configurable = Configuration.from_runnable_config(config)
    results, _, _ = _report_research_material(state, configurable)
    claims_by_id = _claim_index(results)
    if len(claims_by_id) < 2:
        emit_research_event(
            "claim_conflicts_detected",
            research_run_id=state["research_run_id"],
            conflict_count=0,
            material_conflict_count=0,
            analysis_complete=True,
        )
        return {"claim_conflicts": [], "consistency_analysis_complete": True}

    compact_claims = [
        {
            "claim_id": claim_id,
            "dimension": claim["dimension_title"],
            "claim": claim["claim"],
            "supporting_source_ids": claim.get("supporting_source_ids", []),
            "uncertainty": claim.get("uncertainty_reason", ""),
        }
        for claim_id, claim in claims_by_id.items()
    ]
    prompt = claim_conflict_instructions.format(
        output_schema=json.dumps(
            ClaimConflictAnalysis.model_json_schema(), ensure_ascii=False
        ),
        research_topic=state["normalized_research_topic"],
        claims=json.dumps(compact_claims, ensure_ascii=False),
    )
    structured_model = create_deepseek_model(
        configurable.reflection_model
    ).with_structured_output(ClaimConflictAnalysis, method="json_mode")
    analysis_complete = True
    try:
        result = structured_model.invoke(prompt)
        if not isinstance(result, ClaimConflictAnalysis):
            raise TypeError("Conflict analysis returned an unexpected type")
    except (LengthFinishReasonError, OutputParserException, TypeError, ValueError):
        try:
            result = structured_model.invoke(
                prompt
                + "\nThe previous response was invalid. Return only the compact JSON "
                "object and omit compatible claim pairs."
            )
            if not isinstance(result, ClaimConflictAnalysis):
                raise TypeError("Conflict analysis retry returned an unexpected type")
        except (LengthFinishReasonError, OutputParserException, TypeError, ValueError):
            result = ClaimConflictAnalysis(conflicts=[])
            analysis_complete = False

    conflicts: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for item in result.conflicts:
        left_id = item.left_claim_id
        right_id = item.right_claim_id
        if (
            left_id not in claims_by_id
            or right_id not in claims_by_id
            or left_id == right_id
            or item.relation == "compatible"
        ):
            continue
        pair = (left_id, right_id) if left_id < right_id else (right_id, left_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        left = claims_by_id[left_id]
        right = claims_by_id[right_id]
        conflicts.append(
            {
                "conflict_id": f"CF-{len(conflicts) + 1}",
                **item.model_dump(),
                "left_source_ids": left.get("supporting_source_ids", []),
                "right_source_ids": right.get("supporting_source_ids", []),
                "material": bool(
                    item.relation == "contradiction"
                    and item.resolution_status == "unresolved"
                    and item.severity in {"medium", "high"}
                ),
            }
        )
    emit_research_event(
        "claim_conflicts_detected",
        research_run_id=state["research_run_id"],
        conflict_count=len(conflicts),
        material_conflict_count=sum(item["material"] for item in conflicts),
        analysis_complete=analysis_complete,
    )
    return {
        "claim_conflicts": conflicts,
        "consistency_analysis_complete": analysis_complete,
    }


def _audited_claim_source_ids(results: list[DimensionResult]) -> set[str]:
    """Collect source IDs explicitly attached to audited claims."""
    return {
        source_id
        for dimension_result in results
        for claim in dimension_result.get("claims", [])
        for source_id in [
            *claim["supporting_source_ids"],
            *claim["contradicting_source_ids"],
        ]
    }


def _report_requires_sectioning(
    results: list[DimensionResult], material: str, configurable: Configuration
) -> bool:
    """Choose sectioned drafting before a single response is likely to overflow."""
    claim_count = sum(len(result.get("claims", [])) for result in results)
    return (
        claim_count >= configurable.report_sectioning_claim_threshold
        or len(material) >= configurable.report_sectioning_material_chars
    )


def _uses_chinese(text: str) -> bool:
    """Return whether user-facing report scaffolding should use Chinese."""
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _deterministic_report_section(result: DimensionResult, research_topic: str) -> str:
    """Build a bounded, citation-safe section when model generation cannot finish."""
    chinese = _uses_chinese(research_topic)
    claims = result.get("claims", [])
    if claims:
        findings = "\n".join(
            "- "
            + claim["claim"]
            + " "
            + " ".join(f"[{source_id}]" for source_id in claim["supporting_source_ids"])
            + (
                f" Uncertainty: {claim['uncertainty_reason']}"
                if claim.get("uncertainty_reason")
                else ""
            )
            for claim in claims
        )
    else:
        findings = (
            "该维度没有通过证据审计的结论。"
            if chinese
            else "No claim passed the evidence audit for this dimension."
        )
    limitations = []
    if result.get("completion_status") != "sufficient":
        limitations.append(
            ("调研状态：" if chinese else "Research status: ")
            + f"{result.get('completion_status', 'incomplete')}."
        )
    limitations.extend(
        str(gap.get("question") or gap.get("reason") or gap)
        for gap in result.get("unresolved_gaps", [])[:3]
    )
    limitation_text = (
        ("\n\n局限性：\n" if chinese else "\n\nLimitations:\n")
        + "\n".join(f"- {item}" for item in limitations)
        if limitations
        else ""
    )
    return findings + limitation_text


def _deterministic_report_overview(
    results: list[DimensionResult], research_topic: str
) -> str:
    """Describe section coverage without adding unsupported factual content."""
    completed = sum(result.get("is_sufficient", False) for result in results)
    if _uses_chinese(research_topic):
        return (
            f"本报告基于经过审计的结论—证据记录，综合了 {len(results)} 个调研维度。"
            f"其中 {completed} 个维度达到配置的完成标准；其余证据限制在对应章节中披露。"
        )
    return (
        f"This report synthesizes {len(results)} research dimensions from audited "
        f"claim–evidence records. {completed} dimension(s) met the configured "
        "completion criteria; remaining limitations are disclosed in their sections."
    )


def _generate_report_section(
    result: DimensionResult,
    research_topic: str,
    model: str,
    research_run_id: str,
    conflicts: list[dict],
) -> str:
    """Generate one bounded section with compact retry and deterministic fallback."""
    dimension = result["dimension"]
    material = format_dimension_results(
        [result],
        max_claims_per_dimension=max(1, len(result.get("claims", []))),
        max_evidence_chars=220,
    )

    def prompt_for(section_material: str) -> str:
        return report_section_instructions.format(
            research_topic=research_topic,
            dimension_title=dimension["title"],
            dimension_scope=dimension["scope"],
            dimension_research=section_material,
        ) + (
            "\n\nCross-claim conflict ledger:\n"
            + _format_conflict_ledger(conflicts)
            + "\nExplicitly disclose every relevant material unresolved conflict, "
            "including its conflict ID."
        )

    emit_research_event(
        "report_section_started",
        research_run_id=research_run_id,
        dimension=dimension,
    )
    try:
        response = create_deepseek_model(model).invoke(prompt_for(material))
        content = str(response.content).strip()
    except LengthFinishReasonError:
        emit_research_event(
            "report_section_retry",
            research_run_id=research_run_id,
            dimension=dimension,
            reason="length_limit",
        )
        compact_material = format_dimension_results(
            [result], max_claims_per_dimension=4, max_evidence_chars=120
        )
        retry_prompt = prompt_for(compact_material) + (
            "\nThe previous section exceeded the output limit. Return a concise "
            "section under 600 words or 1,000 Chinese characters. Do not repeat "
            "evidence and stop after the final paragraph."
        )
        try:
            response = create_deepseek_model(model).invoke(retry_prompt)
            content = str(response.content).strip()
        except LengthFinishReasonError:
            content = _deterministic_report_section(result, research_topic)
            emit_research_event(
                "report_section_fallback",
                research_run_id=research_run_id,
                dimension=dimension,
                reason="length_limit",
            )
    if not content:
        content = _deterministic_report_section(result, research_topic)
        emit_research_event(
            "report_section_fallback",
            research_run_id=research_run_id,
            dimension=dimension,
            reason="empty_response",
        )
    emit_research_event(
        "report_section_completed",
        research_run_id=research_run_id,
        dimension=dimension,
        character_count=len(content),
    )
    return content


def _generate_sectioned_report(
    results: list[DimensionResult],
    research_topic: str,
    model: str,
    research_run_id: str,
    conflicts: list[dict],
) -> tuple[str, str, list[dict]]:
    """Generate bounded semantic sections and merge them without an LLM call."""
    emit_research_event(
        "report_sectioning_started",
        research_run_id=research_run_id,
        dimension_count=len(results),
    )
    sections = [
        {
            "dimension_id": result["dimension"]["id"],
            "title": result["dimension"]["title"],
            "content": _generate_report_section(
                result, research_topic, model, research_run_id, conflicts
            ),
        }
        for result in results
    ]
    overview_material = format_dimension_results(
        results, max_claims_per_dimension=3, max_evidence_chars=100
    )
    overview_prompt = report_overview_instructions.format(
        research_topic=research_topic,
        dimension_research=overview_material,
    ) + (
        "\n\nCross-claim conflict ledger:\n"
        + _format_conflict_ledger(conflicts)
        + "\nDo not silently choose one side of an unresolved material conflict; "
        "name its conflict ID when discussing it."
    )
    try:
        overview = str(
            create_deepseek_model(model).invoke(overview_prompt).content
        ).strip()
    except LengthFinishReasonError:
        overview = _deterministic_report_overview(results, research_topic)
        emit_research_event(
            "report_overview_fallback",
            research_run_id=research_run_id,
            reason="length_limit",
        )
    if not overview:
        overview = _deterministic_report_overview(results, research_topic)
        emit_research_event(
            "report_overview_fallback",
            research_run_id=research_run_id,
            reason="empty_response",
        )
    report = _assemble_sectioned_report(research_topic, overview, sections)
    emit_research_event(
        "report_draft_sectioned",
        research_run_id=research_run_id,
        section_count=len(sections),
        character_count=len(report),
    )
    return report, overview, sections


def _assemble_sectioned_report(
    research_topic: str, overview: str, sections: list[dict]
) -> str:
    """Merge independently generated report parts without another model call."""
    if _uses_chinese(research_topic):
        report_parts = ["# 调研报告", "## 执行摘要\n\n" + overview]
    else:
        report_parts = ["# Research Report", "## Executive Summary\n\n" + overview]
    report_parts.extend(
        f"## {section['title']}\n\n{section['content']}" for section in sections
    )
    return "\n\n".join(report_parts)


def draft_report(state: OverallState, config: RunnableConfig):
    """Draft the report from audited dimension claims."""
    configurable = Configuration.from_runnable_config(config)
    model = state.get("reasoning_model") or configurable.answer_model
    current_results, _, material = _report_research_material(state, configurable)
    conflicts = state.get("claim_conflicts", [])
    emit_research_event("drafting_report", research_run_id=state["research_run_id"])
    if _report_requires_sectioning(current_results, material, configurable):
        report_draft, overview, sections = _generate_sectioned_report(
            current_results,
            state["normalized_research_topic"],
            model,
            state["research_run_id"],
            conflicts,
        )
        return {
            "report_draft": report_draft,
            "report_generation_mode": "sectioned",
            "report_overview": overview,
            "report_sections": sections,
            "report_revision_count": 0,
            "max_report_revisions": configurable.max_report_revisions,
        }
    prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=state["normalized_research_topic"],
        dimension_research=material,
    ) + (
        "\n\nCross-claim conflict ledger:\n"
        + _format_conflict_ledger(conflicts)
        + "\nExplicitly reconcile or disclose every material unresolved conflict "
        "and name its conflict ID."
    )
    try:
        result = create_deepseek_model(model).invoke(prompt)
    except LengthFinishReasonError:
        emit_research_event(
            "report_draft_switching_to_sections",
            research_run_id=state["research_run_id"],
            reason="length_limit",
        )
        report_draft, overview, sections = _generate_sectioned_report(
            current_results,
            state["normalized_research_topic"],
            model,
            state["research_run_id"],
            conflicts,
        )
        return {
            "report_draft": report_draft,
            "report_generation_mode": "sectioned",
            "report_overview": overview,
            "report_sections": sections,
            "report_revision_count": 0,
            "max_report_revisions": configurable.max_report_revisions,
        }
    report_draft = str(result.content).strip()
    if not report_draft:
        emit_research_event(
            "report_draft_switching_to_sections",
            research_run_id=state["research_run_id"],
            reason="empty_response",
        )
        report_draft, overview, sections = _generate_sectioned_report(
            current_results,
            state["normalized_research_topic"],
            model,
            state["research_run_id"],
            conflicts,
        )
        return {
            "report_draft": report_draft,
            "report_generation_mode": "sectioned",
            "report_overview": overview,
            "report_sections": sections,
            "report_revision_count": 0,
            "max_report_revisions": configurable.max_report_revisions,
        }
    return {
        "report_draft": report_draft,
        "report_generation_mode": "single_pass",
        "report_overview": "",
        "report_sections": [],
        "report_revision_count": 0,
        "max_report_revisions": configurable.max_report_revisions,
    }


def audit_report(state: OverallState, config: RunnableConfig):
    """Independently audit report coverage, claims, and citations."""
    configurable = Configuration.from_runnable_config(config)
    current_results, sources, _ = _report_research_material(state, configurable)
    max_claims = max(
        (len(result.get("claims", [])) for result in current_results), default=1
    )
    material = format_dimension_results(
        current_results,
        max_claims_per_dimension=max(1, max_claims),
        max_evidence_chars=180,
    )
    prompt = report_audit_instructions.format(
        research_topic=state["normalized_research_topic"],
        dimension_research=material,
        draft_report=state["report_draft"],
        output_schema=json.dumps(ReportAudit.model_json_schema(), ensure_ascii=False),
    )
    try:
        result = (
            create_deepseek_model(configurable.reflection_model)
            .with_structured_output(ReportAudit, method="json_mode")
            .invoke(prompt)
        )
        if not isinstance(result, ReportAudit):
            raise TypeError("Report audit returned an unexpected type")
        audit = result.model_dump()
    except (
        AttributeError,
        LengthFinishReasonError,
        OutputParserException,
        TypeError,
        ValueError,
    ) as error:
        reason = (
            "length_limit"
            if isinstance(error, LengthFinishReasonError)
            else "structured_output_failure"
        )
        audit = ReportAudit(
            passes=False,
            issues=[
                "The independent model audit could not complete; deterministic citation checks were still applied."
            ],
            revision_instructions=[
                "Preserve only audited claims, valid source markers, and explicit evidence limitations."
            ],
        ).model_dump()
        emit_research_event(
            "report_audit_fallback",
            research_run_id=state["research_run_id"],
            reason=reason,
        )
    valid_ids = {source["source_id"] for source in sources}
    claim_ids = _audited_claim_source_ids(current_results)
    cited_ids = set(re.findall(r"\[(S[A-Za-z0-9-]+)\]", state["report_draft"]))
    invalid_ids = sorted(cited_ids - valid_ids)
    unclaimed_ids = sorted((cited_ids & valid_ids) - claim_ids)
    if invalid_ids:
        audit["passes"] = False
        audit["issues"] = [
            *audit["issues"],
            f"Unknown source markers: {', '.join(invalid_ids)}",
        ]
        audit["revision_instructions"] = [
            *audit["revision_instructions"],
            "Remove every unknown source marker or replace it with valid evidence.",
        ]
    if unclaimed_ids:
        audit["passes"] = False
        audit["issues"] = [
            *audit["issues"],
            f"Source markers are not attached to audited claims: {', '.join(unclaimed_ids)}",
        ]
        audit["revision_instructions"] = [
            *audit["revision_instructions"],
            "Remove citations that are not attached to the audited claim set.",
        ]

    conflicts = state.get("claim_conflicts", [])
    conflict_prompt = report_consistency_audit_instructions.format(
        output_schema=json.dumps(
            ReportConsistencyAudit.model_json_schema(), ensure_ascii=False
        ),
        research_topic=state["normalized_research_topic"],
        dimension_research=material,
        conflict_ledger=_format_conflict_ledger(conflicts),
        draft_report=state["report_draft"],
    )
    try:
        consistency_result = (
            create_deepseek_model(configurable.reflection_model)
            .with_structured_output(ReportConsistencyAudit, method="json_mode")
            .invoke(conflict_prompt)
        )
        if not isinstance(consistency_result, ReportConsistencyAudit):
            raise TypeError("Consistency audit returned an unexpected type")
        consistency_audit = consistency_result.model_dump()
    except (
        AttributeError,
        LengthFinishReasonError,
        OutputParserException,
        TypeError,
        ValueError,
    ):
        consistency_audit = ReportConsistencyAudit(
            passes=False,
            issues=["The independent consistency audit could not complete."],
            revision_instructions=[
                "Preserve both sides of every material conflict and state the uncertainty explicitly."
            ],
        ).model_dump()

    valid_conflict_ids = {item["conflict_id"] for item in conflicts}
    material_conflict_ids = {
        item["conflict_id"] for item in conflicts if item.get("material")
    }
    consistency_audit["covered_conflict_ids"] = sorted(
        set(consistency_audit.get("covered_conflict_ids", [])) & valid_conflict_ids
    )
    reported_omitted = set(consistency_audit.get("omitted_conflict_ids", []))
    omitted_conflict_ids = (reported_omitted & valid_conflict_ids) | (
        material_conflict_ids - set(consistency_audit["covered_conflict_ids"])
    )
    claims_by_id = _claim_index(current_results)
    for conflict in conflicts:
        if not conflict.get("material"):
            continue
        left = claims_by_id.get(conflict["left_claim_id"], {})
        right = claims_by_id.get(conflict["right_claim_id"], {})
        left_cited = bool(set(left.get("supporting_source_ids", [])) & cited_ids)
        right_cited = bool(set(right.get("supporting_source_ids", [])) & cited_ids)
        explicitly_identified = conflict["conflict_id"] in state["report_draft"]
        if not (left_cited and right_cited and explicitly_identified):
            omitted_conflict_ids.add(conflict["conflict_id"])
    consistency_audit["omitted_conflict_ids"] = sorted(omitted_conflict_ids)
    if omitted_conflict_ids:
        consistency_audit["passes"] = False
        consistency_audit["issues"] = [
            *consistency_audit.get("issues", []),
            "Material conflicts are not fully represented: "
            + ", ".join(sorted(omitted_conflict_ids)),
        ]
        consistency_audit["revision_instructions"] = [
            *consistency_audit.get("revision_instructions", []),
            "Present both accepted-evidence sides of every omitted material conflict and explain why they differ.",
        ]
    if not state.get("consistency_analysis_complete", True):
        consistency_audit["passes"] = False
        consistency_audit["issues"] = [
            *consistency_audit.get("issues", []),
            "The pre-draft cross-claim consistency analysis was incomplete.",
        ]
        consistency_audit["revision_instructions"] = [
            *consistency_audit.get("revision_instructions", []),
            "Use conservative language and avoid a unique conclusion where accepted claims may conflict.",
        ]
    if not consistency_audit.get("passes"):
        audit["passes"] = False
        audit["issues"] = [
            *audit["issues"],
            *[f"Consistency: {issue}" for issue in consistency_audit["issues"]],
        ]
        audit["revision_instructions"] = [
            *audit["revision_instructions"],
            *consistency_audit["revision_instructions"],
        ]
    emit_research_event(
        "report_audit_completed",
        research_run_id=state["research_run_id"],
        passes=audit["passes"],
        revision_count=state.get("report_revision_count", 0),
        consistency_passes=consistency_audit["passes"],
        omitted_conflict_count=len(consistency_audit["omitted_conflict_ids"]),
    )
    emit_research_event(
        "report_consistency_audited",
        research_run_id=state["research_run_id"],
        passes=consistency_audit["passes"],
        covered_conflict_count=len(consistency_audit["covered_conflict_ids"]),
        omitted_conflict_count=len(consistency_audit["omitted_conflict_ids"]),
        new_contradiction_count=len(consistency_audit["new_contradictions"]),
    )
    return {
        "report_audit": audit,
        "report_consistency_audit": consistency_audit,
    }


def route_report_audit(state: OverallState):
    """Revise material failures or switch to a citation-safe final report."""
    if state["report_audit"].get("passes"):
        return "finalize_answer"
    if state.get("report_revision_count", 0) >= state.get("max_report_revisions", 2):
        return "build_safe_report"
    return "revise_report"


def build_safe_report(state: OverallState, config: RunnableConfig):
    """Deterministically publish only validated claims after audit exhaustion."""
    configurable = Configuration.from_runnable_config(config)
    results, _, _ = _report_research_material(state, configurable)
    topic = state["normalized_research_topic"]
    overview = _deterministic_report_overview(results, topic)
    sections = [
        {
            "dimension_id": result["dimension"]["id"],
            "title": result["dimension"]["title"],
            "content": _deterministic_report_section(result, topic),
        }
        for result in results
    ]
    conflicts = [
        item for item in state.get("claim_conflicts", []) if item.get("material")
    ]
    if conflicts:
        claims_by_id = _claim_index(results)
        chinese = _uses_chinese(topic)
        conflict_lines = []
        for conflict in conflicts:
            left = claims_by_id.get(conflict["left_claim_id"], {})
            right = claims_by_id.get(conflict["right_claim_id"], {})
            left_markers = " ".join(
                f"[{source_id}]" for source_id in left.get("supporting_source_ids", [])
            )
            right_markers = " ".join(
                f"[{source_id}]" for source_id in right.get("supporting_source_ids", [])
            )
            conflict_lines.append(
                f"- {conflict['conflict_id']}: {left.get('claim', '')} {left_markers} / "
                f"{right.get('claim', '')} {right_markers}. "
                + (
                    "现有合格证据存在冲突，无法安全地选择单一结论。"
                    if chinese
                    else "Accepted evidence conflicts; no single conclusion can be selected safely."
                )
            )
        sections.append(
            {
                "dimension_id": "conflicts",
                "title": "未解决的证据矛盾"
                if chinese
                else "Unresolved Evidence Conflicts",
                "content": "\n".join(conflict_lines),
            }
        )
    report = _assemble_sectioned_report(topic, overview, sections)
    analysis_complete = state.get("consistency_analysis_complete", True)
    consistency_audit = ReportConsistencyAudit(
        passes=analysis_complete,
        covered_conflict_ids=[
            conflict["conflict_id"] for conflict in conflicts if analysis_complete
        ],
        omitted_conflict_ids=[],
        issues=[]
        if analysis_complete
        else ["The pre-draft cross-claim consistency analysis was incomplete."],
        revision_instructions=[]
        if analysis_complete
        else [
            "Treat the fallback as an evidence inventory, not a reconciled synthesis."
        ],
    ).model_dump()
    emit_research_event(
        "safe_report_built",
        research_run_id=state["research_run_id"],
        section_count=len(sections),
        material_conflict_count=len(conflicts),
    )
    return {
        "report_draft": report,
        "report_generation_mode": "safe_fallback",
        "report_overview": overview,
        "report_sections": sections,
        "report_safe_fallback_used": True,
        "report_consistency_audit": consistency_audit,
    }


def _bounded_report_part_revision(
    *,
    prompt: str,
    fallback: str,
    model: str,
    research_run_id: str,
    event_prefix: str,
    event_data: dict | None = None,
) -> str:
    """Revise one bounded report part without risking the entire report."""
    event_data = event_data or {}
    try:
        content = str(create_deepseek_model(model).invoke(prompt).content).strip()
    except LengthFinishReasonError:
        emit_research_event(
            f"{event_prefix}_retry",
            research_run_id=research_run_id,
            reason="length_limit",
            **event_data,
        )
        retry_prompt = prompt + (
            "\nThe previous revision exceeded the output limit. Preserve the essential "
            "supported findings, remove repetition, and return only the complete "
            "revised part in at most half the requested length."
        )
        try:
            content = str(
                create_deepseek_model(model).invoke(retry_prompt).content
            ).strip()
        except LengthFinishReasonError:
            content = ""
    if content:
        return content
    emit_research_event(
        f"{event_prefix}_skipped",
        research_run_id=research_run_id,
        reason="length_limit_or_empty_response",
        **event_data,
    )
    return fallback


def _revise_sectioned_report(
    state: OverallState, current_results: list[DimensionResult], model: str
) -> dict:
    """Revise a long report by bounded parts and merge it deterministically."""
    research_topic = state["normalized_research_topic"]
    research_run_id = state["research_run_id"]
    audit_findings = json.dumps(state["report_audit"], ensure_ascii=False)
    conflict_context = _format_conflict_ledger(state.get("claim_conflicts", []))
    results_by_id = {result["dimension"]["id"]: result for result in current_results}
    revised_sections = []
    for section in state.get("report_sections", []):
        result = results_by_id.get(str(section.get("dimension_id", "")))
        if result is None:
            revised_sections.append(section)
            continue
        dimension = result["dimension"]
        material = format_dimension_results(
            [result],
            max_claims_per_dimension=max(1, len(result.get("claims", []))),
            max_evidence_chars=160,
        )
        prompt = report_section_revision_instructions.format(
            research_topic=research_topic,
            dimension_title=dimension["title"],
            dimension_scope=dimension["scope"],
            dimension_research=material,
            current_section=section.get("content", ""),
            audit_findings=audit_findings,
        ) + (
            "\n\nCross-claim conflict ledger:\n"
            + conflict_context
            + "\nPreserve both accepted-evidence sides and the conflict ID of each "
            "unresolved material conflict."
        )
        revised_sections.append(
            {
                **section,
                "content": _bounded_report_part_revision(
                    prompt=prompt,
                    fallback=str(section.get("content", "")),
                    model=model,
                    research_run_id=research_run_id,
                    event_prefix="report_section_revision",
                    event_data={"dimension": dimension},
                ),
            }
        )
    overview_material = format_dimension_results(
        current_results, max_claims_per_dimension=3, max_evidence_chars=100
    )
    current_overview = state.get("report_overview", "")
    overview_prompt = report_overview_revision_instructions.format(
        research_topic=research_topic,
        dimension_research=overview_material,
        current_overview=current_overview,
        audit_findings=audit_findings,
    ) + (
        "\n\nCross-claim conflict ledger:\n"
        + conflict_context
        + "\nDo not silently select one side of an unresolved material conflict; "
        "name its conflict ID when discussing it."
    )
    revised_overview = _bounded_report_part_revision(
        prompt=overview_prompt,
        fallback=current_overview
        or _deterministic_report_overview(current_results, research_topic),
        model=model,
        research_run_id=research_run_id,
        event_prefix="report_overview_revision",
    )
    revised_report = _assemble_sectioned_report(
        research_topic, revised_overview, revised_sections
    )
    emit_research_event(
        "report_sections_revised",
        research_run_id=research_run_id,
        section_count=len(revised_sections),
        character_count=len(revised_report),
    )
    return {
        "report_draft": revised_report,
        "report_overview": revised_overview,
        "report_sections": revised_sections,
        "report_generation_mode": "sectioned",
        "report_revision_count": state.get("report_revision_count", 0) + 1,
    }


def revise_report(state: OverallState, config: RunnableConfig):
    """Revise only the issues identified by the independent audit."""
    configurable = Configuration.from_runnable_config(config)
    model = state.get("reasoning_model") or configurable.answer_model
    current_results, _, _ = _report_research_material(state, configurable)
    if state.get("report_generation_mode") == "sectioned" and state.get(
        "report_sections"
    ):
        return _revise_sectioned_report(state, current_results, model)
    material = format_dimension_results(
        current_results, max_claims_per_dimension=5, max_evidence_chars=120
    )
    prompt = report_revision_instructions.format(
        research_topic=state["normalized_research_topic"],
        dimension_research=material,
        draft_report=state["report_draft"],
        audit_findings=json.dumps(state["report_audit"], ensure_ascii=False),
    ) + (
        "\n\nCross-claim conflict ledger:\n"
        + _format_conflict_ledger(state.get("claim_conflicts", []))
        + "\nResolve or explicitly disclose every material conflict and name its "
        "conflict ID."
    )
    try:
        result = create_deepseek_model(model).invoke(prompt)
    except LengthFinishReasonError:
        emit_research_event(
            "report_revision_retry",
            research_run_id=state["research_run_id"],
            reason="length_limit",
        )
        retry_prompt = (
            prompt
            + "\nThe previous revision exceeded the output limit. Return the complete "
            "revised report in under 800 words or 1,600 Chinese characters. Use short "
            "sections, do not repeat evidence, and stop after the conclusion."
        )
        try:
            result = create_deepseek_model(model).invoke(retry_prompt)
        except LengthFinishReasonError:
            emit_research_event(
                "report_revision_skipped",
                research_run_id=state["research_run_id"],
                reason="length_limit",
            )
            return {
                "report_draft": state["report_draft"],
                "report_revision_count": state.get("report_revision_count", 0) + 1,
            }
    return {
        "report_draft": str(result.content),
        "report_revision_count": state.get("report_revision_count", 0) + 1,
    }


def finalize_answer(state: OverallState):
    """Render validated source markers from the audited report."""
    current_results, sources, _ = _report_research_material(state)
    claim_ids = _audited_claim_source_ids(current_results)
    sources = [source for source in sources if source["source_id"] in claim_ids]
    emit_research_event("finalizing_answer", research_run_id=state["research_run_id"])
    answer, _ = render_source_citations(state["report_draft"], sources)
    return {"messages": [AIMessage(content=answer)]}


builder = StateGraph(OverallState, config_schema=Configuration)
builder.add_node("initialize_research_topic", initialize_research_topic)
builder.add_node("analyze_research_topic", analyze_research_topic)
builder.add_node("request_topic_clarification", request_topic_clarification)
builder.add_node("generate_research_dimensions", generate_research_dimensions)
builder.add_node("review_research_dimensions", review_research_dimensions)
builder.add_node("research_dimension", research_dimension)
builder.add_node("prepare_report_evidence", prepare_report_evidence)
builder.add_node("detect_claim_conflicts", detect_claim_conflicts)
builder.add_node("draft_report", draft_report)
builder.add_node("audit_report", audit_report)
builder.add_node("revise_report", revise_report)
builder.add_node("build_safe_report", build_safe_report)
builder.add_node("finalize_answer", finalize_answer)
builder.add_edge(START, "initialize_research_topic")
builder.add_edge("initialize_research_topic", "analyze_research_topic")
builder.add_conditional_edges(
    "analyze_research_topic",
    route_topic_analysis,
    ["request_topic_clarification", "generate_research_dimensions"],
)
builder.add_conditional_edges(
    "request_topic_clarification",
    route_topic_clarification,
    ["analyze_research_topic", "generate_research_dimensions"],
)
builder.add_edge("generate_research_dimensions", "review_research_dimensions")
builder.add_conditional_edges(
    "review_research_dimensions",
    route_dimension_review,
    ["generate_research_dimensions", "research_dimension"],
)
builder.add_edge("research_dimension", "prepare_report_evidence")
builder.add_edge("prepare_report_evidence", "detect_claim_conflicts")
builder.add_edge("detect_claim_conflicts", "draft_report")
builder.add_edge("draft_report", "audit_report")
builder.add_conditional_edges(
    "audit_report",
    route_report_audit,
    ["revise_report", "build_safe_report", "finalize_answer"],
)
builder.add_edge("revise_report", "audit_report")
builder.add_edge("build_safe_report", "finalize_answer")
builder.add_edge("finalize_answer", END)
graph = builder.compile(name="deepseek-tavily-multidimensional-research-agent")
