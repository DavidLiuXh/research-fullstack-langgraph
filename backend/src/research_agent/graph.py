"""LangGraph workflow for clarification, dimension research, and reporting."""

# ruff: noqa: E402

import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
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
    claim_extraction_instructions,
    dimension_instructions,
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
    report_audit_instructions,
    report_revision_instructions,
    source_evaluation_instructions,
    topic_clarification_instructions,
)
from research_agent.state import (
    DimensionInput,
    DimensionResult,
    DimensionState,
    OverallState,
    QueryGenerationState,
    WebSearchState,
)
from research_agent.tools_and_schemas import (
    ClaimExtraction,
    Reflection,
    ReportAudit,
    ResearchDimensionList,
    ResearchGap,
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


def _prioritized_gaps(state: DimensionState) -> list[dict]:
    """Return unresolved gaps in impact order, or a scoped initial evidence need."""
    gaps = list(state.get("reflection_assessment", {}).get("missing_questions", []))
    if not gaps:
        gaps = [
            {
                "gap_id": "initial-dimension-scope",
                "question": state["dimension"]["scope"],
                "reason": "This is the initial evidence pass for the dimension.",
                "priority": "high",
                "required_source_types": _default_source_types(
                    state["research_topic"], state["dimension"]
                ),
                "expected_evidence": state["dimension"]["scope"],
                "suggested_query_focus": state["dimension"]["scope"],
            }
        ]
    priority = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(gaps, key=lambda gap: priority.get(gap.get("priority"), 3))
    high_priority = [gap for gap in ordered if gap.get("priority") == "high"]
    other = [gap for gap in ordered if gap.get("priority") != "high"]
    if len(high_priority) > 1:
        offset = max(state.get("research_loop_count", 0) - 1, 0) % len(high_priority)
        high_priority = high_priority[offset:] + high_priority[:offset]
    return [*high_priority, *other]


def _source_rank_key(source: Mapping[str, Any]) -> tuple:
    """Rank sources by requested fit, acceptance, primacy, authority, and score."""
    return (
        bool(source.get("matches_requested_source_type")),
        source.get("quality_status") == "accepted",
        bool(source.get("is_primary_source")),
        bool(source.get("is_authoritative_source")),
        float(source.get("evidence_score", 0)),
    )


def _gap_has_required_source(
    gap: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> bool:
    """Check that a gap has accepted evidence of one requested source type."""
    gap_id = gap.get("gap_id", "")
    if gap_id == "quality-accepted-sources":
        return any(source.get("quality_status") == "accepted" for source in sources)
    if gap_id == "quality-authoritative-source":
        return any(
            source.get("quality_status") == "accepted"
            and source.get("is_authoritative_source")
            for source in sources
        )
    if gap_id == "quality-primary-source":
        return any(
            source.get("quality_status") == "accepted"
            and source.get("is_primary_source")
            for source in sources
        )
    required_types = set(gap.get("required_source_types", []))
    return any(
        source.get("quality_status") == "accepted"
        and gap_id in source.get("gap_ids", [])
        and (not required_types or source.get("source_type", "") in required_types)
        for source in sources
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


def generate_query(
    state: DimensionState, config: RunnableConfig
) -> QueryGenerationState:
    """Generate searches for a dimension and its latest knowledge gap."""
    configurable = Configuration.from_runnable_config(config)
    query_count = (
        state.get("initial_search_query_count")
        or configurable.number_of_initial_queries
    )
    assessment = state.get("reflection_assessment", {})
    gaps = _prioritized_gaps(state)
    required_source_types = sorted(
        {
            source_type
            for gap in gaps
            for source_type in gap.get("required_source_types", [])
        }
    )
    strategies = assessment.get("recommended_search_strategy", [])
    do_not_repeat = assessment.get("do_not_repeat", [])
    query_history = state.get("query_history", [])
    prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=state["research_topic"],
        dimension_title=state["dimension"]["title"],
        dimension_scope=state["dimension"]["scope"],
        knowledge_gap=state.get("current_knowledge_gap")
        or "None; this is the first pass.",
        gap_records=json.dumps(gaps, ensure_ascii=False),
        required_source_types=required_source_types or "No special requirement.",
        recommended_search_strategy=strategies or "No special strategy.",
        query_history=[*query_history, *do_not_repeat]
        or "None; this is the first pass.",
        number_queries=query_count,
    )
    llm = create_deepseek_model(configurable.query_generator_model)
    result = llm.with_structured_output(SearchQueryList, method="json_mode").invoke(
        prompt
    )
    normalized_history = {query.casefold().strip() for query in query_history}
    queries = []
    for query in result.query:
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
        queries = [query.strip() for query in result.query if query.strip()][:1]
    if not queries:
        raise ValueError("DeepSeek did not generate any usable search queries")
    search_tasks = []
    for index, query in enumerate(queries):
        gap = gaps[index % len(gaps)]
        search_tasks.append(
            {
                "query": query,
                "gap_id": gap.get("gap_id", "initial-dimension-scope"),
                "requested_source_types": gap.get("required_source_types", [])
                or _default_source_types(state["research_topic"], state["dimension"]),
                "expected_evidence": gap.get("expected_evidence")
                or gap.get("reason", ""),
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


def reflection(state: DimensionState, config: RunnableConfig):
    """Produce a structured evidence audit and deterministic completion status."""
    configurable = Configuration.from_runnable_config(config)
    loop_count = state.get("research_loop_count", 0) + 1
    selected_sources = state.get("selected_sources", [])
    accepted_count = sum(
        source.get("quality_status") == "accepted" for source in selected_sources
    )
    authoritative_count = sum(
        source.get("quality_status") == "accepted"
        and bool(source.get("is_authoritative_source"))
        for source in selected_sources
    )
    primary_count = sum(
        source.get("quality_status") == "accepted"
        and bool(source.get("is_primary_source"))
        for source in selected_sources
    )
    current_source_ids = sorted(source["source_id"] for source in selected_sources)
    source_id_history = state.get("evidence_source_id_history", [])
    previous_source_ids = set(source_id_history[-1]) if source_id_history else set()
    new_source_ids = set(current_source_ids) - previous_source_ids
    minimum_source_requirements = {
        "accepted": configurable.min_accepted_sources_per_dimension,
        "authoritative": configurable.min_authoritative_sources_per_dimension,
        "primary": configurable.min_primary_sources_per_dimension,
        "current": {
            "accepted": accepted_count,
            "authoritative": authoritative_count,
            "primary": primary_count,
        },
    }
    prompt = reflection_instructions.format(
        research_topic=state["research_topic"],
        dimension_title=state["dimension"]["title"],
        dimension_scope=state["dimension"]["scope"],
        summaries=format_sources_for_research(state.get("selected_sources", [])),
        rejected_source_summary=format_rejected_source_summary(
            state.get("rejected_sources", [])
        ),
        previous_reflection=json.dumps(
            state.get("reflection_history", [])[-1:] or [], ensure_ascii=False
        ),
        query_history=state.get("query_history", []) or "None.",
        evidence_gain_history=json.dumps(
            [
                *state.get("evidence_gain_history", []),
                {"new_source_count": len(new_source_ids)},
            ][-3:],
            ensure_ascii=False,
        ),
        minimum_source_requirements=json.dumps(
            minimum_source_requirements, ensure_ascii=False
        ),
    )
    llm = create_deepseek_model(configurable.reflection_model)
    try:
        result = llm.with_structured_output(Reflection, method="json_mode").invoke(
            prompt
        )
    except OutputParserException as error:
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
                {
                    "question": f"What evidence is still required for {state['dimension']['title']}?",
                    "reason": "The structured reflection could not be parsed safely.",
                    "priority": "high",
                    "required_source_types": [],
                    "expected_evidence": state["dimension"]["scope"],
                    "suggested_query_focus": state["dimension"]["scope"],
                }
            ],
            unsupported_claims=[],
            contradictions=[],
            source_quality_issues=[
                "Structured reflection failed; conservative follow-up research was requested."
            ],
            recommended_search_strategy=[
                "Generate a focused query for the unresolved dimension scope."
            ],
            do_not_repeat=state.get("query_history", []),
            completion_reason="Reflection parsing failed, so evidence cannot be declared sufficient.",
            confidence=0.0,
        )
    missing_questions = list(result.missing_questions)
    if not result.is_sufficient and not missing_questions:
        missing_questions.append(
            ResearchGap(
                gap_id="reflection-unresolved-evidence",
                question=f"What material evidence is still missing for {state['dimension']['title']}?",
                reason=result.completion_reason
                or "The reflection did not declare the dimension sufficient.",
                priority="high",
                required_source_types=_default_source_types(
                    state["research_topic"], state["dimension"]
                ),
                expected_evidence="Direct evidence that resolves the remaining uncertainty or conflict.",
                suggested_query_focus=(
                    result.recommended_search_strategy[0]
                    if result.recommended_search_strategy
                    else state["dimension"]["scope"]
                ),
            )
        )
    quality_gaps = []
    if accepted_count < configurable.min_accepted_sources_per_dimension:
        quality_gaps.append(
            ResearchGap(
                gap_id="quality-accepted-sources",
                question="Which additional relevant sources can independently support this dimension?",
                reason="The minimum accepted-source requirement has not been met.",
                priority="high",
                required_source_types=sorted(AUTHORITATIVE_SOURCE_TYPES),
                expected_evidence="Independent relevant evidence from an accepted source.",
                suggested_query_focus="Find additional authoritative evidence for the dimension.",
            )
        )
    if authoritative_count < configurable.min_authoritative_sources_per_dimension:
        quality_gaps.append(
            ResearchGap(
                gap_id="quality-authoritative-source",
                question="What authoritative source directly addresses this dimension?",
                reason="No sufficient authoritative evidence has been selected.",
                priority="high",
                required_source_types=sorted(AUTHORITATIVE_SOURCE_TYPES),
                expected_evidence="A direct statement or data point from an authoritative source.",
                suggested_query_focus="Search official, academic, standards, or institutional sources.",
            )
        )
    if primary_count < configurable.min_primary_sources_per_dimension:
        quality_gaps.append(
            ResearchGap(
                gap_id="quality-primary-source",
                question="What primary source directly supports this dimension?",
                reason="The minimum primary-source requirement has not been met.",
                priority="high",
                required_source_types=[
                    "government",
                    "official_company",
                    "standards_body",
                    "academic",
                ],
                expected_evidence="First-party data, documentation, regulation, or research text.",
                suggested_query_focus="Find the original official document, dataset, or publication.",
            )
        )
    prior_history = state.get("reflection_history", [])
    prior_gap_values = [
        gap
        for item in prior_history[-1:]
        for gap in item.get("missing_questions", [])
        if gap.get("gap_id")
    ]
    prior_gap_by_id = {gap["gap_id"]: gap for gap in prior_gap_values}
    requested_resolved_ids = set(result.resolved_gap_ids) & set(prior_gap_by_id)
    validated_model_resolved_ids = {
        gap_id
        for gap_id in requested_resolved_ids
        if _gap_has_required_source(prior_gap_by_id[gap_id], selected_sources)
    }
    rejected_resolved_ids = requested_resolved_ids - validated_model_resolved_ids
    requirement_resolved_ids = set()
    if accepted_count >= configurable.min_accepted_sources_per_dimension:
        requirement_resolved_ids.add("quality-accepted-sources")
    if authoritative_count >= configurable.min_authoritative_sources_per_dimension:
        requirement_resolved_ids.add("quality-authoritative-source")
    if primary_count >= configurable.min_primary_sources_per_dimension:
        requirement_resolved_ids.add("quality-primary-source")
    gaps_by_id = {gap.gap_id: gap for gap in missing_questions}
    gaps_by_id.update({gap.gap_id: gap for gap in quality_gaps})
    for prior_gap in prior_gap_values:
        gap_id = prior_gap["gap_id"]
        if gap_id in validated_model_resolved_ids or gap_id in requirement_resolved_ids:
            continue
        gaps_by_id.setdefault(gap_id, ResearchGap.model_validate(prior_gap))
    missing_questions = list(gaps_by_id.values())
    assessment = result.model_dump()
    assessment["missing_questions"] = [gap.model_dump() for gap in missing_questions]
    if any(gap.priority == "high" for gap in missing_questions):
        assessment["is_sufficient"] = False
    if quality_gaps:
        assessment["source_quality_issues"] = list(
            dict.fromkeys(
                [
                    *assessment.get("source_quality_issues", []),
                    "Deterministic minimum source requirements are not met.",
                ]
            )
        )
    if rejected_resolved_ids:
        assessment["source_quality_issues"] = list(
            dict.fromkeys(
                [
                    *assessment.get("source_quality_issues", []),
                    "Resolved gaps were retained because no accepted source of the requested type was associated with them: "
                    + ", ".join(sorted(rejected_resolved_ids)),
                ]
            )
        )
    prior_missing_ids = {
        gap.get("gap_id", "")
        for item in prior_history[-1:]
        for gap in item.get("missing_questions", [])
        if gap.get("gap_id")
    }
    current_gap_ids = {gap.gap_id for gap in missing_questions}
    declared_resolved_ids = validated_model_resolved_ids & (
        set(state.get("gap_registry", {})) | prior_missing_ids
    )
    deterministically_resolved_ids = requirement_resolved_ids & (
        set(state.get("gap_registry", {})) | prior_missing_ids
    )
    newly_resolved_ids = (
        declared_resolved_ids | deterministically_resolved_ids
    ) - current_gap_ids
    # A model may reopen a previously resolved gap when later evidence exposes a
    # new uncertainty. Current missing gaps are therefore authoritative over the
    # historical resolved set.
    resolved_gap_ids = (
        set(state.get("resolved_gap_ids", [])) | newly_resolved_ids
    ) - current_gap_ids
    gap_registry = dict(state.get("gap_registry", {}))
    for item in missing_questions:
        gap_registry[item.gap_id] = item.model_dump()
    gap_source_coverage_ids = {
        gap_id
        for gap_id, gap in gap_registry.items()
        if _gap_has_required_source(gap, selected_sources)
    }
    evidence_gain = {
        "loop": loop_count,
        "new_source_count": len(new_source_ids),
        "new_accepted_source_count": sum(
            source["source_id"] in new_source_ids
            and source.get("quality_status") == "accepted"
            for source in selected_sources
        ),
        "new_primary_source_count": sum(
            source["source_id"] in new_source_ids
            and bool(source.get("is_primary_source"))
            for source in selected_sources
        ),
        "resolved_gap_count": len(newly_resolved_ids),
    }
    evidence_gain["total_gain"] = (
        evidence_gain["new_source_count"] + evidence_gain["resolved_gap_count"]
    )
    knowledge_gap = "\n".join(
        f"[{gap.gap_id}] {gap.question}: {gap.reason}. "
        f"Expected evidence: {gap.expected_evidence}. "
        f"Search focus: {gap.suggested_query_focus}"
        for gap in missing_questions
    )
    emit_research_event(
        "reflection_completed",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        is_sufficient=assessment["is_sufficient"],
        knowledge_gap=knowledge_gap,
        missing_questions=[gap.model_dump() for gap in missing_questions],
        contradictions=[item.model_dump() for item in result.contradictions],
        confidence=result.confidence,
        loop=loop_count,
        evidence_gain=evidence_gain,
    )
    gain_history = state.get("evidence_gain_history", [])
    stalled = bool(
        gain_history
        and gain_history[-1].get("total_gain", 0) == 0
        and evidence_gain["total_gain"] == 0
    )
    high_priority_gap = any(gap.priority == "high" for gap in missing_questions) or any(
        gap_id not in resolved_gap_ids and gap.get("priority") == "high"
        for gap_id, gap in gap_registry.items()
    )
    unresolved_conflict = any(item.requires_follow_up for item in result.contradictions)
    search_unavailable = bool(
        not selected_sources
        and state.get("search_failures")
        and state.get("search_success_count", 0) == 0
    )
    if search_unavailable:
        completion_status = "search_unavailable"
        is_sufficient = False
    elif (
        assessment["is_sufficient"]
        and not high_priority_gap
        and not unresolved_conflict
    ):
        completion_status = "sufficient"
        is_sufficient = True
    elif stalled:
        completion_status = "stalled"
        is_sufficient = False
    elif loop_count >= state["max_research_loops"]:
        completion_status = "budget_exhausted"
        is_sufficient = False
    else:
        completion_status = "researching"
        is_sufficient = False
    return {
        "is_sufficient": is_sufficient,
        "current_knowledge_gap": knowledge_gap,
        "research_loop_count": loop_count,
        "reflection_assessment": assessment,
        "reflection_history": [assessment],
        "evidence_source_count_history": [len(state.get("selected_sources", []))],
        "evidence_source_id_history": [current_source_ids],
        "evidence_gain_history": [evidence_gain],
        "gap_registry": gap_registry,
        "resolved_gap_ids": sorted(resolved_gap_ids),
        "gap_source_coverage_ids": sorted(gap_source_coverage_ids),
        "completion_status": completion_status,
    }


def route_dimension_research(state: DimensionState):
    """Return gaps to query generation, or extract claims when research ends."""
    if state.get("completion_status") != "researching":
        return "extract_claims"
    return "generate_query"


def extract_claims(state: DimensionState, config: RunnableConfig):
    """Convert selected evidence into an auditable claim set."""
    configurable = Configuration.from_runnable_config(config)
    selected = state.get("selected_sources", [])
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
            validated_claims.append(
                {
                    "claim": claim.claim,
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
    return {"claims": claims, "dimension_summary": dimension_summary}


dimension_builder = StateGraph(DimensionState, input_schema=DimensionInput)
dimension_builder.add_node("generate_query", generate_query)
dimension_builder.add_node("web_research", web_research)
dimension_builder.add_node("evaluate_sources", evaluate_sources)
dimension_builder.add_node("reflection", reflection)
dimension_builder.add_node("extract_claims", extract_claims)
dimension_builder.add_edge(START, "generate_query")
dimension_builder.add_conditional_edges(
    "generate_query", dispatch_search_queries, ["web_research"]
)
dimension_builder.add_edge("web_research", "evaluate_sources")
dimension_builder.add_edge("evaluate_sources", "reflection")
dimension_builder.add_conditional_edges(
    "reflection", route_dimension_research, ["generate_query", "extract_claims"]
)
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
    dimension_result = {
        "research_run_id": result["research_run_id"],
        "dimension": result["dimension"],
        "research_content": result.get("dimension_summary", ""),
        "sources": result.get("selected_sources", []),
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
        "known_gap_count": len(result.get("gap_registry", {})),
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
        "sources_gathered": result.get("selected_sources", []),
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


def _partial_length_limited_content(error: LengthFinishReasonError) -> str:
    """Recover usable text returned before a provider output limit was reached."""
    completion = getattr(error, "completion", None)
    choices = getattr(completion, "choices", ())
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    raise error


def draft_report(state: OverallState, config: RunnableConfig):
    """Draft the report from audited dimension claims."""
    configurable = Configuration.from_runnable_config(config)
    model = state.get("reasoning_model") or configurable.answer_model
    _, _, material = _current_research_material(state)
    emit_research_event("drafting_report", research_run_id=state["research_run_id"])
    prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=state["normalized_research_topic"],
        dimension_research=material,
    )
    try:
        result = create_deepseek_model(model).invoke(prompt)
    except LengthFinishReasonError:
        current_results, _, _ = _current_research_material(state)
        compact_material = format_dimension_results(
            current_results, max_claims_per_dimension=5, max_evidence_chars=120
        )
        emit_research_event(
            "report_draft_retry",
            research_run_id=state["research_run_id"],
            reason="length_limit",
        )
        compact_prompt = (
            answer_instructions.format(
                current_date=get_current_date(),
                research_topic=state["normalized_research_topic"],
                dimension_research=compact_material,
            )
            + "\nThe first draft exceeded the output limit. Write a concise executive "
            "report under 800 words or 1,600 Chinese characters. Use short sections, "
            "do not repeat evidence, and stop immediately after the conclusion."
        )
        try:
            result = create_deepseek_model(model).invoke(compact_prompt)
            report_draft = str(result.content)
        except LengthFinishReasonError as retry_error:
            report_draft = _partial_length_limited_content(retry_error)
            emit_research_event(
                "report_draft_partial_recovered",
                research_run_id=state["research_run_id"],
                reason="length_limit",
            )
        return {
            "report_draft": report_draft,
            "report_revision_count": 0,
            "max_report_revisions": configurable.max_report_revisions,
        }
    return {
        "report_draft": str(result.content),
        "report_revision_count": 0,
        "max_report_revisions": configurable.max_report_revisions,
    }


def audit_report(state: OverallState, config: RunnableConfig):
    """Independently audit report coverage, claims, and citations."""
    configurable = Configuration.from_runnable_config(config)
    _, _, material = _current_research_material(state)
    prompt = report_audit_instructions.format(
        research_topic=state["normalized_research_topic"],
        dimension_research=material,
        draft_report=state["report_draft"],
        output_schema=json.dumps(ReportAudit.model_json_schema(), ensure_ascii=False),
    )
    result = (
        create_deepseek_model(configurable.reflection_model)
        .with_structured_output(ReportAudit, method="json_mode")
        .invoke(prompt)
    )
    audit = result.model_dump()
    current_results, sources, _ = _current_research_material(state)
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
    emit_research_event(
        "report_audit_completed",
        research_run_id=state["research_run_id"],
        passes=audit["passes"],
        revision_count=state.get("report_revision_count", 0),
    )
    return {"report_audit": audit}


def route_report_audit(state: OverallState):
    """Revise material audit failures within a bounded loop."""
    if state["report_audit"].get("passes"):
        return "finalize_answer"
    if state.get("report_revision_count", 0) >= state.get("max_report_revisions", 2):
        return "finalize_answer"
    return "revise_report"


def revise_report(state: OverallState, config: RunnableConfig):
    """Revise only the issues identified by the independent audit."""
    configurable = Configuration.from_runnable_config(config)
    model = state.get("reasoning_model") or configurable.answer_model
    current_results, _, _ = _current_research_material(state)
    material = format_dimension_results(
        current_results, max_claims_per_dimension=5, max_evidence_chars=120
    )
    prompt = report_revision_instructions.format(
        research_topic=state["normalized_research_topic"],
        dimension_research=material,
        draft_report=state["report_draft"],
        audit_findings=json.dumps(state["report_audit"], ensure_ascii=False),
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
    current_results, sources, _ = _current_research_material(state)
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
builder.add_node("draft_report", draft_report)
builder.add_node("audit_report", audit_report)
builder.add_node("revise_report", revise_report)
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
builder.add_edge("research_dimension", "draft_report")
builder.add_edge("draft_report", "audit_report")
builder.add_conditional_edges(
    "audit_report", route_report_audit, ["revise_report", "finalize_answer"]
)
builder.add_edge("revise_report", "audit_report")
builder.add_edge("finalize_answer", END)
graph = builder.compile(name="deepseek-tavily-multidimensional-research-agent")
