"""LangGraph workflow for clarification, dimension research, and reporting."""

# ruff: noqa: E402

import json
import os
import re
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
    normalize_search_score,
    render_source_citations,
    tavily_results_to_sources,
)

load_dotenv()


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
    assessment = state.get("reflection_assessment", {})
    gaps = assessment.get("missing_questions", [])
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
        "query_history": queries,
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
        search_score = source.get("score")
        normalized_search_score = normalize_search_score(search_score)
        evidence_score = (
            relevance * 0.4
            + authority * 0.25
            + recency * 0.15
            + (0.1 if primary else 0.0)
            + normalized_search_score * 0.1
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
            }
        )
    domain_counts: dict[str, int] = {}
    for source in sorted(
        evaluated, key=lambda item: item["evidence_score"], reverse=True
    ):
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
        key=lambda item: (
            item["quality_status"] == "accepted",
            item["evidence_score"],
        ),
        reverse=True,
    )
    for source in quality_ranked[configurable.max_selected_sources_per_dimension :]:
        source["quality_status"] = "rejected"
        source["rejection_reasons"] = [
            *source["rejection_reasons"],
            "The per-dimension evidence budget was reached.",
        ]
    selected = [
        source
        for source in evaluated
        if source["quality_status"] in {"accepted", "supplementary"}
    ]
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
    knowledge_gap = "\n".join(
        f"{gap.question}: {gap.reason}. Search focus: {gap.suggested_query_focus}"
        for gap in result.missing_questions
    )
    emit_research_event(
        "reflection_completed",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        is_sufficient=result.is_sufficient,
        knowledge_gap=knowledge_gap,
        missing_questions=[gap.model_dump() for gap in result.missing_questions],
        contradictions=[item.model_dump() for item in result.contradictions],
        confidence=result.confidence,
        loop=loop_count,
    )
    assessment = result.model_dump()
    prior_history = state.get("reflection_history", [])
    current_gap_keys = {
        gap.question.casefold().strip() for gap in result.missing_questions
    }
    prior_gap_keys = {
        gap.get("question", "").casefold().strip()
        for item in prior_history[-1:]
        for gap in item.get("missing_questions", [])
    }
    source_counts = state.get("evidence_source_count_history", [])
    stalled = bool(
        prior_history
        and current_gap_keys
        and current_gap_keys == prior_gap_keys
        and source_counts
        and len(state.get("selected_sources", [])) <= source_counts[-1]
    )
    high_priority_gap = any(gap.priority == "high" for gap in result.missing_questions)
    unresolved_conflict = any(item.requires_follow_up for item in result.contradictions)
    if result.is_sufficient and not high_priority_gap and not unresolved_conflict:
        completion_status = "sufficient"
        is_sufficient = True
    elif loop_count >= state["max_research_loops"]:
        completion_status = "loop_limit_reached"
        is_sufficient = False
    elif stalled:
        completion_status = "stalled"
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
    try:
        result = structured_model.invoke(
            build_prompt(
                selected,
                configurable.max_claim_source_chars,
                configurable.max_claims_per_dimension,
            )
        )
    except LengthFinishReasonError:
        # A concise second pass is preferable to failing the complete dimension when
        # DeepSeek consumes its response budget on a large evidence collection.
        retry_source_limit = min(len(selected), 6)
        retry_claim_limit = min(configurable.max_claims_per_dimension, 8)
        emit_research_event(
            "claim_extraction_retry",
            research_run_id=state["research_run_id"],
            dimension=state["dimension"],
            reason="length_limit",
            source_count=retry_source_limit,
            max_claims=retry_claim_limit,
        )
        result = structured_model.invoke(
            build_prompt(
                selected[:retry_source_limit],
                min(configurable.max_claim_source_chars, 1000),
                retry_claim_limit,
            )
        )
    valid_ids = {source["source_id"] for source in selected}
    source_by_id = {source["source_id"]: source for source in selected}
    claims = []
    for claim in result.claims[: configurable.max_claims_per_dimension]:
        supporting_ids = [
            source_id
            for source_id in claim.source_ids
            if source_id in valid_ids
        ]
        contradicting_ids = [
            source_id
            for source_id in claim.counter_source_ids
            if source_id in valid_ids
        ]
        if not supporting_ids:
            continue
        supporting_evidence = " ".join(
            source_by_id[source_id]["content"][:400].strip()
            for source_id in supporting_ids
        )
        claims.append(
            {
                "claim": claim.claim,
                "supporting_source_ids": supporting_ids,
                "supporting_evidence": supporting_evidence,
                "contradicting_source_ids": contradicting_ids,
                "confidence": 0.5,
                "uncertainty_reason": claim.uncertainty,
            }
        )
    emit_research_event(
        "claims_extracted",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        claim_count=len(claims),
    )
    return {"claims": claims, "dimension_summary": result.summary}


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
