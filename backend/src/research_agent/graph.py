"""LangGraph workflow for clarification, dimension research, and reporting."""

# ruff: noqa: E402

import json
import os
import re
import sys
import time
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit
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
    report_conclusion_instructions,
    report_conclusion_revision_instructions,
    report_consistency_audit_instructions,
    report_overview_instructions,
    report_overview_revision_instructions,
    report_planning_instructions,
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
    ReportPlan,
    ReportSectionPlan,
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


_WIRE_SERVICE_PATTERN = re.compile(
    r"(?:^|[\n(])\s*(?:xinhua|reuters|associated press|"
    r"agence\s+france-presse|afp)\b|"
    r"\b(?:source|reported by|according to|via)\s*:?-?\s*(?:xinhua|reuters|"
    r"associated press|agence\s+france-presse|afp)\b|"
    r"\b(?:xinhua|reuters|associated press|afp)\s+(?:reports?|reported)\b",
    re.IGNORECASE,
)
_SCOPE_STOPWORDS = {
    "about",
    "after",
    "against",
    "analysis",
    "answer",
    "answers",
    "authoritative",
    "before",
    "beyond",
    "china",
    "chinas",
    "current",
    "direct",
    "evidence",
    "including",
    "market",
    "official",
    "potential",
    "recent",
    "report",
    "research",
    "result",
    "since",
    "source",
    "sources",
    "statement",
    "study",
    "what",
    "which",
    "with",
}


def _apply_source_provenance_guardrails(
    source: Mapping[str, Any],
    *,
    source_type: str,
    authority: float,
    primary: bool,
    repost: bool,
    rejection_reasons: list[str],
) -> tuple[str, float, bool, bool, list[str]]:
    """Correct source-type claims contradicted by deterministic provenance."""
    parsed = urlsplit(str(source.get("canonical_url") or source.get("url", "")))
    domain = parsed.netloc.casefold()
    path = parsed.path.casefold()
    attribution_text = " ".join(
        [str(source.get("title", "")), str(source.get("content", ""))[:500]]
    )
    attributed_to_wire_service = bool(_WIRE_SERVICE_PATTERN.search(attribution_text))
    chinese_government_domain = domain == "gov.cn" or domain.endswith(".gov.cn")
    government_news_page = chinese_government_domain and "/news/" in path
    reasons = list(rejection_reasons)
    if attributed_to_wire_service and (
        source_type == "government" or chinese_government_domain
    ):
        source_type = "major_media"
        authority = min(authority, 0.75)
        primary = False
        repost = True
        reasons.append(
            "The page is attributed to a wire service; an official host does not "
            "make republished reporting a government primary source."
        )
    elif government_news_page and primary:
        primary = False
        reasons.append(
            "A government-portal news page is secondary reporting unless its "
            "content identifies an original policy, regulation, or dataset."
        )
    return source_type, authority, primary, repost, list(dict.fromkeys(reasons))


def _scope_terms(value: str) -> set[str]:
    """Return compact English terms useful for conservative scope matching."""
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{3,}", value.casefold())
        if token not in _SCOPE_STOPWORDS and not token.isdigit()
    }


def _gap_similarity_tokens(value: str) -> set[str]:
    """Tokenize English and Chinese gap requirements for conservative deduping."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    english = {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", normalized)
        if token not in _SCOPE_STOPWORDS
    }
    chinese_chunks = re.findall(r"[\u3400-\u9fff]+", normalized)
    chinese = {
        chunk[index : index + 2]
        for chunk in chinese_chunks
        for index in range(max(len(chunk) - 1, 0))
    }
    return english | chinese


def _gap_requirement_text(gap: Mapping[str, Any]) -> str:
    return " ".join(
        str(gap.get(field, ""))
        for field in ("question", "expected_evidence", "suggested_query_focus")
    ).strip()


def _gaps_are_semantically_equivalent(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Reject only high-confidence paraphrases of an already tracked gap."""
    if left.get("gap_id") and left.get("gap_id") == right.get("gap_id"):
        return True
    left_tokens = _gap_similarity_tokens(_gap_requirement_text(left))
    right_tokens = _gap_similarity_tokens(_gap_requirement_text(right))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    containment = overlap / min(len(left_tokens), len(right_tokens))
    union = len(left_tokens | right_tokens)
    jaccard = overlap / union if union else 0
    return containment >= 0.85 and jaccard >= 0.65


def _same_gap_requirement(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether a repeated stable ID has materially unchanged requirements."""
    left_types = sorted(str(item) for item in left.get("required_source_types", []))
    right_types = sorted(str(item) for item in right.get("required_source_types", []))
    left_without_id = {**left, "gap_id": ""}
    right_without_id = {**right, "gap_id": ""}
    return (
        _gaps_are_semantically_equivalent(left_without_id, right_without_id)
        and left_types == right_types
        and str(left.get("priority", "medium")) == str(right.get("priority", "medium"))
    )


def _is_material_actionable_gap(gap: Mapping[str, Any]) -> bool:
    """Keep only consequential gaps that specify a searchable evidence target."""
    if str(gap.get("priority", "low")) not in {"high", "medium"}:
        return False
    question = str(gap.get("question", "")).strip()
    expected = str(gap.get("expected_evidence", "")).strip()
    focus = str(gap.get("suggested_query_focus", "")).strip()
    return bool(question and focus and (expected or len(question) >= 20))


def _reflection_progress_snapshot(
    state: DimensionState, accepted_sources: list[ResearchSource]
) -> dict[str, list[str]]:
    """Capture durable progress signals between whole-dimension audits."""
    claim_keys = []
    for claim in state.get("gap_claim_ledger", []):
        claim_keys.append(
            "|".join(
                [
                    str(claim.get("claim", "")).strip(),
                    *sorted(str(item) for item in claim.get("supporting_source_ids", [])),
                ]
            )
        )
    return {
        "accepted_source_ids": sorted(
            str(source.get("source_id", ""))
            for source in accepted_sources
            if source.get("source_id")
        ),
        "verified_claim_keys": sorted(set(claim_keys)),
        "resolved_gap_ids": sorted(str(item) for item in state.get("resolved_gap_ids", [])),
    }


def _reflection_made_progress(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Count only newly added durable evidence or closures as progress."""
    return any(
        set(current.get(field, [])) - set(previous.get(field, []))
        for field in (
            "accepted_source_ids",
            "verified_claim_keys",
            "resolved_gap_ids",
        )
    )


def _claim_scope_rejection_reasons(
    gap: Mapping[str, Any],
    claim: str,
    evidence: list[dict],
    source_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Reject quote-grounded claims outside a gap's explicit topic or time scope."""
    requirement = " ".join(
        [str(gap.get("question", "")), str(gap.get("expected_evidence", ""))]
    )
    evidence_text = " ".join(
        [claim, *(str(item.get("quote", "")) for item in evidence)]
    )
    reasons = []
    required_terms = _scope_terms(requirement)
    evidence_terms = _scope_terms(evidence_text)
    if len(required_terms) >= 3 and not required_terms.intersection(evidence_terms):
        reasons.append("semantic_scope_mismatch")

    after_years = [
        int(year)
        for pattern in (
            r"\b(?:after|beyond|post)[-\s]*(20\d{2})\b",
            r"(20\d{2})\s*(?:之后|以后)",
        )
        for year in re.findall(pattern, requirement, flags=re.IGNORECASE)
    ]
    since_years = [
        int(year)
        for pattern in (
            r"\b(?:since|from)[-\s]*(20\d{2})\b",
            r"(20\d{2})\s*(?:以来|起)",
        )
        for year in re.findall(pattern, requirement, flags=re.IGNORECASE)
    ]
    content_years = {int(year) for year in re.findall(r"\b20\d{2}\b", evidence_text)}
    publication_years = set()
    if source_by_id:
        publication_years.update(
            int(year)
            for item in evidence
            for source in [source_by_id.get(str(item.get("source_id", "")), {})]
            for year in re.findall(
                r"\b20\d{2}\b", str(source.get("published_date", ""))
            )
        )
    # A publication date after a boundary does not prove that the quoted claim
    # itself covers that future horizon. It may only support "since/from" recency.
    if after_years and not any(year > max(after_years) for year in content_years):
        reasons.append("temporal_scope_mismatch")
    elif since_years and not any(
        year >= max(since_years) for year in content_years | publication_years
    ):
        reasons.append("temporal_scope_mismatch")
    return reasons


def _validate_gap_ledger_claim(
    claim: Mapping[str, Any],
    gap: Mapping[str, Any],
    selected_by_id: Mapping[str, Mapping[str, Any]],
    configurable: Configuration,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Revalidate one Gap claim against accepted evidence and explicit scope."""
    gap_id = str(gap["gap_id"])
    supporting_evidence = []
    for evidence in claim.get("supporting_evidence", []):
        if not isinstance(evidence, Mapping):
            continue
        source_id = str(evidence.get("source_id", ""))
        source = selected_by_id.get(source_id)
        if not source or not (
            gap_id == source.get("gap_id")
            or gap_id in (source.get("gap_ids", []) or [])
        ):
            continue
        located = locate_evidence_quote(
            str(source.get("content", "")),
            str(evidence.get("quote", "")),
            min_chars=configurable.min_evidence_quote_chars,
        )
        if located:
            quote, locator = located
            supporting_evidence.append(
                {"source_id": source_id, "quote": quote, "locator": locator}
            )
    if not supporting_evidence:
        return None, ["missing_verified_quote"]
    scope_rejections = _claim_scope_rejection_reasons(
        gap,
        str(claim.get("claim", "")),
        supporting_evidence,
        selected_by_id,
    )
    if scope_rejections:
        return None, scope_rejections
    counter_evidence = []
    for evidence in claim.get("contradicting_evidence", []):
        if not isinstance(evidence, Mapping):
            continue
        source_id = str(evidence.get("source_id", ""))
        source = selected_by_id.get(source_id)
        if not source:
            continue
        located = locate_evidence_quote(
            str(source.get("content", "")),
            str(evidence.get("quote", "")),
            min_chars=configurable.min_evidence_quote_chars,
        )
        if located:
            quote, locator = located
            counter_evidence.append(
                {"source_id": source_id, "quote": quote, "locator": locator}
            )
    normalized = {
        **claim,
        "gap_ids": [gap_id],
        "supporting_source_ids": list(
            dict.fromkeys(item["source_id"] for item in supporting_evidence)
        ),
        "supporting_evidence": supporting_evidence,
        "contradicting_source_ids": list(
            dict.fromkeys(item["source_id"] for item in counter_evidence)
        ),
        "contradicting_evidence": counter_evidence,
    }
    return normalized, []


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
    sends = []
    for dimension in state["research_dimensions"]:
        dimension_input = {
            "research_topic": topic,
            "research_run_id": state["research_run_id"],
            "dimension": dimension,
            "initial_search_query_count": state.get("initial_search_query_count", 3),
            "max_research_loops": state.get("max_research_loops", 3),
        }
        for field in (
            "dimension_reflection_soft_limit",
            "max_dimension_reflections",
        ):
            if state.get(field) is not None:
                dimension_input[field] = state[field]
        sends.append(Send("research_dimension", dimension_input))
    return sends


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
            "retained_evidence_source_ids": [],
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
        "reflection_no_progress_count": 0,
        "last_reflection_progress_snapshot": {},
        "research_loop_count": 0,
        "completion_status": "researching",
        "termination_reason": "",
        "is_sufficient": False,
        "resolved_gap_ids": [],
        "gap_source_coverage_ids": [],
        "gap_claim_ledger": [],
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
    resolved_gap_ids = set(state.get("resolved_gap_ids", []))
    protected_gap_ids_by_source: dict[str, set[str]] = {}
    for gap_id, gap in state.get("gap_registry", {}).items():
        protected_source_ids = set(gap.get("retained_evidence_source_ids", []))
        if gap.get("status") == "closed" or gap_id in resolved_gap_ids:
            # Rolling checkpoints created before retained evidence was recorded.
            protected_source_ids.update(gap.get("matched_source_ids", []))
        for source_id in protected_source_ids:
            protected_gap_ids_by_source.setdefault(source_id, set()).add(gap_id)
    previous_accepted_by_id = {
        source["source_id"]: source
        for source in state.get("selected_sources", [])
        if source.get("quality_status") == "accepted"
    }
    protected_sources = {
        source_id: previous_accepted_by_id[source_id]
        for source_id in protected_gap_ids_by_source
        if source_id in previous_accepted_by_id
    }
    all_candidates = deduplicate_sources(state.get("sources_gathered", []))
    candidate_by_url = {
        source.get("canonical_url") or source.get("url", ""): source
        for source in all_candidates
    }
    protected_urls = {
        source.get("canonical_url") or source.get("url", "")
        for source in protected_sources.values()
    }
    all_candidates = [
        source
        for source in all_candidates
        if (source.get("canonical_url") or source.get("url", "")) not in protected_urls
    ]
    for source_id, protected in protected_sources.items():
        canonical_url = protected.get("canonical_url") or protected.get("url", "")
        candidate = candidate_by_url.get(canonical_url, {})
        all_candidates.append(
            {
                **candidate,
                **protected,
                "gap_ids": list(
                    dict.fromkeys(
                        [
                            *candidate.get("gap_ids", []),
                            *protected.get("gap_ids", []),
                        ]
                    )
                ),
                "requested_source_types": sorted(
                    set(candidate.get("requested_source_types", []))
                    | set(protected.get("requested_source_types", []))
                ),
                "protected_gap_ids": sorted(protected_gap_ids_by_source[source_id]),
            }
        )
    protected_candidates = [
        source for source in all_candidates if source["source_id"] in protected_sources
    ]
    ordinary_candidates = sorted(
        (
            source
            for source in all_candidates
            if source["source_id"] not in protected_sources
        ),
        key=lambda source: normalize_search_score(source.get("score"), default=0),
        reverse=True,
    )
    # Candidate and selection limits are soft budgets for new evidence. Evidence
    # already used to close a gap is never truncated by a later research pass.
    ordinary_budget = max(
        configurable.max_source_candidates_per_dimension - len(protected_candidates),
        0,
    )
    candidates = [
        *protected_candidates,
        *ordinary_candidates[:ordinary_budget],
    ]
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
        source_type, authority, primary, repost, rejection_reasons = (
            _apply_source_provenance_guardrails(
                source,
                source_type=source_type,
                authority=authority,
                primary=primary,
                repost=repost,
                rejection_reasons=rejection_reasons,
            )
        )
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
    evaluated_by_id = {source["source_id"]: source for source in evaluated}
    for source_id, protected in protected_sources.items():
        current = evaluated_by_id.get(source_id)
        if current is None:
            continue
        stable = {
            **current,
            **protected,
            "gap_ids": list(
                dict.fromkeys(
                    [*current.get("gap_ids", []), *protected.get("gap_ids", [])]
                )
            ),
            "protected_gap_ids": sorted(protected_gap_ids_by_source[source_id]),
            "quality_status": "accepted",
            "rejection_reasons": list(protected.get("rejection_reasons", [])),
        }
        current.clear()
        current.update(stable)
    domain_counts: dict[str, int] = {}
    domain_ranked = sorted(
        evaluated,
        key=lambda source: (
            source["source_id"] in protected_sources,
            *_source_rank_key(source),
        ),
        reverse=True,
    )
    for source in domain_ranked:
        if source["quality_status"] == "rejected":
            continue
        domain = source.get("domain", "")
        if source["source_id"] in protected_sources:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            continue
        if domain_counts.get(domain, 0) >= configurable.max_sources_per_domain:
            source["quality_status"] = "rejected"
            source["rejection_reasons"] = [
                *source["rejection_reasons"],
                "The per-domain evidence limit was reached.",
            ]
            continue
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    protected_selected = [
        source
        for source in evaluated
        if source["source_id"] in protected_sources
        and source["quality_status"] in {"accepted", "supplementary"}
    ]
    quality_ranked = sorted(
        (
            source
            for source in evaluated
            if source["source_id"] not in protected_sources
            if source["quality_status"] in {"accepted", "supplementary"}
        ),
        key=_source_rank_key,
        reverse=True,
    )
    remaining_budget = max(
        configurable.max_selected_sources_per_dimension - len(protected_selected), 0
    )
    for source in quality_ranked[remaining_budget:]:
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
        protected_source_count=len(protected_selected),
        protected_gap_count=len(
            {
                gap_id
                for gap_ids in protected_gap_ids_by_source.values()
                for gap_id in gap_ids
            }
        ),
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
    verified_claims: list[dict[str, Any]] = []
    scope_rejection_reasons: list[str] = []
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
            accepted_claims = []
            rejected_scopes = []
            for claim in assessment_result.claims[:6]:
                if claim.gap_ids and gap["gap_id"] not in claim.gap_ids:
                    continue
                supporting_evidence = []
                for evidence in claim.evidence:
                    source = candidate_by_id.get(evidence.source_id)
                    located = (
                        locate_evidence_quote(
                            source.get("content", ""),
                            evidence.quote,
                            min_chars=configurable.min_evidence_quote_chars,
                        )
                        if source
                        else None
                    )
                    if located:
                        quote, locator = located
                        supporting_evidence.append(
                            {
                                "source_id": evidence.source_id,
                                "quote": quote,
                                "locator": locator,
                            }
                        )
                if not supporting_evidence:
                    continue
                rejected = _claim_scope_rejection_reasons(
                    gap, claim.claim, supporting_evidence, candidate_by_id
                )
                if rejected:
                    rejected_scopes.extend(rejected)
                    continue
                counter_evidence = []
                for evidence in claim.counter_evidence:
                    source = candidate_by_id.get(evidence.source_id)
                    located = (
                        locate_evidence_quote(
                            source.get("content", ""),
                            evidence.quote,
                            min_chars=configurable.min_evidence_quote_chars,
                        )
                        if source
                        else None
                    )
                    if located:
                        quote, locator = located
                        counter_evidence.append(
                            {
                                "source_id": evidence.source_id,
                                "quote": quote,
                                "locator": locator,
                            }
                        )
                accepted_claims.append(
                    {
                        "claim": claim.claim,
                        "gap_ids": [gap["gap_id"]],
                        "supporting_source_ids": list(
                            dict.fromkeys(
                                item["source_id"] for item in supporting_evidence
                            )
                        ),
                        "supporting_evidence": supporting_evidence,
                        "contradicting_source_ids": list(
                            dict.fromkeys(
                                item["source_id"] for item in counter_evidence
                            )
                        ),
                        "contradicting_evidence": counter_evidence,
                        "confidence": claim.confidence,
                        "uncertainty_reason": claim.uncertainty,
                    }
                )
            matched_source_ids = list(
                dict.fromkeys(
                    source_id
                    for claim in accepted_claims
                    for source_id in claim["supporting_source_ids"]
                )
            )
            contradictory_source_ids = list(
                dict.fromkeys(
                    source_id
                    for claim in accepted_claims
                    for source_id in claim["contradicting_source_ids"]
                )
            )
            assessment = GapEvidenceAssessment(
                gap_id=gap["gap_id"],
                directly_answers_gap=bool(accepted_claims),
                matched_source_ids=matched_source_ids,
                supported_claims=list(
                    dict.fromkeys(claim["claim"] for claim in accepted_claims)
                ),
                contradictory_source_ids=contradictory_source_ids,
                remaining_evidence=(
                    ""
                    if accepted_claims
                    else gap.get("expected_evidence", "")
                    if rejected_scopes
                    else assessment_result.summary or gap.get("expected_evidence", "")
                ),
            )
            return assessment, accepted_claims, list(dict.fromkeys(rejected_scopes))

        try:
            result, verified_claims, scope_rejection_reasons = invoke_assessment()
        except (
            AttributeError,
            LengthFinishReasonError,
            OutputParserException,
            TypeError,
            ValueError,
        ) as error:
            try:
                result, verified_claims, scope_rejection_reasons = invoke_assessment(
                    retry=True
                )
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
        "scope_rejection_reasons": scope_rejection_reasons,
    }
    if assessment_status != "deterministic_quality_check":
        assessment["verified_claims"] = verified_claims
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

    ledger = list(state.get("gap_claim_ledger", []))
    if "verified_claims" in assessment:
        retained_other_gap_claims = [
            claim for claim in ledger if gap_id not in claim.get("gap_ids", [])
        ]
        current_gap_claims = []
        scope_rejection_reasons = list(assessment.get("scope_rejection_reasons", []))
        seen_claims = set()
        for claim in [
            *(claim for claim in ledger if gap_id in claim.get("gap_ids", [])),
            *assessment.get("verified_claims", []),
        ]:
            validated, rejected = _validate_gap_ledger_claim(
                claim, gap, selected_by_id, configurable
            )
            scope_rejection_reasons.extend(rejected)
            if validated is None:
                continue
            claim_key = (
                re.sub(r"\s+", " ", validated["claim"]).strip().casefold(),
                tuple(validated["supporting_source_ids"]),
            )
            if claim_key in seen_claims:
                continue
            seen_claims.add(claim_key)
            current_gap_claims.append(validated)
        ledger = [*retained_other_gap_claims, *current_gap_claims]
        matched_ids = list(
            dict.fromkeys(
                source_id
                for claim in current_gap_claims
                for source_id in claim["supporting_source_ids"]
            )
        )
        supported_claims = list(
            dict.fromkeys(claim["claim"] for claim in current_gap_claims)
        )
        contradictory_ids = list(
            dict.fromkeys(
                source_id
                for claim in current_gap_claims
                for source_id in claim["contradicting_source_ids"]
            )
        )
        direct_evidence = bool(current_gap_claims)
    else:
        # Compatibility for deterministic quality gaps and rolling checkpoints
        # created before the structured Gap claim ledger existed.
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
        contradictory_ids = list(
            dict.fromkeys(assessment.get("contradictory_source_ids", []))
        )
        direct_evidence = bool(
            gap.get("direct_evidence_confirmed")
            or assessment.get("directly_answers_gap")
        )
        scope_rejection_reasons = []
    snapshot = _gap_closure_snapshot(
        gap,
        matched_ids=matched_ids,
        supported_claims=supported_claims,
        contradictory_ids=contradictory_ids,
        selected_by_id=selected_by_id,
        configurable=configurable,
        direct_evidence_confirmed=direct_evidence,
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
            "retained_evidence_source_ids": matched_ids,
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
            "verified_claim_count": len(
                [claim for claim in ledger if gap_id in claim.get("gap_ids", [])]
            ),
            "scope_rejection_reasons": list(dict.fromkeys(scope_rejection_reasons)),
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
        "gap_claim_ledger": ledger,
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
    scope_guidance = {
        "temporal_scope_mismatch": (
            "Include the exact required year or horizon in every query and reject "
            "documents whose quoted facts only cover earlier periods."
        ),
        "semantic_scope_mismatch": (
            "Search the exact unresolved subquestion and its required metric or "
            "policy instrument, not the broader topic."
        ),
    }
    guidance_parts.extend(
        scope_guidance[reason]
        for reason in gap.get("scope_rejection_reasons", [])
        if reason in scope_guidance
    )
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


def _filter_reflection_candidates(
    candidates: list[dict], registry: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict], list[dict]]:
    """Keep material novel gaps and explain why repeated/optional ones were skipped."""
    pending: list[dict] = []
    skipped: list[dict] = []
    comparison_pool: list[Mapping[str, Any]] = list(registry.values())
    for candidate in candidates:
        if not _is_material_actionable_gap(candidate):
            skipped.append(
                {
                    "gap_id": candidate.get("gap_id", ""),
                    "reason": "optional_or_not_actionable",
                }
            )
            continue
        exact_existing = registry.get(str(candidate.get("gap_id", "")))
        if (
            str(candidate.get("gap_id", "")).startswith("quality-")
            and exact_existing
            and exact_existing.get("status") == "closed"
            and not any(
                item.get("gap_id") == candidate.get("gap_id") for item in pending
            )
        ):
            pending.append(candidate)
            comparison_pool.append(candidate)
            continue
        equivalent = next(
            (
                existing
                for existing in comparison_pool
                if _gaps_are_semantically_equivalent(candidate, existing)
            ),
            None,
        )
        if equivalent is not None and not _same_gap_requirement(
            candidate, equivalent
        ):
            equivalent = None
        if equivalent is not None:
            skipped.append(
                {
                    "gap_id": candidate.get("gap_id", ""),
                    "equivalent_gap_id": equivalent.get("gap_id", ""),
                    "reason": "duplicate_or_already_tracked",
                }
            )
            continue
        pending.append(candidate)
        comparison_pool.append(candidate)
    return pending, skipped


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
    pending, skipped_candidates = _filter_reflection_candidates(candidates, registry)
    for candidate in pending:
        existing = registry.get(candidate["gap_id"])
        if existing and existing.get("status") in {"closed", "unresolvable"}:
            candidate["origin"] = "reopened"
            candidate["status"] = "reopened"

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
    progress_snapshot = _reflection_progress_snapshot(state, accepted)
    previous_snapshot = state.get("last_reflection_progress_snapshot") or None
    if previous_snapshot is None or _reflection_made_progress(
        previous_snapshot, progress_snapshot
    ):
        reflection_no_progress_count = 0
    else:
        reflection_no_progress_count = (
            int(state.get("reflection_no_progress_count", 0)) + 1
        )
    soft_limit = min(
        6,
        max(
            1,
            int(
                state.get(
                    "dimension_reflection_soft_limit",
                    configurable.dimension_reflection_soft_limit,
                )
            ),
        ),
    )
    hard_limit = min(
        8,
        max(
            soft_limit,
            int(
                state.get(
                    "max_dimension_reflections",
                    configurable.max_dimension_reflections,
                )
            )
        ),
    )
    within_soft_budget = reflection_count < soft_limit
    adaptive_extension = bool(
        pending
        and reflection_count < hard_limit
        and reflection_no_progress_count
        < configurable.max_reflection_no_progress_rounds
    )
    should_continue = bool(pending and (within_soft_budget or adaptive_extension))

    termination_reason = ""
    if search_unavailable:
        completion_status = "search_unavailable"
        termination_reason = "search_unavailable"
    elif deterministic_sufficient:
        completion_status = "sufficient"
    elif should_continue:
        completion_status = "discovering_gaps"
    else:
        completion_status = "completed_with_limitations"
        if pending and reflection_no_progress_count >= (
            configurable.max_reflection_no_progress_rounds
        ):
            termination_reason = "no_progress"
        elif pending and reflection_count >= hard_limit:
            termination_reason = "reflection_limit_reached"
        elif unresolved_conflict:
            termination_reason = "unresolved_contradiction"
        elif quality_gaps:
            termination_reason = "source_quality_unmet"
        elif unresolved_terminal:
            termination_reason = "gap_attempts_exhausted"
        elif skipped_candidates:
            termination_reason = "optional_or_duplicate_gaps_deferred"
        else:
            termination_reason = "unresolved_evidence"

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
        termination_reason=termination_reason,
        reflection_soft_limit=soft_limit,
        reflection_hard_limit=hard_limit,
        reflection_no_progress_count=reflection_no_progress_count,
        knowledge_gap=knowledge_gap,
    )
    return {
        "is_sufficient": deterministic_sufficient,
        "completion_status": completion_status,
        "reflection_assessment": assessment,
        "reflection_history": [assessment],
        "pending_reflection_gaps": pending,
        "dimension_reflection_count": reflection_count,
        "reflection_no_progress_count": reflection_no_progress_count,
        "last_reflection_progress_snapshot": progress_snapshot,
        "current_knowledge_gap": knowledge_gap,
        "termination_reason": termination_reason,
        "skipped_reflection_gaps": skipped_candidates,
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
        if not gap_id.startswith("quality-") and _same_gap_requirement(
            candidate, existing
        ):
            continue
        candidate.update(
            {
                "origin": "reopened",
                "status": "reopened",
                "attempt_count": 0,
                "no_progress_count": 0,
                "strategy_level": int(existing.get("strategy_level", 0)) + 1,
                "matched_source_ids": existing.get("matched_source_ids", []),
                "retained_evidence_source_ids": existing.get(
                    "retained_evidence_source_ids",
                    existing.get("matched_source_ids", []),
                ),
                "supported_claims": existing.get("supported_claims", []),
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
        "termination_reason": "",
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
                    "retained_evidence_source_ids": matched_ids,
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
            else:
                gap["status"] = "partial"
                gap["closure_reason"] = (
                    "Claim reconciliation found final-ledger blockers: "
                    + ", ".join(snapshot["closure_blockers"])
                )
                resolved_ids.discard(gap_id)
            if (
                snapshot["direct_evidence_confirmed"]
                and snapshot["requested_type_satisfied"]
            ):
                coverage_ids.add(gap_id)
            else:
                coverage_ids.discard(gap_id)
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
    source_by_id = {source["source_id"]: source for source in selected}
    gap_registry = state.get("gap_registry", {})
    ledger_claims = []
    ledger_seen = set()
    for claim in state.get("gap_claim_ledger", []):
        for gap_id in claim.get("gap_ids", []):
            gap = gap_registry.get(gap_id)
            if not gap:
                continue
            validated, _ = _validate_gap_ledger_claim(
                claim, gap, source_by_id, configurable
            )
            if validated is None:
                continue
            key = (
                re.sub(r"\s+", " ", validated["claim"]).strip().casefold(),
                tuple(validated["supporting_source_ids"]),
            )
            if key not in ledger_seen:
                ledger_seen.add(key)
                ledger_claims.append(validated)

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
            result = ClaimExtraction(
                claims=[],
                summary="No additional claim passed structured evidence extraction.",
            )
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
                    gap = gap_registry[gap_id]
                    if not _claim_scope_rejection_reasons(
                        gap, claim.claim, supporting_evidence, source_by_id
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
    combined_claims = {}
    for claim in [*ledger_claims, *claims]:
        key = (
            re.sub(r"\s+", " ", claim["claim"]).strip().casefold(),
            tuple(claim.get("supporting_source_ids", [])),
        )
        combined_claims.setdefault(key, claim)
    claims = list(combined_claims.values())[: configurable.max_claims_per_dimension]
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


def audit_final_gap_ledger(state: DimensionState, config: RunnableConfig):
    """Revalidate every gap against the immutable final accepted evidence ledger."""
    configurable = Configuration.from_runnable_config(config)
    selected_by_id = {
        source["source_id"]: source
        for source in state.get("selected_sources", [])
        if source.get("quality_status") == "accepted"
    }
    claims_by_gap: dict[str, list[dict]] = {}
    claim_source_ids_by_gap: dict[str, list[str]] = {}
    rejected_claim_reasons_by_gap: dict[str, list[str]] = {}
    gap_registry = state.get("gap_registry", {})
    candidate_claims = [
        *state.get("gap_claim_ledger", []),
        *state.get("claims", []),
    ]
    for claim in candidate_claims:
        for gap_id in claim.get("gap_ids", []):
            gap = gap_registry.get(gap_id)
            if not gap:
                continue
            validated, rejected = _validate_gap_ledger_claim(
                claim, gap, selected_by_id, configurable
            )
            if validated is None:
                rejected_claim_reasons_by_gap.setdefault(gap_id, []).extend(rejected)
                continue
            existing_keys = {
                (
                    item.get("claim", "").casefold(),
                    tuple(item.get("supporting_source_ids", [])),
                )
                for item in claims_by_gap.get(gap_id, [])
            }
            key = (
                validated.get("claim", "").casefold(),
                tuple(validated.get("supporting_source_ids", [])),
            )
            if key not in existing_keys:
                claims_by_gap.setdefault(gap_id, []).append(validated)
            claim_source_ids_by_gap.setdefault(gap_id, []).extend(
                validated["supporting_source_ids"]
            )

    registry = {}
    prior_resolved_ids = set(state.get("resolved_gap_ids", []))
    resolved_ids: set[str] = set()
    coverage_ids: set[str] = set()
    revoked_gap_ids = []
    retryable_gap_ids = []
    removed_source_ids: set[str] = set()
    for gap_id, original_gap in state.get("gap_registry", {}).items():
        gap = dict(original_gap)
        prior_status = gap.get("status", "open")
        prior_matched_ids = list(dict.fromkeys(gap.get("matched_source_ids", [])))
        missing_ids = [
            source_id
            for source_id in prior_matched_ids
            if source_id not in selected_by_id
        ]
        removed_source_ids.update(missing_ids)
        if gap_id == "quality-primary-source":
            matched_ids = [
                source_id
                for source_id, source in selected_by_id.items()
                if source.get("is_primary_source")
            ]
        elif gap_id == "quality-authoritative-source":
            matched_ids = [
                source_id
                for source_id, source in selected_by_id.items()
                if source.get("is_authoritative_source")
            ]
        elif gap_id == "quality-accepted-sources":
            matched_ids = list(selected_by_id)
        else:
            matched_ids = list(
                dict.fromkeys(
                    [
                        *(
                            source_id
                            for source_id in prior_matched_ids
                            if source_id in selected_by_id
                        ),
                        *claim_source_ids_by_gap.get(gap_id, []),
                    ]
                )
            )
        final_claims = claims_by_gap.get(gap_id, [])
        # Only quote-verified claims that still point to accepted final-ledger
        # evidence may satisfy the final supported-claim requirement. Earlier
        # model assessments are useful during research but are not immutable
        # evidence provenance.
        supported_claims = list(
            dict.fromkeys(claim.get("claim", "") for claim in final_claims)
        )
        supported_claims = [claim for claim in supported_claims if claim]
        if gap_id.startswith("quality-") and matched_ids:
            supported_claims = [
                f"The final ledger has {len(matched_ids)} qualifying accepted source(s)."
            ]
        contradictory_ids = list(
            dict.fromkeys(
                source_id
                for source_id in gap.get("contradictory_source_ids", [])
                if source_id in selected_by_id
            )
        )
        direct_evidence_confirmed = bool(
            matched_ids
            if gap_id.startswith("quality-")
            else claim_source_ids_by_gap.get(gap_id)
        )
        snapshot = _gap_closure_snapshot(
            gap,
            matched_ids=matched_ids,
            supported_claims=supported_claims,
            contradictory_ids=contradictory_ids,
            selected_by_id=selected_by_id,
            configurable=configurable,
            direct_evidence_confirmed=direct_evidence_confirmed,
        )
        blockers = snapshot["closure_blockers"]
        was_resolved = prior_status == "closed" or gap_id in prior_resolved_ids
        if blockers:
            if was_resolved:
                revoked_gap_ids.append(gap_id)
            exhausted = bool(
                prior_status == "unresolvable"
                or int(gap.get("attempt_count", 0))
                >= int(state.get("max_research_loops", 1))
                or int(gap.get("no_progress_count", 0))
                >= configurable.max_gap_no_progress_attempts
            )
            status = "unresolvable" if exhausted else "partial"
            if was_resolved and not exhausted:
                retryable_gap_ids.append(gap_id)
            closure_reason = (
                "Final evidence ledger audit revoked closure because: "
                + ", ".join(blockers)
            )
        else:
            status = "closed"
            closure_reason = (
                "Final accepted evidence ledger satisfies every deterministic "
                "gap closure requirement."
            )
            resolved_ids.add(gap_id)
        if direct_evidence_confirmed and snapshot["requested_type_satisfied"]:
            coverage_ids.add(gap_id)
        gap.update(
            {
                "status": status,
                "matched_source_ids": matched_ids,
                "retained_evidence_source_ids": (
                    matched_ids
                    if gap_id.startswith("quality-")
                    else list(dict.fromkeys(claim_source_ids_by_gap.get(gap_id, [])))
                ),
                "removed_matched_source_ids": missing_ids,
                "supported_claims": supported_claims,
                "verified_claim_count": len(final_claims),
                "contradictory_source_ids": contradictory_ids,
                "direct_evidence_confirmed": direct_evidence_confirmed,
                "requested_source_type_satisfied": snapshot["requested_type_satisfied"],
                "independent_source_count": len(snapshot["independent_domains"]),
                "required_independent_source_count": snapshot["required_independent"],
                "closure_blockers": blockers,
                "closure_reason": closure_reason,
                "remaining_evidence": (
                    ""
                    if not blockers
                    else gap.get("remaining_evidence")
                    or gap.get("expected_evidence", "")
                ),
                "final_ledger_audit_status": (
                    "passed"
                    if not blockers
                    else "closure_revoked"
                    if was_resolved
                    else "blocked"
                ),
                "scope_rejection_reasons": list(
                    dict.fromkeys(
                        [
                            *gap.get("scope_rejection_reasons", []),
                            *rejected_claim_reasons_by_gap.get(gap_id, []),
                        ]
                    )
                ),
            }
        )
        registry[gap_id] = gap

    unresolved = [gap for gap in registry.values() if gap.get("status") != "closed"]
    unresolved_conflict = any(
        bool(conflict.get("requires_follow_up"))
        for conflict in state.get("reflection_assessment", {}).get("contradictions", [])
        if isinstance(conflict, dict)
    )
    is_sufficient = bool(registry and not unresolved and not unresolved_conflict)
    prior_completion = state.get("completion_status", "partial")
    termination_reason = state.get("termination_reason", "")
    if retryable_gap_ids:
        completion_status = "researching"
        termination_reason = ""
        is_sufficient = False
    elif prior_completion == "search_unavailable":
        completion_status = "search_unavailable"
        termination_reason = "search_unavailable"
        is_sufficient = False
    elif is_sufficient:
        completion_status = "sufficient"
        termination_reason = ""
    elif prior_completion in {"completed_with_limitations", "budget_exhausted"}:
        completion_status = "completed_with_limitations"
        termination_reason = termination_reason or "reflection_limit_reached"
    else:
        completion_status = "completed_with_limitations"
        termination_reason = termination_reason or "final_gap_audit_unresolved"
    reflection_assessment = dict(state.get("reflection_assessment", {}))
    reflection_assessment["is_sufficient"] = is_sufficient
    reflection_assessment["missing_questions"] = unresolved
    knowledge_gap = "\n".join(
        f"[{gap['gap_id']}] {gap.get('question', '')}: "
        f"{gap.get('remaining_evidence') or gap.get('expected_evidence', '')}"
        for gap in unresolved
    )
    final_audit = {
        "passes": not unresolved,
        "gap_count": len(registry),
        "resolved_gap_count": len(resolved_ids),
        "revoked_gap_ids": sorted(revoked_gap_ids),
        "retryable_gap_ids": sorted(retryable_gap_ids),
        "removed_matched_source_ids": sorted(removed_source_ids),
        "unresolved_gap_ids": sorted(gap["gap_id"] for gap in unresolved),
    }
    emit_research_event(
        "final_gap_ledger_audited",
        research_run_id=state["research_run_id"],
        dimension=state["dimension"],
        audit=final_audit,
    )
    result = {
        "gap_registry": registry,
        "resolved_gap_ids": sorted(resolved_ids),
        "gap_source_coverage_ids": sorted(coverage_ids),
        "is_sufficient": is_sufficient,
        "completion_status": completion_status,
        "termination_reason": termination_reason,
        "reflection_assessment": reflection_assessment,
        "current_knowledge_gap": knowledge_gap,
        "final_gap_audit": final_audit,
        "gap_claim_ledger": [
            claim for claims in claims_by_gap.values() for claim in claims
        ],
    }
    if retryable_gap_ids:
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        retry_gap_id = min(
            retryable_gap_ids,
            key=lambda item: (
                priority_rank.get(registry[item].get("priority", "low"), 3),
                int(registry[item].get("attempt_count", 0)),
                item,
            ),
        )
        result.update(
            {
                "active_gap_id": retry_gap_id,
                "active_gap": registry[retry_gap_id],
                "gap_processing_complete": False,
                "gap_route": "final_audit_retry",
            }
        )
    return result


def route_final_gap_audit(state: DimensionState):
    """Retry an audit-revoked Gap while its bounded search budget remains."""
    if state.get("final_gap_audit", {}).get("retryable_gap_ids"):
        return "replan_search"
    return END


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
dimension_builder.add_node("audit_final_gap_ledger", audit_final_gap_ledger)
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
dimension_builder.add_edge("extract_claims", "audit_final_gap_ledger")
dimension_builder.add_conditional_edges(
    "audit_final_gap_ledger",
    route_final_gap_audit,
    ["replan_search", END],
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
                "verified_claim_count": int(gap.get("verified_claim_count", 0)),
                "scope_rejection_reasons": gap.get("scope_rejection_reasons", []),
                "final_ledger_audit_status": gap.get(
                    "final_ledger_audit_status", "not_audited"
                ),
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
                "retained_evidence_source_ids": gap.get(
                    "retained_evidence_source_ids", []
                ),
                "matched_source_types": matched_source_types,
                "matched_sources": matched_sources,
                "search_strategy": gap.get("search_strategy", []),
                "excluded_domains": gap.get("excluded_domains", []),
                "missing_matched_source_ids": [
                    source_id
                    for source_id in matched_source_ids
                    if source_id not in accepted_by_id
                ],
                "removed_matched_source_ids": gap.get("removed_matched_source_ids", []),
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
        "termination_reason": result.get("termination_reason", ""),
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
        "final_gap_audit": result.get("final_gap_audit", {}),
    }
    emit_research_event(
        "dimension_completed",
        research_run_id=result["research_run_id"],
        dimension=result["dimension"],
        is_sufficient=result["is_sufficient"],
        loops=result["research_loop_count"],
        completion_status=result["completion_status"],
        termination_reason=result.get("termination_reason", ""),
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


def _fallback_report_plan(
    results: list[DimensionResult], research_topic: str
) -> dict[str, Any]:
    """Build a complete evidence-safe editorial plan without factual invention."""
    sections = []
    for index, result in enumerate(results):
        dimension = result["dimension"]
        sections.append(
            ReportSectionPlan(
                dimension_id=str(dimension["id"]),
                objective=f"Explain how {dimension['title']} answers its assigned scope.",
                synthesis_direction=(
                    "Connect the audited findings through chronology, causality, "
                    "comparison, or implications instead of enumerating them."
                ),
                claim_ids=[
                    str(claim["claim_id"])
                    for claim in result.get("claims", [])
                    if claim.get("claim_id")
                ],
                transition=(
                    "Establish the foundation for the argument."
                    if index == 0
                    else "Build on the preceding section without repeating it."
                ),
            ).model_dump()
        )
    return ReportPlan(
        thesis=(
            f"Answer {research_topic} by integrating the audited findings across "
            "the research dimensions and keeping material uncertainty explicit."
        ),
        narrative_strategy=(
            "Develop one cumulative argument across the dimensions, moving from "
            "context and evidence to interpretation and implications."
        ),
        sections=[ReportSectionPlan.model_validate(item) for item in sections],
        conclusion_direction=(
            "Answer the main question directly by integrating the strongest findings "
            "and the limitations that materially affect interpretation."
        ),
        limitation_strategy=(
            "Consolidate limitations at the point where they affect interpretation "
            "and avoid repeating generic caveats after individual facts."
        ),
    ).model_dump()


def _normalize_report_plan(
    plan: ReportPlan,
    results: list[DimensionResult],
    research_topic: str,
) -> dict[str, Any]:
    """Constrain model planning to known dimensions and audited claim IDs."""
    fallback = _fallback_report_plan(results, research_topic)
    result_by_dimension = {
        str(result["dimension"]["id"]): result for result in results
    }
    proposed_by_dimension = {}
    proposed_order = []
    for section in plan.sections:
        dimension_id = str(section.dimension_id)
        if (
            dimension_id not in result_by_dimension
            or dimension_id in proposed_by_dimension
        ):
            continue
        proposed_by_dimension[dimension_id] = section
        proposed_order.append(dimension_id)
    ordered_dimension_ids = [
        *proposed_order,
        *[
            dimension_id
            for dimension_id in result_by_dimension
            if dimension_id not in proposed_by_dimension
        ],
    ]
    fallback_by_dimension = {
        str(section["dimension_id"]): section for section in fallback["sections"]
    }
    normalized_sections = []
    for dimension_id in ordered_dimension_ids:
        result = result_by_dimension[dimension_id]
        fallback_section = fallback_by_dimension[dimension_id]
        valid_claim_ids = [
            str(claim["claim_id"])
            for claim in result.get("claims", [])
            if claim.get("claim_id")
        ]
        proposed = proposed_by_dimension.get(dimension_id)
        proposed_ids = (
            [claim_id for claim_id in proposed.claim_ids if claim_id in valid_claim_ids]
            if proposed
            else []
        )
        claim_ids = list(dict.fromkeys([*proposed_ids, *valid_claim_ids]))
        normalized_sections.append(
            {
                "dimension_id": dimension_id,
                "objective": (
                    proposed.objective.strip()
                    if proposed and proposed.objective.strip()
                    else fallback_section["objective"]
                ),
                "synthesis_direction": (
                    proposed.synthesis_direction.strip()
                    if proposed and proposed.synthesis_direction.strip()
                    else fallback_section["synthesis_direction"]
                ),
                "claim_ids": claim_ids,
                "transition": (
                    proposed.transition.strip()
                    if proposed and proposed.transition.strip()
                    else fallback_section["transition"]
                ),
            }
        )
    return {
        "thesis": plan.thesis.strip() or fallback["thesis"],
        "narrative_strategy": (
            plan.narrative_strategy.strip() or fallback["narrative_strategy"]
        ),
        "sections": normalized_sections,
        "conclusion_direction": (
            plan.conclusion_direction.strip() or fallback["conclusion_direction"]
        ),
        "limitation_strategy": (
            plan.limitation_strategy.strip() or fallback["limitation_strategy"]
        ),
    }


def generate_report_plan(state: OverallState, config: RunnableConfig):
    """Plan a single narrative before drafting independent report sections."""
    configurable = Configuration.from_runnable_config(config)
    results = state.get("report_dimension_results", [])
    topic = state["normalized_research_topic"]
    claim_catalog = [
        {
            "dimension_id": str(result["dimension"]["id"]),
            "dimension_title": result["dimension"]["title"],
            "dimension_scope": result["dimension"]["scope"],
            "claims": [
                {
                    "claim_id": claim.get("claim_id", ""),
                    "claim": claim.get("claim", ""),
                    "uncertainty": claim.get("uncertainty_reason", ""),
                }
                for claim in result.get("claims", [])
            ],
        }
        for result in results
    ]
    prompt = report_planning_instructions.format(
        output_schema=json.dumps(ReportPlan.model_json_schema(), ensure_ascii=False),
        research_topic=topic,
        claim_catalog=json.dumps(claim_catalog, ensure_ascii=False),
        conflict_ledger=_format_conflict_ledger(state.get("claim_conflicts", [])),
    )
    model = state.get("reasoning_model") or configurable.answer_model
    fallback = _fallback_report_plan(results, topic)
    try:
        proposed = (
            create_deepseek_model(model)
            .with_structured_output(ReportPlan, method="json_mode")
            .invoke(prompt)
        )
        if not isinstance(proposed, ReportPlan):
            raise TypeError("Report planning returned an unexpected type")
        plan = _normalize_report_plan(proposed, results, topic)
        planning_mode = "model"
    except (
        AttributeError,
        LengthFinishReasonError,
        OutputParserException,
        TypeError,
        ValueError,
    ) as error:
        plan = fallback
        planning_mode = "deterministic_fallback"
        emit_research_event(
            "report_planning_fallback",
            research_run_id=state["research_run_id"],
            error=str(error),
        )
    emit_research_event(
        "report_plan_created",
        research_run_id=state["research_run_id"],
        planning_mode=planning_mode,
        section_count=len(plan["sections"]),
        thesis=plan["thesis"],
    )
    return {"report_plan": plan}


def _order_results_by_report_plan(
    results: list[DimensionResult], report_plan: Mapping[str, Any]
) -> list[DimensionResult]:
    """Apply the validated editorial section order without dropping dimensions."""
    result_by_dimension = {
        str(result["dimension"]["id"]): result for result in results
    }
    ordered = []
    for section in report_plan.get("sections", []):
        result = result_by_dimension.pop(str(section.get("dimension_id", "")), None)
        if result is not None:
            ordered.append(result)
    ordered.extend(result_by_dimension.values())
    return ordered


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


def _research_limitation_message(result: DimensionResult, *, chinese: bool) -> str:
    """Translate internal stop reasons into concise report-facing language."""
    reason = result.get("termination_reason", "unresolved_evidence")
    chinese_reasons = {
        "gap_attempts_exhausted": "部分证据缺口已达到搜索次数上限",
        "reflection_limit_reached": "已达到维度复核的硬上限",
        "no_progress": "连续复核未获得新的有效证据",
        "source_quality_unmet": "来源质量或来源类型要求仍未完全满足",
        "unresolved_contradiction": "仍存在需要进一步核实的矛盾证据",
        "search_unavailable": "网页搜索服务不可用",
        "optional_or_duplicate_gaps_deferred": "其余缺口属于重复或低影响问题",
        "final_gap_audit_unresolved": "最终证据审计仍发现未关闭的缺口",
        "unresolved_evidence": "仍有证据缺口未完全解决",
    }
    english_reasons = {
        "gap_attempts_exhausted": "some evidence gaps reached their search-attempt limit",
        "reflection_limit_reached": "the hard dimension-reflection limit was reached",
        "no_progress": "consecutive audits found no new qualifying evidence",
        "source_quality_unmet": "source quality or source-type requirements remain unmet",
        "unresolved_contradiction": "material contradictory evidence remains unresolved",
        "search_unavailable": "web search was unavailable",
        "optional_or_duplicate_gaps_deferred": "remaining gaps were duplicate or low impact",
        "final_gap_audit_unresolved": "the final evidence audit found unresolved gaps",
        "unresolved_evidence": "some evidence gaps remain unresolved",
    }
    if chinese:
        return "已完成并保留局限性说明：" + chinese_reasons.get(
            reason, chinese_reasons["unresolved_evidence"]
        ) + "。"
    return "Completed with limitations: " + english_reasons.get(
        reason, english_reasons["unresolved_evidence"]
    ) + "."


def _deterministic_report_section(result: DimensionResult, research_topic: str) -> str:
    """Build a bounded, citation-safe section when model generation cannot finish."""
    chinese = _uses_chinese(research_topic)
    claims = result.get("claims", [])
    if claims:
        sentences = []
        for claim in claims:
            sentence = claim["claim"].strip().rstrip(".!?。！？")
            markers = " ".join(
                f"[{source_id}]" for source_id in claim["supporting_source_ids"]
            )
            sentence = f"{sentence} {markers}".strip()
            if claim.get("uncertainty_reason"):
                sentence += (
                    f" 这一判断仍需注意：{claim['uncertainty_reason']}"
                    if chinese
                    else f" This finding remains qualified by {claim['uncertainty_reason']}"
                )
            if sentence[-1:] not in ".!?。！？":
                sentence += "。" if chinese else "."
            sentences.append(sentence)
        separator = "" if chinese else " "
        findings = "\n\n".join(
            separator.join(sentences[index : index + 3])
            for index in range(0, len(sentences), 3)
        )
    else:
        findings = (
            "该维度没有通过证据审计的结论。"
            if chinese
            else "No claim passed the evidence audit for this dimension."
        )
    limitations = []
    if result.get("completion_status") != "sufficient":
        limitations.append(_research_limitation_message(result, chinese=chinese))
    limitations.extend(
        str(gap.get("question") or gap.get("reason") or gap)
        for gap in result.get("unresolved_gaps", [])[:3]
    )
    if limitations:
        separator = "；" if chinese else "; "
        limitation_text = (
            "\n\n局限性方面，" if chinese else "\n\nRegarding limitations, "
        ) + separator.join(item.rstrip("。.") for item in limitations)
        limitation_text += "。" if chinese else "."
    else:
        limitation_text = ""
    return findings + limitation_text


def _deterministic_report_overview(
    results: list[DimensionResult], research_topic: str
) -> str:
    """Describe section coverage without adding unsupported factual content."""
    completed = sum(result.get("is_sufficient", False) for result in results)
    representative_claims = [
        claim
        for result in results
        for claim in result.get("claims", [])[:1]
    ]
    finding_text = " ".join(
        claim["claim"].strip().rstrip("。.")
        + " "
        + " ".join(
            f"[{source_id}]" for source_id in claim["supporting_source_ids"]
        )
        for claim in representative_claims
    )
    if _uses_chinese(research_topic):
        overview = (
            f"本报告基于经过审计的结论—证据记录，综合了 {len(results)} 个调研维度。"
            f"其中 {completed} 个维度达到配置的完成标准；其余证据限制在对应章节中披露。"
        )
        return overview + (f"综合现有证据，{finding_text}。" if finding_text else "")
    overview = (
        f"This report synthesizes {len(results)} research dimensions from audited "
        f"claim–evidence records. {completed} dimension(s) met the configured "
        "completion criteria; remaining limitations are disclosed in their sections."
    )
    return overview + (f" Taken together, {finding_text}." if finding_text else "")


def _deterministic_report_conclusion(
    results: list[DimensionResult], research_topic: str
) -> str:
    """Conclude safely from representative audited claims when generation fails."""
    representative_claims = [
        claim
        for result in results
        for claim in result.get("claims", [])[-1:]
    ]
    findings = " ".join(
        claim["claim"].strip().rstrip("。.")
        + " "
        + " ".join(
            f"[{source_id}]" for source_id in claim["supporting_source_ids"]
        )
        for claim in representative_claims
    )
    if _uses_chinese(research_topic):
        return (
            (f"综合各维度，{findings}。" if findings else "现有证据不足以形成综合结论。")
            + "这些结论仅以通过审计的证据为边界，尚未关闭的缺口应作为后续研究重点。"
        )
    return (
        (f"Across the dimensions, {findings}. " if findings else "The available evidence does not support an integrated conclusion. ")
        + "These conclusions remain bounded by the audited evidence, and unresolved gaps should guide further research."
    )


def _generate_report_section(
    result: DimensionResult,
    research_topic: str,
    model: str,
    research_run_id: str,
    conflicts: list[dict],
    report_plan: dict,
    previous_context: str = "None; this is the first body section.",
) -> str:
    """Generate one bounded section with compact retry and deterministic fallback."""
    dimension = result["dimension"]
    material = format_dimension_results(
        [result],
        max_claims_per_dimension=max(1, len(result.get("claims", []))),
        max_evidence_chars=220,
    )
    section_plan = next(
        (
            section
            for section in report_plan.get("sections", [])
            if str(section.get("dimension_id", "")) == str(dimension["id"])
        ),
        {},
    )

    def prompt_for(section_material: str) -> str:
        return report_section_instructions.format(
            research_topic=research_topic,
            dimension_title=dimension["title"],
            dimension_scope=dimension["scope"],
            report_plan=json.dumps(report_plan, ensure_ascii=False),
            section_plan=json.dumps(section_plan, ensure_ascii=False),
            previous_context=previous_context,
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


def _generate_report_conclusion(
    results: list[DimensionResult],
    research_topic: str,
    model: str,
    research_run_id: str,
    report_plan: dict,
) -> str:
    """Generate a bounded cross-dimension conclusion from audited claims."""
    material = format_dimension_results(
        results, max_claims_per_dimension=3, max_evidence_chars=100
    )
    prompt = report_conclusion_instructions.format(
        research_topic=research_topic,
        report_plan=json.dumps(report_plan, ensure_ascii=False),
        dimension_research=material,
    )
    try:
        conclusion = str(create_deepseek_model(model).invoke(prompt).content).strip()
    except LengthFinishReasonError:
        conclusion = ""
    if not conclusion:
        conclusion = _deterministic_report_conclusion(results, research_topic)
        emit_research_event(
            "report_conclusion_fallback",
            research_run_id=research_run_id,
            reason="length_limit_or_empty_response",
        )
    return conclusion


def _generate_sectioned_report(
    results: list[DimensionResult],
    research_topic: str,
    model: str,
    research_run_id: str,
    conflicts: list[dict],
    report_plan: dict,
) -> tuple[str, str, list[dict], str]:
    """Generate bounded semantic sections and merge them without an LLM call."""
    emit_research_event(
        "report_sectioning_started",
        research_run_id=research_run_id,
        dimension_count=len(results),
    )
    results = _order_results_by_report_plan(results, report_plan)
    sections = []
    previous_context = "None; this is the first body section."
    for result in results:
        content = _generate_report_section(
            result,
            research_topic,
            model,
            research_run_id,
            conflicts,
            report_plan,
            previous_context,
        )
        sections.append(
            {
                "dimension_id": result["dimension"]["id"],
                "title": result["dimension"]["title"],
                "content": content,
            }
        )
        completed_plan = next(
            (
                section
                for section in report_plan.get("sections", [])
                if str(section.get("dimension_id", ""))
                == str(result["dimension"]["id"])
            ),
            {},
        )
        previous_context = (
            f"Previous section: {result['dimension']['title']}. "
            f"Editorial objective: {completed_plan.get('objective', result['dimension']['scope'])}."
        )
    overview_material = format_dimension_results(
        results, max_claims_per_dimension=3, max_evidence_chars=100
    )
    overview_prompt = report_overview_instructions.format(
        research_topic=research_topic,
        report_plan=json.dumps(report_plan, ensure_ascii=False),
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
    conclusion = _generate_report_conclusion(
        results, research_topic, model, research_run_id, report_plan
    )
    report = _assemble_sectioned_report(
        research_topic, overview, sections, conclusion
    )
    emit_research_event(
        "report_draft_sectioned",
        research_run_id=research_run_id,
        section_count=len(sections),
        character_count=len(report),
    )
    return report, overview, sections, conclusion


def _assemble_sectioned_report(
    research_topic: str,
    overview: str,
    sections: list[dict],
    conclusion: str = "",
) -> str:
    """Merge independently generated report parts without another model call."""
    if _uses_chinese(research_topic):
        report_parts = ["# 调研报告", "## 执行摘要\n\n" + overview]
    else:
        report_parts = ["# Research Report", "## Executive Summary\n\n" + overview]
    report_parts.extend(
        f"## {section['title']}\n\n{section['content']}" for section in sections
    )
    if conclusion:
        heading = "结论" if _uses_chinese(research_topic) else "Conclusion"
        report_parts.append(f"## {heading}\n\n{conclusion}")
    return "\n\n".join(report_parts)


def draft_report(state: OverallState, config: RunnableConfig):
    """Draft the report from audited dimension claims."""
    configurable = Configuration.from_runnable_config(config)
    model = state.get("reasoning_model") or configurable.answer_model
    current_results, _, material = _report_research_material(state, configurable)
    conflicts = state.get("claim_conflicts", [])
    report_plan = state.get("report_plan") or _fallback_report_plan(
        current_results, state["normalized_research_topic"]
    )
    emit_research_event("drafting_report", research_run_id=state["research_run_id"])
    if _report_requires_sectioning(current_results, material, configurable):
        report_draft, overview, sections, conclusion = _generate_sectioned_report(
            current_results,
            state["normalized_research_topic"],
            model,
            state["research_run_id"],
            conflicts,
            report_plan,
        )
        return {
            "report_draft": report_draft,
            "report_generation_mode": "sectioned",
            "report_overview": overview,
            "report_sections": sections,
            "report_conclusion": conclusion,
            "report_revision_count": 0,
            "max_report_revisions": configurable.max_report_revisions,
        }
    prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=state["normalized_research_topic"],
        report_plan=json.dumps(report_plan, ensure_ascii=False),
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
        report_draft, overview, sections, conclusion = _generate_sectioned_report(
            current_results,
            state["normalized_research_topic"],
            model,
            state["research_run_id"],
            conflicts,
            report_plan,
        )
        return {
            "report_draft": report_draft,
            "report_generation_mode": "sectioned",
            "report_overview": overview,
            "report_sections": sections,
            "report_conclusion": conclusion,
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
        report_draft, overview, sections, conclusion = _generate_sectioned_report(
            current_results,
            state["normalized_research_topic"],
            model,
            state["research_run_id"],
            conflicts,
            report_plan,
        )
        return {
            "report_draft": report_draft,
            "report_generation_mode": "sectioned",
            "report_overview": overview,
            "report_sections": sections,
            "report_conclusion": conclusion,
            "report_revision_count": 0,
            "max_report_revisions": configurable.max_report_revisions,
        }
    return {
        "report_draft": report_draft,
        "report_generation_mode": "single_pass",
        "report_overview": "",
        "report_sections": [],
        "report_conclusion": "",
        "report_revision_count": 0,
        "max_report_revisions": configurable.max_report_revisions,
    }


def _report_article_style_findings(
    draft: str, research_topic: str = ""
) -> list[str]:
    """Detect deterministic signs that a draft is an evidence inventory, not prose."""
    lines = [line.strip() for line in draft.splitlines() if line.strip()]
    body_lines = [line for line in lines if not line.startswith("#")]
    bullet_lines = [
        line for line in body_lines if re.match(r"^(?:[-*+] |\d+[.)]\s+)", line)
    ]
    findings = []
    list_requested = any(
        cue in research_topic.casefold()
        for cue in (
            "list",
            "checklist",
            "ranking",
            "top ",
            "列表",
            "清单",
            "排名",
            "逐条",
        )
    )
    if (
        not list_requested
        and len(bullet_lines) >= 4
        and len(bullet_lines) / max(len(body_lines), 1) >= 0.3
    ):
        findings.append(
            "The draft is dominated by bullet-like claim enumeration instead of connected prose."
        )
    paragraphs = [
        re.sub(r"^#+\s*", "", paragraph.strip())
        for paragraph in re.split(r"\n\s*\n", draft)
        if paragraph.strip() and not paragraph.lstrip().startswith(("- ", "* ", "+ "))
    ]
    substantial_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if len(paragraph) >= 160 and "\n- " not in paragraph
    ]
    if len(draft) >= 1000 and len(substantial_paragraphs) < 3:
        findings.append(
            "The draft lacks enough developed prose paragraphs to form a coherent article."
        )
    repetitive_caveats = sum(
        draft.casefold().count(phrase)
        for phrase in (
            "current evidence",
            "available evidence",
            "现有证据",
            "现有材料",
            "证据不足",
        )
    )
    if repetitive_caveats >= 6:
        findings.append(
            "Evidence limitations are repeated excessively instead of being consolidated where they affect interpretation."
        )
    return findings


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
        report_plan=json.dumps(
            state.get("report_plan")
            or _fallback_report_plan(
                current_results, state["normalized_research_topic"]
            ),
            ensure_ascii=False,
        ),
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
    style_findings = _report_article_style_findings(
        state["report_draft"], state["normalized_research_topic"]
    )
    if style_findings:
        audit["passes"] = False
        audit["issues"] = [*audit["issues"], *style_findings]
        audit["revision_instructions"] = [
            *audit["revision_instructions"],
            "Rewrite list-like fragments as developed paragraphs that synthesize related claims under the report plan's thesis.",
            "Consolidate repetitive limitations and strengthen transitions between sections and paragraphs.",
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
    report_plan = state.get("report_plan") or _fallback_report_plan(results, topic)
    results = _order_results_by_report_plan(results, report_plan)
    overview = _deterministic_report_overview(results, topic)
    sections = [
        {
            "dimension_id": result["dimension"]["id"],
            "title": result["dimension"]["title"],
            "content": _deterministic_report_section(result, topic),
        }
        for result in results
    ]
    conclusion = _deterministic_report_conclusion(results, topic)
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
                f"{conflict['conflict_id']} 涉及两项不能同时采信的结论："
                if chinese
                else f"{conflict['conflict_id']} concerns two findings that cannot both be accepted: "
            )
            conflict_lines[-1] += (
                f"{left.get('claim', '')} {left_markers}；"
                f"{right.get('claim', '')} {right_markers}。"
                if chinese
                else f"{left.get('claim', '')} {left_markers}; "
                f"{right.get('claim', '')} {right_markers}. "
            ) + (
                "现有合格证据存在冲突，无法安全地选择单一结论。"
                if chinese
                else "Accepted evidence conflicts; no single conclusion can be selected safely."
            )
        sections.append(
            {
                "dimension_id": "conflicts",
                "title": "未解决的证据矛盾"
                if chinese
                else "Unresolved Evidence Conflicts",
                "content": "\n\n".join(conflict_lines),
            }
        )
    report = _assemble_sectioned_report(topic, overview, sections, conclusion)
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
        "report_conclusion": conclusion,
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
    report_plan = state.get("report_plan") or _fallback_report_plan(
        current_results, research_topic
    )
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
            report_plan=json.dumps(report_plan, ensure_ascii=False),
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
        report_plan=json.dumps(report_plan, ensure_ascii=False),
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
    current_conclusion = state.get("report_conclusion", "")
    conclusion_prompt = report_conclusion_revision_instructions.format(
        research_topic=research_topic,
        report_plan=json.dumps(report_plan, ensure_ascii=False),
        dimension_research=overview_material,
        current_conclusion=current_conclusion,
        audit_findings=audit_findings,
    )
    revised_conclusion = _bounded_report_part_revision(
        prompt=conclusion_prompt,
        fallback=current_conclusion
        or _deterministic_report_conclusion(current_results, research_topic),
        model=model,
        research_run_id=research_run_id,
        event_prefix="report_conclusion_revision",
    )
    revised_report = _assemble_sectioned_report(
        research_topic, revised_overview, revised_sections, revised_conclusion
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
        "report_conclusion": revised_conclusion,
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
        report_plan=json.dumps(
            state.get("report_plan")
            or _fallback_report_plan(
                current_results, state["normalized_research_topic"]
            ),
            ensure_ascii=False,
        ),
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
builder.add_node("generate_report_plan", generate_report_plan)
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
builder.add_edge("detect_claim_conflicts", "generate_report_plan")
builder.add_edge("generate_report_plan", "draft_report")
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
