import importlib
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage

from research_agent.graph import (
    analyze_research_topic,
    audit_report,
    dispatch_research_dimensions,
    evaluate_sources,
    extract_claims,
    finalize_answer,
    graph,
    initialize_research_topic,
    reflection,
    request_topic_clarification,
    review_research_dimensions,
    revise_report,
    route_dimension_research,
    route_dimension_review,
    route_report_audit,
    route_topic_analysis,
    route_topic_clarification,
    web_research,
)
from research_agent.tools_and_schemas import (
    ClaimExtraction,
    EvidenceClaim,
    EvidenceConflict,
    Reflection,
    ReportAudit,
    ResearchDimension,
    ResearchDimensionList,
    ResearchGap,
    SearchQueryList,
    SourceAssessment,
    SourceAssessmentList,
    TopicClarificationAssessment,
)


def _dimension_state(**overrides):
    state = {
        "research_topic": "topic",
        "research_run_id": "run",
        "dimension": {"id": "0", "title": "Market", "scope": "scope"},
        "current_knowledge_gap": "missing evidence",
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "initial_search_query_count": 2,
        "max_research_loops": 3,
        "research_loop_count": 1,
        "is_sufficient": False,
        "completion_status": "researching",
    }
    state.update(overrides)
    return state


def test_dimension_gap_routes_back_to_query_generation():
    assert route_dimension_research(_dimension_state()) == "generate_query"


def test_sufficient_dimension_routes_to_end():
    assert (
        route_dimension_research(
            _dimension_state(is_sufficient=True, completion_status="sufficient")
        )
        == "extract_claims"
    )


def test_dimension_stops_at_loop_limit():
    assert (
        route_dimension_research(
            _dimension_state(
                research_loop_count=3, completion_status="loop_limit_reached"
            )
        )
        == "extract_claims"
    )


def test_parent_dispatches_isolated_dimension_inputs():
    state = {
        "messages": [],
        "normalized_research_topic": "Normalized topic",
        "research_run_id": "run",
        "research_dimensions": [
            {"id": "0", "title": "Market", "scope": "market scope"},
            {"id": "1", "title": "Technology", "scope": "tech scope"},
        ],
        "initial_search_query_count": 2,
        "max_research_loops": 3,
    }

    sends = dispatch_research_dimensions(state)

    assert len(sends) == 2
    assert sends[0].arg["dimension"]["id"] == "0"
    assert sends[1].arg["dimension"]["id"] == "1"
    assert sends[0].arg is not sends[1].arg
    assert sends[0].arg["research_topic"] == "Normalized topic"


def test_initialize_research_topic_resets_previous_clarification_state():
    result = initialize_research_topic(
        {
            "messages": [HumanMessage(content="Research Apple")],
            "topic_clarification_history": [
                {"questions": ["Old question"], "response": "Old response"}
            ],
        }
    )

    assert result["original_research_topic"] == "Research Apple"
    assert result["normalized_research_topic"] == "Research Apple"
    assert result["topic_clarification_history"] == []


def test_clear_topic_routes_to_dimension_generation():
    assert (
        route_topic_analysis({"topic_needs_clarification": False})
        == "generate_research_dimensions"
    )


def test_ambiguous_topic_routes_to_human_clarification():
    assert (
        route_topic_analysis({"topic_needs_clarification": True})
        == "request_topic_clarification"
    )


def test_clarification_response_is_recorded_and_reanalyzed(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda value: {"action": "clarify", "response": "Apple Inc."},
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    state = {
        "topic_ambiguities": ["Apple is ambiguous"],
        "topic_clarification_questions": ["Company or fruit?"],
        "topic_assumptions": ["Assume Apple Inc."],
        "topic_clarification_reason": "Two interpretations",
        "topic_clarification_history": [],
    }

    result = request_topic_clarification(state)

    assert result["topic_clarification_action"] == "clarify"
    assert result["topic_clarification_history"] == [
        {"questions": ["Company or fruit?"], "response": "Apple Inc."}
    ]
    assert route_topic_clarification(result) == "analyze_research_topic"


def test_accepting_assumptions_continues_to_dimension_generation(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda value: {"action": "accept_assumptions"},
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    result = request_topic_clarification(
        {
            "topic_ambiguities": ["Time range is unclear"],
            "topic_clarification_questions": ["Which time range?"],
            "topic_assumptions": ["Use the past 12 months."],
            "topic_clarification_reason": "Time range changes the evidence",
            "topic_clarification_history": [],
        }
    )

    assert result["topic_needs_clarification"] is False
    assert route_topic_clarification(result) == "generate_research_dimensions"


def test_topic_analysis_uses_clarification_history(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    captured_prompts = []

    class FakeAssessmentModel:
        def with_structured_output(self, schema, method):
            assert schema is TopicClarificationAssessment
            assert method == "json_mode"
            return self

        def invoke(self, prompt):
            captured_prompts.append(prompt)
            return TopicClarificationAssessment(
                needs_clarification=False,
                ambiguities=[],
                clarification_questions=[],
                assumptions=[],
                normalized_topic="Research Apple Inc. globally.",
                reason="The subject is now clear.",
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FakeAssessmentModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    result = analyze_research_topic(
        {
            "original_research_topic": "Research Apple",
            "topic_clarification_history": [
                {"questions": ["Company or fruit?"], "response": "Apple Inc."}
            ],
        },
        {},
    )

    assert result["topic_needs_clarification"] is False
    assert result["normalized_research_topic"] == "Research Apple Inc. globally."
    assert "Apple Inc." in captured_prompts[0]


def test_dimension_review_rejection_requires_regeneration(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda value: {"approved": False, "feedback": "Add regulation"},
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)

    result = review_research_dimensions(
        {
            "research_run_id": "run",
            "research_dimensions": [
                {"id": "0", "title": "Market", "scope": "market scope"}
            ],
        }
    )

    assert result == {
        "dimension_approved": False,
        "dimension_feedback": "Add regulation",
    }
    assert (
        route_dimension_review({"dimension_approved": False})
        == "generate_research_dimensions"
    )


def test_dimension_review_approval_dispatches_research(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda value: {"approved": True, "feedback": ""},
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    state = {
        "messages": [],
        "normalized_research_topic": "Normalized topic",
        "research_run_id": "run",
        "research_dimensions": [
            {"id": "0", "title": "Market", "scope": "market scope"}
        ],
        "initial_search_query_count": 1,
        "max_research_loops": 1,
    }

    result = review_research_dimensions(state)
    routed = route_dimension_review({**state, **result})

    assert result["dimension_approved"] is True
    assert len(routed) == 1
    assert routed[0].node == "research_dimension"


def test_compiled_parent_graph_has_dimension_pipeline():
    assert graph.name == "deepseek-tavily-multidimensional-research-agent"
    assert {
        "generate_research_dimensions",
        "initialize_research_topic",
        "analyze_research_topic",
        "request_topic_clarification",
        "review_research_dimensions",
        "research_dimension",
        "draft_report",
        "audit_report",
        "revise_report",
        "finalize_answer",
    }.issubset(graph.nodes)


def test_parent_graph_runs_parallel_dimension_subgraphs(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeModel:
        schema = None

        def with_structured_output(self, schema, method):
            assert method == "json_mode"
            self.schema = schema
            return self

        def invoke(self, prompt):
            if self.schema is TopicClarificationAssessment:
                return TopicClarificationAssessment(
                    needs_clarification=False,
                    ambiguities=[],
                    clarification_questions=[],
                    assumptions=[],
                    normalized_topic="Research this topic",
                    reason="Clear",
                )
            if self.schema is ResearchDimensionList:
                return ResearchDimensionList(
                    dimensions=[
                        ResearchDimension(title="Market", scope="market scope"),
                        ResearchDimension(title="Technology", scope="tech scope"),
                    ]
                )
            if self.schema is SearchQueryList:
                query = "market query" if "Market" in prompt else "technology query"
                return SearchQueryList(
                    query=[query, f"{query} official"], rationale="test"
                )
            if self.schema is SourceAssessmentList:
                source_ids = list(
                    dict.fromkeys(re.findall(r"\[(S[A-Za-z0-9-]+)\]", prompt))
                )
                return SourceAssessmentList(
                    assessments=[
                        SourceAssessment(
                            source_id=source_id,
                            source_type="research_institute",
                            authority_score=0.9,
                            relevance_score=0.95,
                            recency_score=0.8,
                            is_primary_source=True,
                            is_likely_repost=False,
                            supported_topics=["test evidence"],
                            rejection_reasons=[],
                        )
                        for source_id in source_ids
                    ]
                )
            if self.schema is Reflection:
                return Reflection(
                    is_sufficient=True,
                    covered_questions=["The dimension scope"],
                    missing_questions=[],
                    unsupported_claims=[],
                    contradictions=[],
                    source_quality_issues=[],
                    recommended_search_strategy=[],
                    do_not_repeat=[],
                    completion_reason="The evidence is sufficient.",
                    confidence=0.9,
                )
            if self.schema is ClaimExtraction:
                source_ids = list(
                    dict.fromkeys(re.findall(r"\[(S[A-Za-z0-9-]+)\]", prompt))
                )
                return ClaimExtraction(
                    claims=[
                        EvidenceClaim(
                            claim=f"Supported evidence from {source_id}",
                            source_ids=[source_id],
                            counter_source_ids=[],
                            uncertainty="",
                        )
                        for source_id in source_ids
                    ],
                    summary="Supported dimension summary.",
                )
            if self.schema is ReportAudit:
                return ReportAudit(
                    passes=True,
                    issues=[],
                    revision_instructions=[],
                )
            source_ids = list(
                dict.fromkeys(re.findall(r"\[(S[A-Za-z0-9-]+)\]", prompt))
            )
            return AIMessage(
                content=" ".join(f"Evidence [{item}]." for item in source_ids)
            )

    class FakeTavilyClient:
        def __init__(self, api_key):
            assert api_key == "test-tavily-key"

        def search(self, query, **kwargs):
            slug = query.replace(" ", "-")
            return {
                "results": [
                    {
                        "title": query.title(),
                        "url": f"https://example.com/{slug}",
                        "content": f"Evidence for {query}",
                        "score": 0.9,
                    }
                ]
            }

    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FakeModel()
    )
    monkeypatch.setattr(graph_module, "TavilyClient", FakeTavilyClient)
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda value: {"approved": True, "feedback": ""},
    )

    graph_input = {
        "messages": [HumanMessage(content="Research this topic")],
        "initial_search_query_count": 2,
        "max_research_loops": 1,
        "reasoning_model": "deepseek-v4-pro",
    }
    result = graph.invoke(graph_input)

    assert len(result["dimension_results"]) == 2
    assert {item["dimension"]["title"] for item in result["dimension_results"]} == {
        "Market",
        "Technology",
    }
    assert "https://example.com/market-query" in result["messages"][-1].content
    assert "https://example.com/market-query-official" in result["messages"][-1].content
    assert "https://example.com/technology-query" in result["messages"][-1].content
    assert (
        "https://example.com/technology-query-official"
        in result["messages"][-1].content
    )

    custom_events = list(graph.stream(graph_input, stream_mode="custom"))
    event_types = {event["type"] for event in custom_events}
    assert sum(event["type"] == "sources_evaluated" for event in custom_events) == 2
    assert {
        "planning_dimensions",
        "topic_analyzed",
        "dimensions_created",
        "dimensions_reviewed",
        "queries_generated",
        "search_started",
        "search_completed",
        "sources_evaluated",
        "reflection_completed",
        "claims_extracted",
        "dimension_completed",
        "drafting_report",
        "report_audit_completed",
        "finalizing_answer",
    }.issubset(event_types)


def test_tavily_failure_degrades_to_empty_evidence(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FailingTavilyClient:
        def __init__(self, api_key):
            pass

        def search(self, query, **kwargs):
            raise TimeoutError("Tavily timed out")

    events = []
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(graph_module, "TavilyClient", FailingTavilyClient)
    monkeypatch.setattr(
        graph_module,
        "emit_research_event",
        lambda event_type, **data: events.append({"type": event_type, **data}),
    )

    result = web_research(
        {
            "research_run_id": "run",
            "search_query": "unavailable query",
            "search_id": "run-0-0-0",
        },
        {"configurable": {"tavily_max_retries": 0}},
    )

    assert result["sources_gathered"] == []
    assert "Search failed" in result["web_research_result"][0]
    assert events[-1]["type"] == "search_failed"


def test_final_answer_uses_only_current_research_run(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)

    def dimension_result(run_id, content):
        return {
            "research_run_id": run_id,
            "dimension": {"id": "0", "title": "Market", "scope": "scope"},
            "research_content": content,
            "sources": [],
            "research_loop_count": 1,
            "is_sufficient": True,
            "claims": [
                {
                    "claim": "Current fact",
                    "supporting_source_ids": ["Snew-0-0-0-0"],
                    "supporting_evidence": content,
                    "contradicting_source_ids": [],
                    "confidence": 0.9,
                    "uncertainty_reason": "",
                }
            ],
        }

    def source(run_id, source_id, content):
        return {
            "research_run_id": run_id,
            "source_id": source_id,
            "query": "query",
            "title": f"{run_id} source",
            "url": f"https://example.com/{run_id}",
            "content": content,
        }

    result = finalize_answer(
        {
            "messages": [HumanMessage(content="Current question")],
            "normalized_research_topic": "Current normalized question",
            "research_run_id": "new",
            "dimension_results": [
                dimension_result("old", "OLD DIMENSION CONTENT"),
                dimension_result("new", "NEW DIMENSION CONTENT"),
            ],
            "sources_gathered": [
                source("old", "Sold-0-0-0-0", "OLD SOURCE CONTENT"),
                source("new", "Snew-0-0-0-0", "NEW SOURCE CONTENT"),
            ],
            "reasoning_model": "deepseek-v4-pro",
            "report_draft": ("Current fact [Snew-0-0-0-0]. Old marker [Sold-0-0-0-0]."),
        },
    )

    assert "https://example.com/new" in result["messages"][0].content
    assert "https://example.com/old" not in result["messages"][0].content


def test_source_quality_selection_rejects_unassessed_and_excess_domain_sources(
    monkeypatch,
):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeAssessmentModel:
        def with_structured_output(self, schema, method):
            assert schema is SourceAssessmentList
            assert method == "json_mode"
            return self

        def invoke(self, prompt):
            return SourceAssessmentList(
                assessments=[
                    SourceAssessment(
                        source_id=f"S{index}",
                        source_type="government",
                        authority_score=0.9,
                        relevance_score=0.9,
                        recency_score=0.9,
                        is_primary_source=True,
                        is_likely_repost=False,
                        supported_topics=["market"],
                        rejection_reasons=[],
                    )
                    for index in range(3)
                ]
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FakeAssessmentModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    sources = [
        {
            "research_run_id": "run",
            "source_id": f"S{index}",
            "query": "query",
            "title": f"Unique title {index}",
            "url": f"https://example.com/{index}",
            "content": "evidence",
            "score": 0.9,
        }
        for index in range(3)
    ]
    sources.append(
        {
            "research_run_id": "run",
            "source_id": "Sunassessed",
            "query": "query",
            "title": "Unassessed source",
            "url": "https://other.example/unassessed",
            "content": "evidence",
            "score": 0.9,
        }
    )

    result = evaluate_sources(_dimension_state(sources_gathered=sources), {})

    assert len(result["selected_sources"]) == 2
    assert {item["source_id"] for item in result["rejected_sources"]} == {
        "S2",
        "Sunassessed",
    }


def test_compact_source_assessment_response_is_conservatively_normalized():
    result = SourceAssessmentList.model_validate(
        {"assessments": {"S1": 0.9, "S2": 0.2}}
    )

    assert [assessment.source_id for assessment in result.assessments] == ["S1", "S2"]
    assert result.assessments[0].source_type == "unknown"
    assert result.assessments[0].relevance_score == 0.9
    assert result.assessments[0].authority_score == 0.35
    assert result.assessments[0].rejection_reasons


def test_source_evaluation_parser_failure_uses_conservative_fallback(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FailingAssessmentModel:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            raise OutputParserException("invalid source assessment")

    events = []
    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FailingAssessmentModel()
    )
    monkeypatch.setattr(
        graph_module,
        "emit_research_event",
        lambda event_type, **data: events.append({"type": event_type, **data}),
    )
    result = evaluate_sources(
        _dimension_state(
            sources_gathered=[
                {
                    "research_run_id": "run",
                    "source_id": "Sfallback",
                    "query": "query",
                    "title": "Fallback source",
                    "url": "https://example.com/fallback",
                    "content": "evidence",
                    "score": 0.9,
                }
            ]
        ),
        {},
    )

    assert result["selected_sources"][0]["source_id"] == "Sfallback"
    assert result["selected_sources"][0]["quality_status"] == "supplementary"
    assert events[0]["type"] == "source_evaluation_fallback"


def test_report_audit_route_is_bounded():
    assert (
        route_report_audit(
            {"report_audit": {"passes": True}, "report_revision_count": 0}
        )
        == "finalize_answer"
    )
    assert (
        route_report_audit(
            {
                "report_audit": {"passes": False},
                "report_revision_count": 0,
                "max_report_revisions": 2,
            }
        )
        == "revise_report"
    )
    assert (
        route_report_audit(
            {
                "report_audit": {"passes": False},
                "report_revision_count": 2,
                "max_report_revisions": 2,
            }
        )
        == "finalize_answer"
    )


def test_report_revision_keeps_draft_when_both_attempts_hit_length_limit(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class TestLengthError(Exception):
        pass

    class LengthLimitedModel:
        def invoke(self, prompt):
            raise TestLengthError("limit")

    events = []
    monkeypatch.setattr(graph_module, "LengthFinishReasonError", TestLengthError)
    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: LengthLimitedModel()
    )
    monkeypatch.setattr(
        graph_module,
        "emit_research_event",
        lambda event_type, **data: events.append({"type": event_type, **data}),
    )
    result = revise_report(
        {
            "normalized_research_topic": "Current topic",
            "research_run_id": "run",
            "dimension_results": [],
            "sources_gathered": [],
            "report_draft": "Existing audited draft [S1].",
            "report_audit": {
                "passes": False,
                "issues": ["Tighten wording."],
                "revision_instructions": ["Be concise."],
            },
            "report_revision_count": 0,
        },
        {},
    )

    assert result["report_draft"] == "Existing audited draft [S1]."
    assert result["report_revision_count"] == 1
    assert [event["type"] for event in events] == [
        "report_revision_retry",
        "report_revision_skipped",
    ]


def test_report_audit_rejects_unknown_source_markers(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class PassingAuditModel:
        def with_structured_output(self, schema, method):
            assert schema is ReportAudit
            return self

        def invoke(self, prompt):
            return ReportAudit(
                passes=True,
                issues=[],
                revision_instructions=[],
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: PassingAuditModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    result = audit_report(
        {
            "normalized_research_topic": "Current topic",
            "research_run_id": "run",
            "dimension_results": [
                {
                    "research_run_id": "run",
                    "dimension": {"id": "0", "title": "Market", "scope": "scope"},
                    "research_content": "summary",
                    "sources": [],
                    "research_loop_count": 1,
                    "is_sufficient": True,
                }
            ],
            "sources_gathered": [],
            "report_draft": "Unsupported marker [Sinvented].",
            "report_revision_count": 0,
        },
        {},
    )

    assert result["report_audit"]["passes"] is False
    assert "Sinvented" in result["report_audit"]["issues"][0]


def test_report_audit_schema_recovers_wrapped_alternative_fields():
    audit = ReportAudit.model_validate(
        {
            "audit": {
                "draft_answers_material_parts": True,
                "factual_statements_without_support": [
                    {
                        "statement": "Unsupported statement",
                        "explanation": "No source supports it.",
                    }
                ],
                "overstated_claims": [
                    {"claim": "Overstated claim", "explanation": "Too definitive."}
                ],
                "citation_marker_issues": [
                    {"issue": "Unknown marker", "explanation": "It is absent."}
                ],
                "contradictions_counterarguments_uncertainty": [
                    {
                        "issue": "Missing counterargument about authorization",
                        "explanation": "Authorized use may be viable.",
                    }
                ],
                "structure_duplication_clarity": [
                    {"issue": "Duplicated section", "explanation": "Tighten it."}
                ],
                "pass": False,
                "required_corrections": ["Correct and qualify the findings."],
            }
        }
    )

    assert audit.passes is False
    assert len(audit.issues) == 5
    assert "Unknown marker: It is absent." in audit.issues
    assert "Duplicated section: Tighten it." in audit.issues
    assert audit.revision_instructions == ["Correct and qualify the findings."]


def test_reflection_detects_stalled_repeated_gap(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeReflectionModel:
        def with_structured_output(self, schema, method):
            assert schema is Reflection
            return self

        def invoke(self, prompt):
            return Reflection(
                is_sufficient=False,
                covered_questions=[],
                missing_questions=[
                    ResearchGap(
                        question="What is the official market size?",
                        reason="No official data was found.",
                        priority="high",
                        required_source_types=["government"],
                        suggested_query_focus="Find official statistics.",
                    )
                ],
                unsupported_claims=[],
                contradictions=[],
                source_quality_issues=["Only secondary sources were found."],
                recommended_search_strategy=["Search official statistics."],
                do_not_repeat=["generic market size"],
                completion_reason="A material gap remains.",
                confidence=0.3,
            )

    prior = {"missing_questions": [{"question": "What is the official market size?"}]}
    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FakeReflectionModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)

    result = reflection(
        _dimension_state(
            reflection_history=[prior],
            evidence_source_count_history=[1],
            selected_sources=[
                {
                    "research_run_id": "run",
                    "source_id": "S1",
                    "query": "query",
                    "title": "Source",
                    "url": "https://example.com",
                    "content": "evidence",
                }
            ],
            query_history=["generic market size"],
        ),
        {},
    )

    assert result["completion_status"] == "stalled"
    assert result["is_sufficient"] is False


def test_compact_reflection_response_normalizes_provider_aliases():
    result = Reflection.model_validate(
        {
            "dimension": "Technical feasibility",
            "sufficient": False,
            "missing_questions": [
                {
                    "question": "What is the tool maintenance status?",
                    "impact": "high",
                    "source_types": ["official documentation", "GitHub"],
                    "search_focus": "tool maintenance status 2026",
                }
            ],
            "do_not_repeat": ["generic tool overview"],
        }
    )

    assert result.is_sufficient is False
    assert result.missing_questions[0].priority == "high"
    assert result.missing_questions[0].required_source_types == [
        "official documentation",
        "GitHub",
    ]
    assert (
        result.missing_questions[0].suggested_query_focus
        == "tool maintenance status 2026"
    )
    assert result.recommended_search_strategy == ["tool maintenance status 2026"]


def test_reflection_parser_failure_requests_conservative_follow_up(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FailingReflectionModel:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            raise OutputParserException("invalid reflection")

    events = []
    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FailingReflectionModel()
    )
    monkeypatch.setattr(
        graph_module,
        "emit_research_event",
        lambda event_type, **data: events.append({"type": event_type, **data}),
    )
    result = reflection(
        _dimension_state(
            selected_sources=[],
            query_history=["previous query"],
            reflection_history=[],
            evidence_source_count_history=[],
        ),
        {},
    )

    assert result["is_sufficient"] is False
    assert result["completion_status"] == "researching"
    assert result["reflection_assessment"]["missing_questions"][0]["priority"] == "high"
    assert events[0]["type"] == "reflection_fallback"


def test_claim_extraction_drops_unknown_source_ids(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeClaimModel:
        def with_structured_output(self, schema, method):
            assert schema is ClaimExtraction
            return self

        def invoke(self, prompt):
            return ClaimExtraction(
                claims=[
                    EvidenceClaim(
                        claim="Valid claim",
                        source_ids=["Svalid", "Sinvented"],
                        counter_source_ids=["Sinvented"],
                        uncertainty="",
                    ),
                    EvidenceClaim(
                        claim="Invented claim",
                        source_ids=["Sinvented"],
                        counter_source_ids=[],
                        uncertainty="",
                    ),
                ],
                summary="Summary",
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FakeClaimModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    result = extract_claims(
        _dimension_state(
            selected_sources=[
                {
                    "research_run_id": "run",
                    "source_id": "Svalid",
                    "query": "query",
                    "title": "Valid source",
                    "url": "https://example.com/valid",
                    "content": "Supported fact.",
                }
            ],
            reflection_assessment={},
        ),
        {},
    )

    assert len(result["claims"]) == 1
    assert result["claims"][0]["supporting_source_ids"] == ["Svalid"]
    assert result["claims"][0]["contradicting_source_ids"] == []


def test_claim_extraction_retries_with_compact_evidence_after_length_limit(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeLengthError(Exception):
        pass

    class FakeClaimModel:
        prompts = []

        def with_structured_output(self, schema, method):
            assert schema is ClaimExtraction
            return self

        def invoke(self, prompt):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                raise FakeLengthError()
            return ClaimExtraction(claims=[], summary="Compact summary")

    events = []
    model = FakeClaimModel()
    monkeypatch.setattr(graph_module, "LengthFinishReasonError", FakeLengthError)
    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: model
    )
    monkeypatch.setattr(
        graph_module,
        "emit_research_event",
        lambda event_type, **data: events.append({"type": event_type, **data}),
    )
    result = extract_claims(
        _dimension_state(
            selected_sources=[
                {
                    "research_run_id": "run",
                    "source_id": f"S{index}",
                    "query": "query",
                    "title": f"Source {index}",
                    "url": f"https://example.com/{index}",
                    "content": f"MARKER-{index} " + ("evidence " * 1000),
                }
                for index in range(8)
            ],
            reflection_assessment={},
        ),
        {},
    )

    assert result["dimension_summary"] == "Compact summary"
    assert len(model.prompts) == 2
    assert "MARKER-5" in model.prompts[1]
    assert "MARKER-6" not in model.prompts[1]
    assert len(model.prompts[1]) < len(model.prompts[0])
    assert events[0]["type"] == "claim_extraction_retry"
    assert events[0]["reason"] == "length_limit"


def test_claim_schema_recovers_source_ids_misplaced_by_model():
    result = ClaimExtraction.model_validate(
        {
            "claims": [
                {
                    "claim": "A supported claim",
                    "supporting_evidence": ["S1", "S2"],
                }
            ],
            "dimension_summary": "Summary",
        }
    )

    claim = result.claims[0]
    assert claim.source_ids == ["S1", "S2"]
    assert claim.counter_source_ids == []
    assert claim.uncertainty == ""
    assert result.summary == "Summary"


def test_reflection_schema_rejects_unresolved_sufficient_evidence():
    try:
        Reflection(
            is_sufficient=True,
            covered_questions=[],
            missing_questions=[
                ResearchGap(
                    question="Missing fact",
                    reason="It changes the conclusion.",
                    priority="high",
                    required_source_types=["government"],
                    suggested_query_focus="Find the fact.",
                )
            ],
            unsupported_claims=[],
            contradictions=[
                EvidenceConflict(
                    description="Material conflict",
                    source_ids=["S1", "S2"],
                    requires_follow_up=True,
                )
            ],
            source_quality_issues=[],
            recommended_search_strategy=[],
            do_not_repeat=[],
            completion_reason="Incorrectly sufficient.",
            confidence=0.9,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Inconsistent sufficient reflection was accepted")
