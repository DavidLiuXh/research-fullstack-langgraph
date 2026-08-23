import importlib
import re

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage

from research_agent.graph import (
    analyze_research_topic,
    audit_report,
    dispatch_research_dimensions,
    draft_report,
    evaluate_sources,
    extract_claims,
    finalize_answer,
    generate_query,
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
    EvidenceQuote,
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


def _report_state(*, dimensions=2, claims_per_dimension=1, topic="Research topic"):
    dimension_results = []
    sources = []
    for dimension_index in range(dimensions):
        claims = []
        dimension_sources = []
        for claim_index in range(claims_per_dimension):
            source_id = f"S{dimension_index}-{claim_index}"
            source = {
                "research_run_id": "run",
                "source_id": source_id,
                "query": "query",
                "title": f"Source {source_id}",
                "url": f"https://example.com/{source_id}",
                "content": f"Verified evidence for claim {claim_index}.",
            }
            sources.append(source)
            dimension_sources.append(source)
            claims.append(
                {
                    "claim": f"Verified claim {dimension_index}-{claim_index}.",
                    "supporting_source_ids": [source_id],
                    "supporting_evidence": [
                        {
                            "source_id": source_id,
                            "quote": source["content"],
                            "locator": "chars:0-30",
                        }
                    ],
                    "contradicting_source_ids": [],
                    "contradicting_evidence": [],
                    "confidence": 0.9,
                    "uncertainty_reason": "",
                }
            )
        dimension_results.append(
            {
                "research_run_id": "run",
                "dimension": {
                    "id": str(dimension_index),
                    "title": f"Dimension {dimension_index}",
                    "scope": f"Scope {dimension_index}",
                },
                "research_content": "summary",
                "sources": dimension_sources,
                "research_loop_count": 1,
                "is_sufficient": True,
                "completion_status": "sufficient",
                "covered_questions": [],
                "unresolved_gaps": [],
                "contradictions": [],
                "source_quality_issues": [],
                "confidence": 0.9,
                "claims": claims,
            }
        )
    return {
        "normalized_research_topic": topic,
        "research_run_id": "run",
        "dimension_results": dimension_results,
        "sources_gathered": sources,
    }


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
                research_loop_count=3, completion_status="budget_exhausted"
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


def test_generated_queries_are_bound_to_prioritized_gaps(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeQueryModel:
        def with_structured_output(self, schema, method):
            assert schema is SearchQueryList
            return self

        def invoke(self, prompt):
            assert "gap-primary" in prompt
            return SearchQueryList(
                query=["official primary data", "academic comparison"],
                rationale="Close the two gaps.",
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FakeQueryModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    result = generate_query(
        _dimension_state(
            research_loop_count=1,
            reflection_assessment={
                "missing_questions": [
                    {
                        "gap_id": "gap-primary",
                        "question": "What does the primary source say?",
                        "reason": "Primary evidence is missing.",
                        "priority": "high",
                        "required_source_types": ["government"],
                        "expected_evidence": "Official data",
                        "suggested_query_focus": "official data",
                    },
                    {
                        "gap_id": "gap-comparison",
                        "question": "What does independent research report?",
                        "reason": "Independent comparison is missing.",
                        "priority": "medium",
                        "required_source_types": ["academic"],
                        "expected_evidence": "Comparative study",
                        "suggested_query_focus": "academic comparison",
                    },
                ]
            },
        ),
        {},
    )

    assert result["search_tasks"] == [
        {
            "query": "official primary data",
            "gap_id": "gap-primary",
            "requested_source_types": ["government"],
            "expected_evidence": "Official data",
        },
        {
            "query": "academic comparison",
            "gap_id": "gap-comparison",
            "requested_source_types": ["academic"],
            "expected_evidence": "Comparative study",
        },
    ]


def test_high_priority_gap_scheduling_rotates_between_loops():
    graph_module = importlib.import_module("research_agent.graph")
    gaps = [
        {
            "gap_id": gap_id,
            "question": gap_id,
            "reason": "missing",
            "priority": "high",
            "required_source_types": ["government"],
            "expected_evidence": "official evidence",
            "suggested_query_focus": gap_id,
        }
        for gap_id in ("gap-one", "gap-two", "gap-three")
    ]

    ordered = graph_module._prioritized_gaps(
        _dimension_state(
            research_loop_count=2,
            reflection_assessment={"missing_questions": gaps},
        )
    )

    assert [gap["gap_id"] for gap in ordered] == [
        "gap-two",
        "gap-three",
        "gap-one",
    ]


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
                source_evidence = list(
                    dict.fromkeys(
                        re.findall(
                            r"\[(S[A-Za-z0-9-]+)\][\s\S]*?Content: ([^\n]+)",
                            prompt,
                        )
                    )
                )
                return ClaimExtraction(
                    claims=[
                        EvidenceClaim(
                            claim=f"Supported evidence from {source_id}",
                            evidence=[
                                EvidenceQuote(source_id=source_id, quote=content)
                            ],
                            uncertainty="",
                        )
                        for source_id, content in source_evidence
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
    assert len(result["search_failures"]) == 1
    assert "Search failed" in result["web_research_result"][0]
    assert events[-1]["type"] == "search_failed"


def test_reflection_stops_when_search_is_unavailable(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class MissingEvidenceReflectionModel:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            return Reflection(
                is_sufficient=False,
                covered_questions=[],
                missing_questions=[],
                unsupported_claims=[],
                contradictions=[],
                source_quality_issues=["Search was unavailable."],
                recommended_search_strategy=[],
                do_not_repeat=[],
                completion_reason="No evidence could be retrieved.",
                confidence=0,
            )

    monkeypatch.setattr(
        graph_module,
        "create_deepseek_model",
        lambda *a, **k: MissingEvidenceReflectionModel(),
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    result = reflection(
        _dimension_state(
            selected_sources=[],
            search_failures=["query: usage limit exceeded"],
            search_success_count=0,
        ),
        {},
    )

    assert result["completion_status"] == "search_unavailable"


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


def test_source_selection_prioritizes_requested_authoritative_type(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeAssessmentModel:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            return SourceAssessmentList(
                assessments=[
                    SourceAssessment(
                        source_id="Sofficial",
                        source_type="government",
                        authority_score=0.75,
                        relevance_score=0.75,
                        recency_score=0.8,
                        is_primary_source=False,
                        is_likely_repost=False,
                        supported_topics=["official evidence"],
                        rejection_reasons=[],
                    ),
                    SourceAssessment(
                        source_id="Sblog",
                        source_type="blog",
                        authority_score=0.99,
                        relevance_score=0.99,
                        recency_score=0.9,
                        is_primary_source=False,
                        is_likely_repost=False,
                        supported_topics=["commentary"],
                        rejection_reasons=[],
                    ),
                ]
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: FakeAssessmentModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    common = {
        "research_run_id": "run",
        "query": "official evidence",
        "content": "A sufficiently complete evidence snippet.",
        "score": 0.9,
        "requested_source_types": ["government"],
    }
    result = evaluate_sources(
        _dimension_state(
            sources_gathered=[
                {
                    **common,
                    "source_id": "Sofficial",
                    "title": "Official source",
                    "url": "https://gov.example/evidence",
                },
                {
                    **common,
                    "source_id": "Sblog",
                    "title": "Blog source",
                    "url": "https://blog.example/evidence",
                },
            ]
        ),
        {
            "configurable": {
                "max_selected_sources_per_dimension": 1,
                "min_accepted_sources_per_dimension": 1,
                "min_authoritative_sources_per_dimension": 1,
                "min_primary_sources_per_dimension": 0,
            }
        },
    )

    assert [source["source_id"] for source in result["selected_sources"]] == [
        "Sofficial"
    ]
    assert result["selected_sources"][0]["matches_requested_source_type"] is True


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


def test_small_report_uses_single_pass_drafting(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class ShortReportModel:
        def invoke(self, prompt):
            return AIMessage(content="A concise report [S0-0].")

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: ShortReportModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)

    result = draft_report(
        _report_state(dimensions=1),
        {
            "configurable": {
                "report_sectioning_claim_threshold": 100,
                "report_sectioning_material_chars": 100000,
            }
        },
    )

    assert result["report_generation_mode"] == "single_pass"
    assert result["report_sections"] == []
    assert result["report_draft"] == "A concise report [S0-0]."


def test_large_report_is_generated_and_merged_by_dimension(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class SectionModel:
        prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            if "executive overview" in prompt:
                return AIMessage(content="Cross-dimension overview.")
            dimension = "0" if "Dimension 0" in prompt else "1"
            return AIMessage(content=f"Section {dimension} " + ("x" * 5000))

    model = SectionModel()
    monkeypatch.setattr(graph_module, "create_deepseek_model", lambda *a, **k: model)
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)

    result = draft_report(
        _report_state(dimensions=2, claims_per_dimension=2),
        {"configurable": {"report_sectioning_claim_threshold": 4}},
    )

    assert result["report_generation_mode"] == "sectioned"
    assert len(result["report_sections"]) == 2
    assert len(result["report_draft"]) > 10000
    assert "Section 0" in result["report_draft"]
    assert "Section 1" in result["report_draft"]
    assert len(model.prompts) == 3


def test_sectioned_generation_keeps_all_dimension_claims(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class CapturingModel:
        prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return AIMessage(content="Bounded report part.")

    model = CapturingModel()
    monkeypatch.setattr(graph_module, "create_deepseek_model", lambda *a, **k: model)
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)

    draft_report(
        _report_state(dimensions=1, claims_per_dimension=10),
        {"configurable": {"report_sectioning_claim_threshold": 4}},
    )

    assert "Verified claim 0-0." in model.prompts[0]
    assert "Verified claim 0-9." in model.prompts[0]


def test_length_limited_draft_switches_to_citation_safe_section_fallback(
    monkeypatch,
):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeLengthError(Exception):
        pass

    class LengthLimitedModel:
        def invoke(self, prompt):
            raise FakeLengthError("limit")

    events = []
    monkeypatch.setattr(graph_module, "LengthFinishReasonError", FakeLengthError)
    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: LengthLimitedModel()
    )
    monkeypatch.setattr(
        graph_module,
        "emit_research_event",
        lambda event_type, **data: events.append({"type": event_type, **data}),
    )

    result = draft_report(
        _report_state(dimensions=1, topic="中文调研主题"),
        {
            "configurable": {
                "report_sectioning_claim_threshold": 100,
                "report_sectioning_material_chars": 100000,
            }
        },
    )

    assert result["report_generation_mode"] == "sectioned"
    assert result["report_draft"].startswith("# 调研报告")
    assert "Verified claim 0-0. [S0-0]" in result["report_draft"]
    assert [event["type"] for event in events] == [
        "drafting_report",
        "report_draft_switching_to_sections",
        "report_sectioning_started",
        "report_section_started",
        "report_section_retry",
        "report_section_fallback",
        "report_section_completed",
        "report_overview_fallback",
        "report_draft_sectioned",
    ]


def test_sectioned_report_revision_preserves_long_report(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class SectionRevisionModel:
        def invoke(self, prompt):
            if "executive overview" in prompt:
                return AIMessage(content="Revised overview.")
            dimension = "0" if "Dimension 0" in prompt else "1"
            return AIMessage(content=f"Revised section {dimension} " + ("x" * 5000))

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: SectionRevisionModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    state = {
        **_report_state(dimensions=2, claims_per_dimension=2),
        "report_generation_mode": "sectioned",
        "report_overview": "Original overview.",
        "report_sections": [
            {"dimension_id": "0", "title": "Dimension 0", "content": "Old 0"},
            {"dimension_id": "1", "title": "Dimension 1", "content": "Old 1"},
        ],
        "report_draft": "Old complete report.",
        "report_audit": {
            "passes": False,
            "issues": ["Improve clarity."],
            "revision_instructions": ["Revise relevant sections."],
        },
        "report_revision_count": 0,
    }

    result = revise_report(state, {})

    assert result["report_generation_mode"] == "sectioned"
    assert result["report_overview"] == "Revised overview."
    assert len(result["report_draft"]) > 10000
    assert "Revised section 0" in result["report_draft"]
    assert "Revised section 1" in result["report_draft"]
    assert result["report_revision_count"] == 1


def test_length_limited_section_revisions_keep_existing_parts(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeLengthError(Exception):
        pass

    class LengthLimitedModel:
        def invoke(self, prompt):
            raise FakeLengthError("limit")

    events = []
    monkeypatch.setattr(graph_module, "LengthFinishReasonError", FakeLengthError)
    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: LengthLimitedModel()
    )
    monkeypatch.setattr(
        graph_module,
        "emit_research_event",
        lambda event_type, **data: events.append({"type": event_type, **data}),
    )
    state = {
        **_report_state(dimensions=1),
        "report_generation_mode": "sectioned",
        "report_overview": "Existing overview.",
        "report_sections": [
            {
                "dimension_id": "0",
                "title": "Dimension 0",
                "content": "Existing section [S0-0].",
            }
        ],
        "report_draft": "Existing report.",
        "report_audit": {
            "passes": False,
            "issues": ["Improve clarity."],
            "revision_instructions": ["Revise relevant sections."],
        },
        "report_revision_count": 0,
    }

    result = revise_report(state, {})

    assert "Existing overview." in result["report_draft"]
    assert "Existing section [S0-0]." in result["report_draft"]
    assert result["report_revision_count"] == 1
    assert [event["type"] for event in events] == [
        "report_section_revision_retry",
        "report_section_revision_skipped",
        "report_overview_revision_retry",
        "report_overview_revision_skipped",
        "report_sections_revised",
    ]


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


def test_report_audit_length_limit_uses_conservative_fallback(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class FakeLengthError(Exception):
        pass

    class LengthLimitedAuditModel:
        def with_structured_output(self, schema, method):
            assert schema is ReportAudit
            return self

        def invoke(self, prompt):
            raise FakeLengthError("limit")

    events = []
    monkeypatch.setattr(graph_module, "LengthFinishReasonError", FakeLengthError)
    monkeypatch.setattr(
        graph_module,
        "create_deepseek_model",
        lambda *a, **k: LengthLimitedAuditModel(),
    )
    monkeypatch.setattr(
        graph_module,
        "emit_research_event",
        lambda event_type, **data: events.append({"type": event_type, **data}),
    )
    state = {
        **_report_state(dimensions=1),
        "report_draft": "Verified claim [S0-0].",
        "report_revision_count": 1,
    }

    result = audit_report(state, {})

    assert result["report_audit"]["passes"] is False
    assert "could not complete" in result["report_audit"]["issues"][0]
    assert [event["type"] for event in events] == [
        "report_audit_fallback",
        "report_audit_completed",
    ]


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
            evidence_source_id_history=[["S1"]],
            evidence_gain_history=[{"total_gain": 0}],
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


def test_reflection_adds_quality_gaps_before_declaring_sufficient(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class PassingReflectionModel:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            return Reflection(
                is_sufficient=True,
                covered_questions=["Scope covered"],
                missing_questions=[],
                unsupported_claims=[],
                contradictions=[],
                source_quality_issues=[],
                recommended_search_strategy=[],
                do_not_repeat=[],
                completion_reason="Sufficient",
                confidence=0.9,
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: PassingReflectionModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    result = reflection(
        _dimension_state(
            research_loop_count=0,
            selected_sources=[
                {
                    "research_run_id": "run",
                    "source_id": "Ssecondary",
                    "query": "query",
                    "title": "Secondary source",
                    "url": "https://example.com",
                    "content": "Secondary evidence",
                    "quality_status": "supplementary",
                    "is_primary_source": False,
                    "is_authoritative_source": False,
                }
            ],
        ),
        {},
    )

    gap_ids = {
        gap["gap_id"] for gap in result["reflection_assessment"]["missing_questions"]
    }
    assert result["is_sufficient"] is False
    assert result["completion_status"] == "researching"
    assert {
        "quality-accepted-sources",
        "quality-authoritative-source",
        "quality-primary-source",
    }.issubset(gap_ids)


def test_reflection_synthesizes_gap_for_incomplete_empty_assessment(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class EmptyGapReflectionModel:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            return Reflection(
                is_sufficient=False,
                covered_questions=[],
                missing_questions=[],
                unsupported_claims=["A key claim remains unsupported."],
                contradictions=[],
                source_quality_issues=[],
                recommended_search_strategy=["Find direct official evidence."],
                do_not_repeat=[],
                completion_reason="A key claim remains unsupported.",
                confidence=0.4,
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: EmptyGapReflectionModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    selected = [
        {
            "research_run_id": "run",
            "source_id": source_id,
            "query": "query",
            "title": "Primary source",
            "url": f"https://example.com/{source_id}",
            "content": "Direct primary evidence.",
            "quality_status": "accepted",
            "is_primary_source": True,
            "is_authoritative_source": True,
        }
        for source_id in ("S1", "S2")
    ]
    result = reflection(
        _dimension_state(research_loop_count=0, selected_sources=selected), {}
    )

    gaps = result["reflection_assessment"]["missing_questions"]
    assert [gap["gap_id"] for gap in gaps] == ["reflection-unresolved-evidence"]
    assert gaps[0]["suggested_query_focus"] == "Find direct official evidence."


def test_reflection_keeps_prior_gap_until_explicitly_resolved(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class OptimisticReflectionModel:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            return Reflection(
                is_sufficient=True,
                covered_questions=["Some evidence was found."],
                resolved_gap_ids=[],
                missing_questions=[],
                unsupported_claims=[],
                contradictions=[],
                source_quality_issues=[],
                recommended_search_strategy=[],
                do_not_repeat=[],
                completion_reason="Sufficient",
                confidence=0.9,
            )

    monkeypatch.setattr(
        graph_module,
        "create_deepseek_model",
        lambda *a, **k: OptimisticReflectionModel(),
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    prior_gap = {
        "gap_id": "gap-unresolved",
        "question": "What does the regulator require?",
        "reason": "The legal requirement remains unknown.",
        "priority": "high",
        "required_source_types": ["government"],
        "expected_evidence": "The operative regulatory text.",
        "suggested_query_focus": "official regulation text",
    }
    selected = [
        {
            "research_run_id": "run",
            "source_id": source_id,
            "query": "query",
            "title": "Primary source",
            "url": f"https://example.com/{source_id}",
            "content": "Direct primary evidence.",
            "quality_status": "accepted",
            "is_primary_source": True,
            "is_authoritative_source": True,
        }
        for source_id in ("S1", "S2")
    ]
    result = reflection(
        _dimension_state(
            research_loop_count=1,
            selected_sources=selected,
            reflection_history=[{"missing_questions": [prior_gap]}],
            gap_registry={"gap-unresolved": prior_gap},
        ),
        {},
    )

    assert result["completion_status"] == "researching"
    assert result["is_sufficient"] is False
    assert result["reflection_assessment"]["missing_questions"] == [prior_gap]


def test_reflection_requires_requested_type_source_to_resolve_gap(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class ResolvedReflectionModel:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            return Reflection(
                is_sufficient=True,
                covered_questions=["Regulatory requirement found."],
                resolved_gap_ids=["gap-regulation"],
                missing_questions=[],
                unsupported_claims=[],
                contradictions=[],
                source_quality_issues=[],
                recommended_search_strategy=[],
                do_not_repeat=[],
                completion_reason="The official requirement was found.",
                confidence=0.9,
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: ResolvedReflectionModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    prior_gap = {
        "gap_id": "gap-regulation",
        "question": "What does the regulator require?",
        "reason": "Official text is required.",
        "priority": "high",
        "required_source_types": ["government"],
        "expected_evidence": "Operative regulatory text.",
        "suggested_query_focus": "official regulation",
    }

    def selected_sources(gap_ids):
        return [
            {
                "research_run_id": "run",
                "source_id": source_id,
                "query": "query",
                "title": "Official source",
                "url": f"https://example.com/{source_id}",
                "content": "Direct regulatory evidence.",
                "quality_status": "accepted",
                "source_type": "government",
                "is_primary_source": True,
                "is_authoritative_source": True,
                "gap_ids": gap_ids,
            }
            for source_id in ("S1", "S2")
        ]

    base_state = {
        "research_loop_count": 1,
        "reflection_history": [{"missing_questions": [prior_gap]}],
        "gap_registry": {"gap-regulation": prior_gap},
    }
    unmatched = reflection(
        _dimension_state(**base_state, selected_sources=selected_sources([])), {}
    )
    matched = reflection(
        _dimension_state(
            **base_state,
            selected_sources=selected_sources(["gap-regulation"]),
        ),
        {},
    )

    assert unmatched["completion_status"] == "researching"
    assert unmatched["resolved_gap_ids"] == []
    assert matched["completion_status"] == "sufficient"
    assert matched["resolved_gap_ids"] == ["gap-regulation"]
    assert matched["gap_source_coverage_ids"] == ["gap-regulation"]


def test_reflection_reopens_a_previously_resolved_gap(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class ReopenedGapModel:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            return Reflection(
                is_sufficient=False,
                covered_questions=[],
                resolved_gap_ids=[],
                missing_questions=[
                    {
                        "gap_id": "gap-regulation",
                        "question": "Which exceptions apply to the regulation?",
                        "reason": "New evidence indicates that exceptions may apply.",
                        "priority": "high",
                        "required_source_types": ["government"],
                        "expected_evidence": "The official exception clauses.",
                        "suggested_query_focus": "official regulation exceptions",
                    }
                ],
                unsupported_claims=[],
                contradictions=[],
                source_quality_issues=[],
                recommended_search_strategy=[],
                do_not_repeat=[],
                completion_reason="A resolved issue has reopened.",
                confidence=0.4,
            )

    monkeypatch.setattr(
        graph_module, "create_deepseek_model", lambda *a, **k: ReopenedGapModel()
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)
    selected = [
        {
            "research_run_id": "run",
            "source_id": source_id,
            "query": "query",
            "title": "Official source",
            "url": f"https://example.com/{source_id}",
            "content": "Direct regulatory evidence.",
            "quality_status": "accepted",
            "source_type": "government",
            "is_primary_source": True,
            "is_authoritative_source": True,
        }
        for source_id in ("S1", "S2")
    ]
    result = reflection(
        _dimension_state(
            research_loop_count=1,
            selected_sources=selected,
            resolved_gap_ids=["gap-regulation"],
            gap_registry={
                "gap-regulation": {
                    "gap_id": "gap-regulation",
                    "question": "What does the regulation require?",
                    "reason": "Official text is required.",
                    "priority": "high",
                    "required_source_types": ["government"],
                    "expected_evidence": "Operative regulatory text.",
                    "suggested_query_focus": "official regulation",
                }
            },
        ),
        {},
    )

    assert result["completion_status"] == "researching"
    assert result["resolved_gap_ids"] == []


def test_claim_extraction_skips_model_when_no_screened_evidence(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")
    monkeypatch.setattr(
        graph_module,
        "create_deepseek_model",
        lambda *a, **k: pytest.fail("The model must not run without evidence."),
    )
    monkeypatch.setattr(graph_module, "emit_research_event", lambda *a, **k: None)

    result = extract_claims(_dimension_state(selected_sources=[]), {})

    assert result["claims"] == []
    assert "no quality-screened evidence" in result["dimension_summary"]


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
    assert result.missing_questions[0].required_source_types == ["official_company"]
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
                        evidence=[
                            EvidenceQuote(source_id="Svalid", quote="Supported fact."),
                            EvidenceQuote(
                                source_id="Sinvented", quote="Invented evidence."
                            ),
                        ],
                        counter_evidence=[
                            EvidenceQuote(
                                source_id="Sinvented", quote="Invented evidence."
                            )
                        ],
                        uncertainty="",
                    ),
                    EvidenceClaim(
                        claim="Invented claim",
                        evidence=[
                            EvidenceQuote(
                                source_id="Sinvented", quote="Invented evidence."
                            )
                        ],
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
    assert result["claims"][0]["supporting_evidence"] == [
        {
            "source_id": "Svalid",
            "quote": "Supported fact.",
            "locator": "chars:0-15",
        }
    ]


def test_claim_extraction_repairs_legacy_source_id_only_output(monkeypatch):
    graph_module = importlib.import_module("research_agent.graph")

    class RepairingClaimModel:
        calls = 0

        def with_structured_output(self, schema, method):
            return self

        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return ClaimExtraction(
                    claims=[
                        EvidenceClaim(
                            claim="Supported claim",
                            source_ids=["Svalid"],
                        )
                    ],
                    summary="Initial summary",
                )
            assert "verbatim quote text" in prompt
            return ClaimExtraction(
                claims=[
                    EvidenceClaim(
                        claim="Supported claim",
                        evidence=[
                            EvidenceQuote(
                                source_id="Svalid",
                                quote="Supported fact from primary source.",
                            )
                        ],
                    )
                ],
                summary="Repaired summary",
            )

    events = []
    model = RepairingClaimModel()
    monkeypatch.setattr(graph_module, "create_deepseek_model", lambda *a, **k: model)
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
                    "source_id": "Svalid",
                    "query": "query",
                    "title": "Valid source",
                    "url": "https://example.com/valid",
                    "content": "Supported fact from primary source.",
                }
            ],
            reflection_assessment={},
        ),
        {},
    )

    assert model.calls == 2
    assert len(result["claims"]) == 1
    assert result["claims"][0]["supporting_source_ids"] == ["Svalid"]
    assert any(
        event["type"] == "claim_extraction_retry"
        and event["reason"] == "invalid_evidence_quotes"
        for event in events
    )


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
    monkeypatch.setattr(graph_module, "create_deepseek_model", lambda *a, **k: model)
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
