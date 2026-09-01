"""Tests for evaluation adapters, judges, aggregation, and reports."""

import asyncio
from types import SimpleNamespace

import pytest

from research_agent.evals.evaluators.deterministic import (
    evaluate_deterministic_quality,
)
from research_agent.evals.evaluators.groundedness import (
    DeepSeekGroundednessEvaluator,
)
from research_agent.evals.evaluators.report_quality import (
    DeepSeekReportQualityEvaluator,
)
from research_agent.evals.run_evaluate import (
    aggregate_results,
    evaluate_example,
    render_markdown,
    validate_args,
)
from research_agent.evals.schemas import GroundednessAssessment, ReportQualityScores
from research_agent.evals.target import ResearchEvaluationTarget, _interrupt_decision
from research_agent.tools_and_schemas import SearchQueryList


def _scores_by_key(scores):
    return {score["key"]: score["score"] for score in scores}


def test_interrupt_decisions_cover_both_human_gates():
    assert _interrupt_decision({"type": "research_topic_clarification"}) == {
        "action": "accept_assumptions"
    }
    assert _interrupt_decision({"type": "research_dimension_review"}) == {
        "approved": True
    }
    with pytest.raises(ValueError, match="Unsupported graph interrupt type"):
        _interrupt_decision({"type": "unexpected"})


def test_single_search_query_string_is_normalized_for_bounded_evaluations():
    result = SearchQueryList.model_validate(
        {"query": "single focused query", "rationale": "Only one was requested."}
    )
    assert result.query == ["single focused query"]


def test_deterministic_evaluator_scores_valid_evidence_and_trajectory():
    required_events = [
        "planning_dimensions",
        "initial_gaps_planned",
        "gap_selected",
        "queries_generated",
        "search_completed",
        "sources_evaluated",
        "gap_evidence_assessed",
        "gap_status_updated",
        "reflection_completed",
        "claims_extracted",
        "report_evidence_prepared",
        "claim_conflicts_detected",
        "drafting_report",
        "report_audit_completed",
        "report_consistency_audited",
        "finalizing_answer",
    ]
    outputs = {
        "report_draft": "A supported statement [S1].",
        "sources": [
            {
                "source_id": "S1",
                "quality_status": "accepted",
                "evidence_score": 0.9,
                "domain": "example.com",
                "content": "A directly supported statement from the source.",
                "is_primary_source": True,
                "is_authoritative_source": True,
                "source_type": "government",
            },
            {
                "source_id": "S2",
                "quality_status": "accepted",
                "evidence_score": 0.9,
                "domain": "academic.example",
                "content": "Independent academic evidence.",
                "source_type": "academic",
            },
            {
                "source_id": "S3",
                "quality_status": "accepted",
                "evidence_score": 0.9,
                "domain": "standards.example",
                "content": "Independent standards evidence.",
                "source_type": "standards_body",
            },
        ],
        "dimension_results": [
            {
                "is_sufficient": True,
                "sources": [
                    {
                        "source_id": "S1",
                        "quality_status": "accepted",
                        "is_primary_source": True,
                        "is_authoritative_source": True,
                        "domain": "example.com",
                        "source_type": "government",
                    },
                    {
                        "source_id": "S2",
                        "quality_status": "accepted",
                        "domain": "academic.example",
                        "source_type": "academic",
                    },
                ],
                "known_gap_count": 1,
                "resolved_gap_count": 1,
                "high_priority_gap_count": 1,
                "resolved_high_priority_gap_count": 1,
                "high_priority_gap_source_coverage_count": 1,
                "evidence_gain_history": [{"total_gain": 1}],
                "direct_evidence_gap_count": 1,
                "supported_claim_gap_count": 1,
                "requested_type_gap_count": 1,
                "independent_source_gap_count": 1,
                "gap_assessment_failure_count": 0,
                "completion_status": "sufficient",
                "claims": [
                    {
                        "supporting_source_ids": ["S1"],
                        "supporting_evidence": [
                            {
                                "source_id": "S1",
                                "quote": "A directly supported statement from the source.",
                                "locator": "chars:0-47",
                            }
                        ],
                        "contradicting_source_ids": [],
                    }
                ],
            }
        ],
        "custom_events": [{"type": event} for event in required_events],
        "node_trajectory": [],
        "report_revision_count": 1,
        "max_report_revisions": 1,
        "consistency_analysis_complete": True,
        "claim_conflicts": [],
        "report_consistency_audit": {"omitted_conflict_ids": []},
    }
    scores = _scores_by_key(
        evaluate_deterministic_quality(
            {"messages": []}, outputs, {"expects_clarification": False}
        )
    )
    assert all(value == 1 for value in scores.values())


def test_claim_evidence_coverage_requires_a_quote_present_in_its_source():
    outputs = {
        "sources": [
            {
                "source_id": "S1",
                "content": "The official result was 42 percent in 2025.",
            }
        ],
        "dimension_results": [
            {
                "claims": [
                    {
                        "supporting_source_ids": ["S1"],
                        "supporting_evidence": [
                            {"source_id": "S1", "quote": "A fabricated quote."}
                        ],
                        "contradicting_source_ids": [],
                    }
                ]
            }
        ],
    }

    scores = _scores_by_key(evaluate_deterministic_quality({"messages": []}, outputs))

    assert scores["claim_evidence_coverage"] == 0


def test_deterministic_evaluator_does_not_fall_back_past_empty_report_ledger():
    outputs = {
        "report_draft": "Rejected marker [Srejected].",
        "report_sources": [],
        "report_dimension_results": [],
        "sources": [
            {
                "source_id": "Srejected",
                "quality_status": "rejected",
                "content": "Rejected content.",
            }
        ],
        "dimension_results": [{"claims": [{"supporting_source_ids": ["Srejected"]}]}],
    }

    scores = _scores_by_key(evaluate_deterministic_quality({}, outputs))

    assert scores["citation_validity"] == 0
    assert scores["accepted_evidence_isolation"] == 0


def test_material_conflict_disclosure_checks_final_report_not_stale_audit():
    conflict = {
        "conflict_id": "CF-1",
        "material": True,
        "left_source_ids": ["S1"],
        "right_source_ids": ["S2"],
    }
    outputs = {
        "report_draft": "CF-1 presents both accepted positions [S1] [S2].",
        "report_sources": [
            {"source_id": "S1", "quality_status": "accepted"},
            {"source_id": "S2", "quality_status": "accepted"},
        ],
        "report_dimension_results": [],
        "claim_conflicts": [conflict],
        "report_consistency_audit": {
            "passes": False,
            "omitted_conflict_ids": ["CF-1"],
        },
    }

    scores = _scores_by_key(evaluate_deterministic_quality({}, outputs))

    assert scores["material_conflict_disclosure"] == 1
    assert scores["exact_quote_validity"] == 0


def test_target_resumes_both_interrupts_and_collects_events(monkeypatch):
    class FakeGraph:
        calls = 0

        async def astream(self, graph_input, config, stream_mode):
            del config, stream_mode
            self.calls += 1
            if self.calls == 1:
                assert isinstance(graph_input, dict)
                yield (
                    "updates",
                    {
                        "__interrupt__": (
                            SimpleNamespace(
                                value={"type": "research_topic_clarification"}
                            ),
                        )
                    },
                )
                return
            if self.calls == 2:
                assert graph_input.resume == {"action": "accept_assumptions"}
                yield (
                    "updates",
                    {
                        "__interrupt__": (
                            SimpleNamespace(
                                value={"type": "research_dimension_review"}
                            ),
                        )
                    },
                )
                return
            assert graph_input.resume == {"approved": True}
            yield "custom", {"type": "finalizing_answer"}
            yield "updates", {"finalize_answer": {}}
            yield (
                "values",
                {
                    "messages": [{"content": "Final report"}],
                    "report_draft": "Draft",
                    "report_generation_mode": "sectioned",
                    "report_overview": "Overview",
                    "report_sections": [{"dimension_id": "0"}],
                    "dimension_results": [],
                    "sources_gathered": [],
                },
            )

    fake_graph = FakeGraph()
    monkeypatch.setattr(
        "research_agent.evals.target.builder.compile", lambda **kwargs: fake_graph
    )
    result = asyncio.run(
        ResearchEvaluationTarget()(
            {"messages": [{"role": "user", "content": "Question"}]}
        )
    )
    assert result["final_report"] == "Final report"
    assert result["report_generation_mode"] == "sectioned"
    assert result["report_overview"] == "Overview"
    assert result["report_sections"] == [{"dimension_id": "0"}]
    assert result["node_trajectory"] == ["finalize_answer"]
    assert result["custom_events"] == [{"type": "finalizing_answer"}]


def test_deterministic_evaluator_does_not_reward_missing_citations_or_claims():
    outputs = {
        "report_draft": "No citations.",
        "sources": [],
        "dimension_results": [],
        "custom_events": [],
        "node_trajectory": [],
        "report_revision_count": 0,
        "max_report_revisions": 1,
    }
    scores = _scores_by_key(
        evaluate_deterministic_quality(
            {"messages": []}, outputs, {"expects_clarification": False}
        )
    )
    assert scores["citation_validity"] == 0
    assert scores["claim_source_validity"] == 0
    assert scores["dimension_completion"] == 0


def test_report_quality_evaluator_uses_deepseek_structured_output(monkeypatch):
    class FakeModel:
        def with_structured_output(self, schema, method):
            assert schema is ReportQualityScores
            assert method == "json_mode"
            return self

        def with_retry(self, **kwargs):
            assert kwargs == {"stop_after_attempt": 3}
            return self

        def invoke(self, prompt):
            assert "Expected topics" in prompt
            return ReportQualityScores(
                relevance=0.9,
                structure=0.8,
                completeness=0.7,
                source_quality=0.6,
                analytical_rigor=0.5,
                balance_and_objectivity=0.4,
                comment="Useful baseline.",
            )

    monkeypatch.setattr(
        "research_agent.evals.evaluators.report_quality.create_deepseek_model",
        lambda model: FakeModel(),
    )
    feedback = DeepSeekReportQualityEvaluator("deepseek-test")(
        {"messages": [{"content": "Question"}]},
        {"final_report": "Report"},
        {"expected_topics": ["topic"]},
    )
    assert len(feedback) == 6
    assert feedback[0]["key"] == "relevance_score"
    assert feedback[0]["score"] == 0.9


def test_groundedness_evaluator_filters_invalid_indexes(monkeypatch):
    class FakeModel:
        def with_structured_output(self, schema, method):
            assert schema is GroundednessAssessment
            return self

        def with_retry(self, **kwargs):
            assert kwargs == {"stop_after_attempt": 3}
            return self

        def invoke(self, prompt):
            assert "JSON object" in prompt
            return GroundednessAssessment(
                unsupported_indexes=[1, 99, -1], comment="One unsupported claim."
            )

    monkeypatch.setattr(
        "research_agent.evals.evaluators.groundedness.create_deepseek_model",
        lambda model: FakeModel(),
    )
    result = DeepSeekGroundednessEvaluator("deepseek-test")(
        {},
        {
            "dimension_results": [
                {
                    "claims": [
                        {
                            "claim": "Supported",
                            "supporting_evidence": [
                                {"source_id": "S1", "quote": "Evidence"}
                            ],
                        },
                        {
                            "claim": "Unsupported",
                            "supporting_evidence": [
                                {"source_id": "S2", "quote": "Other"}
                            ],
                        },
                    ]
                }
            ]
        },
    )
    assert result["score"] == 0.5


def test_evaluate_example_retains_judge_failure_and_deterministic_scores():
    class FakeTarget:
        async def __call__(self, inputs):
            return {
                "report_draft": "No citations.",
                "sources": [],
                "dimension_results": [
                    {
                        "completion_status": "partial",
                        "gap_diagnostics": [
                            {
                                "dimension_title": "Market",
                                "gap_id": "gap-market",
                                "priority": "medium",
                                "status": "unresolvable",
                                "attempt_count": 2,
                                "closure_blockers": ["missing_requested_source_type"],
                                "required_source_types": ["government"],
                                "direct_evidence_confirmed": True,
                                "supported_claim_count": 1,
                                "requested_source_type_satisfied": False,
                                "independent_source_count": 1,
                                "required_independent_source_count": 1,
                                "assessment_status": "completed",
                                "matched_sources": [
                                    {
                                        "source_id": "S1",
                                        "source_type": "news",
                                        "domain": "example.com",
                                    }
                                ],
                                "remaining_evidence": "An official source",
                            }
                        ],
                    }
                ],
                "custom_events": [],
                "node_trajectory": [],
                "report_revision_count": 0,
                "max_report_revisions": 1,
            }

    def failing_judge(*args):
        raise RuntimeError("judge unavailable")

    result = asyncio.run(
        evaluate_example(
            {
                "id": "case",
                "inputs": {"messages": [{"content": "Question"}]},
                "reference_outputs": {},
            },
            FakeTarget(),
            [failing_judge],
        )
    )
    assert result["scores"]
    assert "judge unavailable" in result["errors"][0]
    assert result["summary"]["unresolved_gaps"] == 1
    assert result["gap_diagnostics"][0]["gap_id"] == "gap-market"


def test_aggregate_and_markdown_report_expose_metric_coverage():
    results = [
        {
            "id": "one",
            "duration_seconds": 2,
            "scores": [{"key": "metric", "score": 1}],
            "errors": [],
        },
        {
            "id": "two",
            "duration_seconds": 4,
            "scores": [],
            "errors": ["judge failed"],
        },
    ]
    aggregate = aggregate_results(results)
    assert aggregate["metrics"]["metric"] == {
        "mean": 1.0,
        "count": 1,
        "coverage": 0.5,
    }
    markdown = render_markdown(
        {
            "generated_at": "now",
            "dataset": "smoke",
            "evaluation_model": "deepseek",
            "aggregate": aggregate,
            "results": results,
        }
    )
    assert "| metric | 1.000 | 1/2 |" in markdown


def test_markdown_report_renders_gap_diagnostics():
    result = {
        "id": "case",
        "duration_seconds": 1,
        "scores": [],
        "errors": [],
        "gap_diagnostics": [
            {
                "dimension_title": "Market | Scope",
                "gap_id": "gap-source-type",
                "priority": "medium",
                "status": "unresolvable",
                "attempt_count": 2,
                "closure_blockers": ["missing_requested_source_type"],
                "required_source_types": ["government"],
                "direct_evidence_confirmed": True,
                "supported_claim_count": 1,
                "requested_source_type_satisfied": False,
                "independent_source_count": 1,
                "required_independent_source_count": 1,
                "assessment_status": "completed",
                "matched_sources": [
                    {
                        "source_id": "S1",
                        "source_type": "news",
                        "domain": "example.com",
                    }
                ],
                "remaining_evidence": "An official source",
            }
        ],
    }
    report = {
        "generated_at": "now",
        "dataset": "smoke",
        "evaluation_model": "deepseek",
        "aggregate": aggregate_results([result]),
        "results": [result],
    }

    markdown = render_markdown(report)

    assert "#### Gap Diagnostics" in markdown
    assert "Market \\| Scope" in markdown
    assert "missing_requested_source_type" in markdown
    assert "direct=True; claims=1; requested_type=False" in markdown
    assert "S1:news:example.com" in markdown


def test_runner_rejects_deadlocking_or_empty_settings():
    base = {
        "limit": 1,
        "concurrency": 1,
        "target_retries": 1,
    }
    validate_args(SimpleNamespace(**base))
    with pytest.raises(ValueError, match="concurrency"):
        validate_args(SimpleNamespace(**{**base, "concurrency": 0}))
    with pytest.raises(ValueError, match="target-retries"):
        validate_args(SimpleNamespace(**{**base, "target_retries": -1}))
