"""Regressions from the automotive report: dates, scopes and coherent delivery."""

import importlib
from datetime import date

import pytest
from langchain_core.messages import AIMessage

from research_agent.reporting import (
    comparison_scope_note,
    editorial_packets,
    unfinished_period,
    writing_evidence,
)
from research_agent.tools_and_schemas import (
    ClaimConflictAnalysis,
    ClaimConflictItem,
    ReportAudit,
    ReportConsistencyAudit,
)


@pytest.mark.parametrize(
    "topic",
    ["调研2026年前9个月国内汽车产销", "2026年1-9月汽车市场", "first 9 months of 2026"],
)
def test_requested_period_not_yet_observable(topic):
    assert unfinished_period(topic, date(2026, 9, 13)) == "2026-09-30"
    assert unfinished_period(topic, date(2026, 10, 1)) == ""


def test_temporal_gate_has_actionable_assumption_and_resumes(monkeypatch):
    module = importlib.import_module("research_agent.graph")

    class Today(date):
        @classmethod
        def today(cls):
            return cls(2026, 9, 13)

    monkeypatch.setattr(module, "date", Today)
    state = {
        "original_research_topic": "调研2026年前9个月国内汽车市场",
        "topic_clarification_history": [],
    }
    result = module.analyze_research_topic(state, {})
    assert result["topic_needs_clarification"]
    assert "2026-09-30" in result["topic_clarification_questions"][0]
    assert "最新已公布完整月份" in result["normalized_research_topic"]
    monkeypatch.setattr(
        module, "interrupt", lambda payload: {"action": "accept_assumptions"}
    )
    monkeypatch.setattr(module, "emit_research_event", lambda *a, **k: None)
    resumed = module.request_topic_clarification({**state, **result})
    assert module.route_topic_clarification(resumed) == "generate_research_dimensions"


@pytest.mark.parametrize(
    "left,right",
    [
        (
            "2026年1月新能源汽车销量94.5万辆，增长0.1%",
            "2026年1月新能源乘用车零售59.6万辆，下降20%",
        ),
        ("2026年第一季度新能源汽车销量下降3.7%", "2026年第一季度国内销量下降20%"),
        (
            "2026年上半年传统汽车销量下降30%，电动汽车销量下降不到20%",
            "2026年上半年汽车产量下降6%，国内销量下降20%",
        ),
        ("2026年1月销量100万辆", "2026年2月销量90万辆"),
    ],
)
def test_automotive_scope_differences_are_not_proven_contradictions(left, right):
    assert comparison_scope_note(left, right)


def test_same_scope_contradictions_remain_auditable():
    assert not comparison_scope_note(
        "2026年1月乘用车零售100万辆", "2026年1月乘用车零售80万辆"
    )
    assert not comparison_scope_note(
        "2026年1月乘用车零售100万辆",
        "2026年1月乘用车零售80万辆，2026年2月乘用车零售90万辆",
    )


def test_conflict_node_downgrades_noncomparable_sales_without_hiding_same_scope(
    monkeypatch,
):
    module = importlib.import_module("research_agent.graph")

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, prompt):
            assert "explicit_metric_scope" in prompt
            return ClaimConflictAnalysis(
                conflicts=[
                    ClaimConflictItem(
                        left_claim_id="C0",
                        right_claim_id="C1",
                        relation="contradiction",
                        severity="high",
                    )
                ]
            )

    monkeypatch.setattr(module, "create_deepseek_model", lambda *a, **k: Model())
    monkeypatch.setattr(module, "emit_research_event", lambda *a, **k: None)
    results = evidence_results()
    results[0]["claims"][0]["claim"] = "2026年1月新能源汽车销量94.5万辆，同比增长0.1%"
    results[1]["claims"][0]["claim"] = "2026年1月新能源乘用车零售59.6万辆，同比下降20%"
    state = {
        "report_dimension_results": results,
        "report_sources": [],
        "research_run_id": "run",
        "normalized_research_topic": "汽车市场",
    }
    conflict = module.detect_claim_conflicts(state, {})["claim_conflicts"][0]
    assert conflict["relation"] == "scope_difference"
    assert not conflict["material"]
    results[1]["claims"][0]["claim"] = "2026年1月新能源汽车销量80万辆，同比下降5%"
    assert module.detect_claim_conflicts(state, {})["claim_conflicts"][0]["material"]


def evidence_results():
    return [
        {
            "dimension": {"id": str(i), "title": f"Dimension {i}", "scope": "scope"},
            "research_run_id": "run",
            "research_loop_count": 3,
            "is_sufficient": False,
            "completion_status": "budget_exhausted",
            "confidence": 0,
            "claims": [
                {
                    "claim_id": f"C{i}",
                    "claim": f"Fact {i}",
                    "contradicting_source_ids": [],
                    "uncertainty_reason": "",
                    "supporting_source_ids": [f"S{i}"],
                    "supporting_evidence": [
                        {"source_id": f"S{i}", "quote": f"Evidence {i}"}
                    ],
                }
            ],
            "sources": [],
            "unresolved_gaps": [
                {"question": "What is the reporting cutoff?", "attempt_count": 3}
            ],
        }
        for i in range(2)
    ]


def test_editorial_chapter_combines_dimensions_and_preserves_provenance():
    results = evidence_results()
    plan = {
        "sections": [
            {
                "dimension_id": "demand-drivers",
                "title": "Why demand changed",
                "objective": "Explain drivers",
                "claim_ids": ["C1", "C0", "invented"],
            }
        ]
    }
    packets = editorial_packets(results, plan)
    assert len(packets) == 1
    assert packets[0]["dimension"]["title"] == "Why demand changed"
    assert [c["claim_id"] for c in packets[0]["claims"]] == ["C1", "C0"]
    assert packets[0]["claims"][0]["supporting_source_ids"] == ["S1"]
    assert len(results) == 2
    material = writing_evidence(results)
    assert "Evidence 1" in material and "reporting cutoff" in material
    assert '"allowed_citation_markers": ["[S1]"]' in material
    assert (
        "budget_exhausted" not in material
        and "confidence" not in material
        and "attempt_count" not in material
    )


def test_whole_report_context_reaches_overview_conclusion_and_next_chapter(monkeypatch):
    module = importlib.import_module("research_agent.graph")
    prompts = []

    class Model:
        def invoke(self, prompt):
            prompts.append(prompt)
            return AIMessage(content=f"Completed analysis {len(prompts)}.")

    monkeypatch.setattr(module, "create_deepseek_model", lambda *a, **k: Model())
    monkeypatch.setattr(module, "emit_research_event", lambda *a, **k: None)
    module._generate_sectioned_report(
        evidence_results(), "Market research", "model", "run", [], {"sections": []}
    )
    assert "Completed analysis 1." in prompts[1]
    assert (
        "Completed analysis 1." in prompts[2] and "Completed analysis 2." in prompts[2]
    )
    assert (
        "Completed analysis 1." in prompts[3] and "Completed analysis 2." in prompts[3]
    )


def test_revision_can_use_verified_facts_allocated_to_another_chapter(monkeypatch):
    module = importlib.import_module("research_agent.graph")
    prompts = []

    def revise(**kwargs):
        prompts.append(kwargs["prompt"])
        return "Revised connected prose."

    monkeypatch.setattr(module, "_bounded_report_part_revision", revise)
    monkeypatch.setattr(module, "emit_research_event", lambda *a, **k: None)
    state = {
        "normalized_research_topic": "Market research",
        "research_run_id": "run",
        "report_audit": {"issues": ["Include the company comparison"]},
        "report_sections": [
            {"dimension_id": "0", "title": "Comparison", "content": "Draft"}
        ],
        "report_plan": {"sections": []},
    }
    module._revise_sectioned_report(state, evidence_results(), "model")
    assert "Global audited evidence" in prompts[0]
    assert "Evidence 1" in prompts[0]
    assert "budget_exhausted" not in prompts[0]


@pytest.mark.parametrize(
    "targets,expected_calls",
    [(["overview"], ["report_overview_revision"]),
     (["body:0"], ["report_section_revision", "report_overview_revision", "report_conclusion_revision"]),
     (["unknown"], ["report_section_revision", "report_section_revision", "report_overview_revision", "report_conclusion_revision"])],
)
def test_scoped_revision_preserves_unaffected_parts(monkeypatch, targets, expected_calls):
    module = importlib.import_module("research_agent.graph")
    calls = []

    def revise(**kwargs):
        calls.append(kwargs["event_prefix"])
        return "Corrected prose."

    monkeypatch.setattr(module, "_bounded_report_part_revision", revise)
    monkeypatch.setattr(module, "emit_research_event", lambda *a, **k: None)
    sections = [{"dimension_id": str(i), "title": f"Chapter {i}", "content": f"Preserve table {i}"} for i in range(2)]
    state = {
        "normalized_research_topic": "Market research", "research_run_id": "run",
        "report_audit": {"issues": ["Fix comparison"], "revision_targets": targets},
        "report_plan": {"sections": []}, "report_sections": sections,
        "report_overview": "Old overview", "report_conclusion": "Old conclusion",
    }
    output = module._revise_sectioned_report(state, evidence_results(), "model")
    assert calls == expected_calls
    if targets == ["overview"]:
        assert output["report_sections"] == sections
        assert output["report_conclusion"] == "Old conclusion"
    if targets == ["body:0"]:
        assert output["report_sections"][1] == sections[1]


def test_editorial_failure_retains_fact_checked_prose(monkeypatch):
    module = importlib.import_module("research_agent.graph")
    model_calls = []

    class Model:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self

        def invoke(self, prompt):
            if self.schema is ReportAudit:
                return ReportAudit(
                    passes=False,
                    factual_passes=True,
                    issues=["Improve transitions"],
                    revision_instructions=["Connect chapters"],
                )
            return ReportConsistencyAudit(passes=True)

    def factory(model, **kwargs):
        model_calls.append((model, kwargs))
        return Model()

    monkeypatch.setattr(module, "create_deepseek_model", factory)
    monkeypatch.setattr(module, "emit_research_event", lambda *a, **k: None)
    state = {
        "report_dimension_results": [],
        "report_sources": [],
        "normalized_research_topic": "Market",
        "research_run_id": "run",
        "report_draft": "A connected article.",
        "report_revision_count": 2,
    }
    audit = module.audit_report(state, {"configurable": {"answer_model": "final-auditor"}})
    assert model_calls == [("final-auditor", {"thinking": True})] * 2
    assert audit["last_fact_checked_report"] == state["report_draft"]
    result = module.build_safe_report({**state, **audit}, {})
    assert result["report_draft"] == "A connected article."
    assert result["report_generation_mode"] == "fact_checked_draft"


def test_fact_failure_cannot_be_marked_overall_passing():
    with pytest.raises(ValueError):
        ReportAudit(passes=True, factual_passes=False)


def test_alternative_audit_shape_never_discards_findings_or_coerces_false_to_true():
    audit = ReportAudit.model_validate(
        {"pass": "false", "issues": ["Unsupported causal claim"],
         "revision_instructions": ["Qualify the mechanism"]}
    )
    assert not audit.passes
    assert audit.issues == ["Unsupported causal claim"]
    with pytest.raises(ValueError):
        ReportAudit.model_validate({"pass": True, "issues": ["Unsupported claim"]})


def test_unknown_citation_cannot_replace_last_fact_checked_draft(monkeypatch):
    module = importlib.import_module("research_agent.graph")

    class Model:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self

        def invoke(self, prompt):
            if self.schema is ReportAudit:
                return ReportAudit(passes=True, factual_passes=True, revision_targets=["overview"])
            return ReportConsistencyAudit(passes=True)

    monkeypatch.setattr(module, "create_deepseek_model", lambda *a, **k: Model())
    monkeypatch.setattr(module, "emit_research_event", lambda *a, **k: None)
    state = {
        "report_dimension_results": evidence_results(),
        "report_sources": [{"source_id": "S0"}, {"source_id": "S1"}],
        "normalized_research_topic": "Market",
        "research_run_id": "run",
        "report_draft": "Unsupported invented citation [S0-0-1].",
        "last_fact_checked_report": "Previous safe prose [S0].",
        "report_revision_count": 2,
    }
    audit = module.audit_report(state, {})
    assert not audit["report_audit"]["passes"]
    assert not audit["report_audit"]["factual_passes"]
    assert audit["report_audit"]["revision_targets"] == []
    assert audit["last_fact_checked_report"] == state["last_fact_checked_report"]
    assert module.route_report_audit({**state, **audit}) == "build_safe_report"
    assert module.build_safe_report({**state, **audit}, {})["report_draft"] == state[
        "last_fact_checked_report"
    ]


def test_partial_inventory_is_not_scored_as_a_complete_article():
    from research_agent.evals.evaluators.deterministic import (
        evaluate_deterministic_quality,
    )

    scores = evaluate_deterministic_quality(
        {},
        {
            "report_draft": "Readable prose. " * 100,
            "report_generation_mode": "safe_fallback",
            "sources": [],
            "dimension_results": [],
            "custom_events": [],
            "node_trajectory": [],
        },
        {},
    )
    assert next(s["score"] for s in scores if s["key"] == "article_coherence") == 0


def test_report_table_contract_does_not_leak_into_query_schema():
    from research_agent.language import format_research_prompt

    assert "REPORTING CONTRACT" not in format_research_prompt(
        "Generate JSON queries", language_context={}
    )
    assert "REPORTING CONTRACT" in format_research_prompt(
        "Draft {report_plan}", language_context={}, report_plan="plan"
    )


def test_comparison_tables_belong_to_body_not_summary():
    module = importlib.import_module("research_agent.graph")
    table = "\n| Indicator | Value |\n| --- | --- |\n| Sales | 100 |\n"
    for heading in ("执行摘要", "结论", "Executive overview", "Conclusion"):
        assert module._report_article_style_findings(f"## {heading}\n{table}")
    assert not module._report_article_style_findings(f"## 市场产销比较\n{table}")
