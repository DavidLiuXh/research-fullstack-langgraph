"""Deterministic evaluators for citations, evidence, coverage, and trajectory."""

from __future__ import annotations

import re
from typing import Any

from research_agent.utils import locate_evidence_quote


def _score(key: str, value: float, comment: str = "") -> dict[str, Any]:
    return {"key": key, "score": max(0.0, min(float(value), 1.0)), "comment": comment}


def _ratio(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    return numerator / denominator if denominator else empty


def evaluate_deterministic_quality(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return low-cost regression metrics without an LLM judge."""
    del inputs
    reference = reference_outputs or {}
    sources = (
        outputs.get("report_sources", [])
        if "report_sources" in outputs
        else outputs.get("sources", [])
    )
    source_ids = {source.get("source_id") for source in sources}
    source_by_id = {source.get("source_id"): source for source in sources}
    dimensions = (
        outputs.get("report_dimension_results", [])
        if "report_dimension_results" in outputs
        else outputs.get("dimension_results", [])
    )
    claims = [claim for result in dimensions for claim in result.get("claims", [])]
    claim_source_ids = [
        source_id
        for claim in claims
        for source_id in [
            *claim.get("supporting_source_ids", []),
            *claim.get("contradicting_source_ids", []),
        ]
    ]
    draft_markers = re.findall(r"\[(S[A-Za-z0-9-]+)\]", outputs.get("report_draft", ""))
    valid_markers = [marker for marker in draft_markers if marker in source_ids]
    valid_claim_ids = [
        source_id for source_id in claim_source_ids if source_id in source_ids
    ]
    accepted_source_ids = {
        source.get("source_id")
        for source in sources
        if source.get("quality_status") == "accepted"
    }
    report_evidence_ids = set(draft_markers) | set(claim_source_ids)
    accepted_report_evidence_ids = report_evidence_ids & accepted_source_ids

    selected = [
        source
        for source in sources
        if source.get("quality_status") in {"accepted", "supplementary"}
    ]
    quality_scored = [
        source for source in selected if source.get("evidence_score") is not None
    ]
    domains = {source.get("domain") for source in selected if source.get("domain")}
    source_types = {
        source.get("source_type")
        for source in selected
        if source.get("source_type") and source.get("source_type") != "unknown"
    }
    research_sources = outputs.get("sources", [])
    research_selected = [
        source
        for source in research_sources
        if source.get("quality_status") == "accepted"
    ]
    research_domains = {
        source.get("domain") for source in research_selected if source.get("domain")
    }
    sufficient_dimensions = [
        result for result in dimensions if result.get("is_sufficient")
    ]
    evidence_items = [
        evidence
        for claim in claims
        for evidence in claim.get("supporting_evidence", [])
        if isinstance(evidence, dict)
    ]
    valid_evidence_items = [
        evidence
        for evidence in evidence_items
        if (source := source_by_id.get(evidence.get("source_id")))
        and locate_evidence_quote(
            str(source.get("content", "")), str(evidence.get("quote", ""))
        )
    ]
    claims_with_evidence = [
        claim
        for claim in claims
        if any(
            isinstance(evidence, dict)
            and (source := source_by_id.get(evidence.get("source_id")))
            and locate_evidence_quote(
                str(source.get("content", "")), str(evidence.get("quote", ""))
            )
            for evidence in claim.get("supporting_evidence", [])
        )
    ]
    known_gap_count = sum(
        int(result.get("known_gap_count", 0)) for result in dimensions
    )
    resolved_gap_count = sum(
        int(result.get("resolved_gap_count", 0)) for result in dimensions
    )
    high_gap_count = sum(
        int(result.get("high_priority_gap_count", 0)) for result in dimensions
    )
    resolved_high_gap_count = sum(
        int(result.get("resolved_high_priority_gap_count", 0)) for result in dimensions
    )
    high_gap_source_coverage_count = sum(
        int(result.get("high_priority_gap_source_coverage_count", 0))
        for result in dimensions
    )
    gain_history = [
        gain
        for result in dimensions
        for gain in result.get("evidence_gain_history", [])
    ]
    gainful_loops = [gain for gain in gain_history if gain.get("total_gain", 0) > 0]
    direct_evidence_gap_count = sum(
        int(result.get("direct_evidence_gap_count", 0)) for result in dimensions
    )
    supported_claim_gap_count = sum(
        int(result.get("supported_claim_gap_count", 0)) for result in dimensions
    )
    requested_type_gap_count = sum(
        int(result.get("requested_type_gap_count", 0)) for result in dimensions
    )
    independent_source_gap_count = sum(
        int(result.get("independent_source_gap_count", 0)) for result in dimensions
    )
    gap_assessment_failure_count = sum(
        int(result.get("gap_assessment_failure_count", 0)) for result in dimensions
    )
    available_dimensions = [
        result
        for result in dimensions
        if result.get("completion_status") != "search_unavailable"
    ]
    dimensions_with_primary = [
        result
        for result in dimensions
        if any(
            source.get("quality_status") == "accepted"
            and source.get("is_primary_source")
            for source in result.get("sources", [])
        )
    ]
    dimensions_with_authority = [
        result
        for result in dimensions
        if any(
            source.get("quality_status") == "accepted"
            and source.get("is_authoritative_source")
            for source in result.get("sources", [])
        )
    ]
    dimensions_with_independent_publishers = [
        result
        for result in dimensions
        if len(
            {
                source.get("domain")
                for source in result.get("sources", [])
                if source.get("quality_status") == "accepted" and source.get("domain")
            }
        )
        >= 2
    ]

    event_types = {
        event.get("type")
        for event in outputs.get("custom_events", [])
        if isinstance(event, dict)
    }
    required_events = {
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
    }
    observed_required_events = required_events & event_types
    expected_clarification = bool(reference.get("expects_clarification", False))
    actual_clarification = any(
        event.get("type") == "topic_analyzed" and event.get("needs_clarification")
        for event in outputs.get("custom_events", [])
        if isinstance(event, dict)
    )
    revision_count = int(outputs.get("report_revision_count", 0))
    max_revisions = int(outputs.get("max_report_revisions", 0))
    material_conflicts = [
        conflict
        for conflict in outputs.get("claim_conflicts", [])
        if conflict.get("material")
    ]
    disclosed_material_conflicts = [
        conflict
        for conflict in material_conflicts
        if conflict.get("conflict_id") in outputs.get("report_draft", "")
        and any(
            f"[{source_id}]" in outputs.get("report_draft", "")
            for source_id in conflict.get("left_source_ids", [])
        )
        and any(
            f"[{source_id}]" in outputs.get("report_draft", "")
            for source_id in conflict.get("right_source_ids", [])
        )
    ]

    comments = []
    if draft_markers and len(valid_markers) != len(draft_markers):
        comments.append("The draft contains unknown source markers.")
    if claim_source_ids and len(valid_claim_ids) != len(claim_source_ids):
        comments.append("Some audited claims reference unknown sources.")

    return [
        _score(
            "citation_validity",
            _ratio(len(valid_markers), len(draft_markers), empty=0.0),
            "; ".join(comments) or "All emitted citation markers are valid.",
        ),
        _score(
            "accepted_evidence_isolation",
            _ratio(
                len(accepted_report_evidence_ids),
                len(report_evidence_ids),
                empty=1.0,
            ),
            "Every report and claim source is accepted evidence."
            if accepted_report_evidence_ids == report_evidence_ids
            else "Rejected, supplementary, or unknown evidence reached report material.",
        ),
        _score(
            "claim_source_validity",
            _ratio(len(valid_claim_ids), len(claim_source_ids), empty=0.0),
        ),
        _score(
            "claim_evidence_coverage",
            _ratio(len(claims_with_evidence), len(claims), empty=0.0),
        ),
        _score(
            "exact_quote_validity",
            _ratio(len(valid_evidence_items), len(evidence_items), empty=0.0),
        ),
        _score(
            "dimension_completion",
            _ratio(len(sufficient_dimensions), len(dimensions), empty=0.0),
            f"{len(sufficient_dimensions)}/{len(dimensions)} dimensions declared sufficient.",
        ),
        _score(
            "source_quality_metadata",
            _ratio(len(quality_scored), len(selected), empty=0.0),
        ),
        _score(
            "source_domain_diversity",
            _ratio(len(domains), len(selected), empty=0.0),
            f"{len(domains)} unique report-evidence domains across {len(selected)} selected sources.",
        ),
        _score(
            "research_source_domain_diversity",
            _ratio(len(research_domains), len(research_selected), empty=0.0),
            f"{len(research_domains)} unique accepted research domains across {len(research_selected)} sources.",
        ),
        _score(
            "source_type_diversity",
            _ratio(len(source_types), min(len(selected), 3), empty=0.0),
            f"{len(source_types)} distinct recognized source types in report evidence; target=3.",
        ),
        _score(
            "independent_publisher_dimension_coverage",
            _ratio(
                len(dimensions_with_independent_publishers),
                len(dimensions),
                empty=0.0,
            ),
            f"{len(dimensions_with_independent_publishers)}/{len(dimensions)} dimensions have at least two accepted publisher domains.",
        ),
        _score(
            "primary_source_dimension_coverage",
            _ratio(len(dimensions_with_primary), len(dimensions), empty=0.0),
        ),
        _score(
            "authoritative_source_dimension_coverage",
            _ratio(len(dimensions_with_authority), len(dimensions), empty=0.0),
        ),
        _score(
            "gap_resolution",
            _ratio(resolved_gap_count, known_gap_count, empty=1.0),
            f"{resolved_gap_count}/{known_gap_count} known gaps resolved.",
        ),
        _score(
            "gap_direct_evidence_coverage",
            _ratio(direct_evidence_gap_count, known_gap_count, empty=1.0),
            f"{direct_evidence_gap_count}/{known_gap_count} gaps have direct accepted evidence.",
        ),
        _score(
            "gap_supported_claim_coverage",
            _ratio(supported_claim_gap_count, known_gap_count, empty=1.0),
            f"{supported_claim_gap_count}/{known_gap_count} gaps have supported claims.",
        ),
        _score(
            "gap_requested_type_coverage",
            _ratio(requested_type_gap_count, known_gap_count, empty=1.0),
            f"{requested_type_gap_count}/{known_gap_count} gaps satisfy their requested source type.",
        ),
        _score(
            "gap_independent_source_coverage",
            _ratio(independent_source_gap_count, known_gap_count, empty=1.0),
            f"{independent_source_gap_count}/{known_gap_count} gaps meet their independent-source requirement.",
        ),
        _score(
            "gap_assessment_reliability",
            1
            - _ratio(
                gap_assessment_failure_count,
                len(gain_history),
                empty=0.0,
            ),
            f"{gap_assessment_failure_count}/{len(gain_history)} gap assessments used conservative structured-output fallback.",
        ),
        _score(
            "high_priority_gap_resolution",
            _ratio(resolved_high_gap_count, high_gap_count, empty=1.0),
            f"{resolved_high_gap_count}/{high_gap_count} high-priority gaps resolved.",
        ),
        _score(
            "high_priority_gap_source_coverage",
            _ratio(high_gap_source_coverage_count, high_gap_count, empty=1.0),
            f"{high_gap_source_coverage_count}/{high_gap_count} high-priority gaps have accepted requested-type evidence.",
        ),
        _score(
            "evidence_gain_per_loop",
            _ratio(len(gainful_loops), len(gain_history), empty=0.0),
            f"{len(gainful_loops)}/{len(gain_history)} loops added evidence or resolved gaps.",
        ),
        _score(
            "search_availability",
            _ratio(len(available_dimensions), len(dimensions), empty=0.0),
            f"{len(available_dimensions)}/{len(dimensions)} dimensions had usable search access.",
        ),
        _score(
            "trajectory_completeness",
            _ratio(len(observed_required_events), len(required_events)),
            "Missing events: " + ", ".join(sorted(required_events - event_types))
            if required_events - event_types
            else "All required workflow stages were observed.",
        ),
        _score(
            "clarification_routing",
            float(expected_clarification == actual_clarification),
            f"expected={expected_clarification}, actual={actual_clarification}",
        ),
        _score(
            "revision_budget_compliance",
            float(revision_count <= max_revisions),
            f"revisions={revision_count}, limit={max_revisions}",
        ),
        _score(
            "consistency_analysis_completion",
            float(outputs.get("consistency_analysis_complete", False)),
        ),
        _score(
            "material_conflict_disclosure",
            _ratio(
                len(disclosed_material_conflicts),
                len(material_conflicts),
                empty=1.0,
            ),
            f"{len(disclosed_material_conflicts)}/{len(material_conflicts)} material conflicts are explicitly disclosed with both evidence sides.",
        ),
    ]
