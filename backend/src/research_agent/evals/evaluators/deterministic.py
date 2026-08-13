"""Deterministic evaluators for citations, evidence, coverage, and trajectory."""

from __future__ import annotations

import re
from typing import Any


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
    sources = outputs.get("sources", [])
    source_ids = {source.get("source_id") for source in sources}
    dimensions = outputs.get("dimension_results", [])
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

    selected = [
        source
        for source in sources
        if source.get("quality_status") in {"accepted", "supplementary"}
    ]
    quality_scored = [
        source for source in selected if source.get("evidence_score") is not None
    ]
    domains = {source.get("domain") for source in selected if source.get("domain")}
    sufficient_dimensions = [
        result for result in dimensions if result.get("is_sufficient")
    ]

    event_types = {
        event.get("type")
        for event in outputs.get("custom_events", [])
        if isinstance(event, dict)
    }
    required_events = {
        "planning_dimensions",
        "queries_generated",
        "search_completed",
        "sources_evaluated",
        "reflection_completed",
        "claims_extracted",
        "drafting_report",
        "report_audit_completed",
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
            "claim_source_validity",
            _ratio(len(valid_claim_ids), len(claim_source_ids), empty=0.0),
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
    ]
