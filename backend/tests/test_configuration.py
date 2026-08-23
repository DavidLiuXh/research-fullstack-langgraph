import pytest
from pydantic import ValidationError

from research_agent.configuration import Configuration


def test_source_quality_thresholds_are_ordered():
    with pytest.raises(ValidationError):
        Configuration(
            source_acceptance_threshold=0.5,
            source_supplementary_threshold=0.7,
        )


def test_source_quality_limits_are_configurable():
    configuration = Configuration(
        max_source_candidates_per_dimension=25,
        max_selected_sources_per_dimension=8,
        max_sources_per_domain=3,
        max_claims_per_dimension=10,
        min_accepted_sources_per_dimension=3,
    )

    assert configuration.max_source_candidates_per_dimension == 25
    assert configuration.max_selected_sources_per_dimension == 8
    assert configuration.max_sources_per_domain == 3
    assert configuration.max_claims_per_dimension == 10
    assert configuration.min_accepted_sources_per_dimension == 3


def test_quality_defaults_allow_adaptive_research():
    configuration = Configuration()

    assert configuration.max_research_loops == 3
    assert configuration.max_initial_gaps_per_dimension == 4
    assert configuration.max_gap_no_progress_attempts == 2
    assert configuration.min_independent_sources_per_high_gap == 2
    assert configuration.max_dimension_reflections == 3
    assert configuration.min_accepted_sources_per_dimension == 2
    assert configuration.min_authoritative_sources_per_dimension == 1
    assert configuration.min_primary_sources_per_dimension == 1
    assert configuration.report_sectioning_claim_threshold == 18
    assert configuration.report_sectioning_material_chars == 18000
