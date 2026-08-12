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
    )

    assert configuration.max_source_candidates_per_dimension == 25
    assert configuration.max_selected_sources_per_dimension == 8
    assert configuration.max_sources_per_domain == 3
    assert configuration.max_claims_per_dimension == 10
