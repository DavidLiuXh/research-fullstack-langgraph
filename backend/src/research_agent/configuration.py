import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="deepseek-v4-flash",
        description="The model used for dimension planning and query generation.",
    )

    reflection_model: str = Field(
        default="deepseek-v4-flash",
        description="The model used to reflect on each research dimension.",
    )

    answer_model: str = Field(
        default="deepseek-v4-pro",
        description="The model used to synthesize the final answer.",
    )

    number_of_initial_queries: int = Field(
        default=3,
        description="The number of search queries generated per dimension loop.",
    )

    number_of_research_dimensions: int = Field(
        default=3,
        ge=2,
        le=8,
        description="Number of complementary research dimensions.",
    )

    max_research_loops: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Maximum focused search attempts allowed for each evidence gap.",
    )

    max_initial_gaps_per_dimension: int = Field(
        default=4,
        ge=1,
        le=6,
        description="Maximum evidence gaps planned before researching a dimension.",
    )

    max_gap_no_progress_attempts: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Consecutive no-progress attempts before a gap is unresolvable.",
    )

    min_independent_sources_per_high_gap: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Independent accepted sources required to close a high-priority gap.",
    )

    dimension_reflection_soft_limit: int = Field(
        default=3,
        ge=1,
        le=6,
        description=(
            "Reflection rounds normally allowed before only material, progressing "
            "gaps may extend the research."
        ),
    )

    max_dimension_reflections: int = Field(
        default=5,
        ge=1,
        le=8,
        description="Hard cap on whole-dimension audits, including adaptive extensions.",
    )

    max_reflection_no_progress_rounds: int = Field(
        default=2,
        ge=1,
        le=3,
        description=(
            "Consecutive dimension reflections without new accepted evidence, "
            "verified claims, or resolved gaps before stopping early."
        ),
    )

    max_report_revisions: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum report revisions after independent quality audits.",
    )

    report_sectioning_claim_threshold: int = Field(
        default=18,
        ge=4,
        le=100,
        description="Use sectioned drafting when the audited claim count reaches this value.",
    )

    report_sectioning_material_chars: int = Field(
        default=18000,
        ge=4000,
        le=100000,
        description="Use sectioned drafting when compact evidence exceeds this size.",
    )

    tavily_search_depth: str = Field(
        default="advanced",
        description="Tavily search depth: basic or advanced.",
    )

    tavily_max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum Tavily results returned per query.",
    )

    tavily_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries after the initial Tavily search attempt.",
    )

    @model_validator(mode="after")
    def validate_source_thresholds(self):
        """Ensure supplementary evidence cannot outrank accepted evidence."""
        if self.source_supplementary_threshold > self.source_acceptance_threshold:
            raise ValueError(
                "source_supplementary_threshold must not exceed "
                "source_acceptance_threshold"
            )
        for field_name in (
            "min_accepted_sources_per_dimension",
            "min_authoritative_sources_per_dimension",
            "min_primary_sources_per_dimension",
        ):
            if getattr(self, field_name) > self.max_selected_sources_per_dimension:
                raise ValueError(
                    f"{field_name} must not exceed max_selected_sources_per_dimension"
                )
        if self.dimension_reflection_soft_limit > self.max_dimension_reflections:
            if "dimension_reflection_soft_limit" not in self.model_fields_set:
                self.dimension_reflection_soft_limit = self.max_dimension_reflections
                return self
            raise ValueError(
                "dimension_reflection_soft_limit must not exceed "
                "max_dimension_reflections"
            )
        return self

    max_source_candidates_per_dimension: int = Field(
        default=40,
        ge=5,
        le=100,
        description="Maximum deduplicated sources assessed per dimension.",
    )

    max_selected_sources_per_dimension: int = Field(
        default=12,
        ge=1,
        le=30,
        description="Maximum quality-screened sources retained per dimension.",
    )

    max_sources_per_domain: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Maximum retained sources from one publisher domain per dimension; "
            "the diversity-first default prevents one institution from crowding "
            "out independent evidence."
        ),
    )

    source_acceptance_threshold: float = Field(
        default=0.65,
        ge=0,
        le=1,
        description="Evidence score required for an accepted source.",
    )

    source_supplementary_threshold: float = Field(
        default=0.45,
        ge=0,
        le=1,
        description="Minimum evidence score for supplementary evidence.",
    )

    min_accepted_sources_per_dimension: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum accepted sources required before a dimension is sufficient.",
    )

    min_authoritative_sources_per_dimension: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Minimum authoritative sources required per completed dimension.",
    )

    min_primary_sources_per_dimension: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Minimum primary sources required per completed dimension.",
    )

    max_claims_per_dimension: int = Field(
        default=12,
        ge=1,
        le=30,
        description="Maximum auditable claims retained per dimension.",
    )

    max_claim_source_chars: int = Field(
        default=1800,
        ge=500,
        le=6000,
        description="Maximum evidence characters per source sent to claim extraction.",
    )

    min_evidence_quote_chars: int = Field(
        default=12,
        ge=6,
        le=100,
        description="Minimum normalized length of an evidence quote.",
    )

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
