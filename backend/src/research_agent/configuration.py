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
        default=2,
        description="Maximum research loops performed independently per dimension.",
    )

    max_report_revisions: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum report revisions after independent quality audits.",
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
        return self

    max_source_candidates_per_dimension: int = Field(
        default=40,
        ge=5,
        le=100,
        description="Maximum deduplicated sources assessed per dimension.",
    )

    max_selected_sources_per_dimension: int = Field(
        default=12,
        ge=3,
        le=30,
        description="Maximum quality-screened sources retained per dimension.",
    )

    max_sources_per_domain: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum retained sources from one domain per dimension.",
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
