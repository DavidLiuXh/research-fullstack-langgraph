import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


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
