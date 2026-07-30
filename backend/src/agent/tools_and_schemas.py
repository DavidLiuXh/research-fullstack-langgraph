from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: list[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant."
    )


class ResearchDimension(BaseModel):
    title: str = Field(description="A concise name for this research dimension.")
    scope: str = Field(
        description="A self-contained description of what this dimension must investigate."
    )


class ResearchDimensionList(BaseModel):
    dimensions: list[ResearchDimension] = Field(
        description="Distinct, complementary dimensions that jointly cover the topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the evidence is sufficient for this research dimension."
    )
    knowledge_gap: str = Field(
        description="What remains unknown; empty when the evidence is sufficient."
    )
