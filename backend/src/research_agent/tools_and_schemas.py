from pydantic import BaseModel, Field


class TopicClarificationAssessment(BaseModel):
    needs_clarification: bool = Field(
        description="Whether material ambiguity prevents a reliable research plan."
    )
    ambiguities: list[str] = Field(
        description="Material ambiguities or missing information that change the plan."
    )
    clarification_questions: list[str] = Field(
        description="One to three prioritized questions for the user."
    )
    assumptions: list[str] = Field(
        description="Reasonable defaults the user may accept instead of answering."
    )
    normalized_topic: str = Field(
        description="A self-contained research brief using all known context and assumptions."
    )
    reason: str = Field(description="A concise explanation of the assessment.")


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
