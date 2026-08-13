"""Compact schemas used by the DeepSeek evaluation judges."""

from pydantic import BaseModel, Field, model_validator


class ReportQualityScores(BaseModel):
    """Score complementary final-report qualities on a zero-to-one scale."""

    relevance: float = Field(ge=0, le=1)
    structure: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    analytical_rigor: float = Field(ge=0, le=1)
    balance_and_objectivity: float = Field(ge=0, le=1)
    comment: str = ""


class GroundednessAssessment(BaseModel):
    """Identify claim indexes not supported by their attached evidence."""

    unsupported_indexes: list[int] = Field(default_factory=list)
    comment: str = ""

    @model_validator(mode="after")
    def deduplicate_indexes(self):
        """Normalize duplicate indexes without hiding invalid values."""
        self.unsupported_indexes = list(dict.fromkeys(self.unsupported_indexes))
        return self
