import hashlib
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SOURCE_TYPE_VALUES = {
    "government",
    "academic",
    "official_company",
    "standards_body",
    "international_organization",
    "industry_association",
    "research_institute",
    "major_media",
    "specialist_media",
    "commercial_report",
    "blog",
    "forum",
    "aggregator",
    "unknown",
}
SOURCE_TYPE_ALIASES = {
    "official documentation": "official_company",
    "official docs": "official_company",
    "vendor documentation": "official_company",
    "official repository": "official_company",
    "github": "official_company",
    "government website": "government",
    "regulator": "government",
    "official statistics": "government",
    "academic paper": "academic",
    "research paper": "academic",
    "peer reviewed": "academic",
    "standards": "standards_body",
    "standard": "standards_body",
    "industry report": "research_institute",
}


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

    @field_validator("query", mode="before")
    @classmethod
    def normalize_single_query(cls, value):
        """Accept a single query string when only one query was requested."""
        if isinstance(value, str):
            return [value]
        return value


class ResearchDimension(BaseModel):
    title: str = Field(description="A concise name for this research dimension.")
    scope: str = Field(
        description="A self-contained description of what this dimension must investigate."
    )


class ResearchDimensionList(BaseModel):
    dimensions: list[ResearchDimension] = Field(
        description="Distinct, complementary dimensions that jointly cover the topic."
    )


class SourceAssessment(BaseModel):
    source_id: str = Field(description="The exact source ID being assessed.")
    source_type: Literal[
        "government",
        "academic",
        "official_company",
        "standards_body",
        "international_organization",
        "industry_association",
        "research_institute",
        "major_media",
        "specialist_media",
        "commercial_report",
        "blog",
        "forum",
        "aggregator",
        "unknown",
    ]
    authority_score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    recency_score: float = Field(ge=0, le=1)
    is_primary_source: bool
    is_likely_repost: bool
    supported_topics: list[str]
    rejection_reasons: list[str]


class SourceAssessmentList(BaseModel):
    assessments: list[SourceAssessment]

    @field_validator("assessments", mode="before")
    @classmethod
    def normalize_compact_assessments(cls, value):
        """Accept conservative fallbacks for compact provider responses."""
        if not isinstance(value, dict):
            return value
        normalized = []
        for source_id, assessment in value.items():
            if isinstance(assessment, dict):
                normalized.append({"source_id": source_id, **assessment})
                continue
            try:
                score = max(0.0, min(float(assessment), 1.0))
            except (TypeError, ValueError):
                score = 0.0
            normalized.append(
                {
                    "source_id": source_id,
                    "source_type": "unknown",
                    "authority_score": 0.35,
                    "relevance_score": score,
                    "recency_score": 0.5,
                    "is_primary_source": False,
                    "is_likely_repost": False,
                    "supported_topics": [],
                    "rejection_reasons": [
                        "The model returned only a compact relevance score; "
                        "authority metadata was unavailable."
                    ],
                }
            )
        return normalized


class ResearchGap(BaseModel):
    gap_id: str = ""
    question: str
    reason: str
    priority: Literal["high", "medium", "low"]
    required_source_types: list[str]
    expected_evidence: str = ""
    suggested_query_focus: str

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_aliases(cls, value):
        """Normalize common compact field names returned by providers."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.setdefault(
            "reason",
            "This question remains unanswered and may affect dimension coverage.",
        )
        normalized.setdefault("priority", normalized.pop("impact", "medium"))
        normalized.setdefault(
            "required_source_types", normalized.pop("source_types", [])
        )
        source_types = normalized.get("required_source_types") or []
        if isinstance(source_types, str):
            source_types = [source_types]
        normalized["required_source_types"] = list(
            dict.fromkeys(
                source_type
                if source_type in SOURCE_TYPE_VALUES
                else SOURCE_TYPE_ALIASES.get(source_type.casefold().strip(), "unknown")
                for item in source_types
                for source_type in [str(item).casefold().strip()]
            )
        )
        normalized.setdefault(
            "suggested_query_focus", normalized.pop("search_focus", "")
        )
        normalized.setdefault(
            "expected_evidence",
            normalized.pop("evidence_needed", normalized.get("reason", "")),
        )
        if not normalized.get("gap_id"):
            question = unicodedata.normalize(
                "NFKC", str(normalized.get("question", ""))
            ).casefold()
            question = re.sub(r"\s+", " ", question).strip()
            digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
            normalized["gap_id"] = f"gap-{digest}"
        return normalized


class ResearchGapPlan(BaseModel):
    """Initial evidence gaps planned for one approved research dimension."""

    gaps: list[ResearchGap] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def normalize_planned_gaps(self):
        """Remove duplicate stable IDs from provider output."""
        unique: dict[str, ResearchGap] = {}
        for gap in self.gaps:
            unique.setdefault(gap.gap_id, gap)
        self.gaps = list(unique.values())
        return self


class GapEvidenceAssessment(BaseModel):
    """Semantic mapping between accepted sources and one active gap."""

    gap_id: str
    directly_answers_gap: bool = False
    matched_source_ids: list[str] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    contradictory_source_ids: list[str] = Field(default_factory=list)
    remaining_evidence: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_compact_assessment(cls, value):
        """Accept conservative defaults for compact provider responses."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.setdefault("directly_answers_gap", normalized.pop("direct", False))
        normalized.setdefault("matched_source_ids", normalized.pop("source_ids", []))
        normalized.setdefault("supported_claims", normalized.pop("claims", []))
        normalized.setdefault(
            "contradictory_source_ids", normalized.pop("conflicting_source_ids", [])
        )
        normalized.setdefault(
            "remaining_evidence", normalized.pop("missing_evidence", "")
        )
        return normalized


class EvidenceConflict(BaseModel):
    description: str
    source_ids: list[str]
    likely_explanation: str = ""
    requires_follow_up: bool = False


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the evidence is sufficient for this research dimension."
    )
    covered_questions: list[str]
    resolved_gap_ids: list[str] = Field(default_factory=list)
    missing_questions: list[ResearchGap] = Field(max_length=3)
    unsupported_claims: list[str]
    contradictions: list[EvidenceConflict]
    source_quality_issues: list[str]
    recommended_search_strategy: list[str]
    do_not_repeat: list[str]
    completion_reason: str
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_compact_reflection(cls, value):
        """Fill conservative defaults for compact provider reflections."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.setdefault("is_sufficient", normalized.pop("sufficient", False))
        normalized.setdefault("covered_questions", [])
        normalized.setdefault("resolved_gap_ids", [])
        normalized.setdefault("missing_questions", [])
        normalized.setdefault("unsupported_claims", [])
        normalized.setdefault("contradictions", [])
        normalized.setdefault("source_quality_issues", [])
        gaps = normalized.get("missing_questions") or []
        normalized.setdefault(
            "recommended_search_strategy",
            [
                gap.get("search_focus") or gap.get("suggested_query_focus")
                for gap in gaps
                if isinstance(gap, dict)
                and (gap.get("search_focus") or gap.get("suggested_query_focus"))
            ],
        )
        normalized.setdefault("do_not_repeat", [])
        normalized.setdefault(
            "completion_reason",
            "Evidence is sufficient."
            if normalized["is_sufficient"]
            else "Material evidence gaps remain.",
        )
        normalized.setdefault(
            "confidence", 0.75 if normalized["is_sufficient"] else 0.35
        )
        return normalized

    @model_validator(mode="after")
    def validate_sufficiency(self):
        """Reject internally inconsistent completion decisions."""
        if self.is_sufficient and any(
            gap.priority == "high" for gap in self.missing_questions
        ):
            raise ValueError("Sufficient evidence cannot have high-priority gaps")
        if self.is_sufficient and any(
            conflict.requires_follow_up for conflict in self.contradictions
        ):
            raise ValueError("Sufficient evidence cannot have unresolved conflicts")
        return self


class EvidenceQuote(BaseModel):
    """A verbatim excerpt attributed to one selected source."""

    source_id: str
    quote: str = ""
    locator: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_quote_aliases(cls, value):
        """Normalize common evidence quote aliases from model providers."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.setdefault(
            "quote", normalized.pop("excerpt", normalized.pop("text", ""))
        )
        normalized.setdefault(
            "locator", normalized.pop("location", normalized.pop("section", ""))
        )
        return normalized


class EvidenceClaim(BaseModel):
    """A factual claim backed by directly quoted selected evidence."""

    claim: str
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    counter_evidence: list[EvidenceQuote] = Field(default_factory=list)
    uncertainty: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)

    @property
    def source_ids(self) -> list[str]:
        """Expose supporting source IDs for rolling compatibility."""
        return list(dict.fromkeys(item.source_id for item in self.evidence))

    @property
    def counter_source_ids(self) -> list[str]:
        """Expose counter-source IDs for rolling compatibility."""
        return list(dict.fromkeys(item.source_id for item in self.counter_evidence))

    @model_validator(mode="before")
    @classmethod
    def normalize_misplaced_source_ids(cls, value):
        """Recover DeepSeek JSON that puts source IDs in the evidence field."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        evidence = normalized.pop("supporting_evidence", normalized.get("evidence", []))
        source_ids = normalized.pop(
            "source_ids", normalized.pop("supporting_source_ids", [])
        )
        if isinstance(evidence, dict):
            evidence = [
                {"source_id": source_id, "quote": quote}
                for source_id, quote in evidence.items()
            ]
        if not isinstance(evidence, list):
            evidence = []
        if evidence and all(isinstance(item, str) for item in evidence):
            if not source_ids:
                source_ids = evidence
            evidence = []
        existing_evidence_ids = {
            item.get("source_id") for item in evidence if isinstance(item, dict)
        }
        evidence.extend(
            {"source_id": str(source_id), "quote": ""}
            for source_id in source_ids
            if source_id not in existing_evidence_ids
        )
        normalized["evidence"] = evidence

        counter_evidence = normalized.pop(
            "contradicting_evidence", normalized.get("counter_evidence", [])
        )
        counter_source_ids = normalized.pop(
            "counter_source_ids", normalized.pop("contradicting_source_ids", [])
        )
        if isinstance(counter_evidence, dict):
            counter_evidence = [
                {"source_id": source_id, "quote": quote}
                for source_id, quote in counter_evidence.items()
            ]
        if not isinstance(counter_evidence, list):
            counter_evidence = []
        if counter_evidence and all(isinstance(item, str) for item in counter_evidence):
            if not counter_source_ids:
                counter_source_ids = counter_evidence
            counter_evidence = []
        existing_counter_ids = {
            item.get("source_id") for item in counter_evidence if isinstance(item, dict)
        }
        counter_evidence.extend(
            {"source_id": str(source_id), "quote": ""}
            for source_id in counter_source_ids
            if source_id not in existing_counter_ids
        )
        normalized["counter_evidence"] = counter_evidence
        normalized.setdefault("uncertainty", normalized.pop("uncertainty_reason", ""))
        normalized.setdefault("confidence", 0.5)
        return normalized


class ClaimExtraction(BaseModel):
    claims: list[EvidenceClaim]
    summary: str

    @model_validator(mode="before")
    @classmethod
    def normalize_summary_alias(cls, value):
        """Accept the previous summary field during rolling upgrades."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.setdefault("summary", normalized.pop("dimension_summary", ""))
        return normalized


class ReportAudit(BaseModel):
    passes: bool
    issues: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_alternative_audit_shape(cls, value):
        """Recover a semantically valid DeepSeek audit with renamed fields."""
        if not isinstance(value, dict):
            return value
        raw = value.get("audit", value)
        if not isinstance(raw, dict):
            return value

        def render_findings(items, *keys):
            rendered = []
            for item in items or []:
                if isinstance(item, str):
                    rendered.append(item)
                    continue
                if not isinstance(item, dict):
                    rendered.append(str(item))
                    continue
                subject = next((item.get(key) for key in keys if item.get(key)), "")
                explanation = item.get("explanation", "")
                rendered.append(
                    f"{subject}: {explanation}"
                    if subject and explanation
                    else str(subject or explanation)
                )
            return [item for item in rendered if item]

        if "passes" in raw:
            return raw
        normalized = dict(raw)
        normalized["passes"] = bool(raw.get("pass", False))
        normalized["issues"] = [
            *render_findings(
                raw.get("factual_statements_without_support"), "statement", "claim"
            ),
            *render_findings(raw.get("overstated_claims"), "claim", "statement"),
            *render_findings(raw.get("citation_marker_issues"), "issue"),
            *render_findings(
                raw.get("contradictions_counterarguments_uncertainty"), "issue"
            ),
            *render_findings(raw.get("structure_duplication_clarity"), "issue"),
        ]
        normalized["revision_instructions"] = [
            str(item) for item in raw.get("required_corrections", []) if item
        ]
        if raw.get("draft_answers_material_parts") is False:
            normalized["issues"] = [
                *normalized["issues"],
                "The draft does not answer every material part of the request.",
            ]
        if not normalized["passes"] and not normalized["revision_instructions"]:
            normalized["revision_instructions"] = [
                "Resolve every material finding identified by the report audit."
            ]
        return normalized

    @model_validator(mode="after")
    def validate_pass_status(self):
        """Prevent a passing audit from carrying material findings."""
        if self.passes and self.issues:
            raise ValueError("A passing report audit cannot contain findings")
        if not self.passes and not self.revision_instructions:
            raise ValueError("A failing report audit requires revision instructions")
        return self
