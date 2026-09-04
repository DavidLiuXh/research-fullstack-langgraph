"""DeepSeek-as-judge report quality evaluation."""

from __future__ import annotations

import json
import os
from typing import Any, cast

from research_agent.evals.schemas import ReportQualityScores
from research_agent.llm import create_deepseek_model

REPORT_QUALITY_PROMPT = """You are an independent evaluator of a deep-research report.
Score each criterion from 0 to 1. Judge the report against the user request and
expected topics. Do not reward length by itself. Penalize unsupported certainty,
weak sources, missing counterarguments, and failure to follow the request.

Criteria:
- relevance: directly answers the request
- structure: a coherent article with a central thesis, developed prose paragraphs,
  logical transitions, and an integrated conclusion; penalize claim-by-claim lists
  and repetitive section summaries unless the user explicitly requested a list
- completeness: covers material expected topics and requirements
- source_quality: uses authoritative and diverse sources appropriately
- analytical_rigor: relates multiple findings through chronology, causality,
  comparison, or implications instead of merely enumerating evidence
- balance_and_objectivity: represents uncertainty and material counterevidence

Return JSON matching this schema:
{schema}

User request:
{question}

Expected topics:
{expected_topics}

Report:
{report}
"""


class DeepSeekReportQualityEvaluator:
    """Evaluate six report qualities in one bounded DeepSeek request."""

    def __init__(self, model: str | None = None):
        """Configure the independent DeepSeek judge model."""
        self.model: str = model or os.getenv("EVALUATION_MODEL") or "deepseek-v4-pro"

    def __call__(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return normalized feedback for each report-quality criterion."""
        reference = reference_outputs or {}
        question = str(inputs["messages"][0]["content"])
        prompt = REPORT_QUALITY_PROMPT.format(
            schema=json.dumps(
                ReportQualityScores.model_json_schema(), ensure_ascii=False
            ),
            question=question,
            expected_topics=json.dumps(
                reference.get("expected_topics", []), ensure_ascii=False
            ),
            report=outputs["final_report"],
        )
        result = cast(
            ReportQualityScores,
            (
                create_deepseek_model(self.model)
                .with_structured_output(ReportQualityScores, method="json_mode")
                .with_retry(stop_after_attempt=3)
                .invoke(prompt)
            ),
        )
        scores = result.model_dump(exclude={"comment"})
        return [
            {"key": f"{key}_score", "score": value, "comment": result.comment}
            for key, value in scores.items()
        ]
