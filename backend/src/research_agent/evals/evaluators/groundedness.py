"""DeepSeek judge for claim-to-evidence groundedness."""

from __future__ import annotations

import json
import os
from typing import Any, cast

from research_agent.evals.schemas import GroundednessAssessment
from research_agent.llm import create_deepseek_model

GROUNDEDNESS_PROMPT = """For each numbered claim, decide whether the attached
evidence text materially supports the claim. A source ID alone is not support.
Mark an index unsupported when the evidence is absent, irrelevant, contradictory,
or substantially weaker than the wording of the claim. Return only a JSON object
containing unsupported zero-based indexes and a concise comment.

Schema:
{schema}

Claims and attached evidence:
{claims}
"""


class DeepSeekGroundednessEvaluator:
    """Score audited claim grounding with one compact DeepSeek request."""

    def __init__(self, model: str | None = None, max_claims: int = 24):
        """Configure the judge model and bounded claim sample size."""
        self.model: str = model or os.getenv("EVALUATION_MODEL") or "deepseek-v4-pro"
        self.max_claims = max_claims

    def __call__(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score the fraction of audited claims supported by attached evidence."""
        del inputs, reference_outputs
        claims = [
            {
                "claim": claim.get("claim", ""),
                "evidence": [
                    {
                        "source_id": item.get("source_id", ""),
                        "quote": item.get("quote", "")[:800],
                        "locator": item.get("locator", ""),
                    }
                    for item in claim.get("supporting_evidence", [])
                    if isinstance(item, dict)
                ][:3],
            }
            for result in outputs.get("dimension_results", [])
            for claim in result.get("claims", [])
        ][: self.max_claims]
        if not claims:
            return {
                "key": "groundedness_score",
                "score": 0.0,
                "comment": "No audited claims were produced.",
            }
        prompt = GROUNDEDNESS_PROMPT.format(
            schema=json.dumps(
                GroundednessAssessment.model_json_schema(), ensure_ascii=False
            ),
            claims=json.dumps(claims, ensure_ascii=False),
        )
        result = cast(
            GroundednessAssessment,
            (
                create_deepseek_model(self.model)
                .with_structured_output(GroundednessAssessment, method="json_mode")
                .with_retry(stop_after_attempt=3)
                .invoke(prompt)
            ),
        )
        unsupported = {
            index for index in result.unsupported_indexes if 0 <= index < len(claims)
        }
        return {
            "key": "groundedness_score",
            "score": (len(claims) - len(unsupported)) / len(claims),
            "comment": result.comment,
        }
