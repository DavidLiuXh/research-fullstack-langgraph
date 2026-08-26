"""Run local evaluation experiments or an existing LangSmith dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from research_agent.evals.evaluators import (
    DeepSeekGroundednessEvaluator,
    DeepSeekReportQualityEvaluator,
    evaluate_deterministic_quality,
)
from research_agent.evals.target import EvaluationRunConfig, ResearchEvaluationTarget

BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "smoke.jsonl"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load validated input/reference pairs from a JSONL dataset."""
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            example = json.loads(line)
            if "inputs" not in example or "reference_outputs" not in example:
                raise ValueError(f"Invalid evaluation example at line {line_number}")
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return examples


def _flatten_feedback(feedback: Any) -> list[dict[str, Any]]:
    if isinstance(feedback, dict):
        feedback = [feedback]
    if not isinstance(feedback, list):
        raise TypeError("Evaluator feedback must be a dict or list of dicts")
    normalized = []
    for item in feedback:
        if not isinstance(item, dict) or "key" not in item or "score" not in item:
            raise ValueError(f"Invalid evaluator feedback: {item!r}")
        normalized.append(item)
    return normalized


async def evaluate_example(
    example: dict[str, Any],
    target: ResearchEvaluationTarget,
    llm_evaluators: list[Any],
    target_retries: int = 1,
) -> dict[str, Any]:
    """Run the graph and all evaluators while retaining individual failures."""
    started = time.perf_counter()
    inputs = example["inputs"]
    reference = example["reference_outputs"]
    target_errors = []
    outputs = None
    for attempt in range(target_retries + 1):
        try:
            outputs = await target(inputs)
            break
        except Exception as error:
            target_errors.append(
                f"attempt {attempt + 1}: {type(error).__name__}: {error}"
            )
    if outputs is None:
        return {
            "id": example.get("id", "unknown"),
            "duration_seconds": time.perf_counter() - started,
            "scores": [],
            "target_attempts": target_retries + 1,
            "errors": [f"target: {' | '.join(target_errors)}"],
        }

    scores = _flatten_feedback(
        evaluate_deterministic_quality(inputs, outputs, reference)
    )
    errors = []
    for evaluator in llm_evaluators:
        try:
            feedback = await asyncio.to_thread(evaluator, inputs, outputs, reference)
            scores.extend(_flatten_feedback(feedback))
        except Exception as error:
            errors.append(
                f"{type(evaluator).__name__}: {type(error).__name__}: {error}"
            )
    return {
        "id": example.get("id", "unknown"),
        "question": inputs["messages"][0]["content"],
        "duration_seconds": time.perf_counter() - started,
        "target_attempts": len(target_errors) + 1,
        "scores": scores,
        "errors": errors,
        "summary": {
            "dimensions": len(outputs.get("dimension_results", [])),
            "sources": len(outputs.get("sources", [])),
            "claims": sum(
                len(result.get("claims", []))
                for result in (
                    outputs.get("report_dimension_results")
                    or outputs.get("dimension_results", [])
                )
            ),
            "revisions": outputs.get("report_revision_count", 0),
            "report_generation_mode": outputs.get("report_generation_mode", "unknown"),
            "completion_statuses": [
                result.get("completion_status", "unknown")
                for result in outputs.get("dimension_results", [])
            ],
            "search_failures": sum(
                int(result.get("search_failure_count", 0))
                for result in outputs.get("dimension_results", [])
            ),
            "known_gaps": sum(
                int(result.get("known_gap_count", 0))
                for result in outputs.get("dimension_results", [])
            ),
            "resolved_gaps": sum(
                int(result.get("resolved_gap_count", 0))
                for result in outputs.get("dimension_results", [])
            ),
            "no_gain_loops": sum(
                int(result.get("no_gain_loop_count", 0))
                for result in outputs.get("dimension_results", [])
            ),
            "gap_assessment_failures": sum(
                int(result.get("gap_assessment_failure_count", 0))
                for result in outputs.get("dimension_results", [])
            ),
            "material_conflicts": sum(
                bool(conflict.get("material"))
                for conflict in outputs.get("claim_conflicts", [])
            ),
            "consistency_audit_passed": outputs.get("report_consistency_audit", {}).get(
                "passes", False
            ),
            "safe_fallback_used": outputs.get("report_safe_fallback_used", False),
        },
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute macro averages while making missing evaluator scores visible."""
    scores_by_key: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for score in result.get("scores", []):
            scores_by_key[score["key"]].append(float(score["score"]))
    total = len(results)
    return {
        "examples": total,
        "successful_targets": sum(
            not any(error.startswith("target:") for error in result.get("errors", []))
            for result in results
        ),
        "examples_with_errors": sum(bool(result.get("errors")) for result in results),
        "average_duration_seconds": statistics.fmean(
            result["duration_seconds"] for result in results
        ),
        "metrics": {
            key: {
                "mean": statistics.fmean(values),
                "count": len(values),
                "coverage": len(values) / total,
            }
            for key, values in sorted(scores_by_key.items())
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable experiment report beside the raw JSON output."""
    aggregate = report["aggregate"]
    lines = [
        "# Research Agent Evaluation Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Dataset: `{report['dataset']}`",
        f"- Evaluation model: `{report['evaluation_model']}`",
        f"- Examples: {aggregate['examples']}",
        f"- Successful graph runs: {aggregate['successful_targets']}",
        f"- Examples with errors: {aggregate['examples_with_errors']}",
        f"- Average duration: {aggregate['average_duration_seconds']:.1f}s",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | Coverage |",
        "| --- | ---: | ---: |",
    ]
    for key, metric in aggregate["metrics"].items():
        lines.append(
            f"| {key} | {metric['mean']:.3f} | {metric['count']}/{aggregate['examples']} |"
        )
    lines.extend(["", "## Per-example Results", ""])
    for result in report["results"]:
        lines.extend(
            [
                f"### {result['id']}",
                "",
                f"- Duration: {result['duration_seconds']:.1f}s",
                f"- Target attempts: {result.get('target_attempts', 1)}",
                f"- Summary: `{json.dumps(result.get('summary', {}), ensure_ascii=False)}`",
            ]
        )
        if result.get("errors"):
            lines.append("- Errors: " + " | ".join(result["errors"]))
        for score in result.get("scores", []):
            comment = f" — {score.get('comment')}" if score.get("comment") else ""
            lines.append(f"- `{score['key']}`: {float(score['score']):.3f}{comment}")
        lines.append("")
    return "\n".join(lines)


async def run_local(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run a local JSONL experiment and persist raw and Markdown reports."""
    examples = load_jsonl(args.dataset, args.limit)
    target = ResearchEvaluationTarget(
        EvaluationRunConfig(
            number_of_research_dimensions=args.dimensions,
            number_of_initial_queries=args.queries,
            max_research_loops=args.research_loops,
            max_report_revisions=args.report_revisions,
            tavily_max_results=args.search_results,
        )
    )
    llm_evaluators = (
        []
        if args.skip_llm_judges
        else [
            DeepSeekReportQualityEvaluator(args.evaluation_model),
            DeepSeekGroundednessEvaluator(args.evaluation_model),
        ]
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(example):
        async with semaphore:
            return await evaluate_example(
                example, target, llm_evaluators, args.target_retries
            )

    results = await asyncio.gather(*(bounded(example) for example in examples))
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "evaluation_model": args.evaluation_model,
        "run_config": {
            "dimensions": args.dimensions,
            "queries": args.queries,
            "research_loops": args.research_loops,
            "report_revisions": args.report_revisions,
            "search_results": args.search_results,
            "target_retries": args.target_retries,
            "llm_judges": not args.skip_llm_judges,
        },
        "aggregate": aggregate_results(results),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"evaluation-{stamp}.json"
    markdown_path = args.output_dir / f"evaluation-{stamp}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


async def run_langsmith(args: argparse.Namespace):
    """Evaluate against an existing LangSmith dataset using the same target."""
    if not args.langsmith_dataset:
        raise ValueError("--langsmith-dataset is required in LangSmith mode")
    from langsmith import Client

    target = ResearchEvaluationTarget()
    evaluators: list[Any] = [evaluate_deterministic_quality]
    if not args.skip_llm_judges:
        evaluators.extend(
            [
                DeepSeekReportQualityEvaluator(args.evaluation_model),
                DeepSeekGroundednessEvaluator(args.evaluation_model),
            ]
        )
    return await Client().aevaluate(
        target,
        data=args.langsmith_dataset,
        evaluators=evaluators,
        experiment_prefix=args.experiment_name,
        max_concurrency=args.concurrency,
        metadata={"evaluation_model": args.evaluation_model},
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for local and LangSmith evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--evaluation-model",
        default=os.getenv("EVALUATION_MODEL", "deepseek-v4-pro"),
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--queries", type=int, default=2)
    parser.add_argument("--research-loops", type=int, default=1)
    parser.add_argument("--report-revisions", type=int, default=1)
    parser.add_argument("--search-results", type=int, default=4)
    parser.add_argument("--target-retries", type=int, default=1)
    parser.add_argument("--skip-llm-judges", action="store_true")
    parser.add_argument("--langsmith-dataset")
    parser.add_argument("--experiment-name", default="deepseek-research-agent")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject settings that would deadlock or silently skip an experiment."""
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.target_retries < 0:
        raise ValueError("--target-retries cannot be negative")


def main() -> None:
    """Load local credentials and dispatch the selected evaluation mode."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(BACKEND_ROOT / ".env")
    args = build_parser().parse_args()
    validate_args(args)
    if args.langsmith_dataset:
        result = asyncio.run(run_langsmith(args))
        logging.info("%s", result)
    else:
        json_path, markdown_path = asyncio.run(run_local(args))
        logging.info("JSON report: %s", json_path)
        logging.info("Markdown report: %s", markdown_path)


if __name__ == "__main__":
    main()
