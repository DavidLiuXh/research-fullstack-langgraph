# Evaluation

This package adapts the interrupt-driven research graph to offline and LangSmith
evaluation. It automatically accepts proposed clarification assumptions and
approves generated research dimensions, then records graph nodes, custom events,
final state, sources, claims, audit results, and the rendered report.

The first-phase suite combines deterministic checks with DeepSeek-as-judge:

- citation and claim-source validity
- dimension sufficiency and source-quality metadata
- domain diversity and workflow trajectory completeness
- clarification routing and report revision budget compliance
- relevance, structure, completeness, source quality, analytical rigor, balance
- claim-to-evidence groundedness

Run a bounded local smoke evaluation from `backend/`:

```bash
uv run python -m research_agent.evals.run_evaluate --limit 3
```

Results are written as JSON and Markdown under
`src/research_agent/evals/results/`. Use
`--skip-llm-judges` for deterministic-only development checks. The evaluator
model defaults to `deepseek-v4-pro` and can be changed with `EVALUATION_MODEL`.
Failed graph examples are retried once by default; use `--target-retries` to
change that bounded retry policy. Reports retain the attempt count and final
failure stage.

To evaluate an existing LangSmith dataset:

```bash
uv run python -m research_agent.evals.run_evaluate \
  --langsmith-dataset "Deep Research Bench" \
  --experiment-name "deepseek-baseline"
```

Full Deep Research Bench runs are intentionally separate from the smoke suite
because they are substantially more expensive.
