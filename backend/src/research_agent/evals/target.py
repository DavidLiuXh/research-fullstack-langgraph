"""Adapt the interrupt-driven research graph to repeatable evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from research_agent.graph import builder


@dataclass(frozen=True)
class EvaluationRunConfig:
    """Bound expensive graph settings for repeatable evaluation experiments."""

    number_of_research_dimensions: int = 2
    number_of_initial_queries: int = 2
    max_research_loops: int = 1
    max_report_revisions: int = 1
    tavily_max_results: int = 4
    tavily_max_retries: int = 1


def _message_content(message: Any) -> str:
    if isinstance(message, BaseMessage):
        return str(message.content)
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(message)


def _interrupt_decision(interrupt_value: Any) -> dict[str, Any]:
    """Approve evaluation gates and accept proposed topic assumptions."""
    payload = getattr(interrupt_value, "value", interrupt_value)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported graph interrupt payload: {payload!r}")
    interrupt_type = payload.get("type")
    if interrupt_type == "research_topic_clarification":
        return {"action": "accept_assumptions"}
    if interrupt_type == "research_dimension_review":
        return {"approved": True}
    raise ValueError(f"Unsupported graph interrupt type: {interrupt_type!r}")


class ResearchEvaluationTarget:
    """Run one isolated graph invocation and expose evaluation-friendly output."""

    def __init__(self, run_config: EvaluationRunConfig | None = None):
        """Configure bounded graph settings for every target invocation."""
        self.run_config = run_config or EvaluationRunConfig()

    async def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute and resume the graph until it emits a final report."""
        graph = builder.compile(checkpointer=MemorySaver(), name="evaluation-graph")
        config: RunnableConfig = {
            "configurable": {
                "thread_id": f"eval-{uuid4().hex}",
                **asdict(self.run_config),
            }
        }
        graph_input: Any = {
            "messages": inputs["messages"],
            "initial_search_query_count": self.run_config.number_of_initial_queries,
            "max_research_loops": self.run_config.max_research_loops,
        }
        node_trajectory: list[str] = []
        custom_events: list[dict[str, Any]] = []
        final_state: dict[str, Any] = {}
        stream_modes: list[Literal["updates", "custom", "values"]] = [
            "updates",
            "custom",
            "values",
        ]

        for _ in range(6):
            interrupts = []
            try:
                async for mode, chunk in graph.astream(
                    graph_input,
                    config,
                    stream_mode=stream_modes,
                ):
                    if mode == "custom" and isinstance(chunk, dict):
                        custom_events.append(chunk)
                    elif mode == "updates" and isinstance(chunk, dict):
                        node_trajectory.extend(
                            node for node in chunk if node != "__interrupt__"
                        )
                        interrupts.extend(chunk.get("__interrupt__", ()))
                    elif mode == "values" and isinstance(chunk, dict):
                        final_state = chunk
            except Exception as error:
                last_node = node_trajectory[-1] if node_trajectory else "graph_start"
                last_event = (
                    custom_events[-1].get("type", "unknown")
                    if custom_events
                    else "none"
                )
                raise RuntimeError(
                    f"Evaluation graph failed after node={last_node}, "
                    f"event={last_event}: {type(error).__name__}: {error}"
                ) from error
            if not interrupts:
                break
            if len(interrupts) != 1:
                raise RuntimeError("Evaluation supports one interrupt at a time")
            graph_input = Command(resume=_interrupt_decision(interrupts[0]))
        else:
            raise RuntimeError("Evaluation exceeded the interrupt-resume safety limit")

        messages = final_state.get("messages", [])
        if not messages:
            raise RuntimeError("Evaluation graph completed without a final message")
        return {
            "final_report": _message_content(messages[-1]),
            "report_draft": final_state.get("report_draft", ""),
            "normalized_research_topic": final_state.get(
                "normalized_research_topic", ""
            ),
            "research_dimensions": final_state.get("research_dimensions", []),
            "dimension_results": final_state.get("dimension_results", []),
            "sources": final_state.get("sources_gathered", []),
            "report_dimension_results": final_state.get("report_dimension_results", []),
            "report_sources": final_state.get("report_sources", []),
            "report_evidence_ledger": final_state.get("report_evidence_ledger", {}),
            "claim_conflicts": final_state.get("claim_conflicts", []),
            "consistency_analysis_complete": final_state.get(
                "consistency_analysis_complete", False
            ),
            "report_consistency_audit": final_state.get("report_consistency_audit", {}),
            "report_safe_fallback_used": final_state.get(
                "report_safe_fallback_used", False
            ),
            "report_generation_mode": final_state.get(
                "report_generation_mode", "unknown"
            ),
            "report_overview": final_state.get("report_overview", ""),
            "report_sections": final_state.get("report_sections", []),
            "report_audit": final_state.get("report_audit", {}),
            "report_revision_count": final_state.get("report_revision_count", 0),
            "max_report_revisions": final_state.get("max_report_revisions", 0),
            "node_trajectory": node_trajectory,
            "custom_events": custom_events,
        }
