from typing import Any

__all__ = ["graph"]


def __getattr__(name: str) -> Any:
    """Load the compiled graph lazily so utility modules remain independently testable."""
    if name == "graph":
        from research_agent.graph import graph

        return graph
    raise AttributeError(name)
