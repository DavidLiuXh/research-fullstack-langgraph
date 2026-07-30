import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from agent.state import ResearchSource


def get_research_topic(messages: list[AnyMessage]) -> str:
    """Build the research topic, retaining prior human/assistant context."""
    if len(messages) == 1:
        return str(messages[-1].content)

    parts: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            parts.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            parts.append(f"Assistant: {message.content}")
    return "\n".join(parts)


def tavily_results_to_sources(
    response: dict[str, Any], query: str, search_id: str
) -> list[ResearchSource]:
    """Normalize a Tavily response into stable, prompt-safe research sources."""
    sources: list[ResearchSource] = []
    for index, result in enumerate(response.get("results") or []):
        url = str(result.get("url") or "").strip()
        content = str(result.get("content") or "").strip()
        if not url or not content:
            continue
        sources.append(
            {
                "source_id": f"S{search_id}-{index}",
                "query": query,
                "title": str(result.get("title") or url).strip(),
                "url": url,
                "content": content,
                "score": result.get("score"),
                "published_date": result.get("published_date"),
            }
        )
    return sources


def format_sources_for_research(sources: list[ResearchSource]) -> str:
    """Format sources for reflection and answer prompts with stable source IDs."""
    if not sources:
        return "No usable search results were returned."

    blocks = []
    for source in sources:
        published = source.get("published_date")
        metadata = f"Published: {published}\n" if published else ""
        blocks.append(
            f"[{source['source_id']}]\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"{metadata}Content: {source['content']}"
        )
    return "\n\n".join(blocks)


def deduplicate_sources(sources: list[ResearchSource]) -> list[ResearchSource]:
    """Keep the first occurrence of each URL while preserving search order."""
    unique: list[ResearchSource] = []
    seen_urls: set[str] = set()
    for source in sources:
        if source["url"] in seen_urls:
            continue
        seen_urls.add(source["url"])
        unique.append(source)
    return unique


def render_source_citations(
    text: str, sources: list[ResearchSource]
) -> tuple[str, list[ResearchSource]]:
    """Replace valid [source-id] markers with Markdown links and report used sources."""
    source_map = {source["source_id"]: source for source in sources}
    used_ids: list[str] = []

    def replace(match: re.Match[str]) -> str:
        source_id = match.group(1)
        source = source_map.get(source_id)
        if source is None:
            return match.group(0)
        if source_id not in used_ids:
            used_ids.append(source_id)
        safe_title = source["title"].replace("[", "").replace("]", "")
        return f"[{safe_title}]({source['url']})"

    rendered = re.sub(r"\[(S[0-9-]+)\]", replace, text)
    return rendered, [source_map[source_id] for source_id in used_ids]
