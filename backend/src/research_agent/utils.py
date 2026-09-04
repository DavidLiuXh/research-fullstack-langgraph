import re
import unicodedata
from math import isfinite
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from research_agent.state import DimensionResult, ResearchSource

TRACKING_QUERY_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}
MAX_SOURCE_SNIPPET_CHARS = 6000


def _normalized_text_with_offsets(value: str) -> tuple[str, list[int]]:
    """Normalize searchable text while retaining offsets into the original value."""
    normalized: list[str] = []
    offsets: list[int] = []
    previous_was_space = False
    for index, character in enumerate(value):
        expanded = unicodedata.normalize("NFKC", character).casefold()
        for item in expanded:
            if item.isspace():
                if previous_was_space:
                    continue
                item = " "
                previous_was_space = True
            else:
                previous_was_space = False
            normalized.append(item)
            offsets.append(index)
    while normalized and normalized[0] == " ":
        normalized.pop(0)
        offsets.pop(0)
    while normalized and normalized[-1] == " ":
        normalized.pop()
        offsets.pop()
    return "".join(normalized), offsets


def locate_evidence_quote(
    source_content: str, quote: str, *, min_chars: int = 12
) -> tuple[str, str] | None:
    """Return the exact source excerpt and character locator for a verified quote."""
    normalized_source, offsets = _normalized_text_with_offsets(source_content)
    normalized_quote, _ = _normalized_text_with_offsets(quote)
    if len(normalized_quote) < min_chars or not normalized_source or not offsets:
        return None
    match_start = normalized_source.find(normalized_quote)
    if match_start < 0:
        return None
    match_end = match_start + len(normalized_quote) - 1
    if match_end >= len(offsets):
        return None
    original_start = offsets[match_start]
    original_end = offsets[match_end] + 1
    excerpt = source_content[original_start:original_end].strip()
    if not excerpt:
        return None
    return excerpt, f"chars:{original_start}-{original_end}"


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
    response: dict[str, Any], query: str, search_id: str, research_run_id: str
) -> list[ResearchSource]:
    """Normalize a Tavily response into stable, prompt-safe research sources."""
    sources: list[ResearchSource] = []
    for index, result in enumerate(response.get("results") or []):
        url = str(result.get("url") or "").strip()
        content = str(result.get("content") or "").strip()[:MAX_SOURCE_SNIPPET_CHARS]
        if not url or not content:
            continue
        canonical_url = canonicalize_url(url)
        if not canonical_url:
            continue
        sources.append(
            {
                "research_run_id": research_run_id,
                "source_id": f"S{search_id}-{index}",
                "query": query,
                "title": str(result.get("title") or url).strip(),
                "url": url,
                "canonical_url": canonical_url,
                "domain": urlsplit(canonical_url).netloc,
                "content": content,
                "score": result.get("score"),
                "published_date": result.get("published_date"),
            }
        )
    return sources


def format_sources_for_research(
    sources: list[ResearchSource], *, max_content_chars: int | None = None
) -> str:
    """Format sources for reflection and answer prompts with stable source IDs."""
    if not sources:
        return "No usable search results were returned."

    blocks = []
    for source in sources:
        content = source["content"]
        if max_content_chars is not None and len(content) > max_content_chars:
            content = f"{content[:max_content_chars].rstrip()}…"
        published = source.get("published_date")
        metadata = f"Published: {published}\n" if published else ""
        quality_metadata = ""
        if source.get("quality_status"):
            quality_metadata = (
                f"Source type: {source.get('source_type', 'unknown')}\n"
                f"Quality: {source['quality_status']}\n"
                f"Evidence score: {source.get('evidence_score', 0):.2f}\n"
            )
        gap_ids = list(
            dict.fromkeys(
                [
                    *source.get("gap_ids", []),
                    *([source["gap_id"]] if source.get("gap_id") else []),
                ]
            )
        )
        provenance = f"Gap provenance: {', '.join(gap_ids)}\n" if gap_ids else ""
        blocks.append(
            f"[{source['source_id']}]\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"{metadata}{quality_metadata}{provenance}Content: {content}"
        )
    return "\n\n".join(blocks)


def deduplicate_sources(sources: list[ResearchSource]) -> list[ResearchSource]:
    """Deduplicate sources while preferring stronger, more complete results."""
    unique: list[ResearchSource] = []
    seen_urls: dict[str, ResearchSource] = {}
    seen_titles: dict[str, ResearchSource] = {}
    ranked_sources = sorted(
        sources,
        key=lambda source: (
            normalize_search_score(source.get("score"), default=0),
            bool(source.get("published_date")),
            len(source.get("content", "")),
        ),
        reverse=True,
    )
    for source in ranked_sources:
        canonical_url = source.get("canonical_url") or canonicalize_url(source["url"])
        normalized_title = normalize_title(source["title"])
        title_is_distinctive = len(normalized_title) >= 20
        duplicate = seen_urls.get(canonical_url)
        if duplicate is None and title_is_distinctive:
            duplicate = seen_titles.get(normalized_title)
        if duplicate is not None:
            duplicate["gap_ids"] = list(
                dict.fromkeys(
                    [
                        *duplicate.get("gap_ids", []),
                        *source.get("gap_ids", []),
                        *([source["gap_id"]] if source.get("gap_id") else []),
                    ]
                )
            )
            duplicate["requested_source_types"] = sorted(
                set(duplicate.get("requested_source_types", []))
                | set(source.get("requested_source_types", []))
            )
            expectations = list(
                dict.fromkeys(
                    item
                    for item in (
                        duplicate.get("expected_evidence", ""),
                        source.get("expected_evidence", ""),
                    )
                    if item
                )
            )
            duplicate["expected_evidence"] = " | ".join(expectations)
            continue
        normalized_source: ResearchSource = {
            **source,
            "canonical_url": canonical_url,
            "domain": urlsplit(canonical_url).netloc,
        }
        source_gap_ids = list(
            dict.fromkeys(
                [
                    *source.get("gap_ids", []),
                    *([source["gap_id"]] if source.get("gap_id") else []),
                ]
            )
        )
        if source_gap_ids:
            normalized_source["gap_ids"] = source_gap_ids
        seen_urls[canonical_url] = normalized_source
        if title_is_distinctive:
            seen_titles[normalized_title] = normalized_source
        unique.append(normalized_source)
    return unique


def deduplicate_sources_by_id(sources: list[ResearchSource]) -> list[ResearchSource]:
    """Preserve dimension-specific citation aliases while removing repeated IDs."""
    unique: list[ResearchSource] = []
    seen_ids: set[str] = set()
    for source in sources:
        if source["source_id"] in seen_ids:
            continue
        seen_ids.add(source["source_id"])
        unique.append(source)
    return unique


def canonicalize_url(url: str) -> str:
    """Remove fragments and common tracking parameters from a URL."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return ""
    scheme = parts.scheme.lower() or "https"
    if scheme not in {"http", "https"} or not parts.hostname:
        return ""
    hostname = (parts.hostname or "").lower()
    if port and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname
    path = parts.path.rstrip("/") or "/"
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
            and key.lower() not in TRACKING_QUERY_PARAMETERS
        )
    )
    return urlunsplit((scheme, netloc, path, query, ""))


def normalize_search_score(value: Any, default: float = 0.5) -> float:
    """Coerce an external ranking score into a finite zero-to-one value."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if not isfinite(score):
        return default
    return max(0.0, min(score, 1.0))


def normalize_title(title: str) -> str:
    """Normalize a title for deterministic duplicate detection."""
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE).strip()


def format_source_candidates(sources: list[ResearchSource]) -> str:
    """Format compact candidate metadata for source-quality assessment."""
    if not sources:
        return "No candidate sources."
    return "\n\n".join(
        f"[{source['source_id']}]\n"
        f"Title: {source['title']}\n"
        f"Domain: {source.get('domain', '')}\n"
        f"Published: {source.get('published_date') or 'unknown'}\n"
        f"Requested source types: {source.get('requested_source_types', [])}\n"
        f"Evidence expected: {source.get('expected_evidence', '')}\n"
        f"Search score: {source.get('score')}\n"
        f"Snippet: {source['content']}"
        for source in sources
    )


def format_rejected_source_summary(sources: list[ResearchSource]) -> str:
    """Format rejection reasons without returning rejected content to the model."""
    if not sources:
        return "No sources were rejected."
    return "\n".join(
        f"- [{source['source_id']}] {source['title']}: "
        f"{'; '.join(source.get('rejection_reasons', [])) or 'below quality threshold'}"
        for source in sources
    )


def format_dimension_results(
    results: list[DimensionResult],
    *,
    max_claims_per_dimension: int = 8,
    max_evidence_chars: int = 220,
) -> str:
    """Group compact audited claims and limitations by dimension."""
    if not results:
        return "No dimension research was completed."

    sections = []
    for result in sorted(results, key=lambda item: int(item["dimension"]["id"])):
        dimension = result["dimension"]
        status = result.get("completion_status") or (
            "sufficient" if result["is_sufficient"] else "loop_limit_reached"
        )
        termination_reason = result.get("termination_reason", "")
        if status in {"completed_with_limitations", "budget_exhausted"}:
            status = "completed with limitations"
            if termination_reason:
                status += f"; stop reason: {termination_reason.replace('_', ' ')}"
        elif status == "search_unavailable":
            status = "incomplete because web search was unavailable"
        claims = result.get("claims", [])[:max_claims_per_dimension]
        claim_text = (
            "\n".join(
                f"- {claim['claim']} "
                + " ".join(
                    f"[{source_id}]" for source_id in claim["supporting_source_ids"]
                )
                + (
                    f"\n  Counterevidence: {', '.join(claim['contradicting_source_ids'])}"
                    if claim["contradicting_source_ids"]
                    else ""
                )
                + (
                    f"\n  Uncertainty: {claim['uncertainty_reason']}"
                    if claim["uncertainty_reason"]
                    else ""
                )
                + "\n  Evidence: "
                + _format_claim_evidence(
                    claim.get("supporting_evidence", []), max_evidence_chars
                )
                for claim in claims
            )
            or "No auditable claims were extracted for this dimension."
        )
        sections.append(
            f"## Dimension {dimension['id']}: {dimension['title']}\n"
            f"Scope: {dimension['scope']}\n"
            f"Research status: {status} after {result['research_loop_count']} loop(s)\n"
            f"Confidence: {result.get('confidence', 0):.2f}\n"
            f"Source quality issues: {result.get('source_quality_issues', [])}\n"
            f"Unresolved gaps: {result.get('unresolved_gaps', [])}\n"
            f"Contradictions: {result.get('contradictions', [])}\n\n"
            f"Audited claims and evidence excerpts:\n{claim_text}"
        )
    return "\n\n=====\n\n".join(sections)


def _format_claim_evidence(evidence: object, max_chars: int) -> str:
    """Render verified evidence quotes while tolerating legacy string claims."""
    if isinstance(evidence, str):
        return evidence[:max_chars]
    if not isinstance(evidence, list):
        return "No verified evidence quote."
    rendered: list[str] = []
    remaining = max_chars
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", ""))
        quote = str(item.get("quote", ""))
        locator = str(item.get("locator", ""))
        block = f'[{source_id}] "{quote}" ({locator})'
        if rendered and len(block) > remaining:
            break
        block = block[:remaining]
        rendered.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break
    return " | ".join(rendered) or "No verified evidence quote."


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
            return ""
        if source_id not in used_ids:
            used_ids.append(source_id)
        safe_title = source["title"].replace("[", "").replace("]", "")
        return f"[{safe_title}]({source['url']})"

    rendered = re.sub(r"\[(S[A-Za-z0-9-]+)\]", replace, text)
    return rendered, [source_map[source_id] for source_id in used_ids]
