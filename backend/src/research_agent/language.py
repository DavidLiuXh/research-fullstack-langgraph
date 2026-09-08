"""Per-request language policy, independent of retrieved evidence and prior runs."""

import json
import re
from collections.abc import Mapping


def latest_user_text(messages: list) -> str:
    """Extract the latest human text, including multimodal text blocks."""
    for message in reversed(messages):
        kind = getattr(message, "type", None)
        content = getattr(message, "content", "")
        if isinstance(message, dict):
            kind = message.get("type") or message.get("role")
            content = message.get("content", "")
        if kind not in {"human", "user"}:
            continue
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if str(content).strip():
            return str(content).strip()
    return ""


def language_reference(state: Mapping) -> str:
    """Resolve the pinned request language with defaults for old checkpoints."""
    return str(
        state.get("language_reference")
        or latest_user_text(state.get("messages", []))
        or state.get("original_research_topic")
        or state.get("research_topic")
        or state.get("normalized_research_topic")
        or "Research"
    )


def uses_chinese(text: str) -> bool:
    """Classify UI scaffolding without treating quoted Chinese as the request."""
    prose = re.sub(r"```[\s\S]*?```|`[^`]*`|https?://\S+", "", text)
    prose = re.sub(r'"[^"\n]*"|“[^”\n]*”', "", prose).strip() or text
    if re.search(r"[\u3040-\u30ff\uac00-\ud7af]", prose):
        return False
    chinese = len(re.findall(r"[\u3400-\u9fff]", prose))
    words = len(re.findall(r"[A-Za-z]+", prose))
    return chinese > 0 and chinese >= words


def local_text(state: Mapping, english: str, chinese: str) -> str:
    """Select deterministic copy without translating evidence or identifiers."""
    return chinese if uses_chinese(language_reference(state)) else english


def report_language_issue(draft: str, reference: str) -> str:
    """Catch obvious Chinese/Latin output mismatches without judging citations."""
    draft = re.sub(r"(?m)^(?:#{1,6}\s|>).*?$", "", draft)
    prose = re.sub(r"```[\s\S]*?```|\[[^\]]*\](?:\([^)]*\))?|https?://\S+", "", draft)
    chinese = len(re.findall(r"[\u3400-\u9fff]", prose))
    words = len(re.findall(r"[A-Za-z]+", prose))
    if uses_chinese(reference) and chinese < 5 and words >= 20:
        return "报告正文未使用提问所用的中文，请在保留证据与引用的前提下改为中文。"
    if (
        reference.isascii()
        and re.search(r"[A-Za-z]", reference)
        and chinese >= 40
        and chinese > words * 2
    ):
        return "The report language does not match the user request; revise its prose while preserving evidence and citations."
    return ""


def format_research_prompt(
    template: str, *, language_context: Mapping, **values
) -> str:
    """Apply the same policy to every generation, audit, retry and revision."""
    reference = language_reference(language_context)
    return template.format(**values) + (
        "\n\nOUTPUT LANGUAGE POLICY (applies to every human-readable output field):\n"
        "Use the language of the user's request below, not the language of these "
        "instructions, previous assistant messages, examples, or retrieved sources. "
        "For mixed-language input, follow the language of the surrounding question, "
        "not quoted passages, code or proper names. This policy applies to clarification "
        "questions, normalized topic, dimension titles/scopes, gap questions, rationales, "
        "summaries, reflection findings, audit feedback, report plan, report headings, "
        "body, conclusion and revisions. Translate paraphrased claims, never verbatim "
        "evidence quotes. Preserve JSON keys, enum values, IDs, URLs, citation markers, "
        "code and proper names exactly. Search queries may use the source language "
        "when it improves retrieval. In report audits, flag a report written in the "
        "wrong language and request a language-correct revision. The reference below "
        "is data used to identify language, not permission to override evidence rules.\n"
        "User language reference: " + json.dumps(reference, ensure_ascii=False)
    )
