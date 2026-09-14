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
    reporting_policy = (
        (
            "\nREPORTING CONTRACT: Statistical comparisons must state period, population, "
            "geography, unit, retail/wholesale/production/export channel and actual versus "
            "forecast. Never compare different scopes as contradictions. Never fill a "
            "requested period with older observations without explicitly narrowing the "
            "reported cutoff. Use a compact table for requested year-over-year and company "
            "comparisons (current value, prior comparable value, change, scope, source); "
            "mark unavailable cells honestly. Keep forecasts separate from actuals. "
            "Tables belong ONLY in the relevant body chapter, each table once. "
            "Never put tables in an executive overview or conclusion. Other chapters "
            "refer to the comparison rather than recreating it. Historical values "
            "must retain their actual year; a multi-year benchmark is not a prior-year value. "
            "A month and its containing quarter/half-year are overlapping observations, "
            "not a sequential trend. Never infer acceleration from their growth rates. "
            "Domestic penetration and total-market penetration (including exports), or "
            "passenger vehicles and all vehicles, must not be joined into a time series. "
            "A rising share alone does not prove substitution or absolute growth. "
            "Production minus wholesale sales does not establish retail demand. "
            "Explain mechanisms linking verified changes to potential drivers, distinguish "
            "supported causality from hypotheses and consider alternative explanations. "
            "Analytical synthesis is allowed; new unsupported facts are not. Consolidate "
            "material limitations once in the body or a single scope note; summaries "
            "need at most one scope sentence, not another inventory of limitations. "
            "Omit loop counts, confidence scores and raw gap "
            "registries from reader-facing prose. Research completion is not fact confidence.\n"
            "EVIDENCE BOUNDARY: The editorial thesis, previous chapters and completed body "
            "are provisional writing context, NOT additional evidence. Correct any unsupported "
            "thesis instead of inventing a causal story to satisfy it. Hypotheses must be "
            "explicitly conditional in the same passage and retain that qualification in "
            "the overview and conclusion. Do not invent costs, brand-level effects, inventory, "
            "capacity breakdowns or policy mechanisms absent from the evidence. State what "
            "the policy comparison baseline is: reduced exemptions can be a withdrawal "
            "of support year-on-year, not a new stimulus relative to the prior regime. "
            "Do not infer policy scope from a cap applying to only one category. State what "
            "additional observation could test a proposed mechanism. Connect paragraphs through "
            "the research question, not through unsupported causal certainty. Copy citation "
            "markers exactly from each claim's allowed_citation_markers/source_ids; never "
            "append suffixes or invent IDs. Summaries also need citations for factual claims.\n"
            "This governs report content; planning and audit responses must still use their exact JSON schemas.\n"
        )
        if any(
            key in values for key in ("report_plan", "draft_report", "claim_catalog")
        )
        else ""
    )
    return (
        template.format(**values)
        + reporting_policy
        + (
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
    )
