"""Reporting contracts for time ranges, comparable metrics and editorial evidence."""

import calendar
import json
import re
from datetime import date


def unfinished_period(topic: str, today: date) -> str:
    """Identify an explicitly requested year-to-date period that has not ended."""
    match = re.search(
        r"(20\d{2})\s*年?\s*(?:前|1\s*[-—至到]\s*)([1-9]|1[0-2])\s*个?月", topic
    )
    if not match:
        match = re.search(r"first\s+(\d{1,2})\s+months?\s+of\s+(20\d{2})", topic, re.I)
        if not match:
            return ""
        month, year = map(int, match.groups())
    else:
        year, month = map(int, match.groups())
    if not 1 <= month <= 12:
        return ""
    end = date(year, month, calendar.monthrange(year, month)[1])
    return end.isoformat() if end >= today else ""


def metric_scope(text: str) -> dict[str, list[str]]:
    """Extract only explicit scope cues; unknown scope is never assumed equal."""
    groups = {
        "measure": {
            "production": r"产量|生产|production",
            "sales": r"销量|销售|sales",
            "utilization": r"利用率|utilization",
        },
        "channel": {
            "retail": r"零售|retail",
            "wholesale": r"批发|wholesale",
            "export": r"出口|exports?",
            "domestic": r"国内销量|国内销售|domestic sales",
        },
        "population": {
            "passenger": r"乘用车|passenger",
            "commercial": r"商用车|commercial vehicle",
            "nev": r"新能源|new energy",
            "bev": r"纯电|battery electric",
            "ice": r"燃油|内燃机|传统汽车|internal combustion",
        },
        "basis": {
            "forecast": r"预计|预测|forecast|projected",
            "actual": r"实际|actual",
        },
        "geography": {
            "china": r"中国|国内|China|Chinese",
            "global": r"全球|global|worldwide",
        },
    }
    result = {
        key: [
            name for name, pattern in entries.items() if re.search(pattern, text, re.I)
        ]
        for key, entries in groups.items()
    }
    result["period"] = re.findall(
        r"20\d{2}年(?:第[一二三四]季度|上半年|下半年|\d{1,2}月)?", text
    )
    return result


def comparison_scope_note(left: str, right: str) -> str:
    """Block numeric contradiction claims when explicit scope is incompatible."""
    if not (re.search(r"\d", left) and re.search(r"\d", right)):
        return ""
    a, b = metric_scope(left), metric_scope(right)
    for field in ("measure", "channel", "population", "basis", "period", "geography"):
        if a[field] and b[field] and set(a[field]).isdisjoint(b[field]):
            return f"Different or compound {field} scopes require metric-level reconciliation; these statements are not established as mutually exclusive."
    if bool(a["channel"]) != bool(b["channel"]):
        return "The sales channel is unspecified on one side; comparability has not been established."
    if bool(a["population"]) != bool(b["population"]):
        return "Vehicle coverage is unspecified on one side; comparability has not been established."
    return ""


def writing_evidence(results: list[dict]) -> str:
    """Provide facts and material limitations without operational diagnostics."""
    packets = []
    for result in results:
        packets.append(
            {
                "research_dimension": result["dimension"],
                "claims": [
                    {
                        "claim_id": claim.get("claim_id", ""),
                        "claim": claim["claim"],
                        "source_ids": claim.get("supporting_source_ids", []),
                        "allowed_citation_markers": [
                            f"[{source_id}]"
                            for source_id in claim.get("supporting_source_ids", [])
                        ],
                        "evidence": claim.get("supporting_evidence", []),
                        "uncertainty": claim.get("uncertainty_reason", ""),
                        "metric_scope": metric_scope(claim["claim"]),
                    }
                    for claim in result.get("claims", [])
                ],
                "unanswered_questions": [
                    gap.get("question", "") for gap in result.get("unresolved_gaps", [])
                ],
            }
        )
    return json.dumps(packets, ensure_ascii=False)


def editorial_packets(results: list[dict], plan: dict) -> list[dict]:
    """Build chapter evidence from validated global claim allocations."""
    catalog = {
        c.get("claim_id"): (c, r)
        for r in results
        for c in r.get("claims", [])
        if c.get("claim_id")
    }
    packets = []
    for section in plan.get("sections", []):
        selected = [
            catalog[cid] for cid in section.get("claim_ids", []) if cid in catalog
        ]
        if not selected:
            continue
        origins = {r["dimension"]["id"]: r for _, r in selected}
        packet = dict(selected[0][1])
        packet.update(
            {
                "dimension": {
                    "id": section["dimension_id"],
                    "title": section.get("title")
                    or selected[0][1]["dimension"]["title"],
                    "scope": section["objective"],
                },
                "claims": [c for c, _ in selected],
                "sources": list(
                    {
                        s["source_id"]: s
                        for r in origins.values()
                        for s in r.get("sources", [])
                    }.values()
                ),
                "unresolved_gaps": [
                    g for r in origins.values() for g in r.get("unresolved_gaps", [])
                ],
            }
        )
        packets.append(packet)
    if packets:
        return packets
    # Old checkpoints have chapter order but no allocations.
    by_id = {str(r["dimension"]["id"]): r for r in results}
    ordered = [
        by_id[s["dimension_id"]]
        for s in plan.get("sections", [])
        if s.get("dimension_id") in by_id
    ]
    included = {r["dimension"]["id"] for r in ordered}
    return ordered + [r for r in results if r["dimension"]["id"] not in included]
