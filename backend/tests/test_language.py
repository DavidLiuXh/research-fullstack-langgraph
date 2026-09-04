"""Language policy and deterministic scaffolding regression tests."""

import ast
import inspect

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from research_agent.language import (
    format_research_prompt,
    language_reference,
    latest_user_text,
    report_language_issue,
    uses_chinese,
)


@pytest.mark.parametrize(
    "text, chinese",
    [
        ("请调研 DeepSeek 和 OpenAI 的技术差异", True),
        ('Explain the policy called "中国制造2025" in detail.', False),
        ("请研究 AI", True),
        ("Compare DeepSeek and OpenAI", False),
        ("Explain this code: ```python\n标题 = '你好'```", False),
        ("日本の研究について教えてください", False),
    ],
)
def test_ui_language_ignores_foreign_quotes_and_code(text, chinese):
    assert uses_chinese(text) is chinese


def test_latest_human_controls_new_run_not_old_answer():
    from research_agent.graph import initialize_research_topic

    messages = [
        HumanMessage(content="请研究市场"),
        AIMessage(content="中文报告" * 300),
        HumanMessage(content="Now investigate the risks"),
    ]
    state = initialize_research_topic(
        {"messages": messages, "language_reference": "请研究市场"}
    )
    assert state["language_reference"] == "Now investigate the risks"
    assert language_reference(state) == "Now investigate the risks"


def test_multimodal_message_and_legacy_checkpoint():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {}},
                {"type": "text", "text": "请研究能源政策"},
            ],
        }
    ]
    assert latest_user_text(messages) == "请研究能源政策"
    assert language_reference({"messages": messages}) == "请研究能源政策"
    assert language_reference({"research_topic": "Legacy request"}) == "Legacy request"


def test_language_policy_preserves_schema_and_quotes():
    prompt = format_research_prompt(
        "Evidence: {evidence}",
        language_context={"language_reference": "请调研能源政策"},
        evidence='{"status": "closed", "quote": "Original English evidence"}',
    )
    assert '"quote": "Original English evidence"' in prompt
    assert 'User language reference: "请调研能源政策"' in prompt
    assert "Preserve JSON keys, enum values, IDs" in prompt
    assert "never verbatim evidence quotes" in prompt
    assert "Search queries may use the source language" in prompt


def test_every_graph_prompt_uses_shared_language_policy():
    import importlib

    graph_module = importlib.import_module("research_agent.graph")
    tree = ast.parse(inspect.getsource(graph_module))
    wrapped = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Name)
        ):
            assert not node.func.value.id.endswith("_instructions")
        if isinstance(node.func, ast.Name) and node.func.id == "format_research_prompt":
            wrapped.append(node)
            assert any(keyword.arg == "language_context" for keyword in node.keywords)
    assert len(wrapped) == 20


def test_obvious_report_language_mismatch():
    assert report_language_issue("# 调研报告\n\n" + "English report findings and evidence. " * 20, "请分析行业风险")
    assert report_language_issue(
        "English report findings and evidence. " * 20, "请分析行业风险"
    )
    assert report_language_issue(
        "这里是很长的中文分析结论。" * 20, "Analyze industry risks"
    )
    assert not report_language_issue(
        "这是正文，保留 OpenAI 名称与 [S1] 引用。" * 20, "请分析行业风险"
    )
    assert not report_language_issue("Analysis of 日本政策. " * 20, "Analyze policy")
    assert not report_language_issue(
        "日本の研究についての結論。" * 20, "日本の研究について教えてください"
    )
