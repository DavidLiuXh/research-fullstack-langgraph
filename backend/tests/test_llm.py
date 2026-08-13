"""Tests for the DeepSeek model factory."""

import pytest
from pydantic import SecretStr

from research_agent import llm


def test_create_deepseek_model_uses_native_integration(monkeypatch):
    """Configure ChatDeepSeek with the expected endpoint and model options."""
    captured = {}

    class FakeChatDeepSeek:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example/api")
    monkeypatch.setattr(llm, "ChatDeepSeek", FakeChatDeepSeek)

    model = llm.create_deepseek_model("deepseek-chat", temperature=0.25, thinking=True)

    assert isinstance(model, FakeChatDeepSeek)
    assert captured == {
        "model": "deepseek-chat",
        "api_key": SecretStr("test-key"),
        "base_url": "https://deepseek.example/api",
        "temperature": 0.25,
        "max_retries": 4,
        "timeout": 180,
        "extra_body": {"thinking": {"type": "enabled"}},
    }


def test_create_deepseek_model_requires_api_key(monkeypatch):
    """Fail early when DeepSeek credentials are missing."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is not set"):
        llm.create_deepseek_model("deepseek-chat")
