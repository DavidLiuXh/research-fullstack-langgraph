"""DeepSeek chat-model construction utilities."""

import os

from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr


def create_deepseek_model(
    model: str,
    *,
    temperature: float = 0,
    thinking: bool = False,
    reasoning_effort: str | None = None,
    max_retries: int = 4,
    timeout: float = 180,
) -> ChatDeepSeek:
    """Create a chat model using LangChain's native DeepSeek integration."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    return ChatDeepSeek(
        model=model,
        api_key=SecretStr(api_key),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
    )
