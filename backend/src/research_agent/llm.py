"""DeepSeek chat-model construction utilities."""

import os

from langchain_deepseek import ChatDeepSeek


def create_deepseek_model(
    model: str, *, temperature: float = 0, thinking: bool = False
) -> ChatDeepSeek:
    """Create a chat model using LangChain's native DeepSeek integration."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    return ChatDeepSeek(
        model=model,
        api_key=api_key,
        api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=temperature,
        max_retries=2,
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )
