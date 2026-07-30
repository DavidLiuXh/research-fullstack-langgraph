import os

from langchain_openai import ChatOpenAI


def create_deepseek_model(
    model: str, *, temperature: float = 0, thinking: bool = False
) -> ChatOpenAI:
    """Create a DeepSeek chat model through its OpenAI-compatible API."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=temperature,
        max_retries=2,
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )
