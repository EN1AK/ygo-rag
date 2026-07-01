from __future__ import annotations

from rag_agent.config import Settings


def create_deepseek_chat_model(settings: Settings):
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required to call DeepSeek.")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is required for DeepSeek chat integration. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from exc
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        timeout=settings.deepseek_timeout_seconds,
    )
