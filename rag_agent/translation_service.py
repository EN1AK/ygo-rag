from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_agent.config import Settings
from rag_agent.llm import create_deepseek_chat_model


DEFAULT_SOURCE_LANG = "auto"
DEFAULT_TARGET_LANG = "zh-CN"


@dataclass(frozen=True)
class TranslationRequest:
    text: str
    source_lang: str = DEFAULT_SOURCE_LANG
    target_lang: str = DEFAULT_TARGET_LANG
    structured_max_block_chars: int | None = None


@dataclass(frozen=True)
class TranslationResponse:
    translation: str
    source_lang: str
    target_lang: str
    warnings: list[str]
    structured: dict[str, Any]


def normalize_translation_request(request: TranslationRequest) -> TranslationRequest:
    text = request.text.strip()
    if not text:
        raise ValueError("text is required.")
    source_lang = (request.source_lang or DEFAULT_SOURCE_LANG).strip() or DEFAULT_SOURCE_LANG
    target_lang = (request.target_lang or DEFAULT_TARGET_LANG).strip() or DEFAULT_TARGET_LANG
    return TranslationRequest(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        structured_max_block_chars=request.structured_max_block_chars,
    )


def build_translation_prompt(request: TranslationRequest) -> str:
    normalized = normalize_translation_request(request)
    return "\n".join(
        [
            "You are a precise translation engine.",
            f"Source language: {normalized.source_lang}",
            f"Target language: {normalized.target_lang}",
            "",
            "Translate only the text between <text> and </text>.",
            "Preserve the original meaning and formatting where practical.",
            "For game or trading-card text, keep terminology consistent and concise.",
            "Do not add explanations, notes, markdown fences, or commentary.",
            "",
            "<text>",
            normalized.text,
            "</text>",
        ]
    )


def execute_translation(
    request: TranslationRequest,
    settings: Settings,
) -> TranslationResponse:
    normalized = normalize_translation_request(request)
    llm = create_deepseek_chat_model(settings)
    response = llm.invoke(build_translation_prompt(normalized))
    content = getattr(response, "content", response)
    translation = str(content).strip()
    return build_translation_response(
        translation=translation,
        source_lang=normalized.source_lang,
        target_lang=normalized.target_lang,
        max_block_chars=normalized.structured_max_block_chars,
    )


def build_translation_response(
    *,
    translation: str,
    source_lang: str = DEFAULT_SOURCE_LANG,
    target_lang: str = DEFAULT_TARGET_LANG,
    max_block_chars: int | None = None,
) -> TranslationResponse:
    warnings: list[str] = []
    structured = build_structured_translation_response(
        translation,
        source_lang=source_lang,
        target_lang=target_lang,
        max_block_chars=max_block_chars,
    )
    return TranslationResponse(
        translation=translation,
        source_lang=source_lang,
        target_lang=target_lang,
        warnings=warnings,
        structured=structured,
    )


def build_structured_translation_response(
    translation: str,
    *,
    source_lang: str,
    target_lang: str,
    max_block_chars: int | None = None,
) -> dict[str, Any]:
    parts = split_translation_text(translation, max_block_chars)
    split = len(parts) > 1
    blocks = [
        {
            "type": "translation",
            "index": index,
            "text": text,
            "truncated": False,
            "split": split,
            "fields": {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "part": index,
                "total_parts": len(parts),
            },
        }
        for index, text in enumerate(parts, start=1)
    ]
    return {
        "version": 1,
        "summary": {
            "block_count": len(blocks),
            "split": split,
            "truncated": False,
        },
        "blocks": blocks,
    }


def split_translation_text(text: str, max_chars: int | None) -> list[str]:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            parts.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, max_chars + 1)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return parts or [""]


def translation_response_to_dict(response: TranslationResponse) -> dict[str, Any]:
    return {
        "translation": response.translation,
        "source_lang": response.source_lang,
        "target_lang": response.target_lang,
        "warnings": response.warnings,
        "structured": response.structured,
    }
