"""Dedicated persona asset for administrator private conversations."""
import os
from pathlib import Path

DEFAULT_PRIVATE_PERSONA=Path(__file__).resolve().parents[1]/"personas/wuwa-aemeath/persona.txt"


def load_private_persona():
    configured=os.getenv("PRIVATE_CHAT_PERSONA_FILE")
    if configured=="":
        raise RuntimeError("Private persona is not configured")
    path=Path(configured) if configured else DEFAULT_PRIVATE_PERSONA
    if not path.is_file():
        raise RuntimeError("Private persona file is unavailable")
    with path.open('r',encoding='utf-8') as source:text=source.read(20001)
    if not text.strip() or len(text)>20000:
        raise RuntimeError("Private persona must contain 1..20000 characters")
    return text
