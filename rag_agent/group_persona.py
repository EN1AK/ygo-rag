"""Load the original persona as an independent, unmodified text asset."""
import os
from pathlib import Path

DEFAULT_PERSONA = Path(__file__).resolve().parents[1]/"personas/deepseek-group/persona.txt"


def load_persona():
    configured = os.getenv("GROUP_CHAT_PERSONA_FILE")
    if configured == "":
        return ""
    path = Path(configured) if configured else DEFAULT_PERSONA
    if not path.is_file():
        if configured:
            raise RuntimeError("Configured group persona is unavailable")
        return ""
    with path.open("r",encoding="utf-8") as source:
        text = source.read(20001)
    if len(text)>20000:
        raise RuntimeError("Group persona exceeds text limit")
    return text
