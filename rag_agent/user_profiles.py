"""Operator-maintained per-user persona context, selected from a scoped session."""
from dataclasses import dataclass
import os
from pathlib import Path
import re

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "personas/user-settings.md"
SESSION = re.compile(r"qq:\d+:(?:(group):[1-9]\d*:user:|private:user:)(\d+)")


@dataclass(frozen=True)
class UserProfile:
    rules: str = ""
    knowledge: str = ""


def load_user_profile(session_id):
    match = SESSION.fullmatch(session_id)
    if not match:
        raise ValueError("Invalid user profile session")
    path = Path(os.getenv("USER_SETTINGS_FILE", str(DEFAULT_FILE)))
    if not path.is_file():
        return UserProfile()
    with path.open(encoding="utf-8") as source:
        text = source.read(65537)
    if len(text) > 65536:
        raise ValueError("User settings document exceeds 65536 characters")
    entries = re.split(r"(?m)^## 用户 ([1-9]\d*)[ \t]*$", text)
    selected = None
    seen = set()
    for index in range(1, len(entries), 2):
        user, body = entries[index:index + 2]
        if user in seen:
            raise ValueError("Duplicate user settings entry")
        seen.add(user)
        if user == match[2]:
            selected = body
    if selected is None:
        return UserProfile()
    sections = re.split(r"(?m)^### (口吻|知识|群聊口吻|群聊知识|私聊口吻|私聊知识)[ \t]*$", selected)
    values = {}
    for index in range(1, len(sections), 2):
        name, body = sections[index:index + 2]
        if name in values:
            raise ValueError("Duplicate user settings section")
        values[name] = body.strip()
    scope = "群聊" if match[1] else "私聊"
    def field(name):
        value = "\n".join(x for x in (values.get(name, ""), values.get(scope + name, "")) if x)
        if len(value) > 4000:
            raise ValueError("User settings field exceeds 4000 characters")
        return value
    return UserProfile(field("口吻"), field("知识"))
