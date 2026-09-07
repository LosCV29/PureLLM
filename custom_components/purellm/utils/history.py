"""History folding for action turns. No Home Assistant imports — unit-testable standalone.

Why this exists (2026-09-06): replaying a text-only assistant turn such as
"Added tomatoes to the shopping list. Anything else?" as a role message makes
Ornith-1.5 imitate it under tool_choice="required" — it writes the spoken
confirmation first, never reaches the tool call, and repeats the sentence
until max_tokens. Folding the prior exchange into the user message gives the
model the context without a template to imitate: immediate clean tool call.
"""
from __future__ import annotations


def fold_history_into_user(history: list[dict] | None, user_text: str) -> str | None:
    """Return a single user-message string carrying the prior exchange, or None.

    None means "do not fold" (no history, or history contains non-string content
    such as Anthropic content blocks) — the caller then replays history as
    role messages.
    """
    if not history:
        return None
    if not all(isinstance(m.get("content"), str) for m in history):
        return None
    parts = []
    for m in history:
        who = "the user said" if m.get("role") == "user" else "you replied"
        parts.append(f'{who} "{m["content"].strip()}"')
    return "[Context: " + " and ".join(parts) + "]\nThe user now says: " + user_text
