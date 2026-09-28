"""Sentence-by-sentence release of a streaming LLM reply (HA-free, unit-testable).

2026-09-28 (v8.8.0): PureLLM used to hold the whole reply until the model
finished, so TTS could not start until the last token. For the synthesis turn
after a tool call, SpokenSentenceStreamer releases each completed sentence as
soon as it exists, applying the same defenses _sanitize_llm_response applies to
whole replies:

- emoji stripped,
- a sentence already said verbatim in this reply is dropped (loop guard),
- the canned "didn't catch that" reply ends the stream,
- a hard character cap ends the stream at a sentence boundary,
- any "<" (leaked tool-call markup, stray think tags) HOLDS the rest so the
  caller can fall back to its whole-text path; nothing after it is released.

Released text is only what will be spoken; the caller never re-sends it.
"""
from __future__ import annotations

import re

# A sentence ends at . ! ? … (plus closing quotes/brackets) followed by whitespace,
# or at a newline. A period inside "45.5" or "p.m." without trailing space never splits.
_RE_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*\s+|\n+")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")
_GARBLE_MARKER = "sorry, i didn't catch that"


class SpokenSentenceStreamer:
    """Feed streamed text in, get speakable sentences out."""

    def __init__(self, *, max_chars: int, garbled_reply: str, emoji_re: re.Pattern | None = None) -> None:
        self._max_chars = max_chars
        self._garbled_reply = garbled_reply
        self._emoji_re = emoji_re
        self._seen: set[str] = set()
        self.buffer = ""
        self.emitted_chars = 0
        self.emitted_any = False
        self.done = False  # cap or garble reached: release nothing more
        self.held = False  # "<" seen: caller handles self.buffer itself

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return sentences that are now complete and speakable."""
        self.buffer += text or ""
        if self.done or self.held:
            return []
        if "<" in self.buffer:
            self.held = True
            return []
        out: list[str] = []
        while not self.done:
            match = _RE_SENTENCE_END.search(self.buffer)
            if not match:
                break
            sentence, self.buffer = self.buffer[: match.end()], self.buffer[match.end():]
            out.extend(self._release(sentence))
        return out

    def finish(self) -> list[str]:
        """Release the final, unterminated sentence (if any)."""
        if self.done or self.held or not self.buffer.strip():
            return []
        tail, self.buffer = self.buffer, ""
        return self._release(tail)

    def _release(self, sentence: str) -> list[str]:
        if self._emoji_re is not None:
            sentence = self._emoji_re.sub("", sentence)
        key = _RE_MULTI_SPACE.sub(" ", sentence).strip().casefold()
        if not key:
            return []
        if _GARBLE_MARKER in key.replace("’", "'"):
            self.done = True
            if self.emitted_any:
                return []
            self.emitted_any = True
            return [self._garbled_reply]
        if key in self._seen:
            return []
        if self.emitted_chars + len(sentence.strip()) > self._max_chars:
            self.done = True
            return []
        self._seen.add(key)
        self.emitted_chars += len(sentence.strip())
        self.emitted_any = True
        return [sentence]
