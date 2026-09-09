"""A timestamped buffer holding the most recent transcript text.

Pure Python, no framework, no I/O. It exists because on LiveKit the
end-of-turn probability and the transcript arrive through two completely
different channels: the detector is fed audio frames and never sees text,
while the transcript comes from a session event. Something has to join them,
and this is that something.

**The timestamp is the whole point.** Nothing in the SDK correlates the text
we hold to the audio a probability was computed over, and there are three
independent ways it can go wrong -- the STT round-trip lags the detector's
own inference, STT is fed a silence frame during agent speech while the
detector gets real audio, and the transcript event is suppressed outright
while the agent is speaking and replayed later. A tail with no age attached
looks authoritative in all three cases. With an age attached, the guard can
decline to use it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

__all__ = ["TailSnapshot", "TranscriptTail"]

#: Only the end of an utterance can matter to a marker match, so the buffer is
#: capped. Generous enough for any multi-word marker plus context.
_MAX_CHARS = 512


@dataclass(frozen=True, slots=True)
class TailSnapshot:
    text: str
    age_ms: float | None
    is_final: bool
    language: str | None


class TranscriptTail:
    """Holds the latest transcript text with the time it arrived.

    Not locked. On LiveKit every writer and the single reader run on the same
    event loop, and neither `push_audio` nor the `predict()` call site has an
    await point, so they cannot interleave. If that ever stops being true this
    needs revisiting rather than a lock bolted on.
    """

    __slots__ = ("_at", "_is_final", "_language", "_now", "_text")

    def __init__(self, now=time.monotonic) -> None:
        # Injectable so tests can control age without sleeping. Monotonic
        # rather than wall clock: we only ever measure elapsed time, and a
        # clock adjustment mid-call must not make a fresh tail look stale.
        self._now = now
        self._text: str = ""
        self._at: float | None = None
        self._is_final: bool = False
        self._language: str | None = None

    def update(
        self,
        text: str,
        *,
        is_final: bool = False,
        language: str | None = None,
    ) -> None:
        """Record new transcript text. Empty text is ignored, not stored.

        An empty interim result is a normal thing for an STT to emit and does
        not mean the previous text is void -- only `reset` means that.
        """
        if not text or not text.strip():
            return
        self._text = text[-_MAX_CHARS:]
        self._at = self._now()
        self._is_final = is_final
        if language:
            self._language = language

    def reset(self) -> None:
        """Forget the current text. Call this at a turn boundary.

        On LiveKit the hook for this is `flush(reason=...)`, never
        `cancel_inference` -- the latter means "this pause turned out not to
        be the end", not "the utterance is over". Clearing on the wrong one
        would wipe the tail mid-sentence every time a caller paused and
        carried on.
        """
        self._text = ""
        self._at = None
        self._is_final = False

    @property
    def language(self) -> str | None:
        """Last language reported by the STT. Survives `reset`, since the
        caller's language does not change just because a turn ended."""
        return self._language

    def set_language(self, language: str | None) -> None:
        if language:
            self._language = language

    def snapshot(self) -> TailSnapshot:
        """The current text and how old it is, in milliseconds."""
        age = None if self._at is None else (self._now() - self._at) * 1000.0
        return TailSnapshot(
            text=self._text,
            age_ms=age,
            is_final=self._is_final,
            language=self._language,
        )
