"""Matching discourse markers against the tail of a partial transcript.

This runs on every partial-transcript update, so it must be microseconds. It
is a dict lookup over a small window, not a regex sweep.

Two things worth knowing before changing this:

**Punctuation is not a signal.** Nigerian ASR output punctuates unreliably,
so "abi?" and "abi" must match identically. Everything is normalised before
comparison and no rule may depend on a question mark.

**Longest match wins.** "the thing be say" means the speaker is still going;
its final word "say" means nothing on its own. Multi-word phrases are checked
before single words, always.
"""

from __future__ import annotations

import unicodedata

from .types import LanguagePack, Marker

__all__ = ["MarkerIndex", "normalise", "tokenise"]

# Kept as a translation table so stripping is a single C-level pass.
#
# The typographic characters at the end are deliberate, not a paste error: STT
# providers emit curly quotes, em dashes and en dashes, and a marker would
# otherwise miss whenever one appeared. RUF001 flags them as "ambiguous",
# which is exactly why they need stripping.
_PUNCT = {
    ord(c): None
    for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~‘’“”—–"  # noqa: RUF001
}


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    NFKC first so that a curly apostrophe or a full-width character from an
    ASR provider does not produce a miss.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower().translate(_PUNCT)
    return " ".join(text.split())


def tokenise(text: str) -> list[str]:
    return normalise(text).split()


class MarkerIndex:
    """Longest-match-first lookup for one language pack.

    Built once per pack at load time. Lookups allocate nothing beyond the
    joined candidate strings.
    """

    __slots__ = ("_ambiguous", "_continuation", "_max_words", "_yield", "pack")

    def __init__(self, pack: LanguagePack) -> None:
        self.pack = pack
        self._continuation = self._index(pack.continuation_markers)
        self._yield = self._index(pack.yield_markers)
        self._ambiguous = self._index(pack.ambiguous_markers)
        widths = [
            len(m.text.split())
            for group in (
                pack.continuation_markers,
                pack.yield_markers,
                pack.ambiguous_markers,
            )
            for m in group
        ]
        self._max_words = max(widths, default=0)

    @staticmethod
    def _index(markers: tuple[Marker, ...]) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in markers:
            key = normalise(m.text)
            if not key:
                continue
            # A marker listed twice keeps its strongest weight rather than
            # whichever happened to be last in the file.
            out[key] = max(out.get(key, 0.0), m.weight)
        return out

    def __bool__(self) -> bool:
        """False for a pack with no markers, so the guard can skip the layer."""
        return bool(self._continuation or self._yield or self._ambiguous)

    def _match(self, tokens: list[str], table: dict[str, float]) -> tuple[str, float] | None:
        """Best match ending at the tail, preferring the longest phrase."""
        if not table or not tokens:
            return None
        span = min(self._max_words, len(tokens))
        for width in range(span, 0, -1):
            candidate = " ".join(tokens[-width:])
            weight = table.get(candidate)
            if weight is not None:
                return candidate, weight
        return None

    def continuation(self, tokens: list[str]) -> tuple[str, float] | None:
        """A marker meaning the speaker has more to say."""
        return self._match(tokens, self._continuation)

    def yields(self, tokens: list[str]) -> tuple[str, float] | None:
        """A marker meaning the speaker is done and inviting a reply."""
        return self._match(tokens, self._yield)

    def ambiguous(self, tokens: list[str]) -> tuple[str, float] | None:
        """Position-dependent markers -- `sha`, trailing `o`, `now`.

        These are only meaningful as the final token, and the guard weights
        them weakly. Never treat one as a hard hand-off.
        """
        if not tokens:
            return None
        last = tokens[-1]
        weight = self._ambiguous.get(last)
        return (last, weight) if weight is not None else None

    def window(self, transcript_tail: str, size: int) -> list[str]:
        """The last `size` tokens of the tail, normalised.

        Callers may hand us the whole accumulated utterance rather than a
        tail, and this runs on every partial-transcript update, so the raw
        string is sliced before normalising instead of after. Without that,
        cost grows with the length of the turn for a result that only ever
        depends on the last few tokens.

        The slice can cut a word in half, which would invent a token that was
        never spoken -- so when it truncates, the first token is discarded.
        """
        if size <= 0 or not transcript_tail:
            return []

        # Generous: ~40 characters per token is far more than any real word,
        # so the window is never starved by the slice.
        budget = size * 40
        truncated = len(transcript_tail) > budget
        raw = transcript_tail[-budget:] if truncated else transcript_tail

        tokens = tokenise(raw)
        if truncated and tokens:
            tokens = tokens[1:]
        return tokens[-size:]
