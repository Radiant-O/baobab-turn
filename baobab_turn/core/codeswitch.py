"""Spotting when a caller switches language mid-utterance.

Switch points are where an inner detector is least reliable: its language
assumptions break exactly there, and Nigerian callers switch constantly --
English to Pidgin to Yoruba inside one sentence.

**No STT gives us this.** The spec assumed we could prefer the provider's own
per-segment language signal, and a survey of every Nigerian provider found
none that reports one in a live stream. So we have to detect it ourselves,
which is why this module exists rather than being a thin wrapper.

The method is deliberately dumb: score the head of the utterance and its tail
against small per-language wordlists carried by the packs, and flag a switch
when the winner changes. No neural language ID -- too slow for a path that
runs on every partial transcript, and unproven benefit for a signal that only
ever buys a couple of hundred milliseconds of extra patience.

What it does with a switch is equally modest. It never vetoes a turn; it asks
for a little more patience, because the one thing we know about a switch
point is that the inner detector's opinion there is worth less than usual.
"""

from __future__ import annotations

from dataclasses import dataclass

from .markers import tokenise
from .types import LanguagePack

__all__ = ["CodeSwitchDetector", "CodeSwitchResult", "LanguageProfile"]

#: Below this many tokens an utterance has no head and tail to compare, and
#: any "switch" is an artefact of having almost no text.
_MIN_TOKENS = 4

#: The head needs enough words to give a density worth comparing.
#:
#: This was missing and it produced a steady false positive on ordinary
#: English. "Yeah. We can do 8AM." is five tokens, so with a four-token
#: tail the head is the single word `yeah` -- which is in the English
#: profile, making head density exactly 1.0 and tail density exactly 0.0.
#: A shift of -1.0 clears any threshold, and the detector confidently
#: reported a switch *out of English* in the middle of an English sentence.
#:
#: One word cannot estimate a proportion. Three is still thin, but it is
#: the point where a single profile word stops being the whole measurement.
#: Below it the honest answer is that there is not enough text to say, and
#: abstaining is cheap here: a detected switch buys a couple of hundred
#: milliseconds, so a false positive costs more than a miss.
_MIN_HEAD_TOKENS = 3


@dataclass(frozen=True, slots=True)
class CodeSwitchResult:
    switched: bool
    head_language: str | None = None
    tail_language: str | None = None

    @property
    def description(self) -> str:
        """Either side may be None: we detect that the tail moved into or out
        of a language, which does not identify the other side."""
        if not self.switched:
            return "no switch"
        return f"{self.head_language or '?'} -> {self.tail_language or '?'}"


class LanguageProfile:
    """A language's characteristic vocabulary, for cheap identification.

    Built from a pack's own data: its markers and fillers are by construction
    the words most specific to that language's conversational register, which
    is exactly what discriminates it from a neighbour. A pack may also carry
    an explicit `profile_words` list when its markers alone are too thin.
    """

    __slots__ = ("code", "words")

    def __init__(self, code: str, words: set[str]) -> None:
        self.code = code
        self.words = words

    @classmethod
    def from_pack(cls, pack: LanguagePack, extra: set[str] | None = None) -> LanguageProfile:
        words: set[str] = set()
        for group in (pack.continuation_markers, pack.yield_markers, pack.ambiguous_markers):
            for marker in group:
                words.update(tokenise(marker.text))
        words.update(tokenise(" ".join(pack.fillers)))
        words.update(tokenise(" ".join(pack.profile_words)))
        if extra:
            words.update(extra)
        return cls(pack.code, words)

    def score(self, tokens: list[str]) -> float:
        """Fraction of tokens belonging to this language's vocabulary."""
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t in self.words) / len(tokens)


class CodeSwitchDetector:
    """Compares the head and tail of an utterance across language profiles.

    Profiles that share a word gain nothing from it -- a token counted for
    both English and Pidgin cannot discriminate between them, so shared
    vocabulary is removed at construction. What remains is each language's
    distinctive core, which is the only part that can identify anything.
    """

    __slots__ = ("_profiles", "min_shift", "window")

    def __init__(
        self,
        profiles: list[LanguageProfile],
        window: int = 4,
        min_shift: float = 0.35,
    ) -> None:
        self._profiles = self._make_distinctive(profiles)
        self.window = window
        #: How much a language's density must move between head and tail
        #: before it counts. Set high on purpose: a switch only buys a couple
        #: of hundred milliseconds of patience, so a false positive on every
        #: other utterance costs more than the occasional miss.
        self.min_shift = min_shift

    @staticmethod
    def _make_distinctive(profiles: list[LanguageProfile]) -> list[LanguageProfile]:
        seen: dict[str, int] = {}
        for profile in profiles:
            for word in profile.words:
                seen[word] = seen.get(word, 0) + 1
        return [
            LanguageProfile(p.code, {w for w in p.words if seen[w] == 1})
            for p in profiles
        ]

    def detect(self, transcript: str) -> CodeSwitchResult:
        """Flag a switch when a language's vocabulary density shifts sharply
        between the head of the utterance and its tail.

        Density shift rather than "identify both halves", because in this
        language pair English is the *unmarked* member: Pidgin announces
        itself with `wetin`, `dey`, `abeg`, while English is recognisable only
        by the absence of them. Asking both halves to be positively identified
        can therefore never fire on exactly the switch we most want to catch.

        Measuring the change instead is symmetric and needs no profile for the
        unmarked language: a jump in Pidgin density means a switch *into*
        Pidgin, and a collapse means a switch out of it.
        """
        tokens = tokenise(transcript)
        if len(tokens) < _MIN_TOKENS or not self._profiles:
            return CodeSwitchResult(False)

        tail_tokens = tokens[-self.window:]
        head_tokens = tokens[: -self.window] or tokens[: len(tokens) // 2]
        if len(head_tokens) < _MIN_HEAD_TOKENS or not tail_tokens:
            # Not enough before the tail to say anything about it. See
            # `_MIN_HEAD_TOKENS`: a one-word head makes every density 0 or
            # 1, and the shift between them is then guaranteed to look
            # decisive whatever the words actually were.
            return CodeSwitchResult(False)

        best_shift = 0.0
        head_code: str | None = None
        tail_code: str | None = None

        for profile in self._profiles:
            head = profile.score(head_tokens)
            tail = profile.score(tail_tokens)
            shift = tail - head
            if abs(shift) <= abs(best_shift):
                continue
            best_shift = shift
            if shift > 0:
                # Density rose: the tail moved into this language.
                head_code, tail_code = None, profile.code
            else:
                # Density fell: the tail moved out of it.
                head_code, tail_code = profile.code, None

        if abs(best_shift) < self.min_shift:
            return CodeSwitchResult(False)
        return CodeSwitchResult(True, head_code, tail_code)
