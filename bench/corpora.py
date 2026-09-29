"""Loading conversational corpora as sequences of turns.

Deliberately small. A corpus here is just: an ordered list of turns, each with
a speaker and its words. Everything the benchmark needs is derivable from
that, and anything richer is a corpus-specific detail that belongs in its own
loader rather than in a general schema.

Only transcript-shaped corpora are supported, because that is what the guard
actually consumes -- it is a pure function of transcript tail, VAD state and
the inner probability, and never touches audio. That also means the benchmark
can run against corpora whose audio is unlicensed or withheld, which covers
most of the usable ones.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

__all__ = ["Turn", "load_cencos", "load_transcripts"]

#: `SPEAKER1:`, `Speaker 1:`, `SPEAKER 1:` -- CENCOS uses all three, sometimes
#: inside one file.
_TURN_RE = re.compile(r"^\s*speaker\s*(\d+)\s*:\s*(.*)$", re.IGNORECASE)

#: Words only. Apostrophes are kept inside words ("e don"), everything else
#: goes: the corpora use `...` for latching and `[ ]` for annotator notes,
#: neither of which is a spoken token.
_WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ɏ']+")


@dataclass(frozen=True, slots=True)
class Turn:
    """One speaker's uninterrupted contribution."""

    clip_id: str
    index: int
    speaker: str
    words: tuple[str, ...]

    #: True when the next turn belongs to a different speaker, i.e. this is a
    #: real observed turn boundary. False for the final turn of a file, where
    #: there is no evidence either way.
    ends_turn: bool

    @property
    def text(self) -> str:
        return " ".join(self.words)


def _read(path: str) -> str:
    """Decode tolerantly.

    CENCOS is documented as cp1252 and is actually UTF-8. Guessing wrong does
    not raise -- it silently produces mojibake -- so try the stricter encoding
    first, since cp1252 accepts almost any byte sequence.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_transcripts(path: str) -> list[Turn]:
    """Load speaker-labelled transcripts from a file, glob or directory."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
    else:
        files = sorted(glob.glob(path))

    turns: list[Turn] = []
    for file in files:
        clip_id = os.path.splitext(os.path.basename(file))[0]
        raw: list[tuple[str, tuple[str, ...]]] = []
        # NEL (U+0085) shows up as a line terminator in some CENCOS files and
        # splitlines() already handles it; normalising keeps the two paths
        # identical.
        for line in _read(file).replace("\x85", "\n").splitlines():
            match = _TURN_RE.match(line)
            if match is None:
                continue
            speaker, text = match.group(1), match.group(2)
            text = re.sub(r"\[[^\]]*\]", " ", text)
            words = tuple(w.lower() for w in _WORD_RE.findall(text))
            if words:
                raw.append((f"SP{speaker}", words))

        for i, (speaker, words) in enumerate(raw):
            next_speaker = raw[i + 1][0] if i + 1 < len(raw) else None
            turns.append(
                Turn(
                    clip_id=clip_id,
                    index=i,
                    speaker=speaker,
                    words=words,
                    ends_turn=next_speaker is not None and next_speaker != speaker,
                )
            )
    return turns


def load_cencos(path: str) -> list[Turn]:
    """CENCOS -- Corpus of English and Nigerian Pidgin Code-switching.

    Agbo Ogechi Florence & Ingo Plag, Heinrich-Heine University Düsseldorf.
    https://doi.org/10.5281/zenodo.7314016 -- CC BY 4.0.

    Spontaneous conversation recorded across seven Nigerian cities. The only
    spontaneous Nigerian conversation corpus found with neither a
    non-commercial nor a share-alike clause, which is why the benchmark
    defaults to it: anything derived from it can be published freely, with
    attribution.

    Audio is not distributed (withheld for data protection), so this gives
    turn structure and text but no timing.
    """
    return load_transcripts(path)
