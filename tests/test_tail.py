"""The transcript buffer.

Small, but it is the thing standing between the guard and a confidently wrong
decision: on LiveKit the probability and the transcript arrive through
unrelated channels with no alignment guarantee, so the age this buffer
records is what lets the guard decline to use stale text.
"""

from __future__ import annotations

from baobab_turn.core.tail import TranscriptTail


class Clock:
    """Controllable monotonic clock, so ageing is tested without sleeping."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_starts_empty_with_no_age() -> None:
    snap = TranscriptTail().snapshot()
    assert snap.text == ""
    assert snap.age_ms is None
    assert not snap.is_final
    assert snap.language is None


def test_records_text_and_flags() -> None:
    tail = TranscriptTail()
    tail.update("e don finish abi", is_final=True, language="pcm")
    snap = tail.snapshot()
    assert snap.text == "e don finish abi"
    assert snap.is_final
    assert snap.language == "pcm"


def test_age_grows_with_the_clock() -> None:
    clock = Clock()
    tail = TranscriptTail(now=clock)
    tail.update("abi")
    assert tail.snapshot().age_ms == 0.0
    clock.advance(0.25)
    assert tail.snapshot().age_ms == 250.0
    clock.advance(1.0)
    assert tail.snapshot().age_ms == 1250.0


def test_a_new_update_resets_the_age() -> None:
    """Each interim result is fresh evidence, so the clock restarts."""
    clock = Clock()
    tail = TranscriptTail(now=clock)
    tail.update("the thing")
    clock.advance(2.0)
    assert tail.snapshot().age_ms == 2000.0
    tail.update("the thing be say")
    assert tail.snapshot().age_ms == 0.0


def test_blank_updates_are_ignored_not_stored() -> None:
    """An empty interim result is normal for an STT and does not mean the
    previous text is void. Only reset() means that -- storing the blank would
    silently destroy a usable tail."""
    tail = TranscriptTail()
    tail.update("e don finish abi", is_final=True)
    for blank in ("", "   ", "\n\t"):
        tail.update(blank)
        assert tail.snapshot().text == "e don finish abi"


def test_reset_clears_text_and_age() -> None:
    tail = TranscriptTail()
    tail.update("abi", is_final=True, language="pcm")
    tail.reset()
    snap = tail.snapshot()
    assert snap.text == ""
    assert snap.age_ms is None
    assert not snap.is_final


def test_reset_keeps_the_language() -> None:
    """The caller's language does not change just because a turn ended, and
    re-deriving it from the next partial would leave the first prediction of
    every turn with no pack."""
    tail = TranscriptTail()
    tail.update("abi", language="pcm")
    tail.reset()
    assert tail.snapshot().language == "pcm"


def test_language_is_sticky_against_missing_values() -> None:
    """Interim events often carry no language. That must not erase a known
    one."""
    tail = TranscriptTail()
    tail.update("abi", language="pcm")
    tail.update("abi o", language=None)
    assert tail.snapshot().language == "pcm"


def test_set_language_works_independently() -> None:
    tail = TranscriptTail()
    tail.set_language("en-NG")
    assert tail.snapshot().language == "en-NG"
    tail.set_language(None)
    assert tail.snapshot().language == "en-NG"


def test_text_is_bounded_and_keeps_the_end() -> None:
    """Only the end of an utterance can match a marker, and an unbounded
    buffer would grow for the whole turn."""
    tail = TranscriptTail()
    tail.update(("blah " * 5000) + "e don finish abi")
    text = tail.snapshot().text
    assert len(text) <= 512
    assert text.endswith("e don finish abi")


def test_is_final_downgrades_on_a_later_interim() -> None:
    """A final result followed by more speech means the turn continued. The
    flag must follow the latest evidence, or an ambiguous marker would keep
    counting after it stopped being terminal."""
    tail = TranscriptTail()
    tail.update("e don finish", is_final=True)
    assert tail.snapshot().is_final
    tail.update("e don finish o", is_final=False)
    assert not tail.snapshot().is_final
