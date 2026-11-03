"""Pins the shape of the Pipecat API the adapter depends on.

Unlike the LiveKit case, nothing here is private -- `BaseTurnAnalyzer` is a
documented extension point, so this is a smaller bet. It is still a bet:
1.15.0 moved the analyzer off `TransportParams` and onto the user-turn stop
strategy, which means the attachment instructions in a tutorial written for
an earlier version are now wrong. CI should tell us the day that happens
again, rather than somebody's production call telling them.

A failure here is not necessarily a bug in our code. Read the SDK diff
first, then decide whether the adapter or the assumption moved.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("pipecat", reason="needs pipecat-ai")

from pipecat.audio.turn import base_turn_analyzer as bta


def test_the_base_class_and_state_enum_still_exist() -> None:
    """If these are renamed, the adapter targets nothing."""
    assert hasattr(bta, "BaseTurnAnalyzer")
    assert hasattr(bta, "EndOfTurnState")
    assert hasattr(bta, "BaseTurnParams")


def test_the_two_verdicts_are_still_the_whole_vocabulary() -> None:
    """The adapter maps a probability onto exactly these. A third member
    would mean a case we silently do not handle."""
    assert {m.name for m in bta.EndOfTurnState} == {"COMPLETE", "INCOMPLETE"}


def test_append_audio_still_takes_bytes_and_a_speech_flag() -> None:
    """Both arguments are load-bearing: the bytes feed the pitch sensor and
    the flag is where pacing comes from on this host."""
    sig = inspect.signature(bta.BaseTurnAnalyzer.append_audio)
    assert list(sig.parameters) == ["self", "buffer", "is_speech"]
    assert sig.parameters["buffer"].annotation is bytes
    assert not inspect.iscoroutinefunction(bta.BaseTurnAnalyzer.append_audio)


def test_analyze_end_of_turn_is_still_async_and_returns_a_pair() -> None:
    """The adapter adjusts the first element and passes the second through
    untouched -- those metrics describe the inner analyzer's inference, not
    our decision."""
    fn = bta.BaseTurnAnalyzer.analyze_end_of_turn
    assert inspect.iscoroutinefunction(fn)
    assert "tuple" in str(inspect.signature(fn).return_annotation).lower()


@pytest.mark.parametrize(
    "name", ["speech_triggered", "params", "append_audio", "analyze_end_of_turn", "clear"]
)
def test_the_abstract_surface_is_what_we_implement(name: str) -> None:
    """Anything added to this set becomes a method we fail to provide, and
    the analyzer stops being instantiable at all."""
    assert name in bta.BaseTurnAnalyzer.__abstractmethods__


def test_no_abstract_method_has_appeared_that_we_do_not_implement() -> None:
    from baobab_turn.adapters.pipecat import BaobabTurnAnalyzer

    assert not getattr(BaobabTurnAnalyzer, "__abstractmethods__", set())


def test_sample_rate_is_still_set_through_set_sample_rate() -> None:
    """The pitch sensor needs the rate, and this is the only way the host
    ever tells us."""
    assert hasattr(bta.BaseTurnAnalyzer, "set_sample_rate")
    assert "sample_rate" in inspect.signature(bta.BaseTurnAnalyzer.__init__).parameters


def test_the_stop_strategy_still_takes_a_turn_analyzer() -> None:
    """This is the attachment point, and the thing that moved in 1.15.0.
    If it moves again the adapter still works but the documented wiring is
    wrong, which is its own kind of broken."""
    from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
        TurnAnalyzerUserTurnStopStrategy,
    )

    params = inspect.signature(TurnAnalyzerUserTurnStopStrategy.__init__).parameters
    assert "turn_analyzer" in params


def test_the_transcription_frame_still_carries_what_the_observer_reads() -> None:
    """`finalized`, not `is_final`. Reading the wrong attribute would make
    every transcript look interim and silence the ambiguous markers."""
    from pipecat.frames.frames import TranscriptionFrame

    frame = TranscriptionFrame(text="abi", user_id="u", timestamp="t", finalized=True)
    assert frame.text == "abi"
    assert frame.finalized is True
    assert hasattr(frame, "language")
