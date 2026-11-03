"""The Pipecat adapter.

The interesting half is the delay. Every other host this package targets can
only be nudged -- LiveKit takes a float and applies its own threshold, so
`adapters/livekit.py` buys patience by raising `unlikely_threshold` and
hoping. Pipecat asks COMPLETE or INCOMPLETE and believes the answer, so a
requested delay can be waited out to the millisecond. That is what
`test_a_requested_delay_is_waited_out_exactly` pins down, and it is the
reason `GuardResult.delay_override_ms` was ever an intent rather than a
number somebody ignored.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pipecat", reason="needs pipecat-ai")

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    BaseTurnParams,
    EndOfTurnState,
)

from baobab_turn.adapters.pipecat import BaobabTurnAnalyzer
from baobab_turn.core.config import BaobabConfig

COMPLETE = EndOfTurnState.COMPLETE
INCOMPLETE = EndOfTurnState.INCOMPLETE


class Clock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


class FakeAnalyzer(BaseTurnAnalyzer):
    """Stands in for Pipecat's own analyzer."""

    def __init__(self, *, verdict: EndOfTurnState = COMPLETE, metrics=None) -> None:
        super().__init__()
        self.verdict = verdict
        self.metrics = metrics
        self.audio: list[bytes] = []
        self.cleared = 0
        self.vad_secs: list[float] = []
        self.cleaned = False

    @property
    def speech_triggered(self) -> bool:
        return True

    @property
    def params(self) -> BaseTurnParams:
        return BaseTurnParams()

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        self.audio.append(buffer)
        return self.verdict

    async def analyze_end_of_turn(self):
        return self.verdict, self.metrics

    def clear(self) -> None:
        self.cleared += 1

    def update_vad_start_secs(self, vad_start_secs: float) -> None:
        self.vad_secs.append(vad_start_secs)

    async def cleanup(self) -> None:
        self.cleaned = True


@pytest.fixture
def clock() -> Clock:
    return Clock()


def _make(clock: Clock, **kw):
    inner = FakeAnalyzer(verdict=kw.pop("verdict", COMPLETE))
    a = BaobabTurnAnalyzer(inner=inner, language="pcm", now=clock, **kw)
    a.set_sample_rate(16_000)
    return a, inner


SILENCE = b"\x00\x00" * 160


# --- conformance -------------------------------------------------------


def test_it_is_a_usable_turn_analyzer(clock) -> None:
    a, _ = _make(clock)
    assert isinstance(a, BaseTurnAnalyzer)
    assert not getattr(type(a), "__abstractmethods__", set())


def test_the_passthrough_parts_reach_the_inner_analyzer(clock) -> None:
    a, inner = _make(clock)
    assert a.speech_triggered is True
    assert isinstance(a.params, BaseTurnParams)
    assert inner.sample_rate == 16_000
    a.update_vad_start_secs(0.2)
    assert inner.vad_secs == [0.2]


async def test_cleanup_and_clear_reach_the_inner_analyzer(clock) -> None:
    a, inner = _make(clock)
    a.clear()
    assert inner.cleared == 1
    await a.cleanup()
    assert inner.cleaned


# --- the verdict -------------------------------------------------------


def test_incomplete_is_never_second_guessed(clock) -> None:
    """The guard only ever argues a turn is *not* over. Turning an
    INCOMPLETE into a COMPLETE would be inventing an end-of-turn that the
    host's own detector never claimed."""
    a, _ = _make(clock, verdict=INCOMPLETE)
    a.on_transcript("e don finish abi", is_final=True, language="pcm")
    assert a.append_audio(SILENCE, False) is INCOMPLETE


def test_a_complete_with_nothing_to_say_about_it_stands(clock) -> None:
    a, _ = _make(clock)
    a.on_transcript("i dey come", is_final=True, language="pcm")
    assert a.append_audio(SILENCE, False) is COMPLETE


def test_a_continuation_frame_overturns_a_complete(clock) -> None:
    """"the thing be say" announces a clause is still coming. This is the
    whole claim of the package, expressed in Pipecat's own vocabulary."""
    a, _ = _make(clock)
    a.on_transcript("the thing be say", is_final=True, language="pcm")
    assert a.append_audio(SILENCE, False) is INCOMPLETE


def test_a_weak_marker_does_not_overrule_a_confident_detector(clock) -> None:
    """complete_probability is 1.0, so with the default veto strength only a
    marker above 0.83 can flip it. "but" is 0.6 and must not."""
    a, _ = _make(clock)
    a.on_transcript("i go come but", is_final=True, language="pcm")
    assert a.append_audio(SILENCE, False) is COMPLETE


async def test_the_vad_stop_path_is_adjusted_too(clock) -> None:
    """Pipecat reaches COMPLETE two ways. Covering only append_audio would
    leave the guard silently absent on half the turns."""
    a, _ = _make(clock)
    a.on_transcript("the thing be say", is_final=True, language="pcm")
    state, _ = await a.analyze_end_of_turn()
    assert state is INCOMPLETE


async def test_inner_metrics_are_passed_through_untouched(clock) -> None:
    """They describe the inner analyzer's inference, not our decision.
    Rewriting them would make the host's own numbers lie."""
    inner = FakeAnalyzer(verdict=COMPLETE, metrics="inner-metrics")
    a = BaobabTurnAnalyzer(inner=inner, language="pcm", now=clock)
    a.on_transcript("the thing be say", is_final=True, language="pcm")
    state, metrics = await a.analyze_end_of_turn()
    assert state is INCOMPLETE
    assert metrics == "inner-metrics"


# --- the delay, which is why this adapter exists -----------------------


def test_a_requested_delay_is_waited_out_exactly(clock) -> None:
    """LiveKit cannot do this. There, a delay becomes a nudge to a
    threshold; here the turn simply stays open for as long as was asked."""
    a, _ = _make(clock)
    # A code-switch asks for min_delay_ms + codeswitch_hold_ms = 500ms.
    a.on_transcript(
        "i think we should go there but wetin dey happen abeg una sabi",
        is_final=True,
        language="pcm",
    )
    assert a.append_audio(SILENCE, False) is INCOMPLETE

    clock.tick(0.4)
    assert a.append_audio(SILENCE, False) is INCOMPLETE

    clock.tick(0.11)  # now past 500ms
    assert a.append_audio(SILENCE, False) is COMPLETE


def test_a_held_turn_is_not_re_judged_while_it_waits(clock) -> None:
    """Re-running the guard every frame would let one late marker restart
    the clock, and a caller could be held open indefinitely."""
    a, _ = _make(clock)
    a.on_transcript(
        "i think we should go there but wetin dey happen abeg una sabi",
        is_final=True,
        language="pcm",
    )
    assert a.append_audio(SILENCE, False) is INCOMPLETE

    # Something that would otherwise ask for a fresh delay.
    a.on_transcript("the thing be say", is_final=True, language="pcm")
    clock.tick(0.6)
    assert a.append_audio(SILENCE, False) is COMPLETE


def test_clear_cancels_a_pending_hold(clock) -> None:
    a, _ = _make(clock)
    a.on_transcript(
        "i think we should go there but wetin dey happen abeg una sabi",
        is_final=True,
        language="pcm",
    )
    assert a.append_audio(SILENCE, False) is INCOMPLETE
    a.clear()
    assert a._hold_until is None


# --- failing open ------------------------------------------------------


def test_a_broken_guard_returns_the_inner_verdict(clock, monkeypatch) -> None:
    """Fail open. The agent must never break because our layer broke."""
    a, _ = _make(clock)

    def boom(*args, **kwargs):
        raise RuntimeError("guard exploded")

    monkeypatch.setattr("baobab_turn.adapters.pipecat.run_guard", boom)
    a.on_transcript("the thing be say", is_final=True, language="pcm")
    assert a.append_audio(SILENCE, False) is COMPLETE


def test_the_inner_analyzer_is_fed_before_anything_else_can_fail(
    clock, monkeypatch
) -> None:
    """A sensor must never cost the detector its input."""
    a, inner = _make(clock)

    def boom(self, *a_, **kw):
        raise RuntimeError("sensor exploded")

    monkeypatch.setattr(BaobabTurnAnalyzer, "_observe", boom)
    a.append_audio(SILENCE, True)
    assert inner.audio == [SILENCE]


def test_disabled_is_a_pure_passthrough(clock) -> None:
    a, _ = _make(clock, enabled=False)
    a.on_transcript("the thing be say", is_final=True, language="pcm")
    assert a.append_audio(SILENCE, False) is COMPLETE


def test_an_odd_length_buffer_does_not_raise(clock) -> None:
    """int16 means two bytes a sample. A truncated frame is a real thing on
    a lossy transport and must not take the call down."""
    a, _ = _make(clock)
    a.append_audio(b"\x00\x00\x01", True)
    assert a.pitch is not None


# --- the sensors -------------------------------------------------------


def test_pacing_is_measured_from_the_speech_flag(clock) -> None:
    """Pipecat hands us is_speech on every frame, so the caller's rhythm
    needs no extra wiring."""
    a, _ = _make(clock, verdict=INCOMPLETE)
    for _ in range(4):
        a.append_audio(SILENCE, True)
        a.append_audio(SILENCE, False)
        clock.tick(0.4)
        a.append_audio(SILENCE, True)
    assert a.pacing.n >= 3


def test_pacing_survives_a_turn_but_the_tail_does_not(clock) -> None:
    a, _ = _make(clock, verdict=INCOMPLETE)
    for _ in range(4):
        a.append_audio(SILENCE, True)
        a.append_audio(SILENCE, False)
        clock.tick(0.4)
        a.append_audio(SILENCE, True)
    a.on_transcript("the thing be say", is_final=True, language="pcm")
    before = a.pacing.n

    a.clear()
    assert a.pacing.n == before
    assert not a.tail.snapshot().text


def test_no_pitch_sensor_when_the_tonal_rule_is_off(clock) -> None:
    a, _ = _make(clock, config=BaobabConfig(enable_tonal=False))
    assert a.pitch is None
    a.append_audio(SILENCE, True)
    assert a._pitch_contour() is None


# --- the transcript observer -------------------------------------------
#
# A BaseTurnAnalyzer is handed audio and never text, so without this the
# marker layer has nothing to read and three quarters of the package is
# inert on Pipecat.


async def test_the_observer_feeds_the_tail(clock) -> None:
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    from baobab_turn.adapters.pipecat import BaobabTranscriptObserver

    a, _ = _make(clock)
    obs = BaobabTranscriptObserver(a)
    frame = TranscriptionFrame(
        text="the thing be say", user_id="u", timestamp="t", finalized=True
    )
    await obs.process_frame(frame, FrameDirection.DOWNSTREAM)

    snap = a.tail.snapshot()
    assert snap.text == "the thing be say"
    assert snap.is_final is True


async def test_an_interim_transcript_is_never_treated_as_final(clock) -> None:
    """Ambiguous markers only count on a finalised transcript. Mid-utterance
    they mean nothing, and treating one as a hand-off is how a detector cuts
    somebody off."""
    from pipecat.frames.frames import InterimTranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    from baobab_turn.adapters.pipecat import BaobabTranscriptObserver

    a, _ = _make(clock)
    obs = BaobabTranscriptObserver(a)
    await obs.process_frame(
        InterimTranscriptionFrame(text="e don finish sha", user_id="u", timestamp="t"),
        FrameDirection.DOWNSTREAM,
    )
    assert a.tail.snapshot().is_final is False


async def test_the_observer_passes_every_frame_through(clock, monkeypatch) -> None:
    """It observes. A processor that swallowed a frame would break the
    pipeline it was dropped into."""
    from pipecat.frames.frames import Frame, TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    from baobab_turn.adapters.pipecat import BaobabTranscriptObserver

    a, _ = _make(clock)
    obs = BaobabTranscriptObserver(a)
    pushed: list[Frame] = []

    async def capture(self, frame, direction):
        pushed.append(frame)

    monkeypatch.setattr(BaobabTranscriptObserver, "push_frame", capture)

    sent = [
        TranscriptionFrame(text="abi", user_id="u", timestamp="t", finalized=True),
        Frame(),
    ]
    for frame in sent:
        await obs.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert pushed == sent


async def test_a_broken_observer_still_passes_the_frame_on(
    clock, monkeypatch
) -> None:
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    from baobab_turn.adapters.pipecat import BaobabTranscriptObserver

    a, _ = _make(clock)
    obs = BaobabTranscriptObserver(a)

    def boom(*args, **kwargs):
        raise RuntimeError("tail exploded")

    monkeypatch.setattr(a, "on_transcript", boom)
    pushed = []

    async def capture(self, frame, direction):
        pushed.append(frame)

    monkeypatch.setattr(BaobabTranscriptObserver, "push_frame", capture)
    frame = TranscriptionFrame(text="abi", user_id="u", timestamp="t", finalized=True)
    await obs.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert pushed == [frame]
