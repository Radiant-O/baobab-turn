"""The LiveKit adapter: conformance, behaviour, and fail-open.

The fail-open section here is the other half of `tests/test_failopen.py`.
That file proves the guard cannot be made to throw; this one proves that even
if it somehow did, the session would not notice.

Why that matters more here than anywhere else: the SDK wraps the legacy text
detector's call site in try/except but *not* the streaming one. An exception
escaping `predict()` or the future it returns kills the end-of-turn task
before `on_end_of_turn`, so the turn never commits and the call hangs. There
is no safety net below this file.
"""

from __future__ import annotations

import asyncio
import math
import time

import pytest

livekit = pytest.importorskip("livekit.agents", reason="needs livekit-agents")

from livekit import rtc  # noqa: E402
from livekit.agents.voice.turn import (  # noqa: E402
    TurnDetectionEvent,
    _StreamingTurnDetector,
    _StreamingTurnDetectorStream,
)

from baobab_turn.adapters import livekit as adapter  # noqa: E402
from baobab_turn.adapters.livekit import BaobabGuard  # noqa: E402


def _event(p: float = 0.75) -> TurnDetectionEvent:
    return TurnDetectionEvent(
        type="eot_prediction",
        end_of_turn_probability=p,
        last_speaking_time=time.time(),
        detection_delay=0.05,
        inference_duration=0.01,
    )


class FakeStream:
    """Stands in for the SDK's own stream. Configurable failure modes."""

    model = "fake"
    provider = "fake"
    is_fallback = False
    prediction_timeout = 1.0

    def __init__(self, *, p: float = 0.75, raise_sync: bool = False,
                 raise_async: bool = False, never_resolve: bool = False) -> None:
        self.p = p
        self.raise_sync = raise_sync
        self.raise_async = raise_async
        self.never_resolve = never_resolve
        self.audio_frames = 0
        self.flushed: list[str | None] = []
        self.cancelled: list[bool] = []
        self.closed = False
        self.last_event: TurnDetectionEvent | None = None
        #: The future handed out by the most recent predict(), so tests can
        #: resolve or cancel it themselves.
        self.pending: asyncio.Future | None = None

    threshold: float | None = 0.36

    async def unlikely_threshold(self, language=None):
        return self.threshold

    async def backchannel_threshold(self, language=None):
        return None

    async def supports_language(self, language=None):
        return language in (None, "en")

    def predict(self):
        if self.raise_sync:
            raise RuntimeError("inner predict exploded")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending = fut
        if self.never_resolve:
            return fut
        if self.raise_async:
            fut.set_exception(RuntimeError("inner inference failed"))
        else:
            self.last_event = _event(self.p)
            fut.set_result(self.last_event)
        return fut

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        self.cancelled.append(timed_out)

    def flush(self, reason: str | None = None) -> None:
        self.flushed.append(reason)

    def push_audio(self, frame) -> None:
        self.audio_frames += 1

    def end_input(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True


class FakeDetector:
    model = "fake"
    provider = "fake"

    def __init__(self, **kw) -> None:
        self.kw = kw
        self.streams: list[FakeStream] = []

    def stream(self, **_):
        s = FakeStream(**self.kw)
        self.streams.append(s)
        return s


def _make(**kw) -> tuple[BaobabGuard, FakeStream]:
    g = BaobabGuard(inner=FakeDetector(**kw))
    return g, g.turn_detection.stream()


# --- conformance -------------------------------------------------------


def test_satisfies_the_sdk_protocols() -> None:
    """Both protocols are @runtime_checkable and every dispatch site in the
    SDK uses isinstance against them, so this is the real gate."""
    g, s = _make()
    assert isinstance(g.turn_detection, _StreamingTurnDetector)
    assert isinstance(s, _StreamingTurnDetectorStream)


async def test_passes_audio_and_lifecycle_calls_through() -> None:
    g, s = _make()
    s.push_audio(object())
    s.push_audio(object())
    s.end_input()
    await s.aclose()
    inner = g.turn_detection._inner.streams[0]  # type: ignore[attr-defined]
    assert inner.audio_frames == 2
    assert inner.closed


def test_prediction_timeout_is_not_extended() -> None:
    """We could ask the session for more time -- it reads this off us -- but
    that would delay every turn for a path that should never be slow."""
    _, s = _make()
    assert s.prediction_timeout == 1.0


# --- the guard applied -------------------------------------------------


async def test_no_transcript_leaves_the_probability_alone() -> None:
    _, s = _make(p=0.75)
    event = await s.predict()
    assert event.end_of_turn_probability == 0.75


async def test_continuation_marker_lowers_the_probability() -> None:
    g, s = _make(p=0.75)
    g.tail.update("i wan tell you say the thing be say", language="pcm")
    event = await s.predict()
    assert event.end_of_turn_probability < 0.75


async def test_yield_marker_raises_the_probability() -> None:
    g, s = _make(p=0.75)
    g.tail.update("e don finish abi", is_final=True, language="pcm")
    event = await s.predict()
    assert event.end_of_turn_probability > 0.75


async def test_adjust_mutates_the_inner_event_in_place() -> None:
    """Returning the same object keeps every other field without copying, and
    is safe because the inner detector allocates a fresh event per
    prediction -- so the session's identity-based hook dedupe still sees a
    distinct object each time."""
    g, s = _make(p=0.75)
    g.tail.update("abi", is_final=True, language="pcm")
    event = await s.predict()
    inner = g.turn_detection._inner.streams[0]  # type: ignore[attr-defined]
    assert event is inner.last_event
    assert event.detection_delay == 0.05
    assert event.inference_duration == 0.01


async def test_disabled_guard_is_a_true_passthrough() -> None:
    """The kill switch for A/B: nothing we do may change the outcome."""
    g = BaobabGuard(inner=FakeDetector(p=0.75), enabled=False)
    s = g.turn_detection.stream()
    g.tail.update("the thing be say", language="pcm")
    event = await s.predict()
    assert event.end_of_turn_probability == 0.75


async def test_on_decision_hook_receives_the_reasons() -> None:
    seen: list[tuple] = []
    g = BaobabGuard(inner=FakeDetector(p=0.75),
                    on_decision=lambda gi, r: seen.append((gi, r)))
    s = g.turn_detection.stream()
    g.tail.update("e don finish abi", is_final=True, language="pcm")
    await s.predict()
    assert len(seen) == 1
    assert "yield_boost:abi" in seen[0][1].reasons


# --- language queries --------------------------------------------------


async def test_pack_threshold_wins_over_the_inner_detector() -> None:
    """The only graded lever LiveKit offers. A pack wanting more patience for
    a tonal language expresses it here, not through a millisecond delay."""
    g, s = _make()
    g._indexes.clear()
    from baobab_turn.core.markers import MarkerIndex
    from baobab_turn.core.types import LanguagePack

    g._indexes["yo"] = MarkerIndex(
        LanguagePack(code="yo", name="Yoruba", unlikely_threshold=0.2)
    )
    assert await s.unlikely_threshold("yo") == 0.2


async def test_inner_threshold_used_when_the_pack_has_none() -> None:
    _, s = _make()
    assert await s.unlikely_threshold("pcm") == 0.36


async def test_supports_language_never_narrows() -> None:
    """The inner detector only claims 'en' here. We must add our languages
    without ever removing one of its own."""
    _, s = _make()
    assert await s.supports_language("en") is True      # inner's
    assert await s.supports_language("pcm") is True     # ours
    assert await s.supports_language("xx-YY") is False  # neither


# --- turn boundaries ---------------------------------------------------


async def test_flush_clears_the_tail() -> None:
    """flush() is the turn boundary. Without this the next turn could match
    a marker left over from the previous one."""
    g, s = _make()
    g.tail.update("e don finish abi", is_final=True, language="pcm")
    s.flush("turn_end")
    assert g.tail.snapshot().text == ""
    inner = g.turn_detection._inner.streams[0]  # type: ignore[attr-defined]
    assert inner.flushed == ["turn_end"]


async def test_cancel_inference_does_not_clear_the_tail() -> None:
    """cancel_inference means 'that pause was not the end after all', not
    'the utterance finished'. Clearing here would wipe the tail every time a
    caller paused mid-sentence and carried on."""
    g, s = _make()
    g.tail.update("the thing be say", language="pcm")
    s.cancel_inference(timed_out=False)
    assert g.tail.snapshot().text == "the thing be say"


async def test_agent_speaking_makes_the_guard_abstain() -> None:
    """Transcript events are suppressed while the agent talks, so the tail is
    stale by construction -- the barge-in case."""
    g, s = _make(p=0.75)
    g.tail.update("the thing be say", language="pcm")
    s.set_agent_speaking(True)
    event = await s.predict()
    assert event.end_of_turn_probability == 0.75


# --- attach ------------------------------------------------------------


def test_attach_subscribes_a_sync_handler() -> None:
    """The SDK's emitter calls handlers inline and never wraps them in a
    task, so an async handler would return an un-awaited coroutine and
    silently do nothing."""
    import inspect

    captured: dict[str, object] = {}

    class FakeSession:
        def on(self, name, handler):
            captured[name] = handler

    g = BaobabGuard(inner=FakeDetector())
    g.attach(FakeSession())

    handler = captured["user_input_transcribed"]
    assert not inspect.iscoroutinefunction(handler)

    class Ev:
        transcript = "e don finish abi"
        is_final = True
        language = "pcm"

    handler(Ev())  # type: ignore[operator]
    snap = g.tail.snapshot()
    assert snap.text == "e don finish abi"
    assert snap.is_final and snap.language == "pcm"


def test_a_broken_transcript_event_does_not_reach_the_emitter() -> None:
    """A transcript we fail to record is a missing opinion, not a broken
    call."""
    captured: dict[str, object] = {}

    class FakeSession:
        def on(self, name, handler):
            captured[name] = handler

    g = BaobabGuard(inner=FakeDetector())
    g.attach(FakeSession())
    handler = captured["user_input_transcribed"]

    class Exploding:
        @property
        def transcript(self):
            raise RuntimeError("bad event")

    handler(Exploding())  # type: ignore[operator]  # must not raise
    assert g.tail.snapshot().text == ""


# --- fail open ---------------------------------------------------------


async def test_inner_predict_raising_resolves_with_a_safe_default() -> None:
    _, s = _make(raise_sync=True)
    fut = s.predict()
    event = await fut
    assert fut.exception() is None
    assert event.end_of_turn_probability == 1.0


async def test_inner_inference_failing_resolves_with_a_safe_default() -> None:
    _, s = _make(raise_async=True)
    fut = s.predict()
    event = await fut
    assert fut.exception() is None
    assert event.end_of_turn_probability == 1.0


async def test_we_never_set_exception_on_the_future() -> None:
    """The session's await on this future is not wrapped in try/except. An
    exception here kills the end-of-turn task and the turn never commits."""
    for kw in ({"raise_sync": True}, {"raise_async": True}):
        _, s = _make(**kw)
        fut = s.predict()
        await fut
        assert fut.exception() is None


async def test_a_broken_guard_passes_the_inner_result_through(monkeypatch) -> None:
    """If our own rules break, the inner detector's opinion is still good."""
    def explode(*_a, **_k):
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(adapter, "run_guard", explode)
    g, s = _make(p=0.42)
    g.tail.update("e don finish abi", is_final=True, language="pcm")
    fut = s.predict()
    event = await fut
    assert fut.exception() is None
    assert event.end_of_turn_probability == 0.42


async def test_a_broken_on_decision_hook_does_not_break_the_turn() -> None:
    def explode(gi, r):
        raise RuntimeError("logging exploded")

    g = BaobabGuard(inner=FakeDetector(p=0.75), on_decision=explode)
    s = g.turn_detection.stream()
    g.tail.update("e don finish abi", is_final=True, language="pcm")
    fut = s.predict()
    event = await fut
    assert fut.exception() is None
    assert event.end_of_turn_probability > 0.75


async def test_each_failopen_event_is_a_distinct_object() -> None:
    """The session dedupes its decision hook with an identity check
    (`prediction_event is not self._last_emitted_prediction`). A shared
    module-level default instance would silently suppress the second hook of
    a turn, so the default must be constructed fresh every time."""
    _, s = _make(raise_sync=True)
    first = await s.predict()
    second = await s.predict()
    assert first is not second


async def test_inner_cancellation_cancels_ours_without_raising() -> None:
    """`cancel_inference` cancels the inner prediction. That must surface as
    a cancelled future, not as an exception the session has to catch."""
    _, s = _make(never_resolve=True)
    fut = s.predict()
    inner_future = s._inner.pending  # type: ignore[attr-defined]

    inner_future.cancel()
    await asyncio.sleep(0)

    assert fut.cancelled()


async def test_a_late_inner_result_after_we_are_cancelled_is_dropped() -> None:
    """The session cancels us on timeout, then the inner detector answers
    anyway. Setting a result on an already-cancelled future raises
    InvalidStateError, and that exception would surface inside the callback
    with nothing to catch it."""
    _, s = _make(never_resolve=True)
    fut = s.predict()
    inner_future = s._inner.pending  # type: ignore[attr-defined]

    fut.cancel()
    await asyncio.sleep(0)
    assert fut.cancelled()

    # The inner detector comes back late. This must be a no-op.
    inner_future.set_result(_event(0.9))
    await asyncio.sleep(0)


# --- the sensors, wired --------------------------------------------------
#
# Pacing, code-switch and the agent floor were all built, tested and then fed
# by nothing. These cover the wiring: the session events that drive them, and
# the one mechanism that makes a delay request mean anything on LiveKit.


class Event(dict):
    """A stand-in for the SDK's pydantic events -- attribute access, and
    missing fields read as None rather than raising."""

    def __getattr__(self, name):
        return self.get(name)


class Session:
    def __init__(self) -> None:
        self.handlers: dict = {}

    def on(self, name, handler):
        self.handlers[name] = handler

    def fire(self, name, **fields):
        self.handlers[name](Event(fields))


@pytest.fixture
def attached():
    g = BaobabGuard(inner=FakeDetector())
    s = Session()
    g.attach(s)
    return g, s, g.turn_detection.stream()


def test_attach_subscribes_to_every_sensor_event(attached) -> None:
    _, session, _ = attached
    assert set(session.handlers) == {
        "user_input_transcribed",
        "user_state_changed",
        "agent_state_changed",
    }


def test_user_state_changes_feed_the_pacing_tracker(attached) -> None:
    """`user_state_changed` is the SDK's own view of when speech starts and
    stops, so it beats re-deriving pauses from transcript arrival times that
    lag by a network round trip."""
    guard, session, _ = attached
    for _ in range(3):
        session.fire("user_state_changed", old_state="speaking", new_state="listening")
        time.sleep(0.05)
        session.fire("user_state_changed", old_state="listening", new_state="speaking")
    assert guard.pacing.n == 3
    mean, _ = guard.pacing.as_guard_inputs()
    assert mean is not None and mean > 0


def test_a_pause_needs_a_stop_before_it_counts(attached) -> None:
    """Speech starting without a recorded stop is not a measurable gap."""
    guard, session, _ = attached
    session.fire("user_state_changed", old_state="listening", new_state="speaking")
    assert guard.pacing.n == 0


def test_agent_state_tracks_the_floor(attached) -> None:
    guard, session, _ = attached
    assert not guard.agent_speaking
    session.fire("agent_state_changed", old_state="listening", new_state="speaking")
    assert guard.agent_speaking
    session.fire("agent_state_changed", old_state="speaking", new_state="listening")
    assert not guard.agent_speaking


async def test_the_agent_floor_makes_the_guard_abstain(attached) -> None:
    guard, session, stream = attached
    session.fire("user_input_transcribed", transcript="the thing be say",
                 is_final=False, language="pcm")
    session.fire("agent_state_changed", old_state="listening", new_state="speaking")
    await stream.predict()
    # The markers must not have fired: the tail is stale by construction
    # while the agent holds the floor.
    assert guard.agent_speaking


def test_transcripts_drive_code_switch_detection(attached) -> None:
    guard, session, _ = attached
    session.fire(
        "user_input_transcribed",
        transcript="i think we should go there but wetin dey happen abeg una sabi",
        is_final=False,
        language="pcm",
    )
    assert guard.code_switch


def test_sustained_one_language_is_not_a_switch(attached) -> None:
    guard, session, _ = attached
    session.fire("user_input_transcribed",
                 transcript="i think that we should probably go there tomorrow",
                 is_final=False, language="en-NG")
    assert not guard.code_switch


def test_flush_clears_the_switch_but_keeps_the_rhythm(attached) -> None:
    """A caller's rhythm belongs to the speaker, not to the utterance that
    just ended -- resetting it every turn would mean never accumulating
    enough observations to be ready."""
    guard, session, stream = attached
    session.fire("user_input_transcribed",
                 transcript="i think we should go there but wetin dey happen abeg una sabi",
                 is_final=False, language="pcm")
    for _ in range(3):
        session.fire("user_state_changed", old_state="speaking", new_state="listening")
        time.sleep(0.05)
        session.fire("user_state_changed", old_state="listening", new_state="speaking")
    assert guard.code_switch and guard.pacing.n == 3

    stream.flush("turn_end")
    assert not guard.code_switch
    assert guard.pacing.n == 3


@pytest.mark.parametrize("name,fields", [
    ("user_state_changed", {}),
    ("agent_state_changed", {}),
    ("user_input_transcribed", {}),
])
def test_a_malformed_event_never_reaches_the_emitter(attached, name, fields) -> None:
    """These handlers run inside the SDK's event loop. A sensor that fails is
    a missing opinion, not a broken call."""
    _, session, _ = attached
    session.fire(name, **fields)


# --- delay intents become real through the threshold ---------------------


async def test_no_patience_requested_leaves_the_threshold_alone(attached) -> None:
    _, _, stream = attached
    await stream.predict()
    assert await stream.unlikely_threshold("pcm") == pytest.approx(0.36)


async def test_a_delay_request_raises_the_threshold(attached) -> None:
    """The point of the mechanism. LiveKit never consumes a millisecond
    delay, so patience has to be spent by raising the bar to commit --
    otherwise every delay the guard computes is silently discarded."""
    _, session, stream = attached
    session.fire("user_input_transcribed",
                 transcript="i think we should go there but wetin dey happen abeg una sabi",
                 is_final=False, language="pcm")
    await stream.predict()
    assert stream._pending_delay_ms is not None
    raised = await stream.unlikely_threshold("pcm")
    assert raised > 0.36


async def test_the_bump_is_capped(attached) -> None:
    """Uncapped, accumulated patience could make committing effectively
    impossible. A confident turn must always get through."""
    guard, _, stream = attached
    base = 0.36
    stream._pending_delay_ms = 10 * guard.config.max_delay_ms
    raised = await stream.unlikely_threshold("pcm")
    ceiling = base + (1.0 - base) * guard.config.max_threshold_bump
    assert raised <= ceiling + 1e-9


async def test_a_delay_at_the_floor_changes_nothing(attached) -> None:
    guard, _, stream = attached
    stream._pending_delay_ms = guard.config.min_delay_ms
    assert await stream.unlikely_threshold("pcm") == pytest.approx(0.36)


async def test_no_base_threshold_means_no_opinion(attached) -> None:
    """Inventing a threshold where the inner detector has none would change
    behaviour for every turn, not just the ones wanting patience."""
    _, _, stream = attached
    stream._inner.threshold = None
    stream._pending_delay_ms = 2_000
    assert await stream.unlikely_threshold("pcm") is None


# --- pitch -------------------------------------------------------------
#
# The tonal rule was the one sensor-less claim left in the system: packs
# said `tonal: true` and the guard waited longer for everyone speaking that
# language, whatever their pitch was doing. These cover the wiring that
# lets it react to one actual fall instead.

np = pytest.importorskip("numpy", reason="pitch needs baobab-turn[prosody]")


def _glide(
    hz_start: float,
    hz_end: float,
    *,
    rate: int = 16_000,
    ms_total: int = 600,
    ms_frame: int = 20,
    channels: int = 1,
) -> list[rtc.AudioFrame]:
    """Synthesise a voiced glide as a list of real `rtc.AudioFrame`s.

    A pure tone rather than recorded speech: the sensor only ever answers
    rising/falling/level, and a synthetic glide pins that down without
    committing a wav file to the repo.
    """
    per = rate * ms_frame // 1000
    total = rate * ms_total // 1000
    frames: list[rtc.AudioFrame] = []
    phase = 0.0
    for start in range(0, total, per):
        buf = bytearray()
        for i in range(per):
            hz = hz_start + (hz_end - hz_start) * ((start + i) / total)
            phase += 2.0 * math.pi * hz / rate
            sample = int(20_000 * math.sin(phase))
            for _ in range(channels):
                buf += sample.to_bytes(2, "little", signed=True)
        frames.append(rtc.AudioFrame(bytes(buf), rate, channels, per))
    return frames


def _tonal(g: BaobabGuard, code: str = "yo", **kw) -> None:
    from baobab_turn.core.markers import MarkerIndex
    from baobab_turn.core.types import LanguagePack

    g._indexes[code] = MarkerIndex(
        LanguagePack(code=code, name="Test", tonal=True, **kw)
    )


def _feed(stream, frames) -> None:
    for frame in frames:
        stream.push_audio(frame)


async def test_a_measured_fall_damps_a_tonal_language() -> None:
    """The whole point of the sensor. A falling contour in a tonal language
    may be a low tone mid-word, so a confident inner probability on a fall
    is exactly the one to discount."""
    g, s = _make(p=0.8)
    _tonal(g)
    _feed(s, _glide(220.0, 120.0))
    g.tail.update("mo ti sọ", is_final=True, language="yo")

    assert g.pitch_contour == "falling"
    event = await s.predict()
    assert event.end_of_turn_probability == pytest.approx(0.8 * 0.7)


async def test_a_rising_contour_is_not_damped() -> None:
    g, s = _make(p=0.8)
    _tonal(g)
    _feed(s, _glide(120.0, 220.0))
    g.tail.update("mo ti sọ", is_final=True, language="yo")

    assert g.pitch_contour == "rising"
    event = await s.predict()
    assert event.end_of_turn_probability == pytest.approx(0.8)


async def test_a_level_contour_is_not_damped() -> None:
    g, s = _make(p=0.8)
    _tonal(g)
    _feed(s, _glide(180.0, 180.0))
    g.tail.update("mo ti sọ", is_final=True, language="yo")

    assert g.pitch_contour == "level"
    event = await s.predict()
    assert event.end_of_turn_probability == pytest.approx(0.8)


async def test_a_fall_in_a_non_tonal_language_is_left_alone() -> None:
    """English speakers fall at the end of sentences and mean it. The rule
    must not fire just because a sensor is attached."""
    g, s = _make(p=0.8)
    _feed(s, _glide(220.0, 120.0))
    g.tail.update("that is all", is_final=True, language="en")

    assert g.pitch_contour == "falling"
    event = await s.predict()
    assert event.end_of_turn_probability == pytest.approx(0.8)


async def test_flush_clears_the_contour() -> None:
    """A fall belongs to the utterance that just ended. Carrying it into the
    next turn would damp a probability on evidence about different audio."""
    g, s = _make()
    _tonal(g)
    _feed(s, _glide(220.0, 120.0))
    assert g.pitch_contour == "falling"

    s.flush("eou")
    assert g.pitch_contour is None


async def test_audio_reaches_the_inner_detector_even_when_the_sensor_throws(
    monkeypatch,
) -> None:
    """push_audio is on the hot path and the SDK does not wrap it. A broken
    sensor must cost pitch, never the call."""
    g, s = _make()

    def boom(self, *a, **kw):
        raise RuntimeError("sensor exploded")

    monkeypatch.setattr(adapter.PitchTracker, "push_samples", boom)
    _feed(s, _glide(220.0, 120.0))

    assert g.turn_detection._inner.streams[0].audio_frames == 30
    assert g.pitch is None  # disabled, not raising once per frame
    assert g.pitch_contour is None


async def test_a_disabled_sensor_stays_disabled() -> None:
    """Re-arming it per frame would mean paying for the failed import on
    every frame of a long call."""
    g, s = _make()
    g.pitch = None
    _feed(s, _glide(220.0, 120.0))
    assert g.pitch is None


async def test_no_sensor_when_the_tonal_rule_is_off() -> None:
    from baobab_turn.core.config import BaobabConfig

    g = BaobabGuard(inner=FakeDetector(), config=BaobabConfig(enable_tonal=False))
    assert g.pitch is None
    assert g.pitch_contour is None


async def test_stereo_frames_take_one_channel_rather_than_every_sample() -> None:
    """Reading an interleaved stereo frame as a flat mono buffer halves the
    apparent frequency, which would turn every stereo call's pitch into
    nonsense without ever raising."""
    g, s = _make()
    _feed(s, _glide(200.0, 200.0, channels=2))

    assert g.pitch is not None
    measured = sorted(g.pitch._f0)[len(g.pitch._f0) // 2]
    assert measured == pytest.approx(200.0, rel=0.1)


async def test_objects_that_are_not_audio_frames_are_ignored() -> None:
    """Callers drive this stream directly in tests and in non-LiveKit
    harnesses. A sensor is not a reason to reject input the inner detector
    accepts."""
    g, s = _make()

    class Weird:
        data = "not a buffer"
        sample_rate = 16_000
        num_channels = 1

    s.push_audio(object())
    s.push_audio(Weird())
    assert g.pitch is not None
    assert g.pitch_contour is None
    assert g.turn_detection._inner.streams[0].audio_frames == 2
