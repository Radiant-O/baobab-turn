"""LiveKit Agents adapter. Framework glue only -- no rules live here.

Usage is two lines in an existing agent:

    from livekit.agents import AgentSession, TurnHandlingOptions, inference
    from baobab_turn.adapters.livekit import BaobabGuard

    guard = BaobabGuard()
    session = AgentSession(
        turn_handling=TurnHandlingOptions(turn_detection=guard.turn_detection),
        stt=..., llm=..., tts=...,
    )
    guard.attach(session)      # subscribes to transcripts

Three things about the LiveKit integration are non-obvious enough to be worth
stating here, because getting any of them wrong is a broken call rather than
a failing test.

**The detector never sees text.** `_StreamingTurnDetectorStream` receives
audio frames via `push_audio` and returns a probability; the protocol that
takes a `ChatContext` is the deprecated text one. So `attach()` subscribes to
the public `user_input_transcribed` event and feeds a `TranscriptTail`, which
`predict()` then reads. That event carries `is_final` and `language` and
fires while the user is still speaking, so it works with any STT.

**An exception here does not fail safe, it hangs the call.** The SDK wraps
the legacy text detector's call site in try/except but *not* the streaming
one. An exception escaping `predict()` or the future it returns kills the
end-of-turn task before `on_end_of_turn`, so the turn never commits. Nothing
in this file may raise, and we never call `set_exception` on the future the
session awaits -- we resolve it with a safe default instead.

**The default event must be freshly built every time.** The session dedupes
its decision hook with `prediction_event is not self._last_emitted_prediction`
-- an identity check. A shared module-level fallback instance would silently
suppress the second hook of a turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from livekit import rtc
from livekit.agents import APIConnectOptions, inference
from livekit.agents.voice.turn import TurnDetectionEvent

from ..core.codeswitch import CodeSwitchDetector, LanguageProfile
from ..core.config import BaobabConfig
from ..core.guard import guard as run_guard
from ..core.markers import MarkerIndex
from ..core.pacing import PacingTracker
from ..core.tail import TranscriptTail
from ..core.types import GuardInput, GuardResult
from ..packs.registry import available_packs, resolve
from ..prosody import PitchTracker

__all__ = ["BaobabGuard", "BaobabTurnDetector", "BaobabTurnDetectorStream"]

log = logging.getLogger("baobab_turn.livekit")

#: Matches the SDK's own fallback: a positive default so that anything
#: waiting on the prediction commits rather than stalling.
_FAILOPEN_PROBABILITY = 1.0


def _default_event() -> TurnDetectionEvent:
    """A fresh safe-default event. Never cache this -- see the module doc."""
    return TurnDetectionEvent(
        type="eot_prediction",
        end_of_turn_probability=_FAILOPEN_PROBABILITY,
        last_speaking_time=time.time(),
    )


class BaobabGuard:
    """Owns the transcript buffer and the detector, and wires them together.

    One object rather than two because the two halves have independent
    lifetimes inside the SDK: an agent handoff can keep the detector stream
    alive while replacing the STT pipeline. Holding the buffer here means its
    lifetime is ours and survives that.
    """

    def __init__(
        self,
        *,
        inner: Any | None = None,
        config: BaobabConfig | None = None,
        enabled: bool = True,
        on_decision: Callable[[GuardInput, GuardResult], None] | None = None,
    ) -> None:
        self.config = config or BaobabConfig()
        self.tail = TranscriptTail()
        #: Learns this caller's pause rhythm, so a deliberate speaker gets
        #: patience a detector tuned to an average speaker would not give.
        self.pacing = PacingTracker()
        self.turn_detection = BaobabTurnDetector(
            inner=inner if inner is not None else inference.TurnDetector(),
            guard=self,
            enabled=enabled,
            on_decision=on_decision,
        )
        self._indexes: dict[str, MarkerIndex | None] = {}
        self._codeswitch: CodeSwitchDetector | None = None
        self._code_switch = False
        #: Measures the shape of the caller's pitch, so the tonal rule can
        #: react to *this* fall rather than to the language in general.
        #: None when the rule is switched off, and set to None permanently
        #: if the optional numpy extra turns out to be missing -- the rule
        #: then stays silent, which is exactly the behaviour from before the
        #: sensor existed.
        self.pitch: PitchTracker | None = (
            PitchTracker() if self.config.enable_tonal else None
        )
        #: Samples awaiting a full analysis window, as plain ints.
        self._pcm: list[int] = []
        self._agent_speaking = False
        #: When the caller last stopped speaking, for measuring the gap to
        #: when they start again.
        self._user_stopped_at: float | None = None

    @property
    def agent_speaking(self) -> bool:
        """Whether the agent currently holds the floor.

        Lives on the guard rather than the stream because an agent handoff
        can replace the stream while the session carries on, and the floor
        belongs to the session.
        """
        return self._agent_speaking

    @property
    def code_switch(self) -> bool:
        """Whether the last transcript update crossed a language boundary."""
        return self._code_switch

    @property
    def pitch_contour(self) -> str | None:
        """The shape of the caller's recent pitch, or None if not yet known.

        "unknown" is flattened to None so the guard sees the same absence it
        sees when the sensor is not installed at all. Two spellings of "no
        opinion" would be two code paths to keep in step.
        """
        tracker = self.pitch
        if tracker is None:
            return None
        contour = tracker.contour()
        return None if contour == "unknown" else contour

    def observe_audio(self, frame: Any) -> None:
        """Feed one audio frame to the pitch tracker.

        Buffered, because f0 cannot be estimated from a single frame: at
        10-20ms a LiveKit frame is shorter than one period of a 70Hz voice,
        so autocorrelation has nothing to lock onto. Samples accumulate
        until there is a window worth analysing.

        Made mono by taking the first channel rather than averaging. On one
        speaker averaging is harmless, but on two it invents a pitch neither
        of them produced -- and the frame layout does not tell us which case
        this is.

        Tolerates objects that are not audio frames at all. Callers drive
        this stream directly in tests and in non-LiveKit harnesses, and a
        sensor is never a reason to reject input the inner detector accepts.
        """
        tracker = self.pitch
        if tracker is None:
            return

        data = getattr(frame, "data", None)
        rate = int(getattr(frame, "sample_rate", 0) or 0)
        if data is None or rate <= 0:
            return

        channels = int(getattr(frame, "num_channels", 1) or 1)
        try:
            samples = data[::channels].tolist() if channels > 1 else data.tolist()
        except (AttributeError, TypeError, ValueError):
            return

        need = max(2, rate * self.config.pitch_window_ms // 1000)
        self._pcm.extend(samples)
        while len(self._pcm) >= need:
            window = self._pcm[:need]
            del self._pcm[:need]
            try:
                tracker.push_samples(window, rate)
            except Exception:
                # numpy is an optional extra and this runs on the audio hot
                # path, which the SDK does not wrap. Disable the sensor for
                # good rather than raise once per frame for the rest of the
                # call.
                self.pitch = None
                self._pcm.clear()
                log.warning(
                    "baobab: pitch tracking disabled; "
                    "install baobab-turn[prosody] for the tonal rule",
                    exc_info=True,
                )
                return

    def _detector(self) -> CodeSwitchDetector:
        """Built lazily from every pack we are allowed to use.

        Lazily because building profiles reads every pack, and a session that
        never sees a transcript should not pay for that.
        """
        if self._codeswitch is None:
            packs = [
                pack
                for pack in available_packs().values()
                if self.config.allow_community_packs or pack.status != "community"
            ]
            self._codeswitch = CodeSwitchDetector(
                [LanguageProfile.from_pack(pack) for pack in packs]
            )
        return self._codeswitch

    def reset_turn(self) -> None:
        """Clear state that belongs to one turn.

        Pacing deliberately survives: a caller's rhythm is a property of the
        speaker, not of the utterance that just ended, and resetting it every
        turn would mean never accumulating enough observations to be ready.

        The pitch contour does not survive. It describes the shape of one
        utterance, and carrying the last turn's fall into this one would
        damp a probability on evidence that is no longer about the audio
        being judged.
        """
        self._code_switch = False
        if self.pitch is not None:
            self.pitch.reset()
        self._pcm.clear()

    def index_for(self, language: str | None) -> MarkerIndex | None:
        """Cached pack lookup. None means no pack, i.e. stock behaviour."""
        key = language or ""
        if key not in self._indexes:
            pack = resolve(language, allow_community=self.config.allow_community_packs)
            self._indexes[key] = MarkerIndex(pack) if pack is not None else None
        return self._indexes[key]

    def attach(self, session: Any) -> None:
        """Subscribe to the session events that feed the guard.

        Without this the guard runs on the inner probability alone: no
        transcript, so no markers, and none of the other sensors. The agent
        still works, it just behaves exactly like stock -- which is a
        confusing way to discover the call is missing.

        Every handler must be a plain `def`. The SDK's emitter calls handlers
        inline and never wraps them in a task, so an `async def` here would
        return an un-awaited coroutine and silently do nothing.

        Every handler also swallows its own exceptions. These run inside the
        SDK's event loop, and a sensor that fails is a missing opinion, not a
        broken call.
        """

        def _on_transcript(event: Any) -> None:
            try:
                text = getattr(event, "transcript", "") or ""
                self.tail.update(
                    text,
                    is_final=bool(getattr(event, "is_final", False)),
                    language=getattr(event, "language", None),
                )
                # Re-read from the tail rather than using `text`: the tail is
                # bounded and normalised, and the detector should see exactly
                # what the marker layer will see.
                snapshot = self.tail.snapshot()
                if snapshot.text:
                    self._code_switch = self._detector().detect(snapshot.text).switched
            except Exception:
                log.exception("baobab: failed to record transcript")

        def _on_user_state(event: Any) -> None:
            """Measure the gap between the caller's utterances.

            `user_state_changed` is the cleanest pause signal available: the
            transitions are the SDK's own view of when speech starts and
            stops, so we are not re-deriving it from transcript arrival times
            that lag by a network round trip.
            """
            try:
                old = getattr(event, "old_state", None)
                new = getattr(event, "new_state", None)
                now = time.monotonic()
                if old == "speaking" and new != "speaking":
                    self._user_stopped_at = now
                elif new == "speaking" and self._user_stopped_at is not None:
                    self.pacing.observe((now - self._user_stopped_at) * 1000.0)
                    self._user_stopped_at = None
            except Exception:
                log.exception("baobab: failed to record a pause")

        def _on_agent_state(event: Any) -> None:
            try:
                self._agent_speaking = getattr(event, "new_state", None) == "speaking"
            except Exception:
                log.exception("baobab: failed to track agent state")

        session.on("user_input_transcribed", _on_transcript)
        session.on("user_state_changed", _on_user_state)
        session.on("agent_state_changed", _on_agent_state)


class BaobabTurnDetector:
    """Implements LiveKit's `_StreamingTurnDetector` by wrapping another."""

    def __init__(
        self,
        *,
        inner: Any,
        guard: BaobabGuard,
        enabled: bool = True,
        on_decision: Callable[[GuardInput, GuardResult], None] | None = None,
    ) -> None:
        self._inner = inner
        self._guard = guard
        self._enabled = enabled
        self._on_decision = on_decision

    @property
    def model(self) -> str:
        return f"baobab+{getattr(self._inner, 'model', 'unknown')}"

    @property
    def provider(self) -> str:
        return f"baobab+{getattr(self._inner, 'provider', 'unknown')}"

    def stream(self, *, conn_options: APIConnectOptions | None = None) -> Any:
        kwargs = {} if conn_options is None else {"conn_options": conn_options}
        return BaobabTurnDetectorStream(
            inner=self._inner.stream(**kwargs),
            guard=self._guard,
            enabled=self._enabled,
            on_decision=self._on_decision,
        )


class BaobabTurnDetectorStream:
    """The per-session stream. Everything here forwards or fails open."""

    def __init__(
        self,
        *,
        inner: Any,
        guard: BaobabGuard,
        enabled: bool = True,
        on_decision: Callable[[GuardInput, GuardResult], None] | None = None,
    ) -> None:
        self._inner = inner
        self._guard = guard
        self._enabled = enabled
        self._on_decision = on_decision
        self._agent_speaking = False
        #: Delay the guard asked for on the most recent prediction. LiveKit
        #: queries `unlikely_threshold` immediately after each prediction
        #: resolves, so this is read while it is still current.
        self._pending_delay_ms: int | None = None

    # --- passthrough properties ---------------------------------------

    @property
    def model(self) -> str:
        return f"baobab+{getattr(self._inner, 'model', 'unknown')}"

    @property
    def provider(self) -> str:
        return f"baobab+{getattr(self._inner, 'provider', 'unknown')}"

    @property
    def is_fallback(self) -> bool:
        return bool(getattr(self._inner, "is_fallback", False))

    @property
    def prediction_timeout(self) -> float:
        # Deliberately not extended. The session reads this off us and would
        # honour a larger value, but that delays every turn to accommodate a
        # path that should never be slow -- the guard's budget is 15ms
        # against the SDK's 1.0s.
        return float(getattr(self._inner, "prediction_timeout", 1.0))

    # --- language queries ---------------------------------------------

    async def unlikely_threshold(self, language: Any = None) -> float | None:
        """A pack's threshold wins, otherwise the inner detector's -- then
        raised by however much patience the last prediction asked for.

        This is where `delay_override_ms` becomes real on LiveKit. The
        session never consumes a millisecond delay from a detector; it flips
        between `min_delay` and `max_delay` depending on whether the
        probability fell below this threshold. So the way to buy patience is
        to raise the bar: a borderline turn then waits, while a confident one
        still commits promptly.

        Without this, every delay the guard computes -- code-switch holds,
        tonal patience, the caller's own rhythm -- would be calculated and
        silently discarded on this adapter.
        """
        base: float | None = None
        try:
            index = self._guard.index_for(_as_code(language))
            if index is not None and index.pack.unlikely_threshold is not None:
                base = index.pack.unlikely_threshold
        except Exception:
            log.exception("baobab: unlikely_threshold failed, deferring to inner")

        if base is None:
            base = await self._inner.unlikely_threshold(language)
        if base is None:
            # The inner detector has no opinion either. Inventing one would
            # change behaviour for everybody, not just when we want patience.
            return None

        try:
            return self._with_patience(base)
        except Exception:
            log.exception("baobab: threshold adjustment failed, using base")
            return base

    def _with_patience(self, base: float) -> float:
        """Raise `base` in proportion to the delay the guard asked for.

        Scaled against the configured delay range, so asking for the maximum
        patience applies the maximum bump and asking for the minimum changes
        nothing. Capped by `max_threshold_bump` so a confident turn always
        commits -- an uncapped version could accumulate enough patience to
        make committing effectively impossible.
        """
        config = self._guard.config
        delay = self._pending_delay_ms
        if delay is None or delay <= config.min_delay_ms:
            return base

        span = config.max_delay_ms - config.min_delay_ms
        if span <= 0:
            return base
        fraction = min(1.0, (delay - config.min_delay_ms) / span)
        bump = (1.0 - base) * fraction * config.max_threshold_bump
        return min(1.0, base + bump)

    async def backchannel_threshold(self, language: Any = None) -> float | None:
        return await self._inner.backchannel_threshold(language)

    async def supports_language(self, language: Any = None) -> bool:
        """Never narrow what the inner detector claims to handle."""
        try:
            if self._guard.index_for(_as_code(language)) is not None:
                return True
        except Exception:
            log.exception("baobab: supports_language failed, deferring to inner")
        return await self._inner.supports_language(language)

    # --- audio and lifecycle ------------------------------------------

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        # Inner first, always. This call site is not wrapped by the SDK, so
        # the detector must have its audio before our sensor gets a chance
        # to misbehave.
        self._inner.push_audio(frame)
        try:
            self._guard.observe_audio(frame)
        except Exception:
            log.exception("baobab: pitch observation failed")

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        # Means "that pause was not the end after all", NOT "the utterance
        # finished" -- so the tail is deliberately left alone here.
        self._inner.cancel_inference(timed_out=timed_out)

    def flush(self, reason: str | None = None) -> None:
        # This is the turn boundary. Drop the tail so the next turn cannot
        # match a marker left over from the previous one.
        try:
            self._guard.tail.reset()
            self._guard.reset_turn()
        except Exception:
            log.exception("baobab: tail reset failed")
        self._inner.flush(reason)

    def end_input(self) -> None:
        self._inner.end_input()

    async def aclose(self) -> None:
        await self._inner.aclose()

    def set_agent_speaking(self, speaking: bool) -> None:
        """Tell the guard the agent has the floor, by hand.

        `attach()` tracks this automatically from `agent_state_changed`, so
        this is only needed when the guard is driven without a session. The
        two are OR-ed: a manual override can assert the floor but never
        release one the session is reporting.

        While the agent speaks, the session suppresses transcript events and
        replays them later, so any tail we hold is stale by construction.
        That is the barge-in case, where being confidently wrong costs the
        most, so the guard abstains instead.
        """
        self._agent_speaking = speaking

    # --- the prediction ------------------------------------------------

    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        """Wrap the inner prediction, adjust it, and never fail loudly.

        Returns our own future rather than the inner one so that a guard
        failure can still resolve with something sane. Note we never call
        `set_exception`: the session's await on this is not guarded, and an
        exception there kills the turn.
        """
        loop = asyncio.get_running_loop()
        out: asyncio.Future[TurnDetectionEvent] = loop.create_future()

        try:
            inner_future = self._inner.predict()
        except Exception:
            log.exception("baobab: inner predict() raised; failing open")
            out.set_result(_default_event())
            return out

        def _finish(fut: asyncio.Future[TurnDetectionEvent]) -> None:
            if out.done():
                # The session cancelled us. Nothing to do, and setting a
                # result on a cancelled future would itself raise.
                return
            try:
                event = fut.result()
            except asyncio.CancelledError:
                out.cancel()
                return
            except Exception:
                log.exception("baobab: inner prediction failed; failing open")
                out.set_result(_default_event())
                return

            try:
                event = self._adjust(event)
            except Exception:
                # The guard broke. The inner detector's opinion is still
                # perfectly good, so hand it back untouched.
                log.exception("baobab: guard failed; passing inner result through")
            out.set_result(event)

        inner_future.add_done_callback(_finish)
        return out

    def _adjust(self, event: TurnDetectionEvent) -> TurnDetectionEvent:
        """Run the guard and write the result back into the event.

        Mutates in place and returns the same object. That is safe because
        the inner detector allocates a fresh event per prediction, so the
        session's identity-based hook dedupe still sees a distinct object,
        and every other field carries through without copying.
        """
        if not self._enabled:
            return event

        snap = self._guard.tail.snapshot()
        mean_pause, stdev_pause = self._guard.pacing.as_guard_inputs()
        gi = GuardInput(
            p_inner=event.end_of_turn_probability,
            transcript_tail=snap.text,
            is_final=snap.is_final,
            tail_age_ms=snap.age_ms,
            language=snap.language,
            # Either source may assert the floor. The session's view comes
            # from `agent_state_changed`; the flag is a manual override for
            # callers driving the stream without a session.
            agent_speaking=self._agent_speaking or self._guard.agent_speaking,
            code_switch=self._guard.code_switch,
            mean_pause_ms=mean_pause,
            stdev_pause_ms=stdev_pause,
            pitch_contour=self._guard.pitch_contour,
        )

        result = run_guard(gi, self._guard.config, self._guard.index_for(snap.language))

        if result.p_adjusted is not None:
            event.end_of_turn_probability = result.p_adjusted

        # Read back by `unlikely_threshold`, which the session calls as soon
        # as this prediction resolves.
        self._pending_delay_ms = result.delay_override_ms

        if self._on_decision is not None:
            try:
                self._on_decision(gi, result)
            except Exception:
                log.exception("baobab: on_decision hook raised")

        return event


def _as_code(language: Any) -> str | None:
    """LiveKit passes a `LanguageCode`, which may be an enum or a plain str."""
    if language is None:
        return None
    return getattr(language, "value", None) or str(language)
