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

from ..core.config import BaobabConfig
from ..core.guard import guard as run_guard
from ..core.markers import MarkerIndex
from ..core.tail import TranscriptTail
from ..core.types import GuardInput, GuardResult
from ..packs.registry import resolve

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
        self.turn_detection = BaobabTurnDetector(
            inner=inner if inner is not None else inference.TurnDetector(),
            guard=self,
            enabled=enabled,
            on_decision=on_decision,
        )
        self._indexes: dict[str, MarkerIndex | None] = {}

    def index_for(self, language: str | None) -> MarkerIndex | None:
        """Cached pack lookup. None means no pack, i.e. stock behaviour."""
        key = language or ""
        if key not in self._indexes:
            pack = resolve(language, allow_community=self.config.allow_community_packs)
            self._indexes[key] = MarkerIndex(pack) if pack is not None else None
        return self._indexes[key]

    def attach(self, session: Any) -> None:
        """Subscribe to the session's transcript events.

        The handler must be a plain `def`: the SDK's emitter calls handlers
        inline and never wraps them in a task, so an `async def` here would
        return an un-awaited coroutine and silently do nothing.
        """

        def _on_transcript(event: Any) -> None:
            try:
                self.tail.update(
                    getattr(event, "transcript", "") or "",
                    is_final=bool(getattr(event, "is_final", False)),
                    language=getattr(event, "language", None),
                )
            except Exception:
                # A transcript we failed to record is a missing opinion, not
                # a broken call. Never let this reach the emitter.
                log.exception("baobab: failed to record transcript")

        session.on("user_input_transcribed", _on_transcript)


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
        """A pack's threshold wins, otherwise the inner detector's.

        This is the only graded lever LiveKit gives us. It never consumes a
        millisecond delay from a detector -- it flips between `min_delay` and
        `max_delay` on this threshold -- so a pack that wants more patience
        for a tonal language expresses it here rather than through
        `delay_override_ms`.
        """
        try:
            index = self._guard.index_for(_as_code(language))
            if index is not None and index.pack.unlikely_threshold is not None:
                return index.pack.unlikely_threshold
        except Exception:
            log.exception("baobab: unlikely_threshold failed, deferring to inner")
        return await self._inner.unlikely_threshold(language)

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
        self._inner.push_audio(frame)

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        # Means "that pause was not the end after all", NOT "the utterance
        # finished" -- so the tail is deliberately left alone here.
        self._inner.cancel_inference(timed_out=timed_out)

    def flush(self, reason: str | None = None) -> None:
        # This is the turn boundary. Drop the tail so the next turn cannot
        # match a marker left over from the previous one.
        try:
            self._guard.tail.reset()
        except Exception:
            log.exception("baobab: tail reset failed")
        self._inner.flush(reason)

    def end_input(self) -> None:
        self._inner.end_input()

    async def aclose(self) -> None:
        await self._inner.aclose()

    def set_agent_speaking(self, speaking: bool) -> None:
        """Tell the guard the agent has the floor.

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
        gi = GuardInput(
            p_inner=event.end_of_turn_probability,
            transcript_tail=snap.text,
            is_final=snap.is_final,
            tail_age_ms=snap.age_ms,
            language=snap.language,
            agent_speaking=self._agent_speaking,
        )

        result = run_guard(gi, self._guard.config, self._guard.index_for(snap.language))

        if result.p_adjusted is not None:
            event.end_of_turn_probability = result.p_adjusted

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
