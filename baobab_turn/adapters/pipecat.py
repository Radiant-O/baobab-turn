"""Pipecat adapter: the framework this engine was shaped for.

## Why this one fits better

LiveKit hands a detector a float and applies its own threshold, and it never
accepts a millisecond delay -- `adapters/livekit.py` has to spend patience by
nudging `unlikely_threshold`, which only quantises a wait into "sooner" or
"later". Pipecat asks a plain question and takes a plain answer: COMPLETE or
INCOMPLETE. That means a delay can be honoured exactly, by continuing to
answer INCOMPLETE until the delay has run out. `GuardResult.delay_override_ms`
was written as an *intent* for precisely this reason, and here it is finally
spent as one.

## How it attaches

Verified against pipecat-ai 1.15.0 source, not its docs. In 1.15.0 the
analyzer is **not** a `TransportParams` field. It hangs off the user-turn
stop strategy:

    from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
        TurnAnalyzerUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair, LLMUserAggregatorParams,
    )

    analyzer = BaobabTurnAnalyzer(inner=LocalSmartTurnAnalyzerV3(), language="pcm")

    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)]
            )
        ),
    )

A `BaseTurnAnalyzer` is only ever given audio, never text, so the lexical
layer needs a second piece. `BaobabTranscriptObserver` is a pass-through
processor that reads transcription frames on their way past and feeds the
tail. Put it anywhere downstream of the STT service; it alters no frames.

## Two ways a turn completes

The strategy calls `append_audio` on every audio frame and treats COMPLETE
from it as the end of the turn. It also calls `analyze_end_of_turn` when VAD
reports the user stopped. Both paths are adjusted here, because a guard that
only covered one would be silently absent half the time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    BaseTurnParams,
    EndOfTurnState,
)
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..core.codeswitch import CodeSwitchDetector, LanguageProfile
from ..core.config import BaobabConfig
from ..core.guard import guard as run_guard
from ..core.markers import MarkerIndex
from ..core.pacing import PacingTracker
from ..core.tail import TranscriptTail
from ..core.types import GuardInput, GuardResult
from ..packs.registry import available_packs, resolve
from ..prosody import PitchTracker

__all__ = ["BaobabTranscriptObserver", "BaobabTurnAnalyzer"]

log = logging.getLogger("baobab_turn.pipecat")


class BaobabTurnAnalyzer(BaseTurnAnalyzer):
    """Wraps another analyzer and corrects it using a language pack.

    Fails open in every direction. If anything here raises, the inner
    analyzer's own verdict is returned untouched -- Pipecat catches
    exceptions at the frame-processor boundary and turns them into an
    ErrorFrame, which aborts the frame rather than the call, but a turn
    detector that drops frames is still a broken conversation.
    """

    def __init__(
        self,
        *,
        inner: BaseTurnAnalyzer,
        language: str | None = None,
        config: BaobabConfig | None = None,
        sample_rate: int | None = None,
        enabled: bool = True,
        allow_community_packs: bool = False,
        on_decision: Callable[[GuardInput, GuardResult], None] | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        self._inner = inner
        self._enabled = enabled
        self.config = config or BaobabConfig()
        self.language = language
        self._allow_community = allow_community_packs
        self._on_decision = on_decision
        self._now = now

        self.tail = TranscriptTail()
        self.pacing = PacingTracker()
        self.pitch: PitchTracker | None = (
            PitchTracker() if self.config.enable_tonal else None
        )
        self._pcm: list[int] = []
        self._codeswitch: CodeSwitchDetector | None = None
        self._code_switch = False

        self._indexes: dict[str, MarkerIndex | None] = {}
        #: When the caller last stopped speaking, for measuring the gap.
        self._silence_started: float | None = None
        self._was_speech = False
        #: Absolute time at which a delayed COMPLETE becomes allowed. This is
        #: the whole reason this adapter exists: on Pipecat a requested delay
        #: can be waited out exactly rather than approximated.
        self._hold_until: float | None = None

    # --- the parts of the protocol that are just pass-through ----------

    @property
    def speech_triggered(self) -> bool:
        return bool(self._inner.speech_triggered)

    @property
    def params(self) -> BaseTurnParams:
        return self._inner.params

    def set_sample_rate(self, sample_rate: int) -> None:
        super().set_sample_rate(sample_rate)
        self._inner.set_sample_rate(sample_rate)

    def update_vad_start_secs(self, vad_start_secs: float) -> None:
        self._inner.update_vad_start_secs(vad_start_secs)

    async def cleanup(self) -> None:
        await self._inner.cleanup()

    def clear(self) -> None:
        """Reset for a new turn.

        Pacing survives deliberately: a caller's rhythm belongs to the
        speaker, not the utterance that just ended. Everything else is about
        this utterance and must not leak into the next one.
        """
        self._inner.clear()
        self.tail.reset()
        self._code_switch = False
        self._hold_until = None
        self._pcm.clear()
        if self.pitch is not None:
            self.pitch.reset()

    # --- transcript, fed from the observer below -----------------------

    def on_transcript(self, text: str, *, is_final: bool, language: str | None) -> None:
        """Feed the lexical layer. Safe to call from any frame handler."""
        self.tail.update(text, is_final=is_final, language=language)
        if self.config.enable_codeswitch:
            try:
                self._code_switch = self._detector().detect(text).switched
            except Exception:
                self._code_switch = False

    def _detector(self) -> CodeSwitchDetector:
        if self._codeswitch is None:
            packs = [
                pack
                for pack in available_packs().values()
                if self._allow_community or pack.status != "community"
            ]
            self._codeswitch = CodeSwitchDetector(
                [LanguageProfile.from_pack(pack) for pack in packs]
            )
        return self._codeswitch

    def _index_for(self, language: str | None) -> MarkerIndex | None:
        key = language or ""
        if key not in self._indexes:
            pack = resolve(language, allow_community=self._allow_community)
            self._indexes[key] = MarkerIndex(pack) if pack is not None else None
        return self._indexes[key]

    # --- audio ---------------------------------------------------------

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Forward the audio, then correct the verdict.

        The inner analyzer is fed first and unconditionally. Everything after
        it is a sensor, and a sensor must never cost the detector its input.
        """
        state = self._inner.append_audio(buffer, is_speech)
        try:
            self._observe(buffer, is_speech)
        except Exception:
            log.exception("baobab: audio observation failed")
        if state != EndOfTurnState.COMPLETE:
            return state
        return self._decide(state)

    def _observe(self, buffer: bytes, is_speech: bool) -> None:
        """Pacing from the speech flag, pitch from the samples."""
        now = self._now()
        if is_speech and not self._was_speech and self._silence_started is not None:
            self.pacing.observe((now - self._silence_started) * 1000.0)
        if not is_speech and self._was_speech:
            self._silence_started = now
        self._was_speech = is_speech

        tracker = self.pitch
        rate = self.sample_rate
        if tracker is None or rate <= 0 or not buffer:
            return
        # Pipecat guarantees raw int16 PCM here; base_smart_turn reads it with
        # np.frombuffer(buffer, dtype=np.int16), so this matches the format
        # the framework's own analyzer assumes.
        usable = len(buffer) - (len(buffer) % 2)
        self._pcm.extend(
            int.from_bytes(buffer[i : i + 2], "little", signed=True)
            for i in range(0, usable, 2)
        )
        need = max(2, rate * self.config.pitch_window_ms // 1000)
        while len(self._pcm) >= need:
            window = self._pcm[:need]
            del self._pcm[:need]
            try:
                tracker.push_samples(window, rate)
            except Exception:
                # numpy is an optional extra. Disable for good rather than
                # raise once per audio frame for the rest of the call.
                self.pitch = None
                self._pcm.clear()
                log.warning(
                    "baobab: pitch tracking disabled; "
                    "install baobab-turn[prosody] for the tonal rule",
                    exc_info=True,
                )
                return

    # --- the verdict ---------------------------------------------------

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        """The VAD-stop path. Adjusted too, or the guard would be absent here.

        The inner analyzer's metrics are passed through untouched: they
        describe its inference, not our decision, and rewriting them would
        make the host's own numbers lie.
        """
        state, metrics = await self._inner.analyze_end_of_turn()
        if state != EndOfTurnState.COMPLETE:
            return state, metrics
        return self._decide(state), metrics

    def _decide(self, state: EndOfTurnState) -> EndOfTurnState:
        """Run the guard over a COMPLETE and decide whether it stands."""
        if not self._enabled:
            return state
        try:
            return self._guarded()
        except Exception:
            log.exception("baobab: guard failed, passing the turn through")
            return state

    def _guarded(self) -> EndOfTurnState:
        now = self._now()

        # A delay already granted is being waited out. Nothing is re-judged
        # until it expires: re-running the guard every frame would let one
        # late marker restart the clock indefinitely.
        if self._hold_until is not None:
            if now < self._hold_until:
                return EndOfTurnState.INCOMPLETE
            self._hold_until = None
            return EndOfTurnState.COMPLETE

        snap = self.tail.snapshot()
        mean_pause, stdev_pause = self.pacing.as_guard_inputs()
        gi = GuardInput(
            # Pipecat gives a decision, not a confidence. The guard's rules
            # are multiplicative, so COMPLETE enters as a probability and the
            # config says how certain that is taken to mean.
            p_inner=self.config.complete_probability,
            transcript_tail=snap.text,
            is_final=snap.is_final,
            tail_age_ms=snap.age_ms,
            language=snap.language or self.language,
            code_switch=self._code_switch,
            mean_pause_ms=mean_pause,
            stdev_pause_ms=stdev_pause,
            pitch_contour=self._pitch_contour(),
        )
        result = run_guard(gi, self.config, self._index_for(gi.language))

        if self._on_decision is not None:
            try:
                self._on_decision(gi, result)
            except Exception:
                log.exception("baobab: on_decision hook raised")

        if not result.decision:
            return EndOfTurnState.INCOMPLETE

        if result.delay_override_ms:
            # The thing LiveKit cannot do. Start the clock and say not yet;
            # the next audio frame that arrives after it expires completes.
            self._hold_until = now + result.delay_override_ms / 1000.0
            return EndOfTurnState.INCOMPLETE

        return EndOfTurnState.COMPLETE

    def _pitch_contour(self) -> str | None:
        tracker = self.pitch
        if tracker is None:
            return None
        contour = tracker.contour()
        return None if contour == "unknown" else contour


class BaobabTranscriptObserver(FrameProcessor):
    """Feeds transcripts to the analyzer without changing the pipeline.

    A `BaseTurnAnalyzer` is handed audio and never text, so without this the
    marker layer -- the part that knows "the thing be say" means more is
    coming -- would have nothing to read. Place it downstream of the STT
    service; every frame passes through untouched.
    """

    def __init__(self, analyzer: BaobabTurnAnalyzer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._analyzer = analyzer

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        try:
            if isinstance(frame, TranscriptionFrame):
                self._analyzer.on_transcript(
                    frame.text,
                    is_final=bool(getattr(frame, "finalized", False)),
                    language=_as_code(frame.language),
                )
            elif isinstance(frame, InterimTranscriptionFrame):
                self._analyzer.on_transcript(
                    frame.text, is_final=False, language=_as_code(frame.language)
                )
        except Exception:
            log.exception("baobab: transcript observation failed")
        await self.push_frame(frame, direction)


def _as_code(language: Any) -> str | None:
    """Pipecat passes a `Language` enum, or None."""
    if language is None:
        return None
    return getattr(language, "value", None) or str(language)
