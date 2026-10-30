"""Turn detection for chat, where the same problem has a different shape.

People do not write one message per thought. They send

    so
    the thing be say
    i no fit come today

as three messages in six seconds, and a bot that answers the first one has
made exactly the mistake this package exists to stop -- it interrupted.

**Chat is a better fit for this engine than voice, not a worse one.** The
guard has always been a pure function of transcript tail, state and a
probability; it never touches audio. Every caveat on the voice benchmark --
no reliable timing, STT lag and revision, a tail that goes stale during
barge-in -- is an artefact of getting text out of speech. In chat the text is
exact, immediate, and never revised.

## What replaces the inner detector

In voice we wrap somebody else's detector and correct it. Chat has no such
thing, so this module *is* the inner detector, built from the two signals a
chat transport actually gives you:

- **the typing indicator** -- unambiguous evidence of more to come
- **silence since the last message** -- evidence of being done

Those produce a baseline probability, and the guard's markers then refine it
with what the words say. A message ending in "the thing be say" is not
finished however long the silence runs; one ending in "abi" is finished the
moment typing stops.

## What it deliberately does not do

No framework is imported here, because there is no framework to adapt: the
event model is the same for Slack, WhatsApp, Telegram or a web socket. Feed
it three events and ask it one question.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.codeswitch import CodeSwitchDetector, LanguageProfile
from ..core.config import BaobabConfig
from ..core.guard import guard as run_guard
from ..core.markers import MarkerIndex
from ..core.pacing import PacingTracker
from ..core.types import GuardInput
from ..packs.registry import available_packs, resolve

__all__ = ["ChatConfig", "ChatTurnGuard", "ReplyDecision"]


@dataclass(frozen=True, slots=True)
class ChatConfig:
    """Timing for the chat baseline. Milliseconds throughout."""

    #: Silence after which a burst is treated as finished, absent any other
    #: evidence. Below this the baseline rises gradually rather than flipping,
    #: so the markers still have room to move the decision either way.
    settle_ms: int = 2_000

    #: Probability floor while the typing indicator is live. Deliberately low:
    #: someone actively typing is the least ambiguous signal in the whole
    #: system, far stronger than anything the words can say.
    typing_probability: float = 0.1

    #: Where the baseline starts the instant a message lands, before any
    #: silence has accumulated.
    fresh_probability: float = 0.3

    #: Where it saturates once `settle_ms` has passed with no typing.
    settled_probability: float = 0.95

    #: Messages closer together than this are one burst. Beyond it the
    #: previous burst is considered answered or abandoned.
    burst_gap_ms: int = 30_000

    #: How long to wait before asking again when the answer is "not yet".
    poll_interval_ms: int = 250

    #: What counts as this person's typing rhythm, rather than an outlier.
    #: Much wider than the speech defaults: a four-second gap between two
    #: chat messages is ordinary, where in speech it would mean the caller
    #: had stopped to look something up.
    min_gap_ms: int = 250
    max_gap_ms: int = 20_000

    #: Ceiling on a settle time extended by measured pacing. Without it a
    #: sender who happens to produce a few long gaps would push the agent
    #: into never answering at all.
    max_settle_ms: int = 8_000


@dataclass(frozen=True, slots=True)
class ReplyDecision:
    """Whether to reply now, and why."""

    reply: bool
    probability: float
    #: How long to wait before re-asking, when `reply` is False.
    wait_ms: int
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return " · ".join(self.reasons)


class ChatTurnGuard:
    """Decides when a chat agent should reply.

    Not thread-safe and deliberately so: one instance belongs to one
    conversation, and conversations are naturally serialised by the transport
    that delivers their events.

        chat = ChatTurnGuard(language="pcm")
        chat.on_typing(True)
        chat.on_message("the thing be say")
        chat.on_typing(False)

        decision = chat.should_reply()
        if decision.reply:
            ...answer chat.burst_text()
    """

    __slots__ = (
        "_allow_community",
        "_burst",
        "_chat",
        "_code_switch",
        "_codeswitch",
        "_config",
        "_index",
        "_last_message_at",
        "_now",
        "_pacing",
        "_replying",
        "_typing",
        "language",
    )

    def __init__(
        self,
        language: str | None = None,
        config: BaobabConfig | None = None,
        chat_config: ChatConfig | None = None,
        now: Callable[[], float] = time.monotonic,
        allow_community_packs: bool = False,
    ) -> None:
        self.language = language
        self._config = config or BaobabConfig()
        self._chat = chat_config or ChatConfig()
        self._now = now
        pack = resolve(language, allow_community=allow_community_packs)
        self._index = MarkerIndex(pack) if pack is not None else None
        self._burst: list[str] = []
        self._last_message_at: float | None = None
        self._typing = False
        self._replying = False
        self._allow_community = allow_community_packs
        #: This person's own rhythm between messages in a burst. Someone who
        #: sends three messages four seconds apart is not finished after two
        #: seconds of quiet just because a faster sender would be.
        self._pacing = PacingTracker(
            min_pause_ms=self._chat.min_gap_ms,
            max_pause_ms=self._chat.max_gap_ms,
        )
        self._codeswitch: CodeSwitchDetector | None = None
        self._code_switch = False

    # --- events -------------------------------------------------------

    def on_message(self, text: str) -> None:
        """A message arrived from the user.

        A long enough gap since the previous one starts a fresh burst: the
        earlier messages have been answered or abandoned, and letting them
        linger would match a marker from a conversation that already moved on.
        """
        if not text or not text.strip():
            return
        now = self._now()
        if self._last_message_at is not None:
            gap_ms = (now - self._last_message_at) * 1000.0
            if gap_ms > self._chat.burst_gap_ms:
                self._burst.clear()
                self._code_switch = False
            else:
                # Only gaps *within* a burst are rhythm. The gap across a
                # burst boundary is however long it took someone to come
                # back to the conversation, which says nothing about how
                # fast they type.
                self._pacing.observe(gap_ms)
        self._burst.append(text.strip())
        self._last_message_at = now
        self._detect_code_switch()
        # A message having arrived means they are no longer mid-keystroke on
        # it. Transports are inconsistent about clearing this themselves.
        self._typing = False

    def on_typing(self, active: bool) -> None:
        """The typing indicator changed."""
        self._typing = bool(active)

    def on_reply_sent(self) -> None:
        """We answered. The burst is closed."""
        self._burst.clear()
        self._last_message_at = None
        self._typing = False
        self._replying = False
        # Pacing deliberately survives. How fast this person types is a
        # property of the person, not of the burst that just ended, and
        # resetting it every reply would mean never gathering enough
        # observations to be ready.
        self._code_switch = False

    def set_replying(self, replying: bool) -> None:
        """Whether the agent is currently generating or streaming a reply.

        The chat equivalent of the agent holding the floor. While it is true
        the markers abstain, for the same reason as in voice: anything the
        user is sending now is a response to a half-delivered message, and
        judging it against a stale burst is worse than not judging it.
        """
        self._replying = bool(replying)

    # --- sensors ------------------------------------------------------

    def _detector(self) -> CodeSwitchDetector:
        """Built lazily from every pack we are allowed to use.

        Lazily because building profiles reads every pack on disk, and a
        conversation that never receives a message should not pay for it.
        """
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

    def _detect_code_switch(self) -> None:
        """Re-measure across the whole burst, not the latest message alone.

        Switching languages is something people do *between* messages as
        readily as within one -- "i no fit come today" then "let me explain"
        is a switch that no single message contains. Measuring the burst as
        one string is the only way to see it.
        """
        if not self._config.enable_codeswitch:
            return
        try:
            self._code_switch = self._detector().detect(self.burst_text()).switched
        except Exception:
            # A sensor is never a reason to stop answering somebody.
            self._code_switch = False

    @property
    def code_switch(self) -> bool:
        """Whether the current burst crosses a language boundary."""
        return self._code_switch

    @property
    def pacing(self) -> PacingTracker:
        """This person's rhythm between messages. Exposed for inspection."""
        return self._pacing

    def _settle_ms(self) -> float:
        """How long the quiet must run before the burst counts as finished.

        Extended by the sender's own rhythm, because the default is a guess
        about an average typist and this is a measurement of *this* one.
        Someone whose messages arrive four seconds apart has not finished
        after two seconds of quiet; treating them as finished is the chat
        form of interrupting, which is the thing this package exists to
        stop.
        """
        base = float(self._chat.settle_ms)
        if not self._config.enable_pacing or not self._pacing.ready:
            return base
        mean, stdev = self._pacing.as_guard_inputs()
        if mean is None:
            return base
        target = mean + (stdev or 0.0)
        return min(max(base, target), float(self._chat.max_settle_ms))

    # --- state --------------------------------------------------------

    def burst_text(self) -> str:
        """Everything the user has said since the last reply, joined."""
        return " ".join(self._burst)

    @property
    def silence_ms(self) -> float | None:
        if self._last_message_at is None:
            return None
        return (self._now() - self._last_message_at) * 1000.0

    def _baseline(self) -> tuple[float, str]:
        """The probability before the words are considered."""
        if self._typing:
            return self._chat.typing_probability, "typing"
        silence = self.silence_ms
        if silence is None:
            return self._chat.fresh_probability, "no_messages"
        settle = self._settle_ms()
        if silence >= settle:
            return self._chat.settled_probability, "settled"
        # Linear ramp between fresh and settled. Linear on purpose: anything
        # cleverer would be a curve fitted to no data.
        fraction = silence / settle
        span = self._chat.settled_probability - self._chat.fresh_probability
        return self._chat.fresh_probability + span * fraction, "settling"

    # --- the decision -------------------------------------------------

    def should_reply(self) -> ReplyDecision:
        """Reply now, or wait?"""
        if not self._burst:
            return ReplyDecision(False, 0.0, self._chat.poll_interval_ms, ("nothing_to_answer",))

        p_base, why = self._baseline()
        mean_pause, stdev_pause = self._pacing.as_guard_inputs()
        result = run_guard(
            GuardInput(
                p_inner=p_base,
                transcript_tail=self.burst_text(),
                # Chat messages are never revised, so the text is always
                # final in the sense the guard means -- which is why the
                # ambiguous markers can be trusted here and not mid-speech.
                is_final=True,
                tail_age_ms=0.0,
                language=self.language,
                agent_speaking=self._replying,
                code_switch=self._code_switch,
                mean_pause_ms=mean_pause,
                stdev_pause_ms=stdev_pause,
            ),
            self._config,
            self._index,
        )

        p = result.p_adjusted if result.p_adjusted is not None else p_base
        reasons = (why, *result.reasons)

        if result.decision:
            return ReplyDecision(True, p, 0, reasons)

        # Not yet. Suggest when to ask again: either the guard's own delay, or
        # long enough for the silence to reach settle_ms, whichever is sooner.
        wait = self._chat.poll_interval_ms
        if result.delay_override_ms is not None:
            wait = max(wait, result.delay_override_ms)
        silence = self.silence_ms
        if silence is not None and not self._typing:
            remaining = self._settle_ms() - silence
            if 0 < remaining < wait:
                wait = int(remaining)
        return ReplyDecision(False, p, int(wait), reasons)
