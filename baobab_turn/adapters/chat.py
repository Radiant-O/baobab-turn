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

from ..core.config import BaobabConfig
from ..core.guard import guard as run_guard
from ..core.markers import MarkerIndex
from ..core.types import GuardInput
from ..packs.registry import resolve

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

    __slots__ = ("_burst", "_chat", "_config", "_index", "_last_message_at",
                 "_now", "_replying", "_typing", "language")

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
        self._burst.append(text.strip())
        self._last_message_at = now
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

    def set_replying(self, replying: bool) -> None:
        """Whether the agent is currently generating or streaming a reply.

        The chat equivalent of the agent holding the floor. While it is true
        the markers abstain, for the same reason as in voice: anything the
        user is sending now is a response to a half-delivered message, and
        judging it against a stale burst is worse than not judging it.
        """
        self._replying = bool(replying)

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
        if silence >= self._chat.settle_ms:
            return self._chat.settled_probability, "settled"
        # Linear ramp between fresh and settled. Linear on purpose: anything
        # cleverer would be a curve fitted to no data.
        fraction = silence / self._chat.settle_ms
        span = self._chat.settled_probability - self._chat.fresh_probability
        return self._chat.fresh_probability + span * fraction, "settling"

    # --- the decision -------------------------------------------------

    def should_reply(self) -> ReplyDecision:
        """Reply now, or wait?"""
        if not self._burst:
            return ReplyDecision(False, 0.0, self._chat.poll_interval_ms, ("nothing_to_answer",))

        p_base, why = self._baseline()
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
            remaining = self._chat.settle_ms - silence
            if 0 < remaining < wait:
                wait = int(remaining)
        return ReplyDecision(False, p, int(wait), reasons)
