"""Chat turn detection.

The case that justifies the whole module is `test_words_outrank_silence`: a
burst ending in "the thing be say" must not be answered however long the user
pauses. A plain inactivity timeout -- which is what most chat bots use --
gets that wrong every time.
"""

from __future__ import annotations

import pytest

from baobab_turn.adapters.chat import ChatConfig, ChatTurnGuard


class Clock:
    """Controllable monotonic clock, so timing is tested without sleeping."""

    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def chat(clock) -> ChatTurnGuard:
    return ChatTurnGuard(language="pcm", now=clock)


# --- the baseline ------------------------------------------------------


def test_nothing_to_answer_before_any_message(chat) -> None:
    d = chat.should_reply()
    assert not d.reply
    assert d.reasons == ("nothing_to_answer",)


def test_typing_suppresses_the_reply(chat, clock) -> None:
    """The least ambiguous signal in the system: someone mid-keystroke is
    not finished, whatever their words so far suggest."""
    chat.on_message("i go come tomorrow")
    clock.tick(10)
    assert chat.should_reply().reply

    chat.on_typing(True)
    d = chat.should_reply()
    assert not d.reply
    assert "typing" in d.reasons


def test_baseline_ramps_with_silence(chat, clock) -> None:
    chat.on_message("i go come tomorrow")
    fresh = chat.should_reply().probability
    clock.tick(1.0)
    half = chat.should_reply().probability
    clock.tick(5.0)
    settled = chat.should_reply().probability
    assert fresh < half < settled


def test_settled_silence_triggers_a_reply(chat, clock) -> None:
    chat.on_message("i go come tomorrow")
    assert not chat.should_reply().reply
    clock.tick(3.0)
    d = chat.should_reply()
    assert d.reply
    assert "settled" in d.reasons


def test_a_message_clears_the_typing_flag(chat, clock) -> None:
    """Transports are inconsistent about clearing this, and a stuck flag
    would mean never replying again."""
    chat.on_typing(True)
    chat.on_message("i go come tomorrow")
    clock.tick(3.0)
    assert chat.should_reply().reply


# --- where the markers earn their place --------------------------------


def test_words_outrank_silence(chat, clock) -> None:
    """The case this module exists for.

    A burst ending in "the thing be say" announces a clause that has not
    arrived. Silence means nothing against that, and an inactivity timeout --
    what most chat bots use -- would answer into the middle of a sentence.
    """
    chat.on_message("so")
    chat.on_message("the thing be say")
    clock.tick(30.0)

    d = chat.should_reply()
    assert not d.reply
    assert any(r.startswith("continuation_veto") for r in d.reasons)


def test_a_yield_marker_brings_the_reply_forward(chat, clock) -> None:
    """'abi' hands over, so it should reply sooner than bare text would."""
    chat.on_message("i no fit come today abi")
    clock.tick(1.5)
    with_marker = chat.should_reply()

    chat.on_reply_sent()
    chat.on_message("i no fit come today")
    clock.tick(1.5)
    without = chat.should_reply()

    assert with_marker.probability > without.probability


def test_markers_alone_never_reply_into_active_typing(chat) -> None:
    """A yield marker must not beat the typing indicator -- they may have
    hit send early and be correcting themselves."""
    chat.on_message("e don finish abi")
    chat.on_typing(True)
    assert not chat.should_reply().reply


# --- bursts ------------------------------------------------------------


def test_burst_accumulates_messages(chat) -> None:
    for part in ("so", "the thing be say", "i no fit come"):
        chat.on_message(part)
    assert chat.burst_text() == "so the thing be say i no fit come"


def test_a_long_gap_starts_a_new_burst(chat, clock) -> None:
    """After a minute the earlier messages were answered or abandoned.
    Matching a marker from them would judge a conversation that moved on."""
    chat.on_message("the thing be say")
    clock.tick(60.0)
    chat.on_message("abi you dey there")
    assert chat.burst_text() == "abi you dey there"


def test_reply_sent_closes_the_burst(chat) -> None:
    chat.on_message("the thing be say")
    chat.on_reply_sent()
    assert chat.burst_text() == ""
    assert not chat.should_reply().reply


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_messages_are_ignored(chat, text: str) -> None:
    chat.on_message(text)
    assert chat.burst_text() == ""


def test_replying_makes_the_markers_abstain(chat, clock) -> None:
    """The chat equivalent of the agent holding the floor: anything arriving
    now answers a half-delivered message, so the burst is stale evidence."""
    chat.on_message("the thing be say")
    chat.set_replying(True)
    clock.tick(30.0)
    d = chat.should_reply()
    assert "tail_absent" in d.reasons
    assert not any(r.startswith("continuation_veto") for r in d.reasons)


# --- the wait hint -----------------------------------------------------


def test_wait_is_zero_when_replying(chat, clock) -> None:
    chat.on_message("i go come tomorrow")
    clock.tick(3.0)
    d = chat.should_reply()
    assert d.reply and d.wait_ms == 0


def test_wait_is_positive_and_bounded_when_holding(chat) -> None:
    chat.on_message("the thing be say")
    d = chat.should_reply()
    assert not d.reply
    assert 0 < d.wait_ms <= ChatConfig().settle_ms


def test_wait_never_overshoots_the_settle_point(chat, clock) -> None:
    """Re-asking after the burst has already settled wastes a round trip."""
    chat.on_message("i go come tomorrow")
    clock.tick(1.9)
    d = chat.should_reply()
    if not d.reply:
        assert d.wait_ms <= ChatConfig().settle_ms - 1_900 + 1


# --- degenerate cases --------------------------------------------------


def test_works_with_no_pack_at_all(clock) -> None:
    """An unknown language must fall back to pure timing rather than break --
    the same null-pack guarantee the voice path gives."""
    chat = ChatTurnGuard(language="sw", now=clock)
    chat.on_message("habari yako rafiki")
    assert not chat.should_reply().reply
    clock.tick(3.0)
    assert chat.should_reply().reply


def test_custom_timing_is_respected(clock) -> None:
    chat = ChatTurnGuard(language="pcm", now=clock,
                         chat_config=ChatConfig(settle_ms=500))
    chat.on_message("i go come tomorrow")
    clock.tick(0.6)
    assert chat.should_reply().reply


def test_hostile_messages_do_not_raise(chat) -> None:
    for text in ("\x00\x01", "!!!???", "a" * 10_000, "🙂" * 200):
        chat.on_message(text)
        chat.should_reply()
