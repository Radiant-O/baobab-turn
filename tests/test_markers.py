"""Marker matching, including the ambiguous cases SPEC 10 names explicitly.

The cases that matter most here are the ones where a naive implementation
looks correct: punctuation independence, longest-match-wins, and the window
boundary. Each of those silently produces wrong turn decisions rather than
errors, so each is pinned.
"""

from __future__ import annotations

import pytest

from baobab_turn.core.markers import MarkerIndex, normalise, tokenise
from baobab_turn.core.types import LanguagePack, Marker


def _pack(**kw: object) -> LanguagePack:
    return LanguagePack(code="test", name="Test", **kw)  # type: ignore[arg-type]


@pytest.fixture
def index() -> MarkerIndex:
    return MarkerIndex(
        _pack(
            continuation_markers=(
                Marker("the thing be say", 1.0),
                Marker("say", 0.5),
                Marker("so", 0.6),
                Marker("because", 0.8),
            ),
            yield_markers=(
                Marker("abi", 0.9),
                Marker("no be so", 0.9),
                Marker("you get me", 0.9),
            ),
            ambiguous_markers=(
                Marker("sha", 0.4),
                Marker("o", 0.25),
                Marker("now", 0.2),
            ),
        )
    )


# --- normalisation -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Abi?", "abi"),
        ("ABI!!!", "abi"),
        ("  so   then  ", "so then"),
        ("e don finish, o", "e don finish o"),
        ("you get me?!", "you get me"),
        ("", ""),
        ("...", ""),
    ],
)
def test_normalise(raw: str, expected: str) -> None:
    assert normalise(raw) == expected


def test_punctuation_is_never_a_signal() -> None:
    """Nigerian ASR punctuates unreliably, so "abi?" and "abi" must be the
    same token. A rule that depended on the question mark would work in
    testing and fail on real transcripts."""
    assert normalise("abi?") == normalise("abi") == "abi"


def test_curly_quotes_and_unicode_forms_normalise() -> None:
    """Providers emit typographic apostrophes and full-width characters;
    neither should cause a miss."""
    assert normalise("that’s all") == "thats all"
    assert normalise("ａｂｉ") == "abi"  # full-width abi


# --- matching ----------------------------------------------------------


def test_longest_match_wins(index: MarkerIndex) -> None:
    """"the thing be say" announces a clause is coming. Its final word "say"
    is a much weaker signal, and matching it instead would under-weight the
    strongest marker in the pack."""
    tokens = tokenise("i wan tell you say the thing be say")
    match = index.continuation(tokens)
    assert match == ("the thing be say", 1.0)


def test_match_must_end_at_the_tail(index: MarkerIndex) -> None:
    """A marker in the middle of the utterance says nothing about whether the
    speaker has finished now."""
    assert index.continuation(tokenise("so i go come tomorrow")) is None
    assert index.yields(tokenise("abi you go come")) is None


def test_window_excludes_distant_markers(index: MarkerIndex) -> None:
    """SPEC 5.2: a "so" fifteen words back is irrelevant."""
    far = "so " + " ".join(["word"] * 15)
    assert index.continuation(index.window(far, 6)) is None
    assert index.continuation(index.window("so", 6)) == ("so", 0.6)


def test_window_bounds_work_on_a_long_utterance(index: MarkerIndex) -> None:
    """Callers may pass the whole accumulated turn, not just a tail. The
    result must depend only on the end of it."""
    long_tail = ("blah " * 5000) + "e don finish abi"
    tokens = index.window(long_tail, 6)
    assert len(tokens) == 6
    assert index.yields(tokens) == ("abi", 0.9)


def test_window_truncation_does_not_invent_a_token(index: MarkerIndex) -> None:
    """Slicing the raw string can cut a word in half. A partial word must not
    become a token that was never spoken."""
    # 'xabi' sliced mid-word could otherwise surface as 'abi'.
    text = ("z" * 400) + "xabi"
    tokens = index.window(text, 6)
    assert "abi" not in tokens


def test_narrow_window_degrades_to_the_weaker_marker(index: MarkerIndex) -> None:
    """A four-word marker cannot be seen through a two-token window, so
    matching falls back to the strongest marker that fits -- here the much
    weaker standalone "say" (0.5 rather than 1.0).

    This is the real hazard of setting `config.marker_window` too low: it
    does not fail, it quietly under-weights the strongest markers in the
    pack. `marker_window` must be at least as wide as the longest phrase in
    any pack in use, and the Pidgin pack's longest is six words.
    """
    tail = "the thing be say"
    assert index.continuation(index.window(tail, 4)) == ("the thing be say", 1.0)
    assert index.continuation(index.window(tail, 2)) == ("say", 0.5)


# --- the ambiguous cases SPEC 5.2 calls out ---------------------------


def test_sha_matches_only_as_final_token(index: MarkerIndex) -> None:
    assert index.ambiguous(tokenise("i dey come sha")) == ("sha", 0.4)
    assert index.ambiguous(tokenise("sha i dey come")) is None


def test_trailing_o_is_ambiguous_not_a_yield(index: MarkerIndex) -> None:
    """"e don finish o" is usually terminal but can precede a continuation,
    so it must never be classed as a yield marker."""
    tokens = tokenise("e don finish o")
    assert index.ambiguous(tokens) == ("o", 0.25)
    assert index.yields(tokens) is None


def test_now_is_weak_and_not_temporal(index: MarkerIndex) -> None:
    """"come now" is emphasis, not the English temporal "now"."""
    assert index.ambiguous(tokenise("come now")) == ("now", 0.2)


# --- index construction ------------------------------------------------


def test_duplicate_marker_keeps_the_strongest_weight() -> None:
    """Whichever entry came last in the YAML should not silently win."""
    idx = MarkerIndex(
        _pack(continuation_markers=(Marker("so", 0.2), Marker("so", 0.9)))
    )
    assert idx.continuation(["so"]) == ("so", 0.9)


def test_marker_text_is_normalised_at_build_time() -> None:
    """A pack author writing "Abi?" should still get a match."""
    idx = MarkerIndex(_pack(yield_markers=(Marker("Abi?", 0.9),)))
    assert idx.yields(["abi"]) == ("abi", 0.9)


def test_empty_pack_is_falsy() -> None:
    """Lets the guard skip the whole lexical layer rather than scanning
    empty tables on every partial transcript."""
    assert not MarkerIndex(_pack())
    assert MarkerIndex(_pack(yield_markers=(Marker("abi", 0.9),)))


def test_weight_out_of_range_is_rejected() -> None:
    """A typo'd weight should fail loudly at load, not skew decisions."""
    with pytest.raises(ValueError, match="weight out of range"):
        Marker("abi", 1.5)


@pytest.mark.parametrize("tail", ["", "   ", "!!!"])
def test_blank_tails_match_nothing(index: MarkerIndex, tail: str) -> None:
    tokens = index.window(tail, 6)
    assert index.continuation(tokens) is None
    assert index.yields(tokens) is None
    assert index.ambiguous(tokens) is None
