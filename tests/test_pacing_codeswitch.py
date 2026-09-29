"""Pacing and code-switch detection.

These two matter more than the lexicon now. The benchmark showed the marker
layer fires on ~4% of decision points and is close to a structural ceiling
there; pacing and code-switching apply on every turn, so they are where a
headline number can still move.
"""

from __future__ import annotations

import math

import pytest

from baobab_turn.core.codeswitch import CodeSwitchDetector, LanguageProfile
from baobab_turn.core.pacing import PacingTracker
from baobab_turn.core.types import LanguagePack, Marker
from baobab_turn.packs.registry import resolve

# --- pacing ------------------------------------------------------------


def test_not_ready_until_enough_samples() -> None:
    """One pause is not a rhythm. Reporting a mean from it would hand the
    guard a number that looks like evidence and is not."""
    t = PacingTracker(min_samples=3)
    t.observe(300)
    t.observe(400)
    assert not t.ready
    assert t.as_guard_inputs() == (None, None)
    t.observe(350)
    assert t.ready
    assert t.mean_ms == pytest.approx(350.0)


def test_mean_and_stdev() -> None:
    t = PacingTracker()
    for p in (300, 400, 500):
        t.observe(p)
    assert t.mean_ms == pytest.approx(400.0)
    assert t.stdev_ms == pytest.approx(100.0)


def test_stdev_needs_two_observations() -> None:
    t = PacingTracker(min_samples=1)
    t.observe(300)
    assert t.mean_ms == pytest.approx(300.0)
    assert t.stdev_ms is None


@pytest.mark.parametrize("pause", [0, 10, 39.9, 5001, 60_000, -100])
def test_implausible_pauses_are_dropped_not_clamped(pause: float) -> None:
    """A clamped outlier still drags the mean toward the boundary. These
    events are not speech rhythm at all -- an 8-second gap is someone looking
    something up -- so they are discarded."""
    t = PacingTracker()
    assert t.observe(pause) is False
    assert t.n == 0


@pytest.mark.parametrize("pause", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_pauses_are_dropped(pause: float) -> None:
    """int(inf) raising OverflowError inside the guard is a hung call, so
    these must never reach it."""
    t = PacingTracker()
    assert t.observe(pause) is False
    assert t.n == 0
    assert t.as_guard_inputs() == (None, None)


def test_window_is_bounded_and_follows_a_changing_caller() -> None:
    """A caller who speeds up mid-call should stop being treated as slow."""
    t = PacingTracker(window=3)
    for p in (900, 900, 900):
        t.observe(p)
    assert t.mean_ms == pytest.approx(900.0)
    for p in (200, 200, 200):
        t.observe(p)
    assert t.n == 3
    assert t.mean_ms == pytest.approx(200.0)


def test_reset_clears_everything() -> None:
    t = PacingTracker()
    for p in (300, 400, 500):
        t.observe(p)
    t.reset()
    assert t.n == 0
    assert t.as_guard_inputs() == (None, None)


def test_guard_inputs_are_finite_when_present() -> None:
    t = PacingTracker()
    for p in (120, 4800, 300):
        t.observe(p)
    mean, stdev = t.as_guard_inputs()
    assert mean is not None and math.isfinite(mean)
    assert stdev is not None and math.isfinite(stdev)


def test_rejects_nonsense_construction() -> None:
    with pytest.raises(ValueError):
        PacingTracker(window=0)
    with pytest.raises(ValueError):
        PacingTracker(min_samples=0)


# --- code-switching ----------------------------------------------------


@pytest.fixture
def detector() -> CodeSwitchDetector:
    packs = [resolve("en-NG"), resolve("pcm")]
    assert all(p is not None for p in packs)
    return CodeSwitchDetector([LanguageProfile.from_pack(p) for p in packs])  # type: ignore[arg-type]


def test_shared_words_are_discarded_as_indiscriminate() -> None:
    """A word in both profiles cannot tell them apart, so keeping it only
    adds noise to both scores."""
    a = LanguageProfile("a", {"shared", "only_a"})
    b = LanguageProfile("b", {"shared", "only_b"})
    d = CodeSwitchDetector([a, b])
    words = {w for p in d._profiles for w in p.words}
    assert "shared" not in words
    assert {"only_a", "only_b"} <= words


def test_english_into_pidgin_is_detected(detector) -> None:
    r = detector.detect("i think we should probably go there but wetin dey happen abeg una sabi")
    assert r.switched
    assert r.tail_language == "pcm"


def test_pidgin_into_english_is_detected(detector) -> None:
    r = detector.detect(
        "wetin dey happen una sabi say oga don comot, i think we should reconsider that"
    )
    assert r.switched
    assert r.head_language == "pcm"


def test_sustained_english_is_not_a_switch(detector) -> None:
    r = detector.detect("i think that we should probably go there tomorrow morning if possible")
    assert not r.switched


def test_sustained_pidgin_is_not_a_switch(detector) -> None:
    """Consistently speaking one language is the opposite of switching. This
    is the false positive that would add patience to every Pidgin turn."""
    r = detector.detect("wetin dey happen for there i no sabi oga abeg wetin una dey do")
    assert not r.switched


@pytest.mark.parametrize("text", ["", "   ", "abi", "wetin dey", "one two three"])
def test_short_or_empty_input_is_never_a_switch(detector, text: str) -> None:
    """With almost no text any apparent switch is an artefact of the window."""
    assert not detector.detect(text).switched


def test_one_profile_is_enough_to_detect_a_switch() -> None:
    """A consequence of measuring density shift rather than identifying both
    halves: you only need a profile for the *marked* language.

    Pidgin density rising from head to tail is a switch into Pidgin, and no
    English profile is required to say so. That matters directly for the
    languages we cannot yet profile -- a Yoruba pack alone would detect
    switching into and out of Yoruba without anyone having to describe
    whatever it switched from.
    """
    pack = resolve("pcm")
    assert pack is not None
    d = CodeSwitchDetector([LanguageProfile.from_pack(pack)])

    into = d.detect("i think we should go there but wetin dey happen abeg una sabi")
    assert into.switched
    assert into.tail_language == "pcm"

    steady = d.detect("i think that we should probably go there tomorrow morning")
    assert not steady.switched


def test_no_profiles_at_all_is_never_a_switch() -> None:
    assert not CodeSwitchDetector([]).detect("wetin dey happen abeg una sabi").switched


def test_min_shift_suppresses_weak_evidence() -> None:
    """A switch buys a little extra patience, so a false positive on every
    other utterance costs more than an occasional miss."""
    pack = LanguagePack(
        code="x", name="X",
        continuation_markers=(Marker("zzz", 0.5),),
        profile_words=("zzz",),
    )
    other = LanguagePack(code="y", name="Y", profile_words=("qqq",))
    strict = CodeSwitchDetector(
        [LanguageProfile.from_pack(pack), LanguageProfile.from_pack(other)],
        min_shift=0.9,
    )
    assert not strict.detect("one two three four five zzz").switched


def test_profile_words_reach_the_profile() -> None:
    """Markers alone are too thin: en-NG and pcm share nearly all of them, so
    the pack-level vocabulary list is what makes discrimination possible."""
    pack = resolve("pcm")
    assert pack is not None
    profile = LanguageProfile.from_pack(pack)
    assert {"wetin", "dey", "abeg", "sabi"} <= profile.words


def test_hostile_input_does_not_raise(detector) -> None:
    for text in ("\x00\x01", "!!!???", "a" * 5000, "🙂" * 100, "ẹ̀ kú ìròlẹ́"):
        detector.detect(text)
