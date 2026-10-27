"""Pitch tracking, and the tonal rule it finally gives evidence to.

The tonal hypothesis is the load-bearing claim for this project being
pan-African rather than Nigerian. Until now it was expressed as a fixed delay
applied to every speaker of a tonal language regardless of what their pitch
was doing -- a knob with no sensor. These tests cover the sensor, and the
rule that now fires only on a measured fall.
"""

from __future__ import annotations

import math

import pytest

from baobab_turn import BaobabConfig, GuardInput, MarkerIndex, guard
from baobab_turn.core.types import LanguagePack
from baobab_turn.prosody import PitchTracker, estimate_f0

np = pytest.importorskip("numpy", reason="prosody needs the [prosody] extra")

SR = 16_000


def voiced(hz: float, ms: int = 40, sample_rate: int = SR):
    """A synthetic voiced frame: a fundamental plus two harmonics, which is
    roughly what a vowel looks like to an autocorrelator."""
    t = np.arange(int(sample_rate * ms / 1000)) / sample_rate
    return (
        np.sin(2 * np.pi * hz * t)
        + 0.5 * np.sin(4 * np.pi * hz * t)
        + 0.25 * np.sin(6 * np.pi * hz * t)
    )


# --- f0 estimation -----------------------------------------------------


@pytest.mark.parametrize("hz", [85, 120, 180, 250, 350])
def test_recovers_the_fundamental(hz: int) -> None:
    est = estimate_f0(voiced(hz), SR)
    assert est is not None
    assert est == pytest.approx(hz, rel=0.05)


def test_octave_errors_are_not_silently_accepted() -> None:
    """Picking a harmonic instead of the fundamental is the classic failure
    of autocorrelation pitch tracking, and it would invent contours."""
    est = estimate_f0(voiced(150), SR)
    assert est is not None
    assert not (70 <= est <= 80), "picked the sub-octave"
    assert not (290 <= est <= 310), "picked the first harmonic"


def test_unvoiced_and_silence_return_none() -> None:
    """A fricative is not a low note, and digital silence is not a pitch.
    Interpolating through either would fabricate a contour."""
    rng = np.random.default_rng(0)
    assert estimate_f0(rng.normal(size=640), SR) is None
    assert estimate_f0(np.zeros(640), SR) is None


def test_a_dc_offset_does_not_fool_it() -> None:
    """A constant correlates perfectly with itself at every lag, so without
    mean removal it swamps the real periodicity."""
    est = estimate_f0(voiced(150) + 5.0, SR)
    assert est is not None
    assert est == pytest.approx(150, rel=0.05)


@pytest.mark.parametrize("samples", [[], [1.0], np.zeros(8)])
def test_too_short_input_returns_none(samples) -> None:
    assert estimate_f0(samples, SR) is None


def test_scaling_does_not_matter(sample_rate: int = SR) -> None:
    """int16 PCM and floats must give the same answer, since the result is
    normalised."""
    quiet = estimate_f0(voiced(150) * 0.001, sample_rate)
    loud = estimate_f0(voiced(150) * 20_000, sample_rate)
    assert quiet is not None and loud is not None
    assert quiet == pytest.approx(loud, rel=0.01)


# --- contour -----------------------------------------------------------


def _tracker(f0s: list[float]) -> PitchTracker:
    t = PitchTracker(sample_rate=SR)
    for f in f0s:
        t.push_f0(f)
    return t


def test_falling_rising_and_level() -> None:
    assert _tracker([200, 195, 185, 172, 160, 150, 142, 135]).contour() == "falling"
    assert _tracker([140, 148, 158, 170, 182, 195, 205, 215]).contour() == "rising"
    assert _tracker([170, 172, 169, 171, 170, 168, 172, 170]).contour() == "level"


def test_unknown_until_enough_voiced_frames() -> None:
    """Two points define a line through noise. The contour stays unknown
    rather than guessing."""
    t = PitchTracker(sample_rate=SR, min_points=4)
    t.push_f0(200)
    t.push_f0(150)
    assert t.contour() == "unknown"
    assert t.semitone_span() is None


def test_span_is_measured_in_semitones_not_hertz() -> None:
    """Pitch is perceived logarithmically: 20 Hz is a large move for a deep
    voice and a small one for a high voice, so hertz would make the threshold
    mean different things for different speakers."""
    low = _tracker([100, 100, 100, 150, 150, 150]).semitone_span()
    high = _tracker([200, 200, 200, 300, 300, 300]).semitone_span()
    assert low is not None and high is not None
    assert low == pytest.approx(high, abs=0.01)
    assert low == pytest.approx(12 * math.log2(1.5), abs=0.01)


def test_one_bad_frame_cannot_invent_a_contour() -> None:
    """Medians over the first and last thirds, so a single octave error does
    not turn a level contour into a falling one."""
    assert _tracker([170, 170, 170, 170, 170, 170, 85, 170, 170]).contour() == "level"


def test_history_is_bounded_to_the_recent_contour() -> None:
    """The turn-final shape carries the turn-taking meaning; what happened
    ten seconds ago does not."""
    t = PitchTracker(sample_rate=SR, history=6)
    for f in (100, 100, 100):
        t.push_f0(f)
    for f in (200, 205, 210, 215, 220, 225):
        t.push_f0(f)
    assert t.n == 6
    assert t.contour() in ("rising", "level")


def test_implausible_and_non_finite_f0_is_rejected() -> None:
    t = PitchTracker(sample_rate=SR)
    for bad in (0, -100, 20, 5_000, float("nan"), float("inf"), None):
        t.push_f0(bad)  # type: ignore[arg-type]
    assert t.n == 0
    assert t.contour() == "unknown"


def test_reset_clears_the_contour() -> None:
    t = _tracker([200, 190, 180, 170])
    assert t.contour() == "falling"
    t.reset()
    assert t.contour() == "unknown"


def test_push_samples_feeds_the_contour() -> None:
    t = PitchTracker(sample_rate=SR)
    for hz in (200, 190, 180, 170, 160, 150):
        assert t.push_samples(voiced(hz), SR) is not None
    assert t.contour() == "falling"


def test_push_samples_ignores_unvoiced_frames() -> None:
    rng = np.random.default_rng(1)
    t = PitchTracker(sample_rate=SR)
    assert t.push_samples(rng.normal(size=640), SR) is None
    assert t.n == 0


def test_rejects_nonsense_construction() -> None:
    with pytest.raises(ValueError):
        PitchTracker(sample_rate=0)
    with pytest.raises(ValueError):
        PitchTracker(history=1)


# --- the tonal rule ----------------------------------------------------

TONAL = MarkerIndex(LanguagePack(code="yo", name="Yoruba", tonal=True, tonal_extra_ms=150))
NON_TONAL = MarkerIndex(LanguagePack(code="en", name="English"))


def _p(index, contour, p_inner=0.8, config=None):
    return guard(
        GuardInput(p_inner=p_inner, transcript_tail="x", tail_age_ms=0.0,
                   pitch_contour=contour),
        config or BaobabConfig(),
        index,
    )


def test_a_fall_in_a_tonal_language_discounts_the_detector() -> None:
    """The hypothesis, made concrete: the detector read a falling pitch as
    the end of a sentence, but in Yoruba that may be a low tone inside a
    word."""
    r = _p(TONAL, "falling")
    assert r.p_adjusted is not None and r.p_adjusted < 0.8
    assert "tonal_falling_damp" in r.reasons


def test_a_fall_in_a_non_tonal_language_is_left_alone() -> None:
    """English really does use falling pitch to end sentences. Damping it
    there would make the detector worse for no reason."""
    r = _p(NON_TONAL, "falling")
    assert r.p_adjusted == 0.8
    assert "tonal_falling_damp" not in r.reasons


@pytest.mark.parametrize("contour", ["rising", "level", "unknown", None])
def test_only_a_measured_fall_damps(contour) -> None:
    """Without a sensor the rule must stay silent rather than damping every
    tonal utterance -- which is exactly what the fixed delay alone did."""
    r = _p(TONAL, contour)
    assert r.p_adjusted == 0.8
    assert "tonal_falling_damp" not in r.reasons


def test_the_rule_can_be_ablated() -> None:
    off = _p(TONAL, "falling", config=BaobabConfig(enable_tonal=False))
    assert off.p_adjusted == 0.8
    zero = _p(TONAL, "falling", config=BaobabConfig(tonal_falling_damp=0.0))
    assert zero.p_adjusted == 0.8


def test_damping_cannot_leave_the_valid_range() -> None:
    for p_inner in (0.0, 0.01, 0.5, 1.0):
        r = _p(TONAL, "falling", p_inner=p_inner)
        assert r.p_adjusted is not None
        assert 0.0 <= r.p_adjusted <= 1.0


def test_config_rejects_an_out_of_range_damp() -> None:
    with pytest.raises(ValueError):
        BaobabConfig(tonal_falling_damp=1.5)
