"""The rule engine, tested as a pure function with fixture inputs.

No framework session is involved, which is the point of the core/adapter
split: every decision the guard makes is reproducible from a dataclass.
"""

from __future__ import annotations

import pytest

from baobab_turn.core.config import BaobabConfig
from baobab_turn.core.guard import guard
from baobab_turn.core.markers import MarkerIndex
from baobab_turn.core.types import GuardInput, LanguagePack, Marker

CONT = (Marker("the thing be say", 1.0), Marker("so", 0.5))
YIELD = (Marker("abi", 1.0), Marker("you get me", 0.5))
AMBIG = (Marker("o", 0.4),)


def _index(**kw: object) -> MarkerIndex:
    return MarkerIndex(
        LanguagePack(
            code="test",
            name="Test",
            continuation_markers=CONT,
            yield_markers=YIELD,
            ambiguous_markers=AMBIG,
            **kw,  # type: ignore[arg-type]
        )
    )


@pytest.fixture
def index() -> MarkerIndex:
    return _index()


def _gi(**kw: object) -> GuardInput:
    base: dict[str, object] = {"p_inner": 0.75, "tail_age_ms": 10.0}
    base.update(kw)
    return GuardInput(**base)  # type: ignore[arg-type]


# --- the lexical rules -------------------------------------------------


def test_continuation_marker_lowers_probability(index: MarkerIndex) -> None:
    r = guard(_gi(transcript_tail="the thing be say"), index=index)
    assert r.p_adjusted is not None and r.p_adjusted < 0.75
    assert r.reasons == ("continuation_veto:the thing be say",)


def test_yield_marker_raises_probability(index: MarkerIndex) -> None:
    r = guard(_gi(transcript_tail="e don finish abi"), index=index)
    assert r.p_adjusted is not None and r.p_adjusted > 0.75
    assert r.reasons == ("yield_boost:abi",)


def test_marker_weight_scales_the_adjustment(index: MarkerIndex) -> None:
    """A 0.5-weight yield marker must move p less than a 1.0-weight one.
    Weights are the mechanism for encoding how much a marker is trusted, so
    if they did not scale the effect they would be decoration."""
    strong = guard(_gi(transcript_tail="abi"), index=index).p_adjusted
    weak = guard(_gi(transcript_tail="you get me"), index=index).p_adjusted
    assert strong is not None and weak is not None
    assert 0.75 < weak < strong


def test_continuation_beats_yield_when_both_could_match() -> None:
    """"so ... abi" is contrived, but the precedence must be deliberate:
    waiting wrongly costs a little latency, interrupting wrongly costs the
    caller's goodwill. Continuation is checked first."""
    idx = MarkerIndex(
        LanguagePack(
            code="t",
            name="T",
            continuation_markers=(Marker("abi", 1.0),),
            yield_markers=(Marker("abi", 1.0),),
        )
    )
    r = guard(_gi(transcript_tail="abi"), index=idx)
    assert r.p_adjusted is not None and r.p_adjusted < 0.75


def test_ambiguous_marker_only_counts_on_a_final_transcript(index: MarkerIndex) -> None:
    """Trailing "o" mid-utterance means nothing. Treating it as a hand-off is
    exactly how a detector cuts someone off."""
    interim = guard(_gi(transcript_tail="e don finish o", is_final=False), index=index)
    final = guard(_gi(transcript_tail="e don finish o", is_final=True), index=index)
    assert interim.p_adjusted == 0.75
    assert interim.reasons == ("no_rules_fired",)
    assert final.p_adjusted is not None and final.p_adjusted > 0.75


def test_ambiguous_marker_is_weaker_than_a_yield(index: MarkerIndex) -> None:
    ambiguous = guard(_gi(transcript_tail="finish o", is_final=True), index=index)
    yielded = guard(_gi(transcript_tail="finish abi", is_final=True), index=index)
    assert ambiguous.p_adjusted is not None and yielded.p_adjusted is not None
    assert 0.75 < ambiguous.p_adjusted < yielded.p_adjusted


def test_unremarkable_speech_is_left_alone(index: MarkerIndex) -> None:
    r = guard(_gi(transcript_tail="i go come tomorrow"), index=index)
    assert r.p_adjusted == 0.75
    assert r.passthrough
    assert r.reasons == ("no_rules_fired",)


# --- abstention: the three ways the tail is untrustworthy --------------


def test_stale_tail_abstains(index: MarkerIndex) -> None:
    """Nothing correlates the tail to the audio the probability came from, so
    an old tail looks authoritative and is not."""
    r = guard(_gi(transcript_tail="the thing be say", tail_age_ms=5000), index=index)
    assert r.p_adjusted == 0.75
    assert r.reasons == ("tail_stale",)
    assert r.passthrough


def test_tail_age_boundary_is_inclusive(index: MarkerIndex) -> None:
    cfg = BaobabConfig(max_tail_age_ms=800)
    at = guard(_gi(transcript_tail="abi", tail_age_ms=800), cfg, index)
    over = guard(_gi(transcript_tail="abi", tail_age_ms=801), cfg, index)
    assert at.reasons == ("yield_boost:abi",)
    assert over.reasons == ("tail_stale",)


def test_agent_speaking_abstains(index: MarkerIndex) -> None:
    """The transcript feed is suppressed while the agent talks, so any tail we
    hold is a leftover from before. This is the barge-in case."""
    r = guard(_gi(transcript_tail="abi", agent_speaking=True), index=index)
    assert r.p_adjusted == 0.75
    assert r.reasons == ("tail_absent",)


def test_missing_tail_abstains(index: MarkerIndex) -> None:
    for tail in (None, ""):
        r = guard(_gi(transcript_tail=tail), index=index)
        assert r.reasons == ("tail_absent",)
        assert r.p_adjusted == 0.75


def test_absent_tail_age_is_trusted(index: MarkerIndex) -> None:
    """None means 'unknown', not 'infinitely old' -- an adapter that cannot
    supply an age should not lose the whole lexical layer."""
    r = guard(_gi(transcript_tail="abi", tail_age_ms=None), index=index)
    assert r.reasons == ("yield_boost:abi",)


# --- patience ----------------------------------------------------------


def test_codeswitch_adds_patience(index: MarkerIndex) -> None:
    cfg = BaobabConfig()
    r = guard(_gi(transcript_tail="abi", code_switch=True), cfg, index)
    assert r.delay_override_ms == cfg.min_delay_ms + cfg.codeswitch_hold_ms
    assert "codeswitch_hold" in r.reasons


def test_tonal_pack_adds_patience() -> None:
    cfg = BaobabConfig()
    idx = _index(tonal=True, tonal_extra_ms=150)
    r = guard(_gi(transcript_tail="abi"), cfg, idx)
    assert r.delay_override_ms == cfg.min_delay_ms + 150
    assert "tonal_extra" in r.reasons


def test_non_tonal_pack_adds_no_patience(index: MarkerIndex) -> None:
    r = guard(_gi(transcript_tail="abi"), index=index)
    assert r.delay_override_ms is None
    assert "tonal_extra" not in r.reasons


def test_pacing_follows_a_slow_caller(index: MarkerIndex) -> None:
    r = guard(_gi(transcript_tail="abi", mean_pause_ms=900, stdev_pause_ms=200), index=index)
    assert r.delay_override_ms == 1100
    assert "pacing" in r.reasons


def test_pacing_never_shortens_below_the_floor(index: MarkerIndex) -> None:
    """A fast talker must not be able to drive the delay under min_delay."""
    cfg = BaobabConfig()
    r = guard(_gi(transcript_tail="abi", mean_pause_ms=10, stdev_pause_ms=1), cfg, index)
    assert r.delay_override_ms is None or r.delay_override_ms >= cfg.min_delay_ms


def test_delay_is_clamped_to_the_ceiling(index: MarkerIndex) -> None:
    cfg = BaobabConfig(max_delay_ms=1000)
    r = guard(_gi(transcript_tail="abi", mean_pause_ms=99_000), cfg, index)
    assert r.delay_override_ms == 1000


def test_pack_delay_bounds_override_config() -> None:
    cfg = BaobabConfig(min_delay_ms=300, max_delay_ms=2500)
    idx = _index(min_delay_ms=100, max_delay_ms=400)
    r = guard(_gi(transcript_tail="abi", mean_pause_ms=9000), cfg, idx)
    assert r.delay_override_ms == 400


# --- the decision ------------------------------------------------------


def test_decision_uses_the_threshold(index: MarkerIndex) -> None:
    cfg = BaobabConfig(decision_threshold=0.5)
    assert guard(_gi(p_inner=0.51, transcript_tail="x"), cfg, index).decision
    assert not guard(_gi(p_inner=0.49, transcript_tail="x"), cfg, index).decision


def test_decision_threshold_is_inclusive(index: MarkerIndex) -> None:
    cfg = BaobabConfig(decision_threshold=0.5)
    assert guard(_gi(p_inner=0.5, transcript_tail="x"), cfg, index).decision


def test_pack_threshold_overrides_config(index: MarkerIndex) -> None:
    cfg = BaobabConfig(decision_threshold=0.5)
    idx = _index(unlikely_threshold=0.9)
    assert not guard(_gi(p_inner=0.8, transcript_tail="x"), cfg, idx).decision
    assert guard(_gi(p_inner=0.8, transcript_tail="x"), cfg, index).decision


def test_probability_stays_in_range(index: MarkerIndex) -> None:
    """A stack of yield boosts must never push p above 1, and vetoes must
    never push it below 0 -- the framework would reject either."""
    for p in (0.0, 0.01, 0.5, 0.99, 1.0):
        for tail in ("abi", "the thing be say", "so"):
            r = guard(_gi(p_inner=p, transcript_tail=tail), index=index)
            assert r.p_adjusted is not None
            assert 0.0 <= r.p_adjusted <= 1.0


# --- the null pack and ablation ---------------------------------------


def test_no_pack_passes_everything_through() -> None:
    """SPEC 5.8.2: an unknown language degrades to stock behaviour, never to
    something worse than stock."""
    r = guard(_gi(transcript_tail="the thing be say abi"), index=None)
    assert r.p_adjusted == 0.75
    assert r.passthrough
    assert r.decision


def test_empty_pack_skips_the_lexical_layer() -> None:
    idx = MarkerIndex(LanguagePack(code="empty", name="Empty"))
    r = guard(_gi(transcript_tail="abi"), index=idx)
    assert r.p_adjusted == 0.75
    assert r.passthrough


@pytest.mark.parametrize(
    "flag,tail,kwargs",
    [
        ("enable_markers", "abi", {}),
        ("enable_codeswitch", "x", {"code_switch": True}),
        ("enable_pacing", "x", {"mean_pause_ms": 2000}),
    ],
)
def test_each_feature_can_be_switched_off(
    index: MarkerIndex, flag: str, tail: str, kwargs: dict[str, object]
) -> None:
    """Every flag exists so the benchmark can ablate it. A flag that does not
    actually disable its rule would make the ablation table a lie."""
    on = guard(_gi(transcript_tail=tail, **kwargs), BaobabConfig(), index)
    off = guard(_gi(transcript_tail=tail, **kwargs), BaobabConfig(**{flag: False}), index)
    assert (on.p_adjusted, on.delay_override_ms) != (off.p_adjusted, off.delay_override_ms)


def test_tonal_can_be_switched_off() -> None:
    idx = _index(tonal=True, tonal_extra_ms=150)
    on = guard(_gi(transcript_tail="x"), BaobabConfig(), idx)
    off = guard(_gi(transcript_tail="x"), BaobabConfig(enable_tonal=False), idx)
    assert on.delay_override_ms is not None
    assert off.delay_override_ms is None


# --- explainability ----------------------------------------------------


def test_reasons_are_never_empty(index: MarkerIndex) -> None:
    """A decision we cannot explain is a decision we cannot tune."""
    cases = [
        _gi(transcript_tail="abi"),
        _gi(transcript_tail="the thing be say"),
        _gi(transcript_tail="nothing here"),
        _gi(transcript_tail=None),
        _gi(p_inner=None, transcript_tail="abi"),
        _gi(transcript_tail="abi", tail_age_ms=9999),
    ]
    for gi in cases:
        assert guard(gi, index=index).reasons, f"no reasons for {gi}"


def test_config_validation_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        BaobabConfig(marker_window=0)
    with pytest.raises(ValueError):
        BaobabConfig(continuation_veto_strength=1.5)
    with pytest.raises(ValueError):
        BaobabConfig(min_delay_ms=3000, max_delay_ms=1000)
