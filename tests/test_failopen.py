"""The most important test file in the project.

SPEC 14: "A turn detector that breaks a call is infinitely worse than one
that's merely mediocre." Everything here exists to prove that a bug in our
code cannot hang, crash, or corrupt a live conversation.

Scope note: the adapter-level half of fail-open -- never raising out of
`predict()`, never calling `set_exception` on the future the SDK awaits, and
resolving with a *freshly constructed* default event so the SDK's
identity-based hook dedupe still works -- cannot be tested until the LiveKit
adapter exists. It belongs in this file and is not yet here. What is here is
the core-level half: the guard cannot be made to throw, and a bad language
pack cannot take down a session.
"""

from __future__ import annotations

import time

import pytest

from baobab_turn.core.config import BaobabConfig
from baobab_turn.core.guard import guard
from baobab_turn.core.markers import MarkerIndex
from baobab_turn.core.types import GuardInput, LanguagePack, Marker
from baobab_turn.packs import registry


@pytest.fixture
def index() -> MarkerIndex:
    return MarkerIndex(
        LanguagePack(
            code="t",
            name="T",
            continuation_markers=(Marker("the thing be say", 1.0), Marker("so", 0.6)),
            yield_markers=(Marker("abi", 0.9),),
            ambiguous_markers=(Marker("o", 0.3),),
            tonal=True,
            tonal_extra_ms=150,
        )
    )


# --- the guard cannot be made to throw --------------------------------

HOSTILE_TAILS: dict[str, str | None] = {
    "none": None,
    "empty": "",
    "whitespace": "   ",
    "punctuation only": "!!!???",
    "control chars": "\x00\x01\x02",
    "newlines": "\n\t\r",
    "repeated marker x5000": "the thing be say" * 5000,
    "100k chars": "a" * 100_000,
    "emoji": "🙂🙃" * 500,
    "full-width": "ｔｈｅ　ｔｈｉｎｇ　ｂｅ　ｓａｙ",
    "zero-width space": "abi​abi",
    "dashes": "-" * 1000,
}


@pytest.mark.parametrize("label", sorted(HOSTILE_TAILS))
def test_hostile_transcripts_do_not_raise(index: MarkerIndex, label: str) -> None:
    tail = HOSTILE_TAILS[label]
    r = guard(GuardInput(p_inner=0.5, transcript_tail=tail, tail_age_ms=10), index=index)
    assert r.reasons
    assert r.p_adjusted is not None and 0.0 <= r.p_adjusted <= 1.0


@pytest.mark.parametrize(
    "p", [None, 0.0, 1.0, -0.0, 1e-300, 0.9999999999999999]
)
def test_edge_probabilities_do_not_raise(index: MarkerIndex, p: float | None) -> None:
    r = guard(GuardInput(p_inner=p, transcript_tail="abi", tail_age_ms=10), index=index)
    assert r.reasons
    if r.p_adjusted is not None:
        assert 0.0 <= r.p_adjusted <= 1.0


@pytest.mark.parametrize("p", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_probability_is_discarded_not_propagated(
    index: MarkerIndex, p: float
) -> None:
    """A NaN slips through every comparison silently -- `nan < 0` and
    `nan > 1` are both False, so clamping does not catch it and we would hand
    a NaN back to the framework. An inner detector producing NaN or infinity
    is malfunctioning, and we have no basis to correct a number that is not
    one, so we report having no opinion instead.
    """
    r = guard(GuardInput(p_inner=p, transcript_tail="abi", tail_age_ms=10), index=index)
    assert r.p_adjusted is None
    assert r.passthrough
    assert "p_inner_not_finite" in r.reasons


@pytest.mark.parametrize("p,expected", [(-3.0, 0.0), (5.0, 1.0), (1.0000001, 1.0)])
def test_out_of_range_but_finite_probability_is_clamped(
    index: MarkerIndex, p: float, expected: float
) -> None:
    """Out of range is wrong, not meaningless, so it is corrected rather than
    discarded. Clamping happens before the marker rules, which assume they
    are working with an actual probability."""
    r = guard(GuardInput(p_inner=p, transcript_tail="nothing", tail_age_ms=10), index=index)
    assert r.p_adjusted == expected


@pytest.mark.parametrize(
    "age", [None, 0.0, -100.0, 1e12, float("inf"), float("nan")]
)
def test_hostile_tail_ages_do_not_raise(index: MarkerIndex, age: float | None) -> None:
    r = guard(
        GuardInput(p_inner=0.5, transcript_tail="abi", tail_age_ms=age), index=index
    )
    assert r.reasons


@pytest.mark.parametrize(
    "pause", [None, 0.0, -500.0, 1e12, float("inf"), float("nan")]
)
def test_hostile_pacing_stats_do_not_raise(
    index: MarkerIndex, pause: float | None
) -> None:
    r = guard(
        GuardInput(p_inner=0.5, transcript_tail="abi", tail_age_ms=10, mean_pause_ms=pause),
        index=index,
    )
    assert r.reasons
    if r.delay_override_ms is not None:
        cfg = BaobabConfig()
        assert cfg.min_delay_ms <= r.delay_override_ms <= cfg.max_delay_ms


def test_delay_is_always_within_bounds_or_absent(index: MarkerIndex) -> None:
    """Every path that can set a delay must land inside [min, max]. A delay
    outside the framework's accepted range is a rejected turn."""
    cfg = BaobabConfig()
    for pause in (None, 0.0, 50.0, 900.0, 99_000.0):
        for switch in (True, False):
            r = guard(
                GuardInput(
                    p_inner=0.5,
                    transcript_tail="abi",
                    tail_age_ms=10,
                    mean_pause_ms=pause,
                    code_switch=switch,
                ),
                cfg,
                index,
            )
            if r.delay_override_ms is not None:
                assert cfg.min_delay_ms <= r.delay_override_ms <= cfg.max_delay_ms


def test_guard_works_with_all_defaults() -> None:
    """No config, no pack, no transcript: the degenerate call must still
    return something coherent rather than blowing up."""
    r = guard(GuardInput())
    assert r.reasons
    assert r.passthrough


# --- performance budget -----------------------------------------------


def test_guard_p95_within_budget() -> None:
    """SPEC 5.1 asks for a 15ms cap. The guard is synchronous with no await
    points, so an asyncio timeout could never interrupt it -- the budget has
    to be asserted here instead of enforced at runtime.

    Uses the real Pidgin pack and a long transcript, i.e. the realistic worst
    case rather than a toy one.
    """
    pack = registry.resolve("pcm")
    assert pack is not None, "pcm pack should be discoverable"
    idx = MarkerIndex(pack)
    tail = ("i wan tell you say " * 200) + "the thing be say"
    gi = GuardInput(
        p_inner=0.75,
        transcript_tail=tail,
        tail_age_ms=10,
        mean_pause_ms=600,
        code_switch=True,
    )
    cfg = BaobabConfig()

    guard(gi, cfg, idx)  # warm the code path

    samples = []
    for _ in range(500):
        start = time.perf_counter()
        guard(gi, cfg, idx)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    p95 = samples[int(0.95 * (len(samples) - 1))]
    assert p95 < 15.0, f"guard p95 {p95:.3f}ms exceeds the 15ms budget"


# --- a bad language pack must not take down a session -----------------

MALFORMED = {
    "not a mapping": "- just\n- a\n- list\n",
    "missing code": "name: No Code\n",
    "bad status": "code: x\nstatus: totally-made-up\n",
    "markers not a list": "code: x\ncontinuation_markers: nope\n",
    "marker without text": "code: x\nyield_markers:\n  - {weight: 0.5}\n",
    "weight out of range": "code: x\nyield_markers:\n  - {text: abi, weight: 9}\n",
    "thresholds not a mapping": "code: x\nthresholds: 5\n",
    "unparseable weight": "code: x\nyield_markers:\n  - {text: abi, weight: abc}\n",
}


@pytest.mark.parametrize("label", sorted(MALFORMED))
def test_malformed_pack_raises_a_clear_error(label: str, tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(MALFORMED[label], encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        registry.load_pack_file(path)


def test_malformed_pack_is_skipped_not_fatal(tmp_path, monkeypatch, caplog) -> None:
    """Someone's YAML typo must not crash a live call. The pack is skipped
    with a warning and every good pack still loads."""
    (tmp_path / "broken.yaml").write_text("code: x\nstatus: bogus\n", encoding="utf-8")
    monkeypatch.setattr(registry, "_COMMUNITY_DIR", tmp_path)
    registry.available_packs.cache_clear()
    try:
        packs = registry.available_packs()
        assert "pcm" in packs and "en-NG" in packs
        assert "x" not in packs
    finally:
        registry.available_packs.cache_clear()


def test_unparseable_yaml_is_skipped(tmp_path, monkeypatch) -> None:
    (tmp_path / "junk.yaml").write_text("code: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(registry, "_COMMUNITY_DIR", tmp_path)
    registry.available_packs.cache_clear()
    try:
        assert "pcm" in registry.available_packs()
    finally:
        registry.available_packs.cache_clear()


# --- language resolution ----------------------------------------------


@pytest.mark.parametrize(
    "code", [None, "", "   ", "und", "auto", "multi", "xx-YY", "!!", "en-NG-x-private"]
)
def test_unknown_language_resolves_to_nothing(code: str | None) -> None:
    """Returning None puts the guard on the null-pack path, where the inner
    decision passes through untouched."""
    assert registry.resolve(code) is None


def test_exact_code_wins() -> None:
    pack = registry.resolve("en-NG")
    assert pack is not None and pack.code == "en-NG"


def test_base_code_falls_back() -> None:
    """`pcm-NG` has no pack of its own, but `pcm` does."""
    pack = registry.resolve("pcm-NG")
    assert pack is not None and pack.code == "pcm"


def test_community_packs_are_opt_in() -> None:
    """SPEC 5.8.3: community packs are written by native speakers but not
    benchmarked, so they must never be enabled by default."""
    community = LanguagePack(code="sw", name="Swahili", status="community")
    packs = dict(registry.available_packs())
    packs["sw"] = community
    registry.available_packs.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(registry, "available_packs", lambda: packs)
            assert registry.resolve("sw") is None
            assert registry.resolve("sw", allow_community=True) is community
    finally:
        registry.available_packs.cache_clear()


def test_shipped_packs_declare_zero_labelled_turns() -> None:
    """Both shipped packs are untested first drafts. If this ever fails it
    should be because real data was labelled and the count was updated
    honestly -- not because someone wanted the packs to look validated."""
    for code in ("en-NG", "pcm"):
        pack = registry.resolve(code)
        assert pack is not None
        assert pack.status == "experimental"
        assert pack.labelled_turns == 0
        assert not pack.is_validated
