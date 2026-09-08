"""Golden-file test for the metric definitions.

SPEC 10: "Benchmark runner must have a golden-file test so metric definitions
can't drift silently." If a change here is intentional, update
`tests/golden/metrics.json` in the same commit and say why in the message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.metrics import Percentiles, ScoredTurn, compute, percentiles

GOLDEN = Path(__file__).parent / "golden" / "metrics.json"


def _fixture() -> list[ScoredTurn]:
    """A small deliberately awkward set: two premature cuts, one exactly on
    the boundary, one very late tail, and one abstention."""
    return [
        # Cut the caller off by 400ms.
        ScoredTurn("c1", 0, true_end_ms=3000, commit_ms=2600, language="pcm",
                   reasons=("continuation_veto",)),
        # Cut off badly -- 1.2s early, mid code-switch.
        ScoredTurn("c1", 1, true_end_ms=8000, commit_ms=6800, language="pcm",
                   code_switch=True, reasons=("codeswitch_hold",)),
        # Exactly on the boundary. Must NOT count as premature.
        ScoredTurn("c2", 0, true_end_ms=1500, commit_ms=1500, language="en-NG"),
        # Patient, 300ms late.
        ScoredTurn("c2", 1, true_end_ms=4000, commit_ms=4300, language="en-NG",
                   reasons=("yield_boost",)),
        # The tail that ruins the p95: 2.4s late.
        ScoredTurn("c3", 0, true_end_ms=2000, commit_ms=4400, language="yo",
                   reasons=("tonal_extra", "continuation_veto")),
        # Lexical layer had nothing to work with.
        ScoredTurn("c3", 1, true_end_ms=5000, commit_ms=5100, language="yo",
                   reasons=("tail_stale",)),
    ]


def test_matches_golden() -> None:
    got = compute(_fixture()).as_dict()
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert got == expected, (
        "Metric definitions changed. If deliberate, update "
        f"{GOLDEN.name} in the same commit and explain why."
    )


def test_boundary_commit_is_not_premature() -> None:
    """Committing at exactly the labelled end is on time, not early. This
    boundary decides whether a whole class of turns is counted against us."""
    on_time = ScoredTurn("x", 0, true_end_ms=1000, commit_ms=1000)
    assert not on_time.is_premature
    assert compute([on_time]).premature_commit_rate == 0.0


def test_premature_turns_excluded_from_patience_tail() -> None:
    """A premature commit has a negative error. Letting it into the delay
    percentiles would make cutting people off look like low latency."""
    turns = [
        ScoredTurn("x", 0, true_end_ms=1000, commit_ms=500),   # -500
        ScoredTurn("x", 1, true_end_ms=1000, commit_ms=1400),  # +400
    ]
    m = compute(turns)
    assert m.premature_commit_rate == 0.5
    assert m.late_commit_delay_ms.p50 == 400.0


def test_fcr_is_none_without_live_timing() -> None:
    """Offline runs must not report an end-to-end number they cannot measure."""
    m = compute(_fixture())
    assert m.fcr is None
    assert m.late_response_delay_ms is None


def test_fcr_computed_when_live_timing_present() -> None:
    turns = [
        ScoredTurn("x", 0, true_end_ms=1000, commit_ms=1100,
                   agent_speech_start_ms=900),   # agent spoke early
        ScoredTurn("x", 1, true_end_ms=1000, commit_ms=1100,
                   agent_speech_start_ms=1600),  # 600ms after
    ]
    m = compute(turns)
    assert m.fcr == 0.5
    assert m.late_response_delay_ms is not None
    assert m.late_response_delay_ms.p50 == 600.0


def test_abstention_rate_counts_stale_and_absent() -> None:
    turns = [
        ScoredTurn("x", 0, true_end_ms=1, commit_ms=1, reasons=("tail_stale",)),
        ScoredTurn("x", 1, true_end_ms=1, commit_ms=1, reasons=("tail_absent",)),
        ScoredTurn("x", 2, true_end_ms=1, commit_ms=1, reasons=("yield_boost",)),
        ScoredTurn("x", 3, true_end_ms=1, commit_ms=1),
    ]
    assert compute(turns).tail_abstention_rate == 0.5


def test_reasons_counted_once_per_turn() -> None:
    """A rule firing twice in one turn is still one turn's worth of evidence."""
    turns = [ScoredTurn("x", 0, true_end_ms=1, commit_ms=1,
                        reasons=("continuation_veto", "continuation_veto"))]
    assert compute(turns).reason_counts == {"continuation_veto": 1}


def test_empty_input_is_safe() -> None:
    m = compute([])
    assert m.n_turns == 0
    assert m.premature_commit_rate == 0.0
    assert m.fcr is None


@pytest.mark.parametrize(
    "values,expected",
    [
        ([], Percentiles(0.0, 0.0, 0.0)),
        ([42.0], Percentiles(42.0, 42.0, 42.0)),
        ([0.0, 100.0], Percentiles(50.0, 90.0, 95.0)),
        ([1.0, 2.0, 3.0, 4.0], Percentiles(2.5, 3.7, 3.85)),
    ],
)
def test_percentile_interpolation(values: list[float], expected: Percentiles) -> None:
    """Pins linear interpolation between closest ranks, so the definition
    cannot shift if we ever swap the implementation."""
    got = percentiles(values)
    assert got.p50 == pytest.approx(expected.p50)
    assert got.p90 == pytest.approx(expected.p90)
    assert got.p95 == pytest.approx(expected.p95)
