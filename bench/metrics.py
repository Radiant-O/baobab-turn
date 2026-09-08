"""Turn-detection metrics.

These definitions are the product. Every published number comes from here, so
the definitions must not drift silently -- `tests/test_metrics.py` pins them
with a golden file.

Two things this module is deliberate about:

**`premature_commit_rate` is not FCR.** FCR as defined in SPEC 7.1 measures
when the *agent began speaking* before the caller finished. That needs an LLM
and a TTS in the loop. Offline, all we can observe is when the *detector
committed*, which is earlier and cheaper. The offline proxy is legitimate;
calling it FCR is not. `fcr` is computed only when a turn carries
`agent_speech_start_ms`.

**Tails, not medians.** A good p50 with a bad p95 is a bad agent -- callers
remember the turn that hung for three seconds, not the fifty that were fine.
Every latency figure is reported at p50/p90/p95.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "ScoredTurn",
    "Percentiles",
    "Metrics",
    "percentiles",
    "compute",
]


@dataclass(frozen=True, slots=True)
class ScoredTurn:
    """One turn's outcome, in milliseconds relative to the clip start.

    `true_end_ms` is the human-labelled moment the caller genuinely finished.
    `commit_ms` is when the detector decided the turn was over. Both are
    required; everything else is optional so that offline replay and live
    calls can share this type.
    """

    clip_id: str
    turn_index: int
    true_end_ms: float
    commit_ms: float

    # Live mode only. Without this, `fcr` is None -- see the module docstring.
    agent_speech_start_ms: float | None = None

    # Provenance, so a report can slice by language or by whether the clip
    # actually contained a code-switch.
    language: str | None = None
    code_switch: bool = False

    # Which guard rules fired, and whether the lexical layer had to abstain
    # because the transcript tail was stale or missing. The abstention rate is
    # a headline number: it caps how much the marker hypothesis can be worth.
    reasons: tuple[str, ...] = ()

    @property
    def is_premature(self) -> bool:
        return self.commit_ms < self.true_end_ms

    @property
    def commit_error_ms(self) -> float:
        """Signed: negative means the detector cut the caller off."""
        return self.commit_ms - self.true_end_ms


@dataclass(frozen=True, slots=True)
class Percentiles:
    p50: float
    p90: float
    p95: float

    def as_dict(self) -> dict[str, float]:
        return {"p50": self.p50, "p90": self.p90, "p95": self.p95}


@dataclass(frozen=True, slots=True)
class Metrics:
    n_turns: int

    # Primary. Fraction of turns the detector cut short. Lower is better.
    premature_commit_rate: float

    # The cost of patience: how long after the true end the detector waited,
    # over the turns it did NOT cut short. Reported as a tail.
    late_commit_delay_ms: Percentiles

    # End-to-end, live mode only. None offline -- see the module docstring.
    fcr: float | None
    late_response_delay_ms: Percentiles | None

    # How often the lexical layer had nothing trustworthy to work with.
    tail_abstention_rate: float

    # Rule name -> how many turns it fired on. Makes the ablation readable and
    # catches rules that never fire at all.
    reason_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_turns": self.n_turns,
            "premature_commit_rate": self.premature_commit_rate,
            "late_commit_delay_ms": self.late_commit_delay_ms.as_dict(),
            "fcr": self.fcr,
            "late_response_delay_ms": (
                self.late_response_delay_ms.as_dict()
                if self.late_response_delay_ms is not None
                else None
            ),
            "tail_abstention_rate": self.tail_abstention_rate,
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


# Reasons that mean "the lexical layer declined to have an opinion". Kept here
# rather than in core/ so bench has no dependency on the guard.
ABSTENTION_REASONS = frozenset({"tail_stale", "tail_absent"})


def percentiles(values: Sequence[float]) -> Percentiles:
    """p50/p90/p95 by linear interpolation between closest ranks.

    Spelled out rather than delegated so the definition cannot change under us
    when a dependency updates. Matches numpy's default `linear` method.
    Returns zeros for an empty input -- callers report `n_turns` alongside, so
    an empty bucket is visible without needing a sentinel.
    """
    if not values:
        return Percentiles(0.0, 0.0, 0.0)

    ordered = sorted(values)

    def q(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = p * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

    return Percentiles(q(0.50), q(0.90), q(0.95))


def compute(turns: Iterable[ScoredTurn]) -> Metrics:
    """Aggregate scored turns into the numbers that go in the report."""
    turns = list(turns)
    n = len(turns)
    if n == 0:
        return Metrics(
            n_turns=0,
            premature_commit_rate=0.0,
            late_commit_delay_ms=percentiles([]),
            fcr=None,
            late_response_delay_ms=None,
            tail_abstention_rate=0.0,
            reason_counts={},
        )

    premature = [t for t in turns if t.is_premature]

    # Patience is only meaningful where we did not cut the caller off; mixing
    # negative errors in would let a premature commit flatter the percentile.
    late_delays = [t.commit_error_ms for t in turns if not t.is_premature]

    live = [t for t in turns if t.agent_speech_start_ms is not None]
    if live:
        fcr = sum(
            1 for t in live if t.agent_speech_start_ms < t.true_end_ms  # type: ignore[operator]
        ) / len(live)
        lrd = percentiles(
            [
                t.agent_speech_start_ms - t.true_end_ms  # type: ignore[operator]
                for t in live
                if t.agent_speech_start_ms >= t.true_end_ms  # type: ignore[operator]
            ]
        )
    else:
        fcr = None
        lrd = None

    reason_counts: dict[str, int] = {}
    for t in turns:
        for r in set(t.reasons):
            reason_counts[r] = reason_counts.get(r, 0) + 1

    abstained = sum(1 for t in turns if ABSTENTION_REASONS & set(t.reasons))

    return Metrics(
        n_turns=n,
        premature_commit_rate=len(premature) / n,
        late_commit_delay_ms=percentiles(late_delays),
        fcr=fcr,
        late_response_delay_ms=lrd,
        tail_abstention_rate=abstained / n,
        reason_counts=reason_counts,
    )
