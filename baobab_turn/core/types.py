"""The types crossing the core/adapter boundary.

Nothing here imports a framework, and nothing here does I/O. An adapter
translates its framework's world into `GuardInput`, and translates
`GuardResult` back.

The shapes are the way they are because the two target frameworks disagree:
LiveKit wants a float probability and applies a threshold itself; Pipecat
wants a COMPLETE/INCOMPLETE decision and hides its threshold. So the guard
produces both, and owns the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Marker", "LanguagePack", "GuardInput", "GuardResult"]


@dataclass(frozen=True, slots=True)
class Marker:
    """A discourse marker and how strongly we trust it.

    Weights are floats rather than booleans because these markers are not
    equally reliable: "abi?" at the tail is a near-certain hand-off, while a
    trailing "o" is a hint.
    """

    text: str
    weight: float
    note: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"marker weight out of range: {self.text}={self.weight}")


@dataclass(frozen=True, slots=True)
class LanguagePack:
    """A language's turn-taking habits, loaded from YAML. Data, never code."""

    code: str
    name: str
    status: str = "experimental"  # validated | community | experimental
    tonal: bool = False
    tonal_extra_ms: int = 0
    maintainer: str = ""
    region_hint: tuple[str, ...] = ()
    commonly_switches_with: tuple[str, ...] = ()

    continuation_markers: tuple[Marker, ...] = ()
    yield_markers: tuple[Marker, ...] = ()
    ambiguous_markers: tuple[Marker, ...] = ()
    fillers: tuple[str, ...] = ()

    unlikely_threshold: float | None = None
    min_delay_ms: int | None = None
    max_delay_ms: int | None = None

    # Honesty fields. `labelled_turns == 0` means untested, and is displayed
    # as 0 wherever the pack is listed. Never hide it.
    labelled_turns: int = 0
    dataset_note: str = ""

    @property
    def is_validated(self) -> bool:
        return self.status == "validated" and self.labelled_turns > 0


#: The pack used when no language matches. It has no markers, so the guard
#: applies no lexical rules and the inner detector's decision passes through
#: untouched. An unknown language must degrade to stock behaviour, never to
#: something worse than stock.
NULL_PACK = LanguagePack(code="und", name="Unknown", status="validated")


@dataclass(frozen=True, slots=True)
class GuardInput:
    """Everything the guard is allowed to look at.

    `p_inner` is None when the host framework gives us a decision rather than
    a probability, or when its own detector failed. `transcript_tail` is None
    or empty when no transcript is available at all -- which is normal, not
    exceptional: on LiveKit the detector is fed audio, and the transcript
    arrives through a separate channel that can lag or be suppressed.
    """

    p_inner: float | None = None
    transcript_tail: str | None = None
    is_final: bool = False

    #: How old the transcript tail is. The tail can be silently stale -- there
    #: is no alignment guarantee between the audio a probability was computed
    #: over and the text we have. Past `config.max_tail_age_ms` the lexical
    #: layer abstains rather than guessing.
    tail_age_ms: float | None = None

    language: str | None = None
    code_switch: bool = False

    #: Observed pause rhythm for this caller, in ms. Lets a slow, deliberate
    #: speaker earn more patience than a fast one.
    mean_pause_ms: float | None = None
    stdev_pause_ms: float | None = None

    #: True while the agent itself is speaking. The transcript feed is
    #: suppressed then, so markers cannot be trusted during barge-in.
    agent_speaking: bool = False


@dataclass(frozen=True, slots=True)
class GuardResult:
    """The guard's opinion.

    `decision` is always populated -- Pipecat cannot accept a float.
    `p_adjusted` is populated when we had a probability to adjust.
    `delay_override_ms` is an *intent*: Pipecat can honour it to the
    millisecond, LiveKit can only quantise it, so each adapter spends it in
    its own currency.

    `reasons` is never empty. If no rule fired it says so. A decision we
    cannot explain is a decision we cannot tune.
    """

    decision: bool
    p_adjusted: float | None = None
    delay_override_ms: int | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    #: True when we deliberately did not touch the inner decision.
    passthrough: bool = False
