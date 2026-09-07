"""Every tunable number in the system.

Nothing is hardcoded elsewhere. Each field is a hypothesis under test, not a
commitment -- a flag that does not change behaviour for the better gets
deleted rather than defended.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BaobabConfig"]


@dataclass(frozen=True, slots=True)
class BaobabConfig:
    # --- lexical layer -------------------------------------------------
    #: How many tokens back from the tail to scan. A "so" fifteen words ago
    #: says nothing about whether this sentence is finished.
    marker_window: int = 6

    #: How much a continuation marker suppresses the end-of-turn probability.
    continuation_veto_strength: float = 0.6

    #: How much a yield marker raises it.
    yield_boost_strength: float = 0.3

    #: Beyond this age the transcript tail is not trusted and the lexical
    #: layer abstains. There is no alignment guarantee between the audio the
    #: probability came from and the text we hold, so a stale tail is worse
    #: than no tail: it looks authoritative and is not.
    max_tail_age_ms: int = 800

    # --- patience ------------------------------------------------------
    codeswitch_hold_ms: int = 200
    tonal_language_extra_ms: int = 150
    min_delay_ms: int = 300
    max_delay_ms: int = 2500

    #: Probability at or above which we call the turn over, when the host
    #: framework wants a decision instead of a float. A pack may override it.
    decision_threshold: float = 0.5

    # --- feature switches ---------------------------------------------
    # Each exists so its contribution can be measured separately.
    enable_markers: bool = True
    enable_codeswitch: bool = True
    enable_pacing: bool = True
    enable_tonal: bool = True

    #: Community packs are written by native speakers but not benchmarked, so
    #: they are opt-in. Never on by default.
    allow_community_packs: bool = False

    def __post_init__(self) -> None:
        if self.marker_window < 1:
            raise ValueError("marker_window must be at least 1")
        if not 0.0 <= self.continuation_veto_strength <= 1.0:
            raise ValueError("continuation_veto_strength must be in [0, 1]")
        if not 0.0 <= self.yield_boost_strength <= 1.0:
            raise ValueError("yield_boost_strength must be in [0, 1]")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")
        if self.min_delay_ms > self.max_delay_ms:
            raise ValueError("min_delay_ms cannot exceed max_delay_ms")
