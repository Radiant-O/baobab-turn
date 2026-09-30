"""Learning one caller's rhythm during a call.

Unlike the marker layer, this applies to **every** turn rather than only to
the ones that happen to end on a discourse marker. The benchmark showed the
lexicon fires on about 4% of decision points, which is close to a structural
ceiling for it -- so the layers that speak on every turn are where a headline
number can actually move.

The idea is small: some people pause a lot mid-sentence and some do not, and
a detector tuned to an average speaker cuts off the deliberate ones. Track
what *this* caller does and give them the patience they have earned.

No ML, and deliberately so. A rolling mean and standard deviation over a
bounded window is enough to separate a speaker who pauses 200 ms from one who
pauses 900 ms, and it costs microseconds with no training data, no model file
and nothing to go stale.
"""

from __future__ import annotations

import math
from collections import deque

__all__ = ["PacingTracker"]

#: Pauses outside this range are not speech rhythm. Below the floor is VAD
#: jitter inside a single word; above the ceiling the caller has stopped to
#: think, look something up, or talk to somebody else, and folding that into
#: a running mean would teach us to wait several seconds for everyone.
_MIN_PLAUSIBLE_MS = 40.0
_MAX_PLAUSIBLE_MS = 5_000.0


class PacingTracker:
    """Rolling pause statistics for a single caller.

    Bounded: only the most recent `window` pauses count, so the tracker keeps
    following a caller who speeds up or slows down mid-call rather than being
    anchored to how they started.
    """

    __slots__ = ("_pauses", "min_samples", "window")

    def __init__(self, window: int = 20, min_samples: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        self.window = window
        #: Below this many observations the estimate is noise. Reporting a
        #: confident mean from one pause would hand the guard a number that
        #: looks like evidence and is not.
        self.min_samples = min_samples
        self._pauses: deque[float] = deque(maxlen=window)

    def observe(self, pause_ms: float) -> bool:
        """Record one inter-utterance pause. Returns whether it was kept.

        Implausible and non-finite values are dropped rather than clamped: a
        clamped outlier still drags the mean toward the boundary, and the
        point of the floor and ceiling is that those events are not rhythm at
        all.
        """
        if not math.isfinite(pause_ms):
            return False
        if not (_MIN_PLAUSIBLE_MS <= pause_ms <= _MAX_PLAUSIBLE_MS):
            return False
        self._pauses.append(float(pause_ms))
        return True

    def reset(self) -> None:
        """Forget this caller. Call it when the session ends, never between
        turns -- rhythm is a property of the speaker, not of one utterance."""
        self._pauses.clear()

    @property
    def n(self) -> int:
        return len(self._pauses)

    @property
    def ready(self) -> bool:
        return self.n >= self.min_samples

    @property
    def mean_ms(self) -> float | None:
        """None until enough pauses have been seen to mean anything."""
        if not self.ready:
            return None
        return sum(self._pauses) / len(self._pauses)

    @property
    def stdev_ms(self) -> float | None:
        """Sample standard deviation, or None when it is not yet meaningful.

        Needs at least two observations regardless of `min_samples`, since
        the sample formula divides by n-1.
        """
        if not self.ready or self.n < 2:
            return None
        mean = sum(self._pauses) / self.n
        variance = sum((p - mean) ** 2 for p in self._pauses) / (self.n - 1)
        return math.sqrt(variance)

    def as_guard_inputs(self) -> tuple[float | None, float | None]:
        """The pair `GuardInput` wants: `(mean_pause_ms, stdev_pause_ms)`.

        Both None until the tracker is ready, which switches the guard's
        pacing rule off rather than feeding it a guess.
        """
        return self.mean_ms, self.stdev_ms
