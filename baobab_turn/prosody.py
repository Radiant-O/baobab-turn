"""Pitch tracking, so the tonal rule has something to react to.

## Why this exists

The tonal hypothesis is the load-bearing claim for this project being
pan-African rather than Nigerian: a detector that reads falling intonation as
"sentence finished" will cut off speakers of Yoruba, Igbo, Zulu, Shona, Twi
and dozens of other languages, because in those languages a falling pitch can
simply be a low tone in the middle of a word.

Until now the packs expressed that as `tonal: true` plus a fixed
`tonal_extra_ms` -- wait a bit longer for everyone speaking a tonal language,
all the time, whatever their pitch happened to be doing. That is a knob with
no sensor attached. It cannot distinguish the fall that means "I have
finished" from the fall that means "this syllable is low", which is the only
distinction that matters.

This module is the sensor. It estimates fundamental frequency from the audio
frames an adapter already receives and reports the shape of the contour, so
the guard can react to *this* fall rather than to the language in general.

## Why it is not in core/

`core/` is a rules engine and a CI check forbids it importing numpy. Pitch
estimation is signal processing, and doing it in pure Python would cost tens
of milliseconds per frame against a 15 ms budget. So it lives out here behind
an optional extra:

    pip install "baobab-turn[prosody]"

The split is the same one `pacing.py` uses: something outside the guard does
the measuring, and the guard consumes plain numbers.

## What it is not

Not a pitch tracker you would use for music or for tone transcription. It is
deliberately coarse, because the guard only ever asks "is this contour rising,
falling or level". Getting that right at the ends of utterances matters;
being accurate to the cent does not.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Literal

__all__ = ["Contour", "PitchTracker", "estimate_f0"]

Contour = Literal["rising", "falling", "level", "unknown"]

#: Human speech f0. Below is almost always creak or noise; above is almost
#: always a harmonic mistaken for the fundamental.
_MIN_F0_HZ = 70.0
_MAX_F0_HZ = 400.0

#: How strongly the waveform must repeat before we believe the frame is
#: voiced. Unvoiced frames carry no pitch and must not be interpolated
#: through -- a fricative is not a low note.
_VOICING_THRESHOLD = 0.3


def estimate_f0(samples, sample_rate: int) -> float | None:
    """Estimate fundamental frequency from one window of PCM.

    Autocorrelation rather than anything cleverer: it is robust on telephone
    audio, needs no model, and the resolution it gives is far finer than the
    three-way answer we need. Returns None for unvoiced or too-short input.

    `samples` is any sequence of numbers -- int16 PCM or floats, scaling does
    not matter because the result is normalised.
    """
    import numpy as np

    x = np.asarray(samples, dtype=np.float64)
    if x.size < 2:
        return None

    # Remove DC. A constant offset correlates perfectly with itself at every
    # lag and would swamp the real periodicity.
    x = x - x.mean()
    energy = float(np.dot(x, x))
    if energy <= 0.0:
        return None  # digital silence

    min_lag = int(sample_rate / _MAX_F0_HZ)
    max_lag = int(sample_rate / _MIN_F0_HZ)
    if max_lag >= x.size or min_lag >= max_lag:
        return None  # window too short to contain a full period

    corr = np.correlate(x, x, mode="full")[x.size - 1:]
    window = corr[min_lag : max_lag + 1]
    if window.size == 0:
        return None

    peak_index = int(np.argmax(window))
    peak = float(window[peak_index])
    # corr[0] is the energy, so this is the normalised autocorrelation: how
    # much the waveform resembles itself one period later.
    if peak / energy < _VOICING_THRESHOLD:
        return None

    return sample_rate / float(min_lag + peak_index)


class PitchTracker:
    """Keeps a short history of f0 and reports the shape of the contour.

    Bounded, so it describes the end of what is being said rather than the
    whole utterance -- the turn-final contour is the part that carries
    turn-taking meaning.
    """

    __slots__ = ("_f0", "min_points", "min_semitones", "sample_rate")

    def __init__(
        self,
        sample_rate: int = 16_000,
        history: int = 12,
        min_points: int = 4,
        min_semitones: float = 1.5,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if history < 2:
            raise ValueError("history must be at least 2")
        self.sample_rate = sample_rate
        self._f0: deque[float] = deque(maxlen=history)
        #: Below this many voiced frames any slope is noise.
        self.min_points = min_points
        #: How far the contour must move to count as rising or falling.
        #: In semitones, not hertz, because pitch is perceived
        #: logarithmically -- 20 Hz is a large move for a deep voice and a
        #: small one for a high voice.
        self.min_semitones = min_semitones

    def push_samples(self, samples, sample_rate: int | None = None) -> float | None:
        """Feed one window of PCM. Returns the f0 estimate, if voiced."""
        f0 = estimate_f0(samples, sample_rate or self.sample_rate)
        if f0 is not None:
            self._f0.append(f0)
        return f0

    def push_f0(self, f0: float | None) -> None:
        """Feed an f0 estimate from somewhere else."""
        if f0 is not None and math.isfinite(f0) and _MIN_F0_HZ <= f0 <= _MAX_F0_HZ:
            self._f0.append(float(f0))

    def reset(self) -> None:
        """Forget the contour. Call at a turn boundary: the previous turn's
        shape says nothing about this one."""
        self._f0.clear()

    @property
    def n(self) -> int:
        return len(self._f0)

    def semitone_span(self) -> float | None:
        """Signed movement from the start of the history to the end.

        Positive is rising. Uses the median of the first and last thirds
        rather than single endpoints, so one octave-error frame cannot invent
        a contour.
        """
        if self.n < self.min_points:
            return None
        values = list(self._f0)
        third = max(1, len(values) // 3)
        first = sorted(values[:third])[third // 2]
        last = sorted(values[-third:])[third // 2]
        if first <= 0 or last <= 0:
            return None
        return 12.0 * math.log2(last / first)

    def contour(self) -> Contour:
        span = self.semitone_span()
        if span is None:
            return "unknown"
        if span >= self.min_semitones:
            return "rising"
        if span <= -self.min_semitones:
            return "falling"
        return "level"
