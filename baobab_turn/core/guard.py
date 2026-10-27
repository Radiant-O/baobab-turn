"""The rule engine. A pure function -- no I/O, no clock, no framework.

Called on every partial-transcript update and every end-of-turn prediction,
so it is synchronous and cheap. It is deliberately *not* async: a sync
function has no await points, so an asyncio timeout could never interrupt it
anyway. Safety comes from the caller wrapping it in try/except and from the
p95 budget being asserted in tests.

The one invariant that matters more than any rule below: **when in doubt, do
nothing.** A turn detector that breaks a call is far worse than one that is
merely mediocre, so every uncertain path here passes the inner decision
through untouched.
"""

from __future__ import annotations

import math

from .config import BaobabConfig
from .markers import MarkerIndex
from .types import NULL_PACK, GuardInput, GuardResult, LanguagePack

__all__ = ["guard"]


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def guard(
    gi: GuardInput,
    config: BaobabConfig | None = None,
    index: MarkerIndex | None = None,
) -> GuardResult:
    """Adjust an end-of-turn decision using a language's turn-taking habits.

    `index` is None when no language pack matched. That is the null-pack case
    and it must behave exactly like stock: no lexical rules, nothing touched.
    """
    config = config or BaobabConfig()
    pack: LanguagePack = index.pack if index is not None else NULL_PACK

    reasons: list[str] = []
    delay: int | None = None

    # Sanitise the inner probability before any arithmetic touches it.
    #
    # NaN is the dangerous one: every comparison against it is False, so it
    # slips through clamping untouched and we would hand a NaN back to the
    # framework. An inner detector that produces NaN or infinity is
    # malfunctioning, and we have no basis to correct a number that isn't one
    # -- so we treat it as having no opinion at all.
    #
    # Out-of-range but finite values are merely wrong, not meaningless, so
    # they are clamped rather than discarded. Clamping happens here, before
    # the marker rules, because those rules assume p is a probability: a
    # yield boost applied to p=5.0 would move it the wrong way.
    p = gi.p_inner
    if p is not None:
        if not math.isfinite(p):
            reasons.append("p_inner_not_finite")
            p = None
        else:
            p = _clamp(float(p), 0.0, 1.0)

    # --- the lexical layer, and when it must keep quiet ----------------
    #
    # Three independent ways the transcript tail is untrustworthy, and all
    # three are normal rather than exceptional. Each one abstains loudly so
    # that the numbers show how often we had nothing to say -- that fraction
    # caps how much this whole layer can ever be worth.
    tail_usable = True
    if not config.enable_markers or index is None or not index:
        tail_usable = False
    elif gi.agent_speaking:
        # The transcript feed is suppressed while the agent talks, so any tail
        # we hold is a leftover from before. This is exactly the barge-in
        # case, where a confident wrong answer does the most damage.
        tail_usable = False
        reasons.append("tail_absent")
    elif not gi.transcript_tail:
        tail_usable = False
        reasons.append("tail_absent")
    elif gi.tail_age_ms is not None and gi.tail_age_ms > config.max_tail_age_ms:
        # Nothing correlates the tail to the audio the probability came from.
        # A stale tail looks authoritative and is not.
        tail_usable = False
        reasons.append("tail_stale")

    if tail_usable and p is not None:
        assert index is not None  # narrowed by tail_usable
        tokens = index.window(gi.transcript_tail or "", config.marker_window)

        # The longest match wins, across all three marker kinds, and only
        # then does kind break a tie.
        #
        # Getting this backwards is subtle and wrong. "no be so" is an
        # unambiguous hand-off, but its last word "so" is a continuation
        # marker; checking continuations first would veto the turn on a
        # phrase that means the exact opposite. Length is a proxy for
        # specificity, so the more specific marker has to win regardless of
        # which list it came from.
        #
        # At equal length, continuation outranks yield: waiting wrongly costs
        # a little latency, interrupting wrongly costs the caller's goodwill.
        #
        # Ambiguous markers ("sha", trailing "o") only count on a finalised
        # transcript. Mid-utterance they mean nothing, and treating one as a
        # hand-off is how a detector cuts someone off.
        candidates: list[tuple[int, int, str, tuple[str, float]]] = []
        for rank, (kind, match) in enumerate(
            (
                ("continuation", index.continuation(tokens)),
                ("yield", index.yields(tokens)),
                ("ambiguous", index.ambiguous(tokens) if gi.is_final else None),
            )
        ):
            if match is not None:
                candidates.append((len(match[0].split()), rank, kind, match))

        if candidates:
            candidates.sort(key=lambda c: (-c[0], c[1]))
            _, _, kind, (phrase, weight) = candidates[0]
            if kind == "continuation":
                p *= 1.0 - (config.continuation_veto_strength * weight)
                reasons.append(f"continuation_veto:{phrase}")
            elif kind == "yield":
                p += (1.0 - p) * config.yield_boost_strength * weight
                reasons.append(f"yield_boost:{phrase}")
            else:
                p += (1.0 - p) * config.yield_boost_strength * weight * 0.5
                reasons.append(f"ambiguous_hint:{phrase}")

    # --- patience ------------------------------------------------------
    # These are delay *intents*. LiveKit cannot honour a millisecond value and
    # will quantise; Pipecat can sleep it exactly. The adapter decides.
    if config.enable_codeswitch and gi.code_switch:
        delay = (delay or config.min_delay_ms) + config.codeswitch_hold_ms
        reasons.append("codeswitch_hold")

    if config.enable_tonal and pack.tonal:
        extra = pack.tonal_extra_ms or config.tonal_language_extra_ms
        delay = (delay or config.min_delay_ms) + extra
        reasons.append("tonal_extra")

        # A falling contour in a tonal language is ambiguous in a way it is
        # not in English: it may be a low tone inside a word rather than the
        # end of a sentence. The inner detector reads pitch and cannot tell
        # the difference, so when it sounds confident on a fall, discount it.
        #
        # Only on a measured fall. Without a sensor this stays silent rather
        # than damping every tonal utterance, which is what the fixed
        # `tonal_extra_ms` alone amounts to.
        if p is not None and gi.pitch_contour == "falling":
            p *= 1.0 - config.tonal_falling_damp
            reasons.append("tonal_falling_damp")

    if config.enable_pacing and gi.mean_pause_ms is not None:
        # A caller who habitually pauses long mid-sentence has earned more
        # patience than one who does not. Rolling mean only; no ML.
        #
        # Both stats are sanitised first. A non-finite value here is not
        # hypothetical: a variance computed from a single observation, or a
        # division by a zero sample count, produces inf or nan, and `int(inf)`
        # raises OverflowError -- which on the LiveKit streaming path would
        # escape into the SDK's unguarded call site and hang the turn.
        mean = gi.mean_pause_ms
        stdev = gi.stdev_pause_ms or 0.0
        if math.isfinite(mean) and math.isfinite(stdev):
            # Clamp before int() so the conversion can never overflow.
            target = _clamp(
                mean + stdev, float(config.min_delay_ms), float(config.max_delay_ms)
            )
            if target > (delay or config.min_delay_ms):
                delay = int(target)
                reasons.append("pacing")
        else:
            reasons.append("pacing_stats_not_finite")

    # --- clamp and decide ----------------------------------------------
    lo = pack.min_delay_ms if pack.min_delay_ms is not None else config.min_delay_ms
    hi = pack.max_delay_ms if pack.max_delay_ms is not None else config.max_delay_ms
    if delay is not None:
        delay = int(_clamp(float(delay), float(lo), float(hi)))

    threshold = (
        pack.unlikely_threshold
        if pack.unlikely_threshold is not None
        else config.decision_threshold
    )

    if p is None:
        # No probability to work with -- the host handed us a decision, or its
        # detector failed. We have no basis to overrule it.
        return GuardResult(
            decision=True,
            p_adjusted=None,
            delay_override_ms=delay,
            reasons=tuple(reasons) or ("no_probability",),
            passthrough=True,
        )

    p = _clamp(p, 0.0, 1.0)
    touched = p != gi.p_inner or delay is not None

    return GuardResult(
        decision=p >= threshold,
        p_adjusted=p,
        delay_override_ms=delay,
        reasons=tuple(reasons) or ("no_rules_fired",),
        passthrough=not touched,
    )
