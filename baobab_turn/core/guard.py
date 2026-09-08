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
    p = gi.p_inner
    delay: int | None = None

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

        # Order matters. A continuation marker is a stronger claim than a
        # yield marker -- "so..." means the speaker is mid-thought, and we
        # would rather wait wrongly than interrupt wrongly.
        cont = index.continuation(tokens)
        if cont is not None:
            phrase, weight = cont
            p *= 1.0 - (config.continuation_veto_strength * weight)
            reasons.append(f"continuation_veto:{phrase}")
        else:
            yielded = index.yields(tokens)
            if yielded is not None:
                phrase, weight = yielded
                p += (1.0 - p) * config.yield_boost_strength * weight
                reasons.append(f"yield_boost:{phrase}")
            else:
                amb = index.ambiguous(tokens)
                # Ambiguous markers only count as the final token of a
                # finalised transcript. Mid-utterance they mean nothing, and
                # treating one as a hand-off is how you cut someone off.
                if amb is not None and gi.is_final:
                    phrase, weight = amb
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

    if config.enable_pacing and gi.mean_pause_ms is not None:
        # A caller who habitually pauses long mid-sentence has earned more
        # patience than one who does not. Rolling mean only; no ML.
        target = gi.mean_pause_ms + (gi.stdev_pause_ms or 0.0)
        if target > (delay or config.min_delay_ms):
            delay = int(target)
            reasons.append("pacing")

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
