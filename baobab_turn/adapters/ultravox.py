"""Ultravox: what a language pack can and cannot do here.

## This is not a guard, and it cannot be

Every other adapter in this package wraps somebody else's detector and
corrects it per turn. That is impossible on hosted Ultravox Realtime, and
the reason is structural rather than a missing feature: the model decides
server-side when the user has finished, and nothing in the protocol lets a
client veto, delay or replace that decision.

Checked rather than assumed, in three places:

- The API exposes `vadSettings` with four tuning fields and no hook. See
  `docs.ultravox.ai/noise/understanding-vad`.
- Pipecat's own `UltravoxRealtimeLLMService` calls
  `_warn_if_realtime_service_emits_no_turn_frames(emits_turn_frames=False)`
  and says in its docstring that locally generated boundaries "may not match
  Ultravox's server-side turn decisions". Its `VADUserStoppedSpeakingFrame`
  handler exists only to report TTFB.
- The LiveKit plugin declares `turn_detection=True` and documents that
  disabling server-side turn detection is unsupported.

So `BaobabTurnAnalyzer` will not gate Ultravox even when Ultravox is running
inside a Pipecat pipeline. Saying otherwise would be the most damaging kind
of wrong: an integration that looks wired and silently does nothing.

## What a pack can still contribute

One number. `turnEndpointDelay` is how long Ultravox waits after speech
stops before deciding the turn is over, and its default is tuned for
English. For a tonal language that default is the exact failure this project
exists to describe -- a falling pitch read as a finished sentence.

`vad_settings()` turns a pack into that one number. It is a one-shot
configuration, not a per-turn decision: no transcript, no markers, no
pacing, no pitch. A caller who switches language mid-sentence gets nothing
here, because there is nowhere to put it.

    from baobab_turn.adapters.ultravox import vad_settings

    settings = vad_settings("yo")       # {"turnEndpointDelay": "0.544s"}
    # pass as vadSettings when creating the call

That is the whole integration, and it is worth being clear that it is small.
"""

from __future__ import annotations

from ..core.config import BaobabConfig
from ..core.types import LanguagePack
from ..packs.registry import resolve

__all__ = ["TURN_ENDPOINT_DELAY_QUANTUM_MS", "ULTRAVOX_DEFAULT_DELAY_MS", "vad_settings"]

#: Ultravox's own documented default for `turnEndpointDelay`, in ms.
#:
#: Deliberately not in `core/config.py`. That file holds hypotheses this
#: project tests and the benchmark ablates; this is a fact about somebody
#: else's product, and inventing our own default here would misreport what
#: an unconfigured Ultravox call actually does.
ULTRAVOX_DEFAULT_DELAY_MS = 384

#: The docs state only multiples of 32ms are meaningful, so a value is
#: rounded to one. Reporting 0.5s when the service will treat it as 0.512s
#: would make every number we print subtly untrue.
TURN_ENDPOINT_DELAY_QUANTUM_MS = 32

#: Past this the agent feels broken rather than patient. Ultravox imposes no
#: ceiling, so this one is ours.
MAX_TURN_ENDPOINT_DELAY_MS = 2_000


def _quantise(ms: float) -> int:
    """Round to the nearest meaningful step, never below Ultravox's default.

    Never below, because this function exists to buy patience. Returning
    less than the stock value would make a language pack actively worse than
    not using one.
    """
    steps = round(ms / TURN_ENDPOINT_DELAY_QUANTUM_MS)
    value = steps * TURN_ENDPOINT_DELAY_QUANTUM_MS
    return max(ULTRAVOX_DEFAULT_DELAY_MS, min(value, MAX_TURN_ENDPOINT_DELAY_MS))


def delay_ms_for(pack: LanguagePack | None, config: BaobabConfig | None = None) -> int:
    """How long Ultravox should wait, in ms, for a speaker of this language."""
    config = config or BaobabConfig()
    if pack is None:
        return ULTRAVOX_DEFAULT_DELAY_MS

    extra = 0
    if config.enable_tonal and pack.tonal:
        # The only pack field that means anything without a transcript. Every
        # other lever this package has -- markers, code-switch, pacing, pitch
        # -- needs per-turn evidence, and there is no per-turn hook here.
        extra = pack.tonal_extra_ms or config.tonal_language_extra_ms

    # A pack that has measured its own floor overrides the vendor default.
    base = max(ULTRAVOX_DEFAULT_DELAY_MS, pack.min_delay_ms or 0)
    ceiling = pack.max_delay_ms
    total = base + extra
    if ceiling is not None:
        total = min(total, ceiling)
    return _quantise(total)


def vad_settings(
    language: str | None,
    config: BaobabConfig | None = None,
    *,
    allow_community_packs: bool = False,
) -> dict[str, str]:
    """Build the `vadSettings` for an Ultravox call from a language pack.

    Returns a dict ready to pass as `vadSettings` at call creation. An
    unknown language returns Ultravox's own default rather than an empty
    dict, so the caller can always send the result unconditionally and see
    exactly what the call will do.

    Only `turnEndpointDelay` is set. The other three fields --
    `minimumTurnDuration`, `minimumInterruptionDuration` and
    `frameActivationThreshold` -- are left alone because nothing in a
    language pack has anything to say about them, and emitting a value we
    cannot justify would be guessing in a dict that looks authoritative.
    """
    pack = resolve(language, allow_community=allow_community_packs)
    ms = delay_ms_for(pack, config)
    return {"turnEndpointDelay": f"{ms / 1000:.3f}s"}
