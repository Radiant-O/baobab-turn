"""baobab-turn -- turn detection for African speech.

The engine is pan-African by architecture. Shipped, tested coverage is much
narrower: see the README, and every pack's `labelled_turns` count.

Typical use is through a framework adapter, but the core is importable
directly for in-house stacks:

    from baobab_turn import BaobabConfig, GuardInput, MarkerIndex, guard, resolve

    index = MarkerIndex(resolve("pcm"))
    result = guard(GuardInput(p_inner=0.75, transcript_tail="the thing be say"), index=index)
    result.p_adjusted  # 0.345 -- the speaker is not finished
"""

from __future__ import annotations

from .core.config import BaobabConfig
from .core.guard import guard
from .core.markers import MarkerIndex, normalise, tokenise
from .core.types import (
    NULL_PACK,
    GuardInput,
    GuardResult,
    LanguagePack,
    Marker,
)
from .packs.registry import available_packs, resolve
from .version import __version__

__all__ = [
    "BaobabConfig",
    "GuardInput",
    "GuardResult",
    "LanguagePack",
    "Marker",
    "MarkerIndex",
    "NULL_PACK",
    "available_packs",
    "guard",
    "normalise",
    "resolve",
    "tokenise",
    "__version__",
]
