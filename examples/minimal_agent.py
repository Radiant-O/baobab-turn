"""The smallest LiveKit agent using baobab-turn.

This will not run without credentials -- a LiveKit URL and key, plus an STT
and TTS. It is here to show the integration, which is two lines:

    guard = BaobabGuard()
    ...turn_detection=guard.turn_detection...
    guard.attach(session)

Run it with:

    pip install "baobab-turn[livekit]"
    export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
    python examples/minimal_agent.py dev

Two things about this file are deliberate.

**`attach()` is not optional.** The turn detector is fed audio frames and
never sees text, so without `attach()` the marker layer has no transcript and
abstains on every turn -- the agent still works, it just behaves exactly like
stock. If markers seem to do nothing, this is the first thing to check.

**The STT must produce interim results.** The guard inspects the tail of a
*partial* transcript while the caller is still speaking. A batch-only STT
gives us text after the turn has already committed, which is too late to be
useful. Any streaming STT works; the marker weights here were written for
Nigerian English and Pidgin, so a Nigerian-capable one gets more out of them.
"""

from __future__ import annotations

import logging

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    inference,
)

from baobab_turn import BaobabConfig
from baobab_turn.adapters.livekit import BaobabGuard

log = logging.getLogger("minimal-agent")


async def entrypoint(ctx: JobContext) -> None:
    guard = BaobabGuard(
        config=BaobabConfig(),
        # Every rule fired is reported here. Worth logging from day one: the
        # useful number is how often the lexical layer had to abstain because
        # no transcript was available or it was stale, since that caps how
        # much the marker layer can be worth at all.
        on_decision=lambda gi, result: log.info(
            "turn: %.3f -> %s  %s",
            gi.p_inner if gi.p_inner is not None else -1.0,
            f"{result.p_adjusted:.3f}" if result.p_adjusted is not None else "n/a",
            ", ".join(result.reasons),
        ),
    )

    session = AgentSession(
        turn_handling=TurnHandlingOptions(
            turn_detection=guard.turn_detection,
        ),
        # Any streaming STT. Set the language to match a pack -- "en-NG" or
        # "pcm" today -- or leave it to the provider and let the transcript
        # event report it.
        stt=inference.STT(),
        llm=inference.LLM(),
        tts=inference.TTS(),
        vad=inference.VAD(),
    )

    # Feeds the transcript tail from the session's public transcript events,
    # so this works with whatever STT you plugged in above.
    guard.attach(session)

    await session.start(
        agent=Agent(
            instructions=(
                "You are speaking with a Nigerian caller who may switch "
                "between English and Pidgin. Keep replies short."
            )
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
