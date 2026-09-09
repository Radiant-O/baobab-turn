"""Type a sentence, see what the turn detector would do with it.

There is no agent and no audio here -- this drives the guard directly so you
can check whether its judgement matches your ear. That check is the whole
point: the marker weights in the packs are guesses, and the only way to know
whether "sha" really deserves 0.4 is for a speaker to disagree with it.

    python examples/try_it.py            # preset phrases, then interactive
    python examples/try_it.py --demo     # preset phrases only
    python examples/try_it.py -l en-NG   # Nigerian English instead of Pidgin

What you are looking at:

    p_inner      what a stock detector thinks -- 0.75 means "they're done,
                 I'll reply now", which for these phrases is often wrong
    p_adjusted   what it thinks after the guard has had its say
    commit       whether the agent would start talking

Lower is better for the phrases where the speaker is mid-sentence. Higher is
better where they have genuinely finished.
"""

from __future__ import annotations

import argparse
import sys

from baobab_turn import BaobabConfig, GuardInput, MarkerIndex, guard, resolve

# Roughly what LiveKit's detector reports on a pause it reads as terminal.
# The interesting question is what the guard does with that opinion.
DEFAULT_P = 0.75

DEMO: list[tuple[str, str]] = [
    ("i wan tell you say the thing be say", "mid-sentence -- should WAIT"),
    ("na him talk am, erm", "hesitating -- should WAIT"),
    ("the reason be say", "clause coming -- should WAIT"),
    ("i no gree because", "clause coming -- should WAIT"),
    ("e don finish abi", "handing over -- should REPLY"),
    ("you dey follow", "handing over -- should REPLY"),
    ("no be so", "handing over -- should REPLY"),
    ("i go come tomorrow", "plain statement -- no opinion"),
    ("e don finish o", "trailing 'o' -- weak hint only"),
    ("i dey come sha", "trailing 'sha' -- weak hint only"),
    ("be like say na only you go enter that side o", "handing over -- should REPLY"),
]


def _render(tail: str, language: str, index: MarkerIndex, config: BaobabConfig,
            is_final: bool = True) -> None:
    result = guard(
        GuardInput(
            p_inner=DEFAULT_P,
            transcript_tail=tail,
            is_final=is_final,
            language=language,
            tail_age_ms=50,
        ),
        config,
        index,
    )

    adjusted = result.p_adjusted if result.p_adjusted is not None else DEFAULT_P
    move = adjusted - DEFAULT_P
    if abs(move) < 1e-9:
        arrow = "  (unchanged)"
    elif move < 0:
        arrow = f"  WAITS   ({move:+.3f})"
    else:
        arrow = f"  REPLIES ({move:+.3f})"

    print(f"    {DEFAULT_P:.2f} -> {adjusted:.3f}{arrow}")
    print(f"    commit now: {'yes' if result.decision else 'no'}")
    if result.delay_override_ms is not None:
        print(f"    extra patience: {result.delay_override_ms} ms")
    print(f"    why: {', '.join(result.reasons)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-l", "--language", default="pcm",
                        help="language code, e.g. pcm or en-NG (default: pcm)")
    parser.add_argument("--demo", action="store_true",
                        help="run the preset phrases and exit")
    args = parser.parse_args()

    pack = resolve(args.language)
    if pack is None:
        print(f"No pack for {args.language!r}. Try: pcm, en-NG")
        return 1

    index = MarkerIndex(pack)
    config = BaobabConfig()

    print(f"\n{pack.name} ({pack.code}) -- status: {pack.status}, "
          f"labelled turns: {pack.labelled_turns}")
    print(f"{len(pack.continuation_markers)} continuation markers, "
          f"{len(pack.yield_markers)} yield markers, "
          f"{len(pack.ambiguous_markers)} ambiguous")
    print("\nA stock detector would say 0.75 -- 'they're done, reply now' -- "
          "for every line below.\n")

    for phrase, expectation in DEMO:
        print(f'  "{phrase}"')
        print(f"    expected: {expectation}")
        _render(phrase, args.language, index, config)
        print()

    if args.demo:
        return 0

    print("-" * 68)
    print("Now type your own. Blank line or Ctrl-C to quit.")
    print("Judge it by ear: does it wait when you would still be talking?\n")

    while True:
        try:
            tail = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not tail:
            break
        _render(tail, args.language, index, config)
        print()

    print("If any of that felt wrong, the weights are in "
          f"baobab_turn/packs/{pack.code}.yaml -- plain YAML, 0.0 to 1.0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
