"""Run a whole phrase bank through the guard and report where it is blind.

    python examples/check_phrases.py
    python examples/check_phrases.py -f my_phrases.txt -l en-NG
    python examples/check_phrases.py --gaps          # only the misses

Typing phrases one at a time tells you very little. Running two hundred at
once tells you the two things that actually matter:

**Agreement** -- of the phrases the engine had an opinion about, how often
did it agree with you.

**Coverage** -- how many phrases it had NO opinion about at all. This is the
more important number and the one people forget. The engine is a dictionary
lookup over the last few words: if a phrase contains no word from the pack,
nothing happens and the stock detector's guess stands unchanged. Coverage is
therefore a direct measurement of how complete the word list is, and the
gap list at the end of the report is a to-do list for the YAML.

A low agreement score means the weights are wrong. Low coverage means the
vocabulary is too small. They are different problems with different fixes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baobab_turn import BaobabConfig, GuardInput, MarkerIndex, guard, resolve

DEFAULT_P = 0.75
EPSILON = 1e-9
VALID = {"wait", "reply", "none"}


def parse(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Returns (cases, complaints). A bad line is reported, never fatal."""
    cases: list[tuple[str, str]] = []
    complaints: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            complaints.append(f"line {number}: no '|' separator -- {line!r}")
            continue
        expected, _, phrase = line.partition("|")
        expected = expected.strip().lower()
        phrase = phrase.strip()
        if expected not in VALID:
            complaints.append(
                f"line {number}: expected must be one of {sorted(VALID)}, got {expected!r}"
            )
            continue
        if not phrase:
            complaints.append(f"line {number}: empty phrase")
            continue
        cases.append((expected, phrase))
    return cases, complaints


def verdict(phrase: str, index: MarkerIndex, config: BaobabConfig) -> tuple[str, float, str]:
    """Returns (wait|reply|none, delta, why)."""
    result = guard(
        GuardInput(
            p_inner=DEFAULT_P,
            transcript_tail=phrase,
            is_final=True,
            tail_age_ms=50,
        ),
        config,
        index,
    )
    p = result.p_adjusted if result.p_adjusted is not None else DEFAULT_P
    delta = p - DEFAULT_P
    why = ", ".join(result.reasons)
    if delta < -EPSILON:
        return "wait", delta, why
    if delta > EPSILON:
        return "reply", delta, why
    return "none", delta, why


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--file", default="examples/phrases.txt")
    parser.add_argument("-l", "--language", default="pcm")
    parser.add_argument("--gaps", action="store_true",
                        help="show only the phrases the engine had no opinion on")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"No such file: {path}")
        return 1

    pack = resolve(args.language)
    if pack is None:
        print(f"No pack for {args.language!r}. Try: pcm, en-NG")
        return 1

    cases, complaints = parse(path)
    if complaints:
        print("Problems in the file (skipped):")
        for c in complaints:
            print(f"  {c}")
        print()
    if not cases:
        print("No usable phrases found.")
        return 1

    index = MarkerIndex(pack)
    config = BaobabConfig()

    agreed = disagreed = 0
    gaps: list[tuple[str, str]] = []
    wrong: list[tuple[str, str, str, str]] = []

    for expected, phrase in cases:
        got, delta, why = verdict(phrase, index, config)

        # "none" from the engine means it recognised nothing at all. That is a
        # coverage gap, not a disagreement -- unless you expected none.
        if got == "none" and expected != "none":
            gaps.append((expected, phrase))
            continue

        if got == expected:
            agreed += 1
            if not args.gaps:
                print(f"  ok    {expected:5}  {phrase}")
                print(f"                 {delta:+.3f}  {why}")
        else:
            disagreed += 1
            wrong.append((expected, got, phrase, why))

    print()
    print("=" * 70)
    print(f"{pack.name} ({pack.code}) -- {len(pack.continuation_markers)} continuation, "
          f"{len(pack.yield_markers)} yield, {len(pack.ambiguous_markers)} ambiguous markers")
    print(f"{len(cases)} phrases checked")
    print()

    opinions = agreed + disagreed
    if opinions:
        print(f"AGREEMENT  {agreed}/{opinions} "
              f"({100 * agreed / opinions:.0f}%) of phrases it had an opinion on")
    covered = len(cases) - len(gaps)
    print(f"COVERAGE   {covered}/{len(cases)} "
          f"({100 * covered / len(cases):.0f}%) -- it recognised something")

    if wrong:
        print()
        print(f"DISAGREED ({len(wrong)}) -- the weights are wrong for these:")
        for expected, got, phrase, why in wrong:
            print(f"  wanted {expected:5} got {got:5}  {phrase}")
            print(f"                            {why}")

    if gaps:
        print()
        print(f"NO OPINION ({len(gaps)}) -- these are the gaps in the word list.")
        print("Each one needs a word or frame adding to "
              f"baobab_turn/packs/{pack.code}.yaml:")
        for expected, phrase in gaps:
            print(f"  wanted {expected:5}  {phrase}")

    print()
    print("Two different problems: DISAGREED means a weight is wrong,")
    print("NO OPINION means the vocabulary is too small.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
