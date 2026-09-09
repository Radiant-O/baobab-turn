"""Mine candidate discourse markers from a spoken treebank.

    python tools/mine_markers.py path/to/treebank/*.conllu
    python tools/mine_markers.py corpus/ --min-count 8 --yaml

**This produces candidates for a speaker to review. It does not produce a
language pack.** The distinction is the whole point. A wrong marker list is
worse than no marker list, and no amount of frequency data tells you whether
a word is a discourse marker or just a common noun that happens to end
sentences -- the run against Hausa broadcast conversation put *laːfiyàː*
("health", as in a greeting response) and *kaːwoː* ("bring") at the top of
the yield list purely because turns ended on them.

So the output is evidence, ranked, with counts shown. A speaker of the
language decides what is real.

## What it measures

Spoken treebanks in the UD/SUD family mark `# speaker_id` per utterance, so a
speaker change is a real turn boundary. For every word that ends an utterance
we count:

- **turn-final**: the next utterance belongs to a different speaker. The
  speaker finished and handed over.
- **mid-turn**: the next utterance is the same speaker. They kept the floor.

A word skewed towards turn-final is a yield-marker candidate; one skewed
towards mid-turn is a continuation candidate. That maps directly onto the two
marker lists a language pack needs.

It also reports words the corpus annotators themselves tagged with the
`discourse` relation. **Trust that list far more than the ratios** -- it is a
linguist's judgement that the word is a discourse marker at all, which is
exactly the judgement frequency cannot make.

## Why no corpus ships with this package

The treebanks worth mining are CC BY-SA 4.0 -- share-alike. Vendoring one
into an MIT package would drag that obligation along with it. So the tool
ships and the data does not: point it at your own copy. Known usable corpora,
all CC BY-SA 4.0, all genre `spoken`, all with `speaker_id`:

- **Naija (Nigerian Pidgin)** -- UD_Naija-NSC, 9,242 sentences / 140,729
  tokens. github.com/UniversalDependencies/UD_Naija-NSC
- **Hausa (Southern)** -- SUD_Hausa-SouthernAutogramm, broadcast conversation.
  github.com/surfacesyntacticud/SUD_Hausa-SouthernAutogramm
- **Hausa (Northern, Ader)** -- SUD_Hausa-NorthernAutogramm, 400 sentences.

Attribute whatever you use. If you publish weights derived from a CC BY-SA
corpus, say which corpus and cite it.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import sys

SPEAKER_RE = re.compile(r"^# speaker_id = (\S+)", re.MULTILINE)

#: Dependency relations that mark a token as a discourse-level element.
#:
#: `discourse` alone is not enough, and getting this wrong silently drops the
#: most interesting class. UD uses `discourse` for markers and fillers, but
#: SUD -- which is what the Naija and Hausa treebanks are natively annotated
#: in -- puts *emphatic and terminal particles* under `mod:emph` instead. The
#: trailing `o` in "for dis Nigeria o //" is `PART` / `mod:emph`, not
#: `discourse`. Filtering on `discourse` only under-counted exactly the
#: ambiguous trailing-particle class the packs care most about: 8,459
#: `discourse` versus 3,524 `mod:emph` in SUD Naija.
MARKER_RELS = ("discourse", "mod:emph")


class Utterance:
    __slots__ = ("discourse", "speaker", "words")

    def __init__(self, speaker: str, words: list[str], discourse: list[str]) -> None:
        self.speaker = speaker
        self.words = words
        self.discourse = discourse


def parse_conllu(path: str) -> list[Utterance]:
    """Utterances with a speaker id. Blocks without one are skipped.

    Punctuation is dropped: these corpora use symbolic markers like `//` and
    `<` for prosodic boundaries, which would otherwise be the most frequent
    "word" at every position.
    """
    out: list[Utterance] = []
    with open(path, encoding="utf-8") as fh:
        blocks = fh.read().split("\n\n")

    for block in blocks:
        if not block.strip():
            continue
        speaker = SPEAKER_RE.search(block)
        if speaker is None:
            continue

        words: list[str] = []
        discourse: list[str] = []
        for line in block.splitlines():
            if not line or line.startswith("#") or "\t" not in line:
                continue
            cols = line.split("\t")
            if len(cols) < 8:
                continue
            form, lemma, upos, deprel = cols[1], cols[2], cols[3], cols[7]
            if upos == "PUNCT":
                continue
            words.append(form.lower())
            if any(deprel.startswith(rel) for rel in MARKER_RELS):
                discourse.append((lemma or form).lower())

        if words:
            out.append(Utterance(speaker.group(1), words, discourse))
    return out


def expand(paths: list[str]) -> list[str]:
    """Accept files, directories and globs. Deduplicate by basename, since
    the same treebank is often present under two directories."""
    found: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            found.extend(sorted(glob.glob(os.path.join(p, "**", "*.conllu"), recursive=True)))
        else:
            found.extend(sorted(glob.glob(p)))

    seen: set[str] = set()
    unique: list[str] = []
    for f in found:
        name = os.path.basename(f)
        if name in seen:
            continue
        seen.add(name)
        unique.append(f)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".conllu files, globs, or a directory")
    ap.add_argument("--min-count", type=int, default=5,
                    help="ignore words seen fewer times in that position (default 5). "
                         "Low counts produce meaningless 100%% ratios.")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--yaml", action="store_true",
                    help="emit a pack skeleton for a speaker to edit")
    args = ap.parse_args()

    files = expand(args.paths)
    if not files:
        print("No .conllu files found.", file=sys.stderr)
        return 1

    turn_final: collections.Counter[str] = collections.Counter()
    mid_turn: collections.Counter[str] = collections.Counter()
    tagged: collections.Counter[str] = collections.Counter()
    n_utts = n_speakers_changes = 0
    speakers: set[str] = set()

    for path in files:
        utts = parse_conllu(path)
        n_utts += len(utts)
        for i, utt in enumerate(utts):
            speakers.add(utt.speaker)
            for d in utt.discourse:
                tagged[d] += 1
            nxt = utts[i + 1].speaker if i + 1 < len(utts) else None
            if nxt is None:
                continue  # last utterance of a file: no evidence either way
            if nxt != utt.speaker:
                n_speakers_changes += 1
                turn_final[utt.words[-1]] += 1
            else:
                mid_turn[utt.words[-1]] += 1

    if n_utts == 0:
        print("No utterances with '# speaker_id' found. This tool needs a spoken\n"
              "treebank that marks speakers -- a written one has no turns.",
              file=sys.stderr)
        return 1

    print(f"{len(files)} files, {n_utts} utterances, {len(speakers)} speaker labels")
    print(f"{n_speakers_changes} speaker changes (real turn boundaries)")
    print()

    if tagged:
        print("ANNOTATED AS DISCOURSE MARKERS by the corpus linguists.")
        print("Trust this more than the ratios below -- it is a judgement that the")
        print("word is a discourse marker at all, which frequency cannot make.")
        print(f"  {'word':20} {'count':>6}")
        for word, count in tagged.most_common(args.top):
            print(f"  {word:20} {count:>6}")
        print()

    def ranked(primary: collections.Counter[str], other: collections.Counter[str]):
        rows = []
        for word, count in primary.items():
            if count < args.min_count:
                continue
            alt = other.get(word, 0)
            rows.append((count / (count + alt), count, alt, word))
        return sorted(rows, reverse=True)[: args.top]

    yields = ranked(turn_final, mid_turn)
    holds = ranked(mid_turn, turn_final)

    # The intersection is the strongest evidence this tool can produce: a
    # linguist judged the word to be a discourse marker, AND its position
    # skews one way. Either signal alone is weak -- the annotation says
    # nothing about which way a marker points, and the ratio says nothing
    # about whether the word is a marker at all.
    both = [
        (kind, ratio, count, alt, word)
        for kind, rows in (("yield", yields), ("continuation", holds))
        for ratio, count, alt, word in rows
        if word in tagged
    ]
    if both:
        print("BEST EVIDENCE -- tagged as a discourse marker AND positionally skewed.")
        print("Start a speaker's review here.")
        print(f"  {'word':20} {'points to':>13} {'ratio':>7} {'tagged':>7}")
        for kind, ratio, _count, _alt, word in sorted(both, key=lambda r: -r[1]):
            print(f"  {word:20} {kind:>13} {ratio:>6.0%} {tagged[word]:>7}")
        print()

    print(f"YIELD CANDIDATES -- ends a turn more than it keeps one (min {args.min_count})")
    print(f"  {'word':20} {'final':>6} {'mid':>5} {'ratio':>7}")
    for ratio, count, alt, word in yields:
        print(f"  {word:20} {count:>6} {alt:>5} {ratio:>6.0%}")
    print()

    print(f"CONTINUATION CANDIDATES -- keeps the floor (min {args.min_count})")
    print(f"  {'word':20} {'mid':>6} {'final':>5} {'ratio':>7}")
    for ratio, count, alt, word in holds:
        print(f"  {word:20} {count:>6} {alt:>5} {ratio:>6.0%}")
    print()

    if args.yaml:
        print("# " + "-" * 68)
        print("# CANDIDATES ONLY -- not a language pack.")
        print("#")
        print("# Frequency cannot tell a discourse marker from a common noun that")
        print("# happens to end sentences. Delete everything that is not really a")
        print("# turn-taking cue, then set weights by judgement. The ratio is")
        print("# evidence, not a weight.")
        print("#")
        print("# Do not merge this into packs/ until a speaker of the language has")
        print("# been through it line by line.")
        print("# " + "-" * 68)
        print("continuation_markers:")
        for ratio, count, alt, word in holds:
            print(f'  - {{text: "{word}", weight: 0.5}}'
                  f"   # mid {count}, final {alt}, {ratio:.0%}")
        print("yield_markers:")
        for ratio, count, alt, word in yields:
            print(f'  - {{text: "{word}", weight: 0.5}}'
                  f"   # final {count}, mid {alt}, {ratio:.0%}")

    print("Ratios are evidence for a speaker to weigh, not weights. A word can top")
    print("the yield list simply because turns happened to end on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
