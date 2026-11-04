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

Every corpus worth mining carries a licence that vendoring would drag into
an MIT package: the treebanks are CC BY-SA 4.0 (share-alike) and SBCSAE is
CC BY-ND 3.0 US (no derivative works). So the tool ships and the data does
not: point it at your own copy.

Known usable corpora, all genre `spoken`, all with real speaker turns:

- **Naija (Nigerian Pidgin)** -- UD_Naija-NSC, 9,242 sentences / 140,729
  tokens, CC BY-SA 4.0.
  github.com/UniversalDependencies/UD_Naija-NSC
- **Hausa (Southern)** -- SUD_Hausa-SouthernAutogramm, broadcast
  conversation, CC BY-SA 4.0.
  github.com/surfacesyntacticud/SUD_Hausa-SouthernAutogramm
- **Hausa (Northern, Ader)** -- SUD_Hausa-NorthernAutogramm, 400 sentences.
- **American English** -- SBCSAE, 60 face-to-face conversations, ~249k
  tokens, CC BY-ND 3.0 US. Transcripts alone are a 1.2MB download:
  linguistics.ucsb.edu/sites/default/files/sitefiles/research/SBC/SBCorpus.zip

No-derivatives deserves a word, since it sounds more restrictive than it is
here. ND restricts *distributing a modified copy* of the corpus. Reading it
and publishing counts derived from it is ordinary analysis, and this tool
never writes the corpus back out -- but it does mean an SBCSAE-derived pack
must cite the corpus and must not ship any of its text.

## Formats it reads

- `.conllu` -- UD/SUD treebanks, speakers from `# speaker_id`.
- `.trn` -- SBCSAE, speakers from the transcript's own speaker column.
- anything else -- a plain `SPEAKERn: text` transcript, one turn per line.

Attribute whatever you use. If you publish weights derived from a corpus,
say which corpus and cite it.
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


#: `SPEAKER1:`, `Speaker 1:`, `SPEAKER 1:` -- CENCOS uses all three, sometimes
#: within one file, so the match has to be case- and space-insensitive.
TURN_RE = re.compile(r"^\s*speaker\s*(\d+)\s*:\s*(.*)$", re.IGNORECASE)


def read_text(path: str) -> str:
    """Decode a transcript, tolerating either encoding.

    CENCOS is documented as cp1252 but is actually UTF-8 on inspection, and
    the two are indistinguishable without trying. UTF-8 first, because a
    cp1252 file rarely decodes as valid UTF-8 while the reverse silently
    produces mojibake.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_transcript(path: str) -> list[Utterance]:
    """A plain speaker-labelled transcript, one turn per line.

    No dependency annotation here, so there is no `discourse` relation to
    cross-reference -- the positional ratios are the only evidence such a
    corpus can give. CENCOS says as much itself: "no annotation other than
    the speakers."
    """
    out: list[Utterance] = []
    for line in read_text(path).replace("\x85", "\n").splitlines():
        match = TURN_RE.match(line)
        if match is None:
            continue
        speaker, text = match.group(1), match.group(2)
        # Strip the transcription conventions rather than counting them as
        # words: '...' marks a latched or trailing turn, and bracketed items
        # are annotator notes.
        text = re.sub(r"\[[^\]]*\]", " ", text)
        words = [w for w in tokenise(text) if w]
        if words:
            out.append(Utterance(f"SP{speaker}", words, []))
    return out


TOKEN_RE = re.compile(r"[0-9a-zÀ-ɏ']+")


def tokenise(text: str) -> list[str]:
    """Lowercase words, punctuation dropped, apostrophes kept inside words."""
    return TOKEN_RE.findall(text.lower())


# --- SBCSAE -------------------------------------------------------------
#
# The Santa Barbara Corpus is the only openly-licensed English conversational
# corpus found that records turn structure. The CC BY-SA treebanks do not:
# UD_English-GUM is non-commercial, and UD_English-CHILDES and
# UD_English-ESLSpok drop speaker turns in conversion, which is why this
# needs a reader of its own rather than another `# speaker_id` corpus.

#: "START END<TAB>SPEAKER:<TAB>TEXT".
#:
#: The two time marks are separated by a space in the early files and a tab
#: in the later ones, so both are accepted. Requiring a real decimal time is
#: also what discards the annotators' own queries, which are written against
#: a `000000000` pseudo-time and contain no speech.
TRN_RE = re.compile(r"^(\d+\.\d+)\s+(\d+\.\d+)\t(.*)$")

#: A speaker label at the start of what follows the times. In some files the
#: label is its own tab-separated column and in others it is space-padded
#: within one, so this matches the label rather than the layout.
#:
#: Crucially, the column is **blank when the same speaker continues** into
#: the next intonation unit. That is the single most valuable thing this
#: corpus records for us: it is the within-turn structure that makes the
#: turn-final versus mid-turn contrast measurable at all.
TRN_SPEAKER_RE = re.compile(r"^\s*([A-Za-z#>*][A-Za-z0-9_/?>*]{0,11})\s*:\s")

#: Labels that are not a person taking a turn.
#:
#: `>ENV`, `>DOG` and `>RADIO` are environmental sound -- a dog barking is
#: not a turn. `X` is an unidentified voice, and two consecutive `X` lines
#: may be two different people, so counting a speaker change between them
#: would invent a turn boundary that nobody can verify.
NON_SPEAKER_RE = re.compile(r"^(>.*|ENV|\*?X(_\d+)?)$")

#: Everything SBCSAE marks that is not a word. Without this the tokeniser
#: produces nonsense that ranks high enough to crowd out the real
#: candidates: `(H)` becomes the word `h`, `XX` becomes `xx`, and the
#: overlap index in `[2 ... 2]` becomes the word `2`.
SBC_COMMENT = re.compile(r"\(\([^)]*\)\)")              # ((AIR_CONDITIONER))
SBC_VOCAL = re.compile(r"\([A-Za-z_]+\)")               # (H) (Hx) (TSK)
SBC_ANGLE = re.compile(r"<[A-Z_0-9]+\b|\b[A-Z_0-9]+>")  # <READ ... READ>
SBC_UNCLEAR = re.compile(r"\bX+\b", re.IGNORECASE)      # XX, XXXX: unintelligible

#: Overlap brackets, with the index that pairs them: `[2 ... 2]`.
#:
#: These are **deleted, not spaced**, and the reason is the whole trick of
#: reading this corpus. SBCSAE puts a bracket at the instant overlap starts,
#: which is very often in the middle of a word -- `righ[t]`, `Deli[2ciou2]s`,
#: `every2]thing`. Replacing a bracket with a space shatters those into
#: `righ` + `t` and `deli` + `ciou` + `s`, and the fragments are frequent
#: enough to reach the top of a candidate list.
#:
#: The digit is only dropped when a single one sits against the bracket, so
#: that a spoken number keeps its digits: `[50` is fifty, not overlap 5.
SBC_OVERLAP = re.compile(r"\[\d(?!\d)|\[|(?<!\d)\d\]|\]")

#: Deleted rather than replaced by a space, because these sit *inside* a
#: word: `=` is lengthening, `%` a glottal stop, `~` an anonymised name, `^`
#: an accent. Substituting a space would cut `ti=me` into `ti` and `me`.
SBC_INTRAWORD = re.compile(r"[=%~!^&]")

#: Replaced by a space, because these separate words. `@` is laughter.
SBC_SEPARATOR = re.compile(r"[@+…—]")


def is_speaker(label: str) -> bool:
    """Is this label a person whose turn we can count?"""
    return NON_SPEAKER_RE.match(label.strip().rstrip("?").upper()) is None


def clean_sbcsae(text: str) -> str:
    """Strip SBCSAE's transcription conventions, leaving spoken words."""
    text = SBC_COMMENT.sub(" ", text)
    text = SBC_ANGLE.sub(" ", text)
    text = SBC_VOCAL.sub(" ", text)
    text = SBC_OVERLAP.sub("", text)
    text = SBC_INTRAWORD.sub("", text)
    text = SBC_SEPARATOR.sub(" ", text)
    return SBC_UNCLEAR.sub(" ", text)


def tokenise_sbcsae(text: str) -> list[str]:
    """Like `tokenise`, but drops words the speaker cut off.

    A trailing hyphen marks a word abandoned mid-production (`y-`, `the-`).
    Those are not lexical items, and keeping them would put fragments near
    the top of a list meant for a human to read.

    The hyphen is not always last: a cut-off word still takes the comma or
    full stop that closes its intonation unit, so `s-,` and `the-.` have to
    be recognised too.
    """
    out: list[str] = []
    for chunk in text.lower().split():
        if chunk.rstrip(".,?!;:\"')").endswith("-"):
            continue
        out.extend(TOKEN_RE.findall(chunk))
    return out


def parse_sbcsae(path: str) -> list[Utterance]:
    """One intonation unit per line, from the Santa Barbara Corpus.

    Unlike the plain transcripts this tool also reads, SBCSAE segments each
    turn into intonation units and names the speaker only when it changes.
    That is what makes the cross-turn measure usable here rather than
    degenerate: across the 60 conversations, 39,945 units continue the
    current speaker against 28,748 that hand over, so "ends a turn" and
    "keeps the floor" are both well attested.
    """
    out: list[Utterance] = []
    speaker: str | None = None
    for line in read_text(path).splitlines():
        match = TRN_RE.match(line)
        if match is None:
            continue
        # Normalise the one- and two-tab layouts to the same thing before
        # looking for a label.
        rest = match.group(3).replace("\t", " ")
        label = TRN_SPEAKER_RE.match(rest)
        if label is not None:
            speaker = label.group(1)
            rest = rest[label.end():]
        if speaker is None or not is_speaker(speaker):
            # A blank column inherits the line above, so dropping a
            # non-speaker also drops the lines that continue it.
            continue
        words = tokenise_sbcsae(clean_sbcsae(rest))
        if words:
            out.append(Utterance(speaker.rstrip("?").upper(), words, []))
    return out


def parse_any(path: str) -> list[Utterance]:
    lower = path.lower()
    if lower.endswith(".conllu"):
        return parse_conllu(path)
    if lower.endswith(".trn"):
        return parse_sbcsae(path)
    return parse_transcript(path)


def expand(paths: list[str]) -> list[str]:
    """Accept files, directories and globs. Deduplicate by basename, since
    the same treebank is often present under two directories."""
    found: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.conllu", "*.txt", "*.trn"):
                found.extend(sorted(glob.glob(os.path.join(p, "**", ext), recursive=True)))
        else:
            found.extend(sorted(glob.glob(p)))

    seen: set[str] = set()
    unique: list[str] = []
    for f in found:
        name = os.path.basename(f)
        # AppleDouble sidecars. The SBCSAE zip carries one per transcript,
        # and they parse as empty files rather than failing loudly.
        if name.startswith("._") or name in seen:
            continue
        seen.add(name)
        unique.append(f)
    return unique


def position_within_turn(files: list[str]):
    """For each word: how often it ends its speaker's turn vs appears earlier.

    The measure for one-line-per-turn transcripts. It answers "is this a
    terminal particle?" rather than "does this hand the floor over?" -- a
    weaker claim, but one the data can actually support.
    """
    final: collections.Counter[str] = collections.Counter()
    elsewhere: collections.Counter[str] = collections.Counter()
    for path in files:
        for utt in parse_any(path):
            for word in utt.words[:-1]:
                elsewhere[word] += 1
            final[utt.words[-1]] += 1
    return final, elsewhere


def report_position(final, elsewhere, args) -> None:
    rows = []
    for word, count in final.items():
        if count < args.min_count:
            continue
        other = elsewhere.get(word, 0)
        rows.append((count / (count + other), count, other, word))
    print(f"TURN-FINAL POSITION -- ends its own turn (min {args.min_count})")
    print(f"  {'word':20} {'final':>6} {'earlier':>8} {'ratio':>7}")
    for ratio, count, other, word in sorted(rows, reverse=True)[: args.top]:
        print(f"  {word:20} {count:>6} {other:>8} {ratio:>6.0%}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help=".conllu or .txt files, globs, or a directory")
    ap.add_argument("--min-count", type=int, default=5,
                    help="ignore words seen fewer times in that position (default 5). "
                         "Low counts produce meaningless 100%% ratios.")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--yaml", action="store_true",
                    help="emit a pack skeleton for a speaker to edit")
    args = ap.parse_args()

    files = expand(args.paths)
    if not files:
        print("No .conllu or .txt files found.", file=sys.stderr)
        return 1

    turn_final: collections.Counter[str] = collections.Counter()
    mid_turn: collections.Counter[str] = collections.Counter()
    tagged: collections.Counter[str] = collections.Counter()
    n_utts = n_speakers_changes = 0
    speakers: set[str] = set()

    for path in files:
        utts = parse_any(path)
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

    # Guard against a corpus shape the turn-final/mid-turn contrast cannot
    # measure. Treebanks segment a turn into several inter-pausal units, so
    # "same speaker continues" is common and the contrast is meaningful.
    # A transcript with one line per turn has almost no within-turn
    # structure, so nearly every utterance is turn-final and every ratio
    # collapses to 100% -- which looks like a strong result and is an
    # artefact. CENCOS is exactly this shape: 5,746 speaker changes across
    # 5,911 utterances.
    degenerate = sum(mid_turn.values()) < 0.05 * max(sum(turn_final.values()), 1)
    if degenerate:
        print("NOTE: this corpus transcribes one line per turn, so there is no")
        print("within-turn structure to contrast against and the turn-final vs")
        print("mid-turn ratios would all collapse to 100%. Reporting position")
        print("WITHIN each turn instead, which is the measure this shape supports.")
        print()
        final_in_turn, elsewhere = position_within_turn(files)
        report_position(final_in_turn, elsewhere, args)
        print("A high ratio means the word tends to come last in its speaker's own")
        print("turn. That is evidence of a terminal particle, but unlike the")
        print("cross-turn measure it cannot distinguish yielding from merely")
        print("finishing a sentence.")
        return 0

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
