"""Turn a recording into labelled turns the benchmark can read.

    python tools/label.py new recording.wav --out bench/datasets/yo/session1.txt
    python tools/label.py check bench/datasets/yo/session1.txt

This exists because every pack in this repo says `labelled_turns: 0`, and
will keep saying it until somebody sits down with audio and marks where the
turns actually ended. There is no openly-licensed conversational corpus for
Yoruba or Igbo to borrow instead -- that was checked across CorporAn,
CoCoON, ORTOLANG, ELRA, LDC, TalkBank and CHILDES.

## What it writes

The format `bench/corpora.py` already reads, and nothing new:

    SPEAKER1: mo ti so fun un
    SPEAKER1: sugbon ko gbo
    SPEAKER2: ah

The benchmark infers a boundary from the speaker changing. So two
consecutive lines from the same speaker are not a formatting detail -- they
are the most valuable thing you can record here. They say "this person
paused and was not finished", which is exactly the judgement the detector
gets wrong, and the only evidence that can prove it.

## What it never writes

The audio. Hard rule: no raw caller audio in the repo, ever. The manifest
beside the transcript records the recording's SHA-256, its size and how long
the session took, so a result can be traced back to a file without the file
travelling with it. Keep the audio somewhere else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.corpora import load_transcripts

HELP = """
  <text>        the next speaker says this
  = <text>      the SAME speaker continues -- a pause that did not end a turn
  .undo         remove the last line
  .stats        lines so far, and how many pauses were not turn ends
  .help         this
  .done         finish and write the manifest

Everything is saved as you go, so a crash costs you nothing.
"""


def digest(path: Path) -> tuple[str, int]:
    """SHA-256 and size, so a transcript can name its audio without carrying it."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def render(lines: list[tuple[int, str]]) -> str:
    return "".join(f"SPEAKER{s}: {t}\n" for s, t in lines)


def parse_existing(text: str) -> list[tuple[int, str]]:
    """Read back what we wrote, so --append can continue a session."""
    out: list[tuple[int, str]] = []
    for raw in text.splitlines():
        head, sep, body = raw.partition(":")
        if not sep or not head.upper().startswith("SPEAKER"):
            continue
        number = head[len("SPEAKER"):].strip()
        if not number.isdigit():
            continue
        body = body.strip()
        if body:
            out.append((int(number), body))
    return out


def counts(lines: list[tuple[int, str]]) -> tuple[int, int]:
    """(turn boundaries, same-speaker continuations)."""
    if len(lines) < 2:
        return 0, 0
    boundaries = sum(1 for i in range(len(lines) - 1) if lines[i][0] != lines[i + 1][0])
    return boundaries, len(lines) - 1 - boundaries


def cmd_new(args: argparse.Namespace) -> int:
    audio = Path(args.audio)
    if not audio.is_file():
        print(f"no such file: {audio}")
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.append:
        print(f"{out} exists. Use --append to continue it, or choose another name.")
        return 2

    lines: list[tuple[int, str]] = []
    if args.append and out.exists():
        lines = parse_existing(out.read_text(encoding="utf-8"))
        print(f"continuing {out} at {len(lines)} lines")

    print(f"Labelling {audio.name} -> {out}")
    print("Play the audio in whatever player you like; type what you hear.")
    print(HELP)

    started = time.time()
    # Starts at 2 so the first alternation lands on SPEAKER1.
    speaker = lines[-1][0] if lines else 2

    while True:
        boundaries, _ = counts(lines)
        try:
            entry = input(f"[{len(lines)} lines, {boundaries} boundaries] > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        entry = entry.strip()

        if not entry:
            continue
        if entry == ".help":
            print(HELP)
            continue
        if entry == ".stats":
            b, same = counts(lines)
            print(f"  {len(lines)} lines, {b} boundaries, {same} continuations")
            continue
        if entry == ".undo":
            if lines:
                dropped = lines.pop()
                speaker = lines[-1][0] if lines else 2
                print(f"  removed: SPEAKER{dropped[0]}: {dropped[1]}")
                out.write_text(render(lines), encoding="utf-8")
            continue
        if entry == ".done":
            break

        if entry.startswith("="):
            text = entry[1:].strip()
            if not text:
                continue
            # Same speaker again: they paused and kept going. This is the
            # observation the whole benchmark turns on.
            lines.append((speaker, text))
        else:
            speaker = 2 if speaker == 1 else 1
            lines.append((speaker, entry))

        # Written every line. A labelling session is an hour of somebody's
        # attention and must never be lost to a crash or a closed terminal.
        out.write_text(render(lines), encoding="utf-8")

    if not lines:
        print("nothing recorded")
        return 1

    sha, size = digest(audio)
    boundaries, same = counts(lines)
    manifest = {
        "transcript": out.name,
        "language": args.language,
        "labeller": args.labeller,
        "audio_sha256": sha,
        "audio_bytes": size,
        "audio_filename": audio.name,
        "lines": len(lines),
        "turn_boundaries": boundaries,
        "same_speaker_continuations": same,
        "labelling_seconds": round(time.time() - started, 1),
        "note": "Audio is referenced by digest and deliberately not stored here.",
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {out} ({len(lines)} lines)")
    print(f"wrote {manifest_path}")
    print(f"{boundaries} turn boundaries, {same} same-speaker continuations")
    if same == 0:
        print(
            "\nNo same-speaker continuations recorded. If every pause in this "
            "recording really did end a turn then fine, but that is unusual -- "
            "and those pauses are the cases this project exists to get right."
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Prove a transcript is benchmark-ready, using the benchmark's own loader."""
    path = Path(args.transcript)
    if not path.is_file():
        print(f"no such file: {path}")
        return 2

    turns = load_transcripts(str(path))
    if not turns:
        print(f"{path.name}: no turns parsed -- are the lines 'SPEAKERn: text'?")
        return 1

    boundaries = sum(1 for t in turns if t.ends_turn)
    speakers = sorted({t.speaker for t in turns})
    words = sum(len(t.words) for t in turns)
    print(path.name)
    print(f"  {len(turns)} turns, {boundaries} boundaries, {words} words")
    print("  speakers: " + ", ".join(speakers))
    print(f"  same-speaker continuations: {len(turns) - boundaries - 1}")
    if len(speakers) < 2:
        print("  warn   only one speaker: there are no turn boundaries to score")
        return 1
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Label a recording for the benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="label a recording")
    new.add_argument("audio")
    new.add_argument("--out", required=True)
    new.add_argument("--language", default="", help="pack code, e.g. yo")
    new.add_argument("--labeller", default="", help="who labelled it")
    new.add_argument("--append", action="store_true", help="continue an existing file")
    new.set_defaults(func=cmd_new)

    check = sub.add_parser("check", help="verify the benchmark can read it")
    check.add_argument("transcript")
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
