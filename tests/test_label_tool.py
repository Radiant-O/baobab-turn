"""The labelling tool, and the loop it closes.

Every pack here says `labelled_turns: 0`. This tool is the only path from a
recording to a number that can change that, so the thing worth testing is
not the prompt -- it is that what comes out the far end is read correctly by
the benchmark's own loader, and that the audio never travels with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.corpora import load_transcripts
from tools.label import counts, digest, main, parse_existing, render


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "rec.wav"
    path.write_bytes(b"not really audio, but it hashes the same way")
    return path


def _run(audio: Path, out: Path, entries: list[str], monkeypatch, *extra: str) -> int:
    """Drive the interactive loop from a scripted list of inputs."""
    feed = iter(entries)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(feed))
    return main(["new", str(audio), "--out", str(out), *extra])


# --- the pieces --------------------------------------------------------


def test_render_and_parse_round_trip() -> None:
    lines = [(1, "mo ti so"), (1, "sugbon ko gbo"), (2, "ah")]
    assert parse_existing(render(lines)) == lines


def test_counts_separates_boundaries_from_continuations() -> None:
    """The distinction the whole benchmark rests on: a speaker change is a
    turn boundary, the same speaker again is a pause that was not one."""
    assert counts([(1, "a"), (1, "b"), (2, "c")]) == (1, 1)
    assert counts([(1, "a"), (2, "b"), (1, "c")]) == (2, 0)
    assert counts([(1, "a")]) == (0, 0)
    assert counts([]) == (0, 0)


def test_digest_matches_the_file(audio: Path) -> None:
    import hashlib

    sha, size = digest(audio)
    assert sha == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert size == len(audio.read_bytes())


# --- a session ---------------------------------------------------------


def test_a_session_writes_a_transcript_and_a_manifest(
    audio: Path, tmp_path: Path, monkeypatch
) -> None:
    out = tmp_path / "s1.txt"
    rc = _run(
        audio,
        out,
        ["mo ti so fun un", "= sugbon ko gbo", "ah i see", ".done"],
        monkeypatch,
        "--language",
        "yo",
        "--labeller",
        "a speaker",
    )
    assert rc == 0
    assert out.read_text(encoding="utf-8") == (
        "SPEAKER1: mo ti so fun un\n"
        "SPEAKER1: sugbon ko gbo\n"
        "SPEAKER2: ah i see\n"
    )

    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["language"] == "yo"
    assert manifest["labeller"] == "a speaker"
    assert manifest["turn_boundaries"] == 1
    assert manifest["same_speaker_continuations"] == 1


def test_the_manifest_names_the_audio_but_never_carries_it(
    audio: Path, tmp_path: Path, monkeypatch
) -> None:
    """Hard rule: no raw caller audio in the repo. A digest is how a result
    stays traceable to a recording that stays on somebody's machine."""
    out = tmp_path / "s1.txt"
    _run(audio, out, ["hello", ".done"], monkeypatch)

    raw = out.with_suffix(".manifest.json").read_bytes()
    assert audio.read_bytes() not in raw
    manifest = json.loads(raw)
    assert manifest["audio_sha256"] == digest(audio)[0]
    assert manifest["audio_filename"] == "rec.wav"
    # The name, not the path: a manifest should not leak somebody's home
    # directory into a repo.
    assert str(audio.parent) not in raw.decode("utf-8")


def test_the_benchmark_reads_what_this_writes(
    audio: Path, tmp_path: Path, monkeypatch
) -> None:
    """The loop this tool exists to close. Anything else can be fixed later;
    if the benchmark cannot read the output, the tool is pointless."""
    out = tmp_path / "s1.txt"
    _run(audio, out, ["first thing", "= still me", "now you", ".done"], monkeypatch)

    turns = load_transcripts(str(out))
    assert [t.speaker for t in turns] == ["SP1", "SP1", "SP2"]
    # The pause inside SPEAKER1 did not end a turn; the handover did; the
    # final turn has no evidence either way.
    assert [t.ends_turn for t in turns] == [False, True, False]


def test_every_line_is_saved_as_it_is_typed(
    audio: Path, tmp_path: Path, monkeypatch
) -> None:
    """A labelling session is an hour of somebody's attention. Losing it to a
    closed terminal would be unforgivable, so the file is written per line
    rather than at .done."""
    out = tmp_path / "s1.txt"
    seen: list[str] = []

    feed = iter(["one", "two", "three"])

    def recording_input(_prompt: str = "") -> str:
        if out.exists():
            seen.append(out.read_text(encoding="utf-8"))
        try:
            return next(feed)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr("builtins.input", recording_input)
    main(["new", str(audio), "--out", str(out)])

    # The file grew while typing, not only at the end.
    assert len(seen) >= 2
    assert seen[-1].count("\n") > seen[0].count("\n")


def test_an_interrupted_session_still_keeps_the_work(
    audio: Path, tmp_path: Path, monkeypatch
) -> None:
    out = tmp_path / "s1.txt"

    feed = iter(["one", "two"])

    def interrupting(_prompt: str = "") -> str:
        try:
            return next(feed)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr("builtins.input", interrupting)
    assert main(["new", str(audio), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").count("\n") == 2


def test_undo_removes_the_last_line(audio: Path, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "s1.txt"
    _run(audio, out, ["one", "two", ".undo", "three", ".done"], monkeypatch)
    text = out.read_text(encoding="utf-8")
    assert "two" not in text
    # Undo also rewinds whose turn it is, so "three" takes the slot "two" had.
    assert text == "SPEAKER1: one\nSPEAKER2: three\n"


def test_nothing_recorded_is_not_a_success(
    audio: Path, tmp_path: Path, monkeypatch
) -> None:
    out = tmp_path / "s1.txt"
    assert _run(audio, out, [".done"], monkeypatch) == 1
    assert not out.with_suffix(".manifest.json").exists()


def test_refuses_to_overwrite_without_append(
    audio: Path, tmp_path: Path, monkeypatch
) -> None:
    out = tmp_path / "s1.txt"
    out.write_text("SPEAKER1: already here\n", encoding="utf-8")
    assert _run(audio, out, ["new text", ".done"], monkeypatch) == 2
    assert "already here" in out.read_text(encoding="utf-8")


def test_append_continues_an_existing_session(
    audio: Path, tmp_path: Path, monkeypatch
) -> None:
    out = tmp_path / "s1.txt"
    out.write_text("SPEAKER1: first\nSPEAKER2: second\n", encoding="utf-8")
    _run(audio, out, ["third", ".done"], monkeypatch, "--append")
    assert out.read_text(encoding="utf-8") == (
        "SPEAKER1: first\nSPEAKER2: second\nSPEAKER1: third\n"
    )


def test_a_missing_recording_is_refused(tmp_path: Path) -> None:
    assert main(["new", str(tmp_path / "nope.wav"), "--out", str(tmp_path / "o.txt")]) == 2


# --- check -------------------------------------------------------------


def test_check_accepts_a_good_transcript(tmp_path: Path, capsys) -> None:
    path = tmp_path / "s1.txt"
    path.write_text("SPEAKER1: a\nSPEAKER1: b\nSPEAKER2: c\n", encoding="utf-8")
    assert main(["check", str(path)]) == 0
    assert "1 boundaries" in capsys.readouterr().out


def test_check_flags_a_single_speaker(tmp_path: Path, capsys) -> None:
    """One speaker means zero turn boundaries, so the file scores nothing.
    Finding that out at benchmark time would waste the labelling."""
    path = tmp_path / "s1.txt"
    path.write_text("SPEAKER1: a\nSPEAKER1: b\n", encoding="utf-8")
    assert main(["check", str(path)]) == 1
    assert "only one speaker" in capsys.readouterr().out


def test_check_rejects_an_unparseable_file(tmp_path: Path, capsys) -> None:
    path = tmp_path / "s1.txt"
    path.write_text("just some prose with no speaker labels\n", encoding="utf-8")
    assert main(["check", str(path)]) == 1
    assert "no turns parsed" in capsys.readouterr().out


def test_check_on_a_missing_file(tmp_path: Path) -> None:
    assert main(["check", str(tmp_path / "nope.txt")]) == 2
