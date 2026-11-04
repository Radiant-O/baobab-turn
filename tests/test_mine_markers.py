"""Reading a corpus whose conventions fight the tokeniser.

Every fixture here is invented, written in SBCSAE's *format* but never its
content. That is deliberate twice over: the corpus is CC BY-ND 3.0 US, so a
copy of its text has no business in an MIT repository, and a parser is
better tested against cases chosen to break it than against whatever the
first file happens to contain.

The case that matters most is `test_an_overlap_bracket_inside_a_word`.
SBCSAE puts a bracket at the instant two people start talking over each
other, which lands mid-word far more often than not. Treating the bracket as
a word boundary splits `righ[t]` into `righ` and `t`, and those fragments
are common enough to reach the top of a candidate list -- which is to say
the failure looks like a finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.mine_markers import (
    clean_sbcsae,
    expand,
    is_sound,
    is_unidentified,
    parse_sbcsae,
    tokenise_sbcsae,
)


def write(tmp_path: Path, body: str, name: str = "SBC001.trn") -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


# --- layout ------------------------------------------------------------


def test_the_two_column_layout(tmp_path: Path) -> None:
    utts = parse_sbcsae(write(tmp_path, "0.00 1.50\tADA:\tgo on then\n"))
    assert len(utts) == 1
    assert utts[0].speaker == "ADA"
    assert utts[0].words == ["go", "on", "then"]


def test_a_blank_speaker_column_continues_the_same_speaker(tmp_path: Path) -> None:
    """The fact the whole cross-turn measure rests on. Without it every
    intonation unit would look like a separate turn."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "0.00 1.50\tADA:\tso i told him\n"
            "1.50 2.10\t    \tand he just left\n"
            "2.10 3.00\tBEN:\treally\n",
        )
    )
    assert [u.speaker for u in utts] == ["ADA", "ADA", "BEN"]


def test_the_speaker_may_share_a_column_with_the_text(tmp_path: Path) -> None:
    """Some files pad the label with spaces instead of closing the column
    with a tab. Same corpus, same release, both layouts."""
    utts = parse_sbcsae(write(tmp_path, "82.28 83.93\tADA:      yes exactly\n"))
    assert utts[0].speaker == "ADA"
    assert utts[0].words == ["yes", "exactly"]


def test_the_time_marks_may_be_tab_separated(tmp_path: Path) -> None:
    """Space in the early files, tab in the later ones."""
    utts = parse_sbcsae(write(tmp_path, "595.320\t596.855\tADA:\tthey were ready\n"))
    assert utts[0].words == ["they", "were", "ready"]


def test_annotator_queries_are_not_speech(tmp_path: Path) -> None:
    """Written against a `000000000` pseudo-time. Requiring a real decimal
    time is what filters them out."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "000000000 000000000 $ COMMA OR PERIOD?\n"
            "0.00 1.50\tADA:\tthe only real line\n",
        )
    )
    assert len(utts) == 1
    assert utts[0].words == ["the", "only", "real", "line"]


# --- who counts as a speaker -------------------------------------------


@pytest.mark.parametrize("label", [">ENV", ">DOG", ">RADIO", "ENV"])
def test_noise_is_not_a_turn(label: str) -> None:
    """A dog barking does not end anybody's turn."""
    assert is_sound(label)
    assert not is_unidentified(label)


@pytest.mark.parametrize("label", ["X", "X_2", "*X"])
def test_an_unidentified_voice_is_a_speaker_we_cannot_name(label: str) -> None:
    """Different from noise: somebody really did speak."""
    assert is_unidentified(label)
    assert not is_sound(label)


@pytest.mark.parametrize("label", ["ADA", "#READ", "MANY", "AUD_2", "NORA?", "TOM_1"])
def test_real_speakers_including_the_awkward_ones(label: str) -> None:
    assert not is_sound(label)
    assert not is_unidentified(label)


def test_a_non_speaker_takes_its_continuation_lines_with_it(tmp_path: Path) -> None:
    """A blank column inherits the line above, so dropping the environment
    must drop what continues it rather than handing those words to whoever
    spoke last."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "0.00 1.00\tADA:\tlisten to this\n"
            "1.00 2.00\t>ENV:\t((DOOR_SLAMS))\n"
            "2.00 3.00\t     \t((FOOTSTEPS_FADE))\n"
            "3.00 4.00\tBEN:\tanyway\n",
        )
    )
    assert [u.speaker for u in utts] == ["ADA", "BEN"]


def test_an_uncertain_attribution_keeps_the_speaker(tmp_path: Path) -> None:
    utts = parse_sbcsae(write(tmp_path, "0.00 1.00\tNORA?:\tmaybe so\n"))
    assert utts[0].speaker == "NORA"


def test_an_unidentified_voice_does_not_bridge_two_turns(tmp_path: Path) -> None:
    """Dropping the unknown line would make ADA adjacent to ADA, so `yes`
    would read as "kept the floor" when she actually yielded and took it
    back. 27 lines in the real release sit exactly here."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "0.00 1.00\tADA:\tyes\n"
            "1.00 2.00\tX:\tno\n"
            "2.00 3.00\tADA:\tright\n",
        )
    )
    assert [u.speaker for u in utts] == ["ADA", "?1", "ADA"]


def test_one_run_of_unknown_lines_is_one_unknown_speaker(tmp_path: Path) -> None:
    """Two consecutive unidentified lines may be two different people, so a
    boundary between them would be invented. A run is treated as one turn."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "0.00 1.00\tADA:\tgo on\n"
            "1.00 2.00\tX:\twell\n"
            "2.00 3.00\tX_2:\tmaybe\n"
            "3.00 4.00\tADA:\tfine\n"
            "4.00 5.00\tX:\tagain\n",
        )
    )
    assert [u.speaker for u in utts] == ["ADA", "?1", "?1", "ADA", "?2"]


def test_noise_does_not_interrupt_an_unknown_run(tmp_path: Path) -> None:
    """Noise is not a turn, so it cannot split one unknown speaker into
    two."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "0.00 1.00\tX:\tsomething\n"
            "1.00 2.00\t>ENV:\t((DOOR))\n"
            "2.00 3.00\tX:\tand more\n",
        )
    )
    assert [u.speaker for u in utts] == ["?1", "?1"]


def test_an_indented_label_is_still_a_label(tmp_path: Path) -> None:
    """Four labels in the real release carry a leading space. Reading one as
    a continuation would weld two people's turns into one."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "0.00 1.00\tADA:\tgo on\n" "1.00 2.00\t BRETT:\texcuse me\n",
        )
    )
    assert [u.speaker for u in utts] == ["ADA", "BRETT"]


def test_a_mixed_case_label_is_still_a_label(tmp_path: Path) -> None:
    """`Doris:`, `Mary:` and `Steven:` appear in mixed case."""
    utts = parse_sbcsae(write(tmp_path, "0.00 1.00\tDoris:\tyeah\n"))
    assert utts[0].speaker == "DORIS"


def test_a_long_label_is_not_silently_demoted(tmp_path: Path) -> None:
    """The failure mode is quiet: an unmatched label reads as a continuation
    of whoever spoke last, merging two people."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "0.00 1.00\tADA:\thello\n" "1.00 2.00\tCHRISTOPHERS:\thi\n",
        )
    )
    assert [u.speaker for u in utts] == ["ADA", "CHRISTOPHERS"]


# --- transcription conventions -----------------------------------------


def test_an_overlap_bracket_inside_a_word(tmp_path: Path) -> None:
    """The headline case. The bracket marks an instant, not a word boundary,
    so it is deleted rather than spaced."""
    assert tokenise_sbcsae(clean_sbcsae("that's righ[t].")) == ["that's", "right"]
    assert tokenise_sbcsae(clean_sbcsae("[2checked every2]thing")) == [
        "checked",
        "everything",
    ]
    assert tokenise_sbcsae(clean_sbcsae("Deli[2ciou2]s.")) == ["delicious"]


def test_a_spoken_number_keeps_its_digits() -> None:
    """The overlap index is a single digit against the bracket. A number is
    not, and eating it would silently rewrite what somebody said."""
    assert tokenise_sbcsae(clean_sbcsae("[50 dollars")) == ["50", "dollars"]
    assert tokenise_sbcsae(clean_sbcsae("about [20 people")) == ["about", "20", "people"]


def test_lengthening_does_not_split_a_word() -> None:
    """`=` marks a drawn-out vowel and sits inside the word. Spacing it
    would turn `ti=me` into `ti` and `me`."""
    assert tokenise_sbcsae(clean_sbcsae("all the ti=me")) == ["all", "the", "time"]
    assert tokenise_sbcsae(clean_sbcsae("s=o ti=red")) == ["so", "tired"]


def test_breath_and_vocal_noise_are_not_words() -> None:
    """`(H)` would otherwise tokenise to the word `h`, and it is one of the
    most frequent things in the corpus."""
    assert tokenise_sbcsae(clean_sbcsae("(H) and then")) == ["and", "then"]
    assert tokenise_sbcsae(clean_sbcsae("(Hx) (TSK) (SNIFF) right")) == ["right"]


def test_transcriber_comments_are_dropped() -> None:
    assert tokenise_sbcsae(clean_sbcsae("((AIR_CONDITIONER_FAN)) quiet now")) == [
        "quiet",
        "now",
    ]


def test_unintelligible_speech_is_dropped() -> None:
    """Any run of X, not just short ones: `XXXX` reached the candidate list
    when the pattern was bounded."""
    assert tokenise_sbcsae(clean_sbcsae("they look a green XXXX")) == [
        "they",
        "look",
        "a",
        "green",
    ]
    assert tokenise_sbcsae(clean_sbcsae("XX XXX XXXXX")) == []


def test_an_x_ray_is_not_unintelligible_speech() -> None:
    """Three in the real release. The unintelligible mark is a standalone
    run of X, so a hyphen is what tells the two apart.

    The hyphen still splits the word, as it does for every hyphenated word
    here. What matters is that the `x` survives as a spoken thing rather
    than being deleted as noise.
    """
    assert tokenise_sbcsae(clean_sbcsae("an X-ray,")) == ["an", "x", "ray"]
    assert tokenise_sbcsae(clean_sbcsae("deteriorating on X-ray")) == [
        "deteriorating",
        "on",
        "x",
        "ray",
    ]


def test_a_cut_off_word_is_not_a_word() -> None:
    """A trailing hyphen marks a word abandoned mid-production. Keeping it
    puts fragments in front of a human reviewer as if they were evidence."""
    assert tokenise_sbcsae(clean_sbcsae("cause y- --")) == ["cause"]
    assert tokenise_sbcsae(clean_sbcsae("aren't they real s-,")) == [
        "aren't",
        "they",
        "real",
    ]


def test_quoted_and_marked_passages_keep_their_words() -> None:
    """`<READ ... READ>` and `<X ... X>` wrap real speech. Only the markers
    go."""
    assert tokenise_sbcsae(clean_sbcsae("<READ wishing you a day READ>")) == [
        "wishing",
        "you",
        "a",
        "day",
    ]
    assert tokenise_sbcsae(clean_sbcsae("<X oh X> --")) == ["oh"]


def test_laughter_separates_rather_than_joins() -> None:
    assert tokenise_sbcsae(clean_sbcsae("@@@ sorry @@")) == ["sorry"]
    assert tokenise_sbcsae(clean_sbcsae("no@way")) == ["no", "way"]


def test_an_anonymised_name_keeps_the_name() -> None:
    """`~` marks a pseudonym substituted for a real one. The word is still a
    word."""
    assert tokenise_sbcsae(clean_sbcsae("~Mae ~Lynne")) == ["mae", "lynne"]


# --- file discovery ----------------------------------------------------


def test_appledouble_sidecars_are_skipped(tmp_path: Path) -> None:
    """The SBCSAE zip carries one per transcript. They parse as empty files
    rather than failing loudly, so nothing would point at them."""
    (tmp_path / "SBC001.trn").write_text("0.00 1.00\tADA:\thello\n", encoding="utf-8")
    (tmp_path / "._SBC001.trn").write_bytes(b"\x00\x05\x16\x07")
    found = expand([str(tmp_path)])
    assert [Path(f).name for f in found] == ["SBC001.trn"]


# --- the measure the tool exists for -----------------------------------


def test_a_speaker_change_is_a_boundary_and_a_repeat_is_not(tmp_path: Path) -> None:
    """End to end: the contrast the candidate lists are built from."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "0.00 1.00\tADA:\ti was saying\n"
            "1.00 2.00\t    \tthat it works\n"
            "2.00 3.00\tBEN:\tmhm\n"
            "3.00 4.00\tADA:\tgood\n",
        )
    )
    changes = [
        utts[i].words[-1]
        for i in range(len(utts) - 1)
        if utts[i + 1].speaker != utts[i].speaker
    ]
    holds = [
        utts[i].words[-1]
        for i in range(len(utts) - 1)
        if utts[i + 1].speaker == utts[i].speaker
    ]
    assert changes == ["works", "mhm"]
    assert holds == ["saying"]


def test_a_line_with_no_words_left_is_not_an_utterance(tmp_path: Path) -> None:
    """Otherwise a breath on its own line becomes a turn, and whatever
    preceded it stops looking turn-final."""
    utts = parse_sbcsae(
        write(
            tmp_path,
            "0.00 1.00\tADA:\treal words here\n"
            "1.00 2.00\t    \t(H)\n"
            "2.00 3.00\t    \t((COUGH))\n",
        )
    )
    assert len(utts) == 1
