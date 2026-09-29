"""Corpus loading and benchmark scoring.

The scoring logic decides what number gets published, so the cases that could
silently inflate a result have their own tests: the last turn of a file is not
evidence, a proper prefix is never a turn end, and "no rule fired" must not be
counted as the lexical layer having had an opinion.
"""

from __future__ import annotations

import pytest

from baobab_turn.core.config import BaobabConfig
from baobab_turn.core.markers import MarkerIndex
from baobab_turn.packs.registry import resolve
from bench.corpora import load_transcripts
from bench.runner import DEFAULT_P_INNER, Outcome, replay, score

CONVERSATION = """\
SPEAKER1: i wan tell you say the thing be say
SPEAKER2: e don finish abi
SPEAKER1: yes
SPEAKER1: i still dey talk
SPEAKER2: okay
"""


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "conv.txt").write_text(CONVERSATION, encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def index() -> MarkerIndex:
    pack = resolve("pcm")
    assert pack is not None
    return MarkerIndex(pack)


# --- loading -----------------------------------------------------------


def test_loads_turns_in_order(corpus) -> None:
    turns = load_transcripts(corpus)
    assert [t.speaker for t in turns] == ["SP1", "SP2", "SP1", "SP1", "SP2"]
    assert turns[1].text == "e don finish abi"


def test_ends_turn_is_true_only_on_a_speaker_change(corpus) -> None:
    """Turn 3 is followed by the same speaker, so the speaker had not
    finished -- counting it as a turn end would invent evidence."""
    turns = load_transcripts(corpus)
    assert [t.ends_turn for t in turns] == [True, True, False, True, False]


def test_last_turn_of_a_file_is_not_evidence(corpus) -> None:
    """Nothing follows it, so we never observed whether the speaker was
    done."""
    turns = load_transcripts(corpus)
    assert turns[-1].ends_turn is False


@pytest.mark.parametrize(
    "line,expected",
    [
        ("SPEAKER1: abi", ["abi"]),
        ("Speaker 1:  abi", ["abi"]),
        ("SPEAKER 1:abi", ["abi"]),
        ("speaker12: abi", ["abi"]),
    ],
)
def test_speaker_label_variants(tmp_path, line, expected) -> None:
    """CENCOS uses all of these, sometimes inside one file."""
    (tmp_path / "a.txt").write_text(line + "\nSPEAKER9: x\n", encoding="utf-8")
    turns = load_transcripts(str(tmp_path))
    assert list(turns[0].words) == expected


def test_annotator_notes_and_punctuation_are_dropped(tmp_path) -> None:
    (tmp_path / "a.txt").write_text(
        "SPEAKER1: e don [laughs] finish, abi?\nSPEAKER2: x\n", encoding="utf-8"
    )
    turns = load_transcripts(str(tmp_path))
    assert list(turns[0].words) == ["e", "don", "finish", "abi"]


def test_lines_without_a_speaker_label_are_skipped(tmp_path) -> None:
    (tmp_path / "a.txt").write_text(
        "some header\n\nSPEAKER1: abi\nnot a turn\nSPEAKER2: x\n", encoding="utf-8"
    )
    assert len(load_transcripts(str(tmp_path))) == 2


def test_cp1252_files_still_load(tmp_path) -> None:
    """The corpus documents cp1252 and ships UTF-8; both must work."""
    (tmp_path / "a.txt").write_bytes(
        "SPEAKER1: café abi\nSPEAKER2: x\n".encode("cp1252")
    )
    turns = load_transcripts(str(tmp_path))
    assert "abi" in turns[0].words


# --- replay ------------------------------------------------------------


def test_replay_scores_every_prefix_and_the_end(corpus, index) -> None:
    turns = [t for t in load_transcripts(corpus) if t.ends_turn]
    first = turns[0]
    outcomes = [o for o in replay(turns, BaobabConfig(), index) if o.turn_index == 0]
    assert len(outcomes) == len(first.words)
    assert sum(o.is_turn_end for o in outcomes) == 1
    assert outcomes[-1].is_turn_end


def test_replay_skips_turns_that_were_never_observed_to_end(corpus, index) -> None:
    turns = load_transcripts(corpus)
    outcomes = replay(turns, BaobabConfig(), index)
    scored = {(o.clip_id, o.turn_index) for o in outcomes}
    assert (turns[2].clip_id, 2) not in scored   # same speaker followed
    assert (turns[4].clip_id, 4) not in scored   # last turn of the file


def test_max_prefixes_caps_long_turns(corpus, index) -> None:
    """One rambling turn must not dominate the rate.

    The cap counts prefixes; the turn end is always scored on top, since
    dropping it would lose the only positive evidence the turn carries.
    """
    turns = [t for t in load_transcripts(corpus) if t.ends_turn]
    assert len(turns[0].words) == 9, "fixture changed"

    capped = [o for o in replay(turns, BaobabConfig(), index, max_prefixes=3)
              if o.turn_index == 0]
    assert sum(not o.is_turn_end for o in capped) == 3
    assert sum(o.is_turn_end for o in capped) == 1
    assert capped[-1].is_turn_end
    # The prefixes kept are the ones nearest the end, where the evidence is.
    assert [o.prefix_len for o in capped] == [6, 7, 8, 9]


def test_continuation_marker_prevents_a_commit(corpus, index) -> None:
    """The point of the whole exercise: 'the thing be say' should stop the
    agent committing."""
    turns = [t for t in load_transcripts(corpus) if t.ends_turn]
    ends = [o for o in replay(turns, BaobabConfig(), index)
            if o.turn_index == 0 and o.is_turn_end]
    assert ends[0].p_adjusted < DEFAULT_P_INNER
    assert not ends[0].committed


# --- scoring -----------------------------------------------------------


def _outcome(**kw) -> Outcome:
    base = dict(clip_id="c", turn_index=0, prefix_len=1, is_turn_end=False,
                committed=False, p_adjusted=0.5, reasons=("no_rules_fired",))
    base.update(kw)
    return Outcome(**base)  # type: ignore[arg-type]


def test_premature_commit_counted_only_mid_turn() -> None:
    s = score([
        _outcome(is_turn_end=False, committed=True),    # premature
        _outcome(is_turn_end=True, committed=True),     # correct
    ])
    assert s.premature_commits == 1
    assert s.premature_commit_rate == 1.0
    assert s.missed_ends == 0


def test_missed_end_counted_only_at_turn_ends() -> None:
    s = score([
        _outcome(is_turn_end=True, committed=False),    # missed
        _outcome(is_turn_end=False, committed=False),   # correct
    ])
    assert s.missed_ends == 1
    assert s.missed_end_rate == 1.0
    assert s.premature_commits == 0


def test_no_rules_fired_is_not_coverage() -> None:
    """Counting a silent layer as covered would make an empty pack look like
    a complete one."""
    s = score([_outcome(reasons=("no_rules_fired",)) for _ in range(10)])
    assert s.coverage == 0.0


@pytest.mark.parametrize("reason", ["tail_stale", "tail_absent"])
def test_abstention_is_not_coverage(reason) -> None:
    s = score([_outcome(reasons=(reason,)) for _ in range(5)])
    assert s.coverage == 0.0


def test_directional_accuracy_uses_the_right_sign_per_position() -> None:
    """Mid-turn the probability should fall; at a turn end it should rise."""
    s = score([
        _outcome(is_turn_end=False, p_adjusted=0.2, reasons=("continuation_veto:x",)),
        _outcome(is_turn_end=True, p_adjusted=0.8, reasons=("yield_boost:y",)),
        _outcome(is_turn_end=False, p_adjusted=0.9, reasons=("yield_boost:z",)),
    ])
    assert s.right_direction == 2
    assert s.wrong_direction == 1
    assert s.directional_accuracy == pytest.approx(2 / 3)


def test_reason_counts_strip_the_matched_phrase() -> None:
    s = score([
        _outcome(reasons=("continuation_veto:the thing be say",)),
        _outcome(reasons=("continuation_veto:wey",)),
    ])
    assert s.reason_counts["continuation_veto"] == 2


def test_empty_input_is_safe() -> None:
    s = score([])
    assert s.n_turns == 0
    assert s.premature_commit_rate == 0.0
    assert s.coverage == 0.0
