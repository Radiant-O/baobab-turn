"""The English pack, and the reach it has that no other pack does.

`resolve()` falls back from a locale to its base code, so `en.yaml` is the
only file here that answers for a caller it was never written for. A speaker
of Kenyan or Indian English gets it without anybody shipping a pack for
them, which makes two things worth pinning: that the fallback actually
happens, and that it never shadows a more specific pack.

The rest of this file guards the pack's content against the two ways a
mined pack goes wrong -- a marker that means both things at once, and a
weight that drifted away from the evidence recorded beside it.
"""

from __future__ import annotations

import pytest

from baobab_turn.core.markers import MarkerIndex, normalise, tokenise
from baobab_turn.packs.registry import available_packs, resolve


@pytest.fixture
def en():
    pack = resolve("en")
    assert pack is not None, "en.yaml did not load"
    return pack


# --- reach -------------------------------------------------------------


@pytest.mark.parametrize(
    "language", ["en-US", "en-GB", "en-IN", "en-KE", "en-ZA", "en-AU", "en-IE"]
)
def test_every_english_locale_without_a_pack_falls_back_here(language: str) -> None:
    """The whole argument for this pack. Before it, each of these resolved
    to None and the guard applied no lexical rules at all."""
    pack = resolve(language)
    assert pack is not None
    assert pack.code == "en"


def test_nigerian_english_still_wins_on_exact_match() -> None:
    """A base-code fallback must never shadow a pack written for the
    specific locale, or adding `en` would have made en-NG callers worse."""
    pack = resolve("en-NG")
    assert pack is not None
    assert pack.code == "en-NG"


def test_a_non_english_code_is_unaffected() -> None:
    assert resolve("pcm").code == "pcm"
    assert resolve("zz-ZZ") is None


def test_the_pack_is_registered() -> None:
    assert "en" in available_packs()


# --- content ------------------------------------------------------------


def test_it_is_honest_about_not_being_validated(en) -> None:
    """Nothing has been labelled and scored against this pack. Claiming
    otherwise is the one thing a mined pack must not do."""
    assert en.status == "experimental"
    assert en.labelled_turns == 0
    assert "Santa Barbara" in en.dataset_note


def test_the_corpus_is_credited(en) -> None:
    """CC BY-ND 3.0 US requires attribution, and the note is the only place
    this pack can carry it."""
    note = en.dataset_note
    assert "CC BY-ND" in note
    assert "Du Bois" in note


def test_no_marker_means_both_things(en) -> None:
    """A word in two lists would push the probability in both directions on
    the same utterance."""
    seen: dict[str, str] = {}
    groups = {
        "continuation": en.continuation_markers,
        "yield": en.yield_markers,
        "ambiguous": en.ambiguous_markers,
    }
    for name, markers in groups.items():
        for marker in markers:
            key = normalise(marker.text)
            assert key not in seen, f"{key!r} is in both {seen.get(key)} and {name}"
            seen[key] = name


def test_weights_respect_the_threshold_the_validator_warns_at(en) -> None:
    """Below 0.5 a marker fires often enough to matter and is wrong often
    enough to hurt. Ambiguous markers are exempt by design -- the guard
    halves them again, and a low number there is caution, not carelessness.
    """
    for marker in en.continuation_markers + en.yield_markers:
        assert marker.weight >= 0.5, f"{marker.text!r} is below the weak-weight floor"
    for marker in en.ambiguous_markers:
        assert 0.0 < marker.weight < 0.5


def test_the_near_even_acknowledgements_are_ambiguous_not_yields(en) -> None:
    """`yeah` ends a turn 1182 times and continues one 1062 times. Treating
    that as a yield would cut people off mid-sentence constantly, because
    the same word opens a continuation just as readily."""
    ambiguous = {normalise(m.text) for m in en.ambiguous_markers}
    assert {"yeah", "okay", "right", "yes"} <= ambiguous
    yields = {normalise(m.text) for m in en.yield_markers}
    assert not {"yeah", "okay", "right", "yes"} & yields


def test_the_strongest_signal_is_a_backchannel(en) -> None:
    """`mhm` is the clearest thing in the corpus: a caller who says only
    that has handed the floor back and is waiting."""
    weights = {normalise(m.text): m.weight for m in en.yield_markers}
    assert weights["mhm"] == max(weights.values())


def test_storytelling_was_cut(en) -> None:
    """`said`, `says` and `goes` top the continuation list on raw counts --
    "and he goes, ..." -- but holding the floor every time somebody quotes
    another person would be wrong exactly where people talk longest."""
    everything = {
        normalise(m.text)
        for m in en.continuation_markers + en.yield_markers + en.ambiguous_markers
    }
    assert not everything & {"said", "says", "goes"}


def test_topic_nouns_were_cut(en) -> None:
    """These scored high only because these 60 conversations happened to be
    about them. Frequency cannot tell a discourse marker from a noun that
    turns landed on, which is why a speaker reviews the list."""
    everything = {
        normalise(m.text)
        for m in en.continuation_markers + en.yield_markers + en.ambiguous_markers
    }
    assert not everything & {
        "amen", "tiger", "horse", "dollars", "morning", "room", "school", "mom",
    }


# --- behaviour ----------------------------------------------------------


def test_the_phrase_is_what_matches_not_its_last_word(en) -> None:
    """"you know" carries the signal and `know` alone does not, which is why
    the bare word is deliberately absent from the pack. Longest-match-first
    is what makes the phrase win."""
    index = MarkerIndex(en)
    found = index.continuation(tokenise("i went there but you know"))
    assert found is not None
    assert found[0] == "you know"
    # The bare word was never added, so nothing matches it on its own.
    assert index.continuation(tokenise("things i know")) is None


def test_punctuation_does_not_change_the_match(en) -> None:
    """ASR punctuates unreliably, so "you know," and "you know" must be the
    same thing."""
    index = MarkerIndex(en)
    assert index.continuation(tokenise("but you know,")) == index.continuation(
        tokenise("but you know")
    )


def test_a_backchannel_reads_as_the_floor_being_handed_back(en) -> None:
    index = MarkerIndex(en)
    found = index.yields(tokenise("mhm"))
    assert found is not None and found[0] == "mhm"


def test_a_filler_holds_the_floor(en) -> None:
    """The case this whole project exists for: a pause after "and uh" is
    somebody thinking, not somebody finished."""
    index = MarkerIndex(en)
    found = index.continuation(tokenise("so i went to the place and uh"))
    assert found is not None
    assert found[0] == "and uh"


def test_an_ambiguous_token_only_counts_at_the_very_end(en) -> None:
    """`okay` as the last word is an acknowledgement; "okay so i went" is
    somebody getting started."""
    index = MarkerIndex(en)
    assert index.ambiguous(tokenise("okay")) is not None
    assert index.ambiguous(tokenise("okay so i went")) is None
