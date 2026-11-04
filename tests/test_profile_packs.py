"""Packs that identify a language without claiming to understand it.

`es` and `hi` carry no markers. That is not an unfinished pack -- it is the
only honest shape available, because nobody who speaks Spanish or Hindi has
worked on this project and writing their discourse markers from a machine's
guess is what the repository forbids.

What such a pack can still do is tell the code-switch detector that a
speaker moved into that language, and a detected switch buys patience by
itself. So the tests here are about reach rather than judgement: that the
packs load, that they identify their language, and that adding them did not
disturb the pair that already worked.
"""

from __future__ import annotations

import pytest

from baobab_turn.core.codeswitch import CodeSwitchDetector, LanguageProfile
from baobab_turn.core.markers import MarkerIndex, normalise
from baobab_turn.packs.registry import available_packs, resolve


@pytest.fixture
def detector() -> CodeSwitchDetector:
    packs = [p for p in available_packs().values() if p.status != "community"]
    return CodeSwitchDetector([LanguageProfile.from_pack(p) for p in packs])


# --- the packs ----------------------------------------------------------


@pytest.mark.parametrize("code", ["es", "hi"])
def test_a_profile_pack_loads_and_claims_nothing_about_turns(code: str) -> None:
    pack = resolve(code)
    assert pack is not None
    assert pack.profile_words, "a profile pack with no profile is nothing"
    assert not pack.continuation_markers
    assert not pack.yield_markers
    assert not pack.ambiguous_markers
    assert pack.labelled_turns == 0


@pytest.mark.parametrize(
    ("language", "expected"),
    [("es-MX", "es"), ("es-AR", "es"), ("es", "es"), ("hi-IN", "hi"), ("hi", "hi")],
)
def test_locales_fall_back_to_the_profile_pack(language: str, expected: str) -> None:
    pack = resolve(language)
    assert pack is not None and pack.code == expected


@pytest.mark.parametrize("code", ["es", "hi"])
def test_a_profile_pack_applies_no_lexical_rules(code: str) -> None:
    """With no markers the index is falsy, so the guard skips the layer
    entirely and the inner detector's decision passes through."""
    pack = resolve(code)
    assert pack is not None
    assert not MarkerIndex(pack)


def test_the_corpus_and_its_licence_are_recorded() -> None:
    assert "CC BY 4.0" in resolve("es").dataset_note
    assert "CC BY-SA 3.0" in resolve("hi").dataset_note


def test_hindi_admits_it_only_covers_devanagari() -> None:
    """Romanised Hinglish matches none of these words, and a pack that let
    somebody find that out in production would be worse than no pack."""
    note = resolve("hi").dataset_note
    assert "Devanagari" in note and "romanised" in note.lower()


def test_no_profile_word_was_eaten_by_yaml() -> None:
    """YAML 1.1 reads a bare `no` as boolean false, and the Spanish word
    `no` reached the profile as the string "false" until it was quoted. The
    pack still loaded and the list still looked populated, which is what
    makes this worth a test rather than a comment.
    """
    for pack in available_packs().values():
        for word in pack.profile_words:
            assert normalise(word) not in {"true", "false"}, (
                f"{pack.code}: {word!r} looks like an unquoted YAML boolean"
            )


# --- what the packs buy -------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("i went down to the shop and then la cosa es que", "es"),
        ("so i was telling him about the thing para los que", "es"),
        ("i was going to say that के में है की", "hi"),
        # The pair that already worked, to prove the new profiles did not
        # crowd it out. Every profile competes for the same best shift.
        ("i was going to say that wetin dey happen abeg", "pcm"),
    ],
)
def test_a_switch_into_each_language_is_detected(
    detector: CodeSwitchDetector, text: str, expected: str
) -> None:
    result = detector.detect(text)
    assert result.switched, f"no switch detected in {text!r}"
    assert result.tail_language == expected


@pytest.mark.parametrize(
    "text",
    [
        "i went down to the shop and bought some bread today",
        "well i mean you know it was really quite good",
        "so i told him about it and he said that was fine",
    ],
)
def test_plain_english_is_not_a_switch(
    detector: CodeSwitchDetector, text: str
) -> None:
    """A false positive costs more than a miss here: a switch buys a couple
    of hundred milliseconds, so firing on every other utterance is worse
    than occasionally missing one."""
    assert not detector.detect(text).switched


def test_a_shared_word_cannot_identify_anything(
    detector: CodeSwitchDetector,
) -> None:
    """Spanish `no` is also in pcm's "no be so", so it discriminates
    nothing and is dropped from both profiles. A profile that kept it would
    look larger and work worse."""
    profiles = {p.code: p.words for p in detector._profiles}
    assert "no" not in profiles["es"]
    assert "no" not in profiles["pcm"]


def test_every_profile_keeps_enough_to_be_useful(
    detector: CodeSwitchDetector,
) -> None:
    """After shared words are discarded, a profile can be left too thin to
    fire. The threshold is the same one validate_pack warns at."""
    for code in ("es", "hi"):
        words = {p.code: p.words for p in detector._profiles}[code]
        assert len(words) >= 10, f"{code} kept only {len(words)} distinctive words"
