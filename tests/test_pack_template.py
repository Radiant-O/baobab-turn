"""The contributor path: a template, and a validator that explains itself.

CLAUDE.md forbids writing marker lists for a language with a model, so the
only way this project reaches Yoruba, Igbo, Hausa or Swahili is a speaker
filling in a file. These cover the two things that path needs: a template
that cannot accidentally load, and a validator that tells somebody who does
not write Python what is wrong with their pack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.validate_pack import check

TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "baobab_turn" / "packs" / "community" / "_template.yaml"
)


def _write(tmp_path: Path, **overrides) -> Path:
    """A minimal pack that passes, with fields overridden per test."""
    data = {
        "code": "zz",
        "name": "Test Language",
        "status": "experimental",
        "maintainer": "A Speaker",
        "continuation_markers": [{"text": "because", "weight": 0.8}],
        "yield_markers": [{"text": "you hear", "weight": 0.9}],
        "profile_words": [f"zzword{i}" for i in range(12)],
        "validation": {"labelled_turns": 0, "dataset_note": "nothing labelled yet"},
    }
    data.update(overrides)
    path = tmp_path / "zz.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- the template ------------------------------------------------------


def test_the_template_exists_and_parses() -> None:
    assert TEMPLATE.is_file()
    assert isinstance(yaml.safe_load(TEMPLATE.read_text(encoding="utf-8")), dict)


def test_the_template_is_never_loaded_as_a_pack() -> None:
    """An underscore-prefixed file is skipped. If that ever stopped being
    true, every install would gain a language called 'Language Name'."""
    from baobab_turn.packs.registry import available_packs

    assert "xx" not in available_packs()
    assert all(p.name != "Language Name" for p in available_packs().values())


def test_the_template_covers_every_field_the_loader_reads() -> None:
    """A field the loader understands but the template omits is a feature no
    contributor will ever discover."""
    data = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    expected = {
        "code", "name", "status", "tonal", "tonal_extra_ms", "maintainer",
        "region_hint", "commonly_switches_with", "continuation_markers",
        "yield_markers", "ambiguous_markers", "fillers", "profile_words",
        "thresholds", "validation",
    }
    assert expected <= set(data)
    assert set(data["thresholds"]) == {
        "unlikely_threshold", "min_delay_ms", "max_delay_ms"
    }
    assert set(data["validation"]) == {"labelled_turns", "dataset_note"}


def test_the_template_itself_is_not_a_valid_pack() -> None:
    """It is a starting point, not a language. Saying so is the point of the
    'no markers at all' error."""
    errors, _ = check(TEMPLATE)
    assert any("no markers" in e for e in errors)


# --- the validator -----------------------------------------------------


def test_a_filled_in_pack_passes(tmp_path: Path) -> None:
    errors, warnings = check(_write(tmp_path))
    assert errors == []
    assert warnings == []


def test_the_shipped_packs_pass() -> None:
    root = Path(__file__).resolve().parent.parent / "baobab_turn" / "packs"
    for name in ("pcm.yaml", "en-NG.yaml"):
        errors, _ = check(root / name)
        assert errors == [], f"{name}: {errors}"


@pytest.mark.parametrize("field,value", [("code", "xx"), ("name", "Language Name")])
def test_placeholders_left_in_are_an_error(tmp_path: Path, field: str, value: str) -> None:
    errors, _ = check(_write(tmp_path, **{field: value}))
    assert any("placeholder" in e for e in errors)


def test_a_marker_cannot_mean_both_things(tmp_path: Path) -> None:
    """The guard resolves ties by kind, so a word in both lists does not
    crash -- it silently always loses one way. Better to refuse it."""
    errors, _ = check(
        _write(
            tmp_path,
            continuation_markers=[{"text": "abi", "weight": 0.8}],
            yield_markers=[{"text": "abi", "weight": 0.9}],
        )
    )
    assert any("cannot mean both" in e for e in errors)


def test_a_marker_that_cannot_survive_normalisation_is_an_error(tmp_path: Path) -> None:
    """Nothing at runtime would ever report this: the entry simply never
    matches, for the life of the pack."""
    errors, _ = check(_write(tmp_path, yield_markers=[{"text": "???", "weight": 0.9}]))
    assert any("empty once normalised" in e for e in errors)


def test_a_marker_needing_normalisation_is_a_warning(tmp_path: Path) -> None:
    errors, warnings = check(
        _write(tmp_path, yield_markers=[{"text": "Abi?", "weight": 0.9}])
    )
    assert errors == []
    assert any("matched as 'abi'" in w for w in warnings)


def test_validated_without_labelled_turns_is_an_error(tmp_path: Path) -> None:
    """The honesty field. A pack claiming validation with nothing labelled
    is the one lie this project cannot ship."""
    errors, _ = check(_write(tmp_path, status="validated"))
    assert any("validated" in e for e in errors)


def test_a_weak_continuation_marker_warns(tmp_path: Path) -> None:
    _, warnings = check(
        _write(tmp_path, continuation_markers=[{"text": "because", "weight": 0.2}])
    )
    assert any("not worth including" in w for w in warnings)


def test_a_low_weight_ambiguous_marker_does_not_warn(tmp_path: Path) -> None:
    """Ambiguous markers are hints and the guard halves them again. A low
    number here is care, not carelessness, and warning would push authors to
    inflate exactly the entries that should stay cautious."""
    _, warnings = check(
        _write(tmp_path, ambiguous_markers=[{"text": "sha", "weight": 0.2}])
    )
    assert not any("not worth including" in w for w in warnings)


def test_a_thin_profile_warns(tmp_path: Path) -> None:
    """Shared words are discarded as indiscriminate, so a profile can look
    populated and detect nothing."""
    _, warnings = check(_write(tmp_path, profile_words=["dey", "wetin", "abeg"]))
    assert any("distinctive profile_words" in w for w in warnings)


def test_a_malformed_file_reports_rather_than_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("code: [this is not a string]\n", encoding="utf-8")
    errors, _ = check(bad)
    assert errors and "will not load" in errors[0]
