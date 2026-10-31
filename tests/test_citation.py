"""The citation file, and the one way it rots.

A DOI is only useful if what it points at is accurate. The failure mode is
not a malformed file -- the schema catches that -- it is a version number
that quietly stops matching the code, so every citation of this software
names a release that is not the one being described.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CITATION = ROOT / "CITATION.cff"

#: Zenodo mints a version DOI per release and one concept DOI that always
#: resolves to the newest. The concept DOI is the one to publish, so it is
#: the one pinned here.
CONCEPT_DOI = "10.5281/zenodo.23822193"


def _cff() -> dict:
    return yaml.safe_load(CITATION.read_text(encoding="utf-8"))


def _declared_version() -> str:
    text = (ROOT / "baobab_turn" / "version.py").read_text(encoding="utf-8")
    return text.split('"')[1]


def test_it_exists_and_parses() -> None:
    assert CITATION.is_file()
    assert isinstance(_cff(), dict)


def test_version_matches_the_package() -> None:
    """The drift that makes a citation wrong. Bumping version.py without
    this file means every citation names the wrong release."""
    assert str(_cff()["version"]) == _declared_version()


def test_the_concept_doi_is_used_not_a_version_doi() -> None:
    """A version DOI here would freeze every citation at 0.1.1 forever,
    including citations written years later."""
    cff = _cff()
    assert str(cff["doi"]) == CONCEPT_DOI
    assert [str(i["value"]) for i in cff["identifiers"]] == [CONCEPT_DOI]


def test_required_fields_are_present() -> None:
    cff = _cff()
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["message"]
    assert cff["authors"] and cff["authors"][0]["family-names"]


def test_licence_agrees_with_the_repo() -> None:
    assert _cff()["license"] == "MIT"
    assert "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
