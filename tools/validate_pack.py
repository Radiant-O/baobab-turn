"""Check a language pack before it is proposed.

    python tools/validate_pack.py baobab_turn/packs/community/yo.yaml

Errors mean the pack will not load or will behave wrongly. Warnings mean it
will load and probably do less than its author intended -- an entry that can
never match, a word that cannot discriminate, a placeholder left in from the
template.

The point is that a contributor who speaks the language but does not write
Python gets told what is wrong with their file, by the same loader the
library uses, rather than by a stack trace at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baobab_turn.core.markers import normalise
from baobab_turn.core.types import LanguagePack, Marker
from baobab_turn.packs.registry import available_packs, load_pack_file

#: Values the template ships with. A pack still carrying one was copied and
#: not filled in.
_PLACEHOLDERS = {"xx", "language name", ""}

#: Below this a continuation or yield marker is doing more harm than good:
#: it fires often enough to matter and is wrong often enough to hurt.
#:
#: Deliberately not applied to ambiguous markers. Those are hints, the guard
#: halves their weight again, and a low number there is the author being
#: careful rather than careless -- warning about it would push contributors
#: to inflate exactly the entries that should stay cautious.
_WEAK_WEIGHT = 0.5

#: Fewer than this and code-switch detection has too little to work with.
_THIN_PROFILE = 10


def _marker_lists(pack: LanguagePack) -> list[tuple[str, tuple[Marker, ...]]]:
    return [
        ("continuation_markers", pack.continuation_markers),
        ("yield_markers", pack.yield_markers),
        ("ambiguous_markers", pack.ambiguous_markers),
    ]


def check(path: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one pack file."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        pack = load_pack_file(path)
    except Exception as exc:
        return [f"will not load: {exc}"], []

    if pack.code.lower() in _PLACEHOLDERS:
        errors.append(f"'code' is still the template placeholder {pack.code!r}")
    if pack.name.lower() in _PLACEHOLDERS:
        errors.append(f"'name' is still the template placeholder {pack.name!r}")
    if not pack.maintainer:
        warnings.append("no 'maintainer' -- nobody can be asked about an entry")

    # An entry that cannot survive normalisation can never match the
    # transcript, and nothing at runtime will ever say so.
    seen: dict[str, str] = {}
    total = 0
    for field, markers in _marker_lists(pack):
        for m in markers:
            total += 1
            norm = normalise(m.text)
            if not norm:
                errors.append(f"{field}: {m.text!r} is empty once normalised")
                continue
            if norm != m.text:
                warnings.append(
                    f"{field}: {m.text!r} is matched as {norm!r} "
                    "-- write it that way to keep the file honest"
                )
            if norm in seen and seen[norm] != field:
                errors.append(
                    f"{norm!r} is in both {seen[norm]} and {field}: "
                    "a marker cannot mean both 'not finished' and 'finished'"
                )
            seen[norm] = field
            if field != "ambiguous_markers" and m.weight < _WEAK_WEIGHT:
                warnings.append(
                    f"{field}: {m.text!r} has weight {m.weight} -- "
                    "below 0.5 it is probably not worth including"
                )

    if total == 0:
        errors.append("no markers at all: this pack would do nothing")

    # Shared words are discarded as indiscriminate, so a profile made only
    # of them detects nothing while looking populated.
    others = {
        w
        for code, other in available_packs().items()
        if code != pack.code
        for w in other.profile_words
    }
    distinctive = [w for w in pack.profile_words if normalise(w) not in others]
    if not pack.profile_words:
        warnings.append(
            "no 'profile_words' -- nothing can detect a switch *into* this "
            "language. Only the unmarked language of a pair can skip these."
        )
    elif len(distinctive) < _THIN_PROFILE:
        warnings.append(
            f"only {len(distinctive)} distinctive profile_words "
            f"(of {len(pack.profile_words)}); shared words are discarded, "
            f"so aim for {_THIN_PROFILE}+ that other packs do not have"
        )

    if pack.status == "validated" and pack.labelled_turns == 0:
        errors.append("status 'validated' with 0 labelled_turns")
    if pack.labelled_turns == 0 and not pack.dataset_note:
        warnings.append("0 labelled_turns and no 'dataset_note' saying how it was built")

    return errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2

    path = Path(argv[0])
    if not path.is_file():
        print(f"no such file: {path}")
        return 2

    errors, warnings = check(path)
    for w in warnings:
        print(f"  warn   {w}")
    for e in errors:
        print(f"  ERROR  {e}")

    if errors:
        print(f"\n{path.name}: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"\n{path.name}: ok ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
