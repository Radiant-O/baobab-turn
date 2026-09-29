"""Pack discovery and language-code resolution.

The only code allowed in this directory; everything else here is data.

Two rules that must not bend:

**A malformed pack never crashes an agent.** It logs a warning and is
skipped. Someone's YAML typo must not take down a live call.

**An unknown language degrades to stock behaviour.** When nothing matches we
return the null pack, which has no markers, so the guard applies no lexical
rules and the inner detector's decision passes through. Unknown must never
mean worse-than-stock.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from ..core.types import LanguagePack, Marker

__all__ = ["available_packs", "load_pack_file", "resolve"]

log = logging.getLogger(__name__)

_PACKS_DIR = Path(__file__).parent
_COMMUNITY_DIR = _PACKS_DIR / "community"
_VALID_STATUS = {"validated", "community", "experimental"}


def _markers(raw: object, where: str) -> tuple[Marker, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{where}: expected a list, got {type(raw).__name__}")
    out: list[Marker] = []
    for entry in raw:
        if not isinstance(entry, dict) or "text" not in entry:
            raise ValueError(f"{where}: each marker needs a 'text' field")
        out.append(
            Marker(
                text=str(entry["text"]),
                weight=float(entry.get("weight", 0.5)),
                note=str(entry.get("note", "")),
            )
        )
    return tuple(out)


def _strs(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(x) for x in raw)


def load_pack_file(path: Path) -> LanguagePack:
    """Parse one pack. Raises ValueError on anything malformed."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: top level must be a mapping")

    code = data.get("code")
    if not code or not isinstance(code, str):
        raise ValueError(f"{path.name}: missing required field 'code'")

    status = str(data.get("status", "experimental"))
    if status not in _VALID_STATUS:
        raise ValueError(
            f"{path.name}: status must be one of {sorted(_VALID_STATUS)}, got {status!r}"
        )

    thresholds = data.get("thresholds") or {}
    validation = data.get("validation") or {}
    if not isinstance(thresholds, dict) or not isinstance(validation, dict):
        raise ValueError(f"{path.name}: 'thresholds' and 'validation' must be mappings")

    def opt_float(key: str) -> float | None:
        v = thresholds.get(key)
        return None if v is None else float(v)

    def opt_int(key: str) -> int | None:
        v = thresholds.get(key)
        return None if v is None else int(v)

    return LanguagePack(
        code=code,
        name=str(data.get("name", code)),
        status=status,
        tonal=bool(data.get("tonal", False)),
        tonal_extra_ms=int(data.get("tonal_extra_ms", 0) or 0),
        maintainer=str(data.get("maintainer", "") or ""),
        region_hint=_strs(data.get("region_hint")),
        commonly_switches_with=_strs(data.get("commonly_switches_with")),
        continuation_markers=_markers(
            data.get("continuation_markers"), f"{path.name}:continuation_markers"
        ),
        yield_markers=_markers(data.get("yield_markers"), f"{path.name}:yield_markers"),
        ambiguous_markers=_markers(
            data.get("ambiguous_markers"), f"{path.name}:ambiguous_markers"
        ),
        fillers=_strs(data.get("fillers")),
        profile_words=_strs(data.get("profile_words")),
        unlikely_threshold=opt_float("unlikely_threshold"),
        min_delay_ms=opt_int("min_delay_ms"),
        max_delay_ms=opt_int("max_delay_ms"),
        labelled_turns=int(validation.get("labelled_turns", 0) or 0),
        dataset_note=str(validation.get("dataset_note", "") or ""),
    )


@lru_cache(maxsize=1)
def available_packs() -> dict[str, LanguagePack]:
    """Every pack that loaded cleanly, keyed by language code.

    Community packs are included here so they can be listed, but `resolve`
    will not hand one back unless the caller opts in.
    """
    packs: dict[str, LanguagePack] = {}
    for directory in (_PACKS_DIR, _COMMUNITY_DIR):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            if path.name.startswith("_"):
                continue  # _template.yaml is a starting point, not a pack
            try:
                pack = load_pack_file(path)
            except Exception as exc:
                log.warning("skipping malformed language pack %s: %s", path.name, exc)
                continue
            if pack.code in packs:
                log.warning("duplicate language code %r in %s", pack.code, path.name)
                continue
            packs[pack.code] = pack
    return packs


def resolve(language: str | None, *, allow_community: bool = False) -> LanguagePack | None:
    """Find the pack for a language code.

    Resolution order is exact code, then base code, then nothing:
    `en-NG` -> `en-NG`; `en-US` -> `en` if such a pack exists; otherwise None,
    which the guard treats as the null pack.

    Returns None rather than raising, and never returns a community pack
    unless `allow_community` is set -- community packs are written by native
    speakers but not benchmarked, so they must be opted into.
    """
    if not language:
        return None

    packs = available_packs()
    code = language.strip()
    if not code or code.lower() in {"und", "auto", "multi"}:
        return None

    for candidate in (code, code.split("-")[0]):
        pack = packs.get(candidate)
        if pack is None:
            continue
        if pack.status == "validated" or pack.status == "experimental":
            return pack
        if allow_community:
            return pack
        log.debug("pack %r is community-status and not enabled", pack.code)
        return None
    return None
