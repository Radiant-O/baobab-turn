# Changelog

Notable changes per release. Dates are the release date.

This file starts at 0.1.2. Earlier releases are described by their git tags
and the commits between them.

## 0.1.2 — 2026-11-04

The release where the project stopped being only about Nigerian speech, and
gained the adapters to be used somewhere other than LiveKit.

### Added

- **Pipecat adapter.** `BaobabTurnAnalyzer` wraps an inner analyzer, plus
  `BaobabTranscriptObserver`. Needs Pipecat 1.15.0 or later, which is where
  the turn analyzer moved off `TransportParams` onto the user-turn stop
  strategy. Install with `pip install "baobab-turn[pipecat]"`.
- **Ultravox configuration helper.** `vad_settings(language)` turns a
  language pack into a `turnEndpointDelay`. This is deliberately *not* a
  guard: hosted Ultravox decides turns server-side and the protocol gives a
  client no way to veto, delay or replace that decision, so a pack can
  contribute one number and the module says so plainly rather than looking
  wired and doing nothing.
- **`en` language pack.** The widest-reaching pack here, because a locale
  falls back to its base code: `en-GB`, `en-IN`, `en-KE` and `en-AU` all
  resolve to it, while `en-NG` still wins on exact match. Markers were mined
  from 60 recorded conversations in the Santa Barbara Corpus and then cut by
  a speaker; weights are measured lift over a 35.7% base rate, and every
  entry carries the counts it came from.
- **`es` and `hi` profile packs.** No markers — nobody who speaks Spanish or
  Hindi has worked on this, and writing their discourse markers from a
  machine's guess is forbidden here. They carry `profile_words` instead,
  which is enough for the code-switch detector to notice a speaker moving
  into that language. `hi` is Devanagari only and says so: romanised
  Hinglish matches none of it.
- **Pacing and code-switching in the chat adapter**, where the settle time
  rather than the delay is what decides whether to answer.
- **`BaobabConfig.patient()`** for speakers the detector will rush. The one
  preset that is about nobody's language.
- **Corpus tooling.** `tools/mine_markers.py` reads SBCSAE `.trn` files,
  ranks multi-word endings (`--phrases`), emits a language profile
  (`--profile`), and reports inter-turn silence (`--gaps`).
  `tools/label.py` turns a recording into turns the benchmark can read, and
  `tools/validate_pack.py` checks a pack before it ships.
- **`CITATION.cff` and a Zenodo DOI**, so the work can be cited.
- **A community pack template**, so a speaker can contribute without
  touching Python.

### Changed

- **Language resolution now walks every subtag** from most specific to
  least, rather than trying only the full code and the bare base code.
  `en-NG-x-private` used to fall past `en-NG` to a base code that did not
  exist; once an `en` pack existed it would have landed there instead,
  quietly dropping every Nigerian marker.
- The package summary and README now describe who this is for — speakers a
  detector was not trained on — rather than African speech alone. Nigerian
  speech remains the proving ground.
- `build` is listed in the `dev` extra. It was missing, so
  `pip install -e ".[dev]"` installed everything except the one tool the
  wheel check needs, and that check had been failing on clean checkouts.

### Fixed

- `validate_pack.py` now errors on a profile word that YAML read as a
  boolean. A bare `no` becomes false and reaches the profile as `"false"`,
  and the pack still loads looking populated — `yes`, `on` and `off` are
  ordinary vocabulary in some language, so this was waiting for exactly the
  contributors the project wants.
- Author metadata is a handle rather than a legal name.

### Note on an earlier claim

A figure of 47% was quoted for overlapping turn transitions in SBCSAE. It
was measured with a throwaway script and is wrong. Measured through the
parser: 43% overlap, and a further 41% sit at exactly 0.000s, which is a
transcription convention rather than a measured silence. Only about 16% of
transitions have a positive gap at all. `--gaps` reproduces this.

### Still not validated

Every pack reports `labelled_turns: 0`. No conversation has been labelled
and scored against any of them. Measured markers are evidence; that acting
on them improves turn-taking remains untested.

## 0.1.1

Packaging fix. The 0.1.0 sdist carried files that had no business in it;
the sdist is now an allowlist rather than an exclude list.

## 0.1.0

First release.
