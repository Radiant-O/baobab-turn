# baobab-turn

**Turn detection for African speech.** Framework-agnostic core, thin adapters,
contributable language packs. MIT licensed.

> **Status: pre-alpha, nothing works yet.** No benchmark has been run. There
> are no numbers on this page because there are no numbers. When there are,
> they go at the top, before the pitch.

## Benchmark

<!-- SPEC 11: this table leads the README. It stays empty until M1 produces
     a real measurement. Do not fill it with estimates or illustrative
     figures. An empty table is honest; a plausible-looking one is not. -->

_Empty until M1 completes. See `docs/SPEC.md` §7 for the methodology and §9
for what M1 is._

## The problem

Voice agents decide when the user has stopped talking. Get it wrong and the
agent talks over the caller — the most damaging failure mode in a voice agent,
because the caller feels disrespected and hangs up.

Every production turn-detection model is trained on Western speech. LiveKit's
audio turn detector covers 14 languages, none of them Nigerian, and falls back
to English thresholds when it detects no language. Tonal languages have pitch
contours that a prosody-reading detector mistakes for the end of a sentence;
code-switching breaks its language assumptions mid-utterance; and Nigerian
English and Pidgin mark turn boundaries with words it has never seen used that
way.

**This is not a transcription problem.** Nigerian ASR is largely solved — the
words come back correct and the agent still interrupts. That gap is what this
package aims to close.

Whether any of that is actually true is an open question, and answering it
with numbers is the first milestone. If the benchmark shows the stock detector
already handles Nigerian speech well, that is a publishable result and this
project stops.

## Scope

The engine is pan-African by *architecture*. Validated coverage is much
narrower and the two must never be blurred.

**Planned v1 validated coverage: Nigerian English (`en-NG`) and Nigerian
Pidgin (`pcm`).** Yoruba, Igbo and Hausa will ship as community packs —
opt-in, off by default — until a speaker of each writes the marker lists and
labels benchmark data. Other African languages are supported by the language
pack mechanism, not by our testing. See `docs/FINDINGS.md` D6.

## Install

Not published yet. Nothing to install.

## Language packs

A language pack is a YAML data file — no Python, no code execution — so that a
speaker of a language can contribute one without touching the engine.

Validated and community packs are listed in **separate tables**, always, with
each pack's labelled-turn count shown. A count of `0` is displayed as `0`.

Marker lists are never machine-translated. Discourse markers do not translate
word for word, and a wrong list is worse than no list.

_`LANGUAGE_PACKS.md` arrives at M5. Contributing a language before the engine
is proven wastes contributor goodwill._

## Repository layout

```
baobab_turn/core/      pure Python rules engine, no framework imports (CI-enforced)
baobab_turn/adapters/  thin framework glue, ~150-250 lines each
baobab_turn/packs/     language packs, data only
bench/                 the measuring tape -- build this first
tools/label.py         CLI for labelling true end-of-turn in clips
docs/SPEC.md           the full engineering specification
docs/FINDINGS.md       verified SDK constraints, corrections to the spec
```

## Principles

- **Measure before you build.** The benchmark comes before the engine.
- **Fail open, always.** A turn detector that breaks a call is worse than one
  that is merely mediocre.
- **Publish what didn't work.** An ablation table showing a feature made no
  difference is more credible than one where everything helped.
- **Small honest datasets beat large vague claims.**
- **Delete features that don't move the numbers.** Every config flag is a
  hypothesis under test, not a commitment.

## License

MIT
