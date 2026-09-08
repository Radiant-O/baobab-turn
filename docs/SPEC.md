# baobab-turn — Engineering Specification

**Turn detection for African speech. Framework-agnostic core, thin adapters, contributable language packs.**

Working name: `baobab-turn`. Rename freely; package names appear in §4.

Document version: 2.0
Audience: an AI coding agent plus a human reviewer.
Status: greenfield. Nothing is built yet.

### Scope in one paragraph

The engine is pan-African by design. The **shipped, validated coverage in v1 is Nigerian only** — Nigerian English, Pidgin, Yoruba, Igbo, Hausa. Other African languages are supported by *architecture*, through contributable language packs (§5.8), not by our own testing. This distinction must be stated plainly in the README and never blurred in marketing. Claiming untested coverage destroys the credibility that is this project's only real asset.

The core runs on any framework that exposes swappable turn detection. LiveKit and Pipecat are the two v1 targets. Vapi, Retell and similar managed platforms own their conversation loop and cannot accept a custom detector — they are permanently out of scope (§2).

---

## 0. How to use this document

You are building a Python package. This spec tells you **what to build, why, and how to know it works.** It does not give you copy-paste code, because the LiveKit Agents API changes frequently and stale code is worse than no code.

**Before you write anything:**

1. Connect to LiveKit's documentation MCP server for coding agents: <https://docs.livekit.io/mcp/>. Use it as your source of truth for every API call.
2. Read the current turn detection docs: <https://docs.livekit.io/agents/logic/turns/> and <https://docs.livekit.io/agents/logic/turns/turn-detector/>
3. Clone and read the SDK source: <https://github.com/livekit/agents>. You need the actual turn-detection protocol definition (see §5.1).
4. Read the existing benchmark harness: <https://github.com/livekit/eot-bench>

Anything in this document marked **[VERIFY]** is my best understanding but must be checked against the live SDK before you depend on it.

---

## 1. The problem

Voice agents decide when the user has stopped talking. Get it wrong and the agent talks over the caller. This is the single most damaging failure in a voice agent — worse than a wrong answer, because the caller feels disrespected and hangs up.

Every production turn-detection model is trained on Western speech. LiveKit's audio turn detector supports 14 languages: English, Arabic, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Dutch, Portuguese, Turkish, Chinese. No Nigerian language. No Pidgin. When no language is detected, it applies **English thresholds by default**.

Four things break for Nigerian callers. **Three of the four generalise across Africa** — which is what makes the engine pan-African rather than a Nigerian hack:

**1. Tonal interference. [PAN-AFRICAN]** Yoruba and Igbo are tonal — pitch carries lexical meaning. LiveKit's audio detector explicitly uses "acoustic cues like intonation, pitch, and rhythm" to decide the turn is over. A falling pitch contour that means "this is a low-tone syllable" gets read as "this sentence has ended." Nigerian English carries a lot of this prosody too.

This is the strongest and most portable hypothesis. Tonal languages are spread across the continent — Yoruba, Igbo, Zulu, Xhosa, Shona, Twi/Akan, Somali and many more. A pitch-reading detector has the same conflict with all of them. **The `tonal: true` flag in a language pack is therefore the single highest-value field in the whole system.**

**2. Code-switching. [PAN-AFRICAN]** Callers switch between a colonial language and one or more local languages mid-sentence. Nigeria: English / Pidgin / Yoruba / Igbo / Hausa. Kenya: English / Swahili / Sheng. South Africa: English / Zulu / Afrikaans. Every switch point is a moment where the model's language assumptions break. The detection mechanism (§5.3) is generic; only the language profiles differ.

**3. Lexical turn cues that don't exist in the training data. [LANGUAGE-SPECIFIC]** Nigerian English and Pidgin mark turn boundaries with words the model has never seen used that way — trailing *o*, *abi*, *sha*, *you get?*, *no be so?*. Conversely, some markers signal *more is coming* (*the thing be say...*, *as e be...*) and a Western model has no idea.

**This is the one component that does not port.** Swahili, Amharic, Twi and Wolof each need their own marker lists, written by someone who speaks the language. Hence language packs (§5.8). Do not machine-translate marker lists — discourse markers are not translatable word-for-word, and a wrong list is worse than no list.

**4. Pacing. [PARTLY PORTABLE]** Different rhythm, different pause lengths, different filler sounds (*ehn*, *ehen*, *hmm*). The adaptation mechanism is generic; the baseline expectations are per-language.

**None of this is a transcription problem.** Intron, Spitch, N-ATLAS and SBPN have largely solved Nigerian ASR. The words come back correct. The agent still interrupts. That gap is what this package closes.

### 1.1 Honest framing of the linguistic claims

The four claims above are **hypotheses to be validated by measurement**, not established findings. The whole point of §7 (the benchmark harness) is to prove or disprove them with numbers before building on them. If the benchmark shows LiveKit's detector already handles Nigerian speech fine, that is a valid and publishable result, and this project stops. Build the measurement first.

---

## 2. Goals and non-goals

### Goals

- Reduce false interruptions of African speakers by a voice agent, measurably.
- Keep the decision engine **framework-agnostic** — pure Python, no framework imports (§3.1).
- Ship thin adapters for LiveKit and Pipecat, each installable as an optional extra.
- Make languages **contributable** via data-only packs, so speakers of a language can add it without writing Python (§5.8).
- Add near-zero latency (rule layer p95 under 15ms).
- Produce a reproducible benchmark that anyone can run against their own agent and their own language.
- Be honest and open: MIT licensed, public numbers, published methodology, explicit statement of what is untested.

### Non-goals (do not build these)

- **Not** a new ASR model. African ASR is being solved by funded teams with datasets we can't match.
- **Not** a TTS model.
- **Not** a hosted service, dashboard, or SaaS product. No auth, no billing, no web UI.
- **Not** a Vapi/Retell/Bland integration. Those platforms own their conversation loop and do not accept custom turn detectors. This is a structural limitation, not a missing feature — do not attempt workarounds.
- **Not** a full replacement for the host framework's detector in v1. We wrap and correct it (see §3).
- **Not** shipping language packs we cannot validate. An unvalidated pack must never be merged to `main` as though it were tested (§5.8.3).

---

## 3. Core architecture decision

**We do not train a turn-detection model from scratch. We wrap an existing detector and veto its bad decisions.**

Reasoning: LiveKit's audio detector is good at general acoustics and gets a lot right. Our value is the Nigerian-specific corrections layered on top. A wrapper ships in weeks; a from-scratch model takes a year and a dataset we don't have yet.

```
              user audio
                  │
                  ▼
         ┌─────────────────┐
         │   Silero VAD    │  (speech / no speech)
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Inner detector │  (LiveKit TurnDetector, or any other)
         │  → probability  │
         └────────┬────────┘
                  │  p(end of turn)
                  ▼
         ┌─────────────────────────────┐
         │      GUARD LAYER       │   ← this is what we build
         │  • continuation veto        │
         │  • yield-marker boost       │
         │  • code-switch hold         │
         │  • adaptive pacing          │
         └────────┬────────────────────┘
                  │  adjusted p + delay override
                  ▼
            AgentSession commits or waits
```

The guard layer is **composable and fail-open**. If it raises, times out, or has no opinion, the inner detector's decision passes through untouched. The agent must never break because our layer broke.

### 3.1 The core/adapter rule — enforce this from commit one

The engine must not know what framework it is running inside.

```
baobab_turn/core/     ← pure Python. NO framework imports. Ever.
                        Input:  transcript tail, p_inner, language, session stats, config
                        Output: p_adjusted, delay_override_ms, reasons

baobab_turn/adapters/ ← thin. ~50-100 lines each. Framework-specific glue only.
                        livekit.py  → implements LiveKit's turn-detection protocol
                        pipecat.py  → implements Pipecat's equivalent
```

**Enforcement:** add a CI check that fails if any module under `core/` imports `livekit`, `pipecat`, or any other framework. This is a one-line grep in the test suite and it prevents the single most likely architectural rot.

An adapter's only jobs are: (a) satisfy the host framework's protocol, (b) translate the framework's inputs into the core's input shape, (c) translate the core's output back. No rules, no lexicon, no decisions in adapter code.

**Why this matters commercially as well as technically:** the core is what survives. LiveKit will change its API again — it already has, twice. Pipecat will change too. When they do, you rewrite 80 lines of adapter, not the whole project.

### 3.2 Framework targets

| Framework | Status | Why |
|---|---|---|
| LiveKit Agents | v1 target | Open, swappable turn detection, you already run it in production |
| Pipecat | v1 target | Open, same component-swap architecture, largest alternative community |
| Vapi / Retell / Bland | **Impossible** | Managed platforms; they own the loop and expose config knobs only |
| Custom / in-house stacks | Supported | The core is importable directly; document this path in the README |

Build the LiveKit adapter first because you can test it against real production traffic today. Build the Pipecat adapter second, purely to prove the core boundary is real — an abstraction with one implementation is not an abstraction.

---

## 4. Package and repo layout

Base package (PyPI): `baobab-turn`
Optional extras: `baobab-turn[livekit]`, `baobab-turn[pipecat]`, `baobab-turn[all]`
Import path: `baobab_turn`
License: MIT
Python: 3.10+

The adapters are extras, not separate packages. One repo, one version number, one benchmark. Split into separate PyPI packages only if adapter dependencies start conflicting.

```
baobab-turn/
├── README.md                  # leads with the benchmark table, not the pitch
├── LICENSE                    # MIT
├── pyproject.toml
├── CONTRIBUTING.md
├── LANGUAGE_PACKS.md          # how to contribute a language — see §5.8
├── .github/workflows/ci.yml
│
├── baobab_turn/
│   ├── __init__.py            # public exports
│   │
│   ├── core/                  # PURE PYTHON — no framework imports, enforced by CI
│   │   ├── __init__.py
│   │   ├── guard.py           # the rule engine (pure function)
│   │   ├── markers.py         # pack loading + tail matching
│   │   ├── codeswitch.py      # language-switch detection
│   │   ├── pacing.py          # adaptive per-caller delay
│   │   ├── config.py          # BaobabConfig dataclass
│   │   ├── types.py           # GuardInput, GuardResult, LanguagePack
│   │   └── log.py             # structured decision logging
│   │
│   ├── adapters/              # thin framework glue, ~50-100 lines each
│   │   ├── __init__.py
│   │   ├── livekit.py         # BaobabTurnDetector for LiveKit AgentSession
│   │   └── pipecat.py         # equivalent for Pipecat
│   │
│   ├── packs/                 # DATA ONLY — no code in this directory
│   │   ├── registry.py        # pack discovery + language-code resolution
│   │   ├── en-NG.yaml         # ✅ validated
│   │   ├── pcm.yaml           # ✅ validated (Nigerian Pidgin)
│   │   ├── yo.yaml            # ✅ validated
│   │   ├── ig.yaml            # ✅ validated
│   │   ├── ha.yaml            # ✅ validated
│   │   ├── _template.yaml     # starting point for contributors
│   │   └── community/         # ⚠️ UNVALIDATED — see §5.8.3
│   │
│   └── version.py
│
├── bench/                     # the measuring tape — see §7
│   ├── __init__.py
│   ├── runner.py              # replays clips, scores decisions
│   ├── metrics.py             # FCR, LTR, response latency
│   ├── caller.py              # synthetic caller for live calls (phase 3)
│   ├── report.py              # markdown + JSON output
│   └── datasets/
│       ├── README.md          # how to obtain / contribute clips
│       └── manifest.schema.json
│
├── tools/
│   └── label.py               # CLI to label true end-of-turn in clips
│
├── examples/
│   ├── minimal_agent.py       # smallest working agent using the plugin
│   └── ab_agent.py            # runs baseline and sabi side by side
│
└── tests/
    ├── test_markers.py
    ├── test_guard.py
    ├── test_codeswitch.py
    ├── test_pacing.py
    └── test_failopen.py
```

---

## 5. Component specifications

### 5.1 `detector.py` — BaobabTurnDetector

The public entry point. Implements LiveKit's turn-detection protocol so it can be passed to `AgentSession`.

**[VERIFY] Current integration shape.** As of the docs read for this spec, turn detection is configured through `TurnHandlingOptions`:

```python
from livekit.agents import AgentSession, TurnHandlingOptions, inference

session = AgentSession(
    turn_handling=TurnHandlingOptions(
        turn_detection=inference.TurnDetector(),
    ),
    # ... stt, tts, llm
)
```

Our target usage:

```python
from livekit.agents import AgentSession, TurnHandlingOptions, inference
from livekit.plugins.baobab_turn import BaobabTurnDetector

session = AgentSession(
    turn_handling=TurnHandlingOptions(
        turn_detection=BaobabTurnDetector(inner=inference.TurnDetector()),
    ),
    stt=...,  # any Nigerian-capable STT
)
```

**[VERIFY] The protocol to implement.** Find the turn-detector protocol/ABC in the `livekit-agents` source (look under `livekit/agents/voice/`). Based on the deprecated text plugin, it is likely a protocol with roughly these members — **confirm the real signatures before implementing**:

- `unlikely_threshold(language: str | None) -> float | None`
- `supports_language(language: str | None) -> bool`
- `predict_end_of_turn(chat_ctx, ...) -> float` (async, returns probability)

**Constructor parameters:**

| Param | Type | Default | Meaning |
|---|---|---|---|
| `inner` | protocol | `inference.TurnDetector()` | Detector we wrap and correct |
| `config` | `BaobabConfig` | `BaobabConfig()` | Rule tuning, see §5.5 |
| `languages` | `list[str]` | `["en-NG","pcm","yo","ig","ha"]` | Which language codes we claim to handle |
| `guard_timeout_ms` | int | `15` | Hard cap; exceed it and we pass through |
| `enabled` | bool | `True` | Kill switch for A/B testing |
| `on_decision` | callable \| None | `None` | Hook for logging every decision |

**Required behaviour:**

- `supports_language` must return `True` for Nigerian codes AND for whatever the inner detector supports, so it never narrows capability.
- `predict_end_of_turn` must call the inner detector, then pass the result through the guard, then return.
- The whole guard path must be `async` and must not block the event loop.
- If `enabled=False`, return the inner result unchanged. This is how we run A/B.
- If the inner detector raises, log and return a safe default (`1.0`, matching LiveKit's own fallback behaviour so the turn isn't blocked).
- If the guard raises or exceeds `guard_timeout_ms`, log and return the inner result unchanged. **Fail open, always.**

### 5.2 `markers.py` — the lexicon

Two ordered lists of Nigerian English and Pidgin discourse markers, matched against the tail of the current partial transcript.

**These lists are a starting hypothesis. They must be revised from real labelled data (§8). Do not treat them as authoritative.**

**Continuation markers** — speaker has MORE to say. Presence at the transcript tail should *lower* end-of-turn probability.

```
so, then, because, as in, like, and, but, if, when, wey, make,
the thing be say, wetin I dey talk be say, wetin happen be say,
as e be, as e dey be, na because, the reason be say,
you know, you see, I mean, let me talk true, make I talk true,
erm, ehm, ehn, eh, hmm, mmm
```

**Yield markers** — speaker is DONE and inviting a response. Presence should *raise* end-of-turn probability and permit a shorter delay.

```
abi, abi?, shey, shey?, no be so, no be so?, abi no be so,
you get?, you get me?, you understand?, e clear?, e clear so?,
you dey hear me?, you dey follow?, that's all, na all, na so,
okay?, right?, that one nko?, nko?
```

**Ambiguous — handle carefully:**

- `sha` — appears mid-utterance and terminally. Only treat as a yield marker when it is the final token AND followed by VAD silence exceeding `min_delay`.
- `o` (trailing emphatic, as in *"e don finish o"*) — usually terminal but can precede a continuation. Weight it weakly, never as a hard yield.
- `now` (as in *"come now"*) — not the temporal English *now*. Weak signal only.

**Implementation requirements:**

- Match on **normalised** text: lowercase, strip punctuation, collapse whitespace. Nigerian ASR output punctuation is unreliable — never depend on a question mark.
- Match multi-word phrases before single words (longest match wins).
- Match only within the **last N tokens** of the partial transcript (`config.marker_window`, default 6). A *"so"* fifteen words back is irrelevant.
- Use a compiled trie or `set`-based lookup. This runs on every partial transcript update; it must be microseconds, not milliseconds.
- Every marker carries a weight in `[0.0, 1.0]`, not a boolean. Store as `dict[str, float]`.
- Keep the lists in a **data file** (`markers.yaml` or similar) loaded at import, not hardcoded in Python, so contributors can add markers without touching code.

### 5.3 `codeswitch.py` — language switch detection

**Purpose:** detect when the caller switches language mid-utterance, because that is where the inner detector is least reliable.

**Approach for v1 — keep it dumb and fast:**

- Maintain small character-n-gram or wordlist profiles for English, Pidgin, Yoruba, Igbo, Hausa.
- Score the last window of tokens against each profile.
- If the dominant language of the tail differs from the dominant language of the head of the same utterance, flag `code_switch=True`.
- Alternative if the STT provider reports per-segment language (Intron's Sahara handles code-switching natively): **prefer the provider's signal over our own guess.** Check what each adapter exposes.

**Effect:** when `code_switch=True`, apply `config.codeswitch_hold_ms` (default 200ms) of extra patience. Do not veto outright — just wait a little longer.

**Do not** attempt real language identification with a neural model in v1. Too slow, too much complexity, unproven benefit.

### 5.4 `pacing.py` — adaptive per-caller delay

LiveKit already offers a dynamic endpointing mode that adapts delay from session pause statistics using an exponential moving average (`alpha`, default `0.9`). **Use it rather than reimplementing it** where possible.

Our addition: track the caller's observed inter-utterance pause distribution within the session and expose it to the guard, so a slow, deliberate speaker gets more patience than a fast one. Keep it to a rolling mean and standard deviation. No ML.

Guard rails: never extend beyond `max_delay`; never shrink below `min_delay`.

### 5.5 `config.py` — BaobabConfig

A frozen dataclass. Every number in the system is here, nothing hardcoded elsewhere.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `marker_window` | int | `6` | Tokens from the tail to scan |
| `continuation_veto_strength` | float | `0.6` | How much a continuation marker suppresses p |
| `yield_boost_strength` | float | `0.3` | How much a yield marker raises p |
| `codeswitch_hold_ms` | int | `200` | Extra patience on a detected switch |
| `min_delay_ms` | int | `300` | Floor, mirrors LiveKit default with audio detector |
| `max_delay_ms` | int | `2500` | Ceiling, mirrors LiveKit default with audio detector |
| `tonal_language_extra_ms` | int | `150` | Extra patience when STT reports yo/ig |
| `enable_pacing` | bool | `True` | Adaptive delay on/off |
| `enable_markers` | bool | `True` | Lexicon layer on/off |
| `enable_codeswitch` | bool | `True` | Switch detection on/off |

Every one of these flags exists so the benchmark can ablate them and prove which ones actually help. **A feature that doesn't move the numbers gets deleted, not defended.**

### 5.6 `guard.py` — the rule engine

Pure function, no I/O, fully unit-testable:

```
guard(p_inner, transcript_tail, vad_state, language, session_stats, config)
    -> GuardResult(p_adjusted, delay_override_ms, reasons: list[str])
```

Order of operations:

1. Start with `p = p_inner`.
2. If a continuation marker is at the tail: `p *= (1 - continuation_veto_strength)`.
3. If a yield marker is at the tail: `p += (1 - p) * yield_boost_strength`.
4. If `code_switch`: add `codeswitch_hold_ms` to the delay.
5. If language is tonal (yo, ig): add `tonal_language_extra_ms`.
6. If pacing enabled: adjust delay toward the caller's observed rhythm.
7. Clamp `p` to `[0,1]` and delay to `[min_delay_ms, max_delay_ms]`.
8. Populate `reasons` with which rules fired — this is what makes the system debuggable and what the benchmark reports on.

`reasons` is not optional. Every decision must be explainable, or you cannot tune the system.

### 5.7 `log.py` — decision logging

Emit one structured record per turn decision: timestamp, `p_inner`, `p_adjusted`, delay used, rules fired, transcript tail, detected language, whether the turn was actually committed. JSONL to a file or stdout.

This is what feeds the benchmark and what you paste into your build-in-public posts.

---

### 5.8 Language packs — how this becomes African rather than Nigerian

A language pack is a **YAML data file. No Python. No code execution.** This is what lets a Swahili or Twi speaker contribute without touching the engine, and it is the mechanism by which coverage grows beyond what you personally can test.

#### 5.8.1 Pack schema

```yaml
code: sw                      # BCP-47 / ISO 639
name: Swahili
region_hint: [KE, TZ, UG]
status: community             # validated | community | experimental
maintainer: "name <contact>"

tonal: false                  # THE most important field — see §1 hypothesis 1
tonal_extra_ms: 0             # extra patience if tonal

commonly_switches_with: [en, sheng]

continuation_markers:         # more is coming — lowers p(end of turn)
  - {text: "kwa sababu", weight: 0.8}
  - {text: "halafu", weight: 0.7}

yield_markers:                # speaker is done — raises p(end of turn)
  - {text: "sivyo?", weight: 0.9}
  - {text: "unaelewa?", weight: 0.85}

ambiguous_markers:            # position-dependent; documented, weakly weighted
  - {text: "basi", weight: 0.3, note: "terminal only after VAD silence"}

fillers: ["eeh", "mmh"]

thresholds:
  unlikely_threshold: null    # null = inherit framework default
  min_delay_ms: null
  max_delay_ms: null

validation:
  labelled_turns: 0           # honesty field — 0 means untested
  fcr_baseline: null
  fcr_with_pack: null
  dataset_note: "no dataset yet"
```

#### 5.8.2 Loading and resolution

- `packs/registry.py` discovers packs at import, indexes by language code.
- Resolution order: exact code (`en-NG`) → base code (`en`) → `null pack`.
- **The `null pack` is mandatory.** When no pack matches, the guard applies no lexical rules and passes the inner detector's decision through unchanged. Unknown language must degrade to stock behaviour, never to worse-than-stock.
- Packs are hot-swappable per session — a code-switch may change the active pack mid-call.
- Validate packs against a JSON schema at load. A malformed pack logs a warning and is skipped; it must never crash an agent.

#### 5.8.3 The validation tiers — this is the credibility mechanism

| Status | Meaning | Where it lives | Enabled by default |
|---|---|---|---|
| `validated` | Benchmarked on ≥100 labelled turns, numbers published | `packs/` | Yes |
| `community` | Written by a native speaker, not yet benchmarked | `packs/community/` | **No** — opt-in via config |
| `experimental` | Draft, incomplete | `packs/community/` | No |

**Rules that must not be bent:**

- The README lists validated and community packs in **separate tables**. Never one merged list.
- Community packs are opt-in (`BaobabConfig(allow_community_packs=True)`), never on by default.
- Every pack's `validation.labelled_turns` count is displayed wherever the pack is listed. A `0` is shown as `0`, not hidden.
- **Never machine-translate a marker list.** Discourse markers don't translate word-for-word. A pack must be written by a speaker of the language or not exist.
- Promotion from `community` to `validated` requires a benchmark run on that language's own labelled data, submitted by the contributor. You do not need to speak the language to verify the *numbers*.

#### 5.8.4 `LANGUAGE_PACKS.md`

The contributor guide. Must contain: the schema, a fully worked example (use `pcm.yaml`), how to record and label 100 turns in your language using `tools/label.py`, how to run the benchmark on your pack, and how to submit. Assume the contributor is a competent developer who has never seen this repo.

This document is as important as the code. It is the difference between a Nigerian project and an African one.

## 6. Dependencies and where to get them

### Required

| Package | Source | Purpose |
|---|---|---|
| `livekit-agents` >= 1.6.1 | PyPI | The framework. 1.6.1+ has the built-in audio `TurnDetector`. |
| `livekit-plugins-silero` | PyPI | VAD. Required by the audio detector; `min_silence_duration` must be >= 0.25s or the session raises `ValueError`. |
| `pytest`, `pytest-asyncio` | PyPI | Tests |
| `pyyaml` | PyPI | Marker lexicon loading |

### Nigerian STT options (Phase 2 adapters)

| Provider | Access | Notes |
|---|---|---|
| **Intron — Sahara v2.5** | Commercial API, intron.io | ~20 African languages, explicitly engineered for code-switching and switch-point accuracy. Strongest candidate. Check whether it exposes streaming and per-segment language. |
| **Spitch** | API, also via Cencori AI Gateway | Yoruba, Hausa, Igbo, English, Amharic. STT and TTS. |
| **N-ATLAS** | State-backed, open source | Yoruba, Hausa, Igbo, Nigerian English. Free. Good baseline. |
| **SBPN** | Open weights (arXiv 2605.17710) | Base 120M / Large 600M. Covers Pidgin. Self-host. |

**Critical question to answer before building adapters:** which of these support *streaming* recognition with partial transcripts? The guard layer needs partial transcripts as the user speaks. A batch-only provider cannot be used for turn detection, only for post-hoc transcription. **Test this first — it may eliminate providers.**

### Reference material to read, not depend on

- `github.com/livekit/eot-bench` — LiveKit's own end-of-turn benchmark harness. Read its methodology and mirror it where sensible so our numbers are comparable to theirs.
- `huggingface.co/datasets/livekit/eot-evals` — their English and multilingual EOT evaluation datasets. Use the **schema** as a model for ours.
- `github.com/livekit/agents` — SDK source. The protocol definitions live here.

### Datasets for training/eval (Phase 3, not needed for v1)

NaijaVoices, Google WAXAL, African Next Voices. All are read-speech or general-purpose corpora — note that **none of them are labelled for turn boundaries**, which is exactly why §8 exists.

---

## 7. The benchmark harness (`bench/`) — BUILD THIS FIRST

This is the most important part of the project and the first thing to write. Without it, every claim about the plugin is an opinion.

### 7.1 Metrics

Define these precisely; they go straight into the README.

**False Cutoff Rate (FCR)** — primary metric.
`(number of turns where the agent began speaking before the labelled true end-of-turn) / (total turns)`. Lower is better. This is the number the whole project exists to reduce.

**Late Response Delay (LRD)**.
`agent_speech_start_time − true_end_of_turn_time`, reported as P50 / P90 / P95. This is the cost of patience. FCR and LRD trade against each other; any honest report shows both.

**Turn Latency**, P50 / P90 / P95.
Full round trip: caller stops → agent starts speaking. Instrument each stage separately (VAD → STT commit → detector → LLM → TTS first byte).

**Measure STT latency honestly:** as the gap between how much audio the model has consumed and the transcript it has committed — **not** wall-clock from request send. Wall-clock flatters you.

**Report the tail, not the median.** A great P50 with a terrible P95 is a bad agent. Users remember the turn that hung for three seconds, not the fifty that were fine.

### 7.2 Offline mode (v1)

Replay labelled audio clips through the detector, compare committed turn boundaries against labels, output metrics. No live calls, no network. Fast enough for CI.

Manifest schema (`bench/datasets/manifest.schema.json`):

```json
{
  "clip_id": "string",
  "audio_path": "string",
  "sample_rate": 16000,
  "language_primary": "en-NG | pcm | yo | ig | ha",
  "code_switch": true,
  "turns": [
    {
      "speaker": "caller",
      "start_ms": 0,
      "true_end_ms": 3420,
      "transcript": "string",
      "notes": "trailing 'sha', long mid-utterance pause"
    }
  ]
}
```

### 7.3 Ablation mode

Run the same dataset with each guard feature toggled off in turn (`enable_markers`, `enable_codeswitch`, `enable_pacing`). Output a table showing each feature's contribution to FCR and LRD.

This is non-negotiable. It is how you find out which of the four hypotheses in §1 are actually true.

### 7.4 Live mode (Phase 3)

A synthetic caller that dials the agent over LiveKit and runs adversarial scripts: interrupt mid-sentence, go silent for eight seconds, switch to Pidgin halfway, speak over the agent, add background noise. Score the resulting transcript and timings.

Note that LiveKit ships its own test framework and agent simulations — check `docs.livekit.io/agents/start/testing/` before building this from scratch. Reuse beats reinvention.

### 7.5 Report output

`bench/report.py` writes both JSON (machine) and Markdown (human). The Markdown table is what goes in the README and in your posts. It must include: dataset name and size, model versions, config used, FCR, LRD percentiles, and the ablation table.

---

## 8. Data collection and labelling

We need audio of real Nigerian conversational speech with **labelled true end-of-turn boundaries**. This does not exist publicly. Creating it is the hardest and most valuable part of the project.

### 8.1 Sources, in order of preference

1. **Your own production agent's call recordings.** Real callers, real conditions, real value. **Requires informed consent and a lawful basis under the Nigeria Data Protection Act.** Do not skip this. Get consent, document it, and honour deletion requests.
2. Recorded conversations with willing volunteers, scripted to include code-switching, hesitations, and trailing markers.
3. Public conversational audio (podcasts, interviews) — check licensing before redistribution.

### 8.2 Labelling tool (`tools/label.py`)

A minimal CLI:
- Play a clip, show the transcript.
- Operator marks the millisecond where each speaker genuinely finished their turn.
- Operator tags: primary language, code-switch present, trailing marker present, hesitation present.
- Writes to the manifest schema in §7.2.

Keep it ugly. It is an internal tool. Do not build a web UI.

### 8.3 Target for v1

**200 labelled turns minimum** across English-NG, Pidgin, and at least one tonal language. This is small, and you must say so in the README. Small and honest beats large and fabricated.

### 8.4 Privacy

- Never commit raw caller audio to the repo.
- Publish the *manifest and metrics*, not the audio, unless consent explicitly covers redistribution.
- Provide a `datasets/README.md` explaining how someone reproduces results with their own data.

---

## 9. Build order (milestones)

Each milestone ends with something publishable. Do not start the next until the previous is real.

### M1 — Measure the baseline (target: 1–2 weekends)
- `bench/` offline runner, metrics, report.
- `tools/label.py`.
- 50 labelled turns from your existing production agent.
- Run LiveKit's stock `inference.TurnDetector()` against them.
- **Deliverable:** the first published FCR number for Nigerian speech on a stock voice stack. Nobody has this. This alone is worth a post.

### M2 — The guard layer (2–3 weekends)
- `markers.py`, `guard.py`, `config.py`, `detector.py`.
- Fail-open behaviour and its tests.
- Unit tests to >80% coverage on `guard.py`.
- **Deliverable:** FCR before vs after, with the ablation table. Post it.

### M3 — Code-switch and pacing (2 weekends)
- `codeswitch.py`, `pacing.py`.
- Expand dataset to 200 turns.
- **Deliverable:** updated benchmark, v0.1.0 on PyPI, public repo.

### M3.5 — Prove the core boundary (1 weekend)
- Build `adapters/pipecat.py`.
- Run the identical benchmark through both adapters; results must match within noise.
- If porting required touching anything in `core/`, the boundary was wrong — fix it now, before more code depends on it.
- **Deliverable:** "same detector, two frameworks, same numbers." This is what proves the architecture claim rather than asserting it.

### M4 — STT adapters (open-ended)
- Wrap Intron / Spitch / N-ATLAS behind LiveKit's STT interface.
- Only build adapters for providers confirmed to support streaming (see §6).
- **Deliverable:** latency comparison across Nigerian STT providers. Another post nobody has written.

### M5 — Open the door to other African languages (2 weekends)
- Write `LANGUAGE_PACKS.md` properly (§5.8.4).
- Ship `_template.yaml` and the pack JSON schema.
- Make `tools/label.py` language-agnostic so a contributor can label their own data.
- Publish a call for contributors naming specific languages: Swahili, Amharic, Zulu, Twi, Wolof, Shona, Somali.
- **Deliverable:** the first outside contributor's pack. That is the moment this stops being your project and becomes an African one.

**Do not attempt this before M4.** A contributor guide pointing at an unproven engine wastes their goodwill, and you get one first impression.

### M6 — Live simulation (open-ended)
- Adversarial synthetic caller.
- Only after M1–M5 are stable.

---

## 10. Testing requirements

- `pytest` with `pytest-asyncio`. CI on GitHub Actions, Python 3.10 / 3.11 / 3.12.
- **`tests/test_failopen.py` is mandatory and is the most important test file.** It must prove: inner detector raises → session continues; guard raises → inner result passes through; guard exceeds timeout → inner result passes through; malformed transcript → no crash.
- `test_markers.py` must cover the ambiguous cases in §5.2 explicitly (`sha`, trailing `o`, `now`).
- Guard must be tested as a pure function with fixture inputs — no LiveKit session needed.
- Benchmark runner must have a golden-file test so metric definitions can't drift silently.

---

## 11. Packaging and publishing

- `pyproject.toml`, hatchling or setuptools, standard namespace package layout so `livekit.plugins.baobab_turn` resolves alongside official plugins.
- Version with semver, start at `0.1.0`. Pre-1.0 signals it is early — that is fine and honest.
- Publish to PyPI via GitHub Actions on tag.
- **README structure, in this order:** benchmark table → one-paragraph problem statement → install → 5-line usage example → how to reproduce the benchmark → limitations → contributing.

Lead with the numbers. The numbers are the product.

---

## 12. Known risks and open questions

Answer these early; several could change the design.

1. **Does the inner audio detector even expose an interceptable probability?** The `v1` model runs on LiveKit Inference (server-side). If we can't see or adjust its output, we may need to wrap the deprecated open-weights text detector instead, or the `v1-mini` local model. **This is the single biggest technical unknown. Resolve it in week one.**
2. **The text turn detector is deprecated** and slated for removal in SDK 2.0. Do not build on it as a permanent foundation.
3. **`unlikely_threshold` accepts a per-language dict.** Investigate whether simply supplying tuned thresholds for Nigerian language codes gets most of the benefit with none of our code. **If it does, that is a great outcome — publish it and build less.**
4. **Streaming support** across Nigerian STT providers is unconfirmed (§6). May eliminate providers.
5. **The marker lexicon is unvalidated.** Treat §5.2 as a first draft to be replaced by data.
6. **Latency from Lagos.** Round trips to us-east-1 on Nigerian mobile networks may dominate the whole budget, making our millisecond-level gains irrelevant. Measure this in M1 and report it honestly even if it undermines the project's premise.
7. **Consent and NDPA compliance** on call recordings is a real legal requirement, not a formality. Contributors in other countries have their own regimes (Kenya's DPA, South Africa's POPIA) — `LANGUAGE_PACKS.md` must tell them to check their own law, not assume ours.
8. **Pipecat's turn-detection interface may differ enough** that the core boundary needs adjusting. Discover this at M3.5, not at v1.0.
9. **The tonal hypothesis may be wrong.** It is the load-bearing claim for pan-African relevance. If M2's ablation shows `tonal_extra_ms` contributes nothing to FCR, the pan-African story weakens considerably and you should say so publicly rather than quietly keep the flag.
10. **Contributor packs may never arrive.** Open source contribution is rare. If nobody writes a Swahili pack in six months, the honest outcome is a well-built Nigerian tool with a pan-African architecture — which is still a good outcome. Do not pad the README with empty language stubs to look bigger than you are.

---

## 13. Definition of done for v1

- [ ] `pip install baobab-turn[livekit]` works.
- [ ] Swapping one line in an existing agent enables it.
- [ ] `core/` imports no framework — enforced by a CI check, not by discipline.
- [ ] The same benchmark runs through both the LiveKit and Pipecat adapters with matching results.
- [ ] Guard layer p95 under 15ms.
- [ ] Agent never breaks when the guard fails — proven by tests.
- [ ] Unknown language falls back to stock behaviour via the null pack — proven by tests.
- [ ] Benchmark reproducible by a stranger with their own clips, in their own language.
- [ ] README shows real FCR before/after with dataset size stated.
- [ ] README separates validated packs from community packs, with turn counts shown.
- [ ] Ablation table published, including features that didn't help.
- [ ] `LANGUAGE_PACKS.md` complete enough that a stranger can add a language unaided.
- [ ] MIT licensed, public repo, CI green.

---

## 14. Principles for whoever builds this

**Measure before you build.** M1 exists so that M2 is aimed at a real problem rather than an assumed one.

**Fail open, always.** A turn detector that breaks a call is infinitely worse than one that's merely mediocre.

**Publish what didn't work.** The ablation table showing a feature made no difference is more credible than any table where everything helped. Credibility is the actual product here.

**Small honest datasets beat large vague claims.** "200 labelled turns, FCR 31% → 6%" is strong. "Trained on thousands of hours" with no methodology is noise.

**Delete features that don't move the numbers.** Every flag in `BaobabConfig` is a hypothesis under test, not a commitment.
