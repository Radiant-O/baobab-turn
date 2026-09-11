# Findings

Verified against `livekit-agents` 1.8.0 and `pipecat` @ `1f8a513`. Every line
below is either a correction to `SPEC.md` or a constraint that will cause a
bug if ignored. Citations are file:line in those SDKs.

## LiveKit — the protocol

- The audio detector is `_StreamingTurnDetector` / `_StreamingTurnDetectorStream`
  (`voice/turn.py:54-92`): `push_audio(frame)` + `predict() -> Future[TurnDetectionEvent]`.
  **It has no `predict_end_of_turn`** — that is the deprecated *text* detector
  (`turn.py:36-51`). SPEC §5.1 documents the wrong one.
- `unlikely_threshold` / `supports_language` are **async** and take
  `LanguageCode`, not `str`. `supports_language` is always effectively `True`
  on the audio path.
- Also required on a wrapper: `backchannel_threshold`, `prediction_timeout`,
  `is_fallback`, `cancel_inference`, `flush`, `end_input`, `aclose`,
  `stream(conn_options=...)`. Adapter is ~150-250 lines, not 50.
- `voice/turn.py` is **private and unexported**. Nothing protects us from a
  minor-release change → `tests/test_livekit_contract.py` asserts the shape.

## LiveKit — fail-open (safety critical)

- Custom streaming detectors are **not** wrapped in try/except
  (`audio_recognition.py:1529-1557` unguarded vs `:1558-1567` guarded). An
  escaped exception kills the EOU task and **the turn never commits**.
- So: never raise, never `set_exception`. Resolve with a fresh default event —
  `end_of_turn_probability=1.0`, `last_speaking_time=time.time()`, rest `None`.
- **Never use a module-level default-event singleton.** The SDK dedupes its
  decision hook by object *identity* (`:1619-1627`), so a reused instance
  silently suppresses the second hook of a turn. Mutating the inner event in
  place and returning it is safe and carries all fields free.
- Copy through: `inference_duration`, `backchannel_probability`,
  `detection_delay`, `type`. `last_speaking_time` is never read by the SDK.

## LiveKit — timing

- `predict()` fires on **200 ms of VAD silence while still speaking**
  (`:1403-1408`), before end-of-speech. We opine mid-pause.
- `prediction_timeout` = **1.0 s** (`inference/eot/base.py:37`). Late → session
  proceeds without us on `min_delay`. Our 15 ms budget is well inside it;
  **do not raise it**, it would delay every turn.
- `_push_audio` and the `predict()` call site are both sync on one loop →
  cannot interleave, **no lock needed** (`:730-756`).
- Turn reset hook is **`flush(reason=...)`**, not `cancel_inference` (which
  means "this pause is void", not "utterance ended").

## LiveKit — no delay channel

- LiveKit never consumes a delay. It is binary: `min_delay`, or `max_delay`
  when `p < unlikely_threshold` (`:1512`, `:1569-1574`). Range ~0.3s/2.5s.
- So `unlikely_threshold` is a **delay switch, not a veto** — 0.05 and 0.35
  behave identically against 0.36. The guard layer is justified (SPEC risk 3
  resolved: tuned thresholds alone are not enough).
- Consequence: `tonal_language_extra_ms` and `codeswitch_hold_ms` are **not
  expressible on LiveKit**. Each adapter spends the intent in its own
  currency — Pipecat sleeps the exact ms; LiveKit tunes that language's
  `unlikely_threshold` (which is the right graded instrument for the tonal
  hypothesis anyway).

## Transcript access

- The detector is **audio-only by construction**. `stream()` takes zero args
  (`:968`); no session/agent/chat_ctx reference is stored.
- But `user_input_transcribed` is **public** (`voice/events.py:294`) and fires
  before commit with `transcript`, `is_final`, `language`, `created_at`. So
  the marker layer works with **any** STT.
- No pull-getter exists (`session.history` is committed context only), so
  on-demand reads are impossible.
- No shared session handle: STT and the detector are both session-blind, and
  the detector is built inside `AgentSession.__init__` before a session
  exists. `get_job_context()` is **job-scoped and would leak across
  sessions** — do not use. Wiring is explicit: one owner object exposing
  `.stt` and `.turn_detection`.
- Dual path behind one `TranscriptTail`: our Intron adapter's buffer
  (primary), the public event (fallback, any STT).

## Transcript staleness — build this in from v1

Three ways the tail goes wrong:

1. No alignment guarantee. Detector snapshots 1.2 s of audio synchronously;
   STT crosses two channel hops plus provider RTT. `AudioFrame` has **no
   timestamp or sequence number**. Only handles are on the STT event
   (`SpeechData.start_time/end_time`, `SpeechEvent.speech_end_time`).
2. During AEC warmup and uninterruptible agent speech, STT gets a **silence
   frame while the detector gets real audio** (`agent_activity.py:1508-1522`).
3. **The transcript gate suppresses the event entirely while the agent
   speaks** (`:1160-1163`, replay `:716-723`) — the tail is stale exactly
   during barge-in.

Therefore: stamp every write, abstain past a hard age budget (pass the inner
probability through unchanged), and **report the abstention rate as a headline
metric** — it caps how much the marker hypothesis can ever be worth.

## Pipecat

- Target `BaseUserTurnStopStrategy` (`turns/user_stop/base_user_turn_stop_strategy.py:38`),
  not `BaseTurnAnalyzer` (audio-only, same trap as LiveKit). Layer B receives
  interim transcripts, `finalized`, VAD frames, and **`language`** — so the
  transcript problem does not exist here.
- **Decision-native**: `EndOfTurnState.COMPLETE/INCOMPLETE`, no `UNKNOWN`
  (`base_turn_analyzer.py:21-30`). Probability is metrics-only, thresholded at
  a hardcoded 0.5.
- No min/max delay; timing is owned by the strategy. Delay is honoured exactly
  by overriding `trigger_user_turn_stopped` and sleeping.
- `append_audio` is **sync, on the loop, per ~20 ms frame** → core splits into
  cheap sync `observe` + async `evaluate`.
- Adapter ~100-150 lines.

## Type consequences

`GuardResult`: `decision: bool` **non-optional** (Pipecat needs it),
`p_adjusted: float | None` (LiveKit needs it), `delay_override_ms: int | None`
(intent; adapters translate), `reasons: list[str]` (never optional; includes
`tail_stale` / `tail_absent`). The **guard owns the threshold** via
`threshold_for(language)` — LiveKit asks for it, Pipecat hides it.
`GuardInput` must carry a branchable `is_final`.

## STT providers

| Provider | Streaming partials | Pidgin | Verdict |
|---|---|---|---|
| **Intron Sahara** | yes, WebSocket, revisable | yes | **only live option** |
| Spitch | no (multipart only) | no | offline transcription only |
| N-ATLAS | no (4x whisper-small) | no | offline only, gated licence |
| SBPN | no (offline NeMo) | yes | offline only, **CC BY-NC-SA** |

- **No Nigerian STT is openly licensed for commercial use**, so this package
  ships none and is STT-agnostic; users bring their own. Intron is closed and
  sales-gated (docs and the `wss://` endpoint are real, but `voice.intron.io`
  is a login wall with no self-serve keys and no published pricing), so it can
  only ever be an optional adapter, never a dependency.
- If we do build the Intron adapter: connection closes after `COMMIT` — **one
  utterance per socket** (per-turn reconnect + pre-warmed spare), 300 s session
  cap, base64-in-JSON audio, no LiveKit plugin exists.
- The marker layer is the **only** part needing a live transcript. Tonal
  patience and pacing run on audio/VAD/probability alone — and tonal is the
  pan-African hypothesis, markers the language-specific one. So no streaming
  STT costs us the least portable component, not the load-bearing one.
- Untested cheap option for markers: a **generic** streaming STT (Deepgram,
  AssemblyAI, Soniox) only has to catch short tail tokens (`abi`, `sha`,
  `you get?`). Measure marker recall against labelled clips before assuming a
  Nigerian provider is required.
- **No provider reports live per-segment language.** SPEC §5.3's preferred
  path does not exist → `codeswitch.py` is required, not optional.
- **SBPN is CC BY-NC-SA (non-commercial)** — the CC-BY-4.0 belongs to the
  paper, not the weights. Never a dependency of an MIT package. Usable locally
  as a research-only language-ID oracle (97-100% F1 from 0.1 s).
- N-ATLAS is gated and capped at 1,000 end-users. "Open source" is a stretch.

## Corrections to SPEC.md

- **§7.1** — "FCR" names two things. Offline there is no LLM/TTS, so only
  detector commit time is observable. Use `premature_commit_rate` (offline,
  CI) and `fcr` (live, end-to-end). Never conflate them in a report.
- **§7.2** — store the **partial-transcript timeline** in the manifest.
  `replay-transcript` mode is then deterministic, network-free and CI-able;
  `replay-audio` is manual. Decide before clips are labelled.
- **§5.1** — the 15 ms `guard_timeout_ms` cannot work: a sync function has no
  await points for `asyncio.wait_for` to interrupt. Guard stays sync;
  fail-open is try/except; 15 ms p95 is a **test assertion**, not a runtime cap.
- **§9 M3.5** — success means **identical `GuardResult`s** for identical
  `GuardInput`s (exact, not statistical), plus `premature_commit_rate` within
  noise. End-to-end **timings will legitimately differ** per the delay finding
  above and must not be read as a boundary failure.
- **§11 / §5.1** — import path is `baobab_turn.adapters.livekit`. Do not squat
  in LiveKit's `livekit.plugins` namespace.
- **Scope / §5.8.3** — v1 validated coverage is **`en-NG` and `pcm` only**
  (the maintainer's languages). `yo`/`ig`/`ha` ship as `community`, opt-in,
  until a speaker writes markers and labels data. Recruit one **Yoruba or
  Igbo** speaker ahead of Hausa: the tonal hypothesis (risk 9) is the
  load-bearing pan-African claim and cannot be tested without tonal data.
- **§8.1 / M1** — there is no production agent, and one is not needed. Data
  comes from volunteer human-human recordings. Caveat for the README: people
  simplify when addressing a bot, so this is a proxy; revalidate at M3.5.
- **§5.8.3** — an LLM generating a marker list is machine translation with
  extra steps. Neither Claude nor any subagent writes marker content.
- **§10** — Python 3.13 is supported (`>=3.10,<3.15`; prebuilt cp313 wheel for
  `livekit-local-inference`). CI 3.13 is required, not allow-failure.

## Corpora and licences

Usable, all **CC BY-SA 4.0**, all genre `spoken`, all marking `# speaker_id`
so turn boundaries are recoverable. Maintainer for all three is Bernard Caron
(`bernard.l.caron@gmail.com`).

| Corpus | Content | Note |
|---|---|---|
| SUD_Naija-NSC | Nigerian Pidgin | Cite **SUD**, not UD — the `non_gold/` data exists nowhere else |
| SUD_Hausa-SouthernAutogramm | Hausa, Zaria dialect, 1,918 sent. | Relations fully manual |
| SUD_Hausa-NorthernAutogramm | Hausa, Ader dialect, 400 sent. | Repos hold more than the READMEs claim; check gold status |

**`liva-ai/code-switching-asr` (`en-pcm/`) is the only CC BY 4.0 spontaneous
Pidgin dialogue found** — no NC, no share-alike, ungated. 777 speaker turns
across 4 conversations with millisecond start/end per turn, in-the-wild
recordings among friends. Only 38 minutes, but it is the one corpus whose
licence imposes nothing.

**Dead ends worth not rediscovering.** `intronhealth/afrispeech-dialog` is
real spontaneous dialogue but CC BY-NC-SA *and* English, not Pidgin.
NaijaLex 2.0 is CC BY-NC-SA (its paper's CC BY does not flow to the data).
NaijaVoices, AfriSpeech-200 and the whole `intronhealth/AfriSwitch*` family
are CC BY-NC-SA. `liva-ai/yapdo-convo` advertises 769 h of Pidgin with **no
licence at all** and ships four sample files with no transcripts. Several
HuggingFace re-uploads have silently stripped upstream share-alike notices —
verify licences upstream, never from a re-upload's badge.

## What we may derive from a CC BY-SA corpus

Not legal advice, but the reasoning is worth recording since it shapes the
design.

**Safe: compute counts locally, ship hand-set weights informed by them, and
attribute.** Share-alike attaches only to *Adapted Material*, which requires
the output to be copyrightable, to contain the licensed material in modified
form, and to result from an act requiring permission. A frequency count fails
all three, and §4(a) expressly grants the right to extract.

**The trap is shipping the frequency table itself.** §4(b) can pull BY-SA onto
our YAML through *database right*, which needs no originality at all — so
"counts are facts" is not a defence. Ship **ordinal tiers**, never corpus
frequencies.

**Marker strings themselves are the safe half.** `abi`, `sey`, `wey` are
lexical items of the language, observed rather than authored, and short words
and phrases are not copyrightable. Extracting example *sentences* is not
safe; vendoring the corpus is worst.

## Mining caveats

**`discourse` alone under-counts badly.** SUD puts emphatic and terminal
particles under `mod:emph`, not `discourse` — the trailing `o` in "for dis
Nigeria o //" is `PART`/`mod:emph`. Filtering on `discourse` only hid the
entire trailing-particle class: including `mod:emph` moves `o` from invisible
to **3,088 occurrences, the most frequent marker in the corpus**, and
surfaces `now` (1,388) and `sef` (779, absent from our pack).

**Most of SUD_Naija-NSC is monologue, and the defensible turn count is 937.**
Of the 88 gold recordings, **8 are suffixed `_DG` (dialogue) and 80 `_MG`
(monologue)** -- the filename suffix is load-bearing. The gold dialogue subset
is 1,614 sentences / 14,623 tokens / **937 turn changes**. The larger 14,791
figure includes `non_gold/`, which is licence-clean CC BY-SA but undocumented,
unvalidated and carries no speaker demographics. Quote 937 unless the
unvalidated data is explicitly in scope.

**A filename trap:** recordings titled "Interview" are *single*-speaker --
the interviewer's turns were never transcribed. Do not infer dialogue from a
title; use the `_DG` suffix.

**Token-level timings are manufactured.** 99.96% of inter-pausal units have
all-identical token durations, so per-word duration, speech rate and
final-particle lengthening are all unavailable. Same-speaker gaps are censored
(52% exactly zero, meaning "not independently timed"). Only cross-speaker gaps
are real, because the two speakers sit on separate annotation tiers.

**Cross-speaker gap at handover: median −230 ms, 63% negative** (13,178
observations). The incoming speaker typically starts *before* the outgoing
utterance ends. Overlap at handover is normal in this data, which is evidence
against a design that simply adds patience — and is the right distribution for
calibrating barge-in. Trim the tail before use; extreme negatives are
alignment errors, not real overlaps.

## Hausa and Zaar: the treebank is licensed, the audio is not

The productive seam is one research lineage -- Bernard Caron and colleagues at
LLACAN (CNRS), across CorpAfroAs, CorporAn, Autogramm and NaijaSynCor. Its
signature in CoNLL-U is `# speaker_id` + `# sound_url` + `# sent_timecode`.
The curated index is `spoken_UD_2.17.json` in `grew-nlp/corpusbank`, whose
African members are exactly Beja, Hausa-Northern, Hausa-Southern, Naija-NSC,
Northwest Gbaya and Zaar. **No Yoruba, no Igbo.**

| Corpus | Turn changes | Licence |
|---|---|---|
| UD_Hausa-SouthernAutogramm (Zaria) | 1,172 | CC BY-SA 4.0 |
| **UD_Zaar-Autogramm** (Bauchi State, Chadic) | 708 | CC BY-SA 4.0 |
| SUD_Naija-NSC, `_DG` files only | 937 | CC BY-SA 4.0 |
| UD_Hausa-NorthernAutogramm (Ader) | 71 | CC BY-SA 4.0 |

Corrections to earlier notes: Hausa-Southern is **face-to-face peer
conversation**, not broadcast; Hausa-Northern holds 1,305 sentences rather
than 400, but only ~215 are genuine dialogue. Take **Zaar from the UD repo** --
the SUD one has no licence file at all. Drop Zaar's 3 `_READ` files, which are
read-aloud.

**The ELAN and audio layer carries no reuse licence.** CorpAfroAs states only
`Copyright (c) CorpAfroAs` with `accessRights: Freely accessible` -- free to
*access*, with no grant to reuse or redistribute. Same for CorporAn, ELAR
deposits and the NaijaSynCor MP3s. So the CoNLL-U treebanks are usable under
CC BY-SA; the recordings and ELAN files they point at are not. Ask Bernard
Caron or Christian Chanard (`christian.chanard@cnrs.fr`) before redistributing
anything audio-derived.

**Published broken link:** `# sound_url` points at `.../media/HAUZ/WAV/...`
which 404s. The working path is `HAU/`.

## Yoruba and Igbo: evidenced negatives

**No Yoruba spontaneous conversational corpus exists**, verified four
independent ways: CorporAn's 43-language file list (Yoruba absent), 1,124
CoCoON LLACAN records via OAI (zero `yor`), the CorpAfroAs set, and 212
ORTOLANG OAI records. ELRA holds only Iroyin-Speech (read speech, free for
academic non-commercial); LDC only translated text and a lexicon.

**No Igbo anywhere in that infrastructure** -- ELRA returns literally nothing
for "Igbo". The only conversational Igbo corpus is IARPA Babel LDC2019S16,
$25 to non-members but non-commercial, no redistribution, and LDC reports
recipients to IARPA. There is no Hausa or Yoruba Babel pack; Igbo is the one
Nigerian language the programme covered.

One trace worth knowing: `SUD_Yoruba-AfriSUD` is declared in Grew's manifest
but its backing repo 404s, so a Yoruba treebank may exist privately. Expect
written source text if it surfaces.
