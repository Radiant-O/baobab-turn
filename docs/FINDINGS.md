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
