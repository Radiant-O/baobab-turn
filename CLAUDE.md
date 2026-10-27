# baobab-turn — working notes for Claude

## What this is

A turn-detection guard layer for African speech in voice agents. Pure-Python
core, thin framework adapters, YAML language packs.

## Read these first

- `docs/SPEC.md` — the full engineering specification. Authoritative on intent.
- `docs/FINDINGS.md` — verified SDK constraints and corrections to the spec.

## Hard rules

1. **`baobab_turn/core/` imports no framework. Ever.** Enforced by
   `tests/test_core_purity.py`, which `scripts/check.py` runs as its own step.
   If a framework type is needed in core, the adapter translates it instead.
2. **Fail open.** If the guard raises, the inner detector's decision passes
   through untouched. The agent must never break because our layer broke.
3. **Never write marker lists for a language.** Generating discourse markers
   with an LLM is machine translation with extra steps, which the spec
   forbids. Build the schema and the validator; a speaker supplies the
   content. This applies to subagents too.
4. **Never commit raw caller audio.** Manifests and metrics only.
5. **Every number lives in `core/config.py`.** Nothing hardcoded elsewhere,
   because the benchmark ablates every flag.
6. **`reasons` is not optional.** Every guard decision records which rules
   fired, or the system cannot be tuned or debugged.
7. **No estimated numbers in the README.** An empty benchmark table is honest.

## LiveKit API churn

The LiveKit Agents API changes frequently and this spec's code samples are
marked `[VERIFY]` for that reason. Check the SDK source before depending on
any signature. Docs lag source; source wins.

## Checks

**`python scripts/check.py` is the gate.** Run it before pushing; if it
passes, the branch is good. It runs lint, the test suite, the core purity
check, and -- importantly -- builds the wheel and installs it into a
throwaway environment to confirm the language packs actually ship.

That last step exists because the package was unbuildable for weeks and
nothing noticed: every install was `pip install -e .`, and editable installs
skip wheel packaging entirely.

Hosted CI is **not** part of this workflow. `.github/workflows/ci.yml` is kept
correct but set to `workflow_dispatch` only, so it never fires on a push.
Do not add automatic triggers, and do not treat a missing green tick as a
problem to solve.

## Environment

- Local Python is 3.13.7. `livekit-agents` 1.8.0 supports 3.10-3.14.
- `uv` is not installed.
- The LiveKit docs MCP server (`https://docs.livekit.io/mcp/`) that SPEC §0
  asks for is **not configured**. Set it up before deep SDK work.

## Build order

Milestones in SPEC §9. Do not start a milestone before the previous one is
real. M2 is gated on the open questions in `docs/FINDINGS.md`.
