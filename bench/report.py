"""Turn benchmark results into a report a stranger could check.

Markdown for humans, JSON for machines. The markdown is what goes in the
README and in build-in-public posts, so it carries its own caveats: a table
that travels without them will be misread, and this project's only real asset
is that its numbers mean what they say.
"""

from __future__ import annotations

__all__ = ["to_markdown"]

_CAVEAT = """> **What this measures.** A lexical-layer ablation on real turns, not an
> end-to-end benchmark. Every turn below is a real human turn boundary,
> replayed word by word as a streaming transcript. A *prefix* is a point where
> the speaker demonstrably kept talking, so committing there would have cut
> them off; the *full turn* is where they demonstrably stopped.
>
> Three things it cannot see: there is **no audio**, so the guard judges on
> text with no silence cue; `p_inner` is held **constant**, so the baseline is
> an uninformative predictor and this is an upper bound on the lexical
> layer's standalone signal rather than its contribution in a real stack; and
> word-by-word reveal is an **idealised** stand-in for a real STT, which lags
> and revises.
>
> It answers "do these markers carry real signal about whether a turn has
> ended". It does not answer "does this make an agent better"."""


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _pct_or_na(x: float, fired: bool) -> str:
    """Directional accuracy is undefined when no rule ever fired -- printing
    0.0% there reads as "always wrong" rather than "never asked"."""
    return _pct(x) if fired else "n/a"


def to_markdown(results: dict) -> str:
    conditions: dict[str, dict] = results["conditions"]  # type: ignore[assignment]
    lines: list[str] = []

    lines.append("# baobab-turn benchmark")
    lines.append("")
    lines.append(_CAVEAT)
    lines.append("")

    any_cond = next(iter(conditions.values()))
    lines.append(
        f"**Corpus:** `{results['corpus']}` — {any_cond['n_turns']:,} turns, "
        f"{any_cond['n_prefixes']:,} mid-turn points, {any_cond['n_ends']:,} turn ends."
    )
    lines.append(
        f"**Pack:** `{results['pack']}` (status `{results['pack_status']}`, "
        f"labelled turns **{results['pack_labelled_turns']}**). "
        f"`p_inner` held at {results['p_inner']}."
    )
    lines.append("")

    lines.append("| Condition | Premature commits | Missed turn ends | Coverage "
                 "| Directional accuracy |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, s in conditions.items():
        lines.append(
            f"| {name} | {_pct(s['premature_commit_rate'])} | {_pct(s['missed_end_rate'])} "
            f"| {_pct(s['coverage'])} "
            f"| {_pct_or_na(s['directional_accuracy'], s['coverage'] > 0)} |"
        )
    lines.append("")

    lines.append("**Premature commits** — the agent talks over a caller who was still "
                 "going. Lower is better; this is the number the project exists to reduce.")
    lines.append("")
    lines.append("**Missed turn ends** — the agent leaves a caller hanging. This is the "
                 "cost of patience, and it trades against the first column. Any honest "
                 "report shows both.")
    lines.append("")
    lines.append("**Coverage** — how often the lexical layer had any opinion at all. It "
                 "caps how much this layer can ever be worth: a rule that never fires "
                 "cannot help.")
    lines.append("")
    lines.append("**Directional accuracy** — of the points where a rule did fire, how "
                 "often it pushed the probability the right way: down mid-turn, up at a "
                 "turn end.")
    lines.append("")

    for name, s in conditions.items():
        counts: dict[str, int] = s["reason_counts"]
        if not counts or set(counts) <= {"no_rules_fired"}:
            continue
        lines.append(f"### Rules fired — {name}")
        lines.append("")
        lines.append("| Rule | Times |")
        lines.append("|---|---:|")
        for rule, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{rule}` | {count:,} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Reproduce: `python -m bench.runner <corpus> --pack "
                 f"{results['pack']} --ablate`")
    return "\n".join(lines)
