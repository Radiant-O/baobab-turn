"""Replay real turns through the guard and score its decisions.

    python -m bench.runner path/to/cencos --pack pcm
    python -m bench.runner path/to/cencos --pack pcm --ablate --json out.json

## What this measures, and what it does not

**It is a lexical-layer ablation on real turns, not an end-to-end benchmark.**
Say so wherever the numbers are published.

Every turn in a conversational corpus is a real, human-observed turn
boundary. We replay each one word by word, as a streaming STT would reveal
it, and ask the guard at every prefix whether the turn is over:

    "i wan"                        -> should WAIT   (speaker has more)
    "i wan tell you say the"       -> should WAIT
    "i wan tell you say the thing be say"  -> ground truth: turn ended here

A **proper prefix** is a point where the speaker demonstrably kept going, so
committing there would have cut them off. The **full turn** is a point where
the speaker demonstrably stopped.

Three things this cannot see, all of which matter end to end:

1. **No audio, so no VAD and no silence.** In a real session the detector has
   already heard a pause before it asks. Here the guard judges on text alone.
2. **No real inner detector.** `p_inner` is held at a constant, so the
   baseline is an uninformative constant predictor by construction. The
   comparison shows what the lexical layer adds to *nothing*, which is an
   upper bound on its standalone signal, not its contribution in situ.
3. **Idealised partials.** A real STT lags, revises and re-segments. Word-by-
   word reveal is the friendliest possible version of that.

So: this answers "do the markers carry real signal about whether a Nigerian
Pidgin turn has ended?" It does not answer "does this make an agent better."
That needs audio, a real detector, and labelled latency.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

from baobab_turn.core.config import BaobabConfig
from baobab_turn.core.guard import guard
from baobab_turn.core.markers import MarkerIndex
from baobab_turn.core.types import GuardInput
from baobab_turn.packs.registry import resolve

from .corpora import Turn, load_transcripts

__all__ = ["Outcome", "Scores", "run", "score"]

#: Held at the decision threshold so the baseline sits exactly on the fence
#: and every decision difference is attributable to the lexical layer. Any
#: other value would bake in an arbitrary prior we have no basis for.
DEFAULT_P_INNER = 0.5


@dataclass(frozen=True, slots=True)
class Outcome:
    """The guard's verdict at one point in one turn."""

    clip_id: str
    turn_index: int
    prefix_len: int
    is_turn_end: bool
    committed: bool
    p_adjusted: float
    reasons: tuple[str, ...]

    @property
    def fired(self) -> bool:
        """Did any lexical rule have an opinion here?"""
        return bool(self.reasons) and self.reasons not in (
            ("no_rules_fired",),
            ("tail_absent",),
            ("tail_stale",),
        )


@dataclass
class Scores:
    n_turns: int = 0
    n_prefixes: int = 0
    n_ends: int = 0

    #: Committed while the speaker still had more to say. The agent would have
    #: talked over them. Lower is better; this is the number that matters.
    premature_commits: int = 0

    #: Failed to commit at a real turn end. The agent would have left the
    #: caller hanging. This is the cost of patience.
    missed_ends: int = 0

    #: How often the lexical layer had an opinion at all. Caps how much it can
    #: possibly be worth.
    fired_on_prefixes: int = 0
    fired_on_ends: int = 0

    #: Of the points where a rule fired, did it push the probability the right
    #: way -- down mid-turn, up at a turn end?
    right_direction: int = 0
    wrong_direction: int = 0

    reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def premature_commit_rate(self) -> float:
        return self.premature_commits / self.n_prefixes if self.n_prefixes else 0.0

    @property
    def missed_end_rate(self) -> float:
        return self.missed_ends / self.n_ends if self.n_ends else 0.0

    @property
    def coverage(self) -> float:
        total = self.n_prefixes + self.n_ends
        fired = self.fired_on_prefixes + self.fired_on_ends
        return fired / total if total else 0.0

    @property
    def directional_accuracy(self) -> float:
        total = self.right_direction + self.wrong_direction
        return self.right_direction / total if total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "n_turns": self.n_turns,
            "n_prefixes": self.n_prefixes,
            "n_ends": self.n_ends,
            "premature_commit_rate": self.premature_commit_rate,
            "missed_end_rate": self.missed_end_rate,
            "coverage": self.coverage,
            "directional_accuracy": self.directional_accuracy,
            "premature_commits": self.premature_commits,
            "missed_ends": self.missed_ends,
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


def replay(
    turns: list[Turn],
    config: BaobabConfig,
    index: MarkerIndex | None,
    p_inner: float = DEFAULT_P_INNER,
    max_prefixes: int | None = None,
) -> list[Outcome]:
    """Walk every turn word by word, asking the guard at each step."""
    out: list[Outcome] = []
    for turn in turns:
        if not turn.ends_turn:
            # The last turn of a file: we never observed whether the speaker
            # was finished, so it is not evidence in either direction.
            continue
        n = len(turn.words)
        # Cap the prefixes per turn so one very long turn cannot dominate the
        # rate. Prefixes nearest the end are the informative ones.
        start = 1 if max_prefixes is None else max(1, n - max_prefixes)
        for k in range(start, n + 1):
            is_end = k == n
            result = guard(
                GuardInput(
                    p_inner=p_inner,
                    transcript_tail=" ".join(turn.words[:k]),
                    is_final=is_end,
                    tail_age_ms=0.0,
                    language=index.pack.code if index is not None else None,
                ),
                config,
                index,
            )
            p = result.p_adjusted if result.p_adjusted is not None else p_inner
            out.append(
                Outcome(
                    clip_id=turn.clip_id,
                    turn_index=turn.index,
                    prefix_len=k,
                    is_turn_end=is_end,
                    committed=result.decision,
                    p_adjusted=p,
                    reasons=result.reasons,
                )
            )
    return out


def score(outcomes: list[Outcome], p_inner: float = DEFAULT_P_INNER) -> Scores:
    s = Scores()
    seen_turns: set[tuple[str, int]] = set()
    for o in outcomes:
        seen_turns.add((o.clip_id, o.turn_index))
        for reason in o.reasons:
            # Strip the matched phrase so the counts stay readable.
            key = reason.split(":", 1)[0]
            s.reason_counts[key] = s.reason_counts.get(key, 0) + 1

        if o.is_turn_end:
            s.n_ends += 1
            if not o.committed:
                s.missed_ends += 1
            if o.fired:
                s.fired_on_ends += 1
                # At a real turn end the probability should rise.
                if o.p_adjusted > p_inner:
                    s.right_direction += 1
                elif o.p_adjusted < p_inner:
                    s.wrong_direction += 1
        else:
            s.n_prefixes += 1
            if o.committed:
                s.premature_commits += 1
            if o.fired:
                s.fired_on_prefixes += 1
                # Mid-turn the probability should fall.
                if o.p_adjusted < p_inner:
                    s.right_direction += 1
                elif o.p_adjusted > p_inner:
                    s.wrong_direction += 1

    s.n_turns = len(seen_turns)
    return s


def run(
    corpus_path: str,
    pack_code: str = "pcm",
    p_inner: float = DEFAULT_P_INNER,
    max_prefixes: int | None = 12,
    ablate: bool = False,
) -> dict[str, object]:
    """Run the benchmark, optionally ablating each feature in turn."""
    turns = load_transcripts(corpus_path)
    if not turns:
        raise SystemExit(f"No speaker-labelled transcripts found in {corpus_path!r}")

    pack = resolve(pack_code)
    if pack is None:
        raise SystemExit(f"No pack for {pack_code!r}. Try: pcm, en-NG")
    index = MarkerIndex(pack)

    conditions: dict[str, tuple[BaobabConfig, MarkerIndex | None]] = {
        # No pack at all: the guard's null-pack path, i.e. exactly stock
        # behaviour. This is the honest floor.
        "baseline (no pack)": (BaobabConfig(), None),
        "markers on": (BaobabConfig(), index),
    }
    if ablate:
        conditions["markers off"] = (BaobabConfig(enable_markers=False), index)

    results: dict[str, object] = {
        "corpus": corpus_path,
        "pack": pack.code,
        "pack_status": pack.status,
        "pack_labelled_turns": pack.labelled_turns,
        "p_inner": p_inner,
        "max_prefixes_per_turn": max_prefixes,
        "conditions": {},
    }
    for name, (config, idx) in conditions.items():
        outcomes = replay(turns, config, idx, p_inner, max_prefixes)
        results["conditions"][name] = score(outcomes, p_inner).as_dict()  # type: ignore[index]
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="directory or glob of speaker-labelled transcripts")
    ap.add_argument("--pack", default="pcm", help="language pack code (default: pcm)")
    ap.add_argument("--p-inner", type=float, default=DEFAULT_P_INNER,
                    help="constant inner probability (default: the decision threshold)")
    ap.add_argument("--max-prefixes", type=int, default=12,
                    help="cap prefixes scored per turn so long turns do not dominate")
    ap.add_argument("--ablate", action="store_true", help="also run with markers disabled")
    ap.add_argument("--json", dest="json_path", help="write raw results here")
    ap.add_argument("--markdown", dest="md_path", help="write the report here")
    args = ap.parse_args(argv)

    results = run(
        args.corpus,
        pack_code=args.pack,
        p_inner=args.p_inner,
        max_prefixes=args.max_prefixes if args.max_prefixes > 0 else None,
        ablate=args.ablate,
    )

    from .report import to_markdown

    report = to_markdown(results)
    print(report)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json_path}", file=sys.stderr)
    if args.md_path:
        with open(args.md_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"wrote {args.md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
