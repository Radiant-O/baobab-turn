"""Watch a chat agent decide when to reply.

    python examples/chat_demo.py

Replays scripted conversations against a controllable clock, so a thirty
second pause costs nothing to demonstrate. Each line shows what the guard
would have done at that moment and why.

The case worth watching is the first one. A user sends "so", then "the thing
be say", then goes quiet for thirty seconds. An inactivity timeout -- which
is what most chat bots use -- answers into the middle of that sentence. This
does not, because the words say the clause has not arrived yet.
"""

from __future__ import annotations

from baobab_turn.adapters.chat import ChatTurnGuard

# (event, argument) pairs. "wait" advances the clock in seconds.
SCRIPTS: list[tuple[str, list[tuple[str, object]]]] = [
    (
        "Unfinished thought, then a long silence",
        [
            ("typing", True),
            ("message", "so"),
            ("wait", 0.4),
            ("typing", True),
            ("message", "the thing be say"),
            ("wait", 30.0),
        ],
    ),
    (
        "Finished clearly, hands over with 'abi'",
        [
            ("message", "i no fit come today abi"),
            ("wait", 0.5),
            ("wait", 2.0),
        ],
    ),
    (
        "Plain statement -- timing decides, no marker involved",
        [
            ("message", "i go come tomorrow"),
            ("wait", 0.5),
            ("wait", 2.0),
        ],
    ),
    (
        "Still typing, even though the words look finished",
        [
            ("message", "e don finish abi"),
            ("wait", 3.0),
            ("typing", True),
        ],
    ),
]


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


def run(title: str, events: list[tuple[str, object]]) -> None:
    clock = Clock()
    chat = ChatTurnGuard(language="pcm", now=clock)

    print(f"\n{title}")
    print("-" * len(title))

    for kind, value in events:
        if kind == "message":
            chat.on_message(str(value))
            label = f'user: "{value}"'
        elif kind == "typing":
            chat.on_typing(bool(value))
            label = "user is typing..."
        else:
            clock.tick(float(value))  # type: ignore[arg-type]
            label = f"({value}s pass)"

        d = chat.should_reply()
        verdict = "REPLY NOW" if d.reply else f"wait {d.wait_ms}ms"
        print(f"  {label:36} {verdict:14} p={d.probability:.2f}  {d.text}")


def main() -> int:
    print("Nigerian Pidgin chat. A stock bot replies on an inactivity timeout;")
    print("this one also reads what was actually said.")
    for title, events in SCRIPTS:
        run(title, events)
    print("\nThe first script is the point: thirty seconds of silence, and it")
    print("still waits, because 'the thing be say' means the sentence is not over.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
