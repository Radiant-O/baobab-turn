"""Pins the shape of the private LiveKit API we depend on.

`livekit/agents/voice/turn.py` is not exported from `livekit.agents`, so no
deprecation policy covers it and any minor release can change it underneath
us. We accepted that risk because wrapping the detector is the only way to
reach the end-of-turn probability at all -- `EotPredictionEvent` exists
internally but is absent from the public `EventTypes` and is forwarded only
to LiveKit's own session host.

Having accepted it, this file makes the risk visible: CI tells us the day an
SDK bump breaks the adapter, rather than a production call telling us. A
failure here is not necessarily a bug in our code -- read the SDK diff first.
"""

from __future__ import annotations

import asyncio
import inspect
import typing

import pytest

pytest.importorskip("livekit.agents", reason="needs livekit-agents")

from livekit.agents.voice import turn as lk_turn


def test_the_streaming_protocols_still_exist() -> None:
    """If these are renamed, the adapter targets nothing."""
    assert hasattr(lk_turn, "_StreamingTurnDetector")
    assert hasattr(lk_turn, "_StreamingTurnDetectorStream")
    assert hasattr(lk_turn, "TurnDetectionEvent")


def test_the_protocols_are_still_runtime_checkable() -> None:
    """The SDK dispatches with isinstance against these Protocols, which is
    what lets a duck-typed wrapper work at all. Lose @runtime_checkable and
    isinstance raises TypeError instead of returning False."""
    for name in ("_StreamingTurnDetector", "_StreamingTurnDetectorStream"):
        proto = getattr(lk_turn, name)
        assert getattr(proto, "_is_runtime_protocol", False), (
            f"{name} is no longer @runtime_checkable"
        )


REQUIRED_STREAM_MEMBERS = {
    # name: is it a coroutine function
    "unlikely_threshold": True,
    "backchannel_threshold": True,
    "supports_language": True,
    "predict": False,
    "cancel_inference": False,
    "flush": False,
    "push_audio": False,
    "end_input": False,
    "aclose": True,
}

REQUIRED_STREAM_PROPERTIES = ("model", "provider", "is_fallback", "prediction_timeout")


@pytest.mark.parametrize("name,is_async", sorted(REQUIRED_STREAM_MEMBERS.items()))
def test_stream_member_exists_with_the_expected_sync_or_async_shape(
    name: str, is_async: bool
) -> None:
    """Sync vs async matters concretely: the adapter awaits some of these and
    calls others directly, and getting it wrong yields an un-awaited
    coroutine that silently does nothing."""
    member = getattr(lk_turn._StreamingTurnDetectorStream, name, None)
    assert member is not None, f"stream protocol lost {name!r}"
    assert inspect.iscoroutinefunction(member) is is_async, (
        f"{name} changed between sync and async"
    )


@pytest.mark.parametrize("name", REQUIRED_STREAM_PROPERTIES)
def test_stream_property_exists(name: str) -> None:
    assert isinstance(
        getattr(lk_turn._StreamingTurnDetectorStream, name, None), property
    ), f"stream protocol lost the {name!r} property"


def test_detector_still_builds_streams_via_stream() -> None:
    member = getattr(lk_turn._StreamingTurnDetector, "stream", None)
    assert member is not None
    params = inspect.signature(member).parameters
    assert "conn_options" in params, "stream() lost its conn_options kwarg"
    assert params["conn_options"].kind is inspect.Parameter.KEYWORD_ONLY


def test_predict_still_returns_a_future() -> None:
    """The adapter returns its own future so a guard failure can resolve with
    a safe default. If this became a coroutine, that whole design changes."""
    predict = lk_turn._StreamingTurnDetectorStream.predict
    assert not inspect.iscoroutinefunction(predict)
    hints = typing.get_type_hints(predict)
    assert hints["return"] is asyncio.Future or typing.get_origin(
        hints["return"]
    ) is asyncio.Future, f"predict() return type changed to {hints['return']!r}"


def test_cancel_inference_still_takes_timed_out_keyword() -> None:
    params = inspect.signature(lk_turn._StreamingTurnDetectorStream.cancel_inference).parameters
    assert "timed_out" in params
    assert params["timed_out"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_event_still_carries_the_fields_we_read_and_write() -> None:
    """`end_of_turn_probability` is the field the whole package exists to
    adjust. The rest we copy through, so losing one silently drops data."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(lk_turn.TurnDetectionEvent)}
    for required in (
        "type",
        "end_of_turn_probability",
        "last_speaking_time",
        "detection_delay",
        "inference_duration",
        "backchannel_probability",
    ):
        assert required in fields, f"TurnDetectionEvent lost {required!r}"


def test_the_event_is_still_mutable() -> None:
    """The adapter adjusts the probability in place and returns the same
    object, which keeps every other field for free. A frozen dataclass here
    would need a different approach."""
    import dataclasses

    params = getattr(lk_turn.TurnDetectionEvent, "__dataclass_params__", None)
    assert params is not None
    assert not params.frozen, "TurnDetectionEvent became frozen"
    assert dataclasses.is_dataclass(lk_turn.TurnDetectionEvent)


def test_the_probability_is_still_a_plain_float() -> None:
    hints = typing.get_type_hints(lk_turn.TurnDetectionEvent)
    assert hints["end_of_turn_probability"] is float


def test_prediction_timeout_default_is_still_around_one_second() -> None:
    """Our 15ms guard budget is chosen against this. If the SDK's tolerance
    shrank dramatically the budget would need revisiting."""
    from livekit.agents.inference.eot import base as eot_base

    timeout = getattr(eot_base, "DEFAULT_PREDICTION_TIMEOUT", None)
    if timeout is None:
        pytest.skip("DEFAULT_PREDICTION_TIMEOUT moved; check the SDK")
    assert 0.3 <= float(timeout) <= 5.0, f"prediction timeout is now {timeout}"
