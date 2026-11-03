"""Ultravox, where the honest answer is "one number".

The test that matters most here is the one asserting a *limit*:
`test_an_unknown_language_gets_ultravox_s_own_default`. This module can only
make a call more patient, never less, and an unknown language must come out
exactly as an unconfigured Ultravox call would. A configuration helper that
quietly degrades the stock behaviour is worse than no helper.
"""

from __future__ import annotations

import pytest

from baobab_turn.adapters.ultravox import (
    MAX_TURN_ENDPOINT_DELAY_MS,
    TURN_ENDPOINT_DELAY_QUANTUM_MS,
    ULTRAVOX_DEFAULT_DELAY_MS,
    delay_ms_for,
    vad_settings,
)
from baobab_turn.core.config import BaobabConfig
from baobab_turn.core.types import LanguagePack


def test_ultravox_default_is_what_their_docs_say() -> None:
    """384ms, verbatim from docs.ultravox.ai/noise/understanding-vad. If this
    changes upstream, every number this module produces is off."""
    assert ULTRAVOX_DEFAULT_DELAY_MS == 384
    assert TURN_ENDPOINT_DELAY_QUANTUM_MS == 32


def test_an_unknown_language_gets_ultravox_s_own_default() -> None:
    """The floor. No pack means no opinion, and no opinion must mean stock
    behaviour rather than something we invented."""
    assert vad_settings("xx-YY") == {"turnEndpointDelay": "0.384s"}
    assert vad_settings(None) == {"turnEndpointDelay": "0.384s"}


def test_a_non_tonal_pack_changes_nothing() -> None:
    """pcm is not tonal, and nothing else in a pack applies without a
    transcript. Patience we cannot justify is patience we do not add."""
    assert vad_settings("pcm") == {"turnEndpointDelay": "0.384s"}


def test_a_tonal_pack_buys_patience() -> None:
    pack = LanguagePack(code="yo", name="Yoruba", tonal=True, tonal_extra_ms=160)
    assert delay_ms_for(pack) == 544  # 384 + 160, already a multiple of 32


def test_a_tonal_pack_without_its_own_extra_uses_the_global_default() -> None:
    pack = LanguagePack(code="ig", name="Igbo", tonal=True)
    expected = 384 + BaobabConfig().tonal_language_extra_ms  # 384 + 150 = 534
    assert delay_ms_for(pack) == 544  # 534 quantised to the nearest 32ms step
    assert expected == 534


def test_every_value_lands_on_a_meaningful_step() -> None:
    """Ultravox documents that only multiples of 32ms matter. Reporting
    0.534s when the service will treat it as 0.544s would make the number we
    print subtly untrue."""
    for extra in range(0, 900, 7):
        pack = LanguagePack(code="t", name="T", tonal=True, tonal_extra_ms=extra)
        assert delay_ms_for(pack) % TURN_ENDPOINT_DELAY_QUANTUM_MS == 0


def test_patience_is_capped() -> None:
    """Past a point the agent feels broken rather than polite, and Ultravox
    imposes no ceiling of its own."""
    pack = LanguagePack(code="t", name="T", tonal=True, tonal_extra_ms=60_000)
    assert delay_ms_for(pack) == MAX_TURN_ENDPOINT_DELAY_MS


def test_it_can_never_make_a_call_less_patient_than_stock() -> None:
    """The one-way guarantee. A pack asking for less than Ultravox's default
    would make using this package actively worse than not using it."""
    pack = LanguagePack(code="t", name="T", tonal=True, max_delay_ms=100)
    assert delay_ms_for(pack) == ULTRAVOX_DEFAULT_DELAY_MS


def test_a_pack_floor_raises_the_base() -> None:
    pack = LanguagePack(code="t", name="T", min_delay_ms=800)
    assert delay_ms_for(pack) == 800


def test_the_tonal_rule_can_be_switched_off() -> None:
    """Every flag is a hypothesis under test, including on a host where it
    is the only flag that reaches."""
    pack = LanguagePack(code="yo", name="Yoruba", tonal=True, tonal_extra_ms=160)
    assert delay_ms_for(pack, BaobabConfig(enable_tonal=False)) == 384


def test_only_turn_endpoint_delay_is_emitted() -> None:
    """The other three vadSettings fields have nothing to do with language.
    Emitting a value we cannot justify, inside a dict that looks
    authoritative, is how a config helper does damage."""
    assert set(vad_settings("pcm")) == {"turnEndpointDelay"}


@pytest.mark.parametrize("language", ["pcm", "en-NG", None, "zz"])
def test_the_result_is_always_in_ultravox_s_own_format(language) -> None:
    """Seconds, three decimals, with an "s" suffix -- what the API expects."""
    value = vad_settings(language)["turnEndpointDelay"]
    assert value.endswith("s")
    seconds = float(value[:-1])
    assert 0.384 <= seconds <= MAX_TURN_ENDPOINT_DELAY_MS / 1000
