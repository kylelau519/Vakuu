"""Tests for state_key.py. Needs the sts2-cli engine (one in-process session), so it is slow-ish:
    pytest agents/whispering_earring/test_state_key.py
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
import buffer_layout as bl  # noqa: E402
from buffer_env import Sts2BufferSession, decode_token  # noqa: E402
from installation import ensure_setup  # noqa: E402
from policy import choose_generic_action  # noqa: E402
from state_key import combine, state_key, visible_key  # noqa: E402


@pytest.fixture(scope="module")
def session():
    ensure_setup()
    s = Sts2BufferSession()
    yield s
    s.close()


def _find(decision, prefix: str) -> int | None:
    for idx, slot in bl.hand_cards(decision.arr):
        card_id = decode_token(bl.card_token(slot)) or ""
        if card_id.startswith(prefix) and bl.hand_can_play(slot):
            return idx
    return None


def _first_combat(session, seed: str):
    """-> (character, seed, actions) of the first combat_play decision of a generic-policy run, or None."""
    random.seed(0)
    d = session.start("Ironclad", seed, 1)
    for _ in range(300):
        if d.decision == "combat_play":
            return list(session.action_log)
        if d.decision == "game_over":
            return None
        name, kwargs, _, _ = choose_generic_action(d, None)
        d = session.step(name, **kwargs)
    return None


def _strike_and_defend_combat(session):
    for n in range(40):
        seed = f"statekey-{n}"
        prefix = _first_combat(session, seed)
        if prefix is None:
            continue
        d = session.open_from_checkpoint("Ironclad", seed, 1, prefix)
        if _find(d, "STRIKE") is not None and _find(d, "DEFEND") is not None:
            return seed, prefix
    pytest.skip("no seed with Strike+Defend in the opening hand")


def _play_by_name(session, d, first: str, second: str):
    for prefix in (first, second):
        idx = _find(d, prefix)
        assert idx is not None, prefix
        d = session.step("play_card", card_index=idx)
    return d


def test_same_cards_different_order_same_key(session):
    seed, prefix = _strike_and_defend_combat(session)
    d = session.open_from_checkpoint("Ironclad", seed, 1, prefix)
    start_key = state_key(session, d)
    strike_defend = _play_by_name(session, d, "STRIKE", "DEFEND")
    k1 = state_key(session, strike_defend)

    d = session.open_from_checkpoint("Ironclad", seed, 1, prefix)
    assert state_key(session, d) == start_key  # reloading is deterministic, key included
    defend_strike = _play_by_name(session, d, "DEFEND", "STRIKE")
    k2 = state_key(session, defend_strike)

    assert k1 == k2
    assert k1 != start_key  # and the key does see that something was played

    # and the merge survives into the next turn: same RNG consumed, same draw
    n1 = state_key(session, session.step("end_turn"))
    d = session.open_from_checkpoint("Ironclad", seed, 1, prefix)
    d = _play_by_name(session, d, "DEFEND", "STRIKE")
    assert state_key(session, session.step("end_turn")) == n1


def test_different_cards_different_key(session):
    seed, prefix = _strike_and_defend_combat(session)
    d = session.open_from_checkpoint("Ironclad", seed, 1, prefix)
    only_strike = state_key(session, session.step("play_card", card_index=_find(d, "STRIKE")))
    d = session.open_from_checkpoint("Ironclad", seed, 1, prefix)
    only_defend = state_key(session, session.step("play_card", card_index=_find(d, "DEFEND")))
    assert only_strike != only_defend


def test_hidden_state_is_read_only_and_stable(session):
    seed, prefix = _strike_and_defend_combat(session)
    d = session.open_from_checkpoint("Ironclad", seed, 1, prefix)
    assert session.hidden_state_key() == session.hidden_state_key() != ""
    assert state_key(session, d) == state_key(session, d)


def test_rng_counter_changes_key(session):
    seed, prefix = _strike_and_defend_combat(session)
    d = session.open_from_checkpoint("Ironclad", seed, 1, prefix)
    visible, hidden = visible_key(d.arr), session.hidden_state_key()
    assert "Counter:" in hidden
    # one more draw on any stream: same visible state, same draw pile, different counter
    bumped = hidden.replace("run.Shuffle=Counter: ", "run.Shuffle=Counter: 1", 1)
    assert bumped != hidden
    assert combine(visible, hidden) != combine(visible, bumped)


def test_draw_pile_order_changes_key(session):
    seed, prefix = _strike_and_defend_combat(session)
    d = session.open_from_checkpoint("Ironclad", seed, 1, prefix)
    visible = visible_key(d.arr)
    before = session.hidden_state_key()
    assert "draw=" in before
    before_draw = [l for l in before.splitlines() if l.startswith("draw=")][0]
    cards = before_draw[len("draw="):].split(",")
    assert len(set(cards)) > 1, "need a draw pile with two different cards to reorder"
    swapped = cards[:]
    i = next(i for i in range(1, len(cards)) if cards[i] != cards[0])
    swapped[0], swapped[i] = swapped[i], swapped[0]
    other = before.replace(before_draw, "draw=" + ",".join(swapped))
    assert combine(visible, before) != combine(visible, other)

    # the real engine agrees: RandomizeDrawPile reorders the same cards (same multiset => same visible key)
    session.randomize_draw_pile(d, seed=12345)
    d2 = session._decision(session._call(session._session.GetState)) if hasattr(session._session, "GetState") else None
    if d2 is not None:
        assert visible_key(d2.arr) == visible
    assert session.hidden_state_key() != before
