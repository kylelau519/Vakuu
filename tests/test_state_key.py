"""state_key.py on synthetic combat records (no engine): the visible part of a position's key ignores what must not matter (the
order of cards inside a hand / pile, sections an earlier decision left behind, the unreproducible card_drawn counter) and sees
everything that does."""
from __future__ import annotations

import numpy as np

from vakuu.buffer import layout as bl
from vakuu.buffer.state_key import combine, visible_key

_CP = bl.COMBAT_PLAY


def _record() -> np.ndarray:
    a = np.zeros(bl.TOTAL_SIZE, np.float32)
    a[bl.DECISION_TYPE] = 3  # combat_play
    return a


def _set_cards(a: np.ndarray, start: int, size: int, tokens: list[int]) -> None:
    for i, tok in enumerate(tokens):
        a[start + i * size + bl.CARD_TOKEN] = tok
        a[start + i * size + bl.CARD_COST] = 1


def _hand(a: np.ndarray, tokens: list[int]) -> None:
    _set_cards(a, _CP + bl._CP_HAND_OFFSET, bl.HAND_SLOT_SIZE, tokens)


def _draw_pile(a: np.ndarray, tokens: list[int]) -> None:
    _set_cards(a, _CP + bl._CP_DRAW_PILE_OFFSET, bl.CARD_SIZE, tokens)


def test_hand_order_does_not_matter():
    a, b = _record(), _record()
    _hand(a, [5, 9, 5, 12])
    _hand(b, [12, 5, 5, 9])
    assert visible_key(a) == visible_key(b)


def test_pile_order_does_not_matter_but_content_does():
    a, b, c = _record(), _record(), _record()
    _draw_pile(a, [3, 4, 5])
    _draw_pile(b, [5, 3, 4])
    _draw_pile(c, [3, 4, 6])
    assert visible_key(a) == visible_key(b)
    assert visible_key(a) != visible_key(c)


def test_different_hand_changes_key():
    a, b = _record(), _record()
    _hand(a, [5, 9])
    _hand(b, [5, 10])
    assert visible_key(a) != visible_key(b)


def test_card_drawn_counter_is_masked_but_other_counters_are_not():
    drawn = bl.HISTORY_COUNTER_NAMES.index("card_drawn") * 2
    other = next(i for i, n in enumerate(bl.HISTORY_COUNTER_NAMES) if n != "card_drawn") * 2
    base = _record()
    masked, kept = base.copy(), base.copy()
    masked[_CP + bl._CP_PLAYER_HISTORY_OFFSET + drawn] = 5
    kept[_CP + bl._CP_PLAYER_HISTORY_OFFSET + other] = 5
    assert visible_key(masked) == visible_key(base)
    assert visible_key(kept) != visible_key(base)


def test_sections_of_earlier_decisions_do_not_matter():
    base = _record()
    stale = base.copy()
    stale[bl.MAP + 3] = 7            # an earlier map decision
    stale[bl.CARD_SELECT + 4] = 2    # an earlier card prompt
    assert visible_key(stale) == visible_key(base)


def test_hidden_state_is_part_of_the_key():
    v = visible_key(_record())
    assert combine(v, "rng=1") == combine(v, "rng=1")
    assert combine(v, "rng=1") != combine(v, "rng=2")
    assert combine(v, "rng=1") != combine(visible_key(_record() + 1), "rng=1")
