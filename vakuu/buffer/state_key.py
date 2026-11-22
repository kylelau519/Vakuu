"""Equality key for a combat position, so rollouts that reached the same outcome can be merged.

Two positions are the same only if BOTH parts match (modeled on spire-zero's StateKey + HiddenKey):
  visible  the whole decision buffer, with hand / discard / exhaust / draw pile / potions / deck as sorted
           multisets of card slots. The buffer carries no per-card uids, so a card IS its slot. The ORDER of
           this turn's plays is not part of it -- that is the point of merging.
  hidden   the engine's own hidden-state key (claws/job.py takes it from the engine): every RNG stream's counter
           and state, the draw pile IN ORDER, each monster's queued move and its own RNG. Two orders that look
           identical but consumed a different number of random draws therefore get different keys.

NEVER feed any of this to a net: hidden state is exactly the look-ahead a policy must not have. The key is
for equality only, and computing it is read-only (no RNG stream advances).
"""
from __future__ import annotations

import hashlib

import numpy as np

from vakuu.buffer import layout as bl

_CP = bl.COMBAT_PLAY
_HAND0 = _CP + bl._CP_HAND_OFFSET
_HAND_N = bl.MAX_HAND_CARDS * bl.HAND_SLOT_SIZE
_PILE_N = bl.MAX_DECK_CARDS * bl.CARD_SIZE
_PILES = tuple(_CP + off for off in (bl._CP_DRAW_PILE_OFFSET, bl._CP_DISCARD_PILE_OFFSET, bl._CP_EXHAUST_PILE_OFFSET))
_POT0 = bl.PLAYER_SUMMARY + bl._PLAYER_POTIONS_OFFSET
_POT_N = bl.MAX_POTIONS * bl.POTION_SIZE
_DECK0 = bl.PLAYER_SUMMARY + bl._PLAYER_DECK_OFFSET

# The engine's history counter "card_drawn" is not reproducible across reloads of the SAME position: the
# first read in a fresh session shows 0 where every later reload shows 5 (measured; everything else in
# the buffer, and the whole hidden state, was identical). It would make identical positions look different,
# so it is masked -- player and every enemy slot. The other history counters are kept.
_DRAWN = bl.HISTORY_COUNTER_NAMES.index("card_drawn") * 2
_MASKED = [_CP + bl._CP_PLAYER_HISTORY_OFFSET + _DRAWN + j for j in (0, 1)] + [
    _CP + bl._CP_ENEMY_HISTORY_OFFSET + e * bl.HISTORY_CREATURE_SIZE + _DRAWN + j
    for e in range(bl.MAX_ENEMIES) for j in (0, 1)
]


def _sorted_rows(arr: np.ndarray, start: int, n_rows: int, size: int) -> bytes:
    rows = [arr[start + i * size: start + (i + 1) * size].tobytes() for i in range(n_rows)]
    return b"".join(sorted(rows))


def visible_key(arr: np.ndarray) -> bytes:
    """The combat decision's own sections (decision tag, player summary, run context, combat play), with every
    order-carrying card region replaced by its sorted multiset.

    Only those sections: the buffer is reused across decisions and a section that was not refreshed keeps
    whatever an EARLIER decision wrote there (a previous fork's card reward, the map, ...), so hashing the
    whole thing makes identical positions look different depending on what the session did before."""
    a = arr.copy()
    a[bl.MAP:bl.RUN_CONTEXT] = 0
    a[bl.CARD_SELECT:] = 0
    multisets = [
        _sorted_rows(a, _HAND0, bl.MAX_HAND_CARDS, bl.HAND_SLOT_SIZE),
        _sorted_rows(a, _POT0, bl.MAX_POTIONS, bl.POTION_SIZE),
        _sorted_rows(a, _DECK0, bl.MAX_DECK_CARDS, bl.CARD_SIZE),
        *(_sorted_rows(a, p, bl.MAX_DECK_CARDS, bl.CARD_SIZE) for p in _PILES),
    ]
    a[_MASKED] = 0
    a[_HAND0:_HAND0 + _HAND_N] = 0
    a[_POT0:_POT0 + _POT_N] = 0
    a[_DECK0:_DECK0 + _PILE_N] = 0
    for p in _PILES:
        a[p:p + _PILE_N] = 0
    return a.tobytes() + b"|" + b"|".join(multisets)


def combine(visible: bytes, hidden: str) -> str:
    """The key proper: hex digest of both parts. Split out so tests can vary one part at a time."""
    h = hashlib.sha1()
    h.update(len(visible).to_bytes(8, "little"))
    h.update(visible)
    h.update(hidden.encode())
    return h.hexdigest()
