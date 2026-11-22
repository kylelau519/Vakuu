"""fast_tokens.fast_pile_tensors must equal claws/model.py's _pile_tensors exactly (it only replaces the slow card-by-card read)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from vakuu.buffer import layout as bl
from vakuu.claws import fast_tokens, model
from vakuu.common.tokenizer import EntityVocab
from vakuu.engine import tables


@pytest.fixture(scope="module")
def gvocab():
    return model.GlobalVocab(EntityVocab.load())


def _random_record(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros(bl.TOTAL_SIZE, np.float32)
    card_tokens = [t for t, key in tables.TOKEN_MAP.items() if key.startswith("CARDS:")]
    cp = bl.COMBAT_PLAY
    for p, off in enumerate((bl._CP_DRAW_PILE_OFFSET, bl._CP_DISCARD_PILE_OFFSET, bl._CP_EXHAUST_PILE_OFFSET)):
        n = int(rng.integers(0, 30))
        a[cp + bl._CP_PILE_COUNTS_OFFSET + p] = n
        for i in range(n):
            o = cp + off + i * bl.CARD_SIZE
            a[o + bl.CARD_TOKEN] = rng.choice(card_tokens)
            a[o + bl.CARD_COST] = rng.integers(-1, 4)
            a[o + bl.CARD_UPGRADED] = rng.integers(0, 2)
            a[o + bl.CARD_TYPE] = rng.integers(0, 7)
    return a


@pytest.mark.parametrize("seed", range(5))
def test_fast_matches_slow(seed, gvocab):
    arr = _random_record(seed)
    slow = model._pile_tensors(arr, gvocab)
    fast = fast_tokens.fast_pile_tensors(arr, gvocab)
    assert slow.keys() == fast.keys()
    for k in slow:
        assert torch.equal(slow[k], fast[k]), k
