"""Pile-token tests (no engine needed -- synthetic tensors):
    pytest agents/claws/model/test_piles.py
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (  # noqa: E402
    BASE_TOKEN_TYPES, FEATURE_DIM, MAX_PILE, PILE_TOKEN_TYPES, TOKEN_TYPES, ClawsNet, GlobalVocab,
    TokenBatch,
)
from common.tokenizer import EntityVocab  # noqa: E402


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    n = ClawsNet(GlobalVocab(EntityVocab.load()))
    return n.eval()


def _make_piles(cards_per_pile: list[list[tuple[int, int, int, int]]], slots: list[int]):
    """cards_per_pile[p] = [(global_id, cost, upgraded, type)...]; placed in the given slot indices."""
    keys = {"pile_ids": torch.zeros(3, MAX_PILE, dtype=torch.long), "pile_cost": torch.zeros(3, MAX_PILE),
            "pile_upg": torch.zeros(3, MAX_PILE), "pile_type": torch.zeros(3, MAX_PILE, dtype=torch.long),
            "pile_mask": torch.zeros(3, MAX_PILE, dtype=torch.bool)}
    counts = []
    for p, cards in enumerate(cards_per_pile):
        counts.append(float(len(cards)))
        for slot, (gid, cost, upg, ctype) in zip(slots, cards):
            keys["pile_ids"][p, slot], keys["pile_cost"][p, slot] = gid, cost
            keys["pile_upg"][p, slot], keys["pile_type"][p, slot] = upg, ctype
            keys["pile_mask"][p, slot] = True
    keys["pile_counts"] = torch.tensor(counts)
    return keys


def _batch(piles) -> TokenBatch:
    types = [TOKEN_TYPES.index("PLAYER"), TOKEN_TYPES.index("ENEMY")] + [
        TOKEN_TYPES.index(t) for t in PILE_TOKEN_TYPES
    ]
    feats = torch.zeros(len(types), FEATURE_DIM)
    feats[0, 0], feats[1, 0] = 1.0, 0.5
    return TokenBatch(
        type_ids=torch.tensor(types), global_ids=torch.tensor([0, 3, 0, 0, 0]), group_ids=torch.zeros(len(types), dtype=torch.long), features=feats,
        player_pos=0, enemy_pos={0: 1}, piles=piles,
    )


CARDS = [[(5, 1, 0, 1), (5, 1, 0, 1), (7, 2, 1, 2), (9, 0, 0, 2)],
         [(5, 1, 0, 1), (11, 3, 0, 3)],
         []]


def test_padding_and_order_invariant(net):
    ref = net.encode(_batch(_make_piles(CARDS, slots=[0, 1, 2, 3])))
    # same cards, scattered through different padded slots (padding amount / placement differs)
    scattered = net.encode(_batch(_make_piles(CARDS, slots=[40, 3, 17, 29])))
    # reversed card order within each pile
    rev = net.encode(_batch(_make_piles([c[::-1] for c in CARDS], slots=[0, 1, 2, 3])))
    assert torch.allclose(ref, scattered, atol=1e-5)
    assert torch.allclose(ref, rev, atol=1e-5)


def test_pile_contents_matter(net):
    ref = net.encode(_batch(_make_piles(CARDS, slots=[0, 1, 2, 3])))
    changed = [list(c) for c in CARDS]
    changed[0][3] = (13, 0, 0, 2)  # different card in the draw pile
    other = net.encode(_batch(_make_piles(changed, slots=[0, 1, 2, 3])))
    assert not torch.allclose(ref, other, atol=1e-4)


def test_batched_matches_single_and_padding_rows(net):
    b1 = _batch(_make_piles(CARDS, slots=[0, 1, 2, 3]))
    b2 = _batch(_make_piles([CARDS[1], CARDS[0], CARDS[2]], slots=[0, 1, 2, 3]))
    single = [net.encode(b) for b in (b1, b2)]
    n = b1.type_ids.shape[0]
    pad = 3  # extra padded token rows
    type_ids = torch.zeros(2, n + pad, dtype=torch.long)
    gids = torch.zeros(2, n + pad, dtype=torch.long)
    owners = torch.zeros(2, n + pad, dtype=torch.long)
    feats = torch.zeros(2, n + pad, FEATURE_DIM)
    pad_mask = torch.ones(2, n + pad, dtype=torch.bool)
    for i, b in enumerate((b1, b2)):
        type_ids[i, :n], gids[i, :n], feats[i, :n], pad_mask[i, :n] = b.type_ids, b.global_ids, b.features, False
    piles = {k: torch.stack([b1.piles[k], b2.piles[k]]) for k in b1.piles}
    out = net.encode_batched(type_ids, gids, feats, pad_mask, piles, owners)
    for i in range(2):
        assert torch.allclose(out[i, :n], single[i], atol=1e-5)


def test_old_checkpoint_loads(net):
    old = {k: v.clone() for k, v in net.state_dict().items() if not k.startswith("pile_encoder.")}
    old["type_emb.weight"] = old["type_emb.weight"][: len(BASE_TOKEN_TYPES)]
    fresh = ClawsNet(GlobalVocab(EntityVocab.load()))
    fresh.load_compatible(old)
    assert torch.equal(fresh.type_emb.weight[: len(BASE_TOKEN_TYPES)], old["type_emb.weight"])


def test_enemy_power_bound_to_its_own_enemy(net):
    """Two same-type enemies; the power sits on enemy A in one batch and on enemy B in the other."""
    def make(owner_group: int) -> TokenBatch:
        t = lambda n: TOKEN_TYPES.index(n)  # noqa: E731
        types = [t("PLAYER"), t("ENEMY"), t("ENEMY"), t("ENEMY_POWER")] + [t(n) for n in PILE_TOKEN_TYPES]
        feats = torch.zeros(len(types), FEATURE_DIM)
        feats[3, 3] = 0.3
        groups = torch.tensor([0, 1, 2, owner_group, 0, 0, 0])
        return TokenBatch(
            type_ids=torch.tensor(types), global_ids=torch.tensor([0, 3, 3, 4, 0, 0, 0]), features=feats,
            group_ids=groups, player_pos=0, enemy_pos={0: 1, 1: 2},
            piles=_make_piles(CARDS, slots=[0, 1, 2, 3]),
        )
    on_a, on_b = net.encode(make(1)), net.encode(make(2))
    assert not torch.allclose(on_a[1], on_b[1], atol=1e-4)  # enemy A's state depends on who has it
