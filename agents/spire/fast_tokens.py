"""Faster, exactly equivalent replacements for the slow parts of Claws' per-decision work.

pile tensors: claws/model/model.py's _pile_tensors reads three 50-slot piles card by card through buffer_layout's accessors (about
70% of tokenize_from_buffer). fast_pile_tensors does the same with a few NumPy slices and a token -> global-id lookup table.
`install()` swaps it in (model._pile_tensors is looked up by name at call time); tests/verify_fast.py checks the two agree exactly.

batched forward: score_batch runs ClawsNet once over many decisions (the forks of one turn advance in lockstep) the way train.py's
_batch_step does for training, and returns each decision's option logits and V."""
from __future__ import annotations

import numpy as np
import torch

import buffer_layout as bl
import buffer_env

_tables: dict[int, np.ndarray] = {}


def _table(gvocab) -> np.ndarray:
    """token -> the global card id gvocab gives the card that token decodes to (UNK for anything else)."""
    key = id(gvocab)
    t = _tables.get(key)
    if t is None:
        from spire import tables as T
        n = max(T.TOKEN_MAP) + 1
        t = np.zeros(n, np.int64)
        for tok in range(n):
            t[tok] = gvocab.entity("card", buffer_env.decode_token(tok))
        _tables[key] = t
    return t


_PILES = (bl._CP_DRAW_PILE_OFFSET, bl._CP_DISCARD_PILE_OFFSET, bl._CP_EXHAUST_PILE_OFFSET)
_C = bl.CARD_SIZE
_N = bl.MAX_DECK_CARDS


def fast_pile_tensors(arr: np.ndarray, gvocab) -> dict[str, torch.Tensor]:
    table = _table(gvocab)
    ids = np.zeros((3, _N), np.int64)
    cost = np.zeros((3, _N), np.float32)
    upg = np.zeros((3, _N), np.float32)
    ctype = np.zeros((3, _N), np.int64)
    mask = np.zeros((3, _N), np.bool_)
    counts = np.zeros(3, np.float32)
    cp = bl.COMBAT_PLAY
    for p, off in enumerate(_PILES):
        blk = arr[cp + off: cp + off + _N * _C].reshape(_N, _C)
        tok = blk[:, bl.CARD_TOKEN].astype(np.int64)
        m = tok != 0
        mask[p] = m
        ids[p] = np.where(m, table[np.where(m, tok, 0)], 0)
        cost[p] = np.where(m, blk[:, bl.CARD_COST], 0)
        upg[p] = np.where(m, blk[:, bl.CARD_UPGRADED] != 0, 0)
        ctype[p] = np.where(m, np.clip(blk[:, bl.CARD_TYPE].astype(np.int64), 0, 6), 0)
        counts[p] = int(arr[cp + bl._CP_PILE_COUNTS_OFFSET + p])
    return {"pile_ids": torch.from_numpy(ids), "pile_cost": torch.from_numpy(cost), "pile_upg": torch.from_numpy(upg),
            "pile_type": torch.from_numpy(ctype), "pile_mask": torch.from_numpy(mask), "pile_counts": torch.from_numpy(counts)}


_installed = False


def install() -> None:
    """model._pile_tensors := fast_pile_tensors (idempotent)."""
    global _installed
    if _installed:
        return
    import model as claws_model
    claws_model._orig_pile_tensors = claws_model._pile_tensors
    claws_model._pile_tensors = fast_pile_tensors
    _installed = True


def _collate(batches: list):
    """TokenBatch list -> padded tensors (the same padding train.py's _collate does for examples)."""
    from model import FEATURE_DIM, PILE_KEYS
    lens = [b.type_ids.shape[0] for b in batches]
    m, n = max(lens), len(batches)
    type_ids = torch.zeros(n, m, dtype=torch.long)
    global_ids = torch.zeros(n, m, dtype=torch.long)
    group_ids = torch.zeros(n, m, dtype=torch.long)
    features = torch.zeros(n, m, FEATURE_DIM)
    pad_mask = torch.ones(n, m, dtype=torch.bool)
    for i, (b, k) in enumerate(zip(batches, lens)):
        type_ids[i, :k], global_ids[i, :k], group_ids[i, :k], features[i, :k] = b.type_ids, b.global_ids, b.group_ids, b.features
        pad_mask[i, :k] = False
    piles = {key: torch.stack([b.piles[key] for b in batches]) for key in PILE_KEYS}
    return type_ids, global_ids, features, pad_mask, piles, group_ids


@torch.no_grad()
def score_batch(net, batches: list, options: list[list]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """(per-decision option logits, V in HP [n]) for n tokenized decisions in one pass."""
    type_ids, global_ids, features, pad_mask, piles, group_ids = _collate(batches)
    h = net.encode_batched(type_ids, global_ids, features, pad_mask, piles, group_ids)
    queries, spans = [], []
    for i, (b, opts) in enumerate(zip(batches, options)):
        hb, start = h[i], len(queries)
        for action, args in opts:
            if action == "end_turn":
                q = net.end_turn_query + hb[b.player_pos]
            elif action == "play_card":
                q = hb[b.hand_pos[args["card_index"]]]
                if "target_index" in args:
                    q = q + hb[b.enemy_pos[args["target_index"]]]
            elif action == "use_potion":
                q = hb[b.potion_pos[args["potion_index"]]]
                if "target_index" in args:
                    q = q + hb[b.enemy_pos[args["target_index"]]]
            else:
                raise ValueError(f"unhandled combat action {action!r}")
            queries.append(q)
        spans.append((start, len(opts)))
    logits = net.policy_mlp(net.with_action_features(torch.stack(queries), [a for opts in options for _, a in opts])).squeeze(-1)
    player_h = torch.stack([h[i, b.player_pos] for i, b in enumerate(batches)])
    values = net.value_in_hp(net.value_mlp(player_h).squeeze(-1))
    return [logits[s:s + c] for s, c in spans], values
