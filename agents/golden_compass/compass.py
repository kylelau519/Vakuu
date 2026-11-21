"""Golden Compass at decision time: score every option's afterstate with the hazard net and sample.

    map_select      one option per reachable child; afterstate = that node entered (run_state.map_afterstates)
    card_reward     one option per offered card (+ skip); afterstate = deck with / without the card

Option score = expected further floors survived (hazard_net.expected_floors). Sampling is a softmax
over scores at temperature COMPASS_TAU, mixed with COMPASS_EPS of uniform -- the exploration that
keeps the next round's training data from only ever containing the states the current net likes.

    rest_site       HEAL vs SMITH; shop: a beam search over buys / removal; card_select (after a smith
                    or removal): one option per distinct card  -- see room_search.py

No checkpoint (round 0, or GOLDEN_COMPASS_CHECKPOINT unset/missing): choose_map returns a uniform
random move and choose_card_reward returns None so the caller can fall back to the tier list.

Every choose_* returns (action name, kwargs, policy dict) where policy is the per-option scores and
probabilities, ready for the recorder.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import room_search  # noqa: E402
import run_state as rs  # noqa: E402

_CHECKPOINT = os.environ.get("GOLDEN_COMPASS_CHECKPOINT")
_EPS = float(os.environ.get("COMPASS_EPS", "0.03"))  # share of uniform-random picks (was 0.1: too many random routes for the labels to describe the deployed compass)
_TAU = float(os.environ.get("COMPASS_TAU", "0.35"))  # softmax temperature, in expected-floors units
# Test switch: COMPASS_DISABLE=map,card,rest,shop,select turns those decisions back to the fallback (random map,
# tier-list cards, ...). Unset = the compass decides everything, as always.
# Card rewards only: how many floors one full max-HP share of predicted combat cost is worth (the aux head is what the
# branch data trains). 0 = the hazard score alone, as before.
_CARD_AUX = float(os.environ.get("COMPASS_CARD_AUX", "0"))
_DISABLED = {k.strip() for k in os.environ.get("COMPASS_DISABLE", "").split(",") if k.strip()}

_net = None
_encoder = None


def get_net():
    """Lazy per-process load. None when no checkpoint exists yet (see module docstring)."""
    global _net, _encoder
    if _net is not None:
        return _net
    if not _CHECKPOINT or not os.path.exists(_CHECKPOINT):
        return None
    import hazard_net as hn
    _encoder = hn.Encoder()
    _net = hn.HazardNet.from_checkpoint(_CHECKPOINT)
    return _net


def score(states: list[rs.RunState], aux_weight: float = 0.0) -> list[float]:
    """Expected further floors survived, per state, minus aux_weight x the aux head's predicted HP cost (the share of
    max HP the next AUX combats cost; 0 = hazard only, the original score). Needs a loaded net."""
    import hazard_net as hn
    net = get_net()
    assert net is not None, "score() needs a hazard net checkpoint"
    batch = _encoder.collate(states)
    with torch.no_grad():
        logits, aux = net(batch)
        # act 1 states: the floors after the boss are asked of the state healed to full HP at act 2's Ancient
        healed = _encoder.collate([rs.at_ancient(st) if st.act <= 1 else st for st in states])
        value = hn.expected_floors(logits, batch["start"], net(healed)[0])
        if aux_weight:
            value = value - aux_weight * aux.sum(1)
        return value.tolist()


def _sample(scores: list[float]) -> tuple[int, list[float]]:
    top = max(scores)
    weights = [pow(2.718281828, (s - top) / max(_TAU, 1e-6)) for s in scores]
    total = sum(weights)
    probs = [(1 - _EPS) * w / total + _EPS / len(scores) for w in weights]
    return random.choices(range(len(scores)), weights=probs)[0], probs


def choose_map(decision, character: str | None, ascension: int):
    """-> (name, kwargs, policy|None) for a map_select decision."""
    arr = decision.arr
    net = get_net()
    if net is None or "map" in _DISABLED:
        col, row = random.choice(rs.map_options(arr))
        return "select_map_node", {"col": col, "row": row}, None

    base = rs.encode_from_buffer(arr, character, ascension, getattr(decision, "remaining", None))
    options = rs.map_afterstates(arr, base)
    scores = score([state for _key, state in options])
    idx, probs = _sample(scores)
    (col, row), _ = options[idx]
    policy = {
        "kind": "map",
        "options": [{"col": c, "row": r, "score": s, "prob": p} for ((c, r), _), s, p in zip(options, scores, probs)],
        "chosen": idx,
    }
    return "select_map_node", {"col": col, "row": row}, policy


def choose_card_reward(decision, cards: list[tuple[int, str | None, int]], can_skip: bool,
                       character: str | None, ascension: int):
    """cards: (index, card_id, upgraded) for each offered card. -> (name, kwargs, policy), or None
    when there is no net (the caller falls back to the tier list)."""
    net = get_net()
    if net is None or "card" in _DISABLED:
        return None
    base = rs.encode_from_buffer(decision.arr, character, ascension, getattr(decision, "remaining", None))
    states = [rs.with_card(base, cid, up) for _i, cid, up in cards]
    labels = [{"card_index": i, "card": cid} for i, cid, _up in cards]
    if can_skip or not cards:
        states.append(base)
        labels.append({"skip": True})
    scores = score(states, _CARD_AUX)
    idx, probs = _sample(scores)
    for label, s, p in zip(labels, scores, probs):
        label["score"], label["prob"] = s, p
    policy = {"kind": "card_reward", "options": labels, "chosen": idx}
    if "skip" in labels[idx]:
        return "skip_card_reward", {}, policy
    return "select_card_reward", {"card_index": labels[idx]["card_index"]}, policy


def _pick(options: list, kind: str):
    """Samples one of room_search.Options; -> (option, policy dict)."""
    idx, probs = _sample([o.value for o in options])
    policy = {
        "kind": kind,
        "options": [{**o.label, "score": o.value, "prob": p} for o, p in zip(options, probs)],
        "chosen": idx,
    }
    return options[idx], policy


def choose_rest(decision, character: str | None, ascension: int):
    """-> (name, kwargs, next_purpose, policy), or None (no net / nothing scoreable: caller falls back)."""
    if get_net() is None or "rest" in _DISABLED:
        return None
    base = rs.encode_from_buffer(decision.arr, character, ascension, getattr(decision, "remaining", None))
    options = room_search.rest_options(decision.arr, base, score)
    if not options:
        return None
    chosen, policy = _pick(options, "rest_site")
    return chosen.name, chosen.kwargs, chosen.purpose, policy


def choose_shop(decision, character: str | None, ascension: int):
    """-> (name, kwargs, next_purpose, policy), or None without a net."""
    if get_net() is None or "shop" in _DISABLED:
        return None
    base = rs.encode_from_buffer(decision.arr, character, ascension, getattr(decision, "remaining", None))
    options = room_search.shop_options(decision.arr, base, score)
    chosen, policy = _pick(options, "shop")
    return chosen.name, chosen.kwargs, chosen.purpose, policy


def choose_card_select(decision, purpose: str | None, character: str | None, ascension: int):
    """The follow-up pick after a Smith ("upgrade") or a shop removal ("remove"). -> (name, kwargs,
    policy), or None without a net or for any other kind of selection (caller picks at random)."""
    if get_net() is None or "select" in _DISABLED or purpose not in ("upgrade", "remove"):
        return None
    base = rs.encode_from_buffer(decision.arr, character, ascension, getattr(decision, "remaining", None))
    options = room_search.select_options(decision.arr, base, purpose, score)
    if not options:
        return None
    chosen, policy = _pick(options, f"card_select_{purpose}")
    return chosen.name, chosen.kwargs, policy
