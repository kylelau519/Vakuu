"""Stage-1 macro policy over the host's typed screens: enough to carry a mainline through map, rewards, events, shops, rests
and card prompts so Claws' combats are played in a realistic deck. (Stage 2 puts the Golden Compass here.) Every function
returns a host action string from the decision's own legal list."""
from __future__ import annotations

import random
import sys

import tier_policy  # noqa: E402  (whispering_earring/, put on sys.path by the caller)


class _Card:
    def __init__(self, index: int, cid: str):
        self.index, self.id = index, cid


def choose(d, rng: random.Random, repeat: int = 0) -> str:
    """`d` is a host Decision on a non-combat surface; `repeat` counts how often this exact screen came back unchanged."""
    s, legal = d.surface, d.legal
    sc = d.obs["screen"]
    run = d.obs["run"]
    pick = lambda prefix: [a for a in legal if a.startswith(prefix)]  # noqa: E731

    if s == "map":
        return rng.choice(pick("map:") or legal)
    if s == "rewards":
        take = pick("reward:")
        return take[0] if take else ("rewards_done" if "rewards_done" in legal else legal[0])
    if s == "card_reward":
        cards = [_Card(i, c["id"]) for i, c in enumerate(sc["cards"])]
        idx = tier_policy.choose_card_reward(cards, "card_alt:Skip" in legal, len(run["deck"]))
        if idx is None:
            return "card_alt:Skip" if "card_alt:Skip" in legal else legal[0]
        return f"card:{idx}"
    if s == "rest":
        hp, mx = run["hp"], run["max_hp"]
        want = "rest:HEAL" if hp < 0.7 * mx else "rest:SMITH"
        if want in legal:
            return want
        return rng.choice(pick("rest:") or legal)
    if s == "shop":
        gold = run["gold"]
        afford = [(e["slot"], e) for e in sc["entries"] if e["kind"] == "card" and e["cost"] <= gold and f"buy:{e['slot']}" in legal]
        cards = [_Card(slot, e["card"]["id"]) for slot, e in afford]
        idx = tier_policy.choose_shop_card(cards, len(run["deck"])) if cards else None
        return f"buy:{idx}" if idx is not None else "leave"  # choose_shop_card returns the card's own .index (its shop slot)
    if s == "treasure":
        return "relic:0" if "relic:0" in legal else legal[0]
    if s == "event":
        if repeat >= 3 and "proceed" in legal:
            return "proceed"
        ev = pick("event:")
        if repeat >= 3 and not ev:
            return legal[0]
        return rng.choice(ev) if ev else ("proceed" if "proceed" in legal else legal[0])
    if s == "card_choice":
        ch = sc["choice"]
        n = len(ch["options"])
        hi = min(ch["max"], n)
        k = rng.randint(ch["min"], hi) if hi >= ch["min"] else 0
        return "pick:" + ",".join(str(i) for i in sorted(rng.sample(range(n), k)))
    return rng.choice(legal) if legal else "pick:0"
