"""Replay in eval: how much HP does the greedy policy's choice cost against the other plays it had? (a policy-improvement check, not training)

At the first decision of a combat turn the live combat is exported to the engine and re-imported `hidden` times, each time with the hidden part drawn again (resample_export): each import is an
independent draw of what the player cannot see (the draw pile's order, the random streams), consistent with what it can. For every first play the
policy could make (potions left out: their cost is not HP), each hidden sample is cloned, the play is applied, and Claws finishes the combat greedily.
g = HP lost from that decision to the combat's end. Averaged over the hidden samples this is the play's expected cost under greedy continuation, so no
information the player lacks is used by the policy; the hidden samples are the same for every play (paired), which keeps the comparison tight.

The regret reported is cross-fitted: the best play is picked on half of the hidden samples and its cost measured on the other half. Picking the minimum
of noisy estimates and reading the same estimates back would always show a gain; this one is zero in expectation when the greedy play is already the best.
"""
from __future__ import annotations

import json
import random
import zlib

import numpy as np


def resample_export(export: bytes, seed: int) -> bytes:
    """The combat's export with what the player cannot see drawn again: the draw pile's order (its contents are known to the player) and every
    random generator's state (later shuffles, random targets and moves). What the player can see -- hand, piles' contents, HP, intents already
    chosen -- is untouched. World.from_export(..., sample=True) does not do this: the flag only changes what `hidden_key` means."""
    e = json.loads(export)
    a = e["arenas"]
    r = random.Random(seed)
    pcs = a["player_combat_state"][0]["PlayerCombatState"]
    draw = a["card_pile"][pcs["draw_pile"][1]]["CardPile"]["_cards"]
    r.shuffle(draw)
    for m in a.get("mega_random", []):
        body = m["MegaRandom"]
        for k in ("_s0", "_s1", "_s2", "_s3"):
            body[k] = r.getrandbits(64) | 1
    return json.dumps(e, separators=(",", ":")).encode()


def _kind(card: dict | None) -> str:
    if card is None:
        return "end"
    v = card.get("vars") or {}
    att, blk = "Damage" in v or bool(card.get("dmg")), "Block" in v
    return "attack+block" if att and blk else "attack" if att else "block" if blk else "other"


def _incoming(obs: dict) -> int:
    return int(sum(i.get("damage", 0) * i.get("hits", 1) for e in obs["enemies"] if e["hp"] > 0 for i in e.get("intents") or [] if i.get("type") == "Attack"))


def replay_decision(J, export: bytes, obs: dict, legal: list[str], ctx: dict, options: list, greedy_idx: int, hidden: int, tag: str) -> dict | None:
    """One decision's replay (see the module docstring). `J` is vakuu.claws.job (loaded). None when there is nothing to compare."""
    from vakuu.engine.actions import to_action
    from vakuu.engine.engine import World

    hp = obs["player"]["hp"]
    legal_set = set(legal)
    cand = []  # (option index, host action)
    for i, o in enumerate(options):
        if o[0] == "use_potion":
            continue
        a = to_action(*o, obs, legal)
        if a in legal_set:
            cand.append((i, a))
    ids = [c[0] for c in cand]
    if greedy_idx not in ids or len(cand) < 2:
        return None
    base, clones, slots = [], [], []
    try:
        for _ in range(hidden):
            base.append(World.from_export(resample_export(export, zlib.crc32(f"{tag}_{_}".encode()))))
        for k, w in enumerate(base):
            for j, (_, act) in enumerate(cand):
                c = w.clone()
                try:
                    c.apply(act)
                except Exception:  # noqa: BLE001
                    c.close()
                    continue
                clones.append(c)
                slots.append((j, k))
        results = J.play_engines(clones, ctx, hp, [f"{tag}_{j}_{k}" for j, k in slots])
    finally:
        for w in base + clones:
            w.close()
    g = np.full((len(cand), hidden), np.nan)
    for (j, k), r in zip(slots, results):
        if r["outcome"] in ("won", "lost"):
            g[j, k] = hp - r["final_hp"]
    gj = ids.index(greedy_idx)
    a, b = np.arange(hidden) % 2 == 0, np.arange(hidden) % 2 == 1
    with np.errstate(all="ignore"):
        mean_a, mean_b, mean_all = np.nanmean(g[:, a], 1), np.nanmean(g[:, b], 1), np.nanmean(g, 1)
    if np.isnan(mean_all).any():
        return None
    star = int(np.argmin(mean_a))
    hand = obs["hand"]
    plays = []
    for i, _ in cand:
        o = options[i]
        card = hand[o[1]["card_index"]] if o[0] == "play_card" else None
        plays.append({"name": o[0] if card is None else card["id"], "kind": _kind(card), "cost": None if card is None else card.get("cost")})
    return {
        "enc": obs.get("encounter"), "round": obs.get("round"), "hp": hp, "block": obs["player"].get("block", 0), "energy": obs["player"].get("energy"),
        "incoming": _incoming(obs), "plays": plays, "greedy": gj, "g": np.round(g, 1).tolist(),
        "regret_cf": float(mean_b[gj] - mean_b[star]), "regret_raw": float(mean_all[gj] - mean_all.min()),
        "best": int(np.argmin(mean_all)), "best_cf": star,
    }


def summarize_replay(recs: list[dict]) -> dict:
    """Aggregates over replayed decisions."""
    if not recs:
        return {}
    rc = np.array([r["regret_cf"] for r in recs])
    out = {"n": len(recs), "regret_cf": round(float(rc.mean()), 2), "regret_cf_se": round(float(rc.std() / max(1, len(rc)) ** 0.5), 2),
           "regret_raw": round(float(np.mean([r["regret_raw"] for r in recs])), 2), "share_regret_gt2": round(float((rc > 2).mean()), 3)}
    flips: dict[str, int] = {}
    for r in recs:
        gk, bk = r["plays"][r["greedy"]]["kind"], r["plays"][r["best_cf"]]["kind"]
        if r["best_cf"] != r["greedy"]:
            flips[f"{gk}->{bk}"] = flips.get(f"{gk}->{bk}", 0) + 1
    out["switches"] = dict(sorted(flips.items(), key=lambda kv: -kv[1])[:6])
    by = {}
    for lo, hi, name in ((-1, 5, "inc<=5"), (5, 15, "inc6-15"), (15, 25, "inc16-25"), (25, 999, "inc>25")):
        s = [r["regret_cf"] for r in recs if lo < r["incoming"] <= hi]
        if len(s) >= 5:
            by[name] = [len(s), round(float(np.mean(s)), 2)]
    out["by_incoming"] = by
    return out
