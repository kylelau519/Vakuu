"""Round-trip check of encode.py over spire-zero's combat fixtures (sim parity --dump): encode every observation, read the
record back through buffer_layout's accessors, and compare each field with the observation. Prints the mismatch count per
field and every exception. Usage: verify_encode.py DUMP_DIR [max_files]"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "whispering_earring"))
import sts2_path  # noqa: F401,E402
import buffer_layout as bl  # noqa: E402
import buffer_env  # noqa: E402
from spire import tables as T, encode as E  # noqa: E402

T.install_decode_map()
dec = buffer_env.decode_token


def check(o: dict, legal: list[str], bad: collections.Counter) -> None:
    a = E.encode_combat(o, legal)
    p = o["player"]
    def eq(name, got, want):
        if got != want:
            bad[name] += 1
    eq("decision", bl.decision_name(a), "combat_play")
    eq("hp", int(a[bl.PLAYER_SUMMARY + 1]), p["hp"]); eq("max_hp", int(a[bl.PLAYER_SUMMARY + 2]), p["max_hp"])
    eq("gold", int(a[bl.PLAYER_SUMMARY + 4]), p.get("gold", 0))
    eq("character", dec(int(a[bl.PLAYER_SUMMARY])), p["character"])
    eq("round", bl.round_number(a), o["round"]); eq("energy", bl.energy(a), p["energy"]); eq("max_energy", bl.max_energy(a), p["max_energy"])
    hand = list(bl.hand_cards(a))
    eq("hand_len", len(hand), len(o["hand"][:10]))
    playable = {int(x.split(":")[1]) for x in legal if x.startswith("play:")}
    for (i, slot), c in zip(hand, o["hand"]):
        eq("hand_id", dec(bl.card_token(slot)), c["id"].rstrip("+"))
        eq("hand_cost", int(slot[bl.CARD_COST]), c.get("cost", 0))
        eq("hand_upgraded", int(slot[bl.CARD_UPGRADED]), 1 if c.get("up", 0) > 0 else 0)
        eq("hand_can_play", bl.hand_can_play(slot), o["hand"][i].get("uid", i) in playable)
        eq("hand_stars", bl.hand_star_cost(slot), c.get("stars", 0) or 0)
        eq("hand_enchant", dec(int(slot[bl.CARD_ENCHANT_TOKEN])) if slot[bl.CARD_ENCHANT_TOKEN] > 0 else None, (c.get("enchant") or {}).get("id"))
        eq("hand_affliction", dec(int(slot[bl.CARD_AFFLICT_TOKEN])) if slot[bl.CARD_AFFLICT_TOKEN] > 0 else None, (c.get("affliction") or {}).get("id"))
        if "damage" in {k.lower() for k in (c.get("vars") or {})}:
            eq("hand_damage", int(bl.hand_stat(slot, "damage")), int({k.lower(): v for k, v in c["vars"].items()}["damage"]))
        eq("hand_target_known", int(slot[bl._HAND_TARGET_TYPE]) >= 0, True)
    enemies = list(bl.enemies(a))
    alive = [e for e in o["enemies"] if e["hp"] > 0][:5]
    eq("enemy_len", len(enemies), len(alive))
    for (i, slot), e in zip(enemies, alive):
        eq("enemy_id", dec(bl.enemy_token(slot)), e["id"]); eq("enemy_hp", bl.enemy_hp(slot), e["hp"])
        eq("enemy_block", bl.enemy_block(slot), e.get("block", 0))
        eq("enemy_powers", [(dec(int(s[0])), int(s[1])) for _, s in bl.enemy_powers(slot)],
           [(x["id"], int(x.get("amount", 0))) for x in (e.get("powers") or [])[:8]])
        want_int = [(T.INTENT_INDEX[x["type"]], x["damage"] * x.get("hits", 1) if "damage" in x else -1) for x in (e.get("intents") or [])[:2]]
        eq("enemy_intents", [bl.enemy_intent(slot, j) for j in range(len(want_int))], want_int)
    eq("player_powers", [(dec(int(s[0])), int(s[1])) for _, s in bl.player_powers(a)],
       [(x["id"], int(x.get("amount", 0))) for x in (p.get("powers") or [])[:16]])
    for name, got, want in (("draw", bl.draw_pile_cards(a), o["draw"]), ("discard", bl.discard_pile_cards(a), o["discard"]),
                            ("exhaust", bl.exhaust_pile_cards(a), o["exhaust"])):
        eq(name + "_ids", [dec(bl.card_token(s)) for _, s in got], [c["id"].rstrip("+") for c in want[:50]])
    eq("relics", [dec(int(s[0])) for _, s in bl.player_relics(a)], [r["id"] for r in o["relics"][:40]])
    eq("potions", [dec(int(s[0])) for _, s in bl.player_potions(a)], [x for x in o["potions"] if x][:10])
    eq("orbs", [dec(int(s[0])) for _, s in bl.orbs(a)], [x["id"] for x in (p.get("orbs") or [])[:10]])
    if "prompt" in o:
        b = E.encode_card_select(o)
        eq("prompt_decision", bl.decision_name(b), "card_select")
        eq("prompt_options", [dec(bl.card_token(s)) for _, s in bl.select_cards(b)], [c["id"].rstrip("+") for c in (o["prompt"].get("options") or [])[:50]])


def main() -> None:
    root, limit = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    files = glob.glob(os.path.join(root, "*", "*.observation.json"))
    random.seed(7); files = random.sample(files, min(limit, len(files)))
    bad, errors, n = collections.Counter(), collections.Counter(), 0
    legal_cache = {}
    for f in files:
        d = os.path.dirname(f); k = int(os.path.basename(f).split(".")[0])
        if d not in legal_cache:
            legal_cache[d] = json.load(open(os.path.join(d, "combat.json")))["decisions"]
        dl = legal_cache[d]
        legal = dl[k]["legal"] if k < len(dl) else []
        try:
            check(json.load(open(f)), legal, bad); n += 1
        except Exception as ex:  # noqa: BLE001
            errors[f"{type(ex).__name__}: {str(ex)[:90]}"] += 1
    print(f"{n} observations encoded and read back, {sum(errors.values())} exceptions")
    for k, v in errors.most_common(8):
        print("  EXC", v, k)
    print("mismatches by field:", dict(bad) if bad else "none")


main()
