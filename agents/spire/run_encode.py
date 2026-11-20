"""The host's typed run-level screens (map, card reward, rest, shop, event, card prompt) -> the buffer record the Golden Compass and
the rest of agents/ read (decision.arr), plus the table that turns the agents' answers back into host action strings.

Same layout as sts2-cli's RunSimulator.States.Buffer.cs writer (buffer_layout.py). A SpireDecision is a BufferDecision with `.meta`
(the host's own indexes: which screen entry each buffer slot stands for)."""
from __future__ import annotations

import numpy as np

import buffer_layout as bl
from buffer_env import BufferDecision
from . import tables as T
from .encode import _EMPTY_CARD, _TEMPLATE, _card

_POINT_TYPE = {v: k for k, v in bl.MAP_POINT_TYPES.items() if v}
_DT = {name: num for num, name in bl.DECISION_NAMES.items()}
_CARD = bl.CARD_SIZE


class SpireDecision(BufferDecision):
    """A BufferDecision that remembers where its slots came from on the host's screen."""

    def __init__(self, arr: np.ndarray, meta: dict):
        super().__init__(arr)
        self.meta = meta


def _template() -> np.ndarray:
    a = _TEMPLATE.copy()
    # sections combat_play leaves at zero but the other writers pad with -1 (an empty card, an absent option)
    for base, n in ((bl.CARD_SELECT + 2, bl.MAX_DECK_CARDS), (bl.CARD_REWARD + 2, 6), (bl.EVENT_CARD_REWARD, 6), (bl.BUNDLE_SELECT, 32)):
        for i in range(n):
            a[base + i * _CARD: base + (i + 1) * _CARD] = _EMPTY_CARD
    for i in range(15):
        o = bl.SHOP + i * (_CARD + 3)
        a[o: o + _CARD] = _EMPTY_CARD
        a[o + _CARD: o + _CARD + 3] = -1
    for i in range(11):
        o = bl.SHOP + 15 * (_CARD + 3) + i * 3
        a[o: o + 3] = -1
    a[bl.SHOP + 15 * (_CARD + 3) + 11 * 3] = -1  # removal cost
    a[bl.REST_SITE: bl.REST_SITE + 12] = -1
    a[bl.EVENT_CHOICE: bl.EVENT_CHOICE + 6] = -1
    return a


_TEMPLATE_RUN = _template()


def act_floor(run: dict) -> int:
    """The buffer's ActFloor: the row of the room the player is in within the act (0 at the act's Ancient)."""
    return max(0, len(run["map"]["visited"]) - 1)


def _player_and_context(a: np.ndarray, run: dict) -> None:
    s = bl.PLAYER_SUMMARY
    a[s] = T.token("characters", run["character"])
    a[s + 1], a[s + 2], a[s + 3], a[s + 4] = run["hp"], run["max_hp"], 0, run["gold"]
    potions = run.get("potions") or []
    a[s + 5] = len(potions)
    deck = run["deck"]
    a[s + 6] = len(deck)
    for i, r in enumerate(run["relics"][:bl.MAX_RELICS]):
        o = s + bl._PLAYER_RELICS_OFFSET + i * 2
        a[o], a[o + 1] = T.token("relics", r["id"]), r.get("counter", -1)
    for i, p in enumerate(potions[:bl.MAX_POTIONS]):
        if p:
            o = s + bl._PLAYER_POTIONS_OFFSET + i * 2
            a[o], a[o + 1] = T.token("potions", p), -1
    for i, c in enumerate(deck[:bl.MAX_DECK_CARDS]):
        _card(a, s + bl._PLAYER_DECK_OFFSET + i * _CARD, c)
    rc = bl.RUN_CONTEXT
    a[rc] = run["act"] + 1
    a[rc + 1] = T.token("acts", run.get("act_id"))
    a[rc + 2] = act_floor(run)
    a[rc + 3] = T.token("encounters", run["map"].get("boss"))
    a[rc + 4] = 0


def _map(a: np.ndarray, run: dict) -> None:
    m = run["map"]
    visited = {(v["col"], v["row"]) for v in m["visited"]}
    current = (m["visited"][-1]["col"], m["visited"][-1]["row"]) if m["visited"] else None

    def write(o: int, p: dict, is_boss: bool = False) -> None:
        a[o] = _POINT_TYPE.get(p["type"], 0)
        a[o + 1] = 1 if (p["col"], p["row"]) in visited else 0
        a[o + 2] = 1 if (p["col"], p["row"]) == current else 0
        kids = sorted(p.get("next") or [], key=lambda k: k["col"])[:bl.MAX_MAP_CHILDREN]
        for j in range(bl.MAX_MAP_CHILDREN):
            a[o + 3 + 2 * j] = kids[j]["col"] if j < len(kids) else -1
            a[o + 4 + 2 * j] = kids[j]["row"] if j < len(kids) else -1

    base = bl.MAP
    for i in range(bl.MAX_MAP_ROWS * bl.MAX_MAP_COLS):  # an empty grid cell
        o = base + i * bl.MAP_POINT_SIZE
        a[o], a[o + 3: o + 9] = -1, -1
    boss_o = base + bl.MAX_MAP_ROWS * bl.MAX_MAP_COLS * bl.MAP_POINT_SIZE
    a[boss_o], a[boss_o + 3: boss_o + 9] = -1, -1
    for p in m["points"]:
        if p["type"] == "Boss":
            write(boss_o, p, True)
        elif p["row"] < bl.MAX_MAP_ROWS and p["col"] < bl.MAX_MAP_COLS:
            write(base + (p["row"] * bl.MAX_MAP_COLS + p["col"]) * bl.MAP_POINT_SIZE, p)


def encode_run(d, character: str | None = None) -> SpireDecision | None:
    """A host Decision on a run-level surface -> SpireDecision, or None where the agents never decide this screen
    (rewards, treasure, plain proceed): the caller answers those itself."""
    s = d.surface
    run = d.obs["run"]
    sc = d.obs.get("screen") or {}
    a = _TEMPLATE_RUN.copy()
    meta: dict = {"surface": s}
    if s == "map":
        a[bl.DECISION_TYPE] = _DT["map_select"]
        _map(a, run)
        meta["legal"] = {(t["col"], t["row"]): f"map:{t['col']},{t['row']}" for t in sc["travelable"]}
    elif s == "card_reward":
        a[bl.DECISION_TYPE] = _DT["card_reward"]
        a[bl.CARD_REWARD] = 1 if "card_alt:Skip" in d.legal else 0
        a[bl.CARD_REWARD + 1] = 0
        for i, c in enumerate(sc["cards"][:6]):
            _card(a, bl.CARD_REWARD + 2 + i * _CARD, c)
        meta["cards"] = [f"card:{i}" for i in range(len(sc["cards"][:6]))]
        meta["skip"] = "card_alt:Skip" if "card_alt:Skip" in d.legal else None
        meta["alts"] = [x for x in d.legal if x.startswith("card_alt:")]
    elif s == "event":
        opts = sc.get("options") or []
        if not opts:
            return None
        a[bl.DECISION_TYPE] = _DT["event_choice"]
        a[bl.EVENT_CHOICE] = -1
        for i in range(5):
            a[bl.EVENT_CHOICE + 1 + i] = (0 if f"event:{opts[i]['index']}" in d.legal else 1) if i < len(opts) else -1
        meta["options"] = [f"event:{o['index']}" for o in opts[:5]]
    elif s == "rest":
        a[bl.DECISION_TYPE] = _DT["rest_site"]
        opts = sc["options"][:6]
        for i, o in enumerate(opts):
            a[bl.REST_SITE + 2 * i] = 1
            a[bl.REST_SITE + 2 * i + 1] = 1 if o.get("enabled", True) else 0
        meta["options"] = [f"rest:{o['id']}" for o in opts]
    elif s == "shop":
        a[bl.DECISION_TYPE] = _DT["shop"]
        cards = [e for e in sc["entries"] if e["kind"] == "card"][:15]
        relics = [e for e in sc["entries"] if e["kind"] == "relic"][:6]
        potions = [e for e in sc["entries"] if e["kind"] == "potion"][:5]
        for i, e in enumerate(cards):
            o = bl.SHOP + i * (_CARD + 3)
            _card(a, o, e["card"])
            a[o + _CARD], a[o + _CARD + 1], a[o + _CARD + 2] = e["cost"], 1, 1 if e.get("on_sale") else 0
        for i, e in enumerate(relics):
            o = bl.SHOP + 15 * (_CARD + 3) + i * 3
            a[o], a[o + 1], a[o + 2] = T.token("relics", e["id"]), e["cost"], 1
        for i, e in enumerate(potions):
            o = bl.SHOP + 15 * (_CARD + 3) + 6 * 3 + i * 3
            a[o], a[o + 1], a[o + 2] = T.token("potions", e["id"]), e["cost"], 1
        a[bl.SHOP + 15 * (_CARD + 3) + 11 * 3] = sc.get("removal_cost", -1) if "remove" in d.legal else -1
        meta["cards"] = [f"buy:{e['slot']}" for e in cards]
        meta["relics"] = [f"buy:{e['slot']}" for e in relics]
        meta["potions"] = [f"buy:{e['slot']}" for e in potions]
    elif s == "card_choice" and "combat" not in sc:
        ch = sc["choice"]
        a[bl.DECISION_TYPE] = _DT["card_select"]
        a[bl.CARD_SELECT], a[bl.CARD_SELECT + 1] = ch["min"], ch["max"]
        for i, c in enumerate(ch["options"][:bl.MAX_DECK_CARDS]):
            _card(a, bl.CARD_SELECT + 2 + i * _CARD, c)
        meta["min"], meta["max"], meta["n"] = ch["min"], ch["max"], len(ch["options"])
    else:
        return None
    _player_and_context(a, run)
    if s != "map":
        _map(a, run)  # the map section is part of every record (the compass's afterstates read it on map screens only)
    return SpireDecision(a, meta)


def to_host_action(name: str, kwargs: dict, dec: SpireDecision, d) -> str:
    """The agents' answer (select_map_node, select_card_reward, choose_option, buy_card, ...) as a host action string."""
    m, s = dec.meta, dec.meta["surface"]
    if name == "select_map_node":
        return m["legal"][(kwargs["col"], kwargs["row"])]
    if name == "select_card_reward":
        return m["cards"][kwargs["card_index"]]
    if name == "skip_card_reward":
        return m["skip"] or (m["alts"][0] if m["alts"] else d.legal[0])
    if name == "choose_option":
        return (m["options"] if s in ("event", "rest") else [])[kwargs["option_index"]]
    if name == "leave_room":
        for x in ("leave", "proceed", "rewards_done"):
            if x in d.legal:
                return x
        return d.legal[0]
    if name == "buy_card":
        return m["cards"][kwargs["card_index"]]
    if name == "buy_relic":
        return m["relics"][kwargs["relic_index"]]
    if name == "buy_potion":
        return m["potions"][kwargs["potion_index"]]
    if name == "remove_card":
        return "remove"
    if name == "select_cards":
        return "pick:" + ",".join(str(i) for i in sorted(kwargs["indices"]))
    if name == "skip_select":
        return "pick:"
    raise ValueError(f"no host action for {name} {kwargs} on {s}")
