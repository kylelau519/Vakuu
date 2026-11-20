"""Typed observation (spire-zero's engine / host JSON, plain dicts) -> the buffer record (decision.arr) that sts2-cli's
C# writer made, at the offsets of buffer_layout.py. No typed objects: dicts in, one preallocated float32 array out.

combat_play is complete except what the engine's observation does not carry (zero-filled, and so are sts2-cli's for an
empty slot): the per-turn history counters, a card's enchantment/affliction, star costs. Everything else follows
RunSimulator.States.Buffer.cs's BufferStateWriter.WriteCombatPlay."""
from __future__ import annotations

import numpy as np

import buffer_layout as bl
from . import tables as T

T.install_decode_map()  # buffer_env.decode_token must read OUR token numbers, not sts2-cli's (old game) CLR table

_DT_COMBAT_PLAY = 3
_STAT_KEYS = bl.HAND_STAT_KEYS  # damage, block, magicnumber, ...
_CARD = bl.CARD_SIZE
_HAND = bl.HAND_SLOT_SIZE
_ENEMY = bl.ENEMY_SLOT_SIZE
_EMPTY_CARD = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0], np.float32)


def _build_template() -> np.ndarray:
    """What a fresh record holds where nothing is written: sts2-cli's empty-slot conventions (an empty card is token 0 with
    -1 for enchantment and affliction, an empty enemy has token -1 and (-1, 0) powers, ...)."""
    a = np.zeros(bl.TOTAL_SIZE, np.float32)
    def cards(base: int, n: int) -> None:
        for i in range(n):
            a[base + i * _CARD: base + (i + 1) * _CARD] = _EMPTY_CARD
    cards(bl.PLAYER_SUMMARY + bl._PLAYER_DECK_OFFSET, bl.MAX_DECK_CARDS)
    for i in range(bl.MAX_RELICS):
        a[bl.PLAYER_SUMMARY + bl._PLAYER_RELICS_OFFSET + i * 2 + 1] = -1
    for i in range(bl.MAX_POTIONS):
        a[bl.PLAYER_SUMMARY + bl._PLAYER_POTIONS_OFFSET + i * 2 + 1] = -1
    cp = bl.COMBAT_PLAY
    for i in range(bl.MAX_HAND_CARDS):
        s = cp + bl._CP_HAND_OFFSET + i * _HAND
        a[s: s + _CARD] = _EMPTY_CARD
        a[s + bl._HAND_TARGET_TYPE] = -1
        a[s + bl._HAND_STATS_OFFSET: s + _HAND] = -1
    for i in range(bl.MAX_ENEMIES):
        s = cp + bl._CP_ENEMIES_OFFSET + i * _ENEMY
        a[s] = -1
        for j in range(bl.MAX_INTENTS_PER_ENEMY):
            a[s + bl._ENEMY_INTENTS_OFFSET + 2 * j] = -1
            a[s + bl._ENEMY_INTENTS_OFFSET + 2 * j + 1] = -1
        for j in range(bl.MAX_ENEMY_POWERS):
            a[s + bl._ENEMY_POWERS_OFFSET + 2 * j] = -1
    for j in range(bl.MAX_PLAYER_POWERS):
        a[cp + bl._CP_PLAYER_POWERS_OFFSET + 2 * j] = -1
    for off in (bl._CP_DRAW_PILE_OFFSET, bl._CP_DISCARD_PILE_OFFSET, bl._CP_EXHAUST_PILE_OFFSET):
        cards(cp + off, bl.MAX_DECK_CARDS)
    for i in range(bl.MAX_ORBS):
        s = cp + bl._CP_ORBS_OFFSET + i * 3
        a[s: s + 3] = -1
    a[cp + bl._CP_STARS_OFFSET] = -1
    a[cp + bl._CP_OSTY_OFFSET: cp + bl._CP_OSTY_OFFSET + 3] = -1
    a[cp + bl._CP_OSTY_OFFSET + 3] = -1
    return a


_TEMPLATE = _build_template()


_KW_BITS = {"Exhaust": 0, "Ethereal": 1, "Innate": 2, "Unplayable": 3, "Retain": 4, "Sly": 5, "Eternal": 6}


def _keyword_mask(c: dict, meta: dict) -> int:
    kws = c.get("keywords")
    if kws is None:
        return meta.get("keywords", 0)
    m = 0
    for k in kws:
        b = _KW_BITS.get(k)
        if b is not None:
            m |= 1 << b
    return m


_ROWS: dict = {}


def _card_row(c: dict) -> np.ndarray:
    """The Card(11) block of one card, cached: a deck repeats the same few hundred card states thousands of times."""
    e, f, kw = c.get("enchant"), c.get("affliction"), c.get("keywords")
    key = (c["id"], c.get("cost", 0), c.get("up", 0), e["id"] if e else None, e.get("amount", -1) if e else 0,
           f["id"] if f else None, f.get("amount", -1) if f else 0, tuple(kw) if kw is not None else None)
    row = _ROWS.get(key)
    if row is None:
        cid = c["id"]
        upgraded = c.get("up", 0) > 0 or cid.endswith("+")
        meta = T.CARDS.get(cid.rstrip("+")) or {}
        row = np.array([T.token("cards", cid), c.get("cost", 0), meta.get("type", 0), meta.get("rarity", 0), 1 if upgraded else 0,
                        0 if upgraded else (1 if meta.get("upgradable") else 0),
                        T.token("enchantments", e["id"]) if e else -1, e.get("amount", -1) if e else -1,
                        T.token("afflictions", f["id"]) if f else -1, f.get("amount", -1) if f else -1,
                        _keyword_mask(c, meta)], np.float32)
        if len(_ROWS) < 200_000:
            _ROWS[key] = row
    return row


def _card(a: np.ndarray, o: int, c: dict) -> None:
    """One Card(11) block. The observation marks an upgrade with `up` (a level; some ids also carry a "+")."""
    a[o:o + _CARD] = _card_row(c)


def _pile(a: np.ndarray, base: int, cards: list) -> None:
    for i, c in enumerate(cards[:bl.MAX_DECK_CARDS]):
        _card(a, base + i * _CARD, c)


def encode_combat(obs: dict, legal: list[str], ctx: dict | None = None) -> np.ndarray:
    """One combat decision. `legal` is the host's/engine's action list ("play:i[:t]", "end_turn", "potion:i", ...): a hand
    card is playable when some play:i is legal. `ctx` may carry the run's {"floor", "act_index", "boss1", "boss2",
    "deck", "relics", "potions"} where the combat observation alone does not know them."""
    ctx = ctx or {}
    a = _TEMPLATE.copy()
    a[bl.DECISION_TYPE] = _DT_COMBAT_PLAY
    p = obs["player"]
    hand, draw, discard, exhaust = obs["hand"], obs["draw"], obs["discard"], obs["exhaust"]

    # PlayerSummary
    s = bl.PLAYER_SUMMARY
    a[s] = T.token("characters", p["character"])
    a[s + 1], a[s + 2], a[s + 3], a[s + 4] = p["hp"], p["max_hp"], p.get("block", 0), p.get("gold", 0)
    potions = obs.get("potions") or []
    a[s + 5] = len(potions)
    deck = ctx.get("deck") or (hand + draw + discard + exhaust)
    a[s + 6] = len(deck)
    for i, r in enumerate((ctx.get("relics") or obs.get("relics") or [])[:bl.MAX_RELICS]):
        o = s + bl._PLAYER_RELICS_OFFSET + i * 2
        a[o] = T.token("relics", r["id"])
        a[o + 1] = r.get("counter", -1)
    for i, pt in enumerate(potions[:bl.MAX_POTIONS]):
        if pt:
            o = s + bl._PLAYER_POTIONS_OFFSET + i * 2
            a[o] = T.token("potions", pt["id"] if isinstance(pt, dict) else pt)
            a[o + 1] = -1
    for i, c in enumerate(deck[:bl.MAX_DECK_CARDS]):
        _card(a, s + bl._PLAYER_DECK_OFFSET + i * _CARD, c)

    # RunContext (what the combat knows: its act; the rest from ctx)
    rc = bl.RUN_CONTEXT
    a[rc] = ctx.get("act_index", obs.get("act", 0) + 1)
    a[rc + 2] = ctx.get("floor", 0)
    a[rc + 3] = T.token("encounters", ctx.get("boss1"))
    a[rc + 4] = T.token("encounters", ctx.get("boss2"))

    cp = bl.COMBAT_PLAY
    a[cp], a[cp + 1], a[cp + 2] = obs["round"], p["energy"], p["max_energy"]
    playable = {int(x.split(":")[1]) for x in legal if x.startswith("play:")}  # play:<card uid>[:<enemy uid>]
    enemies = [e for e in obs["enemies"] if e["hp"] > 0][:bl.MAX_ENEMIES]
    for i, c in enumerate(hand[:bl.MAX_HAND_CARDS]):
        o = cp + bl._CP_HAND_OFFSET + i * _HAND
        _card(a, o, c)
        meta = T.CARDS.get(c["id"].rstrip("+")) or {}
        a[o + bl._HAND_CAN_PLAY] = 1 if c.get("uid", i) in playable else 0
        a[o + bl._HAND_STAR_COST] = c.get("stars", 0) or 0
        a[o + bl._HAND_TARGET_TYPE] = meta.get("target", -1)
        v = {k.lower(): x for k, x in (c.get("vars") or {}).items()}
        if "calc" in c:
            v.setdefault("calculateddamage", c["calc"])
        for j, key in enumerate(_STAT_KEYS):
            a[o + bl._HAND_STATS_OFFSET + j] = v.get(key, -1)
        dmg = v.get("damage")
        if dmg is not None:
            for ti in range(len(enemies)):
                a[o + bl._HAND_DAMAGE_BY_TARGET_OFFSET + 2 * ti] = dmg
                a[o + bl._HAND_DAMAGE_BY_TARGET_OFFSET + 2 * ti + 1] = v.get("repeat", 1)

    for i, e in enumerate(enemies):
        o = cp + bl._CP_ENEMIES_OFFSET + i * _ENEMY
        a[o] = T.token("monsters", e["id"])
        a[o + 1], a[o + 2], a[o + 3] = e["hp"], e["max_hp"], e.get("block", 0)
        intents = e.get("intents") or []
        a[o + 4] = 1 if any(x.get("type") == "Attack" for x in intents) else 0
        for j, x in enumerate(intents[:bl.MAX_INTENTS_PER_ENEMY]):
            a[o + bl._ENEMY_INTENTS_OFFSET + 2 * j] = T.INTENT_INDEX.get(x.get("type"), -1)
            a[o + bl._ENEMY_INTENTS_OFFSET + 2 * j + 1] = x["damage"] * x.get("hits", 1) if "damage" in x else -1
        for j, pw in enumerate((e.get("powers") or [])[:bl.MAX_ENEMY_POWERS]):
            a[o + bl._ENEMY_POWERS_OFFSET + 2 * j] = T.token("powers", pw["id"])
            a[o + bl._ENEMY_POWERS_OFFSET + 2 * j + 1] = pw.get("amount", 0)
    for j, pw in enumerate((p.get("powers") or [])[:bl.MAX_PLAYER_POWERS]):
        a[cp + bl._CP_PLAYER_POWERS_OFFSET + 2 * j] = T.token("powers", pw["id"])
        a[cp + bl._CP_PLAYER_POWERS_OFFSET + 2 * j + 1] = pw.get("amount", 0)

    a[cp + bl._CP_PILE_COUNTS_OFFSET], a[cp + bl._CP_PILE_COUNTS_OFFSET + 1], a[cp + bl._CP_PILE_COUNTS_OFFSET + 2] = \
        len(draw), len(discard), len(exhaust)
    _pile(a, cp + bl._CP_DRAW_PILE_OFFSET, draw)
    _pile(a, cp + bl._CP_DISCARD_PILE_OFFSET, discard)
    _pile(a, cp + bl._CP_EXHAUST_PILE_OFFSET, exhaust)

    for i, orb in enumerate((p.get("orbs") or [])[:bl.MAX_ORBS]):
        o = cp + bl._CP_ORBS_OFFSET + i * 3
        a[o] = T.token("orbs", orb["id"])
        a[o + 1], a[o + 2] = orb.get("passive", -1), orb.get("evoke", -1)
    a[cp + bl._CP_ORB_CAPACITY_OFFSET] = p.get("orb_slots", 0)
    a[cp + bl._CP_STARS_OFFSET] = p.get("stars", 0) if p["character"] == "REGENT" else -1
    allies = obs.get("allies") or []
    osty = next((x for x in allies if x.get("id") == "OSTY"), None)
    if osty:
        o = cp + bl._CP_OSTY_OFFSET
        a[o], a[o + 1], a[o + 2], a[o + 3] = osty["hp"], osty["max_hp"], osty.get("block", 0), 1 if osty["hp"] > 0 else 0
    elif p["character"] == "NECROBINDER":
        a[cp + bl._CP_OSTY_OFFSET + 3] = 0
    return a


def encode_card_select(obs: dict, ctx: dict | None = None) -> np.ndarray:
    """A card prompt (the observation's `prompt`: context, min, max, options) as buffer's card_select decision, on top of the
    combat the prompt interrupts (hand, piles and the rest stay as encode_combat wrote them: the fields are separate
    sections of one record)."""
    a = encode_combat(obs, [], ctx)
    a[bl.DECISION_TYPE] = 10  # card_select
    pr = obs["prompt"]
    base = bl.CARD_SELECT
    a[base], a[base + 1] = pr["min"], pr["max"]
    for i, c in enumerate((pr.get("options") or [])[:bl.MAX_DECK_CARDS]):
        _card(a, base + 2 + i * _CARD, c)
    return a
