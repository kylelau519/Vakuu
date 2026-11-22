"""In-room option search on token states -- spire-zero's "apply each option on a copy of the room and
keep going through the room's further screens", without a game copy.

The engine can't snapshot a room (native saves are map-level), so each option's result is built as a
RunState instead (run_state.py's afterstates) and scored by the hazard net. That is exact for what a
RunState sees (deck, relics, potions, gold, HP) and blind to anything else an option does.

    rest_site   HEAL / SMITH (SMITH is worth its best card to upgrade). Relics only ever append rest
                options after those two (Shovel -> Dig, Girya, Pumpkin Candle, ...), so index 0 is Heal and
                1 is Smith; the Midas run modifier, which removes Heal, would break that and is not handled.
                Appended options (Dig, Cook, Kindle, ...) have no afterstate here and are never chosen
                while Heal or Smith is available.
    shop        a beam search over sequences of buys and one removal (several things can be bought in a
                visit), BEAM states wide and at most MAX_PROBES states scored. The first action of the
                best sequence under each root option gives that option's value; the next shop screen
                re-searches with what is left.
    card_select the follow-up screen of a Smith (upgrade) or removal: one candidate per distinct card.

Every function takes `score(states) -> list[float]` (higher is better) and returns options as
`(label dict, value)` so the caller can sample and log them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vakuu.golden_compass import run_state as rs

BEAM = 16
MAX_PROBES = 1024

Score = Callable[[list], list]


@dataclass
class Option:
    name: str  # the action to take now
    kwargs: dict
    label: dict  # JSON-able description for the policy log
    value: float
    purpose: str | None = None  # what the follow-up card_select is for


# --- rest site ------------------------------------------------------------------------------------

def rest_options(arr, base: rs.RunState, score: Score) -> list[Option]:
    from vakuu.buffer import layout as bl

    slots = list(bl.rest_options(arr))
    enabled = {i for i, slot in slots if int(slot[1]) == 1}
    out: list[Option] = []
    if len(slots) >= 2:
        heal_i, smith_i = slots[0][0], slots[1][0]
        if heal_i in enabled:
            out.append(Option("choose_option", {"option_index": heal_i}, {"option": "HEAL"},
                              score([rs.healed(base)])[0]))
        if smith_i in enabled:
            keys = rs.upgrade_candidates(base)
            if keys:
                values = score([rs.with_upgraded(base, k) for k in keys])
                best = max(range(len(keys)), key=values.__getitem__)
                out.append(Option("choose_option", {"option_index": smith_i},
                                  {"option": "SMITH", "best_card": keys[best][0]}, values[best], purpose="upgrade"))
    return out


# --- card_select follow-ups -----------------------------------------------------------------------

def select_options(arr, base: rs.RunState, purpose: str, score: Score) -> list[Option] | None:
    """One option per distinct offered card, or None when this screen isn't a single-pick upgrade /
    removal this module understands (the caller falls back to its random pick)."""
    from vakuu.buffer import layout as bl

    if purpose not in ("upgrade", "remove") or bl.min_select(arr) != 1 or bl.max_select(arr) != 1:
        return None
    from vakuu.buffer.decision import decode_token
    first_index: dict[tuple, int] = {}
    for i, slot in bl.select_cards(arr):
        key = (decode_token(bl.card_token(slot)), int(slot[bl.CARD_UPGRADED]),
               decode_token(int(slot[bl.CARD_ENCHANT_TOKEN])))
        first_index.setdefault(key, i)
    if not first_index:
        return None
    apply = rs.with_upgraded if purpose == "upgrade" else rs.without_card
    keys = list(first_index)
    values = score([apply(base, k) for k in keys])
    return [
        Option("select_cards", {"indices": [first_index[k]]}, {"card": k[0], "upgraded": k[1]}, v)
        for k, v in zip(keys, values)
    ]


# --- shop ------------------------------------------------------------------------------------------

@dataclass
class _Node:
    state: rs.RunState
    first: tuple | None  # (name, kwargs) of the root action this path began with; None = the root
    bought: frozenset = frozenset()
    removed: bool = False
    value: float = float("-inf")


def _inventory(arr):
    from vakuu.buffer import layout as bl
    from vakuu.buffer.decision import decode_token
    cards = [
        (("card", i), "buy_card", {"card_index": i}, int(slot[bl.CARD_SIZE]),
         lambda st, s=slot: rs.with_card(st, decode_token(bl.card_token(s)), int(s[bl.CARD_UPGRADED]),
                                         decode_token(int(s[bl.CARD_ENCHANT_TOKEN]))))
        for i, slot in bl.shop_cards(arr) if bool(slot[bl.CARD_SIZE + 1]) and slot[bl.CARD_SIZE] >= 0
    ]
    relics = [
        (("relic", i), "buy_relic", {"relic_index": i}, int(slot[1]),
         lambda st, s=slot: rs.with_relic(st, decode_token(int(s[0]))))
        for i, slot in bl.shop_relics(arr) if bool(slot[2])
    ]
    potions = [
        (("potion", i), "buy_potion", {"potion_index": i}, int(slot[1]),
         lambda st, s=slot: rs.with_potion(st, decode_token(int(s[0]))))
        for i, slot in bl.shop_potions(arr) if bool(slot[2])
    ]
    return cards + relics + potions, bl.shop_removal_cost(arr)


def shop_options(arr, base: rs.RunState, score: Score, beam: int = BEAM, max_probes: int = MAX_PROBES) -> list[Option]:
    """Root options (leave, or any first buy / the removal) valued by the best state reachable through
    them. See the module docstring."""
    items, removal_cost = _inventory(arr)
    root = _Node(base, None)
    root.value = score([base])[0]
    probes = 1
    best_by_first: dict[tuple, float] = {}
    frontier = [root]

    while frontier and probes < max_probes:
        children: list[_Node] = []
        seen: set = set()
        for node in frontier:
            gold = node.state.gold
            for key, name, kwargs, cost, build in items:
                if key in node.bought or cost > gold:
                    continue
                if name == "buy_potion" and len(node.state.potions) >= node.state.max_potions:
                    continue
                sig = (node.bought | {key}, node.removed)
                first = node.first or (name, tuple(kwargs.items()))
                if (sig, first) in seen:
                    continue
                seen.add((sig, first))
                children.append(_Node(rs.with_gold(build(node.state), -cost), first, node.bought | {key}, node.removed))
            if removal_cost >= 0 and not node.removed and removal_cost <= gold:
                first = node.first or ("remove_card", ())
                for card in rs.distinct_cards(node.state):
                    sig = (node.bought, ("removed", card))
                    if (sig, first) in seen:
                        continue
                    seen.add((sig, first))
                    children.append(_Node(rs.with_gold(rs.without_card(node.state, card), -removal_cost),
                                          first, node.bought, True))
        children = children[:max(0, max_probes - probes)]
        if not children:
            break
        for child, v in zip(children, score([c.state for c in children])):
            child.value = v
            best_by_first[child.first] = max(best_by_first.get(child.first, float("-inf")), v)
        probes += len(children)
        frontier = sorted(children, key=lambda n: -n.value)[:beam]

    options = [Option("leave_room", {}, {"action": "leave"}, root.value)]
    for (name, kwargs), v in best_by_first.items():
        options.append(Option(name, dict(kwargs), {"action": name, **dict(kwargs)}, v,
                              purpose="remove" if name == "remove_card" else None))
    return options
