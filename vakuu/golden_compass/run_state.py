"""RunState: the hazard net's view of a run at one macro decision, plus the "afterstate" builders
that turn "what if I pick X" into the RunState the net should score.

A RunState is plain JSON-able data (strings and numbers, no sts2 objects), so the facility can log
it straight into a run file and materialize.py / train.py never need the game installed. Only
encode_from_buffer() reads the buffer record (vakuu.buffer), and imports it lazily.

The map is the one non-obvious part. It is encoded as a bag of reachable-node tokens -- "Elite+3"
is "an elite three rows ahead of the node I'm standing on" -- with NO edges. Two rooms on one path
and one room on each of two paths read the same, so the net can't memorise route shapes; it can
only learn that a position with more options ahead, or a rest site before the elite, hazards less.
That is the idea borrowed from spire-zero's run value net (see README.md).

`floor` is the first floor whose death this state predicts: the room the player is about to enter,
i.e. the floor of the room they are in now, plus one. A map choice's afterstate (the chosen node
becomes "here", prepended as "<Type>+0") has the same `floor`; the "+0" token is what tells the
net the room is being entered rather than behind it.

`remaining` is the encounter ids the act can still roll ("you already beat elite A, so you'll only see B or C"). The buffer
doesn't carry it: the facility computes it (spire/pools.py) and passes it to encode_from_buffer. [] for logs before it existed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

MAP_TYPES = ["Unknown", "Shop", "Treasure", "RestSite", "Monster", "Elite", "Boss", "Ancient"]
MAX_DELTA = 20  # MAX_MAP_ROWS; "Boss+9" etc. never exceed it


@dataclass
class RunState:
    character: str
    act: int
    ascension: int
    floor: int  # first floor whose death this state predicts -- see module docstring
    hp: int
    max_hp: int
    gold: int
    max_potions: int
    boss: list[str] = field(default_factory=list)
    deck: list[list] = field(default_factory=list)  # [card_id, upgraded(int), enchant_id | None, upgradable(int)]
    relics: list[list] = field(default_factory=list)  # [relic_id, counter]
    potions: list[str] = field(default_factory=list)
    map: list[str] = field(default_factory=list)  # "<Type>+<rows ahead>" per reachable node
    remaining: list[str] = field(default_factory=list)  # encounter ids still possible this act

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "RunState":
        return cls(**d)


# --- afterstates: the RunState a choice leads to -------------------------------------------------

def with_card(state: RunState, card_id: str | None, upgraded: int = 0, enchant: str | None = None) -> RunState:
    return replace(state, deck=state.deck + [[card_id, int(upgraded), enchant, 0 if upgraded else 1]])


def at_ancient(state: RunState) -> RunState:
    """The state at act 2's Ancient, reached from `state`: the same deck, relics and potions, full HP. What the hazard head
    is asked for the floors after the act 1 boss, where the current HP no longer matters. The act 2 boss is not known yet
    and the map is what an Ancient state carries (the boss's node), as in the logged act 2 starts."""
    return replace(state, act=2, floor=2, hp=state.max_hp, boss=[], map=["Boss+1"], remaining=[])


def with_hp(state: RunState, hp: int) -> RunState:
    return replace(state, hp=max(0, min(state.max_hp, int(hp))))


def card_key(card: list) -> tuple:
    """What makes two deck entries interchangeable: (id, upgraded, enchant)."""
    return (card[0], int(card[1]), card[2])


def _upgradable(card: list) -> bool:
    return bool(card[3]) if len(card) > 3 else True


def upgrade_candidates(state: RunState) -> list[tuple]:
    """Distinct keys of the deck cards a smith could upgrade (duplicates collapse: upgrading either
    Strike is the same afterstate)."""
    seen: dict[tuple, None] = {}
    for c in state.deck:
        if _upgradable(c):
            seen.setdefault(card_key(c))
    return list(seen)


def with_upgraded(state: RunState, key: tuple) -> RunState:
    """One copy of the card `key` upgraded (upgraded=1, no longer upgradable)."""
    deck = [list(c) for c in state.deck]
    for c in deck:
        if card_key(c) == key:
            c[1], c[3] = max(1, int(c[1]) + 1), 0
            break
    return replace(state, deck=deck)


def without_card(state: RunState, key: tuple) -> RunState:
    deck = [list(c) for c in state.deck]
    for i, c in enumerate(deck):
        if card_key(c) == key:
            del deck[i]
            break
    return replace(state, deck=deck)


def distinct_cards(state: RunState) -> list[tuple]:
    seen: dict[tuple, None] = {}
    for c in state.deck:
        seen.setdefault(card_key(c))
    return list(seen)


def with_relic(state: RunState, relic_id: str | None) -> RunState:
    return replace(state, relics=state.relics + [[relic_id, -1]])


def with_potion(state: RunState, potion_id: str | None) -> RunState:
    return replace(state, potions=state.potions + [potion_id])


def with_gold(state: RunState, delta: int) -> RunState:
    return replace(state, gold=max(0, state.gold + int(delta)))


def healed(state: RunState) -> RunState:
    """A rest-site Heal: 30% of max HP (HealRestSiteOption.GetBaseHealAmount), +15 with Regal Pillow."""
    amount = int(state.max_hp * 0.3) + (15 if any(r[0] == "REGAL_PILLOW" for r in state.relics) else 0)
    return with_hp(state, state.hp + amount)


# --- the buffer -> RunState encoder ----------------------------------------------------------------

def _map_graph(arr):
    """({(col,row): (type, [(col,row) children])}, current (col,row) | None). The boss has no grid
    coordinate of its own in the buffer; it is recovered as the child key that isn't a grid node."""
    from vakuu.buffer import layout as bl

    graph: dict[tuple[int, int], tuple[str | None, list[tuple[int, int]]]] = {}
    current = None
    boss_type = None
    for row, col, slot in bl.map_points(arr):
        if row == -1:
            boss_type = bl.map_point_type(slot)
            continue
        graph[(col, row)] = (bl.map_point_type(slot), list(bl.map_point_children(slot)))
        if int(slot[2]) == 1:  # MAP_POINT_SIZE layout: type, visited, current, then children
            current = (col, row)
    for _t, kids in list(graph.values()):
        for kid in kids:
            if kid not in graph:
                graph[kid] = (boss_type or "Boss", [])
    return graph, current


def _reachable(graph, starts) -> set[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    stack = list(starts)
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        stack.extend(graph[key][1])
    return seen


def _tokens(graph, keys, here_row: int) -> list[str]:
    out = []
    for col, row in keys:
        node_type = graph[(col, row)][0]
        if node_type is None or node_type == "Unassigned":
            continue
        out.append(f"{node_type}+{min(max(row - here_row, 0), MAX_DELTA)}")
    return sorted(out)


def map_options(arr) -> list[tuple[int, int]]:
    """The (col, row) nodes a select_map_node can move to right now."""
    graph, current = _map_graph(arr)
    if current is None:  # no node marked current: act start, every row-0 point is offered
        return sorted(k for k in graph if k[1] == 0)
    return list(graph[current][1])


def encode_from_buffer(arr, character: str | None, ascension: int, remaining: list[str] | None = None) -> RunState:
    from vakuu.buffer import layout as bl
    from vakuu.buffer.decision import decode_token
    ctx = bl.run_context(arr)
    graph, current = _map_graph(arr)
    if current is None:
        here_row, children = -1, sorted(k for k in graph if k[1] == 0)
    else:
        here_row, children = current[1], graph[current][1]
    reach = _reachable(graph, children)

    deck = [
        [decode_token(bl.card_token(slot)), int(slot[bl.CARD_UPGRADED]),
         decode_token(int(slot[bl.CARD_ENCHANT_TOKEN])), int(slot[bl.CARD_UPGRADABLE])]
        for _, slot in bl.deck_cards(arr)
    ]
    relics = [[decode_token(int(s[bl.RELIC_TOKEN])), int(s[bl.RELIC_COUNTER])] for _, s in bl.player_relics(arr)]
    potions = [decode_token(int(s[bl.POTION_TOKEN])) for _, s in bl.player_potions(arr)]
    bosses = [b for b in (decode_token(ctx["boss1_token"]), decode_token(ctx["boss2_token"])) if b]

    return RunState(
        character=(character or "UNKNOWN").upper(),
        act=ctx["act_index"],
        ascension=int(ascension),
        floor=ctx["floor"] + 1,
        hp=bl.player_hp(arr),
        max_hp=bl.player_max_hp(arr),
        gold=bl.player_gold(arr),
        max_potions=bl.player_max_potions(arr),
        boss=bosses,
        deck=deck,
        relics=relics,
        potions=potions,
        map=_tokens(graph, reach, here_row),
        remaining=list(remaining or []),
    )


def map_afterstates(arr, base: RunState) -> list[tuple[tuple[int, int], RunState]]:
    """One (col, row, afterstate) per legal select_map_node: the chosen node becomes "here" ("<Type>+0"
    plus everything reachable from it, measured from it). Everything else about the run is unchanged
    -- entering a room hasn't cost anything yet."""
    graph, _current = _map_graph(arr)
    out = []
    for key in map_options(arr):
        reach = _reachable(graph, [key])
        out.append((key, replace(base, map=_tokens(graph, reach, key[1]))))
    return out
