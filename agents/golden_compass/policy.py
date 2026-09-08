"""Golden Compass placeholder policy: a value-to-go path plan over the full act map, backed by
per-node-type weights rather than a trained model -- same "reasonable, not trained" spirit as
choices_paradox/policy.py. Duck-typed on the dict shape sts2-cli's `map` command returns
(RunSimulator.GetFullMap): {"rows": [[{col,row,type,children:[{col,row},...]}, ...], ...],
"boss": {col,row,type,...}, "current_coord": {col,row} | null}. `map_select`'s one-hop `choices`
list only shows the immediate children of the current node -- always plan off the full `map`
query instead, since the whole act graph is visible for free, same as a human sees it.

Strategy: assign each node type a scalar value, then DP backward from the boss (the DAG's only
sink) so every node knows its best achievable cumulative value to the boss. At each junction,
walk toward whichever reachable child carries the highest DP value. This naturally routes through
rest sites before the boss and through treasure/shops along the way without special-casing them --
it falls out of the DP once boss is the fixed endpoint.
"""
from __future__ import annotations

from typing import Any

Coord = tuple[int, int]

_NODE_VALUE = {
    "Treasure": 3.0,   # guaranteed relic/gold, zero combat risk
    "RestSite": 2.0,   # heal or upgrade -- always useful, never wasted
    "Shop": 1.5,        # removal + purchases; choices_paradox needs shops to prune the deck
    "Unknown": 0.5,     # unresolved "?" event -- mildly positive EV, treated as neutral-good
    "Elite": 2.5,       # best rewards in the act, but gated by HP below
    "Monster": 0.5,     # baseline filler combat: a card reward paid for in HP
    "Ancient": 0.0,
    "Boss": 0.0,
}

_ELITE_HP_FLOOR = 0.4  # below this HP fraction an elite's upside isn't worth the risk


def _node_value(node_type: str, hp_frac: float) -> float:
    base = _NODE_VALUE.get(node_type, 0.0)
    if node_type == "Elite":
        return base * hp_frac if hp_frac >= _ELITE_HP_FLOOR else -1.0
    return base


def _coord(d: dict[str, Any]) -> Coord:
    return (d["col"], d["row"])


def _best_path_values(map_json: dict[str, Any], hp_frac: float) -> dict[Coord, float]:
    """DP value-to-go per node, computed row-by-row from the boss backward. Rows are already
    laid out by floor depth (children always sit one row past their parent), so a single reverse
    pass is a valid topological order -- no need for a general graph toposort."""
    rows: list[list[dict]] = map_json["rows"]
    boss = map_json["boss"]

    values: dict[Coord, float] = {_coord(boss): _node_value(boss["type"], hp_frac)}
    for row in reversed(rows):
        for node in row:
            children = node.get("children") or []
            best_child = max((values[_coord(c)] for c in children), default=0.0)
            values[_coord(node)] = _node_value(node["type"], hp_frac) + best_child
    return values


def choose_next_node(map_json: dict[str, Any], hp_frac: float) -> Coord | None:
    """Coord to move to next: the reachable child of the current position with the highest
    value-to-go. `hp_frac` should be current_hp / max_hp, used to gate elite-taking.

    Returns None before the act is entered (`current_coord` is null) -- the engine's own
    `map_select` hands back a single forced choice (the Ancient starting point) at that point,
    not present in `map`'s `rows`, so there is nothing here to decide."""
    current = map_json.get("current_coord")
    rows: list[list[dict]] = map_json["rows"]

    if current is None:
        return None

    current_coord = (current["col"], current["row"])
    node = next((n for row in rows for n in row if _coord(n) == current_coord), None)
    if node is None:
        return None

    children = node.get("children") or []
    if not children:
        return _coord(map_json["boss"])

    values = _best_path_values(map_json, hp_frac)
    best = max(children, key=lambda c: values.get(_coord(c), 0.0))
    return _coord(best)


# --- buffer-protocol counterpart --------------------------------------------------------------
#
# sts2-cli's Sts2BufferSession (the session type both facility_claws.py's and
# facility_choices_paradox.py's main lines actually run on) has no GetFullMap()-shaped JSON to
# read -- the map lives in the flat buffer_layout.MAP grid instead (buffer_layout.map_points()/
# map_point_children()/map_point_type()). Same DP-backward-from-boss reasoning as above, just
# reading that wire format instead of the pydantic one.


def _current_children_from_buffer(arr) -> list[Coord]:
    """Physical one-hop children reachable from the map's current position. Falls back to every
    row-0 point when nothing is marked current yet (the very first move of an act -- the buffer
    protocol has no null current_coord signal the way GetFullMap()'s JSON does, so this is the
    buffer-side equivalent of choose_next_node's own `current is None` case), and to the (-1, -1)
    boss sentinel (see facility_common._map_point_type_at) when the current node has no real
    children left: map_point_children() can never yield that sentinel itself -- a genuinely absent
    child slot and "child is the boss" are both encoded as the same (-1, -1) padding, so that
    function's own filter drops both alike -- this infers "boss" from "no real children remain"
    instead."""
    import buffer_layout as bl

    current_children = None
    for _row, _col, slot in bl.map_points(arr):
        if int(slot[2]) == 1:  # MAP_POINT_SIZE layout: type, visited, current, then children
            current_children = list(bl.map_point_children(slot))
            break
    if current_children is None:
        current_children = [(col, row) for row, col, _slot in bl.map_points(arr) if row == 0]
    return current_children or [(-1, -1)]


def _best_path_values_from_buffer(arr, hp_frac: float) -> dict[Coord, float]:
    """Buffer-array counterpart to _best_path_values -- same reverse-topological-order reasoning
    (children always sit one row past their parent), read off buffer_layout.map_points() instead
    of GetFullMap()'s JSON `rows`."""
    import buffer_layout as bl

    nodes: dict[Coord, tuple[str | None, list[Coord]]] = {}
    max_row = -1
    for row, col, slot in bl.map_points(arr):
        if row == -1:
            continue  # boss point itself -- contributes 0 either way (_NODE_VALUE["Boss"] == 0.0)
        nodes[(col, row)] = (bl.map_point_type(slot), list(bl.map_point_children(slot)))
        max_row = max(max_row, row)

    values: dict[Coord, float] = {}
    for row in range(max_row, -1, -1):
        for coord, (node_type, children) in nodes.items():
            if coord[1] != row:
                continue
            best_child = max((values[c] for c in children if c in values), default=0.0)
            values[coord] = _node_value(node_type, hp_frac) + best_child
    return values


def choose_next_node_from_buffer(arr, hp_frac: float) -> tuple[Coord, list[Coord]]:
    """Buffer-array counterpart to choose_next_node -- returns (best_child, all_children) rather
    than just the best pick, since facility_common._resolve_map_decision needs the full sibling
    list too (to branch over every option, not just adopt the best one), not only whispering_
    earring/policy.py's _choose_map_action_from_buffer (which only wants the pick)."""
    children = _current_children_from_buffer(arr)
    if len(children) == 1:
        return children[0], children
    values = _best_path_values_from_buffer(arr, hp_frac)
    best = max(children, key=lambda c: values.get(c, 0.0))
    return best, children
