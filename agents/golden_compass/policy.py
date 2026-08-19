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
