"""Registry of every extracted MonsterMoveGraph, merged from this package's data_*.py chunk files
(one per extraction batch -- see validate.py) so each chunk can be authored/reviewed independently
without merge conflicts on a single giant file."""

from __future__ import annotations

import importlib
import pkgutil

from .resolver import (
    DYNAMIC_CONDITIONS,
    DYNAMIC_WEIGHTS,
    NoConditionMatchedError,
    NoEligibleBranchError,
    ResolverError,
    UnknownStateError,
    blend_embeddings,
    resolve_next_move_distribution,
)
from .schema import ConditionalTarget, MonsterMoveGraph, MoveTarget

MONSTER_GRAPHS: dict[str, MonsterMoveGraph] = {}


def _load_chunks() -> None:
    for mod_info in pkgutil.iter_modules(__path__):
        if not mod_info.name.startswith("data_"):
            continue
        mod = importlib.import_module(f"{__name__}.{mod_info.name}")
        graphs = getattr(mod, "GRAPHS", None)
        if graphs is None:
            raise ImportError(f"{mod_info.name} has no module-level GRAPHS dict")
        for monster_id, graph in graphs.items():
            if monster_id in MONSTER_GRAPHS:
                raise ValueError(
                    f"{monster_id} defined in both {mod_info.name} and another data_ chunk"
                )
            if monster_id != graph.monster_id:
                raise ValueError(
                    f"{mod_info.name}: GRAPHS key {monster_id!r} != graph.monster_id {graph.monster_id!r}"
                )
            MONSTER_GRAPHS[monster_id] = graph


_load_chunks()


def get_graph(monster_id: str) -> MonsterMoveGraph:
    try:
        return MONSTER_GRAPHS[monster_id]
    except KeyError:
        raise KeyError(
            f"no extracted move graph for monster_id={monster_id!r} -- "
            f"{len(MONSTER_GRAPHS)} monsters loaded, see monster_patterns/data_*.py"
        ) from None


__all__ = [
    "MONSTER_GRAPHS",
    "MonsterMoveGraph",
    "MoveTarget",
    "ConditionalTarget",
    "get_graph",
    "resolve_next_move_distribution",
    "blend_embeddings",
    "ResolverError",
    "UnknownStateError",
    "NoEligibleBranchError",
    "NoConditionMatchedError",
    "DYNAMIC_WEIGHTS",
    "DYNAMIC_CONDITIONS",
]
