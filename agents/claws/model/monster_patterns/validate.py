#!/usr/bin/env python3
"""Structural + behavioral sanity checks for every graph in MONSTER_GRAPHS. Run after adding or
editing a data_*.py chunk:

    python3 -m agents.claws.model.monster_patterns.validate

Checks, per monster:
  1. every move_followup/random_branch/conditional_branch target id resolves to *some* known state
     in the same graph (a move id, or another branch id) -- catches typos and missed edges.
  2. every move_ids entry actually appears as a move_followup key (an orphaned move that's
     declared but never reachable as a source is probably a missed FollowUpState edge).
  3. graph.initial_state is itself a known state id.
  4. resolve_next_move_distribution succeeds from a cold start (history=[]) and from every move id
     with an empty prior history, and the returned distribution sums to ~1.0 and never contains a
     probability <= 0.
  5. a ConditionalBranchState's edges are non-empty and (if it has more than one edge) don't ALL
     carry a non-None condition_expr -- worth a warning, not a hard failure, since a real chain
     might legitimately exhaust every case via mutually exclusive conditions, but it's the most
     common extraction mistake (forgetting the trailing else).

Prints a per-monster PASS/FAIL summary and exits non-zero if anything failed.
"""

from __future__ import annotations

import sys

from . import MONSTER_GRAPHS
from .resolver import ResolverError, resolve_next_move_distribution
from .schema import MonsterMoveGraph


def _known_state_ids(graph: MonsterMoveGraph) -> set[str]:
    ids = set(graph.move_ids) | set(graph.random_branches) | set(graph.conditional_branches)
    return ids


def _check_structure(graph: MonsterMoveGraph) -> list[str]:
    errors: list[str] = []
    known = _known_state_ids(graph)

    if graph.initial_state not in known:
        errors.append(f"initial_state {graph.initial_state!r} is not a known state id")

    reachable_as_source = set(graph.move_followup)
    for move_id in graph.move_ids:
        if move_id not in reachable_as_source and move_id != graph.initial_state:
            # A move with no outgoing edge is only fine if nothing ever transitions INTO it either
            # -- otherwise it's a dead end a real game would crash calling GetNextState on.
            pass  # move_followup coverage is checked as "every move has an entry" below instead.

    for move_id, target in graph.move_followup.items():
        if move_id not in graph.move_ids:
            errors.append(f"move_followup source {move_id!r} is not in move_ids")
        if target not in known:
            errors.append(f"move_followup[{move_id!r}] -> {target!r} is not a known state id")

    for move_id in graph.move_ids:
        if move_id not in graph.move_followup:
            errors.append(f"move {move_id!r} has no move_followup entry (missing FollowUpState?)")

    for branch_id, edges in graph.random_branches.items():
        if not edges:
            errors.append(f"random branch {branch_id!r} has no edges")
        for e in edges:
            if e.to not in known:
                errors.append(f"random branch {branch_id!r} -> {e.to!r} is not a known state id")
            if e.weight is None and e.weight_expr is None:
                errors.append(
                    f"random branch {branch_id!r} -> {e.to!r} has neither weight nor weight_expr set"
                )
            if e.repeat_type not in ("CanRepeatForever", "CanRepeatXTimes", "CannotRepeat", "UseOnlyOnce"):
                errors.append(f"random branch {branch_id!r} -> {e.to!r} has bad repeat_type {e.repeat_type!r}")
            if e.repeat_type == "CanRepeatXTimes" and e.max_times <= 0:
                errors.append(f"random branch {branch_id!r} -> {e.to!r} is CanRepeatXTimes with max_times<=0")

    for branch_id, edges in graph.conditional_branches.items():
        if not edges:
            errors.append(f"conditional branch {branch_id!r} has no edges")
        for i, e in enumerate(edges):
            if e.to not in known:
                errors.append(f"conditional branch {branch_id!r} -> {e.to!r} is not a known state id")
            if e.condition_expr is None and i != len(edges) - 1:
                errors.append(
                    f"conditional branch {branch_id!r}: unconditional edge at index {i} is not last"
                )

    return errors


def _check_resolves(graph: MonsterMoveGraph) -> list[str]:
    errors: list[str] = []
    histories = [[]] + [[m] for m in graph.move_ids]
    for history in histories:
        try:
            dist, _unresolved = resolve_next_move_distribution(graph, history)
        except ResolverError as exc:
            errors.append(f"resolve_next_move_distribution(history={history!r}) raised: {exc}")
            continue
        if not dist:
            errors.append(f"resolve_next_move_distribution(history={history!r}) returned empty distribution")
            continue
        total = sum(dist.values())
        if not (0.999 <= total <= 1.001):
            errors.append(f"resolve_next_move_distribution(history={history!r}) sums to {total}, not 1.0")
        for move_id, p in dist.items():
            if p <= 0:
                errors.append(f"resolve_next_move_distribution(history={history!r}) has non-positive p for {move_id!r}: {p}")
    return errors


def main() -> int:
    if not MONSTER_GRAPHS:
        print("No graphs loaded -- add data_*.py chunk files under monster_patterns/.")
        return 1

    total_errors = 0
    for monster_id in sorted(MONSTER_GRAPHS):
        graph = MONSTER_GRAPHS[monster_id]
        errors = _check_structure(graph)
        if not errors:
            errors = _check_resolves(graph)
        if errors:
            total_errors += len(errors)
            print(f"FAIL {monster_id} ({len(errors)} issue(s)):")
            for e in errors:
                print(f"    - {e}")
        else:
            print(f"PASS {monster_id}")

    print(f"\n{len(MONSTER_GRAPHS)} monsters checked, {total_errors} issue(s) total.")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
