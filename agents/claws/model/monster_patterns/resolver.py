"""Resolves a MonsterMoveGraph (schema.py) into a probability distribution over the monster's
*next* move, given its currently-revealed move and recent move history.

This is a direct Python port of MonsterMoveStateMachine.FindNextMoveState /
RandomBranchState.GetNextState+GetStateWeight / ConditionalBranchState.GetNextState (see
sts2-decompiled's MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine) -- except where the C#
draws one `rng.NextFloat()` sample, this walks every branch and returns the full weighted
distribution over reachable moves instead of a single draw.

Why `history` matters: RandomBranchState weights aren't a fixed per-edge constant -- MoveRepeatType
(CannotRepeat/CanRepeatXTimes/UseOnlyOnce) and per-edge cooldowns gate a candidate's eligibility
against MonsterMoveStateMachine.StateLog, i.e. which moves this monster has actually performed
recently. `history` here is that same signal: the chronological move_id sequence for this monster
instance, most-recent (== the move currently telegraphed as `NextMove`/`move_id`) last -- exactly
what sts2-cli's buffer (`enemy_move_hash`) / JSON (`move_id`) fields now expose turn over turn (see
sts2-cli's RunSimulator.States.Buffer.cs / Writers.Json.cs). StateLog only ever contains moves (both
branch node types set ShouldAppearInLogs=False), so no filtering is needed on the caller's side.
"""

from __future__ import annotations

from typing import Callable

from .schema import ConditionalTarget, MonsterMoveGraph, MoveTarget

# Keyed by (monster_id, branch_state_id, target_state_id) -- filled in by hand for the (expected
# small) set of monsters whose AddBranch weight is a Func<float> reading live game state (HP%,
# ascension, round number, ...) rather than a constant. A MoveTarget with weight=None and no entry
# here falls back to weight=1.0 (even odds against its siblings) -- see resolve_next_move_distribution's
# `unresolved_dynamic` return for how to detect when that fallback fired.
DynamicWeightFn = Callable[[dict | None], float]
DYNAMIC_WEIGHTS: dict[tuple[str, str, str], DynamicWeightFn] = {}

# Keyed the same way, for ConditionalBranchState AddState(...) predicates. A ConditionalTarget with
# condition_expr set and no entry here evaluates to False (never silently assume an unresolved
# game-state gate is satisfied) -- so resolution conservatively falls through to a later/else edge.
DynamicConditionFn = Callable[[dict | None], bool]
DYNAMIC_CONDITIONS: dict[tuple[str, str, str], DynamicConditionFn] = {}


class ResolverError(Exception):
    """Base for resolution failures -- these mirror the C# InvalidOperationExceptions
    FindNextMoveState/RandomBranchState/ConditionalBranchState throw when their own live logic
    hits an inconsistent state; here they mean the extracted graph doesn't match a real reachable
    game state, which should never happen for a `history` actually observed from the buffer/JSON
    protocol."""


class UnknownStateError(ResolverError):
    def __init__(self, monster_id: str, state_id: str):
        super().__init__(f"{monster_id}: unknown state id {state_id!r}")


class NoEligibleBranchError(ResolverError):
    def __init__(self, monster_id: str, branch_id: str, history: list[str]):
        super().__init__(
            f"{monster_id}: every edge out of random branch {branch_id!r} is ineligible given "
            f"history={history!r} -- either the graph is incomplete or history is stale/wrong"
        )


class NoConditionMatchedError(ResolverError):
    def __init__(self, monster_id: str, branch_id: str):
        super().__init__(
            f"{monster_id}: no AddState condition matched in {branch_id!r} and it has no "
            f"trailing unconditional else -- extraction likely missed an edge"
        )


def _repeat_factor(edge: MoveTarget, history: list[str]) -> float:
    """Port of RandomBranchState.GetStateWeight's repeat_type branch. `history` is chronological,
    most-recent last -- mirrors StateLog's append order exactly."""
    if edge.repeat_type == "UseOnlyOnce":
        return 0.0 if edge.to in history else 1.0
    if edge.repeat_type == "CanRepeatForever":
        return 1.0
    threshold = 1 if edge.repeat_type == "CannotRepeat" else edge.max_times
    n = len(history)
    if n < threshold:
        return 1.0
    # Eligible unless the last `threshold` entries are ALL this edge's target (i.e. it's already
    # been picked threshold times in a row) -- matches the C# loop exactly, including its
    # short-circuit on the first differing entry.
    for i in range(threshold):
        idx = n - 1 - i
        if idx < 0:
            break
        if history[idx] != edge.to:
            return 1.0
    return 0.0


def _base_weight(
    edge: MoveTarget, monster_id: str, branch_id: str, context: dict | None,
    dynamic_weights: dict[tuple[str, str, str], DynamicWeightFn], unresolved: list[tuple[str, str, str]],
) -> float:
    if edge.weight is not None:
        return edge.weight
    fn = dynamic_weights.get((monster_id, branch_id, edge.to))
    if fn is not None:
        return fn(context)
    unresolved.append((monster_id, branch_id, edge.to))
    return 1.0


def _edge_weight(
    edge: MoveTarget, history: list[str], monster_id: str, branch_id: str, context: dict | None,
    dynamic_weights: dict[tuple[str, str, str], DynamicWeightFn], unresolved: list[tuple[str, str, str]],
) -> float:
    # Cooldown short-circuits to 0 regardless of repeat_factor -- matches GetStateWeight's early
    # `return 0f` before it ever multiplies in the base weight.
    if edge.cooldown > 0 and edge.to in history[-edge.cooldown:]:
        return 0.0
    return _repeat_factor(edge, history) * _base_weight(
        edge, monster_id, branch_id, context, dynamic_weights, unresolved
    )


def _condition_true(
    edge: ConditionalTarget, monster_id: str, branch_id: str, context: dict | None,
    dynamic_conditions: dict[tuple[str, str, str], DynamicConditionFn], unresolved: list[tuple[str, str, str]],
) -> bool:
    if edge.condition_expr is None:
        return True
    fn = dynamic_conditions.get((monster_id, branch_id, edge.to))
    if fn is not None:
        return fn(context)
    unresolved.append((monster_id, branch_id, edge.to))
    return False


def _walk(
    graph: MonsterMoveGraph, state_id: str, history: list[str], context: dict | None, prob: float,
    acc: dict[str, float], dynamic_weights: dict, dynamic_conditions: dict, unresolved: list,
) -> None:
    if prob <= 0:
        return
    if state_id in graph.move_ids:
        acc[state_id] = acc.get(state_id, 0.0) + prob
        return
    if state_id in graph.random_branches:
        edges = graph.random_branches[state_id]
        weights = [
            _edge_weight(e, history, graph.monster_id, state_id, context, dynamic_weights, unresolved)
            for e in edges
        ]
        total = sum(weights)
        if total <= 0:
            raise NoEligibleBranchError(graph.monster_id, state_id, history)
        for e, w in zip(edges, weights):
            if w > 0:
                _walk(graph, e.to, history, context, prob * w / total, acc,
                      dynamic_weights, dynamic_conditions, unresolved)
        return
    if state_id in graph.conditional_branches:
        for e in graph.conditional_branches[state_id]:
            if _condition_true(e, graph.monster_id, state_id, context, dynamic_conditions, unresolved):
                _walk(graph, e.to, history, context, prob, acc,
                      dynamic_weights, dynamic_conditions, unresolved)
                return
        raise NoConditionMatchedError(graph.monster_id, state_id)
    raise UnknownStateError(graph.monster_id, state_id)


def resolve_next_move_distribution(
    graph: MonsterMoveGraph,
    history: list[str],
    context: dict | None = None,
    dynamic_weights: dict[tuple[str, str, str], DynamicWeightFn] | None = None,
    dynamic_conditions: dict[tuple[str, str, str], DynamicConditionFn] | None = None,
) -> tuple[dict[str, float], list[tuple[str, str, str]]]:
    """Returns (distribution, unresolved) where `distribution` is {move_id: probability} over
    every move reachable from the state after `history[-1]` (the currently-telegraphed move), and
    `unresolved` lists any (monster_id, branch_id, target_id) edges whose weight/condition needed
    a DYNAMIC_WEIGHTS/DYNAMIC_CONDITIONS entry that wasn't registered -- non-empty means the
    distribution used a fallback (even weight / false condition) for at least one edge, so treat it
    as approximate until that case is hand-ported.

    `history` should end with the move currently telegraphed (the one you'd read off
    move_id/enemy_move_hash right now) -- mirrors StateLog including its own most recent entry,
    since that's exactly what RandomBranchState.GetStateWeight sees when the *next* move gets
    rolled. Pass an empty `history` only for the very first move of a fresh combat (no move
    revealed yet) -- this resolves from graph.initial_state instead of a move's followup, matching
    MonsterMoveStateMachine's own _currentState = _initialState start. Pass module-level
    DYNAMIC_WEIGHTS/DYNAMIC_CONDITIONS explicitly (or merge your own on top) rather than relying on
    the default -- the defaults start empty.
    """
    if not history:
        start = graph.initial_state
    else:
        current = history[-1]
        if current not in graph.move_followup:
            raise UnknownStateError(graph.monster_id, current)
        start = graph.move_followup[current]
    dynamic_weights = dynamic_weights if dynamic_weights is not None else DYNAMIC_WEIGHTS
    dynamic_conditions = dynamic_conditions if dynamic_conditions is not None else DYNAMIC_CONDITIONS
    acc: dict[str, float] = {}
    unresolved: list[tuple[str, str, str]] = []
    _walk(graph, start, history, context, 1.0, acc, dynamic_weights, dynamic_conditions, unresolved)
    return acc, unresolved


def blend_embeddings(distribution: dict[str, float], embedding_of: Callable[[str], "object"]):
    """sum_i p_i * embedding_of(move_id_i) -- the weighted-token blend this whole module exists
    to feed. `embedding_of` is whatever the caller's embedding lookup is (a dict, a callable
    wrapping an nn.Embedding row, ...); this function is deliberately torch/numpy-agnostic, just
    scalar-weighted addition, so it works with either."""
    total = None
    for move_id, p in distribution.items():
        term = embedding_of(move_id) * p
        total = term if total is None else total + term
    return total
