"""Typed structures for a monster's move state machine, hand-extracted from
sts2-decompiled/src/Core/Models/Monsters/*.cs's GenerateMoveStateMachine() -- see that repo's
MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine namespace for the C# this mirrors
(MonsterState, MoveState, RandomBranchState, ConditionalBranchState, MoveRepeatType). Never copy
decompiled source into this repo; these dataclasses hold only short state-id strings and weight
values read off it, not game logic or text.

Every monster's state machine is one of three node kinds, keyed by MonsterState.Id:
  - a *move* (MoveState): performs an actual combat move, has exactly one successor (its
    FollowUpState/FollowUpStateId) -- deterministic once you're in it, but that successor can
    itself be a branch node, not necessarily another move.
  - a *random branch* (RandomBranchState): picks its successor by weighted random choice among
    AddBranch(...) edges, each gated by a MoveRepeatType/cooldown rule evaluated against recent
    move history.
  - a *conditional branch* (ConditionalBranchState): picks its successor by evaluating AddState(...)
    edges in order, first true condition wins (mirrors an if/elif/else chain).

resolver.py walks this graph exactly the way MonsterMoveStateMachine.FindNextMoveState does, except
it returns the full probability distribution over reachable moves instead of drawing one rng sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MoveTarget:
    """One AddBranch(...) edge out of a RandomBranchState.

    Mirrors RandomBranchState.StateWeight: `weight` is the resolved float when the C# weight
    argument was a simple constant (a literal, or a trivial expression with no external
    dependency); when the weight is a Func<float> that reads live game state (HP%, ascension,
    round number, ...), leave `weight` as None and put the C# source text in `weight_expr` instead
    -- the resolver then needs a hand-ported callable (see resolver.DYNAMIC_WEIGHTS) or falls back
    to `weight=1.0` (even odds) and flags it, since it can't safely eval arbitrary C# expressions.
    """

    to: str
    weight: float | None
    weight_expr: str | None = None
    repeat_type: str = "CanRepeatForever"  # CanRepeatForever | CanRepeatXTimes | CannotRepeat | UseOnlyOnce
    max_times: int = 0  # only meaningful for CanRepeatXTimes -- StateWeight.maxTimes
    cooldown: int = 0  # 0 = no cooldown gate


@dataclass
class ConditionalTarget:
    """One AddState(...) edge out of a ConditionalBranchState. Order matters -- the resolver
    (like ConditionalBranchState.GetNextState) takes the first edge whose condition evaluates
    true. `condition_expr` is the raw C# boolean lambda text; None means an unconditional
    "else" (AddState(move, null) or AddState(move, condition: null)) and must be the LAST edge.
    Like MoveTarget.weight_expr, a non-None condition needs a hand-ported callable in
    resolver.DYNAMIC_CONDITIONS to be evaluated for real; unresolved conditions default to "false"
    unless they're the trailing else, so the resolver conservatively favors later/default branches
    rather than silently assuming a specific game-state gate is met.
    """

    to: str
    condition_expr: str | None = None


@dataclass
class MonsterMoveGraph:
    """One monster's full move state machine, keyed by MonsterState.Id (== MoveState.StateId for
    move nodes -- the same string sts2-cli's buffer/JSON protocol now exposes as
    move_id/MoveIdHash, see sts2-cli's RunSimulator.Writers.Json.cs / States.Buffer.cs)."""

    monster_id: str  # ModelId entry / Tokenizer "monsters" category id, e.g. "AXE_RUBY_RAIDER"
    class_name: str  # decompiled C# class name, e.g. "AxeRubyRaider" -- source cross-reference only
    initial_state: str  # MonsterState.Id passed as MonsterMoveStateMachine's initialState
    move_ids: list[str] = field(default_factory=list)  # every MoveState.Id in this graph
    move_followup: dict[str, str] = field(default_factory=dict)  # MoveState.Id -> its one successor id
    random_branches: dict[str, list[MoveTarget]] = field(default_factory=dict)  # branch id -> edges
    conditional_branches: dict[str, list[ConditionalTarget]] = field(default_factory=dict)
    notes: str = ""  # extractor remarks: unusual constructs, anything hand-simplified, etc.
