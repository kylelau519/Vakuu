"""Checkpoint / branch-job format for turn-level self-play amplification.

A checkpoint is the replay prefix (character, seed, ascension, actions-so-far) needed to
reconstruct a state in a fresh sts2-cli subprocess -- same shape as sts2-cli's own save
format ({character, seed, actions}), plus the bookkeeping (main_id, floor, combat_index,
turn_index, branch_index) that gives every branch a collision-free id even across many
floors/turns. turn_index is the finest unit here -- branches diverge for a single turn at a
time, not a whole combat (see facility.py's _resolve_turn). combat_hp_before/combat_max_hp
are constant across every turn of a combat and let a turn-branch that happens to end the
combat score it with the same terminal formula facility.py uses everywhere else, even though
the branch itself only ever replayed from the turn's start, not the combat's. A BranchJob is
handed to a branch worker directly through facility.py's job queue (an mp.Queue -- plain
Python objects, no disk round-trip needed for a single-machine facility).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BranchJob:
    main_id: str
    floor: int
    combat_index: int
    turn_index: int
    branch_index: int
    character: str
    seed: str
    ascension: int
    combat_hp_before: int
    combat_max_hp: int
    actions: list[dict] = field(default_factory=list)
    policy: str = "explore"

    @property
    def branch_id(self) -> str:
        return (
            f"{self.main_id}_f{self.floor:02d}c{self.combat_index:02d}"
            f"t{self.turn_index:02d}_b{self.branch_index:02d}"
        )
