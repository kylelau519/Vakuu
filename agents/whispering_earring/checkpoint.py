"""Checkpoint / branch-job format for turn-level self-play amplification.

A checkpoint is the replay prefix (character, seed, ascension, actions-so-far) needed to
reconstruct a state in a fresh sts2-cli subprocess -- same shape as sts2-cli's own save format
({character, seed, actions}), plus id bookkeeping (main_id, act, floor, combat_index, turn_index,
branch_index). combat_index alone (monotonic for the whole trial, never reset) already makes
branch_id collision-free; act/floor are included only for human readability, since floor resets
per act. This relies on main_id naming exactly one combat_index sequence --
facility_choices_paradox.py's option-chains each keep their own locally-reset combat_index, so
that module qualifies main_id per chain (branch_main_id =
f"{main_id}_m{divergence_index:03d}_o{option_index:02d}") before calling into
facility_claws._resolve_combat, or sibling chains running concurrently would mint identical
branch_ids and clobber each other's output file.

is_last_of_act/boss_ids/boss_max_hp are decided once at combat entry from the main line's live
context (context.room_type/context.enemies -- see facility_claws._resolve_combat) and carried on
every turn-branch of that combat, since a branch worker only ever sees a buffer-protocol decision
with no room_type of its own. boss_ids/boss_max_hp let a branch recognize the tracked entities via
decode_token despite that (see facility_common._tracked_enemy_hp); anything summoned mid-fight is
silently not counted. Empty/0 whenever is_last_of_act is False. A BranchJob is handed to a branch
worker directly through the facility's job queue (an mp.Queue -- plain Python objects, no disk
round-trip needed).

`actions` is the full action history since run start, kept so a branch worker's recorder metadata
and output JSON stay identical whichever reconstruction path is used. `checkpoint_path`/
`combat_start_index` are the fast path: a native sts2-cli save taken right before the
select_map_node action that entered this combat (see main_worker_loop), plus that action's index
within `actions`. When set, branch_worker_loop reconstructs state via
Sts2BufferSession.load_and_resume_combat(checkpoint_path, actions[combat_start_index],
actions[combat_start_index + 1:]) -- replaying only this combat -- instead of
open_from_checkpoint(...), which replays the entire run from StartRun. checkpoint_path is None
when no checkpoint was available (e.g. the run's very first combat), falling back to full replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BranchJob:
    main_id: str
    act: int
    floor: int
    combat_index: int
    turn_index: int
    branch_index: int
    character: str
    seed: str
    ascension: int
    combat_hp_before: int
    combat_max_hp: int
    is_last_of_act: bool = False
    boss_ids: frozenset[str] = frozenset()
    boss_max_hp: int = 0
    actions: list[dict] = field(default_factory=list)
    checkpoint_path: str | None = None
    combat_start_index: int = 0
    policy: str = "explore"

    @property
    def branch_id(self) -> str:
        return (
            f"{self.main_id}_a{self.act:02d}f{self.floor:02d}c{self.combat_index:02d}"
            f"t{self.turn_index:02d}_b{self.branch_index:02d}"
        )
