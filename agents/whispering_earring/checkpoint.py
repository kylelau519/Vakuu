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

is_last_of_act/enemy_ids/enemy_max_hp are decided once at combat entry from the main line's live
context (context.room_type/context.enemies -- see facility_claws._resolve_combat) and carried on
every turn-branch of that combat, since a branch worker only ever sees a buffer-protocol decision
with no room_type of its own. enemy_ids/enemy_max_hp let a branch recognize the tracked entities via
decode_token despite that (see facility_common._tracked_enemy_hp); anything summoned mid-fight is
silently not counted. Tracked for every combat, not just boss fights -- is_last_of_act only
changes how the terminal/per-turn reward *weights* HP vs. enemy damage, not whether enemy damage
is tracked at all. A BranchJob is handed to a branch worker directly through the facility's job
queue (an mp.Queue -- plain Python objects, no disk round-trip needed).

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
    enemy_ids: frozenset[str] = frozenset()
    enemy_max_hp: int = 0
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


@dataclass
class MetaOptionJob:
    """One card_reward option's full evaluation chain (facility_choices_paradox.py), dispatched to
    the shared branch-worker pool instead of run in-process on the main worker. Unlike BranchJob
    (one combat turn), this is the WHOLE chain -- reconstructed once via load_and_resume_combat
    (or the full-replay fallback), then played unbranched to natural death entirely inside the
    handler, on its own Sts2BufferSession/CoreCLR in its own OS process. See
    facility_choices_paradox.py's module docstring for why running whole chains as separate
    processes doesn't reintroduce the per-turn subprocess cost or CLR thread-safety hazard that
    an earlier, different pool-based design was reverted for.

    prefix is the full action history since run start (same role as BranchJob.actions) --
    last_combat_checkpoint_path/last_combat_entry_index are the fast-path reconstruction inputs,
    an index into prefix. commit_name/commit_args is the option this chain commits to before
    playing onward; walk_seed is the common-random-numbers seed shared by every sibling option of
    this same divergence (see _run_meta_option_job's docstring)."""
    main_id: str
    character: str
    seed: str
    ascension: int
    decision_kind: str
    divergence_index: int
    option_index: int
    prefix: list[dict]
    last_room_checkpoint_path: str | None
    last_room_entry_index: int | None
    commit_name: str
    commit_args: dict
    walk_seed: int
    horizon_combats: int | None = None  # see FacilityConfig.meta_horizon_combats

    @property
    def meta_id(self) -> str:
        return f"{self.main_id}_m{self.divergence_index:03d}_o{self.option_index:02d}"


@dataclass
class MapOptionJob:
    """One select_map_node sibling's bounded evaluation walk (agents/golden_compass), dispatched
    to the shared branch-worker pool -- structurally identical to MetaOptionJob (reconstructed
    once via load_and_resume_combat/open_from_checkpoint, then played unbranched on its own
    Sts2BufferSession/CoreCLR in its own OS process), but scored differently: see
    facility_common._run_map_option_job's docstring for why this walks a fixed number of ROOMS
    (not combats) and scores by terminal HP fraction (not accumulated per-combat win bonuses) --
    the reward-design writeup in golden_compass/model/README.md's Design section covers why a flat
    per-win bonus is the wrong incentive for a decision that gets to choose whether to fight at
    all, unlike a card_reward pick.

    prefix/last_room_checkpoint_path/last_room_entry_index/commit_name/commit_args/walk_seed all
    mean exactly what they mean on MetaOptionJob -- see that dataclass's own docstring. room_horizon
    is this job's counterpart to MetaOptionJob.horizon_combats: how many further select_map_node
    decisions (rooms) this walk plays past its own commit before scoring, win or lose."""
    main_id: str
    character: str
    seed: str
    ascension: int
    divergence_index: int
    option_index: int
    prefix: list[dict]
    last_room_checkpoint_path: str | None
    last_room_entry_index: int | None
    commit_name: str
    commit_args: dict
    walk_seed: int
    room_horizon: int = 2

    @property
    def map_id(self) -> str:
        return f"{self.main_id}_g{self.divergence_index:03d}_o{self.option_index:02d}"
