"""Checkpoint / branch-job format for turn-level self-play amplification.

A checkpoint is the replay prefix (character, seed, ascension, actions-so-far) needed to
reconstruct a state in a fresh sts2-cli subprocess -- same shape as sts2-cli's own save
format ({character, seed, actions}), plus the bookkeeping (main_id, act, floor, combat_index,
turn_index, branch_index) that gives every branch a collision-free id even across many
floors/turns. combat_index alone (monotonic for the whole trial, never reset) already makes
branch_id collision-free on its own -- act is included in the id purely for human readability,
since floor resets to a low number at the start of every act (STS2's own per-act floor
numbering), so "floor 17" by itself doesn't say which act it's in. turn_index is the finest unit here -- branches diverge for a single turn at a
time, not a whole combat (see facility.py's _resolve_turn). combat_hp_before/combat_max_hp
are constant across every turn of a combat and let a turn-branch that happens to end the
combat score it with the same terminal formula facility.py uses everywhere else, even though
the branch itself only ever replayed from the turn's start, not the combat's. is_last_of_act is
likewise decided once, from the main line's live context.room_type at combat entry (see
facility.py's _resolve_combat), and carried on every turn-branch of that combat so a branch
worker -- which only ever sees a buffer-protocol decision with no room_type of its own --
still scores turns with the boss-specific reward shaping. boss_ids/boss_max_hp are decided at
the same point and for the same reason: the set of enemy entity ids present when the boss combat
was entered (context.enemies, main line, JSON side), and the combined max_hp of just those
entities -- fixed for the whole fight so a branch worker can recognize them via decode_token
despite only ever seeing buffer-protocol enemy tokens, and so anything summoned mid-fight is
silently ignored rather than counted as boss progress (see facility.py's _tracked_enemy_hp).
Empty/0 whenever is_last_of_act is False. A BranchJob is
handed to a branch worker directly through facility.py's job queue (an mp.Queue -- plain
Python objects, no disk round-trip needed for a single-machine facility).

`actions` is still the full action history since run start -- kept so a branch worker's
recorder metadata (`start_step=len(job.actions)`) and output JSON stay identical whichever
reconstruction path is used. `checkpoint_path`/`combat_start_index` are the fast path: a
native sts2-cli save taken by the main line right before the select_map_node action that
entered this combat (see facility.py's main_worker_loop), plus that action's index within
`actions`. When set, branch_worker_loop reconstructs state via
Sts2BufferSession.load_and_resume_combat(checkpoint_path, actions[combat_start_index],
actions[combat_start_index + 1:]) -- replaying only this combat's own actions -- instead of
open_from_checkpoint(character, seed, ascension, actions), which replays the entire run from
StartRun. checkpoint_path is None when no checkpoint was available (e.g. the very first combat
of a run, before any select_map_node has been taken), falling back to the full replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetaBranchJob:
    """Job format for card_reward (and eventually card_select) meta-decision branching -- sibling
    to BranchJob above, not an extension of it: BranchJob's combat-scoring fields
    (is_last_of_act/boss_ids/combat_hp_before/...) are meaningless here, since a meta-branch's
    divergence point is a card_reward decision between combats, not a combat turn. See
    agents/choices_paradox/model/README.md's Data section for the full design and
    facility_choices_paradox.py for what builds/consumes these.

    actions is the replay prefix only -- NOT including the committed option's own action.
    facility_choices_paradox._play_meta_forward re-derives this decision's legal options from the
    replayed state and applies options[option_index] itself, the same way BranchJob's
    checkpoint_path fast path re-derives state rather than being handed a pre-committed one.
    There is no native-checkpoint fast path here (unlike BranchJob's checkpoint_path/
    combat_start_index) -- every replicate reconstructs via a full open_from_checkpoint replay;
    see the README's Data section for why (full-replay latency growing with run depth could
    matter once meta-branches multiply that cost by num_options x M, but that's flagged for
    empirical measurement before a v2 fast path is added, not assumed necessary yet)."""
    main_id: str
    decision_kind: str          # "card_reward" | "card_select_remove" | "card_select_upgrade"
    divergence_index: int       # monotonic per-trial counter, mirrors combat_index
    option_index: int           # which legal_options(...) entry this branch commits to
    replicate_index: int        # which of the M replicates of that option this is
    character: str
    seed: str
    ascension: int
    actions: list[dict] = field(default_factory=list)   # full replay prefix
    horizon_combats: int = 1

    @property
    def branch_id(self) -> str:
        return f"{self.main_id}_m{self.divergence_index:03d}_o{self.option_index:02d}_r{self.replicate_index:02d}"


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
