"""Hand-extracted worked examples used to validate schema.py/resolver.py end to end before
delegating the rest of the 117-monster extraction. Both are fully deterministic (no
RandomBranchState/ConditionalBranchState at all) -- ShrinkerBeetle's cycle was cross-checked
against a live sts2-cli combat (move_id sequence observed: SHRINKER_MOVE, CHOMP_MOVE, STOMP_MOVE,
CHOMP_MOVE, STOMP_MOVE, CHOMP_MOVE -- matches move_followup below exactly).

Source: sts2-decompiled/src/Core/Models/Monsters/AxeRubyRaider.cs, ShrinkerBeetle.cs
(GenerateMoveStateMachine). Never copy decompiled source into this repo -- only short state-id
strings and weight values are captured here, no game logic or text.
"""

from __future__ import annotations

from .schema import MonsterMoveGraph, MoveTarget

GRAPHS: dict[str, MonsterMoveGraph] = {
    "AXE_RUBY_RAIDER": MonsterMoveGraph(
        monster_id="AXE_RUBY_RAIDER",
        class_name="AxeRubyRaider",
        initial_state="SWING_1",
        move_ids=["SWING_1", "SWING_2", "BIG_SWING"],
        move_followup={
            "SWING_1": "SWING_2",
            "SWING_2": "BIG_SWING",
            "BIG_SWING": "SWING_1",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "SHRINKER_BEETLE": MonsterMoveGraph(
        monster_id="SHRINKER_BEETLE",
        class_name="ShrinkerBeetle",
        initial_state="SHRINKER_MOVE",
        move_ids=["SHRINKER_MOVE", "CHOMP_MOVE", "STOMP_MOVE"],
        move_followup={
            "SHRINKER_MOVE": "CHOMP_MOVE",
            "CHOMP_MOVE": "STOMP_MOVE",
            "STOMP_MOVE": "CHOMP_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. Opens with SHRINKER_MOVE once, then "
            "CHOMP_MOVE/STOMP_MOVE alternate forever (STOMP_MOVE's followup is CHOMP_MOVE, not "
            "back to SHRINKER_MOVE)."
        ),
    ),
    "TWO_TAILED_RAT": MonsterMoveGraph(
        monster_id="TWO_TAILED_RAT",
        class_name="TwoTailedRat",
        # StarterMoveIndex == -1 (the default for a normally-spawned instance -- only test/debug
        # code sets it otherwise) starts the machine at the RAND branch itself, so even the first
        # move of a fresh combat is drawn from these four options, not fixed.
        initial_state="RAND",
        move_ids=["SCRATCH_MOVE", "DISEASE_BITE_MOVE", "SCREECH_MOVE", "CALL_FOR_BACKUP_MOVE"],
        move_followup={
            "SCRATCH_MOVE": "RAND",
            "DISEASE_BITE_MOVE": "RAND",
            "SCREECH_MOVE": "RAND",
            "CALL_FOR_BACKUP_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="SCRATCH_MOVE", weight=None,
                           weight_expr="(!CanSummon()) ? 1f : (1f/12f)", repeat_type="CannotRepeat"),
                MoveTarget(to="DISEASE_BITE_MOVE", weight=None,
                           weight_expr="(!CanSummon()) ? 1f : (1f/12f)", repeat_type="CannotRepeat"),
                MoveTarget(to="SCREECH_MOVE", weight=None,
                           weight_expr="(!CanSummon()) ? 1f : (1f/12f)", repeat_type="CannotRepeat",
                           cooldown=3),
                MoveTarget(to="CALL_FOR_BACKUP_MOVE", weight=None,
                           weight_expr="(!CanSummon()) ? 0f : 0.75f", repeat_type="UseOnlyOnce"),
            ],
        },
        notes=(
            "CanSummon() depends on cross-instance/combat-wide state -- a per-instance "
            "TurnsUntilSummonable counter (starts at 2, decremented by SCRATCH/DISEASE_BITE/"
            "SCREECH), a combat-wide CallForBackupCount>=3 cap SHARED across every TwoTailedRat "
            "instance via CombatState, whether an empty encounter slot exists to summon into, and "
            "whether any teammate's NextMove is already CALL_FOR_BACKUP_MOVE. None of that fits a "
            "flat per-decision context dict, so all four weights are deliberately left dynamic "
            "(weight=None, no DYNAMIC_WEIGHTS entry) -- resolve_next_move_distribution falls back "
            "to even odds among them and reports the edges via `unresolved`. A real fix would need "
            "the caller to pass combat-wide state (sibling rat count/positions, each sibling's own "
            "NextMove) through `context`, which the current schema doesn't model."
        ),
    ),
}
