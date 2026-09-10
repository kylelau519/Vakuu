"""Hand-extracted move state machines for a batch of Act 2 (and a few cross-act/companion)
monsters. See schema.py for the dataclasses and data_00_worked_examples.py for the extraction
conventions this follows.

Source: sts2-decompiled/src/Core/Models/Monsters/{LeafSlimeM,LeafSlimeS,LivingFog,LivingShield,
LouseProgenitor,MagiKnight,Mawler,MechaKnight,MultiAttackMoveMonster,Myte,Nibbit,Noisebot,
OneHpMonster,Osty,Ovicopter,OwlMagistrate,PaelsLegion}.cs (GenerateMoveStateMachine). Never copy
decompiled source into this repo -- only short state-id strings, weights, and paraphrased
conditions are captured here, no game logic or text.
"""

from __future__ import annotations

from .schema import ConditionalTarget, MonsterMoveGraph, MoveTarget

GRAPHS: dict[str, MonsterMoveGraph] = {
    "LEAF_SLIME_M": MonsterMoveGraph(
        monster_id="LEAF_SLIME_M",
        class_name="LeafSlimeM",
        initial_state="STICKY_SHOT",
        move_ids=["CLUMP_SHOT", "STICKY_SHOT"],
        move_followup={
            "CLUMP_SHOT": "STICKY_SHOT",
            "STICKY_SHOT": "CLUMP_SHOT",
        },
        notes="Fully deterministic 2-cycle, no branch states. Starts at STICKY_SHOT.",
    ),
    "LEAF_SLIME_S": MonsterMoveGraph(
        monster_id="LEAF_SLIME_S",
        class_name="LeafSlimeS",
        initial_state="RAND",
        move_ids=["TACKLE_MOVE", "GOOP_MOVE"],
        move_followup={
            "TACKLE_MOVE": "RAND",
            "GOOP_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="TACKLE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="GOOP_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Starts at the RAND branch itself, so even the opening move is a 50/50 draw between "
            "TACKLE_MOVE and GOOP_MOVE. Both edges are CannotRepeat, so in practice the two moves "
            "strictly alternate once combat is underway -- the randomness only matters for which "
            "one opens."
        ),
    ),
    "LIVING_FOG": MonsterMoveGraph(
        monster_id="LIVING_FOG",
        class_name="LivingFog",
        initial_state="ADVANCED_GAS_MOVE",
        move_ids=["ADVANCED_GAS_MOVE", "BLOAT_MOVE", "SUPER_GAS_BLAST_MOVE"],
        move_followup={
            "ADVANCED_GAS_MOVE": "BLOAT_MOVE",
            "BLOAT_MOVE": "SUPER_GAS_BLAST_MOVE",
            "SUPER_GAS_BLAST_MOVE": "BLOAT_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. Opens with ADVANCED_GAS_MOVE once, then "
            "BLOAT_MOVE/SUPER_GAS_BLAST_MOVE alternate forever (SUPER_GAS_BLAST_MOVE's followup "
            "is BLOAT_MOVE, not back to ADVANCED_GAS_MOVE) -- same shape as SHRINKER_BEETLE."
        ),
    ),
    "LIVING_SHIELD": MonsterMoveGraph(
        monster_id="LIVING_SHIELD",
        class_name="LivingShield",
        initial_state="SHIELD_SLAM_MOVE",
        move_ids=["SHIELD_SLAM_MOVE", "SMASH_MOVE"],
        move_followup={
            "SHIELD_SLAM_MOVE": "SHIELD_SLAM_BRANCH",
            "SMASH_MOVE": "SMASH_MOVE",
        },
        conditional_branches={
            "SHIELD_SLAM_BRANCH": [
                ConditionalTarget(to="SHIELD_SLAM_MOVE", condition_expr="count of other living allied creatures is > 0"),
                ConditionalTarget(to="SMASH_MOVE"),
            ],
        },
        notes=(
            "Repeats SHIELD_SLAM_MOVE every turn while it still has a living ally; once its last "
            "ally dies it permanently switches to SMASH_MOVE (SMASH_MOVE's own followup is itself, "
            "a self-loop that bypasses the conditional branch entirely -- this transition is "
            "one-way, it never goes back to SHIELD_SLAM_MOVE even if new allies appear). Source "
            "writes the second edge as an explicit '== 0' condition rather than a null lambda, but "
            "it's the exact logical complement of '> 0' over a non-negative count, so it's modeled "
            "here as the trailing unconditional else -- this lets the resolver actually terminate "
            "without a DYNAMIC_CONDITIONS entry, and is behaviorally identical."
        ),
    ),
    "LOUSE_PROGENITOR": MonsterMoveGraph(
        monster_id="LOUSE_PROGENITOR",
        class_name="LouseProgenitor",
        initial_state="WEB_CANNON_MOVE",
        move_ids=["WEB_CANNON_MOVE", "POUNCE_MOVE", "CURL_AND_GROW_MOVE"],
        move_followup={
            "WEB_CANNON_MOVE": "CURL_AND_GROW_MOVE",
            "CURL_AND_GROW_MOVE": "POUNCE_MOVE",
            "POUNCE_MOVE": "WEB_CANNON_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "MAGI_KNIGHT": MonsterMoveGraph(
        monster_id="MAGI_KNIGHT",
        class_name="MagiKnight",
        initial_state="POWER_SHIELD_MOVE",
        move_ids=["POWER_SHIELD_MOVE", "DAMPEN_MOVE", "PREP_MOVE", "MAGIC_BOMB", "RAM_MOVE"],
        move_followup={
            "POWER_SHIELD_MOVE": "DAMPEN_MOVE",
            "DAMPEN_MOVE": "RAM_MOVE",
            "RAM_MOVE": "PREP_MOVE",
            "PREP_MOVE": "MAGIC_BOMB",
            "MAGIC_BOMB": "RAM_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. Opens with POWER_SHIELD_MOVE then DAMPEN_MOVE "
            "once each, then RAM_MOVE/PREP_MOVE/MAGIC_BOMB cycle forever."
        ),
    ),
    "MAWLER": MonsterMoveGraph(
        monster_id="MAWLER",
        class_name="Mawler",
        initial_state="CLAW_MOVE",
        move_ids=["RIP_AND_TEAR_MOVE", "ROAR_MOVE", "CLAW_MOVE"],
        move_followup={
            "RIP_AND_TEAR_MOVE": "RAND",
            "ROAR_MOVE": "RAND",
            "CLAW_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="RIP_AND_TEAR_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="ROAR_MOVE", weight=1.0, repeat_type="UseOnlyOnce"),
                MoveTarget(to="CLAW_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Opens with a fixed CLAW_MOVE (the state machine's initial state), then every "
            "following turn is an even 3-way random draw among RIP_AND_TEAR_MOVE/ROAR_MOVE/"
            "CLAW_MOVE, except RIP_AND_TEAR_MOVE and CLAW_MOVE can never repeat back-to-back and "
            "ROAR_MOVE can only ever be used once per combat."
        ),
    ),
    "MECHA_KNIGHT": MonsterMoveGraph(
        monster_id="MECHA_KNIGHT",
        class_name="MechaKnight",
        initial_state="CHARGE_MOVE",
        move_ids=["CHARGE_MOVE", "FLAMETHROWER_MOVE", "WINDUP_MOVE", "HEAVY_CLEAVE_MOVE"],
        move_followup={
            "CHARGE_MOVE": "FLAMETHROWER_MOVE",
            "FLAMETHROWER_MOVE": "WINDUP_MOVE",
            "WINDUP_MOVE": "HEAVY_CLEAVE_MOVE",
            "HEAVY_CLEAVE_MOVE": "FLAMETHROWER_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. Opens with CHARGE_MOVE once, then "
            "FLAMETHROWER_MOVE/WINDUP_MOVE/HEAVY_CLEAVE_MOVE cycle forever."
        ),
    ),
    "MULTI_ATTACK_MOVE_MONSTER": MonsterMoveGraph(
        monster_id="MULTI_ATTACK_MOVE_MONSTER",
        class_name="MultiAttackMoveMonster",
        initial_state="POKE",
        move_ids=["POKE"],
        move_followup={"POKE": "POKE"},
        notes=(
            "Single move (5-hit poke) that repeats forever, no branch states. No localization "
            "entry for MULTI_ATTACK_MOVE_MONSTER -- template/test class, not a real shipped "
            "monster (its Title even borrows BIG_DUMMY's localization string)."
        ),
    ),
    "MYTE": MonsterMoveGraph(
        monster_id="MYTE",
        class_name="Myte",
        initial_state="INIT_MOVE",
        move_ids=["TOXIC_MOVE", "BITE_MOVE", "SUCK_MOVE"],
        move_followup={
            "TOXIC_MOVE": "BITE_MOVE",
            "BITE_MOVE": "SUCK_MOVE",
            "SUCK_MOVE": "TOXIC_MOVE",
        },
        conditional_branches={
            "INIT_MOVE": [
                ConditionalTarget(to="TOXIC_MOVE", condition_expr="this creature occupies the 'first' encounter slot"),
                ConditionalTarget(to="SUCK_MOVE"),
            ],
        },
        notes=(
            "Myte always spawns in pairs sharing one 3-move cycle (TOXIC_MOVE -> BITE_MOVE -> "
            "SUCK_MOVE -> TOXIC_MOVE), just entered at different points depending on slot: the "
            "'first' Myte opens on TOXIC_MOVE, the 'second' opens on SUCK_MOVE (which, per the "
            "cycle, puts it one step ahead of BITE_MOVE relative to the first). Source writes the "
            "second edge as an explicit SlotName=='second' check rather than a null lambda, but "
            "Myte only ever spawns with slot names 'first'/'second', so it's modeled here as the "
            "trailing unconditional else (matches real behavior, and lets the resolver terminate "
            "without a DYNAMIC_CONDITIONS entry) -- if the slot name were ever anything else the "
            "real C# would in fact fail to find a next state, which this simplification masks."
        ),
    ),
    "NIBBIT": MonsterMoveGraph(
        monster_id="NIBBIT",
        class_name="Nibbit",
        initial_state="INIT_MOVE",
        move_ids=["BUTT_MOVE", "SLICE_MOVE", "HISS_MOVE"],
        move_followup={
            "BUTT_MOVE": "SLICE_MOVE",
            "SLICE_MOVE": "HISS_MOVE",
            "HISS_MOVE": "BUTT_MOVE",
        },
        conditional_branches={
            "INIT_MOVE": [
                ConditionalTarget(to="HISS_MOVE", condition_expr="this Nibbit is not in the front slot (IsFront false)"),
                ConditionalTarget(to="SLICE_MOVE"),
            ],
        },
        notes=(
            "IsAlone (default false, set externally for single-Nibbit encounters) changes which "
            "edges are even built, not just which condition is checked -- this is decided once "
            "when GenerateMoveStateMachine() runs (a plain C# if/else over the field), not a "
            "runtime lambda. This entry models the default IsAlone=false case, which is gated on "
            "IsFront as shown; source writes the second edge as an explicit IsFront==true check "
            "rather than a null lambda, but it's the exact logical complement of IsFront==false "
            "over a bool, so it's modeled here as the trailing unconditional else (lets the "
            "resolver terminate without a DYNAMIC_CONDITIONS entry, behaviorally identical). When "
            "IsAlone is true instead, INIT_MOVE has a single unconditional edge straight to "
            "BUTT_MOVE regardless of IsFront. All three moves (BUTT_MOVE, SLICE_MOVE, HISS_MOVE) "
            "form the same 3-cycle afterward in both cases -- only the entry point differs."
        ),
    ),
    "NOISEBOT": MonsterMoveGraph(
        monster_id="NOISEBOT",
        class_name="Noisebot",
        initial_state="NOISE_MOVE",
        move_ids=["NOISE_MOVE"],
        move_followup={"NOISE_MOVE": "NOISE_MOVE"},
        notes="Single move that repeats forever, no branch states.",
    ),
    "ONE_HP_MONSTER": MonsterMoveGraph(
        monster_id="ONE_HP_MONSTER",
        class_name="OneHpMonster",
        initial_state="NOTHING",
        move_ids=["NOTHING"],
        move_followup={"NOTHING": "NOTHING"},
        notes=(
            "Single hidden-intent no-op move that repeats forever, no branch states. No "
            "localization entry for ONE_HP_MONSTER -- template/test class, not a real shipped "
            "monster (its Title even borrows BIG_DUMMY's localization string)."
        ),
    ),
    "OSTY": MonsterMoveGraph(
        monster_id="OSTY",
        class_name="Osty",
        initial_state="NOTHING_MOVE",
        move_ids=["NOTHING_MOVE"],
        move_followup={"NOTHING_MOVE": "NOTHING_MOVE"},
        notes=(
            "Single no-op move that repeats forever, no branch states. Osty is a real companion "
            "creature (Necrobinder mechanic), not a test class -- has a real OSTY localization "
            "entry, unlike the actual test/template monsters in this batch."
        ),
    ),
    "OVICOPTER": MonsterMoveGraph(
        monster_id="OVICOPTER",
        class_name="Ovicopter",
        initial_state="LAY_EGGS_MOVE",
        move_ids=["LAY_EGGS_MOVE", "SMASH_MOVE", "TENDERIZER_MOVE", "NUTRITIONAL_PASTE_MOVE"],
        move_followup={
            "LAY_EGGS_MOVE": "SMASH_MOVE",
            "NUTRITIONAL_PASTE_MOVE": "SMASH_MOVE",
            "SMASH_MOVE": "TENDERIZER_MOVE",
            "TENDERIZER_MOVE": "SUMMON_BRANCH_STATE",
        },
        conditional_branches={
            "SUMMON_BRANCH_STATE": [
                ConditionalTarget(to="LAY_EGGS_MOVE", condition_expr="living-teammates count (GetTeammatesOf, which appears to include Ovicopter itself) is <= 3"),
                ConditionalTarget(to="NUTRITIONAL_PASTE_MOVE"),
            ],
        },
        notes=(
            "Opens with LAY_EGGS_MOVE (the state machine's fixed initial state) always, then "
            "SMASH_MOVE, then TENDERIZER_MOVE, then branches: LAY_EGGS_MOVE again if there's still "
            "room for another summoned egg minion, otherwise a Strength-buff (NUTRITIONAL_PASTE_"
            "MOVE) instead -- both loop back through SMASH_MOVE -> TENDERIZER_MOVE -> the same "
            "branch every cycle. Unlike LivingShield's GetAllyCount, CanLay's count is not "
            "explicitly filtered to exclude self, so it likely includes Ovicopter itself. Source "
            "writes the second edge as an explicit '> 3' check rather than a null lambda, but it's "
            "the exact logical complement of '<= 3' over a count, so it's modeled here as the "
            "trailing unconditional else (lets the resolver terminate without a DYNAMIC_CONDITIONS "
            "entry, behaviorally identical)."
        ),
    ),
    "OWL_MAGISTRATE": MonsterMoveGraph(
        monster_id="OWL_MAGISTRATE",
        class_name="OwlMagistrate",
        initial_state="MAGISTRATE_SCRUTINY",
        move_ids=["MAGISTRATE_SCRUTINY", "PECK_ASSAULT", "JUDICIAL_FLIGHT", "VERDICT"],
        move_followup={
            "MAGISTRATE_SCRUTINY": "PECK_ASSAULT",
            "PECK_ASSAULT": "JUDICIAL_FLIGHT",
            "JUDICIAL_FLIGHT": "VERDICT",
            "VERDICT": "MAGISTRATE_SCRUTINY",
        },
        notes="Fully deterministic 4-cycle, no branch states.",
    ),
    "PAELS_LEGION": MonsterMoveGraph(
        monster_id="PAELS_LEGION",
        class_name="PaelsLegion",
        initial_state="NOTHING_MOVE",
        move_ids=["NOTHING_MOVE"],
        move_followup={"NOTHING_MOVE": "NOTHING_MOVE"},
        notes=(
            "Single no-op move that repeats forever, no branch states. Fixed 9999 HP, invisible "
            "health bar companion tied to the Pael's Legion relic -- a real shipped entity (has a "
            "real PAELS_LEGION localization entry), not a test class."
        ),
    ),
}
