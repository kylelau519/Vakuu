"""Hand-extracted move state machines for a batch of Act 3 / late-game monsters (plus two
template/test classes that ship in the same source directory).

Source: sts2-decompiled/src/Core/Models/Monsters/{SoulFysh,SoulNexus,SpectralKnight,SpinyToad,
Stabbot,TenHpMonster,TerrorEel,TestSubject,TheAdversaryMkOne,TheAdversaryMkTwo,TheAdversaryMkThree,
TheForgotten,TheInsatiable,TheLost,TheObscura,ThievingHopper,Toadpole}.cs (GenerateMoveStateMachine
only, plus the private fields/consts it directly touches). Never copy decompiled source into this
repo -- only short state-id strings, weight/repeat values, and paraphrased condition descriptions
are captured here, no game logic, damage numbers, or flavor text.
"""

from __future__ import annotations

from .schema import ConditionalTarget, MonsterMoveGraph, MoveTarget

GRAPHS: dict[str, MonsterMoveGraph] = {
    "SOUL_FYSH": MonsterMoveGraph(
        monster_id="SOUL_FYSH",
        class_name="SoulFysh",
        initial_state="BECKON_MOVE",
        move_ids=["BECKON_MOVE", "DE_GAS_MOVE", "GAZE_MOVE", "FADE_MOVE", "SCREAM_MOVE"],
        move_followup={
            "BECKON_MOVE": "DE_GAS_MOVE",
            "DE_GAS_MOVE": "GAZE_MOVE",
            "GAZE_MOVE": "FADE_MOVE",
            "FADE_MOVE": "SCREAM_MOVE",
            "SCREAM_MOVE": "BECKON_MOVE",
        },
        notes="Fully deterministic 5-cycle, no branch states.",
    ),
    "SOUL_NEXUS": MonsterMoveGraph(
        monster_id="SOUL_NEXUS",
        class_name="SoulNexus",
        initial_state="SOUL_BURN_MOVE",
        move_ids=["SOUL_BURN_MOVE", "MAELSTROM_MOVE", "DRAIN_LIFE_MOVE"],
        move_followup={
            "SOUL_BURN_MOVE": "RAND",
            "MAELSTROM_MOVE": "RAND",
            "DRAIN_LIFE_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="SOUL_BURN_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="MAELSTROM_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="DRAIN_LIFE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Opens with a fixed SOUL_BURN_MOVE (constructor's initialState), then every move's "
            "FollowUpState points at the same RAND branch: even-odds pick among all three moves, "
            "each ineligible immediately after itself (CannotRepeat) so it can't play twice in a row."
        ),
    ),
    "SPECTRAL_KNIGHT": MonsterMoveGraph(
        monster_id="SPECTRAL_KNIGHT",
        class_name="SpectralKnight",
        initial_state="HEX",
        move_ids=["HEX", "SOUL_SLASH", "SOUL_FLAME"],
        move_followup={
            "HEX": "SOUL_SLASH",
            "SOUL_SLASH": "RAND",
            "SOUL_FLAME": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="SOUL_SLASH", weight=1.0, repeat_type="CanRepeatXTimes", max_times=2),
                MoveTarget(to="SOUL_FLAME", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Opens with fixed HEX then SOUL_SLASH, then RAND forever: even odds between SOUL_SLASH "
            "(may repeat up to 2 times in a row, i.e. AddBranch(state, 2) -> CanRepeatXTimes/"
            "max_times=2 per the plain-int-literal rule) and SOUL_FLAME (CannotRepeat -- never back "
            "to back)."
        ),
    ),
    "SPINY_TOAD": MonsterMoveGraph(
        monster_id="SPINY_TOAD",
        class_name="SpinyToad",
        initial_state="PROTRUDING_SPIKES_MOVE",
        move_ids=["PROTRUDING_SPIKES_MOVE", "SPIKE_EXPLOSION_MOVE", "TONGUE_LASH_MOVE"],
        move_followup={
            "PROTRUDING_SPIKES_MOVE": "SPIKE_EXPLOSION_MOVE",
            "SPIKE_EXPLOSION_MOVE": "TONGUE_LASH_MOVE",
            "TONGUE_LASH_MOVE": "PROTRUDING_SPIKES_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "STABBOT": MonsterMoveGraph(
        monster_id="STABBOT",
        class_name="Stabbot",
        initial_state="STAB_MOVE",
        move_ids=["STAB_MOVE"],
        move_followup={"STAB_MOVE": "STAB_MOVE"},
        notes="Single move, self-loop -- performs STAB_MOVE every turn forever.",
    ),
    "TEN_HP_MONSTER": MonsterMoveGraph(
        monster_id="TEN_HP_MONSTER",
        class_name="TenHpMonster",
        initial_state="NOTHING",
        move_ids=["NOTHING"],
        move_followup={"NOTHING": "NOTHING"},
        notes=(
            "no localization entry -- template/test class, not a real shipped monster. Its single "
            "move performs no action at all (a hidden-intent no-op), self-loops forever; used as a "
            "fixed-HP dummy target in tests/tools, not part of any real encounter."
        ),
    ),
    "TERROR_EEL": MonsterMoveGraph(
        monster_id="TERROR_EEL",
        class_name="TerrorEel",
        initial_state="CRASH_MOVE",
        move_ids=["CRASH_MOVE", "THRASH_MOVE", "STUN_MOVE", "TERROR_MOVE"],
        move_followup={
            "CRASH_MOVE": "THRASH_MOVE",
            "THRASH_MOVE": "CRASH_MOVE",
            "STUN_MOVE": "TERROR_MOVE",
            "TERROR_MOVE": "CRASH_MOVE",
        },
        notes=(
            "Normal play is a fixed 2-cycle: CRASH_MOVE <-> THRASH_MOVE forever from the initial "
            "state. STUN_MOVE (a no-op move with a StunIntent) and TERROR_MOVE are both declared "
            "and each has its own outgoing edge (STUN_MOVE -> TERROR_MOVE -> CRASH_MOVE), but "
            "nothing in GenerateMoveStateMachine ever transitions INTO STUN_MOVE by name -- it's "
            "presumably forced externally (e.g. by an interrupt/stagger effect elsewhere in the "
            "class not read for this extraction), similar in spirit to TestSubject's externally-"
            "triggered DeadState. Treat STUN_MOVE/TERROR_MOVE as reachable only outside the normal "
            "CRASH/THRASH cycle."
        ),
    ),
    "TEST_SUBJECT": MonsterMoveGraph(
        monster_id="TEST_SUBJECT",
        class_name="TestSubject",
        initial_state="BITE_MOVE",
        move_ids=[
            "RESPAWN_MOVE",
            "BITE_MOVE",
            "SKULL_BASH_MOVE",
            "MULTI_CLAW_MOVE",
            "PHASE3_LACERATE_MOVE",
            "BIG_POUNCE",
            "BURNING_GROWL_MOVE",
        ],
        move_followup={
            "RESPAWN_MOVE": "REVIVE_BRANCH",
            "BITE_MOVE": "SKULL_BASH_MOVE",
            "SKULL_BASH_MOVE": "BITE_MOVE",
            "MULTI_CLAW_MOVE": "MULTI_CLAW_MOVE",
            "PHASE3_LACERATE_MOVE": "BIG_POUNCE",
            "BIG_POUNCE": "BURNING_GROWL_MOVE",
            "BURNING_GROWL_MOVE": "PHASE3_LACERATE_MOVE",
        },
        conditional_branches={
            "REVIVE_BRANCH": [
                ConditionalTarget(
                    to="MULTI_CLAW_MOVE",
                    condition_expr="fewer than 2 prior respawns (this is the first or second life)",
                ),
                ConditionalTarget(to="PHASE3_LACERATE_MOVE"),
            ],
        },
        notes=(
            "Normal play is BITE_MOVE <-> SKULL_BASH_MOVE forever from the initial state. "
            "RESPAWN_MOVE (a MustPerformOnceBeforeTransitioning move) is never reached via "
            "move_followup/branch traversal here -- it's forced by an external TriggerDeadState() "
            "call (a SetMoveImmediate override) when the monster would otherwise die, i.e. it's an "
            "out-of-band entry point like TerrorEel's STUN_MOVE. Once RESPAWN_MOVE resolves, "
            "REVIVE_BRANCH sends it to MULTI_CLAW_MOVE (self-looping forever) for the first two "
            "lives, or to the PHASE3_LACERATE_MOVE/BIG_POUNCE/BURNING_GROWL_MOVE 3-cycle from the "
            "third life on. The second AddState condition (Respawns>=2) is the exact integer "
            "complement of the first (Respawns<2), so it's represented as the trailing unconditional "
            "edge per schema convention rather than a second dynamic condition."
        ),
    ),
    "THE_ADVERSARY_MK_ONE": MonsterMoveGraph(
        monster_id="THE_ADVERSARY_MK_ONE",
        class_name="TheAdversaryMkOne",
        initial_state="SMASH_MOVE",
        move_ids=["SMASH_MOVE", "BEAM_MOVE", "BARRAGE_MOVE"],
        move_followup={
            "SMASH_MOVE": "BEAM_MOVE",
            "BEAM_MOVE": "BARRAGE_MOVE",
            "BARRAGE_MOVE": "SMASH_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "THE_ADVERSARY_MK_TWO": MonsterMoveGraph(
        monster_id="THE_ADVERSARY_MK_TWO",
        class_name="TheAdversaryMkTwo",
        initial_state="BASH_MOVE",
        move_ids=["BASH_MOVE", "FLAME_BEAM_MOVE", "BARRAGE_MOVE"],
        move_followup={
            "BASH_MOVE": "FLAME_BEAM_MOVE",
            "FLAME_BEAM_MOVE": "BARRAGE_MOVE",
            "BARRAGE_MOVE": "BASH_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "THE_ADVERSARY_MK_THREE": MonsterMoveGraph(
        monster_id="THE_ADVERSARY_MK_THREE",
        class_name="TheAdversaryMkThree",
        initial_state="CRASH_MOVE",
        move_ids=["CRASH_MOVE", "FLAME_BEAM_MOVE", "BARRAGE_MOVE"],
        move_followup={
            "CRASH_MOVE": "FLAME_BEAM_MOVE",
            "FLAME_BEAM_MOVE": "BARRAGE_MOVE",
            "BARRAGE_MOVE": "CRASH_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "THE_FORGOTTEN": MonsterMoveGraph(
        monster_id="THE_FORGOTTEN",
        class_name="TheForgotten",
        initial_state="MIASMA",
        move_ids=["MIASMA", "DREAD"],
        move_followup={
            "MIASMA": "DREAD",
            "DREAD": "MIASMA",
        },
        notes="Fully deterministic 2-cycle, no branch states.",
    ),
    "THE_INSATIABLE": MonsterMoveGraph(
        monster_id="THE_INSATIABLE",
        class_name="TheInsatiable",
        initial_state="LIQUIFY_GROUND_MOVE",
        move_ids=["LIQUIFY_GROUND_MOVE", "THRASH_MOVE", "THRASH_MOVE_2", "LUNGING_BITE_MOVE", "SALIVATE_MOVE"],
        move_followup={
            "LIQUIFY_GROUND_MOVE": "THRASH_MOVE",
            "THRASH_MOVE": "LUNGING_BITE_MOVE",
            "LUNGING_BITE_MOVE": "SALIVATE_MOVE",
            "SALIVATE_MOVE": "THRASH_MOVE_2",
            "THRASH_MOVE_2": "THRASH_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. Opens once with LIQUIFY_GROUND_MOVE, then "
            "THRASH_MOVE/LUNGING_BITE_MOVE/SALIVATE_MOVE/THRASH_MOVE_2 cycle forever (THRASH_MOVE "
            "and THRASH_MOVE_2 are two distinct state ids that both perform the same underlying "
            "attack -- kept as separate nodes since the game does). A player-execute 'eat player' "
            "finisher exists as bestiary-only flavor (GenerateBestiaryMoveList insert) and is not "
            "part of GenerateMoveStateMachine, so it's excluded here."
        ),
    ),
    "THE_LOST": MonsterMoveGraph(
        monster_id="THE_LOST",
        class_name="TheLost",
        initial_state="DEBILITATING_SMOG",
        move_ids=["DEBILITATING_SMOG", "EYE_LASERS"],
        move_followup={
            "DEBILITATING_SMOG": "EYE_LASERS",
            "EYE_LASERS": "DEBILITATING_SMOG",
        },
        notes="Fully deterministic 2-cycle, no branch states.",
    ),
    "THE_OBSCURA": MonsterMoveGraph(
        monster_id="THE_OBSCURA",
        class_name="TheObscura",
        initial_state="ILLUSION_MOVE",
        move_ids=["ILLUSION_MOVE", "PIERCING_GAZE_MOVE", "SAIL_MOVE", "HARDENING_STRIKE_MOVE"],
        move_followup={
            "ILLUSION_MOVE": "RAND",
            "PIERCING_GAZE_MOVE": "RAND",
            "SAIL_MOVE": "RAND",
            "HARDENING_STRIKE_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="PIERCING_GAZE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="SAIL_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="HARDENING_STRIKE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Opens with fixed ILLUSION_MOVE (a summon, not itself a RAND target), then RAND forever: "
            "even odds among the other three moves, each CannotRepeat (no immediate repeats). "
            "'SAIL_MOVE' is the literal state-id string used in source for what the private method "
            "and comments call the 'Wail' move -- kept verbatim since it's the actual shipped id, "
            "not a transcription fix."
        ),
    ),
    "THIEVING_HOPPER": MonsterMoveGraph(
        monster_id="THIEVING_HOPPER",
        class_name="ThievingHopper",
        initial_state="THIEVERY_MOVE",
        move_ids=["THIEVERY_MOVE", "NAB_MOVE", "HAT_TRICK_MOVE", "FLUTTER_MOVE", "ESCAPE_MOVE"],
        move_followup={
            "THIEVERY_MOVE": "FLUTTER_MOVE",
            "FLUTTER_MOVE": "HAT_TRICK_MOVE",
            "HAT_TRICK_MOVE": "NAB_MOVE",
            "NAB_MOVE": "ESCAPE_MOVE",
            "ESCAPE_MOVE": "ESCAPE_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states: a fixed 4-move sequence "
            "(THIEVERY_MOVE -> FLUTTER_MOVE -> HAT_TRICK_MOVE -> NAB_MOVE) ending in ESCAPE_MOVE, "
            "which self-loops -- ESCAPE_MOVE removes the monster from combat, so in practice it's "
            "never seen a second time, but the graph edge exists in source."
        ),
    ),
    "TOADPOLE": MonsterMoveGraph(
        monster_id="TOADPOLE",
        class_name="Toadpole",
        initial_state="INIT_MOVE",
        move_ids=["SPIKE_SPIT_MOVE", "WHIRL_MOVE", "SPIKEN_MOVE"],
        move_followup={
            "WHIRL_MOVE": "SPIKEN_MOVE",
            "SPIKEN_MOVE": "SPIKE_SPIT_MOVE",
            "SPIKE_SPIT_MOVE": "WHIRL_MOVE",
        },
        conditional_branches={
            "INIT_MOVE": [
                ConditionalTarget(
                    to="WHIRL_MOVE",
                    condition_expr="this Toadpole's IsFront spawn flag is false (the default for a standalone instance)",
                ),
                ConditionalTarget(to="SPIKEN_MOVE"),
            ],
        },
        notes=(
            "The constructor's initialState is the INIT_MOVE conditional branch itself, not a move "
            "-- it picks the opening move by IsFront, a per-instance bool set once at encounter "
            "composition time (not HP/turn/ascension state): ToadpolesWeak spawns a linked pair with "
            "one IsFront=true and one IsFront=false, while IsFront defaults to false (private field "
            "default) for a standalone/freshly-spawned instance, so a generic instance opens at "
            "WHIRL_MOVE. The second AddState condition (IsFront==true) is the exact boolean "
            "complement of the first, so it's represented as the trailing unconditional edge per "
            "schema convention. After the opening pick, WHIRL_MOVE -> SPIKEN_MOVE -> SPIKE_SPIT_MOVE "
            "cycles forever regardless of which branch was taken."
        ),
    ),
}
