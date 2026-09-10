"""Hand-extracted move state machines for the first alphabetical batch of Act 1 monsters.

Source: sts2-decompiled/src/Core/Models/Monsters/{Aeonglass,Architect,AssassinRubyRaider,Axebot,
BattleFriendV1,BattleFriendV2,BattleFriendV3,BigDummy,BowlbugEgg,BowlbugNectar,BowlbugRock,
BowlbugSilk,BruteRubyRaider,BygoneEffigy,Byrdonis,Byrdpip,CalcifiedCultist}.cs
(GenerateMoveStateMachine). Never copy decompiled source into this repo -- only short state-id
strings and weight values are captured here, no game logic or text.
"""

from __future__ import annotations

from .schema import ConditionalTarget, MonsterMoveGraph

GRAPHS: dict[str, MonsterMoveGraph] = {
    "AEONGLASS": MonsterMoveGraph(
        monster_id="AEONGLASS",
        class_name="Aeonglass",
        initial_state="EBB_MOVE",
        move_ids=["EBB_MOVE", "EYE_LASERS_MOVE", "INCREASING_INTENSITY_MOVE"],
        move_followup={
            "EBB_MOVE": "EYE_LASERS_MOVE",
            "EYE_LASERS_MOVE": "INCREASING_INTENSITY_MOVE",
            "INCREASING_INTENSITY_MOVE": "EBB_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "ARCHITECT": MonsterMoveGraph(
        monster_id="ARCHITECT",
        class_name="Architect",
        initial_state="NOTHING",
        move_ids=["NOTHING"],
        move_followup={"NOTHING": "NOTHING"},
        notes="9999 HP prop/decoy monster with a HiddenIntent no-op move that repeats forever.",
    ),
    "ASSASSIN_RUBY_RAIDER": MonsterMoveGraph(
        monster_id="ASSASSIN_RUBY_RAIDER",
        class_name="AssassinRubyRaider",
        initial_state="KILLSHOT_MOVE",
        move_ids=["KILLSHOT_MOVE"],
        move_followup={"KILLSHOT_MOVE": "KILLSHOT_MOVE"},
        notes="Single repeating attack move, no branch states.",
    ),
    "AXEBOT": MonsterMoveGraph(
        monster_id="AXEBOT",
        class_name="Axebot",
        # `_stockOverrideAmount` (backing field of StockAmount) is null on a normally-spawned
        # instance -- nothing in this file's constructor path calls the StockAmount setter -- so
        # `_stockOverrideAmount.HasValue` is false and the machine starts at HAMMER_UPPERCUT_MOVE,
        # not BOOT_UP_MOVE. BOOT_UP_MOVE is only ever entered as an initial state, never as a
        # FollowUpState target, so with a fresh spawn it is effectively unreachable.
        initial_state="HAMMER_UPPERCUT_MOVE",
        move_ids=["BOOT_UP_MOVE", "ONE_TWO_MOVE", "HAMMER_UPPERCUT_MOVE"],
        move_followup={
            "BOOT_UP_MOVE": "HAMMER_UPPERCUT_MOVE",
            "HAMMER_UPPERCUT_MOVE": "ONE_TWO_MOVE",
            "ONE_TWO_MOVE": "HAMMER_UPPERCUT_MOVE",
        },
        notes=(
            "Deterministic. If _stockOverrideAmount (StockAmount's backing field) has been "
            "explicitly set (e.g. by a respawn path elsewhere in the class), the machine instead "
            "starts at BOOT_UP_MOVE, which then falls into the same HAMMER_UPPERCUT_MOVE/"
            "ONE_TWO_MOVE cycle; a freshly-spawned instance never calls that setter, so "
            "BOOT_UP_MOVE is unreachable in the common case."
        ),
    ),
    "BATTLE_FRIEND_V1": MonsterMoveGraph(
        monster_id="BATTLE_FRIEND_V1",
        class_name="BattleFriendV1",
        initial_state="NOTHING_MOVE",
        move_ids=["NOTHING_MOVE"],
        move_followup={"NOTHING_MOVE": "NOTHING_MOVE"},
        notes="Training-dummy-style monster (75 HP skin variant) with a no-op move that repeats forever.",
    ),
    "BATTLE_FRIEND_V2": MonsterMoveGraph(
        monster_id="BATTLE_FRIEND_V2",
        class_name="BattleFriendV2",
        initial_state="NOTHING_MOVE",
        move_ids=["NOTHING_MOVE"],
        move_followup={"NOTHING_MOVE": "NOTHING_MOVE"},
        notes="Training-dummy-style monster (150 HP skin variant) with a no-op move that repeats forever.",
    ),
    "BATTLE_FRIEND_V3": MonsterMoveGraph(
        monster_id="BATTLE_FRIEND_V3",
        class_name="BattleFriendV3",
        initial_state="NOTHING_MOVE",
        move_ids=["NOTHING_MOVE"],
        move_followup={"NOTHING_MOVE": "NOTHING_MOVE"},
        notes="Training-dummy-style monster (300 HP skin variant) with a no-op move that repeats forever.",
    ),
    "BIG_DUMMY": MonsterMoveGraph(
        monster_id="BIG_DUMMY",
        class_name="BigDummy",
        initial_state="NOTHING",
        move_ids=["NOTHING"],
        move_followup={"NOTHING": "NOTHING"},
        notes=(
            "9999 HP prop/decoy monster with a HiddenIntent no-op move that repeats forever. Has "
            "a real localization entry (BIG_DUMMY.name) in this build's monsters.json despite the "
            "class name suggesting a test/template fixture, so treated as a real (if non-combat) "
            "shipped monster rather than flagged as unmatched."
        ),
    ),
    "BOWLBUG_EGG": MonsterMoveGraph(
        monster_id="BOWLBUG_EGG",
        class_name="BowlbugEgg",
        initial_state="BITE_MOVE",
        move_ids=["BITE_MOVE"],
        move_followup={"BITE_MOVE": "BITE_MOVE"},
        notes="Single repeating attack+block move, no branch states.",
    ),
    "BOWLBUG_NECTAR": MonsterMoveGraph(
        monster_id="BOWLBUG_NECTAR",
        class_name="BowlbugNectar",
        initial_state="THRASH_MOVE",
        move_ids=["THRASH_MOVE", "BUFF_MOVE", "THRASH2_MOVE"],
        move_followup={
            "THRASH_MOVE": "BUFF_MOVE",
            "BUFF_MOVE": "THRASH2_MOVE",
            "THRASH2_MOVE": "THRASH2_MOVE",
        },
        notes=(
            "Fully deterministic. Opens with THRASH_MOVE once, then BUFF_MOVE once, then "
            "THRASH2_MOVE (a second move state performing the identical attack) repeats forever."
        ),
    ),
    "BOWLBUG_ROCK": MonsterMoveGraph(
        monster_id="BOWLBUG_ROCK",
        class_name="BowlbugRock",
        initial_state="HEADBUTT_MOVE",
        move_ids=["HEADBUTT_MOVE", "DIZZY_MOVE"],
        move_followup={
            "HEADBUTT_MOVE": "POST_HEADBUTT",
            "DIZZY_MOVE": "HEADBUTT_MOVE",
        },
        conditional_branches={
            "POST_HEADBUTT": [
                ConditionalTarget(
                    to="DIZZY_MOVE",
                    condition_expr="creature is currently off-balance (IsOffBalance flag set)",
                ),
                ConditionalTarget(
                    to="HEADBUTT_MOVE",
                    condition_expr=None,
                ),
            ],
        },
        notes=(
            "POST_HEADBUTT's two C# conditions are complementary booleans (IsOffBalance / "
            "!IsOffBalance), i.e. exhaustive by construction -- once the first (IsOffBalance) "
            "edge fails to match, !IsOffBalance is guaranteed true, so the second edge is encoded "
            "as an unconditional trailing else (condition_expr=None) rather than as its own "
            "dynamic condition; this is semantically exact, not an approximation, and lets the "
            "graph resolve without a DYNAMIC_CONDITIONS entry. Separately, when IsOffBalance is "
            "true, HEADBUTT_MOVE's own perform method stuns the creature and forces its next move "
            "to DIZZY_MOVE directly (bypassing this branch state entirely) via an external "
            "stun-override mechanism this schema doesn't model; the POST_HEADBUTT conditional edge "
            "to DIZZY_MOVE is effectively a backup path for the same outcome. IsOffBalance is reset "
            "to false by DIZZY_MOVE itself."
        ),
    ),
    "BOWLBUG_SILK": MonsterMoveGraph(
        monster_id="BOWLBUG_SILK",
        class_name="BowlbugSilk",
        initial_state="TOXIC_SPIT_MOVE",
        move_ids=["THRASH_MOVE", "TOXIC_SPIT_MOVE"],
        move_followup={
            "THRASH_MOVE": "TOXIC_SPIT_MOVE",
            "TOXIC_SPIT_MOVE": "THRASH_MOVE",
        },
        notes="Fully deterministic 2-cycle; the machine's initial state is TOXIC_SPIT_MOVE, not THRASH_MOVE.",
    ),
    "BRUTE_RUBY_RAIDER": MonsterMoveGraph(
        monster_id="BRUTE_RUBY_RAIDER",
        class_name="BruteRubyRaider",
        initial_state="BEAT_MOVE",
        move_ids=["BEAT_MOVE", "ROAR_MOVE"],
        move_followup={
            "BEAT_MOVE": "ROAR_MOVE",
            "ROAR_MOVE": "BEAT_MOVE",
        },
        notes="Fully deterministic 2-cycle, no branch states.",
    ),
    "BYGONE_EFFIGY": MonsterMoveGraph(
        monster_id="BYGONE_EFFIGY",
        class_name="BygoneEffigy",
        initial_state="SLEEP_MOVE",
        move_ids=["SLEEP_MOVE", "WAKE_MOVE", "SLEEP_MOVE_2", "SLASHES_MOVE"],
        move_followup={
            "SLEEP_MOVE": "WAKE_MOVE",
            "WAKE_MOVE": "SLASHES_MOVE",
            "SLEEP_MOVE_2": "SLASHES_MOVE",
            "SLASHES_MOVE": "SLASHES_MOVE",
        },
        notes=(
            "SLEEP_MOVE (with a one-time intro speak line) plays once, then WAKE_MOVE grants a "
            "big one-time Strength buff and transitions into SLASHES_MOVE, which then repeats "
            "forever. SLEEP_MOVE_2 (a second, speak-line-free sleep move with the same intent) is "
            "declared and given a followup to SLASHES_MOVE, but nothing in GenerateMoveStateMachine "
            "ever points to it as a FollowUpState target or as the initial state -- it's presumably "
            "wired in from elsewhere in the class (not present in the excerpt read) as an "
            "alternate resume point, e.g. after an external interrupt. Left in move_ids/"
            "move_followup per the extraction rule that every MoveState must appear, but it is "
            "unreachable via any edge recorded here."
        ),
    ),
    "BYRDONIS": MonsterMoveGraph(
        monster_id="BYRDONIS",
        class_name="Byrdonis",
        initial_state="SWOOP_MOVE",
        move_ids=["PECK_MOVE", "SWOOP_MOVE"],
        move_followup={
            "PECK_MOVE": "SWOOP_MOVE",
            "SWOOP_MOVE": "PECK_MOVE",
        },
        notes="Fully deterministic 2-cycle; the machine's initial state is SWOOP_MOVE, not PECK_MOVE.",
    ),
    "BYRDPIP": MonsterMoveGraph(
        monster_id="BYRDPIP",
        class_name="Byrdpip",
        initial_state="NOTHING_MOVE",
        move_ids=["NOTHING_MOVE"],
        move_followup={"NOTHING_MOVE": "NOTHING_MOVE"},
        notes="9999 HP, hidden-health-bar pet/relic-bound decoy monster; no-op move repeats forever.",
    ),
    "CALCIFIED_CULTIST": MonsterMoveGraph(
        monster_id="CALCIFIED_CULTIST",
        class_name="CalcifiedCultist",
        initial_state="INCANTATION_MOVE",
        move_ids=["INCANTATION_MOVE", "DARK_STRIKE_MOVE"],
        move_followup={
            "INCANTATION_MOVE": "DARK_STRIKE_MOVE",
            "DARK_STRIKE_MOVE": "DARK_STRIKE_MOVE",
        },
        notes=(
            "INCANTATION_MOVE (buff) plays once as the opener, then DARK_STRIKE_MOVE (attack) "
            "repeats forever."
        ),
    ),
}
