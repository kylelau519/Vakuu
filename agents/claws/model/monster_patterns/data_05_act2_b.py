"""Hand-extracted move state machines for a batch of Act 2 monsters (alphabetically
Parafright..SneakyGremlin).

Source: sts2-decompiled/src/Core/Models/Monsters/{Parafright,PhantasmalGardener,PhrogParasite,
PunchConstruct,Queen,Rocket,ScrollOfBiting,Seapunk,SewerClam,SingleAttackMoveMonster,
SkulkingColony,SlimedBerserker,SlitheringStrangler,SludgeSpinner,SlumberingBeetle,
SnappingJaxfruit,SneakyGremlin}.cs (GenerateMoveStateMachine). Never copy decompiled source into
this repo -- only short state-id strings and weight values are captured here, no game logic or
text.
"""

from __future__ import annotations

from .schema import ConditionalTarget, MonsterMoveGraph, MoveTarget

GRAPHS: dict[str, MonsterMoveGraph] = {
    "PARAFRIGHT": MonsterMoveGraph(
        monster_id="PARAFRIGHT",
        class_name="Parafright",
        initial_state="SLAM_MOVE",
        move_ids=["SLAM_MOVE"],
        move_followup={"SLAM_MOVE": "SLAM_MOVE"},
        notes="Single repeating attack move, no branch states.",
    ),
    "PHANTASMAL_GARDENER": MonsterMoveGraph(
        monster_id="PHANTASMAL_GARDENER",
        class_name="PhantasmalGardener",
        initial_state="INIT_MOVE",
        move_ids=["BITE_MOVE", "LASH_MOVE", "FLAIL_MOVE", "ENLARGE_MOVE"],
        move_followup={
            "BITE_MOVE": "LASH_MOVE",
            "LASH_MOVE": "FLAIL_MOVE",
            "FLAIL_MOVE": "ENLARGE_MOVE",
            "ENLARGE_MOVE": "BITE_MOVE",
        },
        conditional_branches={
            "INIT_MOVE": [
                ConditionalTarget(to="FLAIL_MOVE", condition_expr="this creature's SlotName is 'first'"),
                ConditionalTarget(to="BITE_MOVE", condition_expr="this creature's SlotName is 'second'"),
                ConditionalTarget(to="LASH_MOVE", condition_expr="this creature's SlotName is 'third'"),
                ConditionalTarget(to="ENLARGE_MOVE", condition_expr=None),
            ],
        },
        notes=(
            "INIT_MOVE is a one-time entry branch selecting the opening move by encounter slot "
            "position (first/second/third/fourth), then the four moves cycle BITE->LASH->FLAIL->"
            "ENLARGE->BITE forever regardless of which one opened. The four SlotName checks are "
            "mutually exclusive and exhaustive (only four slot names exist), so the fourth "
            "('fourth') is encoded as an unconditional trailing else (condition_expr=None) rather "
            "than its own dynamic condition -- semantically exact, not an approximation."
        ),
    ),
    "PHROG_PARASITE": MonsterMoveGraph(
        monster_id="PHROG_PARASITE",
        class_name="PhrogParasite",
        initial_state="INFECT_MOVE",
        move_ids=["INFECT_MOVE", "LASH_MOVE"],
        move_followup={
            "INFECT_MOVE": "LASH_MOVE",
            "LASH_MOVE": "INFECT_MOVE",
        },
        notes=(
            "Fully deterministic 2-cycle. GenerateMoveStateMachine also builds a RandomBranchState "
            "'RAND' with CannotRepeat AddBranch edges to both moves, added to the state list, but "
            "nothing ever sets it as a FollowUpState or as the initial state -- it's dead/"
            "unreachable code, so it's omitted from this graph's random_branches."
        ),
    ),
    "PUNCH_CONSTRUCT": MonsterMoveGraph(
        monster_id="PUNCH_CONSTRUCT",
        class_name="PunchConstruct",
        initial_state="READY_MOVE",
        move_ids=["READY_MOVE", "FAST_PUNCH_MOVE", "STRONG_PUNCH_MOVE"],
        move_followup={
            "READY_MOVE": "FAST_PUNCH_MOVE",
            "FAST_PUNCH_MOVE": "STRONG_PUNCH_MOVE",
            "STRONG_PUNCH_MOVE": "READY_MOVE",
        },
        notes=(
            "Fully deterministic 3-cycle, no branch states. StartsWithFastPunch (false by default "
            "on a normally-spawned instance) can move the initial state to FAST_PUNCH_MOVE instead "
            "of READY_MOVE if set true by an external spawner; the cycle itself is unchanged "
            "either way."
        ),
    ),
    "QUEEN": MonsterMoveGraph(
        monster_id="QUEEN",
        class_name="Queen",
        initial_state="PUPPET_STRINGS_MOVE",
        move_ids=[
            "PUPPET_STRINGS_MOVE",
            "YOU_ARE_MINE_MOVE",
            "BURN_BRIGHT_FOR_ME_MOVE",
            "OFF_WITH_YOUR_HEAD_MOVE",
            "EXECUTION_MOVE",
            "ENRAGE_MOVE",
        ],
        move_followup={
            "PUPPET_STRINGS_MOVE": "YOU_ARE_MINE_MOVE",
            "YOU_ARE_MINE_MOVE": "YOURE_MINE_NOW_BRANCH",
            "BURN_BRIGHT_FOR_ME_MOVE": "BURN_BRIGHT_FOR_ME_BRANCH",
            "OFF_WITH_YOUR_HEAD_MOVE": "EXECUTION_MOVE",
            "EXECUTION_MOVE": "ENRAGE_MOVE",
            "ENRAGE_MOVE": "OFF_WITH_YOUR_HEAD_MOVE",
        },
        conditional_branches={
            "YOURE_MINE_NOW_BRANCH": [
                ConditionalTarget(
                    to="BURN_BRIGHT_FOR_ME_MOVE",
                    condition_expr="the companion TorchHeadAmalgam creature hasn't died yet",
                ),
                ConditionalTarget(to="OFF_WITH_YOUR_HEAD_MOVE", condition_expr=None),
            ],
            "BURN_BRIGHT_FOR_ME_BRANCH": [
                ConditionalTarget(
                    to="BURN_BRIGHT_FOR_ME_MOVE",
                    condition_expr="the companion TorchHeadAmalgam creature hasn't died yet",
                ),
                ConditionalTarget(to="OFF_WITH_YOUR_HEAD_MOVE", condition_expr=None),
            ],
        },
        notes=(
            "Both branches gate on the same complementary boolean pair (HasAmalgamDied / "
            "!HasAmalgamDied) -- exhaustive by construction, so the second edge in each is encoded "
            "as an unconditional trailing else (condition_expr=None) rather than its own dynamic "
            "condition; semantically exact, not an approximation. Separately, AfterDeath contains "
            "an external interrupt this schema doesn't model: if the Amalgam companion dies while "
            "Queen's currently-telegraphed NextMove is BURN_BRIGHT_FOR_ME_MOVE, it force-overrides "
            "NextMove directly to ENRAGE_MOVE (SetMoveImmediate), bypassing the branch logic above "
            "for that one transition."
        ),
    ),
    "ROCKET": MonsterMoveGraph(
        monster_id="ROCKET",
        class_name="Rocket",
        initial_state="TARGETING_RETICLE_MOVE",
        move_ids=[
            "TARGETING_RETICLE_MOVE",
            "PRECISION_BEAM_MOVE",
            "CHARGE_UP_MOVE",
            "LASER_MOVE",
            "RECHARGE_MOVE",
        ],
        move_followup={
            "TARGETING_RETICLE_MOVE": "PRECISION_BEAM_MOVE",
            "PRECISION_BEAM_MOVE": "CHARGE_UP_MOVE",
            "CHARGE_UP_MOVE": "LASER_MOVE",
            "LASER_MOVE": "RECHARGE_MOVE",
            "RECHARGE_MOVE": "TARGETING_RETICLE_MOVE",
        },
        notes=(
            "Fully deterministic 5-cycle, no branch states. Class is internally named/SFX-tagged "
            "after 'kaiser_crab' (this is the right-arm add-on piece of that boss), but its "
            "shipped localization id is ROCKET, matching the mechanical "
            "PascalCase->SCREAMING_SNAKE_CASE guess."
        ),
    ),
    "SCROLL_OF_BITING": MonsterMoveGraph(
        monster_id="SCROLL_OF_BITING",
        class_name="ScrollOfBiting",
        initial_state="CHOMP",
        move_ids=["CHOMP", "CHEW", "MORE_TEETH"],
        move_followup={
            "CHOMP": "MORE_TEETH",
            "MORE_TEETH": "CHEW",
            "CHEW": "rand",
        },
        random_branches={
            "rand": [
                MoveTarget(to="CHOMP", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="CHEW", weight=1.0, repeat_type="CanRepeatXTimes", max_times=2),
            ],
        },
        notes=(
            "Opens CHOMP->MORE_TEETH->CHEW once, then CHEW's followup is the 'rand' branch: CHOMP "
            "(cannot repeat back-to-back) vs. CHEW again (allowed up to 2 times in a row), looping "
            "forever after that. StarterMoveIdx % 3 (0 on a normally-spawned instance) selects the "
            "initial state -- 0=CHOMP (used here), 1=CHEW, otherwise MORE_TEETH; all three "
            "converge into the same cycle."
        ),
    ),
    "SEAPUNK": MonsterMoveGraph(
        monster_id="SEAPUNK",
        class_name="Seapunk",
        initial_state="SEA_KICK_MOVE",
        move_ids=["SEA_KICK_MOVE", "SPINNING_KICK_MOVE", "BUBBLE_BURP_MOVE"],
        move_followup={
            "SEA_KICK_MOVE": "SPINNING_KICK_MOVE",
            "SPINNING_KICK_MOVE": "BUBBLE_BURP_MOVE",
            "BUBBLE_BURP_MOVE": "SEA_KICK_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "SEWER_CLAM": MonsterMoveGraph(
        monster_id="SEWER_CLAM",
        class_name="SewerClam",
        initial_state="JET_MOVE",
        move_ids=["PRESSURIZE_MOVE", "JET_MOVE"],
        move_followup={
            "PRESSURIZE_MOVE": "JET_MOVE",
            "JET_MOVE": "PRESSURIZE_MOVE",
        },
        notes="Fully deterministic 2-cycle; the machine's initial state is JET_MOVE, not PRESSURIZE_MOVE.",
    ),
    "SINGLE_ATTACK_MOVE_MONSTER": MonsterMoveGraph(
        monster_id="SINGLE_ATTACK_MOVE_MONSTER",
        class_name="SingleAttackMoveMonster",
        initial_state="POKE",
        move_ids=["POKE"],
        move_followup={"POKE": "POKE"},
        notes=(
            "Single repeating 1-damage attack move, no branch states. No localization entry -- "
            "template/test class, not a real shipped monster (its Title even borrows "
            "BIG_DUMMY.name for display in test contexts)."
        ),
    ),
    "SKULKING_COLONY": MonsterMoveGraph(
        monster_id="SKULKING_COLONY",
        class_name="SkulkingColony",
        initial_state="ZOOM_MOVE",
        move_ids=["ZOOM_MOVE", "ZOOM_MOVE_2", "INERTIA_MOVE", "PIERCING_STABS_MOVE"],
        move_followup={
            "ZOOM_MOVE": "ZOOM_MOVE_2",
            "ZOOM_MOVE_2": "INERTIA_MOVE",
            "INERTIA_MOVE": "PIERCING_STABS_MOVE",
            "PIERCING_STABS_MOVE": "ZOOM_MOVE",
        },
        notes=(
            "Fully deterministic 4-cycle, no branch states. ZOOM_MOVE_2 is a second move state "
            "that performs the identical ZoomMove method/intent as ZOOM_MOVE under a distinct "
            "state id, used purely to give the cycle two separate steps."
        ),
    ),
    "SLIMED_BERSERKER": MonsterMoveGraph(
        monster_id="SLIMED_BERSERKER",
        class_name="SlimedBerserker",
        initial_state="VOMIT_ICHOR_MOVE",
        move_ids=["VOMIT_ICHOR_MOVE", "FURIOUS_PUMMELING_MOVE", "LEECHING_HUG_MOVE", "SMOTHER_MOVE"],
        move_followup={
            "VOMIT_ICHOR_MOVE": "FURIOUS_PUMMELING_MOVE",
            "FURIOUS_PUMMELING_MOVE": "LEECHING_HUG_MOVE",
            "LEECHING_HUG_MOVE": "SMOTHER_MOVE",
            "SMOTHER_MOVE": "VOMIT_ICHOR_MOVE",
        },
        notes="Fully deterministic 4-cycle, no branch states.",
    ),
    "SLITHERING_STRANGLER": MonsterMoveGraph(
        monster_id="SLITHERING_STRANGLER",
        class_name="SlitheringStrangler",
        initial_state="CONSTRICT",
        move_ids=["CONSTRICT", "THWACK", "LASH"],
        move_followup={
            "CONSTRICT": "rand",
            "THWACK": "CONSTRICT",
            "LASH": "CONSTRICT",
        },
        random_branches={
            "rand": [
                MoveTarget(to="THWACK", weight=1.0, repeat_type="CanRepeatForever"),
                MoveTarget(to="LASH", weight=1.0, repeat_type="CanRepeatForever"),
            ],
        },
        notes=(
            "CONSTRICT always opens/repeats after either follow-up move returns to it; 'rand' is "
            "an even 50/50 pick between THWACK and LASH with no repeat restriction on either edge."
        ),
    ),
    "SLUDGE_SPINNER": MonsterMoveGraph(
        monster_id="SLUDGE_SPINNER",
        class_name="SludgeSpinner",
        initial_state="OIL_SPRAY_MOVE",
        move_ids=["OIL_SPRAY_MOVE", "SLAM_MOVE", "RAGE_MOVE"],
        move_followup={
            "OIL_SPRAY_MOVE": "RAND",
            "SLAM_MOVE": "RAND",
            "RAGE_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="OIL_SPRAY_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="SLAM_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="RAGE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Opens at OIL_SPRAY_MOVE, then every move's followup points at the same RAND branch: "
            "an even three-way pick among all three moves, excluding whichever was just played "
            "(CannotRepeat)."
        ),
    ),
    "SLUMBERING_BEETLE": MonsterMoveGraph(
        monster_id="SLUMBERING_BEETLE",
        class_name="SlumberingBeetle",
        initial_state="SNORE_MOVE",
        move_ids=["SNORE_MOVE", "ROLL_OUT_MOVE"],
        move_followup={
            "SNORE_MOVE": "SNORE_NEXT",
            "ROLL_OUT_MOVE": "ROLL_OUT_MOVE",
        },
        conditional_branches={
            "SNORE_NEXT": [
                ConditionalTarget(to="SNORE_MOVE", condition_expr="still has SlumberPower (asleep)"),
                ConditionalTarget(to="ROLL_OUT_MOVE", condition_expr=None),
            ],
        },
        notes=(
            "SNORE_MOVE (a no-op SleepIntent move) repeats via SNORE_NEXT while SlumberPower "
            "persists (SlumberPower starts at 3 stacks and is decremented by its own power logic, "
            "not modeled here); once it's gone, SNORE_NEXT falls to ROLL_OUT_MOVE, which then "
            "self-loops forever (the wake-up trigger/IsAwake flag happens inside ROLL_OUT_MOVE's "
            "own perform method, external to this branch). The two SNORE_NEXT conditions are exact "
            "complements (HasPower<SlumberPower> / !HasPower<SlumberPower>), so the second is "
            "encoded as an unconditional trailing else (condition_expr=None) -- semantically "
            "exact, not an approximation."
        ),
    ),
    "SNAPPING_JAXFRUIT": MonsterMoveGraph(
        monster_id="SNAPPING_JAXFRUIT",
        class_name="SnappingJaxfruit",
        initial_state="ENERGY_ORB_MOVE",
        move_ids=["ENERGY_ORB_MOVE"],
        move_followup={"ENERGY_ORB_MOVE": "ENERGY_ORB_MOVE"},
        notes="Single repeating attack+self-buff move, no branch states.",
    ),
    "SNEAKY_GREMLIN": MonsterMoveGraph(
        monster_id="SNEAKY_GREMLIN",
        class_name="SneakyGremlin",
        initial_state="SPAWNED_MOVE",
        move_ids=["SPAWNED_MOVE", "TACKLE_MOVE"],
        move_followup={
            "SPAWNED_MOVE": "TACKLE_MOVE",
            "TACKLE_MOVE": "TACKLE_MOVE",
        },
        notes="SPAWNED_MOVE (a one-time stunned wake-up move) plays once, then TACKLE_MOVE repeats forever.",
    ),
}
