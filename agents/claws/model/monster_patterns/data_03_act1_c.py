"""Hand-extracted move state machines for 17 monsters, batch "act1_c".

Source: sts2-decompiled/src/Core/Models/Monsters/{Flyconid,Fogmog,FossilStalker,FrogKnight,
FuzzyWurmCrawler,GasBomb,GlobeHead,GremlinMerc,Guardbot,HauntedShip,HunterKiller,InfestedPrism,
Inklet,KinFollower,KinPriest,KnowledgeDemon,LagavulinMatriarch}.cs (GenerateMoveStateMachine).
Never copy decompiled source into this repo -- only short state-id strings, weight values, and
hand-paraphrased condition/weight summaries are captured here, no game logic or text.
"""

from __future__ import annotations

from .schema import ConditionalTarget, MonsterMoveGraph, MoveTarget

GRAPHS: dict[str, MonsterMoveGraph] = {
    "FLYCONID": MonsterMoveGraph(
        monster_id="FLYCONID",
        class_name="Flyconid",
        # Constructor's initialState is the "INITIAL" random branch (a 2-option variant of "RAND"
        # missing the VULNERABLE_SPORES_MOVE option), not any single move -- a fresh Flyconid rolls
        # its first move from FRAIL_SPORES_MOVE/SMASH_MOVE only.
        initial_state="INITIAL",
        move_ids=["VULNERABLE_SPORES_MOVE", "FRAIL_SPORES_MOVE", "SMASH_MOVE"],
        move_followup={
            "VULNERABLE_SPORES_MOVE": "RAND",
            "FRAIL_SPORES_MOVE": "RAND",
            "SMASH_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="VULNERABLE_SPORES_MOVE", weight=1.0, repeat_type="CannotRepeat", cooldown=3),
                MoveTarget(to="FRAIL_SPORES_MOVE", weight=1.0, repeat_type="CannotRepeat", cooldown=2),
                MoveTarget(to="SMASH_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
            "INITIAL": [
                MoveTarget(to="FRAIL_SPORES_MOVE", weight=1.0, repeat_type="CannotRepeat", cooldown=2),
                MoveTarget(to="SMASH_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Two random branches: RAND (reached after any move) offers all three moves with "
            "per-edge cooldowns (3 turns for VULNERABLE_SPORES_MOVE, 2 for FRAIL_SPORES_MOVE, none "
            "for SMASH_MOVE); INITIAL is a separate, narrower branch used only as the machine's "
            "starting point and excludes VULNERABLE_SPORES_MOVE entirely. All weights are equal "
            "(1.0) constants, no dynamic weights."
        ),
    ),
    "FOGMOG": MonsterMoveGraph(
        monster_id="FOGMOG",
        class_name="Fogmog",
        initial_state="ILLUSION_MOVE",
        # SWIPE_MOVE and SWIPE_RANDOM_MOVE share the same performMethod (SwipeMove) but are
        # distinct MoveState ids with different roles in the graph -- SWIPE_MOVE is the fixed
        # post-ILLUSION/post-HEADBUTT follow-up, SWIPE_RANDOM_MOVE is only reachable via BRANCH.
        move_ids=["ILLUSION_MOVE", "SWIPE_MOVE", "SWIPE_RANDOM_MOVE", "HEADBUTT_MOVE"],
        move_followup={
            "ILLUSION_MOVE": "SWIPE_MOVE",
            "SWIPE_MOVE": "BRANCH",
            "SWIPE_RANDOM_MOVE": "HEADBUTT_MOVE",
            "HEADBUTT_MOVE": "SWIPE_MOVE",
        },
        random_branches={
            "BRANCH": [
                MoveTarget(to="SWIPE_RANDOM_MOVE", weight=0.4, repeat_type="CannotRepeat"),
                MoveTarget(to="HEADBUTT_MOVE", weight=0.6, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Opens with ILLUSION_MOVE (summon) once, then SWIPE_MOVE always follows; after "
            "SWIPE_MOVE, a 40/60 constant-weight roll picks SWIPE_RANDOM_MOVE (which always leads "
            "into HEADBUTT_MOVE) or HEADBUTT_MOVE directly, and HEADBUTT_MOVE always leads back to "
            "SWIPE_MOVE -- so SWIPE_MOVE is the true hub after the opener."
        ),
    ),
    "FOSSIL_STALKER": MonsterMoveGraph(
        monster_id="FOSSIL_STALKER",
        class_name="FossilStalker",
        initial_state="LATCH_MOVE",
        move_ids=["TACKLE_MOVE", "LATCH_MOVE", "LASH_MOVE"],
        move_followup={
            "TACKLE_MOVE": "RAND",
            "LATCH_MOVE": "RAND",
            "LASH_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="LATCH_MOVE", weight=2.0),
                MoveTarget(to="TACKLE_MOVE", weight=2.0),
                MoveTarget(to="LASH_MOVE", weight=2.0),
            ],
        },
        notes=(
            "All three FollowUpState assignments are done via one chained expression in the "
            "source (a.FollowUpState = (b.FollowUpState = (c.FollowUpState = newBranch))), all "
            "pointing at the same RAND branch. RAND's three edges are equal-weight "
            "(2.0 each, so effectively uniform 1/3) with CanRepeatForever (default) -- no repeat "
            "restrictions on any option."
        ),
    ),
    "FROG_KNIGHT": MonsterMoveGraph(
        monster_id="FROG_KNIGHT",
        class_name="FrogKnight",
        initial_state="TONGUE_LASH",
        move_ids=["FOR_THE_QUEEN", "STRIKE_DOWN_EVIL", "TONGUE_LASH", "BEETLE_CHARGE"],
        move_followup={
            "FOR_THE_QUEEN": "HALF_HEALTH",
            "STRIKE_DOWN_EVIL": "FOR_THE_QUEEN",
            "TONGUE_LASH": "STRIKE_DOWN_EVIL",
            "BEETLE_CHARGE": "TONGUE_LASH",
        },
        conditional_branches={
            "HALF_HEALTH": [
                ConditionalTarget(
                    to="TONGUE_LASH",
                    condition_expr=(
                        "has already beetle-charged once this fight, or current HP is at or above "
                        "half of max HP"
                    ),
                ),
                # The C# source's second AddState condition is the exact logical negation of the
                # first (De Morgan's law: !(charged || hp>=half) == !charged && hp<half), so it is
                # guaranteed true whenever the first is false -- encoded here as the trailing
                # unconditional else rather than a second paraphrased predicate, which is 100%
                # behavior-preserving and lets this resolve without a hand-ported condition.
                ConditionalTarget(to="BEETLE_CHARGE", condition_expr=None),
            ],
        },
        notes=(
            "HasBeetleCharged is a per-instance bool reset to false in AfterAddedToRoom (once per "
            "combat) and set true the first time BEETLE_CHARGE_MOVE resolves -- so BEETLE_CHARGE "
            "can only ever be picked once per fight (before HP drops under half) or not at all (if "
            "HP never drops under half before that check is reached); after that it's permanently "
            "routed to TONGUE_LASH from HALF_HEALTH regardless of HP."
        ),
    ),
    "FUZZY_WURM_CRAWLER": MonsterMoveGraph(
        monster_id="FUZZY_WURM_CRAWLER",
        class_name="FuzzyWurmCrawler",
        initial_state="FIRST_ACID_GOOP",
        move_ids=["FIRST_ACID_GOOP", "ACID_GOOP", "INHALE"],
        move_followup={
            "FIRST_ACID_GOOP": "INHALE",
            "INHALE": "ACID_GOOP",
            "ACID_GOOP": "FIRST_ACID_GOOP",
        },
        notes=(
            "Fully deterministic 3-cycle, no branch states. FIRST_ACID_GOOP and ACID_GOOP share "
            "the same performMethod but are distinct move ids (FIRST_ACID_GOOP is excluded from "
            "the bestiary display, suggesting it's meant to look identical to ACID_GOOP in-game)."
        ),
    ),
    "GAS_BOMB": MonsterMoveGraph(
        monster_id="GAS_BOMB",
        class_name="GasBomb",
        initial_state="EXPLODE_MOVE",
        move_ids=["EXPLODE_MOVE"],
        move_followup={"EXPLODE_MOVE": "EXPLODE_MOVE"},
        notes=(
            "Single-move machine: EXPLODE_MOVE is a DeathBlowIntent move whose performMethod kills "
            "the GasBomb itself (CreatureCmd.Kill) right after dealing damage, so the source never "
            "sets a real FollowUpState -- there is no second move. A self-loop was added only to "
            "satisfy this schema's 'every move needs a move_followup entry' invariant; in practice "
            "the creature is dead before that edge would ever be walked."
        ),
    ),
    "GLOBE_HEAD": MonsterMoveGraph(
        monster_id="GLOBE_HEAD",
        class_name="GlobeHead",
        initial_state="SHOCKING_SLAP",
        move_ids=["THUNDER_STRIKE", "SHOCKING_SLAP", "GALVANIC_BURST"],
        move_followup={
            "SHOCKING_SLAP": "THUNDER_STRIKE",
            "THUNDER_STRIKE": "GALVANIC_BURST",
            "GALVANIC_BURST": "SHOCKING_SLAP",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "GREMLIN_MERC": MonsterMoveGraph(
        monster_id="GREMLIN_MERC",
        class_name="GremlinMerc",
        initial_state="GIMME_MOVE",
        move_ids=["GIMME_MOVE", "DOUBLE_SMASH_MOVE", "HEHE_MOVE"],
        move_followup={
            "GIMME_MOVE": "DOUBLE_SMASH_MOVE",
            "DOUBLE_SMASH_MOVE": "HEHE_MOVE",
            "HEHE_MOVE": "GIMME_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "GUARDBOT": MonsterMoveGraph(
        monster_id="GUARDBOT",
        class_name="Guardbot",
        initial_state="GUARD_MOVE",
        move_ids=["GUARD_MOVE"],
        move_followup={"GUARD_MOVE": "GUARD_MOVE"},
        notes=(
            "Single move, explicit self-loop in source (moveState.FollowUpState = moveState) -- "
            "every turn it grants block to allied Fabricator-type creatures. No branch states."
        ),
    ),
    "HAUNTED_SHIP": MonsterMoveGraph(
        monster_id="HAUNTED_SHIP",
        class_name="HauntedShip",
        initial_state="HAUNT_MOVE",
        move_ids=["SWIPE_MOVE", "STOMP_MOVE", "HAUNT_MOVE"],
        move_followup={
            "HAUNT_MOVE": "SWIPE_MOVE",
            "SWIPE_MOVE": "STOMP_MOVE",
            "STOMP_MOVE": "SWIPE_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. Opens with HAUNT_MOVE once (its FollowUpState "
            "is set inline via an object initializer), then SWIPE_MOVE/STOMP_MOVE alternate "
            "forever -- same shape as SHRINKER_BEETLE's cycle."
        ),
    ),
    "HUNTER_KILLER": MonsterMoveGraph(
        monster_id="HUNTER_KILLER",
        class_name="HunterKiller",
        initial_state="TENDERIZING_GOOP_MOVE",
        move_ids=["TENDERIZING_GOOP_MOVE", "BITE_MOVE", "PUNCTURE_MOVE"],
        move_followup={
            "TENDERIZING_GOOP_MOVE": "RAND",
            "BITE_MOVE": "RAND",
            "PUNCTURE_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="BITE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="PUNCTURE_MOVE", weight=2.0),
            ],
        },
        notes=(
            "Opens with TENDERIZING_GOOP_MOVE once (its FollowUpState, along with BITE_MOVE's and "
            "PUNCTURE_MOVE's, is set via a chained assignment straight to RAND), then RAND rolls "
            "BITE_MOVE (weight 1, cannot repeat back-to-back) vs. PUNCTURE_MOVE (weight 2, can "
            "repeat freely) forever -- PUNCTURE_MOVE is twice as likely on any given roll, and gets "
            "relatively more likely still right after a BITE_MOVE (which becomes ineligible)."
        ),
    ),
    "INFESTED_PRISM": MonsterMoveGraph(
        monster_id="INFESTED_PRISM",
        class_name="InfestedPrism",
        initial_state="JAB_MOVE",
        move_ids=["JAB_MOVE", "RADIATE_MOVE", "WHIRLWIND_MOVE", "PULSATE_MOVE"],
        move_followup={
            "JAB_MOVE": "RADIATE_MOVE",
            "RADIATE_MOVE": "WHIRLWIND_MOVE",
            "WHIRLWIND_MOVE": "PULSATE_MOVE",
            "PULSATE_MOVE": "JAB_MOVE",
        },
        notes="Fully deterministic 4-cycle, no branch states.",
    ),
    "INKLET": MonsterMoveGraph(
        monster_id="INKLET",
        class_name="Inklet",
        # MiddleInklet defaults to false for a normally-spawned instance, giving JAB_MOVE as the
        # real starting move.
        initial_state="JAB_MOVE",
        move_ids=["JAB_MOVE", "WHIRLWIND_MOVE", "PIERCING_GAZE_MOVE"],
        move_followup={
            # A chained assignment first points all three moves' FollowUpState at a new "RAND"
            # branch, but two of those are immediately overwritten by later explicit assignments
            # (moveState2.FollowUpState = moveState; moveState3.FollowUpState = moveState) -- the
            # final, actually-effective edges are the ones below.
            "JAB_MOVE": "RAND",
            "WHIRLWIND_MOVE": "JAB_MOVE",
            "PIERCING_GAZE_MOVE": "JAB_MOVE",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="PIERCING_GAZE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="WHIRLWIND_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "The source also constructs a second RandomBranchState, \"INIT_RAND\" (AddBranch(JAB, "
            "maxRepeats=2, weight=1), AddBranch(WHIRLWIND, CannotRepeat, weight=1)), but it is "
            "never added to the state list passed into MonsterMoveStateMachine and never becomes "
            "any move's FollowUpState or the initialState -- it is dead/unreachable code (possibly "
            "a leftover from an earlier design) and is deliberately omitted from this graph since "
            "it can never actually be visited. Separately, MiddleInklet (a settable bool, default "
            "false) swaps the real initial move to WHIRLWIND_MOVE instead of JAB_MOVE for "
            "'middle of the pack' triplet spawns -- not modeled here since a fresh default instance "
            "starts at JAB_MOVE."
        ),
    ),
    "KIN_FOLLOWER": MonsterMoveGraph(
        monster_id="KIN_FOLLOWER",
        class_name="KinFollower",
        # StartsWithDance defaults to false for a normally-spawned instance, giving QUICK_SLASH_MOVE
        # as the real starting move.
        initial_state="QUICK_SLASH_MOVE",
        move_ids=["QUICK_SLASH_MOVE", "BOOMERANG_MOVE", "POWER_DANCE_MOVE"],
        move_followup={
            "QUICK_SLASH_MOVE": "BOOMERANG_MOVE",
            "BOOMERANG_MOVE": "POWER_DANCE_MOVE",
            "POWER_DANCE_MOVE": "QUICK_SLASH_MOVE",
        },
        notes=(
            "Fully deterministic 3-cycle, no branch states. StartsWithDance (settable bool, "
            "default false -- likely set true by whatever spawns/orchestrates Kin encounters for "
            "specific slots) swaps the initial move to POWER_DANCE_MOVE instead of "
            "QUICK_SLASH_MOVE; not modeled here since a fresh default instance starts at "
            "QUICK_SLASH_MOVE."
        ),
    ),
    "KIN_PRIEST": MonsterMoveGraph(
        monster_id="KIN_PRIEST",
        class_name="KinPriest",
        initial_state="ORB_OF_FRAILTY_MOVE",
        move_ids=["ORB_OF_FRAILTY_MOVE", "ORB_OF_WEAKNESS_MOVE", "BEAM_MOVE", "RITUAL_MOVE"],
        move_followup={
            "ORB_OF_FRAILTY_MOVE": "ORB_OF_WEAKNESS_MOVE",
            "ORB_OF_WEAKNESS_MOVE": "BEAM_MOVE",
            "BEAM_MOVE": "RITUAL_MOVE",
            "RITUAL_MOVE": "ORB_OF_FRAILTY_MOVE",
        },
        notes="Fully deterministic 4-cycle, no branch states.",
    ),
    "KNOWLEDGE_DEMON": MonsterMoveGraph(
        monster_id="KNOWLEDGE_DEMON",
        class_name="KnowledgeDemon",
        initial_state="CURSE_OF_KNOWLEDGE_MOVE",
        move_ids=[
            "CURSE_OF_KNOWLEDGE_MOVE",
            "SLAP_MOVE",
            "KNOWLEDGE_OVERWHELMING_MOVE",
            "PONDER_MOVE",
        ],
        move_followup={
            "CURSE_OF_KNOWLEDGE_MOVE": "SLAP_MOVE",
            "SLAP_MOVE": "KNOWLEDGE_OVERWHELMING_MOVE",
            "KNOWLEDGE_OVERWHELMING_MOVE": "PONDER_MOVE",
            "PONDER_MOVE": "CurseOfKnowledgeBranch",
        },
        conditional_branches={
            "CurseOfKnowledgeBranch": [
                ConditionalTarget(
                    to="CURSE_OF_KNOWLEDGE_MOVE",
                    condition_expr=(
                        "per-instance curse-set counter (increments by one each time "
                        "CURSE_OF_KNOWLEDGE_MOVE actually resolves in live combat, out of 3 "
                        "possible curse sets) is still below 3"
                    ),
                ),
                # The source's second AddState condition (counter >= 3) is the exact integer
                # complement of "< 3" -- guaranteed true whenever the first is false -- so it is
                # encoded as the trailing unconditional else, matching FrogKnight's HALF_HEALTH
                # pattern above.
                ConditionalTarget(to="SLAP_MOVE", condition_expr=None),
            ],
        },
        notes=(
            "The counter this branch reads is a plain per-instance int starting at 0 and capped by "
            "construction at 3 (there are only 3 curse-card sets to draw from) -- it's exactly "
            "equal to how many times CURSE_OF_KNOWLEDGE_MOVE has appeared in this monster's own "
            "move history so far, so a resolver with access to full per-monster history could "
            "compute it exactly rather than needing live game state; it isn't wired that way here "
            "since MoveTarget/ConditionalTarget's dynamic hooks only receive a flat `context` dict, "
            "not `history`, in the current resolver signature."
        ),
    ),
    "LAGAVULIN_MATRIARCH": MonsterMoveGraph(
        monster_id="LAGAVULIN_MATRIARCH",
        class_name="LagavulinMatriarch",
        initial_state="SLEEP_MOVE",
        move_ids=["SLEEP_MOVE", "SLASH_MOVE", "SLASH2_MOVE", "DISEMBOWEL_MOVE", "SOUL_SIPHON_MOVE"],
        move_followup={
            "SLEEP_MOVE": "SLEEP_BRANCH",
            "SLASH_MOVE": "DISEMBOWEL_MOVE",
            "DISEMBOWEL_MOVE": "SLASH2_MOVE",
            "SLASH2_MOVE": "SOUL_SIPHON_MOVE",
            "SOUL_SIPHON_MOVE": "SLASH_MOVE",
        },
        conditional_branches={
            "SLEEP_BRANCH": [
                ConditionalTarget(
                    to="SLEEP_MOVE",
                    condition_expr="still carries the Asleep power (hasn't been damaged awake yet)",
                ),
                # The source's second AddState condition (!HasPower<AsleepPower>()) is the exact
                # negation of the first, so it's encoded as the trailing unconditional else.
                ConditionalTarget(to="SLASH_MOVE", condition_expr=None),
            ],
        },
        notes=(
            "SLEEP_MOVE itself is a no-op move (empty performMethod, SleepIntent) representing a "
            "turn spent asleep; AsleepPower (3 stacks) and PlatingPower (12) are applied once at "
            "combat start, and AsleepPower is presumably removed/expires via the game's normal "
            "power-tick or on-damage rules (not visible in GenerateMoveStateMachine itself), at "
            "which point SLEEP_BRANCH permanently routes to the SLASH_MOVE-led attack cycle -- this "
            "monster only ever plays SLEEP_MOVE repeatedly, then transitions once and never returns "
            "to sleep. WakeUpMove is a separate non-state-machine trigger (bestiary/visual only) and "
            "is not part of the move graph."
        ),
    ),
}
