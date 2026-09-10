"""Hand-extracted move state machines for a batch of (mostly) Act 1 monsters.

Source: sts2-decompiled/src/Core/Models/Monsters/{CeremonialBeast,Chomper,CorpseSlug,
CrossbowRubyRaider,Crusher,CubexConstruct,DampCultist,DecimillipedeSegment,DeprecatedMonster,
DevotedSculptor,Entomancer,Exoskeleton,EyeWithTeeth,Fabricator,FakeMerchantMonster,FatGremlin,
FlailKnight}.cs (GenerateMoveStateMachine, plus the handful of private fields/properties it
directly reads for starter-move selection or branch gating). Never copy decompiled source into
this repo -- only short state-id strings and weight values are captured here, no game logic,
damage numbers, or flavor text.
"""

from __future__ import annotations

from .schema import ConditionalTarget, MonsterMoveGraph, MoveTarget

GRAPHS: dict[str, MonsterMoveGraph] = {
    "CEREMONIAL_BEAST": MonsterMoveGraph(
        monster_id="CEREMONIAL_BEAST",
        class_name="CeremonialBeast",
        initial_state="STAMP_MOVE",
        move_ids=["STAMP_MOVE", "PLOW_MOVE", "STUN_MOVE", "BEAST_CRY_MOVE", "STOMP_MOVE", "CRUSH_MOVE"],
        move_followup={
            "STAMP_MOVE": "PLOW_MOVE",
            "PLOW_MOVE": "PLOW_MOVE",
            "STUN_MOVE": "BEAST_CRY_MOVE",
            "BEAST_CRY_MOVE": "STOMP_MOVE",
            "STOMP_MOVE": "CRUSH_MOVE",
            "CRUSH_MOVE": "BEAST_CRY_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. PLOW_MOVE's own FollowUpState points back to "
            "itself, so once reached the beast repeats PLOW_MOVE forever within this graph -- a "
            "separate mechanic (IsStunnedByPlowRemoval / SetStunned, triggered when the player "
            "strips the PlowPower it applies to itself) externally forces it into STUN_MOVE, which "
            "is why STUN_MOVE has a FollowUpState (-> BEAST_CRY_MOVE) but is never targeted by any "
            "other move here -- it's an interrupt entry point set directly by game code outside "
            "GenerateMoveStateMachine, not reachable via normal graph traversal."
        ),
    ),
    "CHOMPER": MonsterMoveGraph(
        monster_id="CHOMPER",
        class_name="Chomper",
        # ScreamFirst is a settable property defaulting to false for a freshly-spawned instance,
        # which starts the machine at CLAMP_MOVE; some encounter setup elsewhere may flip it.
        initial_state="CLAMP_MOVE",
        move_ids=["CLAMP_MOVE", "SCREECH_MOVE"],
        move_followup={
            "CLAMP_MOVE": "SCREECH_MOVE",
            "SCREECH_MOVE": "CLAMP_MOVE",
        },
        notes=(
            "Fully deterministic 2-cycle, no branch states. If ScreamFirst is set true (not the "
            "default), the machine instead starts at SCREECH_MOVE -- same cycle, different entry "
            "point."
        ),
    ),
    "CORPSE_SLUG": MonsterMoveGraph(
        monster_id="CORPSE_SLUG",
        class_name="CorpseSlug",
        # StarterMoveIdx defaults to 0 (int default) => WHIP_SLAP_MOVE for a freshly-constructed
        # instance, though EnsureCorpseSlugsStartWithDifferentMoves rerolls it per-instance at
        # encounter setup so multiple slugs in one fight don't open with the same move.
        initial_state="WHIP_SLAP_MOVE",
        move_ids=["WHIP_SLAP_MOVE", "GLOMP_MOVE", "GOOP_MOVE"],
        move_followup={
            "WHIP_SLAP_MOVE": "GLOMP_MOVE",
            "GLOMP_MOVE": "GOOP_MOVE",
            "GOOP_MOVE": "WHIP_SLAP_MOVE",
        },
        notes=(
            "Fully deterministic 3-cycle, no branch states. StarterMoveIdx % 3 picks the entry "
            "point (0=WHIP_SLAP_MOVE, 1=GLOMP_MOVE, else=GOOP_MOVE); EnsureCorpseSlugsStartWithDifferentMoves "
            "assigns a different offset to each CorpseSlug in an encounter via a shared rng draw, "
            "so 'freshly spawned' in practice usually means a non-zero offset -- only the cycle "
            "shape (not the exact entry point) should be treated as fixed across instances."
        ),
    ),
    "CROSSBOW_RUBY_RAIDER": MonsterMoveGraph(
        monster_id="CROSSBOW_RUBY_RAIDER",
        class_name="CrossbowRubyRaider",
        initial_state="RELOAD_MOVE",
        move_ids=["FIRE_MOVE", "RELOAD_MOVE"],
        move_followup={
            "FIRE_MOVE": "RELOAD_MOVE",
            "RELOAD_MOVE": "FIRE_MOVE",
        },
        notes="Fully deterministic 2-cycle, no branch states. Opens on RELOAD_MOVE (defend/reload).",
    ),
    "CRUSHER": MonsterMoveGraph(
        monster_id="CRUSHER",
        class_name="Crusher",
        initial_state="THRASH_MOVE",
        move_ids=["THRASH_MOVE", "ENLARGING_STRIKE_MOVE", "BUG_STING_MOVE", "ADAPT_MOVE", "GUARDED_STRIKE_MOVE"],
        move_followup={
            "THRASH_MOVE": "ENLARGING_STRIKE_MOVE",
            "ENLARGING_STRIKE_MOVE": "BUG_STING_MOVE",
            "BUG_STING_MOVE": "ADAPT_MOVE",
            "ADAPT_MOVE": "GUARDED_STRIKE_MOVE",
            "GUARDED_STRIKE_MOVE": "THRASH_MOVE",
        },
        notes=(
            "Fully deterministic 5-cycle, no branch states. Sfx/asset paths (kaiser_crab_left, "
            "kaiser_crab_boss_explosion) suggest this is one arm/side of a two-part 'Kaiser Crab' "
            "boss pairing -- irrelevant to the move graph itself, which is a single fixed cycle."
        ),
    ),
    "CUBEX_CONSTRUCT": MonsterMoveGraph(
        monster_id="CUBEX_CONSTRUCT",
        class_name="CubexConstruct",
        initial_state="CHARGE_UP_MOVE",
        move_ids=["CHARGE_UP_MOVE", "REPEATER_BLAST_MOVE", "REPEATER_BLAST_MOVE_2", "EXPEL_MOVE"],
        move_followup={
            "CHARGE_UP_MOVE": "REPEATER_BLAST_MOVE",
            "REPEATER_BLAST_MOVE": "REPEATER_BLAST_MOVE_2",
            "REPEATER_BLAST_MOVE_2": "EXPEL_MOVE",
            "EXPEL_MOVE": "REPEATER_BLAST_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. CHARGE_UP_MOVE only ever happens once (opening "
            "move); after EXPEL_MOVE the cycle is REPEATER_BLAST_MOVE -> REPEATER_BLAST_MOVE_2 -> "
            "EXPEL_MOVE -> REPEATER_BLAST_MOVE forever (REPEATER_BLAST_MOVE and _2 share the same "
            "underlying handler/behavior, just distinct state ids for the two-move buildup)."
        ),
    ),
    "DAMP_CULTIST": MonsterMoveGraph(
        monster_id="DAMP_CULTIST",
        class_name="DampCultist",
        initial_state="INCANTATION_MOVE",
        move_ids=["INCANTATION_MOVE", "DARK_STRIKE_MOVE"],
        move_followup={
            "INCANTATION_MOVE": "DARK_STRIKE_MOVE",
            "DARK_STRIKE_MOVE": "DARK_STRIKE_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. Casts INCANTATION_MOVE exactly once, then "
            "DARK_STRIKE_MOVE loops on itself forever (its own FollowUpState is itself)."
        ),
    ),
    "DECIMILLIPEDE_SEGMENT": MonsterMoveGraph(
        monster_id="DECIMILLIPEDE_SEGMENT",
        class_name="DecimillipedeSegment",
        # abstract base class shared by all Decimillipede body segments; Title itself resolves via
        # L10NMonsterLookup("DECIMILLIPEDE_SEGMENT.name"), confirming this id directly.
        # StarterMoveIdx defaults to 0 => WRITHE_MOVE for a freshly-constructed instance.
        initial_state="WRITHE_MOVE",
        move_ids=["WRITHE_MOVE", "BULK_MOVE", "CONSTRICT_MOVE", "DEAD_MOVE", "REATTACH_MOVE"],
        move_followup={
            "WRITHE_MOVE": "CONSTRICT_MOVE",
            "CONSTRICT_MOVE": "BULK_MOVE",
            "BULK_MOVE": "WRITHE_MOVE",
            "DEAD_MOVE": "REATTACH_MOVE",
            "REATTACH_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="WRITHE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="BULK_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="CONSTRICT_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Normal play is a fixed 3-cycle (WRITHE_MOVE -> CONSTRICT_MOVE -> BULK_MOVE -> ...); "
            "StarterMoveIdx % 3 picks the entry point (0=WRITHE_MOVE, 1=BULK_MOVE, else=CONSTRICT_MOVE). "
            "DEAD_MOVE (a no-op placeholder move) and REATTACH_MOVE (must-perform-once heal move, "
            "gated by the segment's ReattachPower) only enter play via the segment's 'reattach' "
            "revive mechanic, set as current state directly by game code rather than reached through "
            "any other move's FollowUpState -- REATTACH_MOVE's own followup is an even 3-way random "
            "pick among the three regular moves (each CannotRepeat, i.e. excluded only if it was the "
            "immediately preceding move)."
        ),
    ),
    "DEPRECATED_MONSTER": MonsterMoveGraph(
        monster_id="DEPRECATED_MONSTER",
        class_name="DeprecatedMonster",
        initial_state="STUB",
        move_ids=["STUB"],
        move_followup={"STUB": "STUB"},
        notes=(
            "Localization does have a 'Deprecated Monster' entry, but per the class's own doc "
            "comment this represents a monster removed from the game and is 'mostly used for the "
            "run history' -- not a live combat spawn. Its GenerateMoveStateMachine is a single "
            "HiddenIntent STUB move that loops on itself; never expect this monster_id in real "
            "training data."
        ),
    ),
    "DEVOTED_SCULPTOR": MonsterMoveGraph(
        monster_id="DEVOTED_SCULPTOR",
        class_name="DevotedSculptor",
        initial_state="FORBIDDEN_INCANTATION_MOVE",
        move_ids=["FORBIDDEN_INCANTATION_MOVE", "SAVAGE_MOVE"],
        move_followup={
            "FORBIDDEN_INCANTATION_MOVE": "SAVAGE_MOVE",
            "SAVAGE_MOVE": "SAVAGE_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. Casts FORBIDDEN_INCANTATION_MOVE exactly once, "
            "then SAVAGE_MOVE loops on itself forever."
        ),
    ),
    "ENTOMANCER": MonsterMoveGraph(
        monster_id="ENTOMANCER",
        class_name="Entomancer",
        initial_state="BEES_MOVE",
        move_ids=["PHEROMONE_SPIT_MOVE", "BEES_MOVE", "SPEAR_MOVE"],
        move_followup={
            "BEES_MOVE": "SPEAR_MOVE",
            "SPEAR_MOVE": "PHEROMONE_SPIT_MOVE",
            "PHEROMONE_SPIT_MOVE": "BEES_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states. Opens on BEES_MOVE.",
    ),
    "EXOSKELETON": MonsterMoveGraph(
        monster_id="EXOSKELETON",
        class_name="Exoskeleton",
        initial_state="INIT_MOVE",
        move_ids=["SKITTER_MOVE", "MANDIBLES_MOVE", "ENRAGE_MOVE"],
        move_followup={
            "SKITTER_MOVE": "RAND",
            "MANDIBLES_MOVE": "ENRAGE_MOVE",
            "ENRAGE_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="SKITTER_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="MANDIBLES_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        conditional_branches={
            "INIT_MOVE": [
                ConditionalTarget(to="SKITTER_MOVE", condition_expr="creature occupies encounter slot 'first'"),
                ConditionalTarget(to="MANDIBLES_MOVE", condition_expr="creature occupies encounter slot 'second'"),
                ConditionalTarget(to="ENRAGE_MOVE", condition_expr="creature occupies encounter slot 'third'"),
                ConditionalTarget(to="RAND", condition_expr=None),
            ],
        },
        notes=(
            "GenerateMoveStateMachine's initial state is a ConditionalBranchState ('INIT_MOVE') "
            "that picks the opening move by Creature.SlotName -- four Exoskeletons are summoned "
            "together, one per slot ('first'/'second'/'third'/'fourth'), each opening on a "
            "different move. The source's four AddState(...) conditions are all explicit "
            "(SlotName == 'first'/'second'/'third'/'fourth') with no literal else; here the "
            "'fourth' condition is collapsed into the trailing unconditional edge (-> RAND) since "
            "the four slot values are mutually exclusive and exhaustive in every real encounter, "
            "which keeps this graph resolvable by resolver.py's documented conservative fallthrough "
            "without a slot-name DYNAMIC_CONDITIONS entry. Until slot_name is threaded through "
            "context with a real DYNAMIC_CONDITIONS entry for the first three edges, cold-start "
            "resolution will always fall through to the RAND branch (even 50/50 between SKITTER_MOVE "
            "and MANDIBLES_MOVE) rather than the slot-accurate opener. After the opener, SKITTER_MOVE "
            "and MANDIBLES_MOVE->ENRAGE_MOVE both loop back into the same RAND branch."
        ),
    ),
    "EYE_WITH_TEETH": MonsterMoveGraph(
        monster_id="EYE_WITH_TEETH",
        class_name="EyeWithTeeth",
        initial_state="DISTRACT_MOVE",
        move_ids=["DISTRACT_MOVE"],
        move_followup={"DISTRACT_MOVE": "DISTRACT_MOVE"},
        notes="Trivial single-move loop, no branch states -- only ever performs DISTRACT_MOVE.",
    ),
    "FABRICATOR": MonsterMoveGraph(
        monster_id="FABRICATOR",
        class_name="Fabricator",
        initial_state="fabricateBranch",
        move_ids=["FABRICATE_MOVE", "FABRICATING_STRIKE_MOVE", "DISINTEGRATE_MOVE"],
        move_followup={
            "FABRICATE_MOVE": "fabricateBranch",
            "FABRICATING_STRIKE_MOVE": "fabricateBranch",
            "DISINTEGRATE_MOVE": "fabricateBranch",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="FABRICATE_MOVE", weight=1.0, repeat_type="CanRepeatForever"),
                MoveTarget(to="FABRICATING_STRIKE_MOVE", weight=1.0, repeat_type="CanRepeatForever"),
            ],
        },
        conditional_branches={
            "fabricateBranch": [
                ConditionalTarget(
                    to="RAND",
                    condition_expr="CanFabricate: fewer than 4 living teammates currently on the field",
                ),
                ConditionalTarget(to="DISINTEGRATE_MOVE", condition_expr=None),
            ],
        },
        notes=(
            "Every move loops back into a ConditionalBranchState ('fabricateBranch') gating on "
            "CanFabricate (combat-wide living-teammate count < 4). The source writes this as two "
            "explicit AddState calls (CanFabricate / !CanFabricate) with no literal else; represented "
            "here as [CanFabricate-condition, unconditional else] since !CanFabricate is exactly the "
            "logical complement, which keeps resolution well-defined. CanFabricate depends on "
            "combat-wide teammate state not in a flat per-decision context dict (same category as "
            "TWO_TAILED_RAT's CanSummon()) -- unless a DYNAMIC_CONDITIONS entry is hand-ported, "
            "resolution conservatively always falls to the DISINTEGRATE_MOVE else-branch. When "
            "CanFabricate is true, RAND picks evenly between FABRICATE_MOVE (summon-only) and "
            "FABRICATING_STRIKE_MOVE (attack + summon), both CanRepeatForever with equal constant "
            "weight (the source wraps '1f' in a Func<float> lambda for both, but it reads no "
            "external state so it's resolved as a plain constant here)."
        ),
    ),
    "FAKE_MERCHANT_MONSTER": MonsterMoveGraph(
        monster_id="FAKE_MERCHANT_MONSTER",
        class_name="FakeMerchantMonster",
        initial_state="SWIPE_MOVE",
        move_ids=["SWIPE_MOVE", "SPEW_COINS_MOVE", "THROW_RELIC_MOVE", "ENRAGE_MOVE"],
        move_followup={
            "SWIPE_MOVE": "RAND_MOVE",
            "SPEW_COINS_MOVE": "RAND_MOVE",
            "ENRAGE_MOVE": "RAND_MOVE",
            "THROW_RELIC_MOVE": "RAND_ATTACK_MOVE",
        },
        random_branches={
            "RAND_MOVE": [
                MoveTarget(to="SWIPE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="SPEW_COINS_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="THROW_RELIC_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="ENRAGE_MOVE", weight=1.0, repeat_type="CannotRepeat", cooldown=3),
            ],
            "RAND_ATTACK_MOVE": [
                MoveTarget(to="SWIPE_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="SPEW_COINS_MOVE", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="THROW_RELIC_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes=(
            "Despite the class name suggesting a test/template stand-in, localization has a real "
            "entry ('The Merchant???') and sfx paths reference 'reverse_merchant', so this is a "
            "genuine shipped enemy (an impostor-merchant encounter), not a debug class. "
            "THROW_RELIC_MOVE routes into a second, narrower random branch (RAND_ATTACK_MOVE, no "
            "ENRAGE_MOVE option) while every other move routes into the main RAND_MOVE branch "
            "(which includes ENRAGE_MOVE, gated by a 3-turn cooldown on top of its own CannotRepeat "
            "rule). All weights are equal constants, no dynamic weights."
        ),
    ),
    "FAT_GREMLIN": MonsterMoveGraph(
        monster_id="FAT_GREMLIN",
        class_name="FatGremlin",
        initial_state="SPAWNED_MOVE",
        move_ids=["SPAWNED_MOVE", "FLEE_MOVE"],
        move_followup={
            "SPAWNED_MOVE": "FLEE_MOVE",
            "FLEE_MOVE": "FLEE_MOVE",
        },
        notes=(
            "Fully deterministic, no branch states. SPAWNED_MOVE (a stunned wake-up beat) happens "
            "once, then FLEE_MOVE loops on itself -- matches the character's gimmick of trying to "
            "escape combat every turn once awake."
        ),
    ),
    "FLAIL_KNIGHT": MonsterMoveGraph(
        monster_id="FLAIL_KNIGHT",
        class_name="FlailKnight",
        initial_state="RAM_MOVE",
        move_ids=["WAR_CHANT", "FLAIL_MOVE", "RAM_MOVE"],
        move_followup={
            "WAR_CHANT": "RAND",
            "FLAIL_MOVE": "RAND",
            "RAM_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="WAR_CHANT", weight=1.0, repeat_type="CannotRepeat"),
                MoveTarget(to="FLAIL_MOVE", weight=1.0, repeat_type="CanRepeatXTimes", max_times=2),
                MoveTarget(to="RAM_MOVE", weight=1.0, repeat_type="CanRepeatXTimes", max_times=2),
            ],
        },
        notes=(
            "Opens on RAM_MOVE, then every move loops back into a single 3-way RAND branch with "
            "equal constant weight: WAR_CHANT is CannotRepeat (never twice in a row), FLAIL_MOVE and "
            "RAM_MOVE can each repeat up to 2 times in a row (CanRepeatXTimes, maxRepeats=2) before "
            "being excluded for one pick."
        ),
    ),
}
