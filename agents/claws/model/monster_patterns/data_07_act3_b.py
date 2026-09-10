"""Hand-extracted move state machines for a batch of Act 3-ish monsters (alphabetically
Torch Head Amalgam through Zapbot).

Source: sts2-decompiled/src/Core/Models/Monsters/{TorchHeadAmalgam,ToughEgg,TrackerRubyRaider,
Tunneler,TurretOperator,TwigSlimeM,TwigSlimeS,Vantom,VineShambler,WaterfallGiant,Wriggler,
Zapbot}.cs (GenerateMoveStateMachine). Never copy decompiled source into this repo -- only short
state-id strings and weight values are captured here, no game logic or text.
"""

from __future__ import annotations

from .schema import ConditionalTarget, MonsterMoveGraph, MoveTarget

GRAPHS: dict[str, MonsterMoveGraph] = {
    "TORCH_HEAD_AMALGAM": MonsterMoveGraph(
        monster_id="TORCH_HEAD_AMALGAM",
        class_name="TorchHeadAmalgam",
        initial_state="STRONG_TACKLE_MOVE",
        move_ids=["STRONG_TACKLE_MOVE", "TACKLE_2_MOVE", "BEAM_MOVE", "TACKLE_3_MOVE", "TACKLE_4_MOVE"],
        move_followup={
            "STRONG_TACKLE_MOVE": "TACKLE_2_MOVE",
            "TACKLE_2_MOVE": "BEAM_MOVE",
            "BEAM_MOVE": "TACKLE_3_MOVE",
            "TACKLE_3_MOVE": "TACKLE_4_MOVE",
            "TACKLE_4_MOVE": "BEAM_MOVE",
        },
        notes="Fully deterministic, no branch states. Opens with STRONG_TACKLE_MOVE once, then "
              "TACKLE_2_MOVE/BEAM_MOVE, then TACKLE_3_MOVE/TACKLE_4_MOVE/BEAM_MOVE cycle forever "
              "(TACKLE_4_MOVE's followup returns to BEAM_MOVE, not back to STRONG_TACKLE_MOVE).",
    ),
    "TOUGH_EGG": MonsterMoveGraph(
        monster_id="TOUGH_EGG",
        class_name="ToughEgg",
        initial_state="HATCH_MOVE",
        move_ids=["HATCH_MOVE", "NIBBLE_MOVE"],
        move_followup={
            "HATCH_MOVE": "NIBBLE_MOVE",
            "NIBBLE_MOVE": "NIBBLE_MOVE",
        },
        notes="Fully deterministic, no branch states. Fresh spawn opens with HATCH_MOVE (a summon "
              "move that also grows the egg into a Hatchling and rerolls its HP), then loops "
              "NIBBLE_MOVE forever. A pre-hatched instance (IsHatched already true when added to "
              "the room, e.g. one spawned already-hatched by another effect) skips straight to "
              "NIBBLE_MOVE via an explicit ForceCurrentState call outside the normal followup "
              "chain -- not modeled here since a normally-spawned egg always starts unhatched.",
    ),
    "TRACKER_RUBY_RAIDER": MonsterMoveGraph(
        monster_id="TRACKER_RUBY_RAIDER",
        class_name="TrackerRubyRaider",
        initial_state="TRACK_MOVE",
        move_ids=["TRACK_MOVE", "HOUNDS_MOVE"],
        move_followup={
            "TRACK_MOVE": "HOUNDS_MOVE",
            "HOUNDS_MOVE": "HOUNDS_MOVE",
        },
        notes="Fully deterministic, no branch states. TRACK_MOVE once, then HOUNDS_MOVE forever.",
    ),
    "TUNNELER": MonsterMoveGraph(
        monster_id="TUNNELER",
        class_name="Tunneler",
        initial_state="BITE_MOVE",
        move_ids=["BITE_MOVE", "BURROW_MOVE", "BELOW_MOVE", "DIZZY_MOVE"],
        move_followup={
            "BITE_MOVE": "BURROW_MOVE",
            "BURROW_MOVE": "BELOW_MOVE",
            "BELOW_MOVE": "BELOW_MOVE",
            "DIZZY_MOVE": "BITE_MOVE",
        },
        notes="Main cycle (no branch states) is BITE_MOVE -> BURROW_MOVE -> BELOW_MOVE -> "
              "BELOW_MOVE forever. DIZZY_MOVE is never targeted by any FollowUpState in this "
              "graph -- it's only reached externally: a GetStunned() call (triggered by some "
              "outside effect while the Tunneler is burrowed) force-sets the current state to "
              "DIZZY_MOVE, which performs a no-op 'still stunned' beat and then resumes at "
              "BITE_MOVE. Treat DIZZY_MOVE as an interrupt state, not part of the normal cycle.",
    ),
    "TURRET_OPERATOR": MonsterMoveGraph(
        monster_id="TURRET_OPERATOR",
        class_name="TurretOperator",
        initial_state="UNLOAD_MOVE",
        move_ids=["UNLOAD_MOVE", "UNLOAD_MOVE_2", "RELOAD_MOVE"],
        move_followup={
            "UNLOAD_MOVE": "UNLOAD_MOVE_2",
            "UNLOAD_MOVE_2": "RELOAD_MOVE",
            "RELOAD_MOVE": "UNLOAD_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states.",
    ),
    "TWIG_SLIME_M": MonsterMoveGraph(
        monster_id="TWIG_SLIME_M",
        class_name="TwigSlimeM",
        initial_state="STICKY_SHOT_MOVE",
        move_ids=["POKEY_POUNCE_MOVE", "STICKY_SHOT_MOVE"],
        move_followup={
            "POKEY_POUNCE_MOVE": "RAND",
            "STICKY_SHOT_MOVE": "RAND",
        },
        random_branches={
            "RAND": [
                MoveTarget(to="POKEY_POUNCE_MOVE", weight=1.0, repeat_type="CanRepeatXTimes", max_times=2),
                MoveTarget(to="STICKY_SHOT_MOVE", weight=1.0, repeat_type="CannotRepeat"),
            ],
        },
        notes="Opens with STICKY_SHOT_MOVE (the machine's initial state), then every move feeds "
              "into the RAND branch: POKEY_POUNCE_MOVE can be picked at most twice in a row "
              "(CanRepeatXTimes, max 2), STICKY_SHOT_MOVE cannot repeat back-to-back. Both edges "
              "are even weight (1.0) whenever legal, so the resolver's weighted choice is over "
              "whichever of the two isn't currently blocked by its repeat rule.",
    ),
    "TWIG_SLIME_S": MonsterMoveGraph(
        monster_id="TWIG_SLIME_S",
        class_name="TwigSlimeS",
        initial_state="TACKLE_MOVE",
        move_ids=["TACKLE_MOVE"],
        move_followup={
            "TACKLE_MOVE": "TACKLE_MOVE",
        },
        notes="Single move, repeats forever. No branch states.",
    ),
    "VANTOM": MonsterMoveGraph(
        monster_id="VANTOM",
        class_name="Vantom",
        initial_state="INK_BLOT_MOVE",
        move_ids=["INK_BLOT_MOVE", "INKY_LANCE_MOVE", "DISMEMBER_MOVE", "PREPARE_MOVE"],
        move_followup={
            "INK_BLOT_MOVE": "INKY_LANCE_MOVE",
            "INKY_LANCE_MOVE": "DISMEMBER_MOVE",
            "DISMEMBER_MOVE": "PREPARE_MOVE",
            "PREPARE_MOVE": "INK_BLOT_MOVE",
        },
        notes="Fully deterministic 4-cycle, no branch states.",
    ),
    "VINE_SHAMBLER": MonsterMoveGraph(
        monster_id="VINE_SHAMBLER",
        class_name="VineShambler",
        initial_state="SWIPE_MOVE",
        move_ids=["GRASPING_VINES_MOVE", "SWIPE_MOVE", "CHOMP_MOVE"],
        move_followup={
            "SWIPE_MOVE": "GRASPING_VINES_MOVE",
            "GRASPING_VINES_MOVE": "CHOMP_MOVE",
            "CHOMP_MOVE": "SWIPE_MOVE",
        },
        notes="Fully deterministic 3-cycle, no branch states. Opens with SWIPE_MOVE (the "
              "machine's initial state), not GRASPING_VINES_MOVE.",
    ),
    "WATERFALL_GIANT": MonsterMoveGraph(
        monster_id="WATERFALL_GIANT",
        class_name="WaterfallGiant",
        initial_state="PRESSURIZE_MOVE",
        move_ids=[
            "PRESSURIZE_MOVE",
            "STOMP_MOVE",
            "RAM_MOVE",
            "SIPHON_MOVE",
            "PRESSURE_GUN_MOVE",
            "PRESSURE_UP_MOVE",
            "EXPLODE_MOVE",
            "ABOUT_TO_BLOW_MOVE",
        ],
        move_followup={
            "PRESSURIZE_MOVE": "STOMP_MOVE",
            "STOMP_MOVE": "RAM_MOVE",
            "RAM_MOVE": "SIPHON_MOVE",
            "SIPHON_MOVE": "PRESSURE_GUN_MOVE",
            "PRESSURE_GUN_MOVE": "PRESSURE_UP_MOVE",
            "PRESSURE_UP_MOVE": "STOMP_MOVE",
            "ABOUT_TO_BLOW_MOVE": "EXPLODE_MOVE",
            "EXPLODE_MOVE": "EXPLODE_MOVE",
        },
        notes="Main cycle (no branch states) is PRESSURIZE_MOVE once, then STOMP_MOVE -> "
              "RAM_MOVE -> SIPHON_MOVE -> PRESSURE_GUN_MOVE -> PRESSURE_UP_MOVE -> STOMP_MOVE "
              "forever. ABOUT_TO_BLOW_MOVE is never targeted by any FollowUpState in this graph "
              "-- it's forced externally (a TriggerAboutToBlowState() call, elsewhere gated on "
              "the giant's stacking self-buff power reaching a lethal-damage threshold) via an "
              "immediate state override, is flagged to always be performed at least once before "
              "any further transition, and then leads into EXPLODE_MOVE, which self-loops as a "
              "kill-the-giant finisher. Treat ABOUT_TO_BLOW_MOVE/EXPLODE_MOVE as a one-way "
              "end-of-fight branch reached by an external HP/stack trigger, not by normal move "
              "resolution.",
    ),
    "WRIGGLER": MonsterMoveGraph(
        monster_id="WRIGGLER",
        class_name="Wriggler",
        initial_state="INIT_MOVE",
        move_ids=["NASTY_BITE_MOVE", "WRIGGLE_MOVE", "SPAWNED_MOVE"],
        move_followup={
            "NASTY_BITE_MOVE": "WRIGGLE_MOVE",
            "WRIGGLE_MOVE": "NASTY_BITE_MOVE",
            "SPAWNED_MOVE": "INIT_MOVE",
        },
        conditional_branches={
            "INIT_MOVE": [
                ConditionalTarget(to="NASTY_BITE_MOVE", condition_expr="this creature occupies encounter slot 'wriggler1'"),
                ConditionalTarget(to="WRIGGLE_MOVE", condition_expr="this creature occupies encounter slot 'wriggler2'"),
                ConditionalTarget(to="NASTY_BITE_MOVE", condition_expr="this creature occupies encounter slot 'wriggler3'"),
                ConditionalTarget(to="WRIGGLE_MOVE", condition_expr="this creature occupies encounter slot 'wriggler4'"),
                ConditionalTarget(to="NASTY_BITE_MOVE", condition_expr=None),
            ],
        },
        notes="Wriggler spawns as one of several identical adds in a swarm encounter; its very "
              "first move is picked purely from which numbered slot ('wriggler1'..'wriggler4') "
              "it was placed into -- odd slots open with NASTY_BITE_MOVE, even slots open with "
              "WRIGGLE_MOVE -- after which the two alternate forever regardless of slot. In the "
              "decompiled source there is NO trailing unconditional else on INIT_MOVE -- the real "
              "ConditionalBranchState.GetNextState() throws if none of the four slot checks match, "
              "because the encounter design guarantees exactly slots 'wriggler1'..'wriggler4' are "
              "used and no other case is reachable. The 5th edge above (condition_expr=None, "
              "defaulting to NASTY_BITE_MOVE) is NOT in the source -- it's a synthetic fallback "
              "added only so resolve_next_move_distribution stays well-defined when no slot-name "
              "context is supplied (this schema has no field for encounter-slot identity); a real "
              "resolver call should pass slot name via `context` and a DYNAMIC_CONDITIONS entry "
              "for the four real edges so this synthetic one never actually fires. If a fresh "
              "instance is spawned with StartStunned set (only done by "
              "special summon effects, not a normal encounter placement), the machine instead "
              "starts at SPAWNED_MOVE, a one-turn do-nothing stun beat that then falls through to "
              "INIT_MOVE's slot check above -- not modeled as the initial_state here since it "
              "isn't the default for a normally-placed Wriggler.",
    ),
    "ZAPBOT": MonsterMoveGraph(
        monster_id="ZAPBOT",
        class_name="Zapbot",
        initial_state="ZAP",
        move_ids=["ZAP"],
        move_followup={
            "ZAP": "ZAP",
        },
        notes="Single move, repeats forever. No branch states.",
    ),
}
