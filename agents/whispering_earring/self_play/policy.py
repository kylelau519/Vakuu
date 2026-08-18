"""Uniform-random exploration policy -- the simplest possible action selection, used for
the first self-play pass before any trained network exists. No scoring, no candidate
evaluation: every legal action at a decision point is equally likely to be picked.

Returns (action_name, kwargs) rather than a built command, so callers pass it straight
into Sts2Session.step(*choose_action(decision)) -- the session builds/validates the
actual command via actions.action(), no need to duplicate that here.
"""
from __future__ import annotations

import random

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the import below)
from models import CombatPlay, Decision


def choose_action(decision: Decision) -> tuple[str, dict]:
    if isinstance(decision, CombatPlay):
        return _choose_combat_action(decision)
    return _choose_generic_action(decision)


def _choose_combat_action(decision: CombatPlay) -> tuple[str, dict]:
    options: list[tuple[str, dict]] = [("end_turn", {})]

    for card in decision.hand:
        if not card.can_play or card.cost > decision.energy:
            continue
        args = {"card_index": card.index}
        if card.target_type == "AnyEnemy" and len(decision.enemies) >= 2:
            args["target_index"] = random.choice(decision.enemies).index
        options.append(("play_card", args))

    for potion in decision.player.potions:
        args = {"potion_index": potion.index}
        if potion.target_type == "AnyEnemy" and len(decision.enemies) >= 2:
            args["target_index"] = random.choice(decision.enemies).index
        options.append(("use_potion", args))

    return random.choice(options)


def _choose_generic_action(decision: Decision) -> tuple[str, dict]:
    """Non-combat decisions: pick a uniformly random legal option, or the simplest safe
    default (leave/skip) where "random" isn't a meaningful choice yet. Good enough to
    keep a run moving forward -- combat_play is what's actually being trained right now."""
    dec = decision.decision

    if dec == "map_select":
        pick = random.choice(decision.choices)
        return "select_map_node", {"col": pick.col, "row": pick.row}

    if dec == "card_reward":
        if not decision.cards:
            return "skip_card_reward", {}
        if decision.can_skip and random.random() < 0.3:
            return "skip_card_reward", {}
        return "select_card_reward", {"card_index": random.choice(decision.cards).index}

    if dec == "bundle_select":
        return "select_bundle", {"bundle_index": random.choice(decision.bundles).index}

    if dec == "card_select":
        lo, hi = decision.min_select, min(decision.max_select, len(decision.cards))
        n_pick = random.randint(lo, hi) if hi >= lo else 0
        if n_pick == 0:
            return "skip_select", {}
        picks = random.sample(decision.cards, n_pick)
        return "select_cards", {"indices": [c.index for c in picks]}

    if dec == "shop":
        return "leave_room", {}

    if dec == "rest_site":
        enabled = [o for o in decision.options if o.is_enabled]
        if not enabled:
            return "leave_room", {}
        return "choose_option", {"option_index": random.choice(enabled).index}

    if dec == "event_choice":
        unlocked = [o for o in decision.options if not o.is_locked]
        if not unlocked:
            return "leave_room", {}
        return "choose_option", {"option_index": random.choice(unlocked).index}

    raise ValueError(f"no policy for decision type {dec!r}")
