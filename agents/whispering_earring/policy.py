"""Non-combat (meta) decisions delegate to the Choices Paradox placeholder policy (agents/choices_paradox/policy.py), which
scores cards off Baalorlord's STS2 tier lists instead of picking blindly
Map_select likewise delegates to Golden Compass (agents/golden_compass/
policy.py), which value-to-go DPs the full map instead of picking a reachable node blindly.

Returns (action_name, kwargs, policy) rather than a built command, so callers pass the first
two straight into session.step(name, **kwargs) -- the session builds/validates the actual
command via actions.action(), no need to duplicate that here. `

_choose_combat_action_claws mixes AlphaZero-style Dirichlet noise into that softmax before
sampling (see _DIRICHLET_EPS/_DIRICHLET_ALPHA_SCALE below) rather than reintroducing search --
without it, an option whose probability drifts near 0 is rarely/never one of
branches_per_turn's on-policy samples, so it never appears in new training data and its
probability estimate can never correct itself. Confirmed empirically (debug_card_bias.py) on
an overnight training.sh run: over a dozen legal Ironclad cards sat at exactly 0.0 average
probability across hundreds of sightings despite being playable. training.sh loops forever
with --init resuming the previous round's checkpoint, so this lockout compounds round over
round instead of resetting.
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the import below)
from env import Sts2Session
from models import CombatPlay, Decision
import buffer_layout as bl
from buffer_env import BufferDecision

_CLAWS_CHECKPOINT = os.environ.get("CLAWS_CHECKPOINT")
_claws_net = None  # lazy-loaded once per worker process, on first combat decision -- see below

_DIRICHLET_EPS = 0.15  # fraction of root probability mass replaced by noise -- AlphaZero's own
# self-play default
_DIRICHLET_ALPHA_SCALE = 10.0  # concentration = this / n_legal_options, the same "~10 / average
# legal move count" convention AlphaZero-style implementations use so alpha adapts to how many
# options are actually on offer instead of a single fixed value across very different option
# counts (end_turn-only vs. a full hand + potions)

_choices_paradox_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "policy.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_policy", _choices_paradox_path)
assert _spec is not None and _spec.loader is not None
choices_paradox = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(choices_paradox)

_golden_compass_path = Path(__file__).resolve().parents[1] / "golden_compass" / "policy.py"
_spec = importlib.util.spec_from_file_location("golden_compass_policy", _golden_compass_path)
assert _spec is not None and _spec.loader is not None
golden_compass = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden_compass)

_REST_HEAL_HP_THRESHOLD = 35
_pending_card_select_purpose: str | None = None

def choose_action(
    decision: Decision | BufferDecision, session: Sts2Session
) -> tuple[str, dict, dict | None]:
    if decision.decision == "combat_play":
        return _choose_combat_action(decision, session)
    name, kwargs = _choose_generic_action(decision, session)
    return name, kwargs, None


def _load_claws_net():
    global _claws_net
    if _claws_net is not None or not _CLAWS_CHECKPOINT:
        return _claws_net

    claws_model_dir = Path(__file__).resolve().parents[1] / "claws" / "model"
    if str(claws_model_dir) not in sys.path:
        sys.path.insert(0, str(claws_model_dir))
    from model import ClawsNet  # noqa: E402  (path just inserted above)

    _claws_net = ClawsNet.from_checkpoint(_CLAWS_CHECKPOINT)
    return _claws_net


def get_claws_net():
    """Public accessor for facility.py's TD-bootstrap value term (_play_turn_only) -- reuses
    this module's lazily-loaded ClawsNet (see _load_claws_net) rather than a second load in
    the same branch-worker process. Returns None if CLAWS_CHECKPOINT isn't set (no checkpoint
    yet -- round 1)."""
    return _load_claws_net()


def _choose_combat_action_claws(decision: CombatPlay, net) -> tuple[str, dict, dict]:
    """Samples (not argmax) from the network's own softmax over model.py's legal_options(),
    mixed with root Dirichlet noise -- see this module's docstring for why."""
    import torch

    options, logits, _value = net.forward(decision)
    probs = torch.softmax(logits, dim=-1)
    alpha = _DIRICHLET_ALPHA_SCALE / len(options)
    noise = torch.distributions.Dirichlet(torch.full((len(options),), alpha)).sample()
    probs = (1 - _DIRICHLET_EPS) * probs + _DIRICHLET_EPS * noise
    idx = int(torch.multinomial(probs, 1).item())
    policy = [
        {"action": a, "args": args, "prob": p} for (a, args), p in zip(options, probs.tolist())
    ]
    name, args = options[idx]
    return name, args, policy


def _choose_combat_action(
    decision: CombatPlay | BufferDecision, session: Sts2Session
) -> tuple[str, dict, dict]:
    net = _load_claws_net()
    if net is not None:
        return _choose_combat_action_claws(decision, net)

    options = (
        _combat_options_from_buffer(decision.arr)
        if isinstance(decision, BufferDecision)
        else _combat_options_from_decision(decision)
    )

    chosen = random.choice(options)
    # Placeholder: a one-hot weight on the option uniform-random actually picked, standing
    # in for an MCTS visit distribution over the same option list. A list of records rather
    # than a dict keyed by action, since kwargs (e.g. target_index) can differ between two
    # options that share an action name.
    policy = [
        {"action": name, "args": args, "prob": 1.0 if (name, args) == chosen else 0.0}
        for name, args in options
    ]
    name, args = chosen
    return name, args, policy


def _combat_options_from_decision(decision: CombatPlay) -> list[tuple[str, dict]]:
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

    return options


def _combat_options_from_buffer(arr) -> list[tuple[str, dict]]:
    """Buffer counterpart to _combat_options_from_decision -- same single-random-target-per-card
    placeholder policy (see this module's docstring), read via buffer_layout instead of a
    pydantic CombatPlay's attribute tree (see sts2-cli's buffer_env.py/buffer_layout.py)."""
    options: list[tuple[str, dict]] = [("end_turn", {})]
    en = bl.energy(arr)
    enemy_indices = [i for i, _ in bl.enemies(arr)]

    for idx, slot in bl.hand_cards(arr):
        if not bl.hand_can_play(slot) or int(slot[bl.CARD_COST]) > en:
            continue
        args = {"card_index": idx}
        if bl.hand_target_type(slot) == "AnyEnemy" and len(enemy_indices) >= 2:
            args["target_index"] = random.choice(enemy_indices)
        options.append(("play_card", args))

    for idx, slot in bl.player_potions(arr):
        args = {"potion_index": idx}
        target_type = bl.POTION_TARGET_TYPES.get(int(slot[bl.POTION_TARGET_TYPE]))
        if target_type == "AnyEnemy" and len(enemy_indices) >= 2:
            args["target_index"] = random.choice(enemy_indices)
        options.append(("use_potion", args))

    return options


def _choose_generic_action(decision: Decision, session: Sts2Session) -> tuple[str, dict]:
    """Non-combat decisions: pick a uniformly random legal option, or the simplest safe
    default (leave/skip) where "random" isn't a meaningful choice yet. Good enough to
    keep a run moving forward -- combat_play is what's actually being trained right now."""
    dec = decision.decision

    if dec == "map_select":
        return _choose_map_action(decision, session)

    if dec == "card_reward":
        index = choices_paradox.choose_card_reward(
            decision.cards, decision.can_skip, decision.player.deck_size or 0
        )
        if index is None:
            return "skip_card_reward", {}
        return "select_card_reward", {"card_index": index}

    if dec == "bundle_select":
        return "select_bundle", {"bundle_index": random.choice(decision.bundles).index}

    if dec == "card_select":
        return _choose_card_select_action(decision)

    if dec == "shop":
        return _choose_shop_action(decision)

    if dec == "rest_site":
        return _choose_rest_site_action(decision)

    if dec == "event_choice":
        unlocked = [o for o in decision.options if not o.is_locked]
        if not unlocked:
            return "leave_room", {}
        return "choose_option", {"option_index": random.choice(unlocked).index}

    raise ValueError(f"no policy for decision type {dec!r}")


def _choose_map_action(decision: Decision, session: Sts2Session) -> tuple[str, dict]:
    """Delegates to Golden Compass's value-to-go DP over the full act map """
    hp_frac = decision.player.hp / decision.player.max_hp if decision.player.max_hp else 1.0
    target = golden_compass.choose_next_node(session.get_map(), hp_frac=hp_frac)
    if target is not None:
        for choice in decision.choices:
            if (choice.col, choice.row) == target:
                return "select_map_node", {"col": choice.col, "row": choice.row}

    pick = random.choice(decision.choices)
    return "select_map_node", {"col": pick.col, "row": pick.row}


def _choose_card_select_action(decision: Decision | BufferDecision) -> tuple[str, dict]:
    """Dispatches on _pending_card_select_purpose, set by whichever shop/rest_site action
    just triggered this follow-up. Anything else (True Grit's exhaust-select, Wish's
    deck-tutor-select, ...) falls back to a uniformly random legal pick -- CardSelect's shape
    alone can't tell those apart from a remove/upgrade select, and no purpose was staged for
    them.

    A BufferDecision only ever reaches here from within _play_turn_only's loop (a mid-combat
    card_select -- Power Potion, Brand, ...); shop/rest_site (the only actions that ever set
    _pending_card_select_purpose) run on the main line's own pydantic-decision session, never a
    buffer one, so purpose is always None on this path -- no need for a buffer-aware
    choices_paradox.choose_removal_target/choose_upgrade_target call."""
    global _pending_card_select_purpose
    purpose, _pending_card_select_purpose = _pending_card_select_purpose, None

    if isinstance(decision, BufferDecision):
        arr = decision.arr
        indices = [i for i, _ in bl.select_cards(arr)]
        lo, hi = bl.min_select(arr), min(bl.max_select(arr), len(indices))
        n_pick = random.randint(lo, hi) if hi >= lo else 0
        if n_pick == 0:
            return "skip_select", {}
        return "select_cards", {"indices": random.sample(indices, n_pick)}

    if purpose == "remove" and decision.cards:
        index = choices_paradox.choose_removal_target(decision.cards)
        return "select_cards", {"indices": [index]}

    if purpose == "upgrade" and decision.cards:
        index = choices_paradox.choose_upgrade_target(decision.cards)
        return "select_cards", {"indices": [index]}

    lo, hi = decision.min_select, min(decision.max_select, len(decision.cards))
    n_pick = random.randint(lo, hi) if hi >= lo else 0
    if n_pick == 0:
        return "skip_select", {}
    picks = random.sample(decision.cards, n_pick)
    return "select_cards", {"indices": [c.index for c in picks]}


def _choose_shop_action(decision: Decision) -> tuple[str, dict]:
    global _pending_card_select_purpose
    gold = decision.player.gold

    if decision.card_removal_cost is not None and decision.card_removal_cost <= gold:
        _pending_card_select_purpose = "remove"
        return "remove_card", {}

    if len(decision.player.potions) < decision.player.max_potions:
        affordable_potions = [p for p in decision.potions if p.is_stocked and p.cost <= gold]
        if affordable_potions:
            return "buy_potion", {"potion_index": random.choice(affordable_potions).index}

    affordable_cards = [c for c in decision.cards if c.is_stocked and c.cost <= gold]
    if affordable_cards:
        index = choices_paradox.choose_shop_card(affordable_cards, decision.player.deck_size or 0)
        if index is not None:
            return "buy_card", {"card_index": index}

    return "leave_room", {}


def _choose_rest_site_action(decision: Decision) -> tuple[str, dict]:
    global _pending_card_select_purpose
    enabled = {o.option_id: o for o in decision.options if o.is_enabled}
    if not enabled:
        return "leave_room", {}

    preferred = "HEAL" if decision.player.hp < _REST_HEAL_HP_THRESHOLD else "SMITH"
    option = enabled[preferred] if preferred in enabled else random.choice(list(enabled.values()))
    if option.option_id == "SMITH":
        _pending_card_select_purpose = "upgrade"
    return "choose_option", {"option_index": option.index}
