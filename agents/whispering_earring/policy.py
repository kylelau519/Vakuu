from __future__ import annotations

import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the import below)
from env import Sts2Session
from models import CombatPlay, Decision
import buffer_layout as bl
from buffer_env import BufferDecision, decode_token
import torch

_CLAWS_CHECKPOINT = os.environ.get("CLAWS_CHECKPOINT")
_claws_net = None  # lazy-loaded once per worker process, on first combat decision -- see below

_DIRICHLET_EPS = 0.15  # fraction of root probability mass replaced by noise -- AlphaZero's own
_COMBAT_DIRICHLET_EPS = 0.25
_DIRICHLET_ALPHA_SCALE = 10.0  # concentration = this / n_legal_options

# Loading choices_paradox.policy
_choices_paradox_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "policy.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_policy", _choices_paradox_path)
assert _spec is not None and _spec.loader is not None
choices_paradox = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(choices_paradox)

# Loading golden_compass.policy
_golden_compass_path = Path(__file__).resolve().parents[1] / "golden_compass" / "policy.py"
_spec = importlib.util.spec_from_file_location("golden_compass_policy", _golden_compass_path)
assert _spec is not None and _spec.loader is not None
golden_compass = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden_compass)

# Loading golden_compass.model.model.py and choices_paradox.model.model.py 
_golden_compass_model_path = (
    Path(__file__).resolve().parents[1] / "golden_compass" / "model" / "model.py"
)
_spec = importlib.util.spec_from_file_location("golden_compass_model", _golden_compass_model_path)
assert _spec is not None and _spec.loader is not None
_gc_model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _gc_model  
_spec.loader.exec_module(_gc_model)


# Loading choices_paradox.model.model.py
_choices_paradox_model_path = (
    Path(__file__).resolve().parents[1] / "choices_paradox" / "model" / "model.py"
)
_spec = importlib.util.spec_from_file_location("choices_paradox_model", _choices_paradox_model_path)
assert _spec is not None and _spec.loader is not None
_cp_model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cp_model 
_spec.loader.exec_module(_cp_model)

_CHOICES_PARADOX_CHECKPOINT = os.environ.get("CHOICES_PARADOX_CHECKPOINT")
_choices_paradox_net = None  # lazy-loaded once per worker process, on first card_reward decision

_GOLDEN_COMPASS_CHECKPOINT = os.environ.get("GOLDEN_COMPASS_CHECKPOINT")
_golden_compass_net = None  # lazy-loaded once per worker process, on first map_select decision

_REST_HEAL_HP_THRESHOLD = 35


@dataclass
class _BufferCard:
    index: int
    id: str | None

def choose_action(
    decision: Decision | BufferDecision, session: Sts2Session, purpose: str | None = None,
    combat_dirichlet_noise: bool = True, card_reward_dirichlet_noise: bool = True,
) -> tuple[str, dict, dict | None, str | None]:
    if decision.decision == "combat_play":
        name, args, policy = _choose_combat_action(decision, session, combat_dirichlet_noise)
        return name, args, policy, None
    name, kwargs, next_purpose, policy = _choose_generic_action(
        decision, session, purpose, card_reward_dirichlet_noise
    )
    return name, kwargs, policy, next_purpose


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
    return _load_claws_net()


def _load_choices_paradox_net():
    global _choices_paradox_net
    if _choices_paradox_net is not None or not _CHOICES_PARADOX_CHECKPOINT:
        return _choices_paradox_net
    if not os.path.exists(_CHOICES_PARADOX_CHECKPOINT):
        return None  # round 1, or CHOICES_PARADOX_CHECKPOINT unset by the training script yet
    _choices_paradox_net = _cp_model.ChoicesParadoxNet.from_checkpoint(_CHOICES_PARADOX_CHECKPOINT)
    return _choices_paradox_net


def _load_golden_compass_net():
    global _golden_compass_net
    if _golden_compass_net is not None or not _GOLDEN_COMPASS_CHECKPOINT:
        return _golden_compass_net
    if not os.path.exists(_GOLDEN_COMPASS_CHECKPOINT):
        return None  # round 1, or GOLDEN_COMPASS_CHECKPOINT unset by the training script yet
    _golden_compass_net = _gc_model.GoldenCompassNet.from_checkpoint(_GOLDEN_COMPASS_CHECKPOINT)
    return _golden_compass_net


def _choose_map_action_net(
    decision, session: Sts2Session, net, dirichlet_noise: bool = True,
) -> tuple[str, dict, dict]:
    """ Samples (not argmax) from GoldenCompassNet's own softmax for the best room combinations"""

    combos, logits, _value, combo_paths = net.forward(decision, session)
    probs = torch.softmax(logits, dim=-1)
    if dirichlet_noise:
        alpha = _DIRICHLET_ALPHA_SCALE / len(combos)
        noise = torch.distributions.Dirichlet(torch.full((len(combos),), alpha)).sample()
        probs = (1 - _DIRICHLET_EPS) * probs + _DIRICHLET_EPS * noise
    idx = int(torch.multinomial(probs, 1).item())
    policy = {
        "combinations": [
            {"combo": list(c), "prob": p} for c, p in zip(combos, probs.tolist())
        ],
        "chosen": idx,
    }
    name, args = _gc_model.resolve_combination_action(combos[idx], combo_paths)
    return name, args, policy


def _choose_card_reward_net(decision, net, dirichlet_noise: bool = True) -> tuple[str, dict]:
    """Samples from ChoicesParadoxNet's own softmax for the best card_reward option, mixed with root Dirichlet"""

    options, logits, _value = net.forward(decision, "card_reward")
    probs = torch.softmax(logits, dim=-1)
    if dirichlet_noise:
        alpha = _DIRICHLET_ALPHA_SCALE / len(options)
        noise = torch.distributions.Dirichlet(torch.full((len(options),), alpha)).sample()
        probs = (1 - _DIRICHLET_EPS) * probs + _DIRICHLET_EPS * noise
    idx = int(torch.multinomial(probs, 1).item())
    return options[idx]


def _choose_card_reward_decision(
    decision, cards: list, can_skip: bool, deck_size: int,
    card_reward_dirichlet_noise: bool = True,
) -> tuple[str, dict]:
    """card_reward's action chooser with a fallback to choices_paradox's own heuristic if no trained net is loaded."""
    net = _load_choices_paradox_net()
    if net is not None:
        return _choose_card_reward_net(decision, net, card_reward_dirichlet_noise)

    index = choices_paradox.choose_card_reward(cards, can_skip, deck_size)
    if index is None:
        return "skip_card_reward", {}
    return "select_card_reward", {"card_index": index}


def _choose_combat_action_claws(
    decision: CombatPlay, net, combat_dirichlet_noise: bool = True
) -> tuple[str, dict, dict]:
    """Samples from ClawsNet's own softmax for the best combat action, mixed with root Dirichlet noise """

    options, logits, _value = net.forward(decision)
    probs = torch.softmax(logits, dim=-1)
    if combat_dirichlet_noise:
        alpha = _DIRICHLET_ALPHA_SCALE / len(options)
        noise = torch.distributions.Dirichlet(torch.full((len(options),), alpha)).sample()
        probs = (1 - _COMBAT_DIRICHLET_EPS) * probs + _COMBAT_DIRICHLET_EPS * noise
    idx = int(torch.multinomial(probs, 1).item())
    policy = [
        {"action": a, "args": args, "prob": p} for (a, args), p in zip(options, probs.tolist())
    ]
    name, args = options[idx]
    return name, args, policy


def _choose_combat_action(
    decision: CombatPlay | BufferDecision, session: Sts2Session,
    combat_dirichlet_noise: bool = True,
) -> tuple[str, dict, dict]:
    """Samples a combat action from a trained ClawsNet if CLAWS_CHECKPOINT is set, falling back to
    a uniform-random placeholder policy otherwise"""
    net = _load_claws_net()
    if net is not None:
        return _choose_combat_action_claws(decision, net, combat_dirichlet_noise)

    options = (
        _combat_options_from_buffer(decision.arr)
        if isinstance(decision, BufferDecision)
        else _combat_options_from_decision(decision)
    )

    chosen = random.choice(options)
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
    """Buffer counterpart to _combat_options_from_decision"""
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


def _choose_generic_action(
    decision: Decision, session: Sts2Session, purpose: str | None,
    card_reward_dirichlet_noise: bool = True,
) -> tuple[str, dict, str | None, dict | None]:
    if isinstance(decision, BufferDecision):
        return _choose_generic_action_from_buffer(decision, purpose, card_reward_dirichlet_noise)

    dec = decision.decision

    if dec == "map_select":
        name, kwargs, policy = _choose_map_action(decision, session)
        return name, kwargs, None, policy

    if dec == "card_reward":
        name, kwargs = _choose_card_reward_decision(
            decision, decision.cards, decision.can_skip, decision.player.deck_size or 0,
            card_reward_dirichlet_noise,
        )
        return name, kwargs, None, None

    if dec == "bundle_select":
        return "select_bundle", {"bundle_index": random.choice(decision.bundles).index}, None, None

    if dec == "card_select":
        name, kwargs = _choose_card_select_action(decision, purpose)
        return name, kwargs, None, None

    if dec == "shop":
        name, kwargs, next_purpose = _choose_shop_action(decision)
        return name, kwargs, next_purpose, None

    if dec == "rest_site":
        name, kwargs, next_purpose = _choose_rest_site_action(decision)
        return name, kwargs, next_purpose, None

    if dec == "event_choice":
        unlocked = [o for o in decision.options if not o.is_locked]
        if not unlocked:
            return "leave_room", {}, None, None
        return "choose_option", {"option_index": random.choice(unlocked).index}, None, None

    raise ValueError(f"no policy for decision type {dec!r}")


def _choose_generic_action_from_buffer(
    decision: BufferDecision, purpose: str | None, card_reward_dirichlet_noise: bool = True
) -> tuple[str, dict, str | None, dict | None]:
    """Buffer counterpart to _choose_generic_action. """
    dec = decision.decision
    arr = decision.arr

    if dec == "map_select":
        name, kwargs = _choose_map_action_from_buffer(arr)
        return name, kwargs, None, None
    if dec in ("card_reward", "event_card_reward"):
        name, kwargs = _choose_card_reward_from_buffer(decision, card_reward_dirichlet_noise)
        return name, kwargs, None, None
    if dec == "bundle_select":
        return (
            "select_bundle", {"bundle_index": random.choice([b for b, _ in bl.bundles(arr)])},
            None, None,
        )
    if dec == "card_select":
        name, kwargs = _choose_card_select_action(decision, purpose)
        return name, kwargs, None, None
    if dec == "shop":
        name, kwargs, next_purpose = _choose_shop_action_from_buffer(arr)
        return name, kwargs, next_purpose, None
    if dec == "rest_site":
        name, kwargs, next_purpose = _choose_rest_site_action_from_buffer(arr)
        return name, kwargs, next_purpose, None
    if dec == "event_choice":
        unlocked = [i for i, is_locked in bl.event_options(arr) if not is_locked]
        if not unlocked:
            return "leave_room", {}, None, None
        return "choose_option", {"option_index": random.choice(unlocked)}, None, None

    raise ValueError(f"no buffer-side policy for decision type {dec!r}")


def _card_reward_options_from_buffer(decision: BufferDecision) -> tuple[list[_BufferCard], bool]:
    """Decodes both buffer decision kinds a "card_reward" pydantic-side decision can actually be:
    the ordinary post-combat CARD_REWARD section, or the blessing-triggered EVENT_CARD_REWARD one."""
    arr = decision.arr
    if bl.decision_name(arr) == "event_card_reward":
        cards = [
            _BufferCard(i, decode_token(bl.card_token(slot)))
            for i, slot in bl.event_card_reward_cards(arr)
        ]
        return cards, True
    cards = [_BufferCard(i, decode_token(bl.card_token(slot))) for i, slot in bl.reward_cards(arr)]
    return cards, bl.can_skip_reward(arr)


def _choose_card_reward_from_buffer(
    decision: BufferDecision, card_reward_dirichlet_noise: bool = True
) -> tuple[str, dict]:
    cards, can_skip = _card_reward_options_from_buffer(decision)
    return _choose_card_reward_decision(
        decision, cards, can_skip, bl.player_deck_size(decision.arr), card_reward_dirichlet_noise
    )


def _choose_card_reward_heuristic(decision: BufferDecision) -> tuple[str, dict]:
    """Test-only counterpart to _choose_card_reward_from_buffer: skips the trained-checkpoint
    path entirely and always resolves through choices_paradox's tier list."""
    cards, can_skip = _card_reward_options_from_buffer(decision)
    index = choices_paradox.choose_card_reward(cards, can_skip, bl.player_deck_size(decision.arr))
    if index is None:
        return "skip_card_reward", {}
    return "select_card_reward", {"card_index": index}


def _choose_shop_action_from_buffer(arr) -> tuple[str, dict, str | None]:
    gold = bl.player_gold(arr)

    removal_cost = bl.shop_removal_cost(arr)
    if removal_cost != -1 and removal_cost <= gold:
        return "remove_card", {}, "remove"

    if len(list(bl.player_potions(arr))) < bl.player_max_potions(arr):
        affordable_potions = [i for i, slot in bl.shop_potions(arr) if bool(slot[2]) and slot[1] <= gold]
        if affordable_potions:
            return "buy_potion", {"potion_index": random.choice(affordable_potions)}, None

    cards = [
        _BufferCard(i, decode_token(bl.card_token(slot)))
        for i, slot in bl.shop_cards(arr)
        if bool(slot[bl.CARD_SIZE + 1]) and slot[bl.CARD_SIZE] <= gold  # is_stocked, cost
    ]
    if cards:
        index = choices_paradox.choose_shop_card(cards, bl.player_deck_size(arr))
        if index is not None:
            return "buy_card", {"card_index": index}, None

    return "leave_room", {}, None


def _choose_rest_site_action_from_buffer(arr) -> tuple[str, dict, str | None]:
    enabled = [i for i, slot in bl.rest_options(arr) if int(slot[1]) == 1]
    if not enabled:
        return "leave_room", {}, None
    return "choose_option", {"option_index": random.choice(enabled)}, None


def _choose_map_action(
    decision: Decision, session: Sts2Session, map_dirichlet_noise: bool = True,
) -> tuple[str, dict, dict | None]:
    """Samples from a trained GoldenCompassNet, fallback to a heuristic if no net is loaded. """
    net = _load_golden_compass_net()
    if net is not None:
        name, kwargs, policy = _choose_map_action_net(decision, session, net, map_dirichlet_noise)
        return name, kwargs, policy

    hp_frac = decision.player.hp / decision.player.max_hp if decision.player.max_hp else 1.0
    target = golden_compass.choose_next_node(session.get_map(), hp_frac=hp_frac)
    if target is not None:
        for choice in decision.choices:
            if (choice.col, choice.row) == target:
                return "select_map_node", {"col": choice.col, "row": choice.row}, None

    pick = random.choice(decision.choices)
    return "select_map_node", {"col": pick.col, "row": pick.row}, None


def _choose_card_select_action(
    decision: Decision | BufferDecision, purpose: str | None
) -> tuple[str, dict]:
    """purpose comes from the caller -- whichever shop/rest_site action just triggered this follow-up.
    Anything else (True Grit's exhaust-select, Wish's deck-tutor-
    select, ...) falls back to a uniformly random legal pick."""
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


def _choose_shop_action(decision: Decision) -> tuple[str, dict, str | None]:
    gold = decision.player.gold

    if decision.card_removal_cost is not None and decision.card_removal_cost <= gold:
        return "remove_card", {}, "remove"

    if len(decision.player.potions) < decision.player.max_potions:
        affordable_potions = [p for p in decision.potions if p.is_stocked and p.cost <= gold]
        if affordable_potions:
            return "buy_potion", {"potion_index": random.choice(affordable_potions).index}, None

    affordable_cards = [c for c in decision.cards if c.is_stocked and c.cost <= gold]
    if affordable_cards:
        index = choices_paradox.choose_shop_card(affordable_cards, decision.player.deck_size or 0)
        if index is not None:
            return "buy_card", {"card_index": index}, None

    return "leave_room", {}, None


def _choose_rest_site_action(decision: Decision) -> tuple[str, dict, str | None]:
    enabled = {o.option_id: o for o in decision.options if o.is_enabled}
    if not enabled:
        return "leave_room", {}, None

    preferred = "HEAL" if decision.player.hp < _REST_HEAL_HP_THRESHOLD else "SMITH"
    option = enabled[preferred] if preferred in enabled else random.choice(list(enabled.values()))
    purpose = "upgrade" if option.option_id == "SMITH" else None
    return "choose_option", {"option_index": option.index}, purpose


##### Generic action chooser for testing: BufferDecision only, no checkpoint loading anywhere --
##### map_select and card_reward resolve through golden_compass's/choices_paradox's plain
##### heuristic policies instead of the pydantic-side fallbacks' random walk / net-first logic.

def choose_generic_action(
    decision: BufferDecision, purpose: str | None = None,
) -> tuple[str, dict, str | None, dict | None]:
    dec = decision.decision
    arr = decision.arr

    if dec == "combat_play":
        name, kwargs = random.choice(_combat_options_from_buffer(arr))
        return name, kwargs, None, None
    if dec == "map_select":
        name, kwargs = _choose_map_action_heuristic(arr)
        return name, kwargs, None, None
    if dec in ("card_reward", "event_card_reward"):
        name, kwargs = _choose_card_reward_heuristic(decision)
        return name, kwargs, None, None
    if dec == "bundle_select":
        return (
            "select_bundle", {"bundle_index": random.choice([b for b, _ in bl.bundles(arr)])},
            None, None,
        )
    if dec == "card_select":
        name, kwargs = _choose_card_select_action(decision, purpose)
        return name, kwargs, None, None
    if dec == "shop":
        name, kwargs, next_purpose = _choose_shop_action_from_buffer(arr)
        return name, kwargs, next_purpose, None
    if dec == "rest_site":
        name, kwargs, next_purpose = _choose_rest_site_action_from_buffer(arr)
        return name, kwargs, next_purpose, None
    if dec == "event_choice":
        unlocked = [i for i, is_locked in bl.event_options(arr) if not is_locked]
        if not unlocked:
            return "leave_room", {}, None, None
        return "choose_option", {"option_index": random.choice(unlocked)}, None, None

    raise ValueError(f"no buffer-side policy for decision type {dec!r}")


def _map_json_from_buffer(arr) -> dict:
    """Reassembles golden_compass's expected {"rows", "boss", "current_coord"} map shape out of
    the buffer's raw grid encoding, so its tier-weighted DP (choose_next_node) runs unmodified
    against buffer decisions. The trailing boss MapPoint slot carries no coordinate of its own
    (RunSimulator.cs's Map() writes it as a bare extra slot past the grid) -- its (col, row) is
    only ever referenced as a child of the last real row's nodes, so it's recovered here as the
    one child coordinate that isn't itself a real grid node."""
    rows_by_row: dict[int, list[dict]] = {}
    current_coord = None
    boss_type = None

    for row, col, slot in bl.map_points(arr):
        if row == -1:  # trailing boss slot
            boss_type = bl.map_point_type(slot)
            continue
        node = {
            "col": col, "row": row, "type": bl.map_point_type(slot),
            "children": [{"col": c, "row": r} for c, r in bl.map_point_children(slot)],
        }
        rows_by_row.setdefault(row, []).append(node)
        if int(slot[2]) == 1:  # MAP_POINT_SIZE layout: type, visited, current, then children
            current_coord = {"col": col, "row": row}

    grid_coords = {(n["col"], n["row"]) for nodes in rows_by_row.values() for n in nodes}
    boss_coord = next(
        (c["col"], c["row"])
        for nodes in rows_by_row.values() for n in nodes for c in n["children"]
        if (c["col"], c["row"]) not in grid_coords
    )
    boss = {"col": boss_coord[0], "row": boss_coord[1], "type": boss_type}
    rows = [rows_by_row[r] for r in sorted(rows_by_row)]
    return {"rows": rows, "boss": boss, "current_coord": current_coord}


def _choose_map_action_heuristic(arr) -> tuple[str, dict]:
    """Test-only counterpart to _choose_map_action_from_buffer: skips GoldenCompassNet entirely
    and always resolves through golden_compass's node-value DP instead of a uniform random walk."""
    max_hp = bl.player_max_hp(arr)
    hp_frac = bl.player_hp(arr) / max_hp if max_hp else 1.0
    target = golden_compass.choose_next_node(_map_json_from_buffer(arr), hp_frac=hp_frac)
    if target is not None:
        col, row = target
        return "select_map_node", {"col": col, "row": row}

    # current_coord is null -- the very first move of the act, forced from every row-0 point
    # (choose_next_node returns None here since there's nothing to plan a path from yet).
    current_children = [(col, row) for row, col, _slot in bl.map_points(arr) if row == 0]
    col, row = random.choice(current_children)
    return "select_map_node", {"col": col, "row": row}


def _choose_map_action_from_buffer(arr) -> tuple[str, dict]:
    current_children = None
    for _row, _col, slot in bl.map_points(arr):
        if int(slot[2]) == 1:  # MAP_POINT_SIZE layout: type, visited, current, then children
            current_children = list(bl.map_point_children(slot))
            break
    if current_children is None:
        # No point is marked current yet -- the very first move of the act, offered from every
        # row-0 point (map_points() yields (row, col, slot) for every real point on the grid).
        current_children = [(col, row) for row, col, _slot in bl.map_points(arr) if row == 0]
    col, row = random.choice(current_children)
    return "select_map_node", {"col": col, "row": row}