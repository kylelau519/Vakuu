from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the import below)
from env import Sts2Session
from models import Decision
import buffer_layout as bl
from buffer_env import BufferDecision, decode_token
import torch

_CLAWS_CHECKPOINT = os.environ.get("CLAWS_CHECKPOINT")
_CLAWS_GREEDY = bool(os.environ.get("CLAWS_GREEDY"))  # eval: always play Claws's top-probability action
_claws_net = None  # lazy-loaded once per worker process, on first combat decision -- see below

# Fraction of root probability mass replaced by noise; 0.25 is AlphaZero's own. CLAWS_NOISE_EPS overrides it, e.g. the
# compass stage plays near-deployed (training_alternating.sh) so its labels are not mostly combat luck.
_COMBAT_DIRICHLET_EPS = float(os.environ.get("CLAWS_NOISE_EPS", "0.25"))
# At or below this eps the noise is a rare deviation from the top move: (1 - eps) argmax + eps noise, so with 0.05 Claws plays
# its top move ~95% of the time (the compass stage, which wants the deployed agent). Above it the noise is exploration
# blended into Claws's own softmax, which is then sampled (Claws's mainline and forks, which want varied play).
_ARGMAX_MIX_MAX_EPS = 0.1
_DIRICHLET_ALPHA_SCALE = 10.0  # concentration = this / n_legal_options

# The shop / remove / upgrade heuristics, and the card-reward fallback while no hazard net exists.
import tier_policy  # noqa: E402

# Map moves and card rewards come from golden_compass's hazard net (GOLDEN_COMPASS_CHECKPOINT; the
# net lazy-loads once per worker process inside compass.py). Flat module names, no `model.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden_compass"))
import compass  # noqa: E402

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


def _tier_card_reward(cards: list, can_skip: bool, deck_size: int) -> tuple[str, dict]:
    """The tier-list card pick -- the fallback while no hazard net is loaded."""
    index = tier_policy.choose_card_reward(cards, can_skip, deck_size)
    if index is None:
        return "skip_card_reward", {}
    return "select_card_reward", {"card_index": index}


def _choose_combat_action_claws(
    decision: BufferDecision, net, combat_dirichlet_noise: bool = True
) -> tuple[str, dict, dict]:
    """Samples from ClawsNet's own softmax for the best combat action, mixed with root Dirichlet noise """

    options, logits, _value = net.forward(decision)
    probs = torch.softmax(logits, dim=-1)
    if _CLAWS_GREEDY:
        combat_dirichlet_noise = False
    if combat_dirichlet_noise:
        alpha = _DIRICHLET_ALPHA_SCALE / len(options)
        noise = torch.distributions.Dirichlet(torch.full((len(options),), alpha)).sample()
        if _COMBAT_DIRICHLET_EPS <= _ARGMAX_MIX_MAX_EPS:
            probs = torch.nn.functional.one_hot(torch.argmax(probs), len(options)).to(probs.dtype)
        probs = (1 - _COMBAT_DIRICHLET_EPS) * probs + _COMBAT_DIRICHLET_EPS * noise
    idx = int(torch.argmax(probs).item()) if _CLAWS_GREEDY else int(torch.multinomial(probs, 1).item())
    policy = [
        {"action": a, "args": args, "prob": p} for (a, args), p in zip(options, probs.tolist())
    ]
    name, args = options[idx]
    return name, args, policy


def _choose_combat_action(
    decision: BufferDecision, session: Sts2Session, combat_dirichlet_noise: bool = True,
) -> tuple[str, dict, dict]:
    """Samples a combat action from a trained ClawsNet if CLAWS_CHECKPOINT is set, falling back to
    a uniform-random placeholder policy otherwise"""
    net = _load_claws_net()
    if net is not None:
        return _choose_combat_action_claws(decision, net, combat_dirichlet_noise)

    options = _combat_options_from_buffer(decision.arr)
    chosen = random.choice(options)
    policy = [
        {"action": name, "args": args, "prob": 1.0 if (name, args) == chosen else 0.0}
        for name, args in options
    ]
    name, args = chosen
    return name, args, policy


def _combat_options_from_buffer(arr) -> list[tuple[str, dict]]:
    """Uniform-random placeholder policy's options: every playable card (aimed at one random enemy when
    it needs a target), every potion, and end_turn."""
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
    card_reward_dirichlet_noise: bool = True,  # unused since the hazard net owns card rewards; kept for callers
) -> tuple[str, dict, str | None, dict | None]:
    if isinstance(decision, BufferDecision):
        return _choose_generic_action_from_buffer(decision, purpose, session)

    dec = decision.decision

    if dec == "map_select":
        name, kwargs, policy = _choose_map_action(decision, session)
        return name, kwargs, None, policy

    if dec == "card_reward":
        name, kwargs = _tier_card_reward(decision.cards, decision.can_skip, decision.player.deck_size or 0)
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
    decision: BufferDecision, purpose: str | None, session=None,
) -> tuple[str, dict, str | None, dict | None]:
    """Buffer counterpart to _choose_generic_action. `session` supplies the character and ascension
    the hazard net needs (the buffer doesn't carry the character)."""
    dec = decision.decision
    arr = decision.arr

    if dec == "map_select":
        name, kwargs, policy = compass.choose_map(
            decision, getattr(session, "character", None), getattr(session, "ascension", 0),
        )
        return name, kwargs, None, policy
    if dec in ("card_reward", "event_card_reward"):
        name, kwargs, policy = _choose_card_reward_from_buffer(decision, session)
        return name, kwargs, None, policy
    if dec == "bundle_select":
        return (
            "select_bundle", {"bundle_index": random.choice([b for b, _ in bl.bundles(arr)])},
            None, None,
        )
    character, ascension = getattr(session, "character", None), getattr(session, "ascension", 0)
    if dec == "card_select":
        picked = compass.choose_card_select(decision, purpose, character, ascension)
        if picked is not None:
            name, kwargs, policy = picked
            return name, kwargs, None, policy
        name, kwargs = _choose_card_select_action(decision, purpose)
        return name, kwargs, None, None
    if dec == "shop":
        picked = compass.choose_shop(decision, character, ascension)
        if picked is not None:
            return picked
        name, kwargs, next_purpose = _choose_shop_action_from_buffer(arr)
        return name, kwargs, next_purpose, None
    if dec == "rest_site":
        picked = compass.choose_rest(decision, character, ascension)
        if picked is not None:
            return picked
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
    decision: BufferDecision, session
) -> tuple[str, dict, dict | None]:
    """The hazard net's pick (afterstate = deck with / without each card) when a checkpoint is
    loaded, else the tier list. -> (name, kwargs, policy|None)."""
    arr = decision.arr
    if bl.decision_name(arr) == "event_card_reward":
        slots, can_skip = list(bl.event_card_reward_cards(arr)), True
    else:
        slots, can_skip = list(bl.reward_cards(arr)), bl.can_skip_reward(arr)
    offered = [(i, decode_token(bl.card_token(slot)), int(slot[bl.CARD_UPGRADED])) for i, slot in slots]

    picked = compass.choose_card_reward(
        decision, offered, can_skip, getattr(session, "character", None), getattr(session, "ascension", 0),
    )
    if picked is not None:
        return picked

    cards, can_skip = _card_reward_options_from_buffer(decision)
    name, kwargs = _tier_card_reward(cards, can_skip, bl.player_deck_size(arr))
    return name, kwargs, None


def _choose_card_reward_heuristic(decision: BufferDecision) -> tuple[str, dict]:
    """Test-only counterpart to _choose_card_reward_from_buffer: never loads the hazard net and
    always resolves through the tier list."""
    cards, can_skip = _card_reward_options_from_buffer(decision)
    return _tier_card_reward(cards, can_skip, bl.player_deck_size(decision.arr))


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
        index = tier_policy.choose_shop_card(cards, bl.player_deck_size(arr))
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
    """pydantic-side (non-buffer) decisions only: uniform random. The hazard net reads the buffer."""
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
        index = tier_policy.choose_removal_target(decision.cards)
        return "select_cards", {"indices": [index]}

    if purpose == "upgrade" and decision.cards:
        index = tier_policy.choose_upgrade_target(decision.cards)
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
        index = tier_policy.choose_shop_card(affordable_cards, decision.player.deck_size or 0)
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
##### map_select is a uniform random move and card_reward resolves through the tier list.

def choose_generic_action(
    decision: BufferDecision, purpose: str | None = None,
) -> tuple[str, dict, str | None, dict | None]:
    dec = decision.decision
    arr = decision.arr

    if dec == "combat_play":
        name, kwargs = random.choice(_combat_options_from_buffer(arr))
        return name, kwargs, None, None
    if dec == "map_select":
        name, kwargs = _choose_map_action_from_buffer(arr)
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


def _choose_map_action_from_buffer(arr) -> tuple[str, dict]:
    """Uniform random move -- the no-model map policy (the hazard net, when loaded, goes through compass.choose_map)."""
    import run_state  # golden_compass/run_state.py, on sys.path since the compass import above
    col, row = random.choice(run_state.map_options(arr))
    return "select_map_node", {"col": col, "row": row}
