"""Non-combat (meta) decisions delegate to the Choices Paradox placeholder policy (agents/
choices_paradox/policy.py), which scores cards off Baalorlord's STS2 tier lists instead of
picking blindly. map_select likewise delegates to Golden Compass (agents/golden_compass/
policy.py), which value-to-go DPs the full map instead of picking a reachable node blindly.
card_reward specifically upgrades to a trained ChoicesParadoxNet (agents/choices_paradox/model/
model.py), sampled the same Dirichlet-noised-softmax way _choose_combat_action_claws samples
Claws, once CHOICES_PARADOX_CHECKPOINT is set -- see _choose_card_reward_decision -- falling back
to the tier-list heuristic otherwise (round 1, no checkpoint yet).

Returns (action_name, kwargs, policy) rather than a built command, so callers pass the first two
straight into session.step(name, **kwargs) -- the session builds/validates the actual command via
actions.action(), no need to duplicate that here.

_choose_combat_action_claws mixes AlphaZero-style Dirichlet noise into that softmax before
sampling (see _DIRICHLET_EPS/_DIRICHLET_ALPHA_SCALE below) rather than reintroducing search --
without it, an option whose probability drifts near 0 is rarely/never one of branches_per_turn's
on-policy samples, so it never appears in new training data and its probability estimate can
never correct itself. Confirmed empirically (debug_card_bias.py) on an overnight training.sh run:
over a dozen legal Ironclad cards sat at exactly 0.0 average probability across hundreds of
sightings despite being playable, and training.sh's forever-loop (--init resuming the previous
round's checkpoint) would have compounded that lockout round over round.
"""
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

# Loaded by file path rather than a package import -- same reasoning as choices_paradox/
# golden_compass above (no __init__.py-based package structure to import through, and this
# avoids colliding with claws/model/model.py, also just named `model`).
_choices_paradox_model_path = (
    Path(__file__).resolve().parents[1] / "choices_paradox" / "model" / "model.py"
)
_spec = importlib.util.spec_from_file_location("choices_paradox_model", _choices_paradox_model_path)
assert _spec is not None and _spec.loader is not None
_cp_model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cp_model  # dataclasses' class-processing looks itself up via
# sys.modules[cls.__module__] during exec_module -- module_from_spec alone doesn't register it
_spec.loader.exec_module(_cp_model)

_CHOICES_PARADOX_CHECKPOINT = os.environ.get("CHOICES_PARADOX_CHECKPOINT")
_choices_paradox_net = None  # lazy-loaded once per worker process, on first card_reward decision

_REST_HEAL_HP_THRESHOLD = 35


@dataclass
class _BufferCard:
    """Satisfies choices_paradox.policy's _CardLike Protocol (.index/.id) off a buffer Card(11)
    slot -- lets the buffer-side card_reward/shop choosers below reuse the same tier-based
    picker the pydantic path uses, instead of a separate uniform-random duplicate."""
    index: int
    id: str | None

def choose_action(
    decision: Decision | BufferDecision, session: Sts2Session, purpose: str | None = None,
    combat_dirichlet_noise: bool = True, card_reward_dirichlet_noise: bool = True,
) -> tuple[str, dict, dict | None, str | None]:
    """purpose/the trailing return value thread whichever card_select "purpose" (remove/upgrade)
    a preceding shop/rest_site pick just triggered -- see _choose_card_select_action's docstring
    for why this can no longer be a module-level global. Callers that loop over a sequence of
    decisions (main_worker_loop, _play_turn_only, any debug script driving a session end to end)
    must pass back whatever purpose the previous call returned; a caller that only ever needs one
    decision (never card_select) can ignore the returned purpose entirely.

    combat_dirichlet_noise/card_reward_dirichlet_noise: independent on/off switches for the two
    trained-net samplers (_choose_combat_action_claws, _choose_card_reward_net respectively) --
    both on by default, matching self-play training's own need for exploration (see this module's
    docstring for why skipping it starves rare options of any training signal). Kept separate
    rather than one shared flag because the two training loops want different answers for the
    card_reward side: training Claws (facility_claws.py) wants it ON -- card_reward outcomes feed
    back into which decks Claws sees, so a deterministic card_reward net would reinforce whatever
    blind spots it already has instead of exploring around them; training Choices Paradox
    (facility_choices_paradox.py's MetaOptionJob chains) wants it OFF for a chain's own
    downstream card_reward picks, since that noise would add unwanted variance on top of the
    option-chain comparison the chain's value IS the training signal for -- see
    facility_choices_paradox.py's own module docstring on why combat's noise inside a chain is
    accepted variance but this isn't. Neither flag affects either net's untrained/heuristic
    fallback path (round 1, no checkpoint yet)."""
    if decision.decision == "combat_play":
        name, args, policy = _choose_combat_action(decision, session, combat_dirichlet_noise)
        return name, args, policy, None
    name, kwargs, next_purpose = _choose_generic_action(
        decision, session, purpose, card_reward_dirichlet_noise
    )
    return name, kwargs, None, next_purpose


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


def _load_choices_paradox_net():
    """Same lazy-load-once-per-process convention as _load_claws_net, keyed off
    CHOICES_PARADOX_CHECKPOINT -- kept as its own module-global (rather than reusing
    facility_choices_paradox.py's copy) since that module isn't on the path of every process
    that imports this one (facility_claws.py, in particular, never imports it)."""
    global _choices_paradox_net
    if _choices_paradox_net is not None or not _CHOICES_PARADOX_CHECKPOINT:
        return _choices_paradox_net
    if not os.path.exists(_CHOICES_PARADOX_CHECKPOINT):
        return None  # round 1, or CHOICES_PARADOX_CHECKPOINT unset by the training script yet
    _choices_paradox_net = _cp_model.ChoicesParadoxNet.from_checkpoint(_CHOICES_PARADOX_CHECKPOINT)
    return _choices_paradox_net


def _choose_card_reward_net(decision, net, dirichlet_noise: bool = True) -> tuple[str, dict]:
    """Samples (not argmax) from ChoicesParadoxNet's own softmax over model.py's
    legal_options()/legal_options_from_buffer(), mixed with root Dirichlet noise unless
    dirichlet_noise=False -- same convention (and same _DIRICHLET_EPS/_DIRICHLET_ALPHA_SCALE) as
    _choose_combat_action_claws above and facility_choices_paradox.py's own
    _choose_meta_option_claws. `decision` is either a pydantic CardReward or a BufferDecision --
    net.forward dispatches on that itself."""
    import torch

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
    """card_reward's action chooser, shared by the pydantic (_choose_generic_action) and buffer
    (_choose_card_reward_from_buffer, including its event_card_reward variant) paths. Samples
    from a loaded ChoicesParadoxNet when CHOICES_PARADOX_CHECKPOINT is set (see
    _choose_card_reward_net) -- the trained pick, aware of full-deck synergy/curve the tier list
    can't see -- falling back to choices_paradox.policy's static tier-list heuristic otherwise
    (round 1, no checkpoint yet), same "noisy but harmless" fallback Claws's own combat picker
    already follows. `decision` is only used by the net path (net.forward's own dispatch on
    pydantic CardReward vs. BufferDecision); the heuristic path reads only cards/can_skip/
    deck_size, same as before this was wired in. card_reward_dirichlet_noise is passed straight
    through to _choose_card_reward_net and has no effect on the heuristic fallback."""
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
    """Samples (not argmax) from the network's own softmax over model.py's legal_options(),
    mixed with root Dirichlet noise unless combat_dirichlet_noise=False -- see this module's
    docstring for why noise is on by default."""
    import torch

    options, logits, _value = net.forward(decision)
    probs = torch.softmax(logits, dim=-1)
    if combat_dirichlet_noise:
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
    decision: CombatPlay | BufferDecision, session: Sts2Session,
    combat_dirichlet_noise: bool = True,
) -> tuple[str, dict, dict]:
    net = _load_claws_net()
    if net is not None:
        return _choose_combat_action_claws(decision, net, combat_dirichlet_noise)

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


def _choose_generic_action(
    decision: Decision, session: Sts2Session, purpose: str | None,
    card_reward_dirichlet_noise: bool = True,
) -> tuple[str, dict, str | None]:
    """Non-combat decisions: pick a uniformly random legal option, or the simplest safe
    default (leave/skip) where "random" isn't a meaningful choice yet. Good enough to
    keep a run moving forward -- combat_play is what's actually being trained right now.

    Returns a third element, next_purpose: None for every decision type except shop/rest_site,
    which can return "remove"/"upgrade" to be threaded into the caller's *next* choose_action
    call (see that function's docstring) -- the only two actions that ever trigger a follow-up
    card_select whose purpose isn't recoverable from CardSelect's own fields.

    A BufferDecision reaching here is delegated to _choose_generic_action_from_buffer entirely --
    see that function's docstring for why every branch below except card_select assumes a
    pydantic Decision's attribute tree (.player, .cards, .choices, ...) directly.
    card_reward_dirichlet_noise is only read by the card_reward branch (see choose_action's
    docstring)."""
    if isinstance(decision, BufferDecision):
        return _choose_generic_action_from_buffer(decision, purpose, card_reward_dirichlet_noise)

    dec = decision.decision

    if dec == "map_select":
        name, kwargs = _choose_map_action(decision, session)
        return name, kwargs, None

    if dec == "card_reward":
        name, kwargs = _choose_card_reward_decision(
            decision, decision.cards, decision.can_skip, decision.player.deck_size or 0,
            card_reward_dirichlet_noise,
        )
        return name, kwargs, None

    if dec == "bundle_select":
        return "select_bundle", {"bundle_index": random.choice(decision.bundles).index}, None

    if dec == "card_select":
        name, kwargs = _choose_card_select_action(decision, purpose)
        return name, kwargs, None

    if dec == "shop":
        return _choose_shop_action(decision)

    if dec == "rest_site":
        return _choose_rest_site_action(decision)

    if dec == "event_choice":
        unlocked = [o for o in decision.options if not o.is_locked]
        if not unlocked:
            return "leave_room", {}, None
        return "choose_option", {"option_index": random.choice(unlocked).index}, None

    raise ValueError(f"no policy for decision type {dec!r}")


def _choose_generic_action_from_buffer(
    decision: BufferDecision, purpose: str | None, card_reward_dirichlet_noise: bool = True
) -> tuple[str, dict, str | None]:
    """Buffer counterpart to _choose_generic_action. Reached only by a meta-branch's horizon walk
    (facility_choices_paradox._run_meta_option_job) -- a turn-branch worker's own job always
    ends the instant its one combat turn resolves, so it never reaches any of these. Same "uniform
    random / simplest safe default" philosophy as the pydantic path; where a smart heuristic
    already exists (card_reward, shop's card/potion picks), it's reused via the _BufferCard
    wrapper rather than duplicated.

    "card_reward"/"event_card_reward" are handled together here since the buffer protocol splits
    what the JSON protocol reports as one "card_reward" literal into two distinct decision kinds
    (see _choose_card_reward_from_buffer's docstring). map_select can't delegate to Golden Compass
    (that needs session.get_map()'s pydantic-shaped map) so it picks uniformly among the current
    map point's own children instead. rest_site can't distinguish HEAL/SMITH/etc. by option_id
    (buffer_layout's rest_options() has no semantic label), so it picks uniformly and never
    threads a purpose -- an upgrade pick just falls back to card_select's own uniform-random
    choice (see _choose_card_select_action's purpose=None fallback)."""
    dec = decision.decision
    arr = decision.arr

    if dec == "map_select":
        name, kwargs = _choose_map_action_from_buffer(arr)
        return name, kwargs, None
    if dec in ("card_reward", "event_card_reward"):
        name, kwargs = _choose_card_reward_from_buffer(decision, card_reward_dirichlet_noise)
        return name, kwargs, None
    if dec == "bundle_select":
        return "select_bundle", {"bundle_index": random.choice([b for b, _ in bl.bundles(arr)])}, None
    if dec == "card_select":
        name, kwargs = _choose_card_select_action(decision, purpose)
        return name, kwargs, None
    if dec == "shop":
        return _choose_shop_action_from_buffer(arr)
    if dec == "rest_site":
        return _choose_rest_site_action_from_buffer(arr)
    if dec == "event_choice":
        unlocked = [i for i, is_locked in bl.event_options(arr) if not is_locked]
        if not unlocked:
            return "leave_room", {}, None
        return "choose_option", {"option_index": random.choice(unlocked)}, None

    raise ValueError(f"no buffer-side policy for decision type {dec!r}")


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


def _choose_card_reward_from_buffer(
    decision: BufferDecision, card_reward_dirichlet_noise: bool = True
) -> tuple[str, dict]:
    """Handles both buffer decision kinds a "card_reward" pydantic-side decision can actually be:
    the ordinary post-combat CARD_REWARD section, or the blessing-triggered EVENT_CARD_REWARD one
    -- STS2's JSON protocol unifies them under one "card_reward" literal (WriteEventCardReward),
    but the buffer protocol keeps them as separate sections with no shared can_skip flag (the
    event variant is unconditionally skippable -- see model.py's legal_options_from_buffer for the
    same distinction). Takes the whole BufferDecision, not just its .arr, so the net path
    (_choose_card_reward_decision -> _choose_card_reward_net -> net.forward) has `.decision`
    available to tell the two kinds apart the same way model.py's own tokenize_from_buffer/
    legal_options_from_buffer do; the heuristic path only ever reads cards/can_skip/deck_size."""
    arr = decision.arr
    if bl.decision_name(arr) == "event_card_reward":
        cards = [
            _BufferCard(i, decode_token(bl.card_token(slot)))
            for i, slot in bl.event_card_reward_cards(arr)
        ]
        can_skip = True
    else:
        cards = [_BufferCard(i, decode_token(bl.card_token(slot))) for i, slot in bl.reward_cards(arr)]
        can_skip = bl.can_skip_reward(arr)
    return _choose_card_reward_decision(
        decision, cards, can_skip, bl.player_deck_size(arr), card_reward_dirichlet_noise
    )


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


def _choose_map_action(decision: Decision, session: Sts2Session) -> tuple[str, dict]:
    """Delegates to Golden Compass's value-to-go DP over the full act map."""
    hp_frac = decision.player.hp / decision.player.max_hp if decision.player.max_hp else 1.0
    target = golden_compass.choose_next_node(session.get_map(), hp_frac=hp_frac)
    if target is not None:
        for choice in decision.choices:
            if (choice.col, choice.row) == target:
                return "select_map_node", {"col": choice.col, "row": choice.row}

    pick = random.choice(decision.choices)
    return "select_map_node", {"col": pick.col, "row": pick.row}


def _choose_card_select_action(
    decision: Decision | BufferDecision, purpose: str | None
) -> tuple[str, dict]:
    """purpose comes from the caller (see choose_action's docstring) -- whichever shop/rest_site
    action just triggered this follow-up, threaded explicitly rather than read off a module global
    (a global couldn't be set by a shop/rest_site pick in one process and read by a card_select
    resolved in another, which matters once a meta-branch worker's own continuation can wander
    through a shop/rest_site too). Anything else (True Grit's exhaust-select, Wish's deck-tutor-
    select, ...) falls back to a uniformly random legal pick -- CardSelect's shape alone can't
    tell those apart from a remove/upgrade select, and no purpose is passed for them.

    A BufferDecision reaching here always falls through to the uniform-random buffer branch below
    regardless of purpose -- choices_paradox.choose_removal_target/choose_upgrade_target take
    pydantic Card objects, not buffer slots."""
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
