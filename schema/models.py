"""Typed models for sts2-cli's decision-point JSON.

Usage:
    import json
    from schema.models import parse_decision, CombatPlay

    decision = parse_decision(json.loads(line))
    if isinstance(decision, CombatPlay):
        for card in decision.hand:
            if card.can_play:
                ...

`parse_decision` dispatches on the "decision" field and raises pydantic.ValidationError on a
shape mismatch -- that error is the drift signal: if sts2-cli's JSON changes, this is where it
shows up, right at the point of capture rather than silently inside training data. Re-read the
relevant RunSimulator.cs method and update the model when it does.

Source: sts2-cli commit f4d7538 (2026-08-13), src/Sts2Headless/RunSimulator.cs. See VERSION.json.

sts2-cli serializes with JsonIgnoreCondition.WhenWritingNull -- an absent key and a null value are
indistinguishable on the wire, and nearly every field *is* conditionally absent. So almost
everything below is Optional with a None default, and every model allows extra fields
(extra="allow") rather than rejecting anything this file hasn't caught up to yet.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


# --- shared (RunSimulator.cs: PlayerSummary 3001-3068, RunContext 3071-3101) ---


class UpgradeInfo(Loose):
    """GetUpgradedInfo, RunSimulator.cs:2964-2999."""

    cost: int
    description: str
    stats: dict[str, int] | None = None
    added_keywords: list[str] | None = None
    removed_keywords: list[str] | None = None


class Card(Loose):
    """Fields common to every card representation (hand, deck, reward, shop, card_select,
    bundle_select). Context-specific fields (can_play, target_type, damage_by_target, upgraded,
    is_stocked, on_sale, card_cost/cost) are added by each decision's own card subclass."""

    index: int | None = None
    id: str | None = None  # "CARD:ENTRY" form, e.g. CARD:STRIKE. Absent on bundle_select cards.
    name: str
    cost: int  # energy cost
    type: str  # CardType enum as string (Attack/Skill/Power/Status/Curse, not exhaustively known)
    rarity: str | None = None  # absent on PlayerSummary.deck cards -- confirmed via live capture
    description: str
    upgraded: bool | None = None  # present on deck cards; required+re-declared on SelectableCard
    stats: dict[str, int] | None = None
    """combat_play hand cards: live preview (Strength/Weak/Frail/Block applied). Everywhere else:
    base values. The same card can report different numbers depending on which decision it's in."""
    keywords: list[str] | None = None
    enchantment: str | None = None
    enchantment_amount: int | None = None
    affliction: str | None = None
    affliction_amount: int | None = None
    after_upgrade: UpgradeInfo | None = None


class Relic(Loose):
    name: str
    description: str
    vars: dict[str, int] | None = None


class Potion(Loose):
    """NEEDS VERIFICATION: only index and name were confirmed from reading RunSimulator.cs;
    extend once real combat_play/shop output has been captured and diffed against this."""

    index: int
    """Original slot. PlayerSummary.potions drops nulls, so array position != index once any
    potion in the run has been used or discarded -- use this field, not the list position."""
    name: str


class Power(Loose):
    name: str
    description: str | None = None
    amount: int


class Boss(Loose):
    id: str | None = None
    name: str | None = None


class RunContext(Loose):
    act: int  # 1-based
    act_name: str | None = None
    floor: int
    room_type: str
    boss: Boss | None = None


class PlayerSummary(Loose):
    name: str
    hp: int
    """In game_over, forced to 0 on death via a _lastKnownHp workaround -- the engine itself
    resets CurrentHp to max_hp on death."""
    max_hp: int
    block: int | None = None
    gold: int
    relics: list[Relic] = []
    potions: list[Potion] = []
    deck_size: int | None = None
    deck: list[Card] | None = None


# --- combat_play (CombatPlayState, RunSimulator.cs:2099-2376) ---


class DamageByTargetRow(Loose):
    target_index: int
    name: str
    damage: int | None = None
    repeat: int | None = None
    total_damage: int | None = None


class HandCard(Card):
    index: int
    rarity: str  # confirmed always present on hand cards via live capture, unlike the deck-only base
    can_play: bool  # forced false if star_cost exceeds the player's available Regent stars
    target_type: str
    """TargetType enum: None, Self, AnyEnemy, AllEnemies, ... play_card requires target_index
    only when this is AnyEnemy and >=2 enemies are alive."""
    star_cost: int | None = None  # present only when > 0 (Regent)
    damage_by_target: list[DamageByTargetRow] | None = None
    """Present only for Attack-type cards with target_type AnyEnemy/AllEnemies. One row per
    currently-alive enemy -- present (1 element) even in a 1-enemy fight."""


class Intent(Loose):
    type: str  # Intent class name, e.g. AttackIntent
    damage: int | None = None  # AttackIntent: per-hit damage if hits > 1, else the total
    hits: int | None = None  # AttackIntent only, present when > 1 hit
    total_damage: int | None = None  # AttackIntent only, present when hits > 1


class Enemy(Loose):
    """Filtered to alive enemies only, then re-indexed 0..k every call -- index is NOT stable
    across turns once anything dies."""

    index: int
    name: str
    hp: int
    max_hp: int
    block: int
    intends_attack: bool
    intents: list[Intent] | None = None
    powers: list[Power] | None = None


class Orb(Loose):
    """Defect only, present iff pcs.OrbQueue.Orbs.Count > 0."""

    index: int
    name: str
    type: str  # Orb class name with the trailing "Orb" stripped
    passive: object | None = None
    evoke: object | None = None


class Osty(Loose):
    """Necrobinder only. {alive: False} if Osty is null; absent entirely for other characters."""

    alive: bool
    name: str | None = None
    hp: int | None = None
    max_hp: int | None = None
    block: int | None = None


class CombatPlay(Loose):
    type: Literal["decision"] = "decision"
    decision: Literal["combat_play"] = "combat_play"
    context: RunContext
    round: int
    energy: int
    max_energy: int
    draw_pile_count: int | None = None
    discard_pile_count: int | None = None
    hand: list[HandCard]
    enemies: list[Enemy]
    player: PlayerSummary
    player_powers: list[Power] | None = None
    orbs: list[Orb] | None = None  # Defect only
    orb_slots: int | None = None  # Defect only, orb queue capacity, alongside orbs
    stars: int | None = None
    """Regent only. Gated on character.Id.Entry == "REGENT", not a star-cost heuristic --
    present even at 0 stars, absent for every other character."""
    osty: Osty | None = None  # Necrobinder only


# --- map_select (MapSelectState, RunSimulator.cs:2002-2097) ---


class MapChoice(Loose):
    col: int
    row: int
    type: str  # MapPointType enum as string


class MapSelect(Loose):
    """At run start (CurrentMapCoord unset), choices includes the starting node itself plus its
    children; on every later floor only the reachable children are listed."""

    type: Literal["decision"] = "decision"
    decision: Literal["map_select"] = "map_select"
    choices: list[MapChoice]
    player: PlayerSummary
    act: int
    act_name: str | None = None
    floor: int


# --- card_reward (CardRewardState, RunSimulator.cs:2457-2492; event-triggered variant 1841-1875) ---


class RewardCard(Card):
    index: int


class CardReward(Loose):
    type: Literal["decision"] = "decision"
    decision: Literal["card_reward"] = "card_reward"
    cards: list[RewardCard]
    can_skip: bool
    gold_earned: int | None = None  # post-combat variant only -- delta vs pre-combat gold
    from_event: Literal[True] | None = None  # present (always True) only on the event variant
    player: PlayerSummary


# --- card_select (inline, RunSimulator.cs:1877-1912) ---


class SelectableCard(Card):
    index: int
    upgraded: bool


class CardSelect(Loose):
    """Uses key "cards", not "options". Answer with actions.action("select_cards", indices=...).
    Fires for upgrade/remove/transform picks and mid-shop/rest-site sub-selections."""

    type: Literal["decision"] = "decision"
    decision: Literal["card_select"] = "card_select"
    cards: list[SelectableCard]
    min_select: int
    max_select: int
    player: PlayerSummary


# --- bundle_select (inline, RunSimulator.cs:1807-1839 -- Scroll Boxes) ---


class Bundle(Loose):
    index: int
    cards: list[Card]  # same shape as reward cards, minus id/after_upgrade


class BundleSelect(Loose):
    type: Literal["decision"] = "decision"
    decision: Literal["bundle_select"] = "bundle_select"
    bundles: list[Bundle]
    player: PlayerSummary


# --- event_choice (EventChoiceState, RunSimulator.cs:2510-2691) ---


class EventOption(Loose):
    index: int
    title: str
    description: str | None = None
    text_key: str
    is_locked: bool
    vars: dict[str, object] | None = None
    """Numeric DynamicVars, e.g. Gold, HpLoss, Heal. RandomCard is resolved to a card name."""


class EventChoice(Loose):
    """If ALL options are is_locked, sts2-cli auto-leaves the room internally instead of
    returning this decision -- an all-locked event_choice never reaches the client."""

    type: Literal["decision"] = "decision"
    decision: Literal["event_choice"] = "event_choice"
    event_name: str
    description: str | None = None
    options: list[EventOption]
    player: PlayerSummary


# --- rest_site (RestSiteState, RunSimulator.cs:2693-2722) ---


class RestOption(Loose):
    index: int
    option_id: str
    name: str  # C# type name, e.g. RestSiteHealOption
    is_enabled: bool


class RestSite(Loose):
    """If the options list would be empty (choice already resolved), sts2-cli silently forces to
    map and returns map_select instead -- a client never sees an empty-options rest_site."""

    type: Literal["decision"] = "decision"
    decision: Literal["rest_site"] = "rest_site"
    options: list[RestOption]


# --- shop (ShopState, RunSimulator.cs:2724-2804) ---


class ShopCard(Card):
    index: int
    card_cost: int  # energy cost
    cost: int  # gold price -- easy to confuse with card_cost
    is_stocked: bool
    on_sale: bool


class ShopRelic(Loose):
    index: int
    name: str
    description: str | None = None
    cost: int
    is_stocked: bool


class ShopPotion(Loose):
    index: int
    name: str
    description: str | None = None
    cost: int
    is_stocked: bool


class Shop(Loose):
    type: Literal["decision"] = "decision"
    decision: Literal["shop"] = "shop"
    cards: list[ShopCard]
    relics: list[ShopRelic]
    potions: list[ShopPotion]
    card_removal_cost: int | None = None  # absent if no removal entry is on offer
    player: PlayerSummary


# --- game_over (GameOverState, RunSimulator.cs:2892-2909) ---


class GameOver(Loose):
    type: Literal["decision"] = "decision"
    decision: Literal["game_over"] = "game_over"
    victory: bool
    player: PlayerSummary
    act: int
    floor: int


Decision = Annotated[
    Union[
        CombatPlay,
        MapSelect,
        CardReward,
        CardSelect,
        BundleSelect,
        EventChoice,
        RestSite,
        Shop,
        GameOver,
    ],
    Field(discriminator="decision"),
]

_decision_adapter = TypeAdapter(Decision)


def parse_decision(raw: dict) -> Decision:
    """Parse a raw decision-point dict (already json.loads'd) into its typed model, dispatching
    on the "decision" field. Raises pydantic.ValidationError on a shape mismatch."""
    return _decision_adapter.validate_python(raw)
