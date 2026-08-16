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

Source: sts2-cli commit 0f344ab (2026-08-16), src/Sts2Headless/RunSimulator.cs and
RunSimulator.States.cs. Adds history counts, hand-card `upgraded`, and power `id` (see
CombatHistorySummary/HistoryCounts docstrings below) -- not yet reflected in
tests/fixtures/ironclad_run_01.jsonl, which predates this commit; re-capture when convenient.
See VERSION.json.

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
    rarity: str | None = None
    """Absent on PlayerSummary.deck cards through schema 0.1.0 -- a confirmed quirk (RunSimulator.cs's
    deck export inlined its own dict literal and simply never included it). Fixed as of schema 0.2.0,
    when the deck export moved onto the shared CardToJson helper used by every other card list --
    deck cards now always carry rarity too."""
    description: str
    upgraded: bool | None = None
    """Present on deck/pile cards; required+re-declared on SelectableCard/PileCard. Through
    schema 0.2.0, HandCard never carried this (CombatPlayState's hand builder is a separate
    hand-rolled dict from the shared CardToJson helper, and simply never set it) -- an omission,
    not a design choice, fixed in an unreleased sts2-cli patch. Stays Optional here since old
    fixtures were captured before the fix; once a fresh capture confirms it's always present on
    hand cards, HandCard can re-require it the way PileCard does."""
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
    counter: int | None = None
    """RelicModel.ShowCounter/DisplayAmount -- the same generic hook the game's own UI reads to draw
    a relic's progress badge (e.g. Pen Nib: N/10 attacks played toward double damage). Present only
    for relics with a visible counter (confirmed: Pen Nib, Nunchaku, Kunai, Pocketwatch); None for
    everything else (Anchor, Orichalcum, ...) -- not a per-relic-type field, a single base-class
    property read uniformly across player.Relics."""


class Potion(Loose):
    """NEEDS VERIFICATION: only index and name were confirmed from reading RunSimulator.cs;
    extend once real combat_play/shop output has been captured and diffed against this."""

    index: int
    """Original slot. PlayerSummary.potions drops nulls, so array position != index once any
    potion in the run has been used or discarded -- use this field, not the list position."""
    name: str


class Power(Loose):
    id: str | None = None
    """PowerModel.Id.Entry, e.g. "MINION_POWER", "VULNERABLE_POWER" -- added alongside the
    localized `name` so a client can match a specific power without string-matching a locale-
    dependent display string (e.g. detecting is_minion via id == "MINION_POWER"). Optional since
    fixtures captured before this field existed won't have it."""
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


class PileCard(Card):
    """draw_pile/discard_pile/exhaust_pile entries (RunSimulator.cs: PileCardsToJson,
    CombatPlayState). Static/base-value fields only -- no can_play/target_type/damage_by_target,
    unlike HandCard; these cards aren't currently playable. Index 0 = next to draw (draw_pile) or
    most recently discarded/exhausted (discard_pile/exhaust_pile) -- pile order is real and
    meaningful (CardPile is backed by an ordered list; shuffles are discrete RNG events, not lazy
    per-draw reordering), not an unordered set."""

    index: int
    upgraded: bool
    rarity: str  # unlike PlayerSummary.deck historically, pile cards always carry rarity


class EntryCount(Loose):
    """One CombatHistory entry type's count for one creature. `this_combat` is the running total
    since combat started; `this_turn` resets each time HappenedThisTurn's round+side+per-player-
    turn-number triple changes -- i.e. it's scoped like Spite's damage-taken check, not like a
    naive "since I last looked" delta."""

    this_turn: int
    this_combat: int


class HistoryCounts(Loose):
    """Per-creature counts of MegaCrit.Sts2.Core.Combat.History.Entries.* types (RunSimulator.cs:
    HistoryCounts). STS2 has no dedicated counter fields for "how many times has X happened this
    combat" -- cards that scale off event counts (Supermassive/cards-created, Spite/damage-taken-
    this-turn, Murder/cards-drawn) compute it live by filtering CombatManager.Instance.History.
    Entries at DynamicVar preview time. Exposing the same counts here lets a client see the
    scaling input even when the scaling card isn't currently in hand -- combat_play's `stats`
    field already bakes the correct number into a card that *is* in hand, so these are for
    everything else: anticipating a draw, or valuing the trend as board state in its own right.

    Field name -> the CombatHistoryEntry subclass it counts, and what "counted" means for that
    creature (its Actor unless noted):
      block_gained            BlockGainedEntry (Actor = the creature that gained block)
      card_afflicted          CardAfflictedEntry (owner of the afflicted card)
      card_discarded          CardDiscardedEntry (owner of the discarded card)
      card_drawn              CardDrawnEntry (owner of the drawn card)
      card_exhausted          CardExhaustedEntry (owner of the exhausted card)
      card_generated          CardGeneratedEntry (owner of the generated card -- NOT who created
                               it; see card_created_by_player on PlayerHistory for that)
      card_played             CardPlayFinishedEntry (the creature that played the card)
      creature_attacked       CreatureAttackedEntry (the attacker)
      damage_received         DamageReceivedEntry, filtered on Receiver == this creature
      damage_dealt            DamageReceivedEntry, filtered on Dealer == this creature (the same
                               entries as damage_received on the other creature; not a distinct
                               entry type)
      energy_spent            EnergySpentEntry (player only in practice -- enemies don't spend energy)
      monster_move_performed  MonsterPerformedMoveEntry (the monster)
      orb_channeled           OrbChanneledEntry (the orb's owner)
      potion_used             PotionUsedEntry (the drinker)
      power_received          PowerReceivedEntry, filtered on Actor == this creature (who got the power)
      power_applied           PowerReceivedEntry, filtered on Applier == this creature (who caused it --
                               None for e.g. innate/start-of-combat powers, so this can undercount)
      stars_modified          StarsModifiedEntry (Regent only)
      summoned                SummonedEntry (whoever did the summoning)
    """

    block_gained: EntryCount
    card_afflicted: EntryCount
    card_discarded: EntryCount
    card_drawn: EntryCount
    card_exhausted: EntryCount
    card_generated: EntryCount
    card_played: EntryCount
    creature_attacked: EntryCount
    damage_received: EntryCount
    damage_dealt: EntryCount
    energy_spent: EntryCount
    monster_move_performed: EntryCount
    orb_channeled: EntryCount
    potion_used: EntryCount
    power_received: EntryCount
    power_applied: EntryCount
    stars_modified: EntryCount
    summoned: EntryCount


class PlayerHistory(HistoryCounts):
    card_created_by_player: EntryCount
    """CardGeneratedEntry filtered on Creator == player (a Player, not a Creature -- this is why
    it's not folded into the generic card_generated count, which filters on Actor/owner instead).
    This is Supermassive's exact scaling input: `.Count(c => c.Creator == card.Owner)`."""


class EnemyHistory(HistoryCounts):
    index: int
    """Matches Enemy.index for the same combat_play decision -- NOT stable across turns once
    something dies, same caveat as Enemy.index."""


class CombatHistorySummary(Loose):
    player: PlayerHistory
    enemies: list[EnemyHistory]


class CombatPlay(Loose):
    type: Literal["decision"] = "decision"
    decision: Literal["combat_play"] = "combat_play"
    context: RunContext
    round: int
    energy: int
    max_energy: int
    draw_pile_count: int | None = None
    discard_pile_count: int | None = None
    draw_pile: list[PileCard] | None = None
    discard_pile: list[PileCard] | None = None
    exhaust_pile: list[PileCard] | None = None  # no exhaust_pile_count -- exhaust had no prior count field
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
    history: CombatHistorySummary | None = None
    """Wrapped in its own try/catch on the C# side (RunSimulator.States.cs) -- absent (not a
    validation failure) if the aggregation itself throws, same defensive pattern as the
    character-specific block above it."""


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
    Fires for upgrade/remove/transform picks and mid-shop/rest-site sub-selections, including
    effects that restrict the pool to a card type or pile (e.g. "choose a Skill card from your
    hand/discard/draw").

    There is no separate "pickable"/"valid" mask, and none is needed: `cards` is always the
    complete, already-eligible set. The engine applies any type/pile/eligibility filter *before*
    handing the list to the card selector -- verified in the decompiled source's
    `CardSelectCmd` (`FromHand`, `FromCombatPile`, `FromDeckGeneric`, etc.), which all build a
    pre-filtered `List<CardModel>` and only then call `ICardSelector.GetSelectedCards(list, ...)`.
    sts2-cli's `HeadlessCardSelector` (the thing `card_select`'s `cards` field is read from) is
    exactly that `ICardSelector` implementation -- it only ever sees the filtered list, never the
    full hand/pile with some cards marked invalid. So "am I allowed to pick this?" is always
    "is it in `cards`?" -- no client-side filtering logic (by type or otherwise) is needed or
    possible; the constraint that produced this selection isn't exposed at all (min/max_select
    tell you how many to pick, not why these specific cards are the ones offered)."""

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
