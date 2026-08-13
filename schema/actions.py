"""Typed builders for sts2-cli's outgoing `action` commands.

Usage:
    from schema.actions import action

    cmd = action("play_card", card_index=0, target_index=1)
    # -> {"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 1}}
    proc.stdin.write(json.dumps(cmd) + "\\n")

    action("end_turn")
    # -> {"cmd": "action", "action": "end_turn"}

`action()` validates args against the per-action Pydantic model below before returning, so a bad
call (missing card_index, a target_index on a no-target card, an unknown action name) raises
locally instead of costing a subprocess round-trip or silently corrupting a trial.

Source: sts2-cli commit f4d7538 (2026-08-13), Program.cs (router, ExecuteAction dispatch) +
RunSimulator.cs (per-action handlers). See VERSION.json.

Two protocol facts baked into these models:
- Numeric args are parsed via GetInt32 server-side -- a JSON float throws rather than truncating,
  so every arg here is `int`, never `float`.
- Non-numeric/non-string/non-bool arg values get coerced through JsonElement.ToString()
  server-side (i.e. stringified raw JSON) -- this is why select_cards takes a comma-separated
  string, not a JSON array. SelectCardsArgs normalizes a list[int] into that string for you.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectMapNodeArgs(Args):
    """map_select only. DoMapSelect, RunSimulator.cs:922-952."""

    col: int
    row: int


class PlayCardArgs(Args):
    """combat_play only -- errors "Not in combat" otherwise. DoPlayCard, RunSimulator.cs:954-1027."""

    card_index: int
    target_index: int | None = None
    """Required iff the card's target_type is AnyEnemy and >=2 enemies are alive; omit otherwise,
    including for AllEnemies/Self/None."""


class ChooseOptionArgs(Args):
    """DoChooseOption, RunSimulator.cs:1597-1706. Dispatches by the CURRENT ROOM TYPE, not the
    decision name -- the same action means rest_site vs event_choice depending on context. Know
    which from the last decision's context.room_type; this call can't tell you."""

    option_index: int


class SelectCardRewardArgs(Args):
    """card_reward (post-combat and event-triggered). DoSelectCardReward, RunSimulator.cs:1210-1253.
    card_index indexes _pendingCardReward.Cards -- a DIFFERENT index space than card_select's
    cards, despite the similar name."""

    card_index: int


class BuyCardArgs(Args):
    """shop. DoBuyCard, RunSimulator.cs:1275-1301."""

    card_index: int


class BuyRelicArgs(Args):
    """shop. DoBuyRelic, RunSimulator.cs:1303-1352. May yield into card_select/card_reward/
    bundle_select mid-purchase (e.g. Kifuda enchant) -- the reply isn't guaranteed to be shop."""

    relic_index: int


class BuyPotionArgs(Args):
    """shop. DoBuyPotion, RunSimulator.cs:1353-1381."""

    potion_index: int


class SelectBundleArgs(Args):
    """bundle_select. DoSelectBundle, RunSimulator.cs:1422-1443."""

    bundle_index: int


class SelectCardsArgs(Args):
    """card_select. DoSelectCards, RunSimulator.cs:1445-1486."""

    indices: str = Field(pattern=r"^[0-9]+(,[0-9]+)*$")

    @field_validator("indices", mode="before")
    @classmethod
    def _normalize(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return ",".join(str(i) for i in v)
        return v


class UsePotionArgs(Args):
    """Valid in combat and elsewhere. DoUsePotion, RunSimulator.cs:1500-1579."""

    potion_index: int
    """The potion's reported "index" from PlayerSummary.potions, not its position in that
    (gap-filtered) array."""
    target_index: int | None = None
    """Required iff the potion's target_type is AnyEnemy and >=2 enemies are alive.
    Self/TargetedNoCreature potions (Flex, Fortifier) ignore this and always hit the player --
    don't rely on omitting it to redirect a self-targeting potion."""


class DiscardPotionArgs(Args):
    """DoDiscardPotion, RunSimulator.cs:1581-1595. Same index-not-position caveat as use_potion."""

    potion_index: int


# end_turn: combat_play. DoEndTurn, RunSimulator.cs:1038-1208. SILENT NO-OP (returns the current
#   decision unchanged, not an error) if a card_select/card_reward/bundle_select is pending --
#   diff state to detect this, don't trust the response type alone.
# remove_card: shop, uses the shop's fixed removal entry (card_removal_cost).
#   RunSimulator.cs:1383-1420. Opens a card_select to pick which deck card to remove.
# leave_room: non-combat rooms (rest_site/shop/event_choice/treasure) -- forces to map_select.
#   RunSimulator.cs:1708-1730.
_ARGS_MODELS: dict[str, type[Args] | None] = {
    "select_map_node": SelectMapNodeArgs,
    "play_card": PlayCardArgs,
    "end_turn": None,
    "choose_option": ChooseOptionArgs,
    "select_card_reward": SelectCardRewardArgs,
    "skip_card_reward": None,  # card_reward, only when can_skip is true. RunSimulator.cs:1255-1273.
    "buy_card": BuyCardArgs,
    "buy_relic": BuyRelicArgs,
    "buy_potion": BuyPotionArgs,
    "remove_card": None,
    "select_bundle": SelectBundleArgs,
    "select_cards": SelectCardsArgs,
    "skip_select": None,  # card_select, cancels the pending selection. RunSimulator.cs:1488-1498.
    "use_potion": UsePotionArgs,
    "discard_potion": DiscardPotionArgs,
    "leave_room": None,
    "proceed": None,  # post-boss-combat act advance, or a generic terminal-rewards proceed.
}


def action(name: str, **kwargs: Any) -> dict:
    """Build and validate an outgoing {"cmd": "action", ...} command ready to send to sts2-cli."""
    if name not in _ARGS_MODELS:
        raise ValueError(f"unknown action {name!r}")
    args_cls = _ARGS_MODELS[name]
    cmd: dict = {"cmd": "action", "action": name}
    if args_cls is None:
        if kwargs:
            raise ValueError(f"{name!r} takes no args, got {kwargs!r}")
        return cmd
    cmd["args"] = args_cls(**kwargs).model_dump(exclude_none=True)
    return cmd
