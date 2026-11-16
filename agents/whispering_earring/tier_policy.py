"""Choices Paradox placeholder policy: card-value decisions backed by Baalorlord's per-character
STS2 tier lists (see scripts/fetch_tier_lists.py and data/card_tiers.json), rather than a trained
model. Purely functional and duck-typed on card-like objects (anything with `.id` and `.index`)
so it has no dependency on sts2-cli's schema types -- callers pass in whatever card objects their
protocol client already gives them.

Strategy, as specified by the project: prioritize S-tier picks, treat F-tier as something to
avoid taking and actively remove, and let the tier list's own `good_upgrade` flag -- not tier
alone -- decide what's worth upgrading (a card can be a strong S-tier pick while its particular
upgrade is a minor stat bump not worth prioritizing).
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Protocol
import numpy as np

_DATA_PATH = Path(__file__).resolve().parent / "data" / "card_tiers.json"

_TIER_RANK = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
_UNKNOWN_RANK = _TIER_RANK["C"]  # cards missing from the tier list (curses, statuses, new
# patch content the list hasn't caught up to) are treated as neutral, not punished or favored

_TARGET_DECK_SIZE = 25  # _logit's aim_size: rejection probability climbs once the deck passes
# this, with the exact crossover pushed later for higher-ranked cards


class _CardLike(Protocol):
    index: int
    id: str | None


def _load_card_tiers() -> dict[str, dict]:
    return json.loads(_DATA_PATH.read_text())["cards"]


CARD_TIERS = _load_card_tiers()


def _entry_id(card_id: str | None) -> str | None:
    """"CARD.ECHO_FORM" -> "ECHO_FORM" (sts2-cli's card ids are "CARD.<entry>"; the tier data
    is keyed on the bare entry). None-safe: bundle_select cards omit `id` entirely."""
    if not card_id:
        return None
    return card_id.split(".", 1)[1] if "." in card_id else card_id


def _placement(card: _CardLike) -> dict | None:
    entry = _entry_id(getattr(card, "id", None))
    return CARD_TIERS.get(entry) if entry else None


def tier_of(card: _CardLike) -> str | None:
    placement = _placement(card)
    return placement["tier"] if placement else None


def rank_of(card: _CardLike) -> int:
    return _TIER_RANK.get(tier_of(card), _UNKNOWN_RANK)


def is_good_upgrade(card: _CardLike) -> bool:
    placement = _placement(card)
    return bool(placement and placement.get("good_upgrade"))


def choose_card_reward(cards: list[_CardLike], can_skip: bool, deck_size: int) -> int | None:
    """Index of the card to take, or None to skip. Takes the highest-tier option offered,
    gated by _logit's add-probability for the deck-bloat dampener: as deck_size approaches
    and passes _TARGET_DECK_SIZE the odds of adding fall off, with the crossover pushed later
    for higher-ranked cards -- an empty reward beats another card thinning out the draws."""
    if not cards:
        return None
    best = max(cards, key=rank_of)
    accept = random.uniform(0, 1) < _logit(deck_size, _TARGET_DECK_SIZE, rank_of(best))
    if not accept and can_skip:
        return None
    return best.index


def choose_shop_card(cards: list[_CardLike], deck_size: int) -> int | None:
    """Highest-tier card among an already-affordable, already-stocked candidate list, or
    None if the deck-bloat dampener (_logit's add-probability) says not to buy at all -- same
    rule as choose_card_reward, so a full deck stops absorbing gold-bought cards the same way
    it stops absorbing free ones."""
    if not cards:
        return None
    best = max(cards, key=rank_of)
    accept = random.uniform(0, 1) < _logit(deck_size, _TARGET_DECK_SIZE, rank_of(best))
    return best.index if accept else None


def choose_removal_target(deck_cards: list[_CardLike]) -> int:
    """Worst-tier card in the deck slice offered by a remove-card card_select."""
    return min(deck_cards, key=rank_of).index


def choose_upgrade_target(upgradeable_cards: list[_CardLike]) -> int:
    """Prefers a card the tier list flags good_upgrade=true; falls back to the highest-tier
    card offered if none of them carry that flag."""
    flagged = [c for c in upgradeable_cards if is_good_upgrade(c)]
    pool = flagged if flagged else upgradeable_cards
    return max(pool, key=rank_of).index


def _logit(deck_size: int, aim_size: int, rank: int) -> float:
    return 1 / (1 + np.exp(deck_size - (aim_size + rank)))
    