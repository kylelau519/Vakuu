"""Tests for schema/models.py.

`test_fixture_*` run against schema/tests/fixtures/ironclad_run_01.jsonl -- a real captured
sts2-cli run, not hand-written data (see fixtures/README.md). That's the test that actually
matters: it's what caught the Card.rarity bug (absent on PlayerSummary.deck cards) that reading
RunSimulator.cs alone missed. The hand-built samples further down cover decision types the fixture
run never reached (card_select, bundle_select, rest_site, shop) -- they're unverified against real
output and should be replaced once a captured run reaches them.
"""
import collections
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schema.models import (
    BundleSelect,
    CardReward,
    CardSelect,
    CombatPlay,
    EventChoice,
    GameOver,
    MapSelect,
    RestSite,
    Shop,
    parse_decision,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ironclad_run_01.jsonl"


def _load_fixture() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def test_fixture_all_decisions_parse():
    raw = _load_fixture()
    assert raw, "fixture file is empty -- did the copy from scratchpad go wrong?"
    for i, d in enumerate(raw):
        parse_decision(d)  # raises ValidationError on any mismatch; that's the test


def test_fixture_decision_type_counts():
    # Regression guard on the fixture itself -- if this drifts, fixtures/README.md's provenance
    # note is now wrong and needs updating alongside the fixture.
    counts = collections.Counter(d["decision"] for d in _load_fixture())
    assert counts == {
        "combat_play": 70,
        "map_select": 4,
        "card_reward": 3,
        "event_choice": 1,
        "game_over": 1,
    }


def test_fixture_decisions_get_the_right_type():
    type_for = {
        "combat_play": CombatPlay,
        "map_select": MapSelect,
        "card_reward": CardReward,
        "event_choice": EventChoice,
        "game_over": GameOver,
    }
    for d in _load_fixture():
        parsed = parse_decision(d)
        assert isinstance(parsed, type_for[d["decision"]])


def test_fixture_combat_play_hand_cards_have_rarity():
    combat_plays = [d for d in _load_fixture() if d["decision"] == "combat_play" and d["hand"]]
    assert combat_plays, "no combat_play decision in the fixture had a nonempty hand"
    parsed = parse_decision(combat_plays[0])
    assert isinstance(parsed, CombatPlay)
    assert all(card.rarity is not None for card in parsed.hand)


def test_fixture_deck_cards_have_no_rarity_but_do_have_upgraded():
    # The actual regression case: PlayerSummary.deck cards omit "rarity" entirely in real output,
    # unlike hand/reward/shop cards. Card.rarity must stay Optional at the base-class level, and
    # HandCard is the one that re-requires it.
    for d in _load_fixture():
        parsed = parse_decision(d)
        deck = getattr(parsed.player, "deck", None) if hasattr(parsed, "player") else None
        if deck:
            assert all(card.rarity is None for card in deck)
            assert all(card.upgraded is not None for card in deck)
            return
    pytest.fail("no decision in the fixture carried a nonempty player.deck")


def test_fixture_game_over_hp_is_present_and_zero_on_loss():
    game_overs = [d for d in _load_fixture() if d["decision"] == "game_over"]
    assert game_overs
    parsed = parse_decision(game_overs[0])
    assert isinstance(parsed, GameOver)
    if not parsed.victory:
        assert parsed.player.hp == 0


def test_unmodeled_extra_field_does_not_break_parsing():
    raw = next(d for d in _load_fixture() if d["decision"] == "game_over")
    raw = {**raw, "some_future_field_sts2_cli_adds_later": {"nested": True}}
    parsed = parse_decision(raw)
    assert parsed.model_extra["some_future_field_sts2_cli_adds_later"] == {"nested": True}


def test_missing_required_field_raises():
    raw = next(d for d in _load_fixture() if d["decision"] == "game_over")
    broken = {k: v for k, v in raw.items() if k != "victory"}
    with pytest.raises(ValidationError):
        parse_decision(broken)


def test_unrecognized_decision_name_raises():
    with pytest.raises(ValidationError):
        parse_decision({"type": "decision", "decision": "treasure"})


# --- hand-built samples: decision types the fixture run never reached. Not verified against real
# output -- see fixtures/README.md. Replace with real captures when a run reaches these.


def test_card_select_sample_parses():
    sample = {
        "type": "decision",
        "decision": "card_select",
        "cards": [
            {"index": 0, "name": "Strike", "cost": 1, "type": "Attack", "rarity": "Basic",
             "description": "Deal 6 damage.", "upgraded": False},
        ],
        "min_select": 1,
        "max_select": 1,
        "player": {"name": "Ironclad", "hp": 68, "max_hp": 80, "gold": 99, "relics": [], "potions": []},
    }
    parsed = parse_decision(sample)
    assert isinstance(parsed, CardSelect)


def test_bundle_select_sample_parses():
    sample = {
        "type": "decision",
        "decision": "bundle_select",
        "bundles": [
            {"index": 0, "cards": [
                {"name": "Defend", "cost": 1, "type": "Skill", "description": "Gain 5 Block."},
            ]},
        ],
        "player": {"name": "Ironclad", "hp": 68, "max_hp": 80, "gold": 99, "relics": [], "potions": []},
    }
    parsed = parse_decision(sample)
    assert isinstance(parsed, BundleSelect)


def test_rest_site_sample_parses():
    sample = {
        "type": "decision",
        "decision": "rest_site",
        "options": [
            {"index": 0, "option_id": "REST_HEAL", "name": "RestSiteHealOption", "is_enabled": True},
        ],
    }
    parsed = parse_decision(sample)
    assert isinstance(parsed, RestSite)


def test_shop_sample_parses():
    sample = {
        "type": "decision",
        "decision": "shop",
        "cards": [
            {"index": 0, "name": "Anger", "cost": 1, "type": "Attack", "rarity": "Common",
             "description": "Deal 8 damage.", "card_cost": 1, "cost": 50,
             "is_stocked": True, "on_sale": False},
        ],
        "relics": [{"index": 0, "name": "Vajra", "cost": 150, "is_stocked": True}],
        "potions": [{"index": 0, "name": "Fire Potion", "cost": 20, "is_stocked": True}],
        "player": {"name": "Ironclad", "hp": 68, "max_hp": 80, "gold": 99, "relics": [], "potions": []},
    }
    parsed = parse_decision(sample)
    assert isinstance(parsed, Shop)
