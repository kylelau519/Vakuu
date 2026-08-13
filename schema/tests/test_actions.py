"""Tests for schema/actions.py."""
import pytest
from pydantic import ValidationError

from schema.actions import action


@pytest.mark.parametrize(
    "name,kwargs,expected",
    [
        ("select_map_node", {"col": 2, "row": 1}, {"cmd": "action", "action": "select_map_node", "args": {"col": 2, "row": 1}}),
        ("play_card", {"card_index": 0}, {"cmd": "action", "action": "play_card", "args": {"card_index": 0}}),
        ("play_card", {"card_index": 0, "target_index": 1}, {"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 1}}),
        ("end_turn", {}, {"cmd": "action", "action": "end_turn"}),
        ("choose_option", {"option_index": 0}, {"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}),
        ("select_card_reward", {"card_index": 1}, {"cmd": "action", "action": "select_card_reward", "args": {"card_index": 1}}),
        ("skip_card_reward", {}, {"cmd": "action", "action": "skip_card_reward"}),
        ("buy_card", {"card_index": 0}, {"cmd": "action", "action": "buy_card", "args": {"card_index": 0}}),
        ("buy_relic", {"relic_index": 0}, {"cmd": "action", "action": "buy_relic", "args": {"relic_index": 0}}),
        ("buy_potion", {"potion_index": 0}, {"cmd": "action", "action": "buy_potion", "args": {"potion_index": 0}}),
        ("remove_card", {}, {"cmd": "action", "action": "remove_card"}),
        ("select_bundle", {"bundle_index": 0}, {"cmd": "action", "action": "select_bundle", "args": {"bundle_index": 0}}),
        ("select_cards", {"indices": "0,2"}, {"cmd": "action", "action": "select_cards", "args": {"indices": "0,2"}}),
        ("skip_select", {}, {"cmd": "action", "action": "skip_select"}),
        ("use_potion", {"potion_index": 0}, {"cmd": "action", "action": "use_potion", "args": {"potion_index": 0}}),
        ("use_potion", {"potion_index": 0, "target_index": 1}, {"cmd": "action", "action": "use_potion", "args": {"potion_index": 0, "target_index": 1}}),
        ("discard_potion", {"potion_index": 0}, {"cmd": "action", "action": "discard_potion", "args": {"potion_index": 0}}),
        ("leave_room", {}, {"cmd": "action", "action": "leave_room"}),
        ("proceed", {}, {"cmd": "action", "action": "proceed"}),
    ],
)
def test_valid_actions_build_expected_wire_command(name, kwargs, expected):
    assert action(name, **kwargs) == expected


def test_select_cards_accepts_a_list_and_normalizes_to_the_wire_string():
    # The exact footgun this exists to prevent: sts2-cli wants a comma-separated string, and
    # silently mis-parses a raw JSON array (RunSimulator.cs's arg coercion stringifies it).
    cmd = action("select_cards", indices=[0, 2, 5])
    assert cmd["args"]["indices"] == "0,2,5"


def test_select_cards_rejects_malformed_string():
    with pytest.raises(ValidationError):
        action("select_cards", indices="not-a-list-of-ints")


def test_unknown_action_name_raises():
    with pytest.raises(ValueError):
        action("cast_spell")


def test_missing_required_arg_raises():
    with pytest.raises(ValidationError):
        action("play_card")


def test_args_on_no_arg_action_raises():
    with pytest.raises(ValueError):
        action("end_turn", card_index=0)


def test_unexpected_kwarg_on_known_action_raises():
    with pytest.raises(ValidationError):
        action("play_card", card_index=0, some_typo_field=1)
