"""Compact per-trial recorder, passed to Sts2Session(logger=...).

Value is scoped per turn, not per combat or per trial. facility_common.py brackets each
individual turn with begin_turn()/end_turn(value) -- every turn (including the one that ends
the combat) gets its own value, so a turn that took damage to set up a good follow-up isn't
flattened to the same scalar as every other turn in that combat. For a turn that ends the
combat, value is the combat's own terminal HP-loss outcome (README.md's Value head: "HP loss
is used as the metric"); for a turn that doesn't, it's a TD-bootstrapped estimate -- see
facility_common.py's _play_turn_only for both formulas. end_turn() backfills every step
recorded since the matching begin_turn() -- the value can only be known once the turn
resolves, so it's applied retroactively to the range, not per-action like attach_policy().
Branch trials adopted by a main line already carry their own per-turn value in their saved
file; facility_claws.py's _absorb_actions replays those through attach_value() (an immediate,
per-action counterpart to attach_policy()) rather than recomputing it.
"""
from __future__ import annotations

import json
import os


class CompactRecorder:
    def __init__(self, meta: dict, start_step: int = 0):
        self.meta = meta
        self.error: str | None = None
        self._policies: dict[int, dict] = {}
        self._pending_policy: dict | None = None
        self._values: dict[int, float] = {}
        self._pending_value: float | None = None
        self._turn_start: int | None = None
        self._step = start_step
        self._combats: list[dict] = []

    def attach_policy(self, policy: dict) -> None:
        self._pending_policy = policy

    def mark_step(self) -> int:
        """Returns the step index the *next* log_action() call will land on -- for a caller that
        needs to score one specific already-taken-or-about-to-be-taken action only once the
        surrounding play has fully resolved (see set_value), when neither attach_value()'s
        next-call semantics nor end_turn()'s whole-range backfill fit: e.g.
        facility_choices_paradox._run_meta_option_job, which commits to a card_reward option, then
        plays many further turns/combats forward before it knows what that one commit was worth --
        and, on the MAIN line's own recorder, _resolve_meta_decision's own commit_step, whose value
        isn't known until the adopted option's own job reports back asynchronously (see
        main_worker_loop's pending_meta_jobs)."""
        return self._step

    def set_value(self, step: int, value: float) -> None:
        """Retroactively assigns `value` to a specific already-logged step -- see mark_step()."""
        self._values[step] = value

    def attach_value(self, value: float) -> None:
        """Immediate counterpart to attach_policy(): assigns `value` to the *next* logged
        step. For replaying an adopted branch's already-scored actions (see facility.py's
        _absorb_actions) -- for scoring a turn as it's actually played live, use
        begin_turn()/end_turn() instead, since the value isn't known until it resolves."""
        self._pending_value = value

    def begin_turn(self) -> None:
        self._turn_start = self._step

    def end_turn(self, value: float) -> None:
        """Backfills `value` onto every step since the matching begin_turn() -- a turn's
        value (terminal outcome if it ends the combat, TD-bootstrap estimate otherwise) is
        only known once it resolves, so unlike attach_policy()/attach_value() this scores
        retroactively rather than the next step."""
        assert self._turn_start is not None, "end_turn() with no matching begin_turn()"
        for step in range(self._turn_start, self._step):
            self._values[step] = value
        self._turn_start = None

    def log_action(self, cmd: dict) -> None:
        if self._pending_policy is not None:
            self._policies[self._step] = self._pending_policy
            self._pending_policy = None
        if self._pending_value is not None:
            self._values[self._step] = self._pending_value
            self._pending_value = None
        self._step += 1

    def log_state(self, decision) -> None:
        pass  # state is derived by replay at materialization time, never stored here

    def record_combat(self, act: int, floor: int, combat_index: int, outcome: str) -> None:
        """Records one resolved combat's (act, floor, outcome) for the stats module
        (see stats.py) -- cheap, already-known-live bookkeeping, not derived by replay
        like everything else this recorder deliberately omits."""
        self._combats.append({
            "act": act, "floor": floor, "combat_index": combat_index, "outcome": outcome,
        })

    def log_error(self, message: str) -> None:
        self.error = message

    def close(self) -> None:
        pass

    def save(self, path: str, action_log: list[dict], outcome: str | None = None) -> dict:
        actions = []
        for i, cmd in enumerate(action_log):
            entry = dict(cmd)
            if i in self._policies:
                entry["policy"] = self._policies[i]
            if i in self._values:
                entry["value"] = self._values[i]
            actions.append(entry)

        data = dict(self.meta)
        data["actions"] = actions
        if outcome is not None:
            data["outcome"] = outcome
        if self.error is not None:
            data["error"] = self.error
        if self._combats:
            data["combats"] = self._combats

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data
