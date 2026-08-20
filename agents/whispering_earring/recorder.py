"""Compact per-trial recorder, passed to Sts2Session(logger=...).

Unlike a full state-per-step trace, this keeps no state at all: full state at any decision
point is always reconstructible later by replaying a trial's action log through a fresh
session (see checkpoint.py / Sts2Session.open_from_checkpoint), so materializing training
features is a deliberate, one-time offline step rather than something paid automatically,
in full, for every trial and every MCTS branch as it's generated.

The one thing replay *can't* recover after the fact is search statistics -- which actions
an MCTS search considered and how many visits each got -- since that information only
exists during the search itself. attach_policy() is the hook for recording it: call it
right before session.step() with the visit distribution that produced the upcoming action,
and it gets attached to that action's entry in the saved file.

Value is scoped per combat, not per trial. facility.py brackets each individual combat with
begin_combat()/end_combat(value) -- value is computed from that one combat's own HP loss
(README.md's Value head: "HP loss is used as the metric"), not the trial's eventual overall
outcome, so a clean early win and a narrow late win in the same run get different training
signal instead of being flattened to the same +1. end_combat() backfills every step recorded
since the matching begin_combat() -- the value can only be known once the combat resolves,
so it's applied retroactively to the range, not per-action like attach_policy(). Branch trials
adopted by a main line already carry their own per-combat value in their saved file;
facility.py's _absorb_combat replays those through attach_value() (an immediate, per-action
counterpart to attach_policy()) rather than recomputing it.
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
        self._combat_start: int | None = None
        self._step = start_step

    def attach_policy(self, policy: dict) -> None:
        self._pending_policy = policy

    def attach_value(self, value: float) -> None:
        """Immediate counterpart to attach_policy(): assigns `value` to the *next* logged
        step. For replaying an adopted branch's already-scored actions (see facility.py's
        _absorb_combat) -- for scoring a combat as it's actually played live, use
        begin_combat()/end_combat() instead, since the value isn't known until it resolves."""
        self._pending_value = value

    def begin_combat(self) -> None:
        self._combat_start = self._step

    def end_combat(self, value: float) -> None:
        """Backfills `value` onto every step since the matching begin_combat() -- the
        combat's outcome (and thus its value) is only known once it resolves, so unlike
        attach_policy()/attach_value() this scores retroactively rather than the next step."""
        assert self._combat_start is not None, "end_combat() with no matching begin_combat()"
        for step in range(self._combat_start, self._step):
            self._values[step] = value
        self._combat_start = None

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

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data
