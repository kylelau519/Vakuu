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
and it gets attached to that action's entry in the saved file. Not used yet, since
policy.py's combat policy is still uniform-random with no search behind it -- wire it up
once an MCTS-driven policy exists.
"""
from __future__ import annotations

import json
import os

# Placeholder value target: the trial's terminal outcome, backed up unweighted to every
# action in the trial (plain Monte-Carlo return, no discount) -- a stand-in for a learned
# value function until one exists. "error" trials get no value: an aborted trial says
# nothing about whether the actions in it were good.
_OUTCOME_VALUE = {"won": 1.0, "died": -1.0}


class CompactRecorder:
    def __init__(self, meta: dict, start_step: int = 0):
        """start_step: position in the eventual action log (see save()) that this
        recorder's first log_action call corresponds to. 0 for a trial starting fresh;
        len(checkpoint_actions) for a branch resuming from a checkpoint, since
        open_from_checkpoint() replays the prefix directly without going through
        log_action, so this recorder never sees those positions."""
        self.meta = meta
        self.error: str | None = None
        self._policies: dict[int, dict] = {}
        self._pending_policy: dict | None = None
        self._step = start_step

    def attach_policy(self, policy: dict) -> None:
        self._pending_policy = policy

    def log_action(self, cmd: dict) -> None:
        if self._pending_policy is not None:
            self._policies[self._step] = self._pending_policy
            self._pending_policy = None
        self._step += 1

    def log_state(self, decision) -> None:
        pass  # state is derived by replay at materialization time, never stored here

    def log_error(self, message: str) -> None:
        self.error = message

    def close(self) -> None:
        pass

    def save(self, path: str, action_log: list[dict], outcome: str | None = None) -> None:
        """action_log: the session's full action_log (checkpoint prefix + steps taken
        here) -- passed in rather than tracked internally since this recorder only
        observes the steps taken through *this* session, not a replayed prefix."""
        value = _OUTCOME_VALUE.get(outcome)
        actions = []
        for i, cmd in enumerate(action_log):
            entry = dict(cmd)
            if i in self._policies:
                entry["policy"] = self._policies[i]
            if value is not None:
                entry["value"] = value
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
