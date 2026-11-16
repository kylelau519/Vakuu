"""Checkpoint: everything a job needs to start from some point in a run -- see the class docstring."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from buffer_env import Sts2BufferSession, BufferDecision


@dataclass
class Checkpoint:
    """Everything a job needs to start from some point in a run -- the single resume shape.

    `actions` is the full history since StartRun. `save_path` (optional) is a native sts2-cli save
    taken after exactly `actions[:save_index]` had been played; it is always a map-level save
    (SaveCheckpoint rolls any in-room state back to just before that room's entry, and RNG isn't
    rewound, so the room-entry action itself must be replayed). reload() therefore costs
    load_save + replaying `actions[save_index:]` -- bounded by one room -- instead of the whole
    run. With no save it falls back to a full replay, and with no actions either, a fresh start.
    Use a unique save_path per save: a path reused across saves can be overwritten under a job
    that is still waiting to read it.
    """
    main_id: str
    character: str
    seed: str
    ascension: int
    actions: list[dict] = field(default_factory=list)
    save_path: str | None = None
    save_index: int = 0

    def __post_init__(self):
        assert 0 <= self.save_index <= len(self.actions), (
            f"save_index {self.save_index} outside actions (len {len(self.actions)})"
        )

    @classmethod
    def fresh(cls, main_id: str, character: str, seed: str, ascension: int) -> Checkpoint:
        return cls(main_id, character, seed, ascension)

    def with_actions(self, actions: list[dict]) -> Checkpoint:
        """Same save, longer history -- the checkpoint of a later point in the same run."""
        return replace(self, actions=list(actions))

    def saved_from(self, session, save_path: str) -> Checkpoint:
        """The checkpoint for `session`'s current point, given a native save just written to
        `save_path` -- i.e. after everything in session.action_log so far."""
        actions = list(session.action_log)
        return replace(self, actions=actions, save_path=save_path, save_index=len(actions))

    def reload(self, session) -> BufferDecision:
        """Brings `session` to this checkpoint's state; session.action_log ends up == self.actions."""
        if self.save_path is not None:
            suffix = self.actions[self.save_index:]
            if suffix:
                decision = session.load_and_resume_combat(self.save_path, suffix[0], suffix[1:])
            else:
                decision = session.load_save(self.save_path)
            session.action_log = self.actions[:self.save_index] + session.action_log
        elif self.actions:
            decision = session.open_from_checkpoint(
                self.character, self.seed, self.ascension, self.actions)
        else:
            decision = session.start(self.character, self.seed, self.ascension)
        return decision
