"""The macro decisions of a mainline or an eval run: the Golden Compass (policy.choose_action on the run-level buffer) when
GOLDEN_COMPASS_CHECKPOINT names a checkpoint, else macro.py's stage-1 heuristics. Screens the agents never decide (rewards, treasure,
plain proceed) are always macro.py's."""
from __future__ import annotations

import os
import random

from vakuu.buffer.run_encode import encode_run, to_host_action
from vakuu.golden_compass import policy as P
from vakuu.macro import heuristics as macro


class _Session:
    def __init__(self, character: str, ascension: int):
        self.character, self.ascension = character, ascension


class MacroPlayer:
    def __init__(self, rng: random.Random, ascension: int):
        ck = os.environ.get("GOLDEN_COMPASS_CHECKPOINT")
        self.use_compass = bool(ck) and os.path.exists(ck)
        self.rng, self.asc = rng, ascension
        self.purpose: str | None = None
        macro.reset()  # a new run: no card reward has been skipped yet

    def act(self, d, same: int = 0) -> str:
        if not self.use_compass:
            return macro.choose(d, self.rng, repeat=same)
        dec = encode_run(d)
        if dec is None or (d.surface == "event" and same >= 3 and "proceed" in d.legal):
            return macro.choose(d, self.rng, repeat=same)
        session = _Session(d.obs["run"]["character"], self.asc)
        name, kwargs, _policy, self.purpose = P.choose_action(dec, session, self.purpose)
        action = to_host_action(name, kwargs, dec, d)
        if d.surface == "card_reward" and action.startswith("card_alt:"):
            macro.mark_skipped(d)
        return action
