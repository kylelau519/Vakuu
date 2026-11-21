"""The macro decisions of a mainline or an eval run: the Golden Compass (policy.choose_action on the run-level buffer) when
GOLDEN_COMPASS_CHECKPOINT names a checkpoint, else macro.py's stage-1 heuristics. Screens the agents never decide (rewards, treasure,
plain proceed) are always macro.py's."""
from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "whispering_earring")):
    if p not in sys.path:
        sys.path.insert(0, p)
import sts2_path  # noqa: F401,E402
import policy as P  # noqa: E402

from spire import macro  # noqa: E402
from spire.run_encode import encode_run, to_host_action  # noqa: E402


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
