"""Cross-process registry for deduplicating forks by outcome state (see state_key.py).

One instance per mainline run, shared by the mainline job and every fork job of that run through a
multiprocessing Manager dict. The mainline registers the state it reached at the end of every turn
first, so it always keeps its own member; a fork keeps its member only if it is the first to reach
its end-of-turn state."""
from __future__ import annotations


class ForkDedup:
    def __init__(self, shared_dict):
        self._d = shared_dict

    def claim(self, combat_index: int, turn_index: int, key: str, uid: str) -> str:
        """Registers `uid` as the owner of `key` in this group unless someone already owns it.
        Returns the owner's uid: == `uid` means the caller is the first (keep), else a duplicate of it.
        Atomic: the Manager serializes setdefault calls."""
        return self._d.setdefault(f"c{combat_index:03d}t{turn_index:02d}|{key}", uid)
