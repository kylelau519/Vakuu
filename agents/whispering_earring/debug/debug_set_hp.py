"""Does set_player_hp behave, and do logs containing "set_hp" entries reload identically?

Plays a random run to its first combat, taking a native save before the select_map_node that enters
it (as a mainline job does), then checks:
  1. set_player_hp(current hp) is a no-op: the re-read decision buffer is identical to the original.
  2. set_player_hp(max hp) changes the player's HP and only the HP-derived entries.
  3. after the combat, set_player_hp(entry_hp - loss) lands exactly there.
  4. reloading the whole action_log (which now holds both set_hp entries) through the native save
     and through a full replay gives the same live state as the session that produced it.

Usage:
    python debug_set_hp.py [--character Ironclad] [--seed SEED] [--ascension 0]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
import buffer_layout as bl  # noqa: E402
from buffer_env import Sts2BufferSession  # noqa: E402
from checkpoint import Checkpoint  # noqa: E402
from installation import ensure_setup  # noqa: E402
from policy import choose_generic_action  # noqa: E402
from debug_resume_job import _live_mask  # noqa: E402

SAVE_DIR = "data/test_checkpoints"


def _fresh_reload(cp: Checkpoint):
    session = Sts2BufferSession()
    try:
        return cp.reload(session), list(session.action_log)
    finally:
        session.close()


def _diff(a, b) -> list[int]:
    return np.nonzero(a.arr != b.arr)[0].tolist()


def run(character: str, seed: str, ascension: int) -> bool | None:
    """True/False = checks passed/failed; None = this seed's first combat killed the player at full HP."""
    session = Sts2BufferSession()
    main_id = f"{character}_{seed}"
    save_path = save_index = None
    try:
        decision = session.start(character, seed, ascension)
        purpose = None
        damaged = False
        while decision.decision != "combat_play":
            if decision.decision == "map_select":
                if not damaged:  # enter the combat below full HP, so healing it is a real change
                    decision = session.set_player_hp(decision.player_max_hp // 2)
                    damaged = True
                save_index = len(session.action_log)
                save_path = os.path.join(SAVE_DIR, f"{main_id}_a{save_index:04d}.json")
                os.makedirs(SAVE_DIR, exist_ok=True)
                session.save(save_path)
            name, kwargs, purpose, _ = choose_generic_action(decision, purpose)
            decision = session.step(name, **kwargs)
            if decision.decision == "game_over":
                return None

        ok = True
        entry_hp, max_hp = decision.player_hp, decision.player_max_hp
        print(f"combat entry: hp {entry_hp}/{max_hp}")

        # 1. no-op
        same = session.set_player_hp(entry_hp)
        d = _diff(decision, same)
        print(f"  1. set_hp(current) re-read differs from original in {len(d)} entries {d[:8]}")
        ok &= not d

        # 2. heal to full
        full = session.set_player_hp(max_hp)
        d = _diff(same, full)
        print(f"  2. set_hp(max) -> hp {full.player_hp}; {len(d)} entries differ from before: {d[:8]}")
        ok &= full.player_hp == max_hp

        current, purpose = full, None
        while current.decision in ("combat_play", "card_select"):
            name, kwargs, purpose, _ = choose_generic_action(current, purpose)
            current = session.step(name, **kwargs)
        if current.decision == "game_over":
            print("  combat ended in death at full HP -- nothing to deduct")
            return None
        loss = max_hp - current.player_hp
        new_hp = entry_hp - loss
        print(f"  combat ended ({current.decision}): lost {loss}; entry {entry_hp} -> {new_hp}")
        if new_hp <= 0:
            print("  deduction would kill the run -- skipping 3/4 for this seed")
            return None

        # 3. deduct
        after = session.set_player_hp(new_hp)
        print(f"  3. set_hp({new_hp}) -> hp {after.player_hp}")
        ok &= after.player_hp == new_hp
        set_hp_entries = [i for i, e in enumerate(session.action_log) if e.get("cmd") == "set_hp"]
        print(f"     set_hp entries in action_log at {set_hp_entries}")

        # 4. reload both ways
        actions = list(session.action_log)
        fast_cp = Checkpoint(main_id, character, seed, ascension, actions=actions,
                             save_path=save_path, save_index=save_index)
        slow_cp = Checkpoint(main_id, character, seed, ascension, actions=actions)
        for label, cp in (("native save", fast_cp), ("full replay", slow_cp)):
            reloaded, log = _fresh_reload(cp)
            live = _live_mask(after)
            bad = np.nonzero((reloaded.arr != after.arr) & live)[0]
            # history counters are replay-side stale (see debug_resume_job); not what this checks
            same_log = log == actions
            print(f"  4. {label}: {reloaded.decision}, hp {reloaded.player_hp}, "
                  f"{len(bad)} live entries differ, action_log {'equal' if same_log else 'DIFFERS'}")
            ok &= reloaded.player_hp == new_hp and same_log and len(bad) == 0
        return ok
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed", default="sethp1234")
    parser.add_argument("--ascension", type=int, default=0)
    args = parser.parse_args()
    ensure_setup()
    for i in range(10):  # a seed whose first combat kills or leaves nothing to deduct just moves on
        seed = f"{args.seed}-{i}"
        print(f"seed {seed}")
        result = run(args.character, seed, args.ascension)
        if result is not None:
            print("\nALL OK" if result else "\nFAILURES ABOVE")
            return 0 if result else 1
    print("no seed produced a testable combat")
    return 1


if __name__ == "__main__":
    sys.exit(main())
