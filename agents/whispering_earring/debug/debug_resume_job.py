"""Does Checkpoint.reload() (native save + replay of the suffix) reconstruct EXACTLY the state a full
replay of the same actions does -- including from a checkpoint with nothing left to replay, and from
one several actions into a room?

Plays one MainlineJob run (which saves a native checkpoint at every map_select), then for a spread of
those checkpoints compares, on a fresh session each:
  fast: Checkpoint.reload() through the native save
  slow: the same actions, no save (full open_from_checkpoint replay)
and asserts the live decoded buffer and session.action_log are identical (history counters are
reported as WARN only -- see _HISTORY_START). Each checkpoint is checked at
its own point (empty suffix) and at several offsets past it (suffix = the room-entry action plus
whatever followed, i.e. mid-combat/mid-event states).

Usage:
    python debug_resume_job.py [--character Ironclad] [--seed SEED] [--ascension 0] [--max-checkpoints 6]
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sts2_path  # noqa: E402,F401
import buffer_layout as bl  # noqa: E402
from buffer_env import Sts2BufferSession  # noqa: E402
from checkpoint import Checkpoint  # noqa: E402
from installation import ensure_setup  # noqa: E402
from jobs import MainlineJob  # noqa: E402

TEST_CHECKPOINT_DIR = "data/test_checkpoints"
from workers import BufferWorker  # noqa: E402


def _reload(cp: Checkpoint):
    session = Sts2BufferSession()
    try:
        decision = cp.reload(session)
        return decision, list(session.action_log)
    finally:
        session.close()


# Decision sections that follow RUN_CONTEXT in the buffer. Only the one matching the current
# decision is live; the others hold whatever was last written there (a full replay passes through
# Neow's event_choice and leaves its -1 padding behind, a native load never does), so they are
# excluded from the comparison -- they are stale padding, not state.
_SECTIONS = {
    "combat_play": (bl.COMBAT_PLAY, bl.COMBAT_PLAY_SIZE),
    "card_select": (bl.CARD_SELECT, bl.CARD_SELECT_SIZE),
    "shop": (bl.SHOP, bl.SHOP_SIZE),
    "rest_site": (bl.REST_SITE, bl.REST_SITE_SIZE),
    "card_reward": (bl.CARD_REWARD, bl.CARD_REWARD_SIZE),
    "event_choice": (bl.EVENT_CHOICE, bl.EVENT_CHOICE_SIZE),
    "event_card_reward": (bl.EVENT_CARD_REWARD, bl.EVENT_CARD_REWARD_SIZE),
    "bundle_select": (bl.BUNDLE_SELECT, bl.BUNDLE_SELECT_SIZE),
    "game_over": (bl.GAME_OVER, bl.GAME_OVER_SIZE),
}


def _live_mask(decision) -> np.ndarray:
    """True for every buffer entry that is real state for this decision: everything before the
    per-decision sections (header, player, map, run context) plus the active decision's own."""
    mask = np.zeros(len(decision.arr), dtype=bool)
    mask[:bl.COMBAT_PLAY] = True
    if decision.decision in _SECTIONS:
        start, size = _SECTIONS[decision.decision]
        mask[start:start + size] = True
    return mask


def _section_of(i: int) -> str:
    if i < bl.PLAYER_SUMMARY + bl.PLAYER_SUMMARY_SIZE:
        return "header/player"
    if i < bl.RUN_CONTEXT:
        return "map"
    if i < bl.COMBAT_PLAY:
        return "run_context"
    for name, (start, size) in _SECTIONS.items():
        if start <= i < start + size:
            return f"{name}+{i - start}"
    return "?"


# Creature history counters (HistoryCounts) are reported separately: on a full replay they carry
# stale this_combat counts over from earlier combats (e.g. monster_move_performed=7 at the very
# first decision of a combat), so they differ from a native-save reload for a reason that is the
# replay path's, not the checkpoint's. See the findings noted when this script was written.
_HISTORY_START = bl.COMBAT_PLAY + bl._CP_PLAYER_HISTORY_OFFSET
_HISTORY_END = bl.COMBAT_PLAY + bl.COMBAT_PLAY_SIZE


def _compare(label: str, fast: Checkpoint, slow: Checkpoint) -> bool:
    f_dec, f_log = _reload(fast)
    s_dec, s_log = _reload(slow)
    problems = []
    warnings = []
    if f_dec.decision != s_dec.decision:
        problems.append(f"decision type {f_dec.decision!r} != {s_dec.decision!r}")
    else:
        live = _live_mask(f_dec)
        history = np.zeros(len(live), dtype=bool)
        if f_dec.decision == "combat_play":
            history[_HISTORY_START:_HISTORY_END] = True
        differs = (f_dec.arr != s_dec.arr) & live
        n_history = int((differs & history).sum())
        diff = np.nonzero(differs & ~history)[0]
        if n_history:
            warnings.append(f"{n_history} history-counter entries differ (replay-side stale counts)")
        if len(diff):
            shown = ", ".join(
                f"[{i}]{_section_of(i)} fast={f_dec.arr[i]:g} slow={s_dec.arr[i]:g}" for i in diff[:6]
            )
            problems.append(f"{len(diff)} live buffer entries differ ({f_dec.decision}): {shown}")
    if f_log != s_log:
        problems.append(f"action_log differs (fast {len(f_log)} vs slow {len(s_log)} entries)")
    status = "FAIL" if problems else ("WARN" if warnings else "OK  ")
    print(f"  {status} {label} ({f_dec.decision}, {len(slow.actions)} actions)"
          + "".join("\n       " + m for m in problems + warnings))
    return not problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed", default="resume1234")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-checkpoints", type=int, default=6)
    args = parser.parse_args()

    ensure_setup()
    for f in glob.glob(os.path.join(TEST_CHECKPOINT_DIR, "*.json")):
        os.remove(f)

    main_id = f"{args.character}_{args.seed}"
    worker = BufferWorker(mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue(), 0)
    try:
        result = MainlineJob(Checkpoint.fresh(main_id, args.character, args.seed, args.ascension),
                             save_dir=TEST_CHECKPOINT_DIR, keep_room_checkpoints=True).execute(
            worker._work_station)
    finally:
        worker.close_work_station()

    checkpoints: list[Checkpoint] = result.auxiliary_data["room_checkpoints"]
    full = result.action_log
    print(f"MainlineJob {main_id}: {result.outcome}, {len(full)} actions, {len(checkpoints)} saved checkpoints")
    if not checkpoints:
        print("no checkpoints were saved -- nothing to verify")
        return 1

    step = max(1, len(checkpoints) // args.max_checkpoints)
    ok = True
    for i in range(0, len(checkpoints), step):
        cp = checkpoints[i]
        print(f"checkpoint r{i + 1:03d} (save_index={cp.save_index}):")
        for offset in (0, 1, 2, 5, 12):
            k = cp.save_index + offset
            if k > len(full):
                break
            fast = dataclasses.replace(cp, actions=full[:k])
            slow = Checkpoint(cp.main_id, cp.character, cp.seed, cp.ascension, actions=full[:k])
            ok &= _compare(f"+{offset} actions past the save", fast, slow)
    print("\nALL OK" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
