"""Sanity check: does a BufferWorker's underlying Sts2BufferSession actually come up and play a
full MainlineJob run end to end -- buffer-only, network-free choose_generic_action, driven by
golden_compass's/choices_paradox's tier-list heuristics for map/card decisions and plain random
choice for combat_play?

Drives the job directly (job.execute(worker._work_station)) rather than through
BufferWorker.run()'s queue loop -- this script only needs to confirm the worker's session comes
up and a job runs against it, not the multiprocessing plumbing around it.

Usage:
    python debug_worker.py [--character Ironclad] [--seed SEED] [--ascension 0]
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sts2_path  # noqa: E402,F401
from installation import ensure_setup  # noqa: E402
from checkpoint import Checkpoint  # noqa: E402
from jobs import MainlineJob  # noqa: E402

TEST_CHECKPOINT_DIR = "data/test_checkpoints"
from workers import BufferWorker  # noqa: E402
from utils import random_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed", default="test1234", help="defaults to a fresh random_seed()")
    parser.add_argument("--ascension", type=int, default=0)
    args = parser.parse_args()

    ensure_setup()
    seed = args.seed or random_seed()
    main_id = f"{args.character}_{seed}"

    stale = glob.glob(os.path.join(TEST_CHECKPOINT_DIR, "*.json"))
    for f in stale:
        os.remove(f)
    print(f"Cleaned {len(stale)} stale checkpoint(s) from {TEST_CHECKPOINT_DIR}/")

    print(f"Instantiating BufferWorker...")
    worker = BufferWorker(
        job_queue=mp.Queue(), priority_job_queue=mp.Queue(), log_queue=mp.Queue(),
        result_queue=mp.Queue(), worker_index=0,
    )

    job = MainlineJob(Checkpoint.fresh(main_id, args.character, seed, args.ascension),
                      save_dir=TEST_CHECKPOINT_DIR, keep_room_checkpoints=True)

    print(f"Running MainlineJob {main_id} against the worker's session...")
    try:
        result = job.execute(worker._work_station)
    finally:
        worker.close_work_station()

    print("\n=== Result ===")
    print(f"main_id:        {result.main_id}")
    print(f"outcome:        {result.outcome}")
    print(f"actions taken:  {len(result.action_log)}")
    print(f"room checkpoints: {len(result.auxiliary_data['room_checkpoints'])}")
    for c in result.auxiliary_data["combats"]:
        print(f"  combat {c['combat_index']} (act {c['act']} floor {c['floor']}, {c['enemies']}): {c['outcome']}, "
              f"lost {c['loss']}/{c['max_hp']} from full over {len(c['turns'])} turns, entered at {c['entry_hp']}")


if __name__ == "__main__":
    main()
