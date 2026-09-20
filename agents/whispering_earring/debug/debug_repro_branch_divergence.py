"""Repro tool for the intermittent 'Sts2ActionError: No pending card selection' (and similar)
errors seen on branch trials, e.g. data/raw/Ironclad_storm-920659_f06c03_b*.json.

Findings so far (see conversation this was written from):
  - Replaying a failed branch's exact prefix (same character/seed/ascension/actions, taken from
    the parent main-line trial's action log up to the failing combat) through a fresh
    Sts2BufferSession, ALONE in one process, reproduces the ORIGINAL run bit-for-bit -- no error.
  - Running N copies of that same replay CONCURRENTLY (matching how branch_worker_loop's pool
    actually runs) makes some of them diverge from the very first RNG-driven decision (a
    different bundle/event outcome, a different monster in the first combat, etc.) -- the
    scripted actions were recorded for one trajectory and get replayed against a different one,
    eventually hitting a select_cards (or similar) for a pending prompt that no longer exists.
  - Every process, solo or concurrent, logs the identical line
    "Profile-scoped data path initialized: user://default/1/profile0". RunSimulator.cs has no
    per-process user-data-dir isolation, so every worker -- including 10-30 running at once --
    reads/writes that same on-disk save-profile path with no locking. That's the prime suspect:
    concurrent read-modify-write on shared save/profile/history state racing and perturbing
    whatever RNG/state that early game logic consults.

Usage:
    python debug_repro_branch_divergence.py <trial.json> [--n N] [--prefix-len K]

    <trial.json>   A main-line trial file (data/raw/<main_id>.json) whose `actions` will be
                    replayed as the checkpoint prefix. Pass a branch file instead if you want to
                    replay exactly what that branch replayed (its own `actions` is empty on
                    failure -- use the parent main trial, or pass --prefix-len to trim).
    --n N           How many concurrent worker processes to run (default 10, matching a
                    --branch-workers 10 pool).
    --prefix-len K  Only replay the first K actions (default: all of them).

Exits nonzero if any worker diverged (decision=="error" partway through replay), and prints a
per-worker summary of the first divergence point.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
from installation import ensure_setup  # noqa: E402
from buffer_env import Sts2BufferSession  # noqa: E402


def _worker(character: str, seed: str, ascension: int, actions: list[dict], worker_id: int,
            result_queue: mp.Queue) -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)  # silence this worker's [SIM]/[INFO] noise -- we only care about the outcome

    session = Sts2BufferSession()
    raw = session._session.StartRun(character, seed, ascension)
    for i, entry in enumerate(actions):
        args = entry.get("args")
        raw = session._session.Action(entry["action"], json.dumps(args) if args else None)
        decision = session._decision(raw)
        if decision.decision == "error":
            result_queue.put((worker_id, i, entry["action"], str(session._session.LastError)))
            return
    result_queue.put((worker_id, None, None, None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_json")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--prefix-len", type=int, default=None)
    args = parser.parse_args()

    with open(args.trial_json) as f:
        trial = json.load(f)

    actions = [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in trial["actions"]
    ]
    if args.prefix_len is not None:
        actions = actions[:args.prefix_len]

    print(f"Replaying {len(actions)} actions from {trial['character']}/{trial['seed']} "
          f"(ascension {trial['ascension']}) across {args.n} concurrent workers...")

    ensure_setup()

    result_queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=_worker, args=(trial["character"], trial["seed"], trial["ascension"],
                                          actions, i, result_queue))
        for i in range(args.n)
    ]
    for p in procs:
        p.start()
    results = [result_queue.get() for _ in procs]
    for p in procs:
        p.join()

    results.sort()
    n_failed = 0
    for worker_id, index, action_name, error in results:
        if index is None:
            print(f"  worker {worker_id}: ok (all {len(actions)} actions replayed cleanly)")
        else:
            n_failed += 1
            print(f"  worker {worker_id}: DIVERGED at action [{index}] {action_name!r} -> {error!r}")

    print(f"\n{n_failed}/{len(results)} workers diverged.")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
