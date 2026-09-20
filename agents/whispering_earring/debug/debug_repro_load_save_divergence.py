"""Does concurrent load_save (many processes loading the SAME save file, then taking the same
scripted action sequence) diverge the way concurrent StartRun-replay did (see
debug_repro_branch_divergence.py)? That divergence was traced to no per-process user-data-dir
isolation racing on a shared save/profile path during *early-run generation* (map/RNG setup
consulted once at StartRun time). load_save resumes from a save whose RNG state
(SerializableRun.SerializableRng) is already embedded in the file, so it shouldn't need to
re-touch that shared early-game path -- this checks that empirically instead of assuming it.

Usage:
    python debug_repro_load_save_divergence.py <save.json> [--n N] [--post-actions-json '[...]']

    <save.json>   A save file written by write_continue_save (see debug_bench_save_load.py).
    --n N         Concurrent worker processes (default 10).
    --post-actions-json  Optional JSON list of {cmd,action,args?} dicts to replay after load_save,
                          to also exercise re-entering a room/combat off the loaded state.
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
from env import Sts2Session, Sts2ActionError, _to_decision  # noqa: E402


def _worker(save_path: str, post_actions: list[dict], worker_id: int, result_queue: "mp.Queue") -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        session = Sts2Session()
        raw = session._send_raw({"cmd": "load_save", "path": save_path})
        decision = _to_decision(raw)
        trace = [(decision.decision, decision.player.hp)]
        for entry in post_actions:
            raw = session._send_raw(entry)
            decision = _to_decision(raw)
            trace.append((decision.decision, decision.player.hp))
        session.close()
        result_queue.put((worker_id, trace, None))
    except Exception as ex:  # noqa: BLE001
        result_queue.put((worker_id, None, str(ex)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("save_json")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--post-actions-json", default="[]")
    args = parser.parse_args()

    post_actions = json.loads(args.post_actions_json)
    ensure_setup()

    print(f"Loading {args.save_json} across {args.n} concurrent workers"
          f"{f' + {len(post_actions)} post-load action(s)' if post_actions else ''}...")

    result_queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=_worker, args=(args.save_json, post_actions, i, result_queue))
        for i in range(args.n)
    ]
    for p in procs:
        p.start()
    results = [result_queue.get() for _ in procs]
    for p in procs:
        p.join()

    results.sort(key=lambda r: r[0])
    errored = [r for r in results if r[2] is not None]
    for worker_id, _, error in errored:
        print(f"  worker {worker_id}: ERROR -- {error}")

    ok = [r for r in results if r[2] is None]
    if not ok:
        print("All workers errored.")
        sys.exit(1)

    baseline_id, baseline_trace, _ = ok[0]
    n_diverged = 0
    for worker_id, trace, _ in ok[1:]:
        if trace == baseline_trace:
            print(f"  worker {worker_id}: matches worker {baseline_id} bit-for-bit: {trace}")
        else:
            n_diverged += 1
            print(f"  worker {worker_id}: DIVERGED from worker {baseline_id}: {trace} vs {baseline_trace}")
    print(f"\n{len(errored)} errored, {n_diverged}/{len(ok) - 1} diverged from baseline "
          f"(out of {args.n} total).")
    sys.exit(1 if (errored or n_diverged) else 0)


if __name__ == "__main__":
    main()
