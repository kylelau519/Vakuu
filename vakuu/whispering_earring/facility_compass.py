"""Golden Compass data generation on spire-zero: a pool of worker processes, each playing whole real-HP runs (Claws in combat, the
compass deciding the rest) from floor 0 (or from an act-N save) and writing raw_dir/{main_id}.json for golden_compass/materialize.py.

    CLAWS_CHECKPOINT=... GOLDEN_COMPASS_CHECKPOINT=... python -m vakuu.whispering_earring.facility_compass --trials 100 --branch-workers 8 \
        --ascension 1 --raw-dir data/spire/raw_runs
"""
from __future__ import annotations

import argparse
import os
import random
import time

from vakuu.whispering_earring.workers import act_saves, end_pool, exit_now, make_pool


def _work(task: tuple) -> dict:
    save, main_id, asc, raw_dir, character, seed, fork_dir, card_forks, map_forks = task
    import torch
    torch.set_num_threads(1)
    from vakuu.golden_compass.job import CompassRun
    try:
        r = CompassRun(save, main_id, asc, raw_dir, character=character, seed=seed, fork_dir=fork_dir, card_forks=card_forks, map_forks=map_forks).run()
        return {"main_id": main_id, "outcome": r["outcome"], "error": r["error"], "floor": r["total_floor"], "act": r["end_act"], "seconds": r["seconds"]}
    except BaseException as e:  # noqa: BLE001
        return {"main_id": main_id, "outcome": "error", "error": f"{type(e).__name__}: {e}", "floor": 0, "act": 0, "seconds": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--branch-workers", type=int, default=8, help="worker processes")
    ap.add_argument("--ascension", type=int, default=1)
    ap.add_argument("--raw-dir", default="data/spire/raw_runs")
    ap.add_argument("--ckpt-dir", default="data/act_checkpoints")
    ap.add_argument("--resume-act", type=int, default=0)
    ap.add_argument("--fork-dir", default=None, help="keep fork points of the runs here (hosts them in SL mode); read by facility_branch.py --points")
    ap.add_argument("--forks-per-run", type=int, default=1, help="card-reward fork points kept per run (with --fork-dir)")
    ap.add_argument("--map-forks-per-run", type=int, default=1, help="route fork points kept per run (with --fork-dir)")
    ap.add_argument("--characters", nargs="*", default=["IRONCLAD"])
    args, _ = ap.parse_known_args()
    os.makedirs(args.raw_dir, exist_ok=True)
    rng = random.Random(os.urandom(4))
    tasks = []
    if args.resume_act > 0:
        pool_files = act_saves(args.ckpt_dir, args.resume_act)
        for _ in range(args.trials):
            src = rng.choice(pool_files)
            tasks.append((src, f"{os.path.basename(src)[:-5]}_r{os.urandom(3).hex()}", args.ascension, args.raw_dir, None, 1, args.fork_dir, args.forks_per_run, args.map_forks_per_run))
    else:
        for _ in range(args.trials):
            ch = rng.choice(args.characters).upper()
            seed = rng.randrange(1, 2**31 - 1)
            tasks.append((None, f"{ch.capitalize()}_{seed}", args.ascension, args.raw_dir, ch, seed, args.fork_dir, args.forks_per_run, args.map_forks_per_run))
    done = err = 0
    t0 = time.time()
    from concurrent.futures import as_completed
    pool = make_pool(args.branch_workers)
    try:
        futs = [pool.submit(_work, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            done += 1
            err += 1 if r["error"] else 0
            if r["error"]:
                print(f"  ERROR {r['main_id']}: {r['error']}", flush=True)
            print(f"runs {done}/{args.trials} | err {err} | last: {r['main_id']} {r['outcome']} act {r['act']} floor {r['floor']} ({r['seconds']:.0f}s) | {time.time() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        print("stopping (in-flight runs are abandoned)", flush=True)
    finally:
        end_pool(pool)
    exit_now()


if __name__ == "__main__":
    main()
