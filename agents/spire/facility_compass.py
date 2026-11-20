"""Golden Compass data generation on spire-zero: a pool of worker processes, each playing whole real-HP runs (Claws in combat, the
compass deciding the rest) from floor 0 (or from an act-N save) and writing raw_dir/{main_id}.json for golden_compass/materialize.py.
Same command line as whispering_earring/facility_compass.py, so golden_compass/compare.py and the training scripts can call either.

    CLAWS_CHECKPOINT=... GOLDEN_COMPASS_CHECKPOINT=... python agents/spire/facility_compass.py --trials 100 --branch-workers 8 \
        --ascension 1 --raw-dir data/spire/raw_runs
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))


def _work(task: tuple) -> dict:
    save, main_id, asc, raw_dir, character, seed = task
    import torch
    torch.set_num_threads(1)
    from spire.compass_job import CompassRun
    try:
        r = CompassRun(save, main_id, asc, raw_dir, character=character, seed=seed).run()
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
    ap.add_argument("--characters", nargs="*", default=["IRONCLAD"])
    args, _ = ap.parse_known_args()
    os.makedirs(args.raw_dir, exist_ok=True)
    rng = random.Random(os.urandom(4))
    tasks = []
    if args.resume_act > 0:
        pool_files = [p for p in sorted(glob.glob(os.path.join(args.ckpt_dir, f"*_act{args.resume_act}.json"))) if not p.endswith(".ckpt.json")]
        if not pool_files:
            raise SystemExit(f"no act-{args.resume_act} saves in {args.ckpt_dir}")
        for _ in range(args.trials):
            src = rng.choice(pool_files)
            tasks.append((src, f"{os.path.basename(src)[:-5]}_r{os.urandom(3).hex()}", args.ascension, args.raw_dir, None, 1))
    else:
        for _ in range(args.trials):
            ch = rng.choice(args.characters).upper()
            seed = rng.randrange(1, 2**31 - 1)
            tasks.append((None, f"{ch.capitalize()}_{seed}", args.ascension, args.raw_dir, ch, seed))
    done = err = 0
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    pool = ProcessPoolExecutor(max_workers=args.branch_workers, mp_context=mp.get_context("spawn"))
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
        procs = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
    for child in mp.active_children():
        try:
            child.terminate()
        except Exception:  # noqa: BLE001
            pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
