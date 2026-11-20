"""Claws data generation on spire-zero: a pool of worker processes, each playing a mainline from a random act-boundary save and
forking every combat turn in the Rust engine (claws_job.ClawsRun). Replaces whispering_earring/facility_claws.py: no job
queue, no reload, no sts2-cli. --branch-workers sets the number of processes (--main-workers is accepted and ignored).

    CLAWS_CHECKPOINT=... python agents/spire/facility_claws.py --trials 30 --branch-workers 17 --forks-per-turn 40 \
        --ascension 1 --raw-dir data/run2/raw --dedup-forks --ckpt-dir data/act_checkpoints --resume-act 2
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
    save, main_id, asc, raw_dir, forks, dedup, character, seed = task
    import torch
    torch.set_num_threads(1)
    from spire.claws_job import ClawsRun
    try:
        r = ClawsRun(save, main_id, asc, raw_dir, forks_per_turn=forks, dedup=dedup, character=character, seed=seed).run()
        return {"main_id": main_id, "outcome": r["outcome"], "error": r["error"], "combats": len(r["combats"]),
                "forks": r["stats"]["forks"], "dup": r["stats"]["dup"], "err": r["stats"]["err"],
                "loss": sum(c.get("loss", 0) for c in r["combats"]), "seconds": r["seconds"]}
    except BaseException as e:  # noqa: BLE001
        return {"main_id": main_id, "outcome": "error", "error": f"{type(e).__name__}: {e}", "combats": 0, "forks": 0, "dup": 0, "err": 1, "loss": 0, "seconds": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10, help="mainline runs to play")
    ap.add_argument("--branch-workers", type=int, default=8, help="worker processes (each plays a run and its forks)")
    ap.add_argument("--main-workers", type=int, default=None, help="ignored (kept so the old command lines work)")
    ap.add_argument("--forks-per-turn", type=int, default=40)
    ap.add_argument("--ascension", type=int, default=1)
    ap.add_argument("--raw-dir", default="data/run2/raw")
    ap.add_argument("--dedup-forks", action="store_true")
    ap.add_argument("--ckpt-dir", default="data/act_checkpoints")
    ap.add_argument("--resume-act", type=int, default=0, help="0: fresh runs from the start (default); N: start from a random act-N save in --ckpt-dir")
    ap.add_argument("--seeds", default=None, help="ignored")
    ap.add_argument("--characters", nargs="*", default=["IRONCLAD"], help="fresh runs only: the characters, picked at random per run")
    args, _unknown = ap.parse_known_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    rng = random.Random(os.urandom(4))
    tasks = []
    if args.resume_act > 0:
        pool_files = [p for p in sorted(glob.glob(os.path.join(args.ckpt_dir, f"*_act{args.resume_act}.json"))) if not p.endswith(".ckpt.json")]
        if not pool_files:
            raise SystemExit(f"no act-{args.resume_act} saves in {args.ckpt_dir}")
        for _ in range(args.trials):
            src = rng.choice(pool_files)
            tasks.append((src, f"{os.path.basename(src)[:-5]}_r{os.urandom(3).hex()}", args.ascension, args.raw_dir, args.forks_per_turn,
                          args.dedup_forks, None, 1))
    else:
        for _ in range(args.trials):
            ch = rng.choice(args.characters).upper()
            seed = rng.randrange(1, 2**31 - 1)
            tasks.append((None, f"{ch.capitalize()}_{seed}", args.ascension, args.raw_dir, args.forks_per_turn, args.dedup_forks, ch, seed))

    done = forks = dup = err = 0
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=args.branch_workers, mp_context=ctx)
    futures = [pool.submit(_work, t) for t in tasks]
    procs: list = []
    try:
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            forks += r["forks"]
            dup += r["dup"]
            err += r["err"] + (1 if r["error"] else 0)
            if r["error"]:
                print(f"  ERROR {r['main_id']}: {r['error']}", flush=True)
            print(f"mainlines {done}/{args.trials} | forks {forks} | dup {dup} | err {err} | last: {r['main_id']} {r['outcome']}, "
                  f"{r['combats']} combats, {r['loss']:.0f} HP lost ({r['seconds']:.0f}s) | {time.time() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        print("stopping (in-flight runs are abandoned)", flush=True)
    finally:
        procs = procs or list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for proc in procs:  # idle workers hold native threads: end them
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
    for child in mp.active_children():  # the workers (spawned by the executor) hold native threads: end them all
        try:
            child.terminate()
        except Exception:  # noqa: BLE001
            pass
    sys.stdout.flush()
    os._exit(0)  # the workers' native threads (torch, the engine) need not be waited for


if __name__ == "__main__":
    main()
