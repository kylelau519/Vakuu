"""Card-reward and route branching on spire-zero (golden_compass/train.py --branch-dir reads its output files).

Scout runs play act 1 and record fork points; every offered card (and skip) of a card fork point, and every reachable node of a map
fork point, is then played `k` times for `horizon` combats with the draw pile reshuffled at each combat's start (branch_job.py). One JSON
file per fork point in --raw-dir: the encoded RunState, the options, every continuation's metrics and each option's mean.

    CLAWS_CHECKPOINT=... GOLDEN_COMPASS_CHECKPOINT=... python -m vakuu.whispering_earring.facility_branch --trials 10 --branch-workers 16 \
        --forks-per-scout 3 --horizon 6 --raw-dir data/spire/raw_branch
--trials is the number of scout runs. Workers are processes; a scout and its branches share the pool."""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

from vakuu.whispering_earring.workers import end_pool, exit_now, make_pool

METRICS = ("died", "hp_lost_share", "hp_lost", "hp_frac_end", "floors", "combats")


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _scout(task: tuple) -> dict:
    import torch
    torch.set_num_threads(1)
    from vakuu.branch.job import ScoutRun
    main_id, asc, character, seed, clone_dir = task
    try:
        return ScoutRun(main_id, asc, character, seed, clone_dir).run()
    except BaseException as e:  # noqa: BLE001
        return {"main_id": main_id, "outcome": "error", "error": f"{type(e).__name__}: {e}", "forks": [], "map_forks": [], "floor": 0}


def _branch(task: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from vakuu.branch.job import BranchRun
    try:
        return BranchRun(**task).run()
    except BaseException as e:  # noqa: BLE001
        return {"fork_id": task["fork_id"], "option": task["option"], "k": task["k"], "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10, help="scout runs")
    ap.add_argument("--branch-workers", type=int, default=8, help="worker processes")
    ap.add_argument("--ascension", type=int, default=1)
    ap.add_argument("--characters", nargs="*", default=["IRONCLAD"])
    ap.add_argument("--raw-dir", default="data/spire/raw_branch")
    ap.add_argument("--forks-per-scout", type=int, default=3, help="card fork points sampled from each scout run")
    ap.add_argument("--k", type=int, default=None, help="continuations per option; default: scaled by the deck's unique cards")
    ap.add_argument("--k-per-unique", type=float, default=2.0)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--map-forks-per-scout", type=int, default=2, help="map decisions branched per scout run (0 = none)")
    ap.add_argument("--map-k", type=int, default=4, help="continuations per reachable node at a map fork")
    ap.add_argument("--horizon", type=int, default=8, help="most combats each continuation plays; it also ends at the next act (metrics are cumulative per combat)")
    ap.add_argument("--points", default=None, help="a facility_compass.py --fork-dir: replay its fork points (every one) instead of playing scouts")
    ap.add_argument("--max-forks", type=int, default=0, help="with --points: replay at most this many of the kept fork points (random; 0 = all)")
    ap.add_argument("--no-shuffle", action="store_true", help="do not reshuffle the draw pile (control)")
    args, _ = ap.parse_known_args()

    if args.points:
        args.forks_per_scout = args.map_forks_per_scout = 10**6  # the compass runs already kept the forks they want replayed
    os.makedirs(args.raw_dir, exist_ok=True)
    clone_dir = os.path.join(args.raw_dir, "clones")
    rng = random.Random(os.urandom(4))
    from concurrent.futures import as_completed
    pool = make_pool(args.branch_workers)
    futures: dict = {}
    forks: dict[str, dict] = {}   # fork_id -> {"meta", "results", "left"}
    stats = {"scouts": 0, "forks": 0, "done": 0, "branches": 0, "branches_done": 0, "errors": 0, "dropped": 0}
    t0 = time.time()

    def k_for(fork: dict) -> int:
        if args.k:
            return args.k
        unique = len({c[0] for c in fork["state"]["deck"]}) + 1
        return max(args.k_min, min(args.k_max, round(args.k_per_unique * unique)))

    def status() -> None:
        print(f"scouts {stats['scouts']}/{args.trials} | fork points done {stats['done']}/{stats['forks']} | branches {stats['branches_done']}/{stats['branches']} "
              f"| errors {stats['errors']} | dropped {stats['dropped']} | {time.time() - t0:.0f}s", flush=True)

    def handle_scout(r: dict) -> None:
        """A finished scout run (or a compass run's fork-point file): samples its fork points and queues every option's continuations."""
        stats["scouts"] += 1
        if r["error"]:
            stats["errors"] += 1
            print(f"  SCOUT ERROR {r['main_id']}: {r['error']}", flush=True)
        main_id = r["main_id"]
        cards = rng.sample(r["forks"], min(args.forks_per_scout, len(r["forks"])))
        maps = rng.sample(r["map_forks"], min(args.map_forks_per_scout, len(r["map_forks"]))) if args.map_forks_per_scout else []
        for fork in cards:
            fork_id = f"{main_id}_f{fork['index']:02d}"
            options = [c[0] for c in fork["cards"]] + ([None] if fork["can_skip"] else [])
            k = k_for(fork)
            meta = {key: fork[key] for key in ("floor", "cards", "can_skip", "state")}
            meta.update(fork_id=fork_id, k=k, horizon=args.horizon, shuffle=not args.no_shuffle, main_id=main_id)
            forks[fork_id] = {"meta": meta, "results": {}, "left": len(options) * k}
            stats["forks"] += 1
            for opt in options:
                for rep in range(k):
                    f2 = pool.submit(_branch, dict(fork_id=fork_id, clone_path=fork["clone_path"], sig=fork["sig"], option=opt, k=rep,
                                                   asc=args.ascension, horizon=args.horizon, shuffle=not args.no_shuffle,
                                                   script=fork["script"], kind="card", faced=fork.get("faced")))
                    futures[f2] = ("branch", None)
                    stats["branches"] += 1
        for fork in maps:
            fork_id = f"{main_id}_m{fork['index']:02d}"
            meta = {key: fork[key] for key in ("floor", "nodes", "state", "afterstates")}
            meta.update(kind="map", fork_id=fork_id, k=args.map_k, horizon=args.horizon, shuffle=not args.no_shuffle, main_id=main_id)
            forks[fork_id] = {"meta": meta, "results": {}, "left": len(fork["nodes"]) * args.map_k}
            stats["forks"] += 1
            for node in fork["nodes"]:
                for rep in range(args.map_k):
                    f2 = pool.submit(_branch, dict(fork_id=fork_id, clone_path=fork["clone_path"], sig=fork["sig"], option=node, k=rep,
                                                   asc=args.ascension, horizon=args.horizon, shuffle=not args.no_shuffle,
                                                   script=[], kind="map", faced=fork.get("faced")))
                    futures[f2] = ("branch", None)
                    stats["branches"] += 1

    try:
        if args.points:  # fork points from the compass runs (facility_compass.py --fork-dir): no scouts
            allp = []   # (run, "forks" | "map_forks", fork point)
            for path in sorted(glob.glob(os.path.join(args.points, "points", "*.json"))):
                with open(path) as fh:
                    r = json.load(fh)
                allp += [(r, key, f) for key in ("forks", "map_forks") for f in r[key]]
            if args.max_forks and len(allp) > args.max_forks:
                allp = rng.sample(allp, args.max_forks)
            by_run: dict[str, dict] = {}
            for r, key, f in allp:
                by_run.setdefault(r["main_id"], {"main_id": r["main_id"], "error": None, "forks": [], "map_forks": []})[key].append(f)
            for r in by_run.values():
                handle_scout(r)
            args.trials = 0
        for i in range(args.trials):
            ch = rng.choice(args.characters).upper()
            seed = rng.randrange(1, 2**31 - 1)
            fut = pool.submit(_scout, (f"{ch.capitalize()}_{seed}", args.ascension, ch, seed, clone_dir))
            futures[fut] = ("scout", None)
        while futures:
            for fut in as_completed(list(futures)):
                kind, _ = futures.pop(fut)
                r = fut.result()
                if kind == "scout":
                    handle_scout(r)
                else:
                    stats["branches_done"] += 1
                    fork = forks[r["fork_id"]]
                    if r["error"]:
                        stats["errors"] += 1
                        if "replayed to a different state" in r["error"]:
                            stats["dropped"] += 1
                        else:
                            print(f"  BRANCH ERROR {r['fork_id']}: {r['error'][:200]}", flush=True)
                    else:
                        key = "skip" if r["option"] is None else str(r["option"])
                        fork["results"].setdefault(key, []).append({"k": r["k"], **r["metrics"], "taken": r.get("taken")})
                    fork["left"] -= 1
                    if fork["left"] == 0:
                        if fork["results"]:
                            options = {}
                            for key, rows in fork["results"].items():
                                options[key] = {"n": len(rows), "runs": rows, **{f"mean_{m}": _mean([float(x[m]) for x in rows]) for m in METRICS}}
                            with open(os.path.join(args.raw_dir, f"{r['fork_id']}.json"), "w") as f:
                                json.dump({**fork["meta"], "options": options}, f)
                        stats["done"] += 1
                status()
                break  # the dict changed (new futures): rebuild the wait list
    except KeyboardInterrupt:
        print("stopping (in-flight branches are abandoned)", flush=True)
    finally:
        end_pool(pool)
    exit_now()


if __name__ == "__main__":
    main()
