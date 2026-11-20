"""Greedy act-2 evaluation of Claws on spire-zero: real HP, no healing, no forks, from random act-boundary saves; the macro
decisions are macro.py's (stage 1). Prints one summary row and appends it to --log as a JSON line.

    CLAWS_CHECKPOINT=... CLAWS_GREEDY=1 python agents/spire/eval_claws.py --runs 60 --workers 17 --ckpt-dir data/act_checkpoints \
        --log data/run2/eval_claw_spire.jsonl --label 12
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))


def _play(task: tuple) -> dict:
    save, main_id, asc, character, seed = task
    import torch
    torch.set_num_threads(1)
    from spire import claws_job as J, macro
    from spire.actions import random_pick, to_action
    from spire.host import SpireHost
    J._load()
    from spire.macro_player import MacroPlayer
    player = MacroPlayer(random.Random(J.zlib_hash(main_id)), asc)
    J.seed_everything(J.zlib_hash(main_id))
    rng = random.Random(J.zlib_hash(main_id))
    out = {"main_id": main_id, "floor": 0, "act": 0, "won": False, "combats": [], "error": None}
    host = None
    try:
        host = SpireHost.start(save=save, character=character, seed=seed, asc=asc)
        d, n, same, last = host.decision, 0, 0, None
        while d is not None and n < 4000:
            n += 1
            run = d.obs.get("run", {})
            out["floor"], out["act"] = run.get("floor", out["floor"]), run.get("act", out["act"])
            if d.surface == "combat":
                obs = d.obs["screen"]["combat"]
                if not out["combats"] or out["combats"][-1].get("_enc") != (obs["encounter"], run.get("floor")):
                    out["combats"].append({"encounter": obs["encounter"], "act": run.get("act"), "floor": run.get("floor"), "hp_in": obs["player"]["hp"], "_enc": (obs["encounter"], run.get("floor")), "won": None})
                dec, options, idx, _p, _v = J.decide(obs, d.legal, J.run_context(run))
                d = host.act(to_action(*options[idx], obs, d.legal))
                continue
            if d.surface == "card_choice" and "combat" in d.obs["screen"]:
                ch = d.obs["screen"]["choice"]
                d = host.act(random_pick(ch["min"], ch["max"], len(ch["options"]), rng))
                continue
            if out["combats"] and out["combats"][-1]["won"] is None:
                out["combats"][-1]["won"] = True
            sig = (d.surface, json.dumps(d.legal), run.get("hp"))
            same = same + 1 if sig == last else 0
            last = sig
            if same >= 25:
                out["error"] = "stalled"
                break
            d = host.act(player.act(d, same))
        if d is None:
            if out["combats"] and out["combats"][-1]["won"] is None:
                out["combats"][-1]["won"] = False
            out["won"] = bool(host.end and host.end.get("won"))
            if host.end and "floor" in host.end:
                out["floor"] = host.end["floor"]
    except BaseException as e:  # noqa: BLE001
        import traceback
        out["error"] = f"{type(e).__name__}: {e} [" + " < ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(traceback.extract_tb(e.__traceback__)[-4:])) + "]"
    finally:
        if host is not None:
            host.close()
    for c in out["combats"]:
        c.pop("_enc", None)
    return out


def summarize(rows: list[dict], label: str) -> dict:
    ok = [r for r in rows if not r["error"]]
    floors = [r["floor"] for r in ok]
    mean = sum(floors) / max(1, len(floors))
    se = math.sqrt(sum((f - mean) ** 2 for f in floors) / max(1, len(floors) - 1) / max(1, len(floors))) if len(floors) > 1 else 0.0
    combats = [c for r in ok for c in r["combats"] if c["won"] is not None]
    def boss_of(act: int):
        return [c for c in combats if str(c["encounter"]).endswith("_BOSS") and c.get("act") == act]
    b1, b2 = boss_of(0), boss_of(1)
    acts = [r["act"] + 1 for r in ok]
    return {
        "label": label, "runs": len(ok), "errors": len(rows) - len(ok), "mean_end_floor": round(mean, 2), "floor_se": round(se, 2),
        "best_floor": max(floors, default=0), "wins": sum(r["won"] for r in ok),
        "combats": len(combats), "combat_win_rate": round(sum(c["won"] for c in combats) / max(1, len(combats)), 3),
        "act1_reached_boss": len(b1), "act1_beat_boss": sum(c["won"] for c in b1),
        "act2_reached": sum(a >= 2 for a in acts), "act2_reached_boss": len(b2), "act2_beat_boss": sum(c["won"] for c in b2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ascension", type=int, default=1)
    ap.add_argument("--ckpt-dir", default="data/act_checkpoints")
    ap.add_argument("--resume-act", type=int, default=0, help="0: fresh runs from the start (default); N: from random act-N saves")
    ap.add_argument("--characters", nargs="*", default=["IRONCLAD"])
    ap.add_argument("--log", default=None)
    ap.add_argument("--label", default=time.strftime("%d.%H%M"))
    args = ap.parse_args()
    rng = random.Random(os.urandom(4))
    tasks = []
    if args.resume_act > 0:
        saves = [p for p in sorted(glob.glob(os.path.join(args.ckpt_dir, f"*_act{args.resume_act}.json"))) if not p.endswith(".ckpt.json")]
        tasks = [(rng.choice(saves), f"eval{os.urandom(3).hex()}", args.ascension, None, 1) for _ in range(args.runs)]
    else:
        tasks = [(None, f"eval{os.urandom(3).hex()}", args.ascension, rng.choice(args.characters).upper(), rng.randrange(1, 2**31 - 1))
                 for _ in range(args.runs)]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))
    procs, rows = [], []
    try:
        futs = [pool.submit(_play, t) for t in tasks]
        procs = list((getattr(pool, "_processes", None) or {}).values())
        for f in as_completed(futs):
            rows.append(f.result())
            print(f"\r{len(rows)}/{args.runs}", end="", flush=True)
    finally:
        procs = procs or list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
    row = summarize(rows, args.label)
    print()
    print(f"runs {row['runs']}/{args.runs} (errors {row['errors']}) | floor {row['mean_end_floor']}±{row['floor_se']} best {row['best_floor']} | "
          f"combats {row['combats']} win {row['combat_win_rate']:.0%} | act-1 boss {row['act1_beat_boss']}/{row['act1_reached_boss']} | "
          f"reached act 2 {row['act2_reached']} (boss {row['act2_beat_boss']}/{row['act2_reached_boss']}) | wins {row['wins']}")
    for r in rows:
        if r["error"]:
            print("  error:", r["error"][:150])
            break
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        with open(args.log, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    for child in mp.active_children():  # the workers (spawned by the executor) hold native threads: end them all
        try:
            child.terminate()
        except Exception:  # noqa: BLE001
            pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
