"""Raw spire runs (claws_job.py's raw_dir/*.pkl) -> the dataset claws/model/train.py reads (the same example dicts and keys
claws/model/materialize.py writes). No simulator: the examples were tokenized when they were played, so this only groups
(the mainline's turn t + every fork of turn t), computes advantages and writes the file.

    python agents/spire/materialize.py RAW_DIR [more dirs] --out dataset.pt [--target return]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from collections import defaultdict

import torch

_DROP = ("hp", "turn_index")


def _cost(turns: list[dict], i: int, mode: str) -> float:
    """What turn i of a member (the mainline or a fork) cost. "combat": the HP lost from this turn's start to the combat's end (g).
    "turn": the HP lost during this turn plus the value net's estimate of the position the next turn starts from (the `v` stored when
    that turn was played); a turn that ended the combat has no next turn, so its cost is just what it lost (g). Only this turn's choices
    differ between a group's members, so the later turns' luck drops out of the comparison."""
    if mode == "turn" and i + 1 < len(turns):
        return turns[i]["loss"] + turns[i + 1]["v"]
    return turns[i]["g"]


def _advantages(run: dict, mode: str = "combat") -> dict[tuple[int, int], dict]:
    members: dict[tuple[int, int], dict] = defaultdict(dict)
    for c in run["combats"]:
        if c["outcome"] == "stalled":
            continue
        for i, t in enumerate(c["turns"]):
            members[(c["combat_index"], t["turn_index"])]["main"] = _cost(c["turns"], i, mode)
    for f in run["forks"]:
        members[(f["combat_index"], f["turn_index"])][f["fork_index"]] = _cost(f["turns"], 0, mode)
    out = {}
    for key, cost in members.items():
        if len(cost) >= 2 and max(cost.values()) - min(cost.values()) > 1e-9:
            base = sum(cost.values()) / len(cost)
            out[key] = {m: base - c for m, c in cost.items()}
    return out


def _finish(ex: dict, adv: float | None, run: str) -> dict:
    ex = {k: v for k, v in ex.items() if k not in _DROP}
    ex["run"] = run  # the trainer splits its held-out set by run
    has = adv is not None and ex.get("action_index") is not None
    ex["action_index"] = ex["action_index"] if has else None
    ex["advantage"] = float(adv) if has else 0.0
    ex["value"] = float(max(0.0, ex["value"]))
    return ex


def materialize_run(run: dict, mode: str = "combat") -> list[dict]:
    if run.get("error") or not run["combats"]:
        return []
    adv = _advantages(run, mode)
    out: list[dict] = []
    for c in run["combats"]:
        if c["outcome"] == "stalled":
            continue
        for ex in c.get("examples", []):
            out.append(_finish(ex, adv.get((c["combat_index"], ex["turn_index"]), {}).get("main"), run["main_id"]))
    for f in run["forks"]:
        a = adv.get((f["combat_index"], f["turn_index"]), {}).get(f["fork_index"])
        for ex in f["examples"]:
            out.append(_finish(ex, a, run["main_id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="raw dirs (their *.pkl), or .pkl files")
    ap.add_argument("--out", default="dataset.pt")
    ap.add_argument("--target", choices=["return"], default="return", help="a member's cost is the HP it actually lost to the combat's end")
    ap.add_argument("--advantage", choices=["combat", "turn"], default="combat",
                    help="what a group member's cost is: the HP lost to the combat's end (default), or this turn's HP loss plus the value "
                         "net's estimate of the next turn's start (_cost)")
    args = ap.parse_args()
    paths: list[str] = []
    for d in args.dirs:
        paths += sorted(glob.glob(os.path.join(d, "*.pkl"))) if os.path.isdir(d) else [d]
    examples: list[dict] = []
    runs = forks = 0
    # the first dir is the current round's: its runs are marked `fresh` (the net being trained has not seen them; the trainer's
    # held-out set comes only from fresh runs, see claws/model/train.py split_holdout)
    first = os.path.abspath(args.dirs[0]) if os.path.isdir(args.dirs[0]) else None
    for p in paths:
        with open(p, "rb") as fh:
            run = pickle.load(fh)
        ex = materialize_run(run, args.advantage)
        if first is not None and os.path.dirname(os.path.abspath(p)) == first:
            for e in ex:
                e["fresh"] = True
        examples += ex
        runs += 1
        forks += len(run["forks"])
    scored = sum(e["action_index"] is not None for e in examples)
    print(f"{runs} run(s), {forks} fork(s) -> {len(examples)} examples ({scored} with a policy advantage)")
    torch.save(examples, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
