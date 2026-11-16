"""One summary line per round, to watch a loop improve (most useful on a small seed pool: the same few
maps and rewards each round, so a rising floor is the policy getting better rather than an easier map).

    python progress.py <run dir> --log progress.jsonl [--label N]

Reads the run files in <run dir> (facility_compass.py output), appends
{label, runs, mean_end_floor, best_floor, wins, mean_combats, mean_hp_lost_per_combat} to the log and
prints the trend over the last rounds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def summarize(run_dir: str, label: str | None) -> dict | None:
    runs = []
    for path in glob.glob(os.path.join(run_dir, "*.json")):
        with open(path) as f:
            d = json.load(f)
        if d.get("role") == "run" and d.get("outcome") in ("lost", "won", "censored"):
            runs.append(d)
    if not runs:
        return None
    combats = [c for d in runs for c in d.get("combats", []) if c.get("max_hp")]
    floors = [d["end_floor"] for d in runs]
    by_seed: dict[str, list[int]] = {}
    for d in runs:
        by_seed.setdefault(d["seed"], []).append(d["end_floor"])
    return {
        "label": label, "runs": len(runs),
        "mean_end_floor": round(statistics.mean(floors), 2), "best_floor": max(floors),
        "wins": sum(d["outcome"] == "won" for d in runs),
        "per_seed_mean_floor": {s: round(statistics.mean(v), 1) for s, v in sorted(by_seed.items())},
        "mean_combats": round(statistics.mean(len(d.get("combats", [])) for d in runs), 2),
        "mean_hp_lost_per_combat": round(statistics.mean((c["hp_in"] - c["hp_out"]) / c["max_hp"] for c in combats), 3)
        if combats else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--log", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--show", type=int, default=10, help="rounds of trend to print")
    args = parser.parse_args()

    row = summarize(args.run_dir, args.label)
    if row is None:
        print(f"no finished runs in {args.run_dir}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps(row) + "\n")
    with open(args.log) as f:
        rows = [json.loads(line) for line in f if line.strip()][-args.show:]
    print("round  runs  mean_end_floor  best  wins  combats  hp_lost/combat")
    for r in rows:
        print(f"{str(r['label']):>5}  {r['runs']:4d}  {r['mean_end_floor']:14.2f}  {r['best_floor']:4d}  "
              f"{r['wins']:4d}  {r['mean_combats']:7.2f}  {r['mean_hp_lost_per_combat']}"
              + (f"  per seed {r['per_seed_mean_floor']}" if len(r.get("per_seed_mean_floor", {})) > 1 else ""))


if __name__ == "__main__":
    main()
