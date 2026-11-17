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
    won = sum(c.get("outcome") == "won" for c in combats)
    lost = sum(c.get("outcome") == "lost" for c in combats)
    # total_floor counts floors across acts (end_floor restarts every act); older run files only have end_floor.
    floors = [d.get("total_floor", d["end_floor"]) for d in runs]
    by_seed: dict[str, list[int]] = {}
    for d, f in zip(runs, floors):
        by_seed.setdefault(d["seed"], []).append(f)
    def boss_fights(d):
        # c["boss"] is recorded by jobs.RunJob; older files lack it, and act 1's boss is always on floor 17.
        return [c for c in d.get("combats", []) if c.get("boss", c.get("act") == 1 and c.get("floor") == 17)]
    reached_boss = sum(bool(boss_fights(d)) for d in runs)
    beat_boss = sum(any(c.get("outcome") == "won" for c in boss_fights(d)) for d in runs)
    return {
        "label": label, "runs": len(runs), "reached_boss": reached_boss, "beat_boss": beat_boss,
        "mean_end_floor": round(statistics.mean(floors), 2), "best_floor": max(floors),
        "floor_se": round(statistics.stdev(floors) / len(floors) ** 0.5, 2) if len(floors) > 1 else None,
        "act2_rate": round(sum(d.get("acts_reached", 1) >= 2 for d in runs) / len(runs), 3),
        "combats": len(combats),
        "combat_win_rate": round(won / (won + lost), 3) if won + lost else None,
        "mean_turns": round(statistics.mean(c.get("turns", 0) for c in combats), 2) if combats else None,
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
    print("round  runs  mean_total_floor  best  boss  beat  act2  wins  combats  hp_lost/combat")
    for r in rows:
        print(f"{str(r['label']):>5}  {r['runs']:4d}  {r['mean_end_floor']:16.2f}  {r['best_floor']:4d}  "
              f"{r.get('reached_boss', 0):4d}  {r.get('beat_boss', 0):4d}  {r.get('act2_rate', 0):4.0%}  {r['wins']:4d}  {r['mean_combats']:7.2f}  {r['mean_hp_lost_per_combat']}"
              + (f"  per seed {r['per_seed_mean_floor']}" if len(r.get("per_seed_mean_floor", {})) > 1 else ""))


if __name__ == "__main__":
    main()
