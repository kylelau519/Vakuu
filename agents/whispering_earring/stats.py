"""Per-floor combat stats for whispering_earring main-line trials.

Reads every main-line (role == "main") trial file in a raw-dir and reports, per floor: how
many runs got that far, how many combats were fought there, and how many of those were won
vs lost. Combat (act, floor, outcome) is recorded live by facility.py's _resolve_combat via
CompactRecorder.record_combat() -- see recorder.py -- so no replay is needed here, only files
saved after that change carry a "combats" list; older files are silently skipped.

Usage:
    python stats.py [--raw-dir data/raw]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_main_trials(raw_dir: Path) -> list[dict]:
    trials = []
    for path in sorted(raw_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if data.get("role") == "main":
            trials.append(data)
    return trials


def print_floor_stats(trials: list[dict]) -> None:
    per_floor: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"combats": 0, "won": 0, "lost": 0}
    )
    max_floor_per_run: dict[str, tuple[int, int]] = {}

    for trial in trials:
        main_id = trial.get("main_id")
        for combat in trial.get("combats", []):
            key = (combat["act"], combat["floor"])
            per_floor[key]["combats"] += 1
            per_floor[key][combat["outcome"]] += 1
            if main_id not in max_floor_per_run or key > max_floor_per_run[main_id]:
                max_floor_per_run[main_id] = key

    if not per_floor:
        print("No combat stats found (no main-line files with a recorded 'combats' list).")
        return

    reached_floors = sorted(max_floor_per_run.values())
    total_runs = len(max_floor_per_run)

    print(f"{'Act':>3} {'Floor':>5} {'Runs Reached':>12} {'Combats':>7} {'Won':>5} {'Lost':>5}")
    for key in sorted(per_floor):
        act, floor = key
        runs_reached = sum(1 for f in reached_floors if f >= key)
        stats = per_floor[key]
        print(f"{act:>3} {floor:>5} {runs_reached:>12} {stats['combats']:>7} "
              f"{stats['won']:>5} {stats['lost']:>5}")

    outcomes = defaultdict(int)
    for trial in trials:
        outcomes[trial.get("outcome", "unknown")] += 1
    print(f"\n{total_runs} main-line run(s) with combat data -- final outcomes: "
          + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-floor combat stats for whispering_earring trials")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="directory of trial JSON files (default: 'data/raw', resolved "
                        "against the process's cwd if relative)")
    args = parser.parse_args()
    print_floor_stats(_load_main_trials(Path(args.raw_dir)))
