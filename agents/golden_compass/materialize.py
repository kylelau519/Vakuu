"""Turns facility_compass.py's run files into a training dataset. No game needed: every RunState was
logged live by RunJob, so this is just reading JSON.

    python materialize.py <run dirs or files...> --out dataset.pt

One sample per logged RunState. Its labels come from the run it belongs to:
  start    the first floor whose death the state predicts (RunState.floor)
  end      the floor the run ended on; outcome 0 = died there, 1 = won, 2 = censored (stalled alive)
  aux      the share of max HP each of the next AUX combats at floor >= start cost (the aux head's target),
           with aux_mask 0 where the run ended before that many combats
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from hazard_net import AUX, FINAL_FLOOR  # noqa: E402

OUTCOME_CODE = {"lost": 0, "won": 1, "censored": 2}


def _aux_targets(combats: list[dict], start: int) -> tuple[list[float], list[float]]:
    upcoming = sorted((c for c in combats if c["floor"] >= start), key=lambda c: c["floor"])[:AUX]
    shares = [
        min(1.0, max(0.0, (c["hp_in"] - c["hp_out"]) / c["max_hp"])) if c["max_hp"] else 0.0 for c in upcoming
    ]
    mask = [1.0] * len(shares)
    pad = AUX - len(shares)
    return shares + [0.0] * pad, mask + [0.0] * pad


def materialize_run(data: dict) -> list[dict]:
    """Only act 1 is learned: floors restart at 2 in every act, so an act-2 floor would be read as an early
    act-1 floor. A run that got into act 2 is, for its act-1 states, one that survived to the end of act 1
    (censored there); its act-2 states and combats are dropped."""
    code = OUTCOME_CODE.get(data.get("outcome"))
    states = data.get("states") or []
    if code is None or not states or data.get("error"):
        return []
    end = int(data["end_floor"])
    if int(data.get("end_act") or 1) >= 2:
        end, code = FINAL_FLOOR, 2
    combats = [c for c in data.get("combats") or [] if int(c.get("act") or 1) == 1]
    samples = []
    for entry in states:
        state = entry["state"]
        if int(state.get("act") or 1) != 1:
            continue
        start = int(state["floor"])
        if code == 0 and start > end:  # died in the room it was already in: nothing left to predict
            continue
        aux, aux_mask = _aux_targets(combats, start)
        samples.append({
            "run": data["main_id"], "decision": entry["decision"], "state": state,
            "end": end, "outcome": code, "aux": aux, "aux_mask": aux_mask,
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="run directories (every *.json inside) or files")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "dataset.pt"))
    args = parser.parse_args()

    files: list[str] = []
    for p in args.paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]

    samples: list[dict] = []
    runs = skipped = 0
    for path in files:
        with open(path) as f:
            data = json.load(f)
        if data.get("role") != "run":
            continue
        got = materialize_run(data)
        runs += bool(got)
        skipped += not got
        samples += got

    torch.save(samples, args.out)
    outcomes = [s["outcome"] for s in samples]
    print(f"{len(samples)} samples from {runs} runs ({skipped} unusable) -> {args.out}")
    if samples:
        print(f"  by outcome of the sample's run: died {outcomes.count(0)}, won {outcomes.count(1)}, "
              f"censored {outcomes.count(2)}")


if __name__ == "__main__":
    main()
