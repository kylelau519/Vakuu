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
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from vakuu.golden_compass.hazard_net import ACT2_END, AUX, FINAL_FLOOR, head_floor  # noqa: E402

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
    """Act 1 and act 2 are learned; floors are the head index (hazard_net.head_floor, act 2 = 17 + the in-act floor).
    An act 1 state of a run that got into act 2 is censored at the end of act 1: the floors after the boss would be
    labelled from a state whose HP is about to be healed, so act 2's hazards are learned only from act 2 states (the
    first of which, at the Ancient, has full HP). A run that went on to act 3 is censored at the end of act 2; act 3
    states are dropped."""
    code = OUTCOME_CODE.get(data.get("outcome"))
    states = data.get("states") or []
    if code is None or not states or data.get("error"):
        return []
    end_act, end_floor = int(data.get("end_act") or 1), int(data["end_floor"])
    samples = []
    for entry in states:
        state = entry["state"]
        act = int(state.get("act") or 1)
        if act == 1:
            end, c = (FINAL_FLOOR, 2) if end_act >= 2 else (end_floor, code)
        elif act == 2:
            end, c = (ACT2_END, 2) if end_act >= 3 else (head_floor(2, end_floor), code)
        else:
            continue
        start = head_floor(act, int(state["floor"]))
        if c == 0 and start > end:  # died in the room it was already in: nothing left to predict
            continue
        combats = [x for x in data.get("combats") or [] if int(x.get("act") or 1) == act]
        aux, aux_mask = _aux_targets(combats, int(state["floor"]))
        samples.append({
            "run": data["main_id"], "decision": entry["decision"], "state": state,
            "end": end, "outcome": c, "aux": aux, "aux_mask": aux_mask,
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
