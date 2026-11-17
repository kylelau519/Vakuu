"""Step 1 (measure only, no behaviour change): how often do the forks of one (combat, turn) group end
their compared turn in the same state?

Per group = {mainline's turn t} + {every fork of turn t}. Each member's "outcome" is the state at the
start of the next turn (or, if the combat ended that turn, ("end", hp lost)). Raw files only hold
action logs, so this replays every mainline and every fork (same route as claws/model/materialize.py)
and reads the buffer at each next-turn start.

Two keys are reported: "visible" (no hidden state; an upper bound) and "full" (state_key.py: visible + hidden).

Usage (conda activate sts2):
    python scripts/measure_fork_duplicates.py data/raw [-j N] [--limit-runs N] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # checkpoint, sts2_path
import sts2_path  # noqa: E402,F401
import buffer_layout as bl  # noqa: E402
from buffer_env import Sts2BufferSession  # noqa: E402
from installation import ensure_setup  # noqa: E402
from checkpoint import Checkpoint  # noqa: E402
from state_key import combine, visible_key  # noqa: E402

_FORK_FILE = re.compile(r"^(?P<main_id>.+)_c\d{3}t\d{2}_f\d{3}\.json$")

def _apply(session, entry: dict):
    if entry.get("cmd") == "set_hp":
        return session.set_player_hp(entry["hp"])
    return session.step(entry["action"], **entry.get("args", {}))


def _bare(entry: dict) -> dict:
    return {k: entry[k] for k in ("cmd", "action", "args", "hp") if k in entry}


def _outcome(session, decision, loss) -> tuple:
    if decision.decision == "combat_play":
        vis = visible_key(decision.arr)
        return ("state", combine(vis, ""), combine(vis, session.hidden_state_key()))
    return ("end", loss)


def _replay_main(main: dict, session, save_dir: str):
    """-> (saves by action index, {(combat_index, turn): outcome key of that turn's end})."""
    actions = main["actions"]
    save_at = {c["save_index"] for c in main["combats"]}
    want: dict[int, tuple[int, int]] = {}  # action index -> (combat_index, turn whose END is here)
    ends: dict[tuple[int, int], tuple] = {}
    for c in main["combats"]:
        if c["outcome"] == "stalled":
            continue
        turns = c["turns"]
        for t, turn in enumerate(turns, start=1):
            if t < len(turns):
                want[turns[t]["start_index"]] = (c["combat_index"], t)
            else:
                ends[(c["combat_index"], t)] = ("end", turn["loss"])
    saves: dict[int, str] = {}
    decision = session.start(main["character"], main["seed"], main.get("ascension", 0))
    for i, entry in enumerate(actions):
        if i in save_at:
            saves[i] = os.path.join(save_dir, f"{main['main_id']}_a{i:04d}.json")
            session.save(saves[i])
        if i in want:
            ends[want[i]] = _outcome(session, decision, None)
        decision = _apply(session, _bare(entry))
    return saves, ends


def _replay_fork(fork: dict, main: dict, saves: dict, session) -> tuple:
    combat = next(c for c in main["combats"] if c["combat_index"] == fork["combat_index"])
    actions, prefix_len, turns = fork["actions"], fork["prefix_len"], fork["turns"]
    if len(turns) == 1:
        return ("end", turns[0]["loss"])
    decision = Checkpoint(
        main["main_id"], main["character"], main["seed"], main.get("ascension", 0),
        actions=[_bare(e) for e in actions[:prefix_len]],
        save_path=saves[combat["save_index"]], save_index=combat["save_index"],
    ).reload(session)
    stop = turns[1]["start_index"]
    for i in range(prefix_len, stop):
        decision = _apply(session, _bare(actions[i]))
    return _outcome(session, decision, None)


_session = None


def _init_worker() -> None:
    global _session
    fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(fd, 2)
    os.close(fd)
    _session = Sts2BufferSession()


def _task(main_path: str, fork_paths: list[str]):
    import tempfile
    main = json.loads(Path(main_path).read_text())
    if main.get("error") is not None or not main.get("combats"):
        return Path(main_path).stem, [], None
    forks = sorted((json.loads(Path(p).read_text()) for p in fork_paths),
                   key=lambda f: (f["combat_index"], f["turn_index"], f["fork_index"]))
    groups: dict[tuple[int, int], list] = defaultdict(list)  # members in order: main first
    try:
        with tempfile.TemporaryDirectory() as d:
            saves, ends = _replay_main(main, _session, d)
            for key, outcome in ends.items():
                groups[key].append(("main", outcome))
            for f in forks:
                if f["outcome"] == "stalled":
                    continue
                groups[(f["combat_index"], f["turn_index"])].append(
                    (f["fork_index"], _replay_fork(f, main, saves, _session)))
    except Exception as e:  # noqa: BLE001
        return Path(main_path).stem, [], f"{type(e).__name__}: {e}"
    out = []
    for key, members in groups.items():
        out.append({
            "group": key,
            "n": len(members),
            "visible": len({m[1][:2] if m[1][0] == "state" else m[1] for m in members}),
            "full": len({m[1][:3] if m[1][0] == "state" else m[1] for m in members}),
            "ends": sum(m[1][0] == "end" for m in members),
        })
    return Path(main_path).stem, out, None


def _report(rows: list[dict], label: str, field: str) -> dict:
    # a group with only the mainline (no forks) carries no information about forks
    rows = [r for r in rows if r["n"] >= 2]
    n_groups = len(rows)
    forks = sum(r["n"] - 1 for r in rows)
    dup = sum(r["n"] - r[field] for r in rows)  # members beyond the first of each distinct outcome
    dist = Counter(r[field] for r in rows)
    print(f"\n[{label}]  groups with >=2 members: {n_groups}, forks: {forks}")
    print("  distinct members per group:", ", ".join(f"{k}:{v}" for k, v in sorted(dist.items())))
    le1 = sum(v for k, v in dist.items() if k <= 1)
    print(f"  groups with <=1 distinct member: {le1}/{n_groups} = {le1 / max(1, n_groups):.1%}")
    print(f"  duplicate forks (same outcome as an earlier member): {dup}/{forks} = {dup / max(1, forks):.1%}")
    return {"groups": n_groups, "forks": forks, "dup_forks": dup, "distinct_dist": dict(dist)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count())
    ap.add_argument("--limit-runs", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    forks_of: dict[str, list[str]] = defaultdict(list)
    mains: dict[str, str] = {}
    for p in sorted(Path(args.raw_dir).glob("*.json")):
        m = _FORK_FILE.match(p.name)
        if m:
            forks_of[m["main_id"]].append(str(p))
        else:
            mains[p.stem] = str(p)
    ids = [i for i in mains if forks_of.get(i)][: args.limit_runs]
    print(f"{len(ids)} mainline run(s) with forks")
    ensure_setup()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool:
        futs = [pool.submit(_task, mains[i], forks_of[i]) for i in ids]
        for fu in as_completed(futs):
            label, out, err = fu.result()
            if err:
                print(f"skip {label}: {err}")
            else:
                rows.extend(out)
                print(f"  done {label}: {len(out)} groups", flush=True)

    res = {"visible": _report(rows, "visible key only", "visible"),
           "full": _report(rows, "full key: visible + hidden (RNG streams, draw order, monster queue)", "full")}
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
