"""Turns whispering_earring's mainline + fork files into tokenized training examples.

Data (see ../../whispering_earring/jobs.py, facility_claws.py): a mainline plays a run, every combat
measured from full HP; every turn of every combat was then forked N times, each fork playing the
combat out from that turn's start. Per combat turn t the *group* is {the mainline's own turn t} +
{every fork of turn t}: same state, different continuations.

Two kinds of label, one example per combat_play decision:

  value      V(state) = the HP actually lost from that decision to the end of the combat. Every
             fork and mainline decision that is reached carries one, so V is trained on realized
             outcomes (Monte Carlo), not on its own predictions.
  advantage  only on decisions inside a group's compared turn (the first turn of a fork, or the
             mainline's turn t): `baseline - cost` for that member, where cost is the member's
             total remaining loss from that turn's start (`--target return`) or, once a trained V
             was recording values, loss this turn + V(next turn's start) (`--target q`), and
             baseline is the group's mean cost. Cost is HP lost, so lower is better: a member
             that lost less than its group has a positive advantage.

A fork's later turns have no group of their own, so only their turn-start decisions are kept, for V.

Rebuilding states: the facility deletes its native saves, so this replays each mainline once from
the start, taking a native save before every select_map_node, and reloads every fork of that run
from those saves (Checkpoint.reload) -- one combat of replay per fork, not the whole run.

Usage:
    conda activate sts2
    python materialize.py ../../whispering_earring/data/raw --out dataset.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see model.py's top -- two OpenMP
# runtimes (conda's llvm-openmp + torch's bundled copy) otherwise abort on import

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

import sts2_path  # noqa: E402,F401  (sets sys.path to sts2-cli/python before the imports below)
from buffer_env import BufferDecision, Sts2BufferSession  # noqa: E402
from installation import ensure_setup  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "whispering_earring"))  # for `checkpoint`
from checkpoint import Checkpoint  # noqa: E402

from model import GlobalVocab, legal_options_from_buffer, tokenize_from_buffer  # noqa: E402

# Saved as plain dicts of builtins/tensors, not TokenBatch/dataclass instances -- torch.save
# pickles a class by its import path, and TokenBatch would resolve to "__main__.TokenBatch" when
# this script is run directly, unloadable from any other module. Plain dicts sidestep that.
Example = dict[str, Any]

_FORK_FILE = re.compile(r"^(?P<main_id>.+)_c\d{3}t\d{2}_f\d{3}\.json$")


def _bare(entry: dict) -> dict:
    """An action-log entry without the recorder's added policy/value."""
    return {k: entry[k] for k in ("cmd", "action", "args", "hp") if k in entry}


def _apply(session: Sts2BufferSession, entry: dict) -> BufferDecision:
    if entry.get("cmd") == "set_hp":
        return session.set_player_hp(entry["hp"])
    return session.step(entry["action"], **entry.get("args", {}))


def _option_index(options: list[tuple[str, dict]], entry: dict) -> int | None:
    """Which slot in legal_options_from_buffer() matches the action a file recorded."""
    for i, (name, args) in enumerate(options):
        if name == entry["action"] and args == entry.get("args", {}):
            return i
    return None


def _example(
    decision: BufferDecision, gvocab: GlobalVocab, value: float, entry: dict | None, advantage: float | None,
) -> Example:
    """One combat_play decision. `entry` + `advantage` make it a policy example too."""
    options = legal_options_from_buffer(decision)
    batch = tokenize_from_buffer(decision, gvocab)
    action_index = _option_index(options, entry) if entry is not None and advantage is not None else None
    return {
        "type_ids": batch.type_ids, "global_ids": batch.global_ids, "features": batch.features,
        "player_pos": batch.player_pos, "hand_pos": batch.hand_pos, "enemy_pos": batch.enemy_pos,
        "potion_pos": batch.potion_pos, "options": options,
        "value": float(max(0, value)),
        "action_index": action_index,
        "advantage": float(advantage) if action_index is not None else 0.0,
    }


# --- groups --------------------------------------------------------------------------------------


@dataclass
class _Group:
    """The members of one (combat, turn) comparison, keyed "main" or a fork's index."""
    advantages: dict[Any, float] = field(default_factory=dict)  # empty = no signal to learn from


def _costs(turn: dict, next_turn: dict | None) -> tuple[float, float | None]:
    """(return, q): HP lost from this turn's start to the combat's end, and this turn's loss plus
    V(next turn's start) -- None when V wasn't recorded."""
    if next_turn is None:
        return turn["g"], float(turn["loss"])  # the combat ended this turn: nothing left to predict
    v_next = next_turn.get("v")
    return turn["g"], None if v_next is None else turn["loss"] + v_next


def _build_groups(main: dict, forks: list[dict], target: str) -> dict[tuple[int, int], _Group]:
    members: dict[tuple[int, int], dict[Any, tuple[float, float | None]]] = defaultdict(dict)
    for combat in main["combats"]:
        if combat["outcome"] == "stalled":
            continue
        turns = combat["turns"]
        for t, turn in enumerate(turns, start=1):
            members[(combat["combat_index"], t)]["main"] = _costs(turn, turns[t] if t < len(turns) else None)
    for fork in forks:
        turns = fork["turns"]
        key = (fork["combat_index"], fork["turn_index"])
        members[key][fork["fork_index"]] = _costs(turns[0], turns[1] if len(turns) > 1 else None)

    groups: dict[tuple[int, int], _Group] = {}
    for key, costs in members.items():
        use_q = target == "q" and all(q is not None for _, q in costs.values())
        cost = {member: (q if use_q else g) for member, (g, q) in costs.items()}
        group = groups[key] = _Group()
        if len(cost) >= 2 and max(cost.values()) - min(cost.values()) > 1e-9:
            baseline = sum(cost.values()) / len(cost)
            group.advantages = {member: baseline - c for member, c in cost.items()}
    return groups


# --- replay --------------------------------------------------------------------------------------


def _replay_main(
    main: dict, groups: dict, gvocab: GlobalVocab, session: Sts2BufferSession, save_dir: str,
) -> tuple[dict[int, str], list[Example]]:
    """Replays the mainline from the start: native saves before the select_map_nodes its combats
    entered through (returned, by action index), and an example for every combat_play decision."""
    actions = main["actions"]
    save_at = {c["save_index"] for c in main["combats"]}

    # action index -> (combat, turn number) for every action a combat turn's range covers
    turn_of: dict[int, tuple[dict, int]] = {}
    for combat in main["combats"]:
        if combat["outcome"] == "stalled":
            continue
        turns = combat["turns"]
        end_of_combat = next(
            (i for i in range(turns[-1]["start_index"], len(actions)) if actions[i].get("cmd") == "set_hp"),
            len(actions),
        )
        for t, turn in enumerate(turns, start=1):
            end = turns[t]["start_index"] if t < len(turns) else end_of_combat
            for i in range(turn["start_index"], end):
                turn_of[i] = (combat, t)

    saves: dict[int, str] = {}
    examples: list[Example] = []
    decision = session.start(main["character"], main["seed"], main.get("ascension", 0))
    for i, entry in enumerate(actions):
        if i in save_at:
            assert entry["action"] == "select_map_node", f"{main['main_id']}: save index {i} is {entry}"
            saves[i] = os.path.join(save_dir, f"{main['main_id']}_a{i:04d}.json")
            session.save(saves[i])
        if decision.decision == "combat_play" and i in turn_of:
            combat, t = turn_of[i]
            advantage = groups[(combat["combat_index"], t)].advantages.get("main")
            remaining = decision.player_hp - (combat["max_hp"] - combat["loss"])
            examples.append(_example(
                decision, gvocab, remaining, entry if "policy" in entry else None, advantage,
            ))
        decision = _apply(session, _bare(entry))
    return saves, examples


def _replay_fork(
    fork: dict, main: dict, saves: dict[int, str], groups: dict, gvocab: GlobalVocab, session: Sts2BufferSession,
) -> list[Example]:
    combat = next(c for c in main["combats"] if c["combat_index"] == fork["combat_index"])
    actions, prefix_len, turns = fork["actions"], fork["prefix_len"], fork["turns"]
    decision = Checkpoint(
        main["main_id"], main["character"], main["seed"], main.get("ascension", 0),
        actions=[_bare(e) for e in actions[:prefix_len]],
        save_path=saves[combat["save_index"]], save_index=combat["save_index"],
    ).reload(session)

    final_hp = decision.player_hp - turns[0]["g"]
    turn_starts = {t["start_index"] for t in turns}
    compared_end = turns[1]["start_index"] if len(turns) > 1 else len(actions)
    advantage = groups[(fork["combat_index"], fork["turn_index"])].advantages.get(fork["fork_index"])

    examples: list[Example] = []
    for i in range(prefix_len, len(actions)):
        entry = actions[i]
        compared = i < compared_end
        if decision.decision == "combat_play" and (compared or i in turn_starts):
            examples.append(_example(
                decision, gvocab, decision.player_hp - final_hp,
                entry if compared and "policy" in entry else None, advantage if compared else None,
            ))
        decision = _apply(session, _bare(entry))
    return examples


def materialize_run(
    main_path: str, fork_paths: list[str], target: str, gvocab: GlobalVocab, session: Sts2BufferSession,
) -> list[Example]:
    main = json.loads(Path(main_path).read_text())
    if main.get("error") is not None or not main.get("combats"):
        return []  # an aborted run says nothing about its combats
    forks = [json.loads(Path(p).read_text()) for p in fork_paths]
    groups = _build_groups(main, forks, target)
    with tempfile.TemporaryDirectory() as save_dir:
        saves, examples = _replay_main(main, groups, gvocab, session, save_dir)
        for fork in forks:
            examples.extend(_replay_fork(fork, main, saves, groups, gvocab, session))
    return examples


# --- driver --------------------------------------------------------------------------------------

_gvocab: GlobalVocab | None = None
_session: Sts2BufferSession | None = None


def _init_worker() -> None:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)  # the engine logs every step to stderr
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    global _gvocab, _session
    _gvocab = GlobalVocab(EntityVocab.load())
    _session = Sts2BufferSession()


def _task(main_path: str, fork_paths: list[str], target: str) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None  # set by _init_worker before any task runs
    label = f"{Path(main_path).stem} (+{len(fork_paths)} forks)"
    try:
        return label, materialize_run(main_path, fork_paths, target, _gvocab, _session), None
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", nargs="+", help="trial JSON files, or directories to glob *.json from")
    parser.add_argument("--out", default="dataset.pt")
    parser.add_argument("--target", choices=["return", "q"], default="return",
                        help="what a member's cost is: the HP it actually lost to the combat's end "
                             "(return, unbiased -- use while V is untrained), or its loss this turn plus the "
                             "recorded V(next turn) (q; groups without recorded V fall back to return)")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count(),
                        help="parallel worker processes, one in-process sts2-cli session each (default: cpu count)")
    args = parser.parse_args()

    paths: list[str] = []
    for trial in args.trials:
        paths.extend(sorted(str(p) for p in Path(trial).glob("*.json")) if os.path.isdir(trial) else [trial])

    forks_of: dict[str, list[str]] = defaultdict(list)
    mains: dict[str, str] = {}
    for path in paths:
        match = _FORK_FILE.match(Path(path).name)
        if match:
            forks_of[match["main_id"]].append(path)
        else:
            mains[Path(path).stem] = path
    orphans = sum(len(v) for k, v in forks_of.items() if k not in mains)
    print(f"{len(mains)} mainline run(s), {sum(len(v) for v in forks_of.values()) - orphans} fork(s)"
          + (f", {orphans} fork(s) without a mainline file skipped" if orphans else ""))

    ensure_setup()  # once, in the parent, so a first-time build can't race across workers

    examples: list[Example] = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool:
        futures = [pool.submit(_task, path, forks_of.get(main_id, []), args.target) for main_id, path in mains.items()]
        for future in as_completed(futures):
            label, result, error = future.result()
            if error is not None:
                print(f"skip {label}: {error}")
                continue
            examples.extend(result)

    scored = sum(ex["action_index"] is not None for ex in examples)
    print(f"{len(examples)} examples ({scored} with a policy advantage)")
    torch.save(examples, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
