"""Turns whispering_earring trial files into tokenized training examples 

Every branch spawned by whispering_earring's turn-level branch-and-adopt (see
../../whispering_earring/facility_claws.py) replays an identical shared prefix before diverging on
its own first action.

Usage:
    conda activate sts2
    python materialize.py ../../whispering_earring/data/raw/*.json --out dataset.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see model.py's top -- two OpenMP
# runtimes (conda's llvm-openmp + torch's bundled copy) otherwise abort on import

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

import sts2_path  # noqa: E402,F401  (sets sys.path to sts2-cli/python before the imports below)
from buffer_env import Sts2BufferSession  # noqa: E402
from installation import ensure_setup  # noqa: E402

from model import GlobalVocab, legal_options_from_buffer, tokenize_from_buffer  # noqa: E402

# Saved as plain dicts of builtins/tensors, not TokenBatch/dataclass instances -- torch.save
# pickles a class by its import path, and TokenBatch would resolve to "__main__.TokenBatch" when
# this script is run directly, unloadable from any other module. Plain dicts sidestep that.
Example = dict[str, Any]


def _same_option(option: tuple[str, dict], recorded: dict) -> bool:
    return option[0] == recorded["action"] and option[1] == recorded.get("args", {})


def _option_index(options: list[tuple[str, dict]], entry: dict) -> int | None:
    """Which slot in the model's own legal_options_from_buffer() matches the action a trial
    actually recorded"""
    for i, option in enumerate(options):
        if _same_option(option, entry):
            return i
    return None


def _norm_action(entry: dict) -> tuple[str, tuple]:
    """Hashable key for an {action, args} pair, used both to key the shared prefix a branch
    trial replayed and to check how many distinct first actions its group's siblings sampled."""
    return (entry["action"], tuple(sorted(entry.get("args", {}).items())))


@dataclass
class _Group:
    """Everything materialize_group() needs for one (main_id, shared-prefix) group of sibling
    branch trials, captured in _scan_trials's single read pass so no branch file ever needs to
    be opened a second time. `members` holds each sibling's own first-novel-action entry
    (`{action, args, value}`) directly."""
    main_id: str
    character: str
    seed: str
    ascension: int
    prefix_actions: list[dict]
    members: list[dict] = field(default_factory=list)


def _group_baseline(group: _Group) -> float | None:
    """The REINFORCE baseline for this group's turn-start decision:"""
    subkeys = {_norm_action(m) for m in group.members}
    if len(subkeys) < 2:
        return None
    return sum(m["value"] for m in group.members) / len(group.members)


def _read_trial(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _scan_trials(paths: list[str]) -> tuple[dict[str, str], dict[tuple, _Group]]:
    """Single read pass over every trial file: classifies each as main-line or branch (grouped by the exact shared prefix
    it replayed before diverging). The reads themselves run on a thread pool. Returns
    (main_id -> path for every non-errored main-line trial, branch groups) -- main_id comes from
    each main-line file's own recorded field (see facility_common.py's main_worker_loop, which
    writes it as f"{character}_{seed}" -- same value every branch of that trial also carries),
    read for free here since _read_trial already loaded the whole file."""
    main_id_to_path: dict[str, str] = {}
    groups: dict[tuple, _Group] = {}

    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4)) as pool:
        trials = pool.map(_read_trial, paths)
        for path, trial in zip(paths, trials):
            _scan_one(path, trial, main_id_to_path, groups)

    return main_id_to_path, groups


def _scan_one(path: str, trial: dict, main_id_to_path: dict[str, str], groups: dict[tuple, _Group]) -> None:
    if trial.get("role") != "branch":
        if trial.get("error") is None and trial.get("main_id") is not None:
            main_id_to_path[trial["main_id"]] = path
        return
    if trial.get("branch_kind") == "meta":
        return  # choices_paradox's own MetaBranchJob file (different prefix/action shape --
        # see agents/choices_paradox/model/materialize.py, which is what actually reads these);
        # only relevant if this ever points at a raw-dir shared with facility_choices_paradox.py
    if trial.get("error") is not None:
        return

    prefix_len = trial.get("prefix_len")
    actions = trial.get("actions", [])
    if prefix_len is None or len(actions) <= prefix_len:
        return
    first = actions[prefix_len]
    if "value" not in first:
        return  # turn never resolved (branch errored mid-turn) -- no realized outcome

    prefix_actions = actions[:prefix_len]
    prefix_key = (trial["main_id"], tuple(_norm_action(a) for a in prefix_actions))
    group = groups.get(prefix_key)
    if group is None:
        group = _Group(
            main_id=trial["main_id"], character=trial["character"], seed=trial["seed"],
            ascension=trial.get("ascension", 0), prefix_actions=prefix_actions,
        )
        groups[prefix_key] = group
    group.members.append(first)


def materialize_group(
    group: _Group, gvocab: GlobalVocab, session: Sts2BufferSession
) -> list[Example]:
    """Replays this group's shared prefix exactly once, tokenizes the resulting turn-start
    decision once, and emits one policy_real example per sibling directly from its own recorded
    first action + value. Only used as a fallback for a branch group whose main-line file is
    missing/errored (see main()) -- when the main file IS available,
    materialize_main_and_groups() below reaches the same checkpoint without replaying this
    group's prefix from scratch."""
    baseline = _group_baseline(group)
    if baseline is None:
        return []

    decision = session.start(group.character, group.seed, group.ascension)
    for entry in group.prefix_actions:
        decision = session.step(entry["action"], **entry.get("args", {}))
    if decision.decision != "combat_play":
        return []  # defensive -- a turn-start checkpoint should always land here

    options = legal_options_from_buffer(decision)
    batch = tokenize_from_buffer(decision, gvocab)

    examples: list[Example] = []
    for first in group.members:
        action_index = _option_index(options, first)
        if action_index is None:
            continue
        examples.append({
            "type_ids": batch.type_ids,
            "global_ids": batch.global_ids,
            "features": batch.features,
            "player_pos": batch.player_pos,
            "hand_pos": batch.hand_pos,
            "enemy_pos": batch.enemy_pos,
            "potion_pos": batch.potion_pos,
            "options": options,
            "value": first["value"],
            "action_index": action_index,
            "advantage": first["value"] - baseline,
            "policy_real": True,
        })
    return examples


def materialize_main_and_groups(
    path: str, own_groups: list[_Group], gvocab: GlobalVocab, session: Sts2BufferSession
) -> list[Example]:
    """Single sequential replay of one main-line trial, emitting both its own per-turn
    policy_real=False examples (same as this function used to do as materialize_main_trial) and,
    at each shared-prefix boundary one of its OWN sibling branch groups diverged from, that
    group's policy_real=True examples (same as materialize_group, without re-replaying from
    action 0).

    This works because whispering_earring's turn-level branch-and-adopt (facility_claws.py)
    commits each turn's action to the main line before the next turn is even branched -- so every
    branch group spawned off this trial replayed a prefix that is guaranteed to be an EXACT
    prefix of this trial's own action log (verified against real data, 0 mismatches across 840
    branch files). Before this, each of a trial's branch groups independently replayed its own
    prefix from scratch, so a trial with branch points every few turns paid for the sum of every
    prefix length instead of just the trial's own length -- on one real ~126-action trial with 28
    branch groups, that was 1759 replayed steps instead of 126, a ~15x blowup that scales O(L^2)
    in trial length rather than O(L)."""
    trial = json.loads(Path(path).read_text())
    if trial.get("error") is not None:
        return []  # aborted trial -- no value signal, per ../README.md's Data section

    groups_by_len: dict[int, list[_Group]] = defaultdict(list)
    baselines: dict[int, float] = {}
    for g in own_groups:
        baseline = _group_baseline(g)
        if baseline is None:
            continue  # same guard materialize_group applies per-group
        groups_by_len[len(g.prefix_actions)].append(g)
        baselines[id(g)] = baseline

    actions = trial["actions"]
    examples: list[Example] = []
    decision = session.start(trial["character"], trial["seed"], trial.get("ascension", 0))

    for i in range(len(actions) + 1):
        # i == len(actions) is a final sentinel pass -- covers the (unlikely but possible) case
        # of a branch group whose prefix is the trial's ENTIRE action log -- without stepping
        # anything further.
        entry = actions[i] if i < len(actions) else None
        is_policy_point = (
            entry is not None and decision.decision == "combat_play" and "policy" in entry
        )
        groups_here = groups_by_len.get(i, [])
        if is_policy_point or groups_here:
            options = legal_options_from_buffer(decision)
            batch = tokenize_from_buffer(decision, gvocab)
            if is_policy_point:
                examples.append({
                    "type_ids": batch.type_ids,
                    "global_ids": batch.global_ids,
                    "features": batch.features,
                    "player_pos": batch.player_pos,
                    "hand_pos": batch.hand_pos,
                    "enemy_pos": batch.enemy_pos,
                    "potion_pos": batch.potion_pos,
                    "options": options,
                    "value": entry.get("value", 0.0),
                    "action_index": None,
                    "advantage": 0.0,
                    "policy_real": False,
                })
            if decision.decision == "combat_play":  # defensive -- see materialize_group's guard
                for g in groups_here:
                    baseline = baselines[id(g)]
                    for first in g.members:
                        action_index = _option_index(options, first)
                        if action_index is None:
                            continue
                        examples.append({
                            "type_ids": batch.type_ids,
                            "global_ids": batch.global_ids,
                            "features": batch.features,
                            "player_pos": batch.player_pos,
                            "hand_pos": batch.hand_pos,
                            "enemy_pos": batch.enemy_pos,
                            "potion_pos": batch.potion_pos,
                            "options": options,
                            "value": first["value"],
                            "action_index": action_index,
                            "advantage": first["value"] - baseline,
                            "policy_real": True,
                        })
        if entry is None:
            break
        decision = session.step(entry["action"], **entry.get("args", {}))

    return examples


_gvocab: GlobalVocab | None = None
_session: Sts2BufferSession | None = None


def _silence_worker_stderr() -> None:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)


def _init_worker() -> None:
    _silence_worker_stderr()
    global _gvocab, _session
    _gvocab = GlobalVocab(EntityVocab.load())
    _session = Sts2BufferSession()


def _materialize_merged_task(
    path: str, own_groups: list[_Group]
) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None  # set by _init_worker before any task runs
    label = path if not own_groups else f"{path} (+{len(own_groups)} branch group(s))"
    try:
        examples = materialize_main_and_groups(path, own_groups, _gvocab, _session)
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"
    return label, examples, None


def _materialize_group_task(group: _Group) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None
    label = f"group({group.main_id}, {len(group.prefix_actions)}-action prefix, {len(group.members)} branches)"
    try:
        examples = materialize_group(group, _gvocab, _session)
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"
    return label, examples, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trials", nargs="+",
        help="trial JSON files, or directories to glob *.json from (data/raw/*.json) -- "
             "directories let a huge data/raw/ be passed as one argument instead of a shell-"
             "expanded file list, which can exceed the OS's exec argument-length limit",
    )
    parser.add_argument("--out", default="dataset.pt")
    parser.add_argument(
        "--jobs", "-j", type=int, default=os.cpu_count(),
        help="parallel worker processes, each loading its own in-process sts2-cli CLR session "
             "(default: cpu count)",
    )
    args = parser.parse_args()

    trial_paths: list[str] = []
    for trial in args.trials:
        if os.path.isdir(trial):
            trial_paths.extend(sorted(str(p) for p in Path(trial).glob("*.json")))
        else:
            trial_paths.append(trial)

    # Once, here, in the parent -- not per worker, so a first-time DLL copy/build (see
    # ensure_setup) can't race across several worker processes hitting it at once.
    ensure_setup()

    # Single read pass (cheap -- no CLR session needed, just reading the files) that both
    # separates out main-line trials and groups every branch trial by its shared prefix -- see
    # _scan_trials's docstring.
    main_id_to_path, groups = _scan_trials(trial_paths)
    real_groups = {k: g for k, g in groups.items() if _group_baseline(g) is not None}
    print(f"{len(main_id_to_path)} main-line trial(s), {len(groups)} branch group(s) "
          f"({len(real_groups)} with real cross-branch signal)")

    # Every group replays a prefix of its OWN main trial's action log (branch-and-adopt commits
    # each turn before the next is even branched -- see materialize_main_and_groups's docstring),
    # so route each group through one shared walk of its main file instead of replaying its
    # prefix from scratch. A group only falls back to the old from-scratch replay when its main
    # file is missing/errored (excluded from main_id_to_path above).
    groups_by_main_id: dict[str, list[_Group]] = defaultdict(list)
    for group in groups.values():
        groups_by_main_id[group.main_id].append(group)
    orphan_groups = [
        g for main_id, gs in groups_by_main_id.items() if main_id not in main_id_to_path for g in gs
    ]
    if orphan_groups:
        print(f"{len(orphan_groups)} branch group(s) have no matching main-line file -- "
              f"replaying independently from action 0")

    all_examples: list[Example] = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool:
        futures = [
            pool.submit(_materialize_merged_task, path, groups_by_main_id.get(main_id, []))
            for main_id, path in main_id_to_path.items()
        ]
        futures += [pool.submit(_materialize_group_task, group) for group in orphan_groups]
        for future in as_completed(futures):
            label, examples, error = future.result()
            if error is not None:
                print(f"skip {label}: {error}")
                continue
            assert examples is not None
            if examples:
                print(f"{label}: {len(examples)} examples")
            all_examples.extend(examples)

    torch.save(all_examples, args.out)
    print(f"wrote {len(all_examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
