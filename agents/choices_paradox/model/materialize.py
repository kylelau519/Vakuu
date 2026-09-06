"""Turns whispering_earring meta-branch trial files (card_reward, and eventually card_select's
remove/upgrade purposes into tokenized training examples. 
Structural copy of claws/model/materialize.py's shared-prefix grouping, but the
baseline this file is NOT claws' own grand-mean.

Every meta-branch spawned by whispering_earring's _resolve_meta_decision replays an identical
shared prefix up to one card_reward/card_select decision, then commits to one of that decision's
legal options and plays forward on its own.

Baseline: a group's `skip_card_reward`/`skip_select` member's own realized value,

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

# Saved as plain dicts of builtins/tensors, not TokenBatch/dataclass instances -- same reasoning
# as claws/model/materialize.py (torch.save pickles a class by import path, unloadable from a
# different entry script).
Example = dict[str, Any]


def _same_option(option: tuple[str, dict], recorded: dict) -> bool:
    return option[0] == recorded["action"] and option[1] == recorded.get("args", {})


def _option_index(options: list[tuple[str, dict]], entry: dict) -> int | None:
    for i, option in enumerate(options):
        if _same_option(option, entry):
            return i
    return None


def _norm_action(entry: dict) -> tuple[str, tuple]:
    return (entry["action"], tuple(sorted(entry.get("args", {}).items())))


@dataclass
class _Group:
    """Everything materialize_group() needs for one (main_id, decision_kind, shared-prefix)
    group of sibling meta-branch trials."""
    main_id: str
    character: str
    seed: str
    ascension: int
    decision_kind: str
    prefix_actions: list[dict]
    members: list[dict] = field(default_factory=list)


_SKIP_ACTIONS = frozenset({"skip_card_reward", "skip_select", "leave_room"})


def _group_baseline(group: _Group) -> float | None:
    """Causal baseline: this group's own skip-option member's realized `value`."""
    subkeys = {_norm_action(m) for m in group.members}
    if len(subkeys) < 2:
        return None
    for m in group.members:
        if m["action"] in _SKIP_ACTIONS:
            return m["value"]
    return None


def _read_trial(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _scan_trials(paths: list[str]) -> tuple[dict[str, str], dict[tuple, _Group]]:
    """Single read pass, same shape as claws' own _scan_trials -- see its docstring (including
    for why this returns main_id -> path rather than a plain path list)."""
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
    if trial.get("branch_kind") != "meta":
        return  # combat's own turn-branch file -- claws/model/materialize.py's territory, not ours
    if trial.get("error") is not None:
        return

    prefix_len = trial.get("prefix_len")
    actions = trial.get("actions", [])
    if prefix_len is None or len(actions) <= prefix_len:
        return
    first = actions[prefix_len]
    if "value" not in first:
        return  # branch never reached its horizon/died mid-replay -- no realized outcome

    decision_kind = trial.get("decision_kind")
    if decision_kind is None:
        return

    prefix_actions = actions[:prefix_len]
    prefix_key = (trial["main_id"], decision_kind, tuple(_norm_action(a) for a in prefix_actions))
    group = groups.get(prefix_key)
    if group is None:
        group = _Group(
            main_id=trial["main_id"], character=trial["character"], seed=trial["seed"],
            ascension=trial.get("ascension", 0), decision_kind=decision_kind,
            prefix_actions=prefix_actions,
        )
        groups[prefix_key] = group
    group.members.append(first)


def materialize_group(
    group: _Group, gvocab: GlobalVocab, session: Sts2BufferSession
) -> list[Example]:
    """Replays this group's shared prefix exactly once, tokenizes the resulting card_reward/
    card_select decision once, and emits one policy_real example per member directly from its
    own recorded first action + value. Only used as a fallback for a meta-branch group whose
    main-line file is missing/errored (see main()) -- when the main file IS available,
    materialize_main_and_groups() below reaches the same checkpoint without replaying this
    group's prefix from scratch."""
    baseline = _group_baseline(group)
    if baseline is None:
        return []

    decision = session.start(group.character, group.seed, group.ascension)
    for entry in group.prefix_actions:
        decision = session.step(entry["action"], **entry.get("args", {}))
    if decision.decision not in (
        "card_reward", "event_card_reward", "card_select", "rest_site", "event_choice"
    ):
        return []  # defensive -- a meta-branch checkpoint should always land on one of these.
        # event_card_reward is the blessing-triggered buffer-side variant of a "card_reward"
        # decision_kind (STS2's JSON protocol unifies both under one "card_reward" literal, but
        # the buffer protocol splits them -- see model.py's legal_options_from_buffer docstring).

    options = legal_options_from_buffer(decision, group.decision_kind)
    batch = tokenize_from_buffer(decision, group.decision_kind, gvocab)

    examples: list[Example] = []
    for first in group.members:
        action_index = _option_index(options, first)
        if action_index is None:
            continue
        examples.append({
            "type_ids": batch.type_ids,
            "global_ids": batch.global_ids,
            "features": batch.features,
            "mode_idx": batch.mode_idx,
            "player_pos": batch.player_pos,
            "mode_pos": batch.mode_pos,
            "offer_pos": batch.offer_pos,
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
    at each shared-prefix boundary one of its OWN sibling meta-branch groups diverged from, that
    group's policy_real=True examples (same as materialize_group, without re-replaying from
    action 0). See claws/model/materialize.py's own materialize_main_and_groups for why every
    meta-branch group's own recorded prefix is guaranteed to be an exact prefix of THIS trial's
    own action log (whispering_earring's branch-and-adopt commits each turn before the next is
    even branched), and why walking it once here instead of replaying each group's prefix from
    scratch is a large win on trials with many meta-decisions."""
    trial = json.loads(Path(path).read_text())
    if trial.get("error") is not None:
        return []

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
        # of a meta-branch group whose prefix is the trial's ENTIRE action log.
        entry = actions[i] if i < len(actions) else None
        is_policy_point = (
            entry is not None
            and decision.decision in (
                "card_reward", "event_card_reward", "rest_site", "event_choice"
            )
            and "policy" in entry
        )
        if is_policy_point:
            decision_kind = (
                "card_reward" if decision.decision in ("card_reward", "event_card_reward")
                else decision.decision
            )
            options = legal_options_from_buffer(decision, decision_kind)
            batch = tokenize_from_buffer(decision, decision_kind, gvocab)
            examples.append({
                "type_ids": batch.type_ids,
                "global_ids": batch.global_ids,
                "features": batch.features,
                "mode_idx": batch.mode_idx,
                "player_pos": batch.player_pos,
                "mode_pos": batch.mode_pos,
                "offer_pos": batch.offer_pos,
                "options": options,
                "value": entry.get("value", 0.0),
                "action_index": None,
                "advantage": 0.0,
                "policy_real": False,
            })

        groups_here = groups_by_len.get(i, [])
        if groups_here and decision.decision in (
            "card_reward", "event_card_reward", "card_select", "rest_site", "event_choice"
        ):
            # All groups sharing one (main_id, prefix) key share one decision_kind -- same
            # assumption materialize_group itself makes (there's only one real game decision at
            # this exact point in this exact trial).
            decision_kind = groups_here[0].decision_kind
            options = legal_options_from_buffer(decision, decision_kind)
            batch = tokenize_from_buffer(decision, decision_kind, gvocab)
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
                        "mode_idx": batch.mode_idx,
                        "player_pos": batch.player_pos,
                        "mode_pos": batch.mode_pos,
                        "offer_pos": batch.offer_pos,
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
    assert _gvocab is not None and _session is not None
    label = path if not own_groups else f"{path} (+{len(own_groups)} branch group(s))"
    try:
        examples = materialize_main_and_groups(path, own_groups, _gvocab, _session)
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"
    return label, examples, None


def _materialize_group_task(group: _Group) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None
    label = (
        f"group({group.main_id}, {group.decision_kind}, "
        f"{len(group.prefix_actions)}-action prefix, {len(group.members)} branches)"
    )
    try:
        examples = materialize_group(group, _gvocab, _session)
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"
    return label, examples, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trials", nargs="+",
        help="trial JSON files, or directories to glob *.json from (data/raw/*.json)",
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

    ensure_setup()  # once, here, in the parent -- see claws/model/materialize.py's comment

    main_id_to_path, groups = _scan_trials(trial_paths)
    real_groups = {k: g for k, g in groups.items() if _group_baseline(g) is not None}
    print(f"{len(main_id_to_path)} main-line trial(s), {len(groups)} meta-branch group(s) "
          f"({len(real_groups)} with real cross-option signal)")

    # Every group replays a prefix of its OWN main trial's action log, so route each group
    # through one shared walk of its main file instead of replaying its prefix from scratch --
    # see materialize_main_and_groups's docstring. A group only falls back to the old
    # from-scratch replay when its main file is missing/errored.
    groups_by_main_id: dict[str, list[_Group]] = defaultdict(list)
    for group in groups.values():
        groups_by_main_id[group.main_id].append(group)
    orphan_groups = [
        g for main_id, gs in groups_by_main_id.items() if main_id not in main_id_to_path for g in gs
    ]
    if orphan_groups:
        print(f"{len(orphan_groups)} meta-branch group(s) have no matching main-line file -- "
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
