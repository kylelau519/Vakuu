"""Turns whispering_earring trial files into tokenized training examples 

Every branch spawned by whispering_earring's turn-level branch-and-adopt (see
../../whispering_earring/facility.py) replays an identical shared prefix before diverging on its
own first action. 

Usage:
    conda activate sts2
    python materialize.py ../../whispering_earring/data/raw/*.json --out dataset.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
    (`{action, args, value}`) directly -- not the sibling's path -- since that's the only thing
    about a sibling this module still needs once the group's shared prefix has been replayed."""
    main_id: str
    character: str
    seed: str
    ascension: int
    prefix_actions: list[dict]
    members: list[dict] = field(default_factory=list)


def _group_baseline(group: _Group) -> float | None:
    """The REINFORCE baseline for this group's turn-start decision: the plain mean of every
    sibling's own realized `value` (expected outcome under the branch-sampling distribution as
    it actually played out), not a mean of per-action averages -- a heavily-sampled action pulls
    the baseline toward itself exactly as much as it was actually favored by that sampling. None
    if fewer than 2 distinct first actions were observed (a single observed option is no
    comparison, not a real signal)."""
    subkeys = {_norm_action(m) for m in group.members}
    if len(subkeys) < 2:
        return None
    return sum(m["value"] for m in group.members) / len(group.members)


def _scan_trials(paths: list[str]) -> tuple[list[str], dict[tuple, _Group]]:
    """Single read pass over every trial file: classifies each as main-line (its bare path,
    for full replay via materialize_main_trial) or branch (grouped by the exact shared prefix
    it replayed before diverging)."""
    main_paths: list[str] = []
    groups: dict[tuple, _Group] = {}

    for path in paths:
        trial = json.loads(Path(path).read_text())
        if trial.get("role") != "branch":
            if trial.get("error") is None:
                main_paths.append(path)
            continue
        if trial.get("error") is not None:
            continue

        prefix_len = trial.get("prefix_len")
        actions = trial.get("actions", [])
        if prefix_len is None or len(actions) <= prefix_len:
            continue
        first = actions[prefix_len]
        if "value" not in first:
            continue  # turn never resolved (branch errored mid-turn) -- no realized outcome

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

    return main_paths, groups


def materialize_group(
    group: _Group, gvocab: GlobalVocab, session: Sts2BufferSession
) -> list[Example]:
    """Replays this group's shared prefix exactly once, tokenizes the resulting turn-start
    decision once, and emits one policy_real example per sibling directly from its own recorded
    first action + value -- see this module's docstring for why no per-sibling replay is needed."""
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


def materialize_main_trial(
    path: str, gvocab: GlobalVocab, session: Sts2BufferSession
) -> list[Example]:
    """Full ordinary replay of one main-line trial -- unaffected by the branch-group shortcut
    above, since main-line trials are few relative to branch files and never carry a
    policy_real signal (see this module's docstring)."""
    trial = json.loads(Path(path).read_text())
    if trial.get("error") is not None:
        return []  # aborted trial -- no value signal, per ../README.md's Data section

    examples: list[Example] = []
    decision = session.start(trial["character"], trial["seed"], trial.get("ascension", 0))
    for entry in trial["actions"]:
        if decision.decision == "combat_play" and "policy" in entry:
            options = legal_options_from_buffer(decision)
            batch = tokenize_from_buffer(decision, gvocab)
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


def _materialize_main_task(path: str) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None  # set by _init_worker before any task runs
    try:
        examples = materialize_main_trial(path, _gvocab, _session)
    except Exception as e:
        return path, None, f"{type(e).__name__}: {e}"
    return path, examples, None


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
    main_paths, groups = _scan_trials(trial_paths)
    real_groups = {k: g for k, g in groups.items() if _group_baseline(g) is not None}
    print(f"{len(main_paths)} main-line trial(s), {len(groups)} branch group(s) "
          f"({len(real_groups)} with real cross-branch signal)")

    all_examples: list[Example] = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool:
        futures = [pool.submit(_materialize_main_task, path) for path in main_paths]
        futures += [pool.submit(_materialize_group_task, group) for group in groups.values()]
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
