"""Turns whispering_earring MapOptionJob branch files into tokenized training examples for
GoldenCompassNet.

Replaces this package's previous design (whole-trajectory terminal +1/-1-plus-shaping-bonus
tagging on the main line -- see git history / this package's README's Design section for the full
reasoning). That signal was degenerate whenever the underlying run almost never wins: every
select_map_node decision got roughly the same flat penalty regardless of path quality, since a
trial dies at floor 6 whether it took the safe route or the risky one. map_select decisions are
now scored via real per-sibling branch walks instead (facility_common.py's
_resolve_map_decision/_run_map_option_job) -- the same "replay an identical shared prefix, then
commit to one sibling and play forward independently" pattern
choices_paradox/model/materialize.py already reads for card_reward/rest_site/event_choice
branching, see that module's own docstring. A branch's own bounded-horizon HP-fraction score is
available and meaningful regardless of the overall win rate, unlike the terminal tag it replaced.

Grouping: MapOptionJob's own recorder metadata already carries (main_id, divergence_index,
option_index) directly (see facility_common.py's _run_map_option_job), so siblings group on the
first two fields without needing prefix-action matching the way choices_paradox's own
materialize.py does (its meta-branch files predate carrying that field explicitly, so it groups
by replaying-prefix-equality instead).

Baseline: a group's own mean realized value across its siblings -- no natural "skip" reference
option exists for a map choice the way card_reward's skip_card_reward does (every physical child
is a real move), so this uses the group mean rather than choices_paradox's skip-relative
baseline. train.py's own `_normalize_advantages` still zero-means/unit-stds these again across the
whole dataset, same as everywhere else.

No main-trial-fusion optimization (unlike choices_paradox/model/materialize.py's
materialize_main_and_groups): every group here just replays its own recorded prefix once from
scratch. Slower per group than fusing with a single walk of the main trial would be -- a
deliberate v1 tradeoff, same "reasonable, not fastest" spirit this package's README already
accepted for Phase 0; worth revisiting if materialize throughput becomes a bottleneck.

Usage:
    conda activate sts2
    python materialize.py ../../whispering_earring/data/raw/*.json ../../whispering_earring/data/raw_choices/*.json --out dataset.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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

Example = dict[str, Any]


def _same_option(option: tuple[str, dict], recorded: dict) -> bool:
    return option[0] == recorded["action"] and option[1] == recorded.get("args", {})


def _action_index(options: list[tuple[str, dict]], entry: dict) -> int | None:
    for i, option in enumerate(options):
        if _same_option(option, entry):
            return i
    return None


@dataclass
class _Group:
    """Everything materialize_group() needs for one (main_id, divergence_index) group of sibling
    MapOptionJob branch files."""
    main_id: str
    character: str
    seed: str
    ascension: int
    prefix_actions: list[dict] = field(default_factory=list)
    members: list[dict] = field(default_factory=list)  # each sibling's own recorded commit action


def _group_baseline(group: _Group) -> float | None:
    """This group's own mean realized value across siblings -- see module docstring on why a mean
    baseline, not choices_paradox's skip-relative one."""
    values = [m["value"] for m in group.members if "value" in m]
    if len(values) < 2:
        return None
    return sum(values) / len(values)


def _read_trial(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _scan_trials(paths: list[str]) -> dict[tuple[str, int], _Group]:
    groups: dict[tuple[str, int], _Group] = {}
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4)) as pool:
        trials = pool.map(_read_trial, paths)
        for trial in trials:
            _scan_one(trial, groups)
    return groups


def _scan_one(trial: dict, groups: dict[tuple[str, int], _Group]) -> None:
    if trial.get("role") != "branch" or trial.get("branch_kind") != "map":
        return  # a main-line file, or some other branch_kind (combat turn, meta) -- not ours
    if trial.get("error") is not None:
        return

    main_id = trial.get("main_id")
    divergence_index = trial.get("divergence_index")
    if main_id is None or divergence_index is None:
        return

    prefix_len = trial.get("prefix_len")
    actions = trial.get("actions", [])
    if prefix_len is None or len(actions) <= prefix_len:
        return
    commit = actions[prefix_len]
    if "value" not in commit:
        return  # this sibling's walk errored/never resolved -- no realized outcome

    key = (main_id, divergence_index)
    group = groups.get(key)
    if group is None:
        group = _Group(
            main_id=main_id, character=trial["character"], seed=trial["seed"],
            ascension=trial.get("ascension", 0), prefix_actions=actions[:prefix_len],
        )
        groups[key] = group
    group.members.append(commit)


def materialize_group(
    group: _Group, gvocab: GlobalVocab, session: Sts2BufferSession
) -> list[Example]:
    """Replays this group's shared prefix exactly once, tokenizes the resulting map_select
    decision once, and emits one policy_real example per sibling directly from its own recorded
    commit action + value."""
    baseline = _group_baseline(group)
    if baseline is None:
        return []

    decision = session.start(group.character, group.seed, group.ascension)
    for entry in group.prefix_actions:
        decision = session.step(entry["action"], **entry.get("args", {}))
    if decision.decision != "map_select":
        return []  # defensive -- a map branch's own prefix should always land here

    options = legal_options_from_buffer(decision)
    batch = tokenize_from_buffer(decision, gvocab, options=options)

    examples: list[Example] = []
    for commit in group.members:
        action_index = _action_index(options, commit)
        if action_index is None:
            continue
        examples.append({
            "type_ids": batch.type_ids,
            "global_ids": batch.global_ids,
            "features": batch.features,
            "player_pos": batch.player_pos,
            "option_pos": batch.option_pos,
            "n_options": len(options),
            "value": commit["value"],
            "action_index": action_index,
            "advantage": commit["value"] - baseline,
            "policy_real": True,
        })
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


def _materialize_group_task(group: _Group) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None
    label = f"group({group.main_id}, div {group.prefix_actions and len(group.prefix_actions)}, {len(group.members)} siblings)"
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

    groups = _scan_trials(trial_paths)
    real_groups = [g for g in groups.values() if _group_baseline(g) is not None]
    print(f"{len(groups)} map-branch group(s) ({len(real_groups)} with real cross-option signal)")

    all_examples: list[Example] = []
    per_file_counts: list[tuple[str, int]] = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool:
        futures = [pool.submit(_materialize_group_task, group) for group in real_groups]
        for future in as_completed(futures):
            label, examples, error = future.result()
            if error is not None:
                print(f"skip {label}: {error}")
                continue
            assert examples is not None
            if examples:
                per_file_counts.append((label, len(examples)))
            all_examples.extend(examples)

    _print_truncated_counts(per_file_counts)
    torch.save(all_examples, args.out)
    print(f"wrote {len(all_examples)} examples to {args.out}")


def _print_truncated_counts(per_file_counts: list[tuple[str, int]], head_tail: int = 5) -> None:
    """Print "{label}: N examples" per group, but only the first/last `head_tail` when there are
    many -- with hundreds of raw trial files the full per-group listing just floods the terminal."""
    total = len(per_file_counts)
    if total <= 2 * head_tail:
        for label, n in per_file_counts:
            print(f"{label}: {n} examples")
        return
    for label, n in per_file_counts[:head_tail]:
        print(f"{label}: {n} examples")
    print(f"... ({total - 2 * head_tail} more group(s)) ...")
    for label, n in per_file_counts[-head_tail:]:
        print(f"{label}: {n} examples")


if __name__ == "__main__":
    main()
