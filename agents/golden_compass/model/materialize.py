"""Turns whispering_earring main-line trial files into tokenized training examples for
GoldenCompassNet.

Unlike choices_paradox/model/materialize.py (which reads whispering_earring's MetaOptionJob
sibling-branch files to get a causal baseline per decision), Golden Compass has no branching of
its own: every map_select decision on a trial's own main line already carries a `value` once
facility_common.py's pending_map_steps/recorder.set_value wiring tags it at trial end -- see that
module's main_worker_loop. That value is this trial's own terminal outcome (+1 won / -1 lost)
PLUS a backward-discounted local shaping bonus (+2/+3 for a won Elite/Boss fight, decayed by
_GOLDEN_COMPASS_GAMMA per select_map_node step of distance) -- the flat terminal term is kept, not
replaced, so a trial that beats a couple of Elites and still dies nets negative overall rather
than teaching the policy to chase risky Elites over safe clears; the shaping term just densifies
credit assignment along the way instead of smearing the same flat scalar across every decision in
an up-to-~50-decision run. This holds regardless of whether the decision was actually chosen by a
trained GoldenCompassNet or by golden_compass/policy.py's DP heuristic fallback (no
GOLDEN_COMPASS_CHECKPOINT set yet): the `value` tagging is unconditional, so even round-1
(heuristic-driven) data is already usable, same "reasonable, not trained" self-play bootstrap
every other subsystem here starts from.

Advantage is the raw realized value itself (no baseline subtraction -- there are no sibling
branches to compute one from): +1 pushes the taken action's probability up on train.py's
advantage-weighted log-likelihood loss, -1 pushes it down. train.py's own `_normalize_advantages`
still zero-means/unit-stds these across the dataset before training, same as everywhere else.

Replays via the pydantic Sts2Session (not Sts2BufferSession, unlike claws'/choices_paradox's own
materialize.py) -- GoldenCompassNet has no buffer-side counterpart to tokenize()/
combination_options() yet (map_select decisions are recorded on the main line only for now; see
this package's README's Phasing section), so there is no buffer-replay path to use here. Slower
per trial than a buffer replay would be, a deliberate v1 tradeoff -- worth revisiting if
materialize throughput becomes a bottleneck.

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
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see model.py's top -- two OpenMP
# runtimes (conda's llvm-openmp + torch's bundled copy) otherwise abort on import

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

import sts2_path  # noqa: E402,F401  (sets sys.path to sts2-cli/python before the imports below)
from env import Sts2Session  # noqa: E402
from installation import ensure_setup  # noqa: E402

from model import GlobalVocab, _map_node_index, combination_options, tokenize  # noqa: E402

Example = dict[str, Any]


def _action_index(combos: list, combo_paths: dict, entry: dict) -> int | None:
    """Which of combination_options()'s distinct combinations this recorded step actually picked.

    When the live decision went through a trained net (_choose_map_action_net), the exact index
    is already recorded in entry["policy"]["chosen"] -- see whispering_earring/policy.py -- and
    is used directly rather than re-derived, since more than one combination can share the same
    resolved (col, row) (see model.py's own docstring on why that ambiguity exists) and only the
    live softmax draw actually disambiguates it. combination_options()'s key order is
    deterministic given the same decision/map (see that function's docstring), so an index
    recorded live is still valid against combos recomputed here during replay.

    Round-1 (heuristic-driven, no GOLDEN_COMPASS_CHECKPOINT set yet) steps carry no such policy,
    so this falls back to the first combination whose path set contains a path starting at the
    actually-taken (col, row) -- an arbitrary but deterministic tie-break when more than one
    combination matches, since the heuristic never reasoned in terms of combinations at all."""
    policy = entry.get("policy")
    if policy is not None and "chosen" in policy:
        idx = policy["chosen"]
        return idx if 0 <= idx < len(combos) else None

    taken = (entry["args"]["col"], entry["args"]["row"])
    for i, combo in enumerate(combos):
        if any(path[0] == taken for path in combo_paths[combo]):
            return i
    return None


def materialize_main_trial(path: str, gvocab: GlobalVocab, session: Sts2Session) -> list[Example]:
    """Full ordinary replay of one main-line trial, emitting one example per map_select step
    that carries a `value` (i.e. the trial reached game_over as a definite win/loss, not an
    error -- see facility_common.py's outcome handling)."""
    trial = json.loads(Path(path).read_text())
    if trial.get("role", "main") != "main" or trial.get("error") is not None:
        return []

    examples: list[Example] = []
    decision = session.start(trial["character"], trial["seed"], trial.get("ascension", 0))
    for entry in trial["actions"]:
        if decision.decision == "map_select" and "value" in entry:
            node_index = _map_node_index(session.get_map())
            combo_paths = combination_options(decision, node_index)
            combos = list(combo_paths.keys())
            action_index = _action_index(combos, combo_paths, entry)
            if action_index is not None:
                batch = tokenize(decision, gvocab, combos=combos)
                examples.append({
                    "type_ids": batch.type_ids,
                    "global_ids": batch.global_ids,
                    "features": batch.features,
                    "player_pos": batch.player_pos,
                    "option_pos": batch.option_pos,
                    "n_options": len(combos),
                    "value": entry["value"],
                    "action_index": action_index,
                    "advantage": entry["value"],  # no baseline -- see module docstring
                    "policy_real": True,
                })
        decision = session.step(entry["action"], **entry.get("args", {}))
    return examples


_gvocab: GlobalVocab | None = None
_session: Sts2Session | None = None


def _silence_worker_stderr() -> None:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)


def _init_worker() -> None:
    _silence_worker_stderr()
    global _gvocab, _session
    _gvocab = GlobalVocab(EntityVocab.load())
    _session = Sts2Session()


def _materialize_main_task(path: str) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None
    try:
        examples = materialize_main_trial(path, _gvocab, _session)
    except Exception as e:
        return path, None, f"{type(e).__name__}: {e}"
    return path, examples, None


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

    all_examples: list[Example] = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool:
        futures = [pool.submit(_materialize_main_task, path) for path in trial_paths]
        for future in as_completed(futures):
            path, examples, error = future.result()
            if error is not None:
                print(f"skip {path}: {error}")
                continue
            assert examples is not None
            if examples:
                print(f"{path}: {len(examples)} examples")
            all_examples.extend(examples)

    torch.save(all_examples, args.out)
    print(f"wrote {len(all_examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
