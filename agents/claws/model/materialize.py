"""Turns whispering_earring trial files into tokenized training examples -- the "materializing
training features" step CompactRecorder's docstring calls out as a deliberate, one-time offline
step, since trial files store only the action log + recorded policy/value, never full state (see
../README.md's Data section).

Replays each trial's action log through a fresh sts2-cli session (same mechanism as
Sts2Session.open_from_checkpoint, done manually here so the pre-action decision is visible at
each step), and at every combat_play decision that carries a recorded `policy`, tokenizes the
state that was actually in front of the policy when it was recorded.

Usage:
    conda activate sts2
    python materialize.py ../../whispering_earring/data/raw/*.json --out dataset.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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

from model import GlobalVocab, legal_options, tokenize  # noqa: E402

# Saved as plain dicts of builtins/tensors, not TokenBatch/dataclass instances -- torch.save
# pickles a class by its import path, and TokenBatch would resolve to "__main__.TokenBatch" when
# this script is run directly, unloadable from any other module. Plain dicts sidestep that.
Example = dict[str, Any]


def _same_option(option: tuple[str, dict], recorded: dict) -> bool:
    return option[0] == recorded["action"] and option[1] == recorded.get("args", {})


def _target_probs(options: list[tuple[str, dict]], recorded_policy: list[dict]) -> torch.Tensor:
    """Aligns the model's own (fuller) legal_options() against whatever the trial actually
    recorded. whispering_earring's placeholder policy only ever samples one target per card
    (see model.py's module docstring), so recorded_policy is typically a strict subset of
    options -- unmatched options get target probability 0, same as an unvisited MCTS child
    would. Not renormalized: recorded probs already sum to ~1 over the entries they do cover."""
    target = torch.zeros(len(options))
    for i, option in enumerate(options):
        for recorded in recorded_policy:
            if _same_option(option, recorded):
                target[i] = recorded["prob"]
                break
    return target


def materialize_trial(path: str, gvocab: GlobalVocab) -> list[Example]:
    trial = json.loads(Path(path).read_text())
    if trial.get("error") is not None:
        return []  # aborted trial -- no value signal, per ../README.md's Data section

    session = Sts2Session()
    examples: list[Example] = []
    try:
        decision = session.start(trial["character"], trial["seed"], trial.get("ascension", 0))
        for entry in trial["actions"]:
            if decision.decision == "combat_play" and "policy" in entry:
                options = legal_options(decision)
                batch = tokenize(decision, gvocab)
                examples.append({
                    "type_ids": batch.type_ids,
                    "global_ids": batch.global_ids,
                    "features": batch.features,
                    "player_pos": batch.player_pos,
                    "hand_pos": batch.hand_pos,
                    "enemy_pos": batch.enemy_pos,
                    "potion_pos": batch.potion_pos,
                    "options": options,
                    "target": _target_probs(options, entry["policy"]),
                    "value": entry.get("value", 0.0),
                })
            decision = session.step(entry["action"], **entry.get("args", {}))
    finally:
        session.close()
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", help="trial JSON files (data/raw/*.json)")
    parser.add_argument("--out", default="dataset.pt")
    args = parser.parse_args()

    ensure_setup()  # once, here -- not per-trial, which would repeat the same checks
    gvocab = GlobalVocab(EntityVocab.load())

    all_examples: list[Example] = []
    for path in args.trials:
        try:
            examples = materialize_trial(path, gvocab)
        except Exception as e:
            print(f"skip {path}: {type(e).__name__}: {e}")
            continue
        print(f"{path}: {len(examples)} examples")
        all_examples.extend(examples)

    torch.save(all_examples, args.out)
    print(f"wrote {len(all_examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
