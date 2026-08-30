"""Turns whispering_earring trial files into tokenized training examples -- the "materializing
training features" step CompactRecorder's docstring calls out as a deliberate, one-time offline
step, since trial files store only the action log + recorded policy/value, never full state (see
../README.md's Data section).

Replays each trial's action log through a reused-per-worker Sts2BufferSession (same mechanism as
Sts2BufferSession.open_from_checkpoint, done manually here so the pre-action decision is visible
at each step), and at every combat_play decision that carries a recorded `policy`, tokenizes the
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
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _target_probs(options: list[tuple[str, dict]], recorded_policy: list[dict]) -> torch.Tensor:
    """Aligns the model's own (fuller) legal_options_from_buffer() against whatever the trial actually
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


def materialize_trial(path: str, gvocab: GlobalVocab, session: Sts2BufferSession) -> list[Example]:
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
                "target": _target_probs(options, entry["policy"]),
                "value": entry.get("value", 0.0),
            })
        decision = session.step(entry["action"], **entry.get("args", {}))
    return examples


_gvocab: GlobalVocab | None = None
_session: Sts2BufferSession | None = None


def _silence_worker_stderr() -> None:
    """Sts2BufferSession's CLR writes unprompted [SIM]/[INFO] lines straight to this process's
    real fd 2 on every action (see clr_bridge.silence_stderr's docstring), and Console.Error dups
    fd 2 once on first access -- redirecting afterwards is a no-op. So this must run before the
    first Sts2BufferSession() in this process. Unlike facility.py's _tag_worker_stderr (which
    relays through a tagged pipe so several branch workers stay distinguishable on one terminal),
    materialize.py's workers have nothing worth surfacing there -- failures already come back
    through _materialize_one's return value -- so this just drops fd 2 to devnull for good."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)


def _init_worker() -> None:
    # One session per worker, reused across every trial that worker handles via session.start()
    # (not one Sts2BufferSession() per trial) -- constructing a BufferSession is the expensive
    # part (it loads this process's in-process CLR on first use, see buffer_env.py/clr_bridge.py,
    # and allocates the native buffer), while start() just resets it for a new run. Trials still
    # parallelize across worker processes with no shared state beyond this session and the
    # read-only vocab, both built once here rather than once per trial.
    _silence_worker_stderr()
    global _gvocab, _session
    _gvocab = GlobalVocab(EntityVocab.load())
    _session = Sts2BufferSession()


def _materialize_one(path: str) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None  # set by _init_worker before any task runs
    try:
        examples = materialize_trial(path, _gvocab, _session)
    except Exception as e:
        return path, None, f"{type(e).__name__}: {e}"
    return path, examples, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", help="trial JSON files (data/raw/*.json)")
    parser.add_argument("--out", default="dataset.pt")
    parser.add_argument(
        "--jobs", "-j", type=int, default=os.cpu_count(),
        help="parallel worker processes, each loading its own in-process sts2-cli CLR session "
             "(default: cpu count)",
    )
    args = parser.parse_args()

    # Once, here, in the parent -- not per worker, so a first-time DLL copy/build (see
    # ensure_setup) can't race across several worker processes hitting it at once.
    ensure_setup()

    all_examples: list[Example] = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool:
        futures = [pool.submit(_materialize_one, path) for path in args.trials]
        for future in as_completed(futures):
            path, examples, error = future.result()
            if error is not None:
                print(f"skip {path}: {error}")
                continue
            assert examples is not None
            print(f"{path}: {len(examples)} examples")
            all_examples.extend(examples)

    torch.save(all_examples, args.out)
    print(f"wrote {len(all_examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
