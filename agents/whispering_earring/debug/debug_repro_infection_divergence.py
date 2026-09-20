"""Root-cause repro for the "Cannot play card Infection: HasUnplayableKeyword" divergence found
on a real production trial (Ironclad_ember-623230, floor 11 / combat_index 5) -- Kyle flagged this
as a real regression from the checkpoint-based turn branching in facility.py.

Confirmed already (see conversation): the MAIN trial's own recorded absolute action log fails a
single continuous session.start()+step() replay at action index 69, inside combat 5, with this
exact error -- meaning whatever state facility.py's checkpoint+entry_action+per-turn reconstruction
actually reached during live generation is NOT reachable via continuous replay of the same
recorded actions. Since the OLD open_from_checkpoint-based per-turn reconstruction was mechanically
identical to continuous replay (full StartRun replay every turn), this divergence has to come from
the NEW save_checkpoint/load_save/entry_action mechanism specifically.

This script replays combat 5 turn-by-turn TWICE from the same real trial's recorded actions:
  1. "reference": one continuous session, no reconstruction at all.
  2. "reconstructed": mimics facility.py's _resolve_turn exactly -- one save_checkpoint at combat
     entry, then load_and_resume_combat (load_save + replay entry_action + replay this combat's
     actions-so-far) at the start of EVERY subsequent turn, same as _resolve_turn's per-turn
     teardown/rebuild.
and diffs full hand contents at every turn boundary, to find the FIRST turn where they diverge.

Usage:
    python debug_repro_infection_divergence.py <main_trial.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
from installation import ensure_setup  # noqa: E402
from env import Sts2Session, Sts2ActionError  # noqa: E402


def _clean(actions: list[dict]) -> list[dict]:
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


def _hand_snapshot(decision) -> list[tuple]:
    return [(c.id, c.upgraded, sorted(c.keywords or [])) for c in decision.hand]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_json")
    parser.add_argument("--combat-index", type=int, default=5)
    parser.add_argument("--scratch", default="/private/tmp/claude-501/-Users-kylelau519-Programming-Vakuu/51b26a12-6771-434f-99bb-9b02d640b324/scratchpad/infection_repro")
    args = parser.parse_args()

    os.makedirs(args.scratch, exist_ok=True)
    ensure_setup()

    trial = json.loads(open(args.trial_json).read())
    character, seed, ascension = trial["character"], trial["seed"], trial.get("ascension", 0)
    actions = _clean(trial["actions"])

    # --- Pass 1: reference continuous replay, recording turn boundaries within the target combat. ---
    ref = Sts2Session()
    decision = ref.start(character, seed, ascension)
    combat_index = 0
    in_target_combat = False
    entry_idx = None
    turn_boundaries: list[int] = []  # action indices where a new turn's combat_play decision starts
    hand_at_boundary: dict[int, list[tuple]] = {}

    ref_failed_at = None
    for i, cmd in enumerate(actions):
        was_combat = decision.decision == "combat_play"
        try:
            decision = ref.step(cmd["action"], **cmd.get("args", {}))
        except Sts2ActionError as e:
            ref_failed_at = (i, cmd, str(e))
            break
        now_combat = decision.decision == "combat_play"
        if now_combat and not was_combat:
            combat_index += 1
            if combat_index == args.combat_index:
                in_target_combat = True
                entry_idx = i  # cmd at index i was the entry action
                turn_boundaries.append(i + 1)
                hand_at_boundary[i + 1] = _hand_snapshot(decision)
            elif in_target_combat:
                break  # moved past the target combat
        elif in_target_combat and now_combat and cmd["action"] == "end_turn":
            turn_boundaries.append(i + 1)
            hand_at_boundary[i + 1] = _hand_snapshot(decision)

    ref.close()
    if ref_failed_at is not None:
        i, cmd, err = ref_failed_at
        print(f"Reference continuous replay itself failed at action {i}: {cmd} -> {err}")
        print(f"(this is the ALREADY-CONFIRMED failure -- comparing only turn boundaries reached before it)")
    print(f"Target combat_index={args.combat_index}: entry action at index {entry_idx}, "
          f"{len(turn_boundaries)} turn boundaries at {turn_boundaries}")

    if entry_idx is None:
        print("Never reached that combat_index in a continuous replay -- nothing to compare.")
        sys.exit(1)

    prior_actions = actions[:entry_idx]
    entry_action = actions[entry_idx]
    checkpoint_path = os.path.join(args.scratch, "checkpoint.json")

    # Take the checkpoint the same way facility.py's main_worker_loop does: BEFORE the entry
    # action, from a session that has only replayed prior_actions.
    setup = Sts2Session()
    setup.start(character, seed, ascension)
    for cmd in prior_actions:
        setup.step(cmd["action"], **cmd.get("args", {}))
    setup.save_checkpoint(checkpoint_path)
    setup.close()

    # --- Pass 2: reconstructed, mimicking _resolve_turn's per-turn teardown/rebuild. ---
    combat_actions_so_far: list[dict] = []
    recon = Sts2Session()
    decision = recon.load_and_resume_combat(
        checkpoint_path, character, seed, ascension, prior_actions, entry_action, combat_actions_so_far,
    )
    print(f"\nTurn 1 (right after entry) hand: {_hand_snapshot(decision)}")
    ref_hand = hand_at_boundary[turn_boundaries[0]]
    print(f"Reference   turn 1 hand: {ref_hand}")
    if _hand_snapshot(decision) != ref_hand:
        print("DIVERGED already at turn 1 (right after entry_action, before any turn actions).")
        sys.exit(1)
    print("Turn 1 matches.\n")

    for turn_num, boundary_idx in enumerate(turn_boundaries[1:], start=2):
        prev_boundary = turn_boundaries[turn_num - 2]
        new_actions_this_turn = actions[prev_boundary:boundary_idx]
        try:
            for cmd in new_actions_this_turn:
                decision = recon.step(cmd["action"], **cmd.get("args", {}))
        except Sts2ActionError as e:
            print(f"Turn {turn_num}: reconstructed session ERRORED mid-turn: {e}")
            sys.exit(1)
        combat_actions_so_far = actions[entry_idx + 1:boundary_idx]

        recon.close()
        recon = Sts2Session()
        try:
            decision = recon.load_and_resume_combat(
                checkpoint_path, character, seed, ascension,
                prior_actions, entry_action, combat_actions_so_far,
            )
        except Sts2ActionError as e:
            print(f"Turn {turn_num}: reconstruction (load+replay) ERRORED: {e}")
            sys.exit(1)

        recon_hand = _hand_snapshot(decision)
        ref_hand = hand_at_boundary[boundary_idx]
        print(f"Turn {turn_num}: reconstructed hand = {recon_hand}")
        print(f"         reference    hand = {ref_hand}")
        if recon_hand != ref_hand:
            print(f"\n>>> FIRST DIVERGENCE at turn {turn_num} (action index {boundary_idx}) <<<")
            sys.exit(1)
        print(f"Turn {turn_num} matches.\n")

    print("All turn boundaries matched -- no divergence found in this combat.")


if __name__ == "__main__":
    main()
