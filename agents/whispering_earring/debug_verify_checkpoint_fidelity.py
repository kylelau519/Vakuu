"""Does load_and_resume_combat (save_checkpoint + entry_action + in-combat action replay)
reconstruct a state that's actually FAITHFUL to uninterrupted play, not just internally
consistent across repeated loads of itself?

debug_repro_load_save_divergence.py only checked that concurrent load_save calls agree with
EACH OTHER (decision type + HP). That rules out a race/nondeterminism bug, but not a silent
state-fidelity bug: save_checkpoint's serialization could systematically drop or corrupt some
field (deck order, discard/exhaust piles, enemy powers/intents, player powers/block/energy) and
every re-load would still agree with every other re-load -- they'd all be reading the same wrong
save. The only way to catch that is to compare against a session that was NEVER checkpointed at
all: play one reference run straight through with plain session.step() calls, and independently
reconstruct the same point via save_checkpoint+load_save+entry_action+replay, then diff full
combat state field by field.

Usage:
    python debug_verify_checkpoint_fidelity.py [--character Ironclad] [--seed SEED]
        [--ascension 0] [--n-turns-to-check 3]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
from installation import ensure_setup  # noqa: E402
from env import Sts2Session, Sts2ActionError  # noqa: E402
from policy import choose_action  # noqa: E402

_SCRATCH = "/private/tmp/claude-501/-Users-kylelau519-Programming-Vakuu/51b26a12-6771-434f-99bb-9b02d640b324/scratchpad/fidelity_check"


def _clean_actions(actions: list[dict]) -> list[dict]:
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


def _combat_snapshot(decision) -> dict:
    """Every field a training example would actually read -- see policy.py's
    _combat_options_from_decision/models.py's CombatPlay for what's used downstream."""
    return {
        "round": decision.round,
        "energy": decision.energy,
        "player_hp": decision.player.hp,
        "player_block": decision.player.block,
        "player_powers": sorted((p.id, p.amount) for p in (decision.player_powers or [])),
        "hand": [(c.id, c.upgraded, c.cost) for c in decision.hand],
        "draw_pile": [(c.id, c.upgraded) for c in decision.draw_pile] if decision.draw_pile else None,
        "discard_pile": [(c.id, c.upgraded) for c in decision.discard_pile] if decision.discard_pile else None,
        "exhaust_pile": [(c.id, c.upgraded) for c in decision.exhaust_pile] if decision.exhaust_pile else None,
        "potions": [p.id for p in decision.player.potions],
        "enemies": [
            (e.id, e.hp, e.max_hp, e.block,
             sorted((p.id, p.amount) for p in (e.powers or [])),
             [(i.type, i.damage) for i in (e.intents or [])])
            for e in decision.enemies
        ],
    }


def _diff(a: dict, b: dict) -> list[str]:
    diffs = []
    for k in a:
        if a[k] != b[k]:
            diffs.append(f"  {k}: reference={a[k]!r}  reconstructed={b[k]!r}")
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed", default="FIDELITYCHK")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--n-turns-to-check", type=int, default=3,
                         help="check fidelity after this many turns into the first combat reached")
    parser.add_argument("--checkpoint-when", choices=["post-entry", "pre-entry"], default="post-entry",
                         help="post-entry matches facility.py's current _resolve_combat (checkpoint "
                              "after combat_play is first seen); pre-entry checkpoints right before "
                              "the room-entry action instead, to test whether RNG-stream position "
                              "at save time is why post-entry mismatches.")
    args = parser.parse_args()

    os.makedirs(_SCRATCH, exist_ok=True)
    ensure_setup()

    print(f"Playing {args.character}/{args.seed} to the first combat, then {args.n_turns_to_check} "
          f"turns into it (reference, uninterrupted)...")

    ref_session = Sts2Session()
    decision = ref_session.start(args.character, args.seed, args.ascension)
    entered_combat = False
    combat_entry_idx = None
    end_turns_taken = 0
    checkpoint_path = os.path.join(_SCRATCH, "checkpoint.json")
    entry_action = None
    pending_purpose = None  # threaded through choose_action -- see its docstring

    try:
        while decision.decision != "game_over":
            if entered_combat and end_turns_taken >= args.n_turns_to_check:
                break  # stop right at the start of round n_turns_to_check+1 -- decision is combat_play

            if not entered_combat:
                # HYPOTHESIS CHECK: checkpoint BEFORE every non-combat action (overwriting the
                # same file), not after combat_play is detected -- args.checkpoint_when controls
                # which one this run actually does, to compare both against the same reference.
                if args.checkpoint_when == "pre-entry":
                    ref_session.save(checkpoint_path)

            name, kwargs, _, pending_purpose = choose_action(decision, ref_session, pending_purpose)
            decision = ref_session.step(name, **kwargs)

            if not entered_combat and decision.decision == "combat_play":
                entered_combat = True
                combat_entry_idx = len(ref_session.action_log) - 1
                entry_action = _clean_actions(ref_session.action_log[combat_entry_idx:combat_entry_idx + 1])[0]
                if args.checkpoint_when == "post-entry":
                    ref_session.save(checkpoint_path)
                print(f"Entered combat at action index {combat_entry_idx}, "
                      f"checkpoint_when={args.checkpoint_when}.")
                continue

            if entered_combat and name == "end_turn":
                end_turns_taken += 1
    except Sts2ActionError as e:
        print(f"Reference run errored before reaching the target depth: {e}")
        ref_session.close()
        sys.exit(1)

    if not entered_combat or decision.decision != "combat_play":
        print("Reference run ended before/without settling on a combat_play state to compare -- "
              "rerun with a different --seed.")
        ref_session.close()
        sys.exit(1)

    combat_actions_so_far = _clean_actions(ref_session.action_log[combat_entry_idx + 1:])

    reference_snapshot = _combat_snapshot(decision)
    reference_hp = decision.player.hp
    print(f"Reference snapshot taken at round {decision.round}, "
          f"{len(combat_actions_so_far)} in-combat actions replayed to get there.")
    ref_session.close()

    print("\nReconstructing the same point via save_checkpoint + load_save + entry_action + "
          "in-combat replay (the new load_and_resume_combat path)...")
    recon_session = Sts2Session()
    recon_decision = recon_session.load_and_resume_combat(
        checkpoint_path, entry_action, combat_actions_so_far,
    )
    reconstructed_snapshot = _combat_snapshot(recon_decision)
    recon_session.close()

    print(f"Reconstructed at round {recon_decision.round}, player_hp={recon_decision.player.hp} "
          f"(reference was round {decision.round}, player_hp={reference_hp})\n")

    diffs = _diff(reference_snapshot, reconstructed_snapshot)
    if not diffs:
        print("MATCH: every checked field (hand/draw/discard/exhaust/potions/enemies/powers/"
              "energy/block/hp) is identical between uninterrupted play and the "
              "checkpoint-reconstructed session.")
        sys.exit(0)
    else:
        print(f"MISMATCH: {len(diffs)} field(s) differ:")
        for d in diffs:
            print(d)
        sys.exit(1)


if __name__ == "__main__":
    main()
