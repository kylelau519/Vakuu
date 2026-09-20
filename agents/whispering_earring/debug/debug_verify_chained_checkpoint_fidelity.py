"""Does a CHAIN of save/load hops stay faithful to uninterrupted play, or do errors compound?

debug_verify_checkpoint_fidelity.py already answers "is one save/load hop faithful against a
session that was never checkpointed at all" -- necessary but not sufficient for the
choices_paradox redesign, which wants to checkpoint at every combat entry along a multi-combat
option-branch walk, each new checkpoint taken from a session that was itself reconstructed from
the previous checkpoint, not from a pristine from-StartRun session. A per-hop bug too small for
the single-hop check to catch (or one that only manifests on a session that's already been through
a load_save once) could silently compound across many hops without ever showing up there.

Method: play ONE reference run straight through with plain session.step() calls (never
checkpointed), across --n-combats combats, snapshotting combat state --n-turns-to-check turns into
each. Independently build a "chain" session that reconstructs the same walk hop by hop: for each
combat, replay forward to just before the room-entry action, save, then start a FRESH session that
loads that save to continue -- so hop i+1's checkpoint always descends from a session built by
loading hop i's checkpoint, never from the pristine reference. At every hop, an independent
same-checkpoint reload is diffed against the reference's own snapshot at that same point.

Usage:
    CLAWS_CHECKPOINT=agents/claws/model/claws.pt python debug_verify_chained_checkpoint_fidelity.py \\
        [--character Ironclad] [--seed SEED] [--ascension 0] [--n-combats 4] [--n-turns-to-check 3]
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault(
    "CLAWS_CHECKPOINT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "claws", "model", "claws.pt"),
)  # a decent combat policy matters here -- the reference walk needs to actually WIN several
# combats in a row, not just survive one turn; default only applies if the caller didn't already
# set one (empty/missing is handled by policy.py's own _load_claws_net fallback)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
from installation import ensure_setup  # noqa: E402
from env import Sts2Session, Sts2ActionError  # noqa: E402
from policy import choose_action  # noqa: E402

_SCRATCH = "/private/tmp/claude-501/-Users-kylelau519-Programming-Vakuu/6a1bcaa2-b06d-4ddd-90fa-bded7dc84550/scratchpad/chain_fidelity_check"


def _clean_actions(actions: list[dict]) -> list[dict]:
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


def _combat_snapshot(decision) -> dict:
    """Same fields debug_verify_checkpoint_fidelity.py checks -- see its own docstring."""
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
    return [f"  {k}: reference={a[k]!r}  reconstructed={b[k]!r}" for k in a if a[k] != b[k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed", default="CHAINFIDELITY")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--n-combats", type=int, default=4)
    parser.add_argument("--n-turns-to-check", type=int, default=3,
                         help="check fidelity this many turns into each combat")
    args = parser.parse_args()

    os.makedirs(_SCRATCH, exist_ok=True)
    ensure_setup()

    # --- Phase 1: play the reference run straight through, never checkpointed -----------------
    print(f"Playing {args.character}/{args.seed} through {args.n_combats} combat(s), uninterrupted "
          f"(reference)...")
    ref_session = Sts2Session()
    decision = ref_session.start(args.character, args.seed, args.ascension)
    pending_purpose = None

    combats_seen = 0
    in_combat = False
    end_turns_this_combat = 0
    snapshot_taken_this_combat = False
    hops: list[dict] = []  # one entry per combat: entry_index, checkpoint_index, snapshot

    usable_hops = 0
    try:
        while decision.decision != "game_over":
            if not in_combat and decision.decision == "combat_play":
                in_combat = True
                combats_seen += 1
                end_turns_this_combat = 0
                snapshot_taken_this_combat = False
                entry_index = len(ref_session.action_log) - 1
                hops.append({"entry_index": entry_index})
                print(f"  combat {combats_seen}: entered at action index {entry_index}")

            if in_combat and not snapshot_taken_this_combat and end_turns_this_combat >= args.n_turns_to_check:
                hops[-1]["checkpoint_index"] = len(ref_session.action_log)
                hops[-1]["snapshot"] = _combat_snapshot(decision)
                snapshot_taken_this_combat = True
                usable_hops += 1
                print(f"    snapshot taken at round {decision.round}, action index {len(ref_session.action_log)}")
                if usable_hops >= args.n_combats:
                    break  # have every hop we need -- no need to resolve this last combat at all

            name, kwargs, _, pending_purpose = choose_action(decision, ref_session, pending_purpose)
            decision = ref_session.step(name, **kwargs)

            if in_combat and name == "end_turn":
                end_turns_this_combat += 1
            if in_combat and decision.decision != "combat_play" and decision.decision != "card_select":
                # combat resolved (won -- card_reward/etc next) or the run ended (game_over)
                if not snapshot_taken_this_combat:
                    print(f"    combat {combats_seen} ended before reaching "
                          f"{args.n_turns_to_check} turns -- no snapshot for this hop, skipping it")
                    hops.pop()
                in_combat = False
    except Sts2ActionError as e:
        print(f"Reference run errored: {e}")
        ref_session.close()
        sys.exit(1)

    if decision.decision == "game_over":
        print(f"  (reference run ended: {'victory' if decision.victory else 'DEFEAT'} after "
              f"{combats_seen} combat(s) seen)")

    hops = [h for h in hops if "snapshot" in h]
    if not hops:
        print("No hops with a usable snapshot -- rerun with a different --seed or more --n-combats.")
        ref_session.close()
        sys.exit(1)

    full_action_log = _clean_actions(ref_session.action_log)
    ref_session.close()
    print(f"\nReference run complete: {len(hops)} usable hop(s) out of {combats_seen} combat(s) seen, "
          f"{len(full_action_log)} total actions.\n")

    # --- Phase 2: build the chain, one save/load hop per combat, verifying each against the ---
    # --- reference snapshot independently -----------------------------------------------------
    entry_indices = [h["entry_index"] for h in hops]
    all_mismatches: dict[int, list[str]] = {}

    chain_session = Sts2Session()
    chain_session.start(args.character, args.seed, args.ascension)
    replayed_to = 0  # how much of full_action_log chain_session has already stepped through

    for hop_num, hop in enumerate(hops, start=1):
        entry_index = hop["entry_index"]
        checkpoint_index = hop["checkpoint_index"]
        entry_action = full_action_log[entry_index]
        in_combat_actions = full_action_log[entry_index + 1:checkpoint_index]

        # Advance chain_session to just before this combat's entry action (pre-entry, matching
        # facility_common.py's own checkpoint-taking convention and load_and_resume_combat's
        # documented contract), then save -- this checkpoint descends from wherever chain_session
        # currently is, which for hop >= 2 is itself a session reconstructed via load_save below,
        # not a from-StartRun session.
        for cmd in full_action_log[replayed_to:entry_index]:
            chain_session.step(cmd["action"], **cmd.get("args", {}))
        checkpoint_path = os.path.join(_SCRATCH, f"chain_hop_{hop_num:02d}.json")
        chain_session.save(checkpoint_path)
        print(f"Hop {hop_num}: chain checkpoint saved at action index {entry_index} "
              f"({'from-StartRun session' if hop_num == 1 else 'a reconstructed session'}).")

        # Independent verification: a FRESH session loading only this checkpoint (never touching
        # chain_session or ref_session), replaying just entry_action + this combat's own actions
        # up to the snapshot point -- diffed against the reference's pristine snapshot.
        verify_session = Sts2Session()
        verify_session.load_save(checkpoint_path)
        v_decision = None
        for cmd in [entry_action, *in_combat_actions]:
            v_decision = verify_session.step(cmd["action"], **cmd.get("args", {}))
        mismatches = _diff(hop["snapshot"], _combat_snapshot(v_decision))
        verify_session.close()
        if mismatches:
            all_mismatches[hop_num] = mismatches
            print(f"  MISMATCH at hop {hop_num}: {len(mismatches)} field(s) differ")
            for d in mismatches:
                print(d)
        else:
            print(f"  MATCH at hop {hop_num}")

        # Continue the chain FROM ITS OWN JUST-TAKEN CHECKPOINT (not by continuing chain_session
        # in place) -- this is what makes hop i+1's own checkpoint descend from a reconstructed
        # session rather than a continuously-running one, the true test of compounding.
        chain_session.close()
        chain_session = Sts2Session()
        chain_session.load_save(checkpoint_path)
        next_bound = entry_indices[hop_num] if hop_num < len(hops) else len(full_action_log)
        for cmd in full_action_log[entry_index:next_bound]:
            chain_session.step(cmd["action"], **cmd.get("args", {}))
        replayed_to = next_bound

    chain_session.close()

    print()
    if not all_mismatches:
        print(f"MATCH across all {len(hops)} hop(s): checkpoint chaining is faithful to "
              f"uninterrupted play, no compounding drift detected.")
        sys.exit(0)
    else:
        print(f"MISMATCH at {len(all_mismatches)}/{len(hops)} hop(s): {sorted(all_mismatches)} -- "
              f"see per-hop diffs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
