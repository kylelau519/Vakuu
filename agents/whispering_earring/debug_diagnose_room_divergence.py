"""General-purpose diagnostic for "unexpected decision" / non-combat divergence errors -- the
class of bug behind both the map-RNG desync fixed in facility.py's _resolve_combat (see
fix-checkpoint-fast-path-rng-desync) and the still-open branch-worker version of the same issue
(see that commit's message). When a trial or branch errors with something like "turn ended on
unexpected decision 'event_choice'" or a continuous replay hits "No pending card selection" /
"Cannot play card X", the actual cause is essentially always a few non-combat decisions back --
some event, card pick, or room-entry resolved to a DIFFERENT outcome than whatever facility.py's
checkpoint-based reconstruction (main line's per-turn resync, or a branch's own
load_and_resume_combat) actually needed to stay consistent with.

This script does two things, given ANY recorded trial or branch JSON (role=="main" or
role=="branch" -- same {character, seed, ascension, actions} shape either way):

1. CONTEXT TRACE (always runs): replays the trial's actions in one continuous session from
   StartRun, and for every non-combat decision along the way (event_choice, card_reward,
   card_select, shop, rest_site, bundle_select, map_select), records what the decision actually
   offered (event name/options, cards offered, etc.) paired with what action was taken in
   response (which option/card/index was picked, decoded back to the actual identity where
   possible -- not just the raw numeric index the action log stores). Stops at the first
   Sts2ActionError (the natural failure point) or at --stop-at if given, and prints the last
   --context entries -- this directly answers "what event/choice/pick happened right before this
   broke," without having to manually cross-reference action indices against decision dumps.

2. CHECKPOINT FIDELITY CHECK (--check-checkpoint): once the point of interest is known, finds the
   last select_map_node action at or before it (the room-entry that led to whatever's now
   suspect), replays continuously up to right before it, takes ONE native checkpoint there, then
   compares (a) that same session just continuing (stepping the entry action directly) against
   (b) a completely fresh session loading that checkpoint and replaying only the entry action --
   exactly what a branch worker does. A MISMATCH here (same on-disk checkpoint, different
   resulting room) is the confirmed signature of the map-RNG desync bug: SaveCheckpoint does not
   preserve whatever RNG cursor governs map/encounter generation, so an independent reload can
   deterministically land on a different room than the session that took the checkpoint reaches
   by just continuing to play. See this module's git history / the fix-checkpoint-fast-path-rng-
   desync branch for the original investigation this generalizes.

Usage:
    python debug_diagnose_room_divergence.py <trial_or_branch.json>
    python debug_diagnose_room_divergence.py <trial_or_branch.json> --stop-at 44
    python debug_diagnose_room_divergence.py <trial_or_branch.json> --check-checkpoint
    python debug_diagnose_room_divergence.py <trial_or_branch.json> --context 12 --check-checkpoint
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

_SCRATCH_DEFAULT = "/tmp/debug_diagnose_room_divergence"


def _clean(actions: list[dict]) -> list[dict]:
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


def _decode_decision(decision) -> dict:
    """What THIS decision actually offers, for whichever non-combat decision types carry
    interesting content -- everything else just gets its bare decision type."""
    d = decision.decision
    if d == "event_choice":
        return {
            "decision": d, "event_id": decision.event_id, "event_name": decision.event_name,
            "options": [(o.index, o.title, o.is_locked) for o in decision.options],
        }
    if d == "card_reward":
        return {
            "decision": d, "can_skip": decision.can_skip,
            "cards": [(c.index, c.id, c.upgraded) for c in decision.cards],
        }
    if d == "event_card_reward":
        return {"decision": d, "cards": [(c.index, c.id, c.upgraded) for c in decision.cards]}
    if d == "card_select":
        return {
            "decision": d, "min_select": decision.min_select, "max_select": decision.max_select,
            "source_category": decision.source_category, "source_id": decision.source_id,
            "cards": [(c.index, c.id, c.upgraded) for c in decision.cards],
        }
    if d == "bundle_select":
        return {"decision": d, "bundles": [(b.index, [c.id for c in b.cards]) for b in decision.bundles]}
    if d == "map_select":
        return {"decision": d}
    return {"decision": d}


def _decode_pick(action_name: str, args: dict, offered: dict) -> str:
    """Best-effort human-readable summary of what was picked, cross-referenced against the
    decision that was actually showing (offered) rather than just echoing the raw index."""
    if action_name == "choose_option" and offered.get("decision") == "event_choice":
        idx = args.get("option_index")
        for oi, title, locked in offered.get("options", []):
            if oi == idx:
                return f"chose option {idx}: {title!r}" + (" [was locked!]" if locked else "")
        return f"chose option {idx} (not found in offered options -- already a red flag)"
    if action_name == "select_card_reward" and offered.get("decision") in ("card_reward", "event_card_reward"):
        idx = args.get("card_index")
        for ci, cid, upg in offered.get("cards", []):
            if ci == idx:
                return f"picked card_index {idx}: {cid}{' (upgraded)' if upg else ''}"
        return f"picked card_index {idx} (not found in offered cards -- already a red flag)"
    if action_name == "skip_card_reward":
        return "skipped card reward"
    if action_name == "select_cards" and offered.get("decision") == "card_select":
        raw = args.get("indices", "")
        indices = [int(x) for x in raw.split(",") if x != ""]
        by_index = {ci: (cid, upg) for ci, cid, upg in offered.get("cards", [])}
        picks = []
        for i in indices:
            if i in by_index:
                cid, upg = by_index[i]
                picks.append(f"{i}:{cid}{'(up)' if upg else ''}")
            else:
                picks.append(f"{i}:<not found in offered cards -- red flag>")
        return f"selected indices [{raw}] -> {', '.join(picks)}"
    if action_name == "select_map_node":
        return f"moved to ({args.get('col')},{args.get('row')})"
    return f"{action_name}({args})" if args else action_name


def _trace(character: str, seed: str, ascension: int, actions: list[dict], stop_at: int | None):
    """Continuous replay from StartRun, collecting one context entry per non-combat decision
    encountered (paired with what was picked in response), until failure / stop_at / end."""
    session = Sts2Session()
    decision = session.start(character, seed, ascension)
    context: list[dict] = []
    failure = None
    n = len(actions) if stop_at is None else min(stop_at + 1, len(actions))

    for i in range(n):
        cmd = actions[i]
        pre_offer = None
        if decision.decision not in ("combat_play", "game_over"):
            pre_offer = _decode_decision(decision)
        try:
            decision = session.step(cmd["action"], **cmd.get("args", {}))
        except Sts2ActionError as e:
            failure = (i, cmd, str(e), pre_offer)
            break
        if pre_offer is not None:
            context.append({
                "index": i, "offered": pre_offer,
                "picked": _decode_pick(cmd["action"], cmd.get("args", {}) or {}, pre_offer),
            })
        if stop_at is not None and i == stop_at:
            break

    session.close()
    return context, failure


def _fmt_context_entry(e: dict) -> str:
    offered = e["offered"]
    d = offered["decision"]
    lines = [f"  [action {e['index']}] decision={d}"]
    if d == "event_choice":
        lines.append(f"    event: {offered['event_id']} ({offered['event_name']!r})")
        lines.append(f"    options offered: {[(o[0], o[1]) for o in offered['options']]}")
    elif d in ("card_reward", "event_card_reward"):
        lines.append(f"    cards offered: {[(c[0], c[1], c[2]) for c in offered['cards']]}")
    elif d == "card_select":
        lines.append(f"    source: {offered.get('source_category')}/{offered.get('source_id')} "
                      f"(pick {offered['min_select']}-{offered['max_select']})")
        lines.append(f"    cards offered: {[(c[0], c[1], c[2]) for c in offered['cards']]}")
    elif d == "bundle_select":
        lines.append(f"    bundles offered: {offered['bundles']}")
    lines.append(f"    -> {e['picked']}")
    return "\n".join(lines)


def _find_last_map_entry(actions: list[dict], upto: int) -> int | None:
    for i in range(upto, -1, -1):
        if actions[i]["action"] == "select_map_node":
            return i
    return None


def _snapshot_room(decision) -> dict:
    if decision.decision == "event_choice":
        return {"decision": "event_choice", "event_id": decision.event_id, "event_name": decision.event_name}
    if decision.decision == "combat_play":
        return {"decision": "combat_play", "enemies": sorted(e.id for e in decision.enemies)}
    return {"decision": decision.decision}


def _check_checkpoint(character: str, seed: str, ascension: int, actions: list[dict], entry_index: int, scratch: str):
    os.makedirs(scratch, exist_ok=True)
    checkpoint_path = os.path.join(scratch, "checkpoint.json")
    entry_action = actions[entry_index]

    print(f"\n--- Checkpoint fidelity check around action {entry_index} ({entry_action}) ---")
    print(f"Replaying actions[0:{entry_index}] continuously, then taking a checkpoint...")
    ref = Sts2Session()
    decision = ref.start(character, seed, ascension)
    for i in range(entry_index):
        cmd = actions[i]
        decision = ref.step(cmd["action"], **cmd.get("args", {}))
    ref.save(checkpoint_path)

    ref_decision = ref.step(entry_action["action"], **entry_action.get("args", {}))
    ref_snapshot = _snapshot_room(ref_decision)
    ref.close()
    print(f"(a) same session continues (no reload) -> {ref_snapshot}")

    fresh = Sts2Session()
    fresh_decision = fresh.load_and_resume_combat(checkpoint_path, entry_action, [])
    fresh_snapshot = _snapshot_room(fresh_decision)
    fresh.close()
    print(f"(b) fresh session loads checkpoint + replays entry_action -> {fresh_snapshot}")

    if ref_snapshot == fresh_snapshot:
        print("MATCH: this checkpoint reloads faithfully -- the divergence, if any, is coming "
              "from somewhere else (check the context trace above for a mismatched pick/option).")
    else:
        print("MISMATCH: confirmed map-RNG desync -- an independent load of this exact checkpoint "
              "reaches a DIFFERENT room than the saving session's own continuation. This is the "
              "known checkpoint-format limitation (see fix-checkpoint-fast-path-rng-desync's "
              "commit message) -- SaveCheckpoint doesn't preserve map/encounter-generation RNG.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial_json", help="any role=='main' or role=='branch' trial JSON from data/raw")
    parser.add_argument("--stop-at", type=int, default=None,
                         help="stop the context trace at this action index instead of running to "
                              "failure/end (use this if the file has no natural failure -- e.g. "
                              "inspecting the moment right before a suspect select_map_node)")
    parser.add_argument("--context", type=int, default=8, help="how many trailing non-combat "
                         "decisions to print (default 8)")
    parser.add_argument("--check-checkpoint", action="store_true", help="also run the checkpoint "
                         "fidelity check around the last select_map_node before the failure/stop point")
    parser.add_argument("--scratch", default=_SCRATCH_DEFAULT)
    args = parser.parse_args()

    ensure_setup()
    trial = json.loads(open(args.trial_json).read())
    character = trial["character"]
    seed = trial["seed"]
    ascension = trial.get("ascension", 0)
    actions = _clean(trial["actions"])
    print(f"{args.trial_json}: role={trial.get('role')} outcome={trial.get('outcome')} "
          f"error={trial.get('error')!r} character={character} seed={seed} "
          f"({len(actions)} actions)")

    context, failure = _trace(character, seed, ascension, actions, args.stop_at)

    print(f"\n--- Context trace: last {args.context} non-combat decision(s) before "
          f"{'the failure' if failure else ('--stop-at' if args.stop_at is not None else 'end of log')} ---")
    if not context:
        print("  (no non-combat decisions recorded before this point)")
    for e in context[-args.context:]:
        print(_fmt_context_entry(e))

    last_index = None
    if failure is not None:
        i, cmd, err, pre_offer = failure
        print(f"\nFAILED at action {i}: {cmd} -> {err}")
        if pre_offer is not None:
            print(f"  (decision showing at the time: {pre_offer})")
        last_index = i
    elif args.stop_at is not None:
        print(f"\nStopped at --stop-at={args.stop_at} (no error).")
        last_index = args.stop_at
    else:
        print("\nReplayed the entire action log with no error.")
        last_index = len(actions) - 1

    if args.check_checkpoint:
        entry_index = _find_last_map_entry(actions, last_index)
        if entry_index is None:
            print("\nNo select_map_node found at or before this point -- nothing to checkpoint-check.")
        else:
            _check_checkpoint(character, seed, ascension, actions, entry_index, args.scratch)


if __name__ == "__main__":
    main()
