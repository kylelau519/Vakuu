"""Does facility_claws._resolve_turn actually NEED to close and respawn a fresh Sts2Session
(new dotnet subprocess) on every combat turn, or would reusing ONE live session across repeated
load_save() calls (load -> run -> save -> load -> run -> save, never closing the process) work
just as well?

Motivating question, verified empirically rather than assumed: the confirmed map-RNG-desync bug
(debug_diagnose_room_divergence.py's docstring, and facility_claws._resolve_combat's own
post-win "safe_session" comment) is that SaveCheckpoint/load_save does NOT preserve whatever RNG
cursor governs FUTURE map/encounter generation -- proven by comparing "the session that took the
checkpoint, continuing to play" against "a completely fresh session loading that same
checkpoint". That comparison never actually isolated "fresh process" as its own variable -- it's
entirely plausible the desync is caused by load_save() itself (a property of the save format /
engine state it does and doesn't restore), in which case reusing one live process across many
load_save() calls would hit the EXACT SAME desync as today's respawn-every-turn code, no better
and no worse -- meaning respawning buys no RNG-safety benefit at all, and reuse would be a free
efficiency win (skips the dotnet subprocess fork/exec + CLR cold start on every turn) given the
post-win full-replay repair step already exists and would stay necessary either way.

This script checks that directly: play one reference run continuously through a full combat and
a couple of rooms past it (ground truth). Then, from the SAME pre-combat checkpoint and the SAME
recorded action sequence, reconstruct that combat two ways -- (a) RESPAWN: close + spawn a fresh
Sts2Session + load_save + replay-so-far on every turn, exactly matching _resolve_turn's current
code; (b) REUSE: load_save + replay-so-far on ONE session that's never closed. Both variants
then apply the SAME post-win full open_from_checkpoint repair _resolve_combat already does before
continuing into the next room(s), and both are compared against the continuous-play baseline at
each subsequent landmark (room type / encounter identity). A third REUSE-NO-REPAIR variant skips
that repair, to confirm the repair step is still necessary regardless of reuse vs. respawn (i.e.
that the desync source really is load_save() itself, not process freshness) -- diagnostic only,
not part of the go/no-go decision.

Usage:
    python debug_verify_session_reuse.py [--character Ironclad] [--seed SEED] [--ascension 0]
        [--rooms-past-combat 2]

RESOLVED: all three variants matched continuous play on every seed tried, including the
REUSE-NO-REPAIR one this script expected to mismatch -- root-caused afterward to sts2-cli's
RunSimulator.SaveLoad.cs commit 3f47140 ("BugFix: enhance load_save method to preserve
UnknownMapPoint odds during mid-act resume"), already merged and active in this checkout's
sts2-cli. load_save's EnterAct call was unconditionally resetting the "?"-node pity-timer odds
to base even though RunState.FromSerializable had just correctly restored them, which is exactly
the desync this script's REUSE-NO-REPAIR variant should have reproduced if it still existed.
_resolve_combat's post-win full-replay repair has since been removed as a result (see its git
history) -- this script's own "no repair" finding, not just the upstream fix, is the empirical
half of that confirmation.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
from installation import ensure_setup  # noqa: E402
from env import Sts2Session, Sts2ActionError, _to_decision  # noqa: E402
from policy import choose_action  # noqa: E402

_SCRATCH = "/private/tmp/claude-501/-Users-kylelau519-Programming-Vakuu/902201f5-2576-4fe5-afc9-277f6cf6c707/scratchpad/session_reuse_check"


def _clean(actions: list[dict]) -> list[dict]:
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


def _landmark(decision) -> tuple:
    """Identifying signature for comparing across variants -- decision kind, plus (for a combat)
    the actual enemy set (encounter identity), which is exactly what a map-RNG desync would
    change even when the chosen map coordinate is identical."""
    if decision.decision == "combat_play":
        return ("combat_play", tuple(sorted((e.id, e.max_hp) for e in decision.enemies)))
    if decision.decision == "event_choice":
        return ("event_choice", decision.event_name if hasattr(decision, "event_name") else None)
    return (decision.decision, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed", default="REUSECHK")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--rooms-past-combat", type=int, default=2,
                         help="how many further map_select landmarks to compare after the combat ends")
    args = parser.parse_args()

    os.makedirs(_SCRATCH, exist_ok=True)
    ensure_setup()
    checkpoint_path = os.path.join(_SCRATCH, "pre_combat.json")

    # ---- Reference: one continuous session, no checkpointing at all ---------------------------
    print(f"Playing {args.character}/{args.seed} continuously through one combat plus "
          f"{args.rooms_past_combat} room(s) past it (reference)...")
    ref = Sts2Session()
    decision = ref.start(args.character, args.seed, args.ascension)
    pending_purpose = None
    prefix: list[dict] = []

    # Phase 1: play until combat is entered, checkpointing before every select_map_node
    # (overwriting each time -- exactly main_worker_loop's own "don't know which one leads to
    # combat yet" pattern) so the final save is the one right before the room that turns out to
    # be a fight.
    try:
        while decision.decision not in ("combat_play", "game_over"):
            name, kwargs, _, pending_purpose = choose_action(decision, ref, pending_purpose)
            if name == "select_map_node":
                ref.save(checkpoint_path)
                prefix = _clean(ref.action_log)
            decision = ref.step(name, **kwargs)
    except Sts2ActionError as e:
        print(f"Reference run errored before reaching a combat: {e}")
        ref.close()
        sys.exit(1)

    if decision.decision != "combat_play":
        print("Never reached a combat -- rerun with a different --seed.")
        ref.close()
        sys.exit(1)

    combat_entry_idx = len(ref.action_log) - 1
    entry_action = _clean(ref.action_log[combat_entry_idx:combat_entry_idx + 1])[0]
    print(f"  entered combat at action index {combat_entry_idx}")

    # Phase 2: play the combat itself, boundary EXACTLY matching what _resolve_turn/_resolve_combat
    # replay (every action from entry through whatever ends the fight, e.g. the winning end_turn/
    # play_card) -- stops the instant the game itself leaves combat_play/card_select (the same
    # states _play_turn_only's own loop covers), i.e. right at the pending card_reward.
    combat_start_len = len(ref.action_log)
    try:
        while decision.decision in ("combat_play", "card_select"):
            name, kwargs, _, pending_purpose = choose_action(decision, ref, pending_purpose)
            decision = ref.step(name, **kwargs)
    except Sts2ActionError as e:
        print(f"Reference run errored mid-combat: {e}")
        ref.close()
        sys.exit(1)

    if decision.decision != "card_reward":
        print(f"Combat ended in {decision.decision!r}, not a win -- rerun with a different --seed "
              f"(need a win to test the post-win repair path).")
        ref.close()
        sys.exit(1)

    combat_actions = _clean(ref.action_log[combat_start_len:])
    # Recover per-turn action counts by splitting on end_turn, so each variant below can resync
    # turn-by-turn exactly like _resolve_turn does (once per turn, not once for the whole combat).
    turn_boundaries: list[int] = []
    for i, a in enumerate(combat_actions):
        if a["action"] == "end_turn":
            turn_boundaries.append(i + 1)
    if not turn_boundaries or turn_boundaries[-1] != len(combat_actions):
        turn_boundaries.append(len(combat_actions))

    print(f"  combat resolved (win) in {len(turn_boundaries)} turn(s), {len(combat_actions)} "
          f"in-combat action(s) total")

    # Phase 3: continue past the combat (starting with resolving this card_reward itself, same as
    # main_worker_loop's own choose_action-driven path once a combat_resolver hands control back),
    # recording landmarks at each subsequent select_map_node.
    post_combat_start_len = len(ref.action_log)
    landmarks: list[tuple] = []
    rooms_seen = 0
    try:
        while decision.decision != "game_over" and rooms_seen < args.rooms_past_combat:
            name, kwargs, _, pending_purpose = choose_action(decision, ref, pending_purpose)
            decision = ref.step(name, **kwargs)
            if name == "select_map_node":
                landmarks.append(_landmark(decision))
                rooms_seen += 1
    except Sts2ActionError as e:
        print(f"Reference run errored while collecting landmarks: {e}")
    post_combat_actions = _clean(ref.action_log[post_combat_start_len:])
    ref.close()
    print(f"  reference landmarks: {landmarks}")

    def _repair(session: Sts2Session, full_action_log: list[dict]) -> tuple[Sts2Session, object]:
        """Exactly facility_claws._resolve_combat's own post-win repair: full StartRun replay,
        mechanically identical to continuous play, before this session drives future map RNG."""
        session.close()
        safe = Sts2Session()
        d = safe.open_from_checkpoint(args.character, args.seed, args.ascension, full_action_log)
        return safe, d

    def _run_variant(label: str, reuse: bool, repair: bool) -> list[tuple]:
        """Reconstructs the SAME recorded combat turn-by-turn, either respawning a fresh
        Sts2Session each turn (reuse=False, matching _resolve_turn's current code exactly) or
        replaying load_save + the combat's actions-so-far on ONE session that's never closed
        (reuse=True) -- both via load_and_resume_combat's own _send_raw-based replay (its public
        API already does exactly the raw load_save + replay this needs, so no hand-rolled replay
        loop is needed for either variant)."""
        print(f"\n--- variant: {label} ---")
        session = None
        d = None
        for boundary in turn_boundaries:
            so_far = combat_actions[:boundary]
            if reuse:
                session = session or Sts2Session()
            else:
                if session is not None:
                    session.close()
                session = Sts2Session()
            d = session.load_and_resume_combat(checkpoint_path, entry_action, so_far)
        full_log = prefix + [entry_action] + combat_actions
        if repair:
            session, d = _repair(session, full_log)
        else:
            session.action_log = full_log

        variant_landmarks = []
        for cmd in post_combat_actions:
            name = cmd["action"]
            raw = session._send_raw(cmd)
            d = _to_decision(raw)
            session.action_log.append(cmd)
            if name == "select_map_node":
                variant_landmarks.append(_landmark(d))
        session.close()
        print(f"  {label} landmarks: {variant_landmarks}")
        return variant_landmarks

    respawn_landmarks = _run_variant("RESPAWN (today's _resolve_turn behavior) + repair", reuse=False, repair=True)
    reuse_landmarks = _run_variant("REUSE (one live session, load_save repeatedly) + repair", reuse=True, repair=True)
    reuse_no_repair_landmarks = _run_variant("REUSE, no post-win repair (diagnostic)", reuse=True, repair=False)

    print("\n=== Summary ===")
    print(f"reference:            {landmarks}")
    print(f"respawn + repair:     {respawn_landmarks}  {'MATCH' if respawn_landmarks == landmarks else 'MISMATCH'}")
    print(f"reuse + repair:       {reuse_landmarks}  {'MATCH' if reuse_landmarks == landmarks else 'MISMATCH'}")
    print(f"reuse, no repair:     {reuse_no_repair_landmarks}  "
          f"{'MATCH (unexpected!)' if reuse_no_repair_landmarks == landmarks else 'MISMATCH (expected -- repair is still needed)'}")

    ok = respawn_landmarks == landmarks and reuse_landmarks == landmarks
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
