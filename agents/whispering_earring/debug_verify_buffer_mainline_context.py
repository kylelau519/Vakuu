"""Does facility_claws._combat_context and facility_common._map_point_type_at -- the two spots
that had to be rewritten off buffer_layout accessors so the main line could switch from Sts2Session
to Sts2BufferSession -- actually agree with the pydantic Sts2Session ground truth they replaced?

Both functions branch on hasattr(decision, "arr")/isinstance to pick a buffer or pydantic path;
this drives ONE reference run on Sts2Session (via the real choose_action policy, so it exercises
whatever the game actually offers rather than a synthetic script), records its exact action log,
then REPLAYS that identical action log step-by-step on a fresh Sts2BufferSession (same seed => same
underlying game state, the same replay-determinism assumption load_and_resume_combat/
open_from_checkpoint already rely on everywhere else in this codebase) and diffs the two functions'
output at every combat entry and every select_map_node this trial hits.

Usage:
    python debug_verify_buffer_mainline_context.py [--character Ironclad] [--seed SEED]
        [--ascension 1] [--max-decisions 400]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
from installation import ensure_setup  # noqa: E402
from env import Sts2Session  # noqa: E402
from buffer_env import Sts2BufferSession  # noqa: E402
from policy import choose_action  # noqa: E402
from facility_claws import _combat_context  # noqa: E402
from facility_common import _map_point_type_at  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed", default="BUFCTXCHK")
    parser.add_argument("--ascension", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=400)
    args = parser.parse_args()

    ensure_setup()

    print(f"Reference pass: playing {args.character}/{args.seed} (asc {args.ascension}) on "
          f"Sts2Session via the real policy, recording actions + expected checks...")

    ref = Sts2Session()
    decision = ref.start(args.character, args.seed, args.ascension)
    pending_purpose = None
    checks: list[tuple[int, str, tuple | str | None]] = []  # (step_index_after, kind, expected)
    steps = 0
    while decision.decision != "game_over" and steps < args.max_decisions:
        steps += 1
        if decision.decision == "combat_play":
            # Combat entry check -- capture BEFORE playing into it (matches _resolve_combat's own
            # call site, which reads context off the decision that first shows combat_play).
            checks.append((len(ref.action_log), "combat_context", _combat_context(decision)))
        name, kwargs, _, pending_purpose = choose_action(decision, ref, pending_purpose)
        if name == "select_map_node":
            # Evaluated on the PRE-step map_select decision, same as _play_trial's own call site.
            checks.append((
                len(ref.action_log), "map_point_type",
                (_map_point_type_at(decision, kwargs["col"], kwargs["row"]),
                 kwargs["col"], kwargs["row"]),
            ))
        decision = ref.step(name, **kwargs)
    ref.close()

    full_log = ref.action_log
    print(f"Reference reached {'game_over' if decision.decision == 'game_over' else 'max-decisions cutoff'} "
          f"after {len(full_log)} actions, with {len(checks)} check point(s) recorded.")

    if not checks:
        print("No combat/map-select decisions hit at all -- nothing to check. Try a different seed.")
        return

    print("Replay pass: driving the identical action log through Sts2BufferSession...")
    buf = Sts2BufferSession()
    buf_decision = buf.start(args.character, args.seed, args.ascension)
    check_by_index = {idx: (kind, expected) for idx, kind, expected in checks}
    mismatches = []
    for i, cmd in enumerate(full_log):
        if i in check_by_index:
            kind, expected = check_by_index[i]
            if kind == "combat_context":
                actual = _combat_context(buf_decision)
            else:
                expected_type, col, row = expected
                actual = (_map_point_type_at(buf_decision, col, row), col, row)
                expected = (expected_type, col, row)
            if actual != expected:
                mismatches.append((i, kind, expected, actual))
        args_ = cmd.get("args")
        buf_decision = buf.step(cmd["action"], **(args_ or {}))
    # Trailing check (index == len(full_log), i.e. a combat_context captured on the very last
    # decision reference stopped on, or a map check whose step is the final action) is covered by
    # the loop above since check_by_index keys are all < len(full_log) except possibly the last
    # combat entry captured right at the cutoff -- handle that edge case explicitly.
    if len(full_log) in check_by_index:
        kind, expected = check_by_index[len(full_log)]
        actual = _combat_context(buf_decision) if kind == "combat_context" else None
        if actual is not None and actual != expected:
            mismatches.append((len(full_log), kind, expected, actual))
    buf.close()

    print(f"\n{len(checks)} check(s), {len(mismatches)} mismatch(es).")
    for idx, kind, expected, actual in mismatches:
        print(f"  [step {idx}] {kind} MISMATCH\n    reference (pydantic): {expected!r}\n    buffer:               {actual!r}")
    if not mismatches:
        print("PASS -- buffer-derived combat/map context matches the pydantic ground truth exactly.")
    else:
        print("FAIL -- see mismatches above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
