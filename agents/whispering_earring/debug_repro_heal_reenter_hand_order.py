"""Test whether RunSimulator.HealAndEnterCombat (sts2-cli's cheap same-run episode-boundary
reset -- see [[whispering_earring_heal_reenter_branching_plan]]) hands back the SAME opening hand
each time it re-enters a given encounter, or a DIFFERENT one.

This matters for the planned heal-reenter combat-variant architecture: if it's meant to spin up
many policy variants that all face the *same* starting hand (only diverging by policy action after
that), the hand must be reproducible across re-entries. If instead the underlying draw/shuffle RNG
is a single continuously-advancing per-run stream (as StS-style roguelikes typically do), each
re-entry naturally draws a *different* hand -- that's expected behavior, not a bug, but it means
"variants" from repeated HealAndEnterCombat calls are not directly comparable unless each call is
immediately followed by capturing state, with no interleaved combat actions.

Two things are checked:
  1. Within one trial: does calling HealAndEnterCombat N times in a row (same encounter, no cards
     played in between) draw the same hand every time, or a different one each time?
  2. Across independent trials (same character/seed/ascension/encounter/action sequence, fresh
     Sts2Session per trial): does the FULL sequence of N hands reproduce identically trial to
     trial? Run serially first, then concurrently (--concurrent) to check for the same
     shared-user-data-dir race documented in debug_repro_branch_divergence.py, now against
     HealAndEnterCombat specifically instead of StartRun-replay.

Usage:
    python debug_repro_heal_reenter_hand_order.py [--character Ironclad] [--seed SEED]
        [--ascension 0] [--encounter SHRINKER_BEETLE_WEAK] [--n-reenters 5] [--n-trials 4]
        [--concurrent]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
from installation import ensure_setup  # noqa: E402
from env import Sts2Session, Sts2ActionError, _to_decision  # noqa: E402


def _heal_and_enter_combat(session: Sts2Session, encounter: str, hp: int | None = None):
    cmd = {"cmd": "heal_and_enter_combat", "encounter": encounter}
    if hp is not None:
        cmd["hp"] = hp
    raw = session._send_raw(cmd)
    return _to_decision(raw)


def _hand_ids(decision) -> list[str | None]:
    return [c.id for c in decision.hand]


def _run_trial(character: str, seed: str, ascension: int, encounter: str, n_reenters: int,
                worker_id: int, result_queue: "mp.Queue", prefix_actions: list[dict] | None) -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)  # silence this worker's stderr noise

    try:
        session = Sts2Session()
        if prefix_actions:
            # Realistic branch-worker setup: replay a real trial's prefix (map/event/combat RNG
            # consumption included) instead of jumping straight from StartRun into the target
            # encounter -- this is the scenario debug_repro_branch_divergence.py found actually
            # diverges under concurrency, unlike the bare start_run->heal_and_enter_combat case.
            session.open_from_checkpoint(character, seed, ascension, prefix_actions)
        else:
            session.start(character, seed, ascension)
        hands = []
        for _ in range(n_reenters + 1):
            decision = _heal_and_enter_combat(session, encounter)
            hands.append(_hand_ids(decision))
        session.close()
        result_queue.put((worker_id, hands, None))
    except (Sts2ActionError, Exception) as ex:  # noqa: BLE001
        result_queue.put((worker_id, None, str(ex)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed", default="HEALREENTERTEST")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--encounter", default="SHRINKER_BEETLE_WEAK")
    parser.add_argument("--n-reenters", type=int, default=5,
                         help="Extra HealAndEnterCombat calls after the first entry (default 5, "
                              "so 6 hands are captured per trial).")
    parser.add_argument("--n-trials", type=int, default=4,
                         help="Independent trials, each a fresh Sts2Session replaying the exact "
                              "same start_run + heal_and_enter_combat*N sequence (default 4).")
    parser.add_argument("--concurrent", action="store_true",
                         help="Run all trials as concurrent subprocesses instead of serially, to "
                              "check for the same shared-save-path race found in "
                              "debug_repro_branch_divergence.py.")
    parser.add_argument("--trial-json", default=None,
                         help="A main-line trial file (data/raw/<id>.json) whose `actions` are "
                              "replayed as a realistic setup prefix before heal-reentering, "
                              "instead of jumping straight from start_run into --encounter. "
                              "Overrides --character/--seed/--ascension from the file.")
    parser.add_argument("--prefix-len", type=int, default=None,
                         help="Only replay the first K actions of --trial-json.")
    args = parser.parse_args()

    ensure_setup()

    prefix_actions = None
    if args.trial_json:
        import json
        with open(args.trial_json) as f:
            trial = json.load(f)
        args.character, args.seed, args.ascension = trial["character"], trial["seed"], trial["ascension"]
        prefix_actions = [
            {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
            for e in trial["actions"]
        ]
        if args.prefix_len is not None:
            prefix_actions = prefix_actions[:args.prefix_len]

    print(f"Character={args.character} seed={args.seed!r} ascension={args.ascension} "
          f"encounter={args.encounter}")
    setup_desc = f"open_from_checkpoint({len(prefix_actions)} actions)" if prefix_actions else "start_run"
    print(f"Each trial: {setup_desc} -> heal_and_enter_combat x{args.n_reenters + 1} "
          f"({'concurrent' if args.concurrent else 'serial'}, {args.n_trials} trials)\n")

    result_queue: mp.Queue = mp.Queue()
    trial_args = [
        (args.character, args.seed, args.ascension, args.encounter, args.n_reenters, i, result_queue,
         prefix_actions)
        for i in range(args.n_trials)
    ]

    if args.concurrent:
        procs = [mp.Process(target=_run_trial, args=a) for a in trial_args]
        for p in procs:
            p.start()
        results = [result_queue.get() for _ in procs]
        for p in procs:
            p.join()
    else:
        results = []
        for a in trial_args:
            _run_trial(*a)
            results.append(result_queue.get())

    results.sort(key=lambda r: r[0])

    errored = [r for r in results if r[2] is not None]
    for worker_id, _, error in errored:
        print(f"trial {worker_id}: ERROR -- {error}")

    ok_results = [r for r in results if r[2] is None]
    if not ok_results:
        print("\nAll trials errored -- nothing to compare.")
        sys.exit(1)

    # --- 1. Within-trial: does re-entering the same encounter repeatedly draw the same hand? ---
    print("--- Within-trial hand sequence (does re-entry redraw the same hand each time?) ---")
    for worker_id, hands, _ in ok_results:
        all_same = all(h == hands[0] for h in hands)
        print(f"trial {worker_id}: {'IDENTICAL every re-entry' if all_same else 'CHANGES across re-entries'}")
        for i, h in enumerate(hands):
            print(f"    entry {i}: {h}")

    # --- 2. Across-trial: does the same scripted sequence reproduce bit-for-bit trial to trial? ---
    print("\n--- Across-trial reproducibility (same seed+sequence, independent sessions) ---")
    baseline_id, baseline_hands, _ = ok_results[0]
    n_diverged = 0
    for worker_id, hands, _ in ok_results[1:]:
        if hands == baseline_hands:
            print(f"trial {worker_id}: matches trial {baseline_id} bit-for-bit")
        else:
            n_diverged += 1
            first_diff = next(i for i in range(len(hands)) if hands[i] != baseline_hands[i])
            print(f"trial {worker_id}: DIVERGED from trial {baseline_id} at entry {first_diff} "
                  f"({hands[first_diff]} vs {baseline_hands[first_diff]})")

    print(f"\n{n_diverged}/{len(ok_results) - 1} trials diverged from the baseline "
          f"({'concurrent' if args.concurrent else 'serial'} run).")
    sys.exit(1 if (n_diverged or errored) else 0)


if __name__ == "__main__":
    main()
