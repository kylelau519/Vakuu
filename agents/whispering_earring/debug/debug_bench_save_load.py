"""Benchmark: is write_continue_save/load_save cost independent of floor depth, unlike
open_from_checkpoint's full-history replay (see [[sts2cli_history_counts_tradeoff]] and
facility.py's per-turn branch cost, which scales with total actions since floor 1)?

Kyle's proposed fix (2026-08-31): SaveCheckpoint already rolls back mid-combat saves to the
pre-room map state (RunSimulator.SaveLoad.cs's TryRollbackSerializedSaveToPreRoom) -- loading
that save and re-entering the same room deterministically resumes the same encounter, since
event choices and card rewards are now seed-locked. So instead of open_from_checkpoint replaying
every action since the run began, a branch worker could load_save a cached per-combat checkpoint
(captured once, by the main line, at combat start) and replay only *this combat's* actions so
far -- bounded by turn count, not floor count. The only open question before wiring that in:
how fast is load_save itself, and does it scale with the save's embedded history?

This script plays one real run forward with the facility's own default (random-fallback) policy,
and at several floor checkpoints times:
  1. write_continue_save  (main line, in the process that's been playing)
  2. load_save             (fresh subprocess, from that save file)
  3. open_from_checkpoint  (fresh subprocess, full replay of the same actions-so-far)
for direct comparison at the same point in the same run.

Usage:
    python debug_bench_save_load.py [--character Ironclad] [--seed SEED] [--ascension 0]
        [--max-floor 17] [--checkpoint-every-floors 3]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts2_path  # noqa: E402,F401
from installation import ensure_setup  # noqa: E402
from env import Sts2Session, Sts2ActionError, _to_decision  # noqa: E402
from policy import choose_action  # noqa: E402


def _write_continue_save(session: Sts2Session, path: str) -> dict:
    return session._send_raw({"cmd": "write_continue_save", "path": path})


def _load_save(session: Sts2Session, path: str) -> dict:
    return session._send_raw({"cmd": "load_save", "path": path})


def _clean_actions(actions: list[dict]) -> list[dict]:
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--seed", default="SAVEBENCH")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=17, help="stop playing once this floor is reached")
    parser.add_argument("--checkpoint-every-floors", type=int, default=3,
                         help="benchmark save/load every N floors of progress")
    parser.add_argument("--save-dir", default="/private/tmp/claude-501/-Users-kylelau519-Programming-Vakuu/51b26a12-6771-434f-99bb-9b02d640b324/scratchpad/save_bench")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ensure_setup()

    print(f"Playing {args.character}/{args.seed} (ascension {args.ascension}) up to floor "
          f"{args.max_floor}, benchmarking every {args.checkpoint_every_floors} floors...\n")

    main_session = Sts2Session()
    decision = main_session.start(args.character, args.seed, args.ascension)
    last_checkpointed_floor = -1
    pending_purpose = None  # threaded through choose_action -- see its docstring
    results = []

    try:
        while decision.decision != "game_over":
            floor = getattr(getattr(decision, "context", None), "floor", None)
            if (floor is not None and floor >= last_checkpointed_floor + args.checkpoint_every_floors
                    and floor <= args.max_floor):
                last_checkpointed_floor = floor
                n_actions = len(main_session.action_log)
                print(f"--- floor {floor} ({n_actions} actions so far) ---")

                save_path = os.path.join(args.save_dir, f"floor{floor}.json")
                t0 = time.perf_counter()
                save_result = _write_continue_save(main_session, save_path)
                t_save = time.perf_counter() - t0
                save_size = save_result.get("size", "?")
                print(f"  write_continue_save: {t_save*1000:.1f} ms (save size {save_size} bytes, "
                      f"room_type={save_result.get('room_type')})")

                # Fresh subprocess: time load_save alone (subprocess spawn timed separately).
                t0 = time.perf_counter()
                load_session = Sts2Session()
                t_spawn = time.perf_counter() - t0
                t0 = time.perf_counter()
                load_raw = _load_save(load_session, save_path)
                t_load = time.perf_counter() - t0
                try:
                    load_decision = _to_decision(load_raw)
                    print(f"  load_save: {t_load*1000:.1f} ms (subprocess spawn {t_spawn*1000:.1f} ms) "
                          f"-> decision={load_decision.decision}")
                except Sts2ActionError as e:
                    print(f"  load_save: {t_load*1000:.1f} ms -> ERROR: {e}")
                load_session.close()

                # Fresh subprocess: time a full open_from_checkpoint replay of the same actions.
                t0 = time.perf_counter()
                replay_session = Sts2Session()
                t_spawn2 = time.perf_counter() - t0
                actions_so_far = _clean_actions(main_session.action_log)
                t0 = time.perf_counter()
                try:
                    replay_session.open_from_checkpoint(
                        args.character, args.seed, args.ascension, actions_so_far
                    )
                    t_replay = time.perf_counter() - t0
                    print(f"  open_from_checkpoint ({len(actions_so_far)} actions): "
                          f"{t_replay*1000:.1f} ms (subprocess spawn {t_spawn2*1000:.1f} ms)")
                except Sts2ActionError as e:
                    t_replay = time.perf_counter() - t0
                    print(f"  open_from_checkpoint FAILED after {t_replay*1000:.1f} ms: {e}")
                    t_replay = None
                replay_session.close()

                results.append({
                    "floor": floor, "n_actions": n_actions,
                    "save_ms": t_save * 1000, "load_ms": t_load * 1000,
                    "replay_ms": t_replay * 1000 if t_replay is not None else None,
                })
                print()

            name, kwargs, policy, pending_purpose = choose_action(
                decision, main_session, pending_purpose
            )
            decision = main_session.step(name, **kwargs)
    except Sts2ActionError as e:
        print(f"main line errored: {e}")
    finally:
        main_session.close()

    print("=== Summary ===")
    print(f"{'floor':>6} {'actions':>8} {'save (ms)':>10} {'load (ms)':>10} {'replay (ms)':>12}")
    for r in results:
        replay_str = f"{r['replay_ms']:.1f}" if r["replay_ms"] is not None else "FAILED"
        print(f"{r['floor']:>6} {r['n_actions']:>8} {r['save_ms']:>10.1f} {r['load_ms']:>10.1f} "
              f"{replay_str:>12}")


if __name__ == "__main__":
    main()
