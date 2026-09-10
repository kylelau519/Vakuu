"""Single-line (unbranched) rollout comparison for Claws.

facility_claws.py's main line adopts one of `--branches-per-turn` parallel attempts every turn,
so a loss there doesn't mean the policy would actually have lost unassisted. This script plays a
handful of FIXED seeds straight through instead -- no branching, no do-overs -- once per
checkpoint, so the comparison is a genuine single-attempt result. Meant to be run at the end of
each Claws training round against three checkpoints: this round's freshly trained net ("current"),
last round's net before this round's training touched it ("previous"), and a frozen random-init
net ("untrained") -- if "current" isn't clearly ahead of "previous", and "previous"/"current"
aren't clearly ahead of "untrained", something's wrong with the round, not just noisy variance.

policy.py reads CLAWS_CHECKPOINT (and loads the net) once at import time and caches it for the
rest of the process, so comparing multiple checkpoints can't happen in one process -- this script
is its own driver: the no-args/--checkpoints invocation spawns itself once per (checkpoint, seed)
pair as a subprocess (each with its own CLAWS_CHECKPOINT in the environment), all launched at once
rather than one at a time -- a single line has no branching to hide the cost of the plain
Sts2Session replay every combat needs (see _resolve_combat_single_line's docstring), so running
the whole matrix serially would take checkpoints x seeds times as long as one rollout; this way it
takes about as long as the single slowest one. CHOICES_PARADOX_CHECKPOINT/GOLDEN_COMPASS_CHECKPOINT
are left alone (inherited from the parent environment) so Claws is the only thing varying between
columns. Each subprocess's engine output goes to its own log file under a temp dir (worth keeping
around on failure, so it's only cleaned up when everything succeeds) rather than the shared
terminal -- concurrent processes' raw [SIM]/[INFO] lines would otherwise interleave into a mess.

Usage:
    python eval_claws.py --checkpoints current=claws.pt previous=claws_prev.pt untrained=claws_untrained.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

# Fixed (not randomly drawn) so every round compares the same seeds -- see module docstring.
DEFAULT_SEEDS = ["falcon-100001", "ember-100002", "wolf-100003"]


def _resolve_combat_single_line(
    job_queue_priority, results_queue, main_worker_index, cfg, recorder, session, decision,
    character, seed, main_id, combat_index, stop_event, checkpoint_path,
):
    """combat_resolver for _play_trial: plays the whole combat straight through on a throwaway
    Sts2BufferSession (no branch workers, no per-turn adoption -- just one line), then resyncs the
    main line's own plain Sts2Session onto the result by replaying its action log, exactly like
    facility_claws._resolve_turn does after adopting a branch -- just once for the whole combat
    instead of once per turn, since there's only ever one candidate here.

    _play_turn_only is written against BufferDecision (bl.*/decode_token), same as
    facility_claws._run_branch_job's branch sessions -- the main line's plain pydantic Decision
    never flows through it directly. is_last_of_act can't be recovered from a BufferDecision
    (RunContext has no room-type field -- see buffer_layout.py's run_context() comment) but it
    only feeds _play_turn_only's per-turn value-shaping scalar, never which action gets chosen or
    the win/loss outcome, so hardcoding False here doesn't affect anything this script measures."""
    import buffer_layout as bl
    from buffer_env import decode_token, Sts2BufferSession
    from env import Sts2Session
    from facility_common import _play_turn_only, _clean_actions

    prefix = list(session.action_log)
    combat_start_index = len(prefix) - 1
    buffer_session = Sts2BufferSession(logger=recorder)
    try:
        if checkpoint_path is not None:
            entry_action = prefix[combat_start_index]
            in_combat_actions = prefix[combat_start_index + 1:]
            buf_decision = buffer_session.load_and_resume_combat(
                checkpoint_path, entry_action, in_combat_actions
            )
            buffer_session.action_log = prefix[:combat_start_index] + buffer_session.action_log
        else:
            buf_decision = buffer_session.open_from_checkpoint(
                character, seed, cfg.ascension, prefix
            )

        run_context = bl.run_context(buf_decision.arr)
        act, floor = run_context["act_index"], run_context["floor"]
        enemy_ids = frozenset(
            decode_token(bl.enemy_token(slot)) for _, slot in bl.enemies(buf_decision.arr)
        )
        enemy_max_hp = sum(bl.enemy_max_hp(slot) for _, slot in bl.enemies(buf_decision.arr))
        combat_hp_before, combat_max_hp = buf_decision.player_hp, buf_decision.player_max_hp

        outcome = "continue"
        while outcome == "continue":
            outcome, buf_decision = _play_turn_only(
                buffer_session, buf_decision, combat_hp_before, combat_max_hp, False,
                enemy_ids, enemy_max_hp, combat_dirichlet_noise=cfg.combat_dirichlet_noise,
            )
        recorder.record_combat(act, floor, combat_index, outcome)
        cleaned = _clean_actions(buffer_session.action_log)
    finally:
        buffer_session.close()

    if checkpoint_path is not None:
        entry_action = cleaned[combat_start_index]
        in_combat_actions = cleaned[combat_start_index + 1:]
        new_decision = session.load_and_resume_combat(
            checkpoint_path, entry_action, in_combat_actions
        )
        session.action_log = cleaned[:combat_start_index] + session.action_log
    else:
        # Rare (only the very first combat of a run lacks a checkpoint) -- matches
        # facility_claws._resolve_turn's own fallback for the same case.
        session.close()
        session = Sts2Session(logger=recorder)
        new_decision = session.open_from_checkpoint(character, seed, cfg.ascension, cleaned)
    return session, new_decision, (None if outcome == "won" else outcome)


def _run_worker(args) -> None:
    import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
    from env import Sts2Session
    from installation import ensure_setup
    from facility_common import FacilityConfig, _TrialState, _play_trial
    from recorder import CompactRecorder

    ensure_setup()
    cfg = FacilityConfig(
        characters=[args.character], ascension=args.ascension,
        combat_dirichlet_noise=not args.no_combat_dirichlet_noise,
    )
    results = []
    for seed in args.seeds:
        recorder = CompactRecorder({
            "character": args.character, "seed": seed, "ascension": args.ascension,
            "label": args.label, "role": "eval",
        })
        with tempfile.TemporaryDirectory(prefix="claws_eval_") as checkpoint_dir:
            state = _TrialState(
                session=Sts2Session(logger=recorder), checkpoint_dir=checkpoint_dir, meta_budget=0,
            )
            stop_event = threading.Event()
            outcome = None
            try:
                outcome, _decision = _play_trial(
                    cfg, None, None, None, None, 0, recorder, args.character, seed,
                    f"{args.label}_{seed}", None, stop_event, None,
                    _resolve_combat_single_line, None, state,
                )
            except Exception as e:
                recorder.log_error(f"{type(e).__name__}: {e}")
                outcome = "error"
            finally:
                state.session.close()
            data = recorder.save(
                os.path.join(checkpoint_dir, "trial.json"), state.session.action_log, outcome=outcome
            )

        combats = data.get("combats", [])
        last = combats[-1] if combats else None
        result = {
            "seed": seed,
            "outcome": outcome,
            "combats_won": sum(1 for c in combats if c["outcome"] == "won"),
            "combats_total": len(combats),
            "final_act": last["act"] if last else None,
            "final_floor": last["floor"] if last else None,
        }
        results.append(result)
        print(f"[{args.label}] seed={seed} -> {outcome} "
              f"(act {result['final_act']} floor {result['final_floor']}, "
              f"{result['combats_won']}/{result['combats_total']} combats won)", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(results, f)


def _print_comparison(label_results: dict[str, list[dict | None]], seeds: list[str]) -> None:
    col_w = 24
    print("\n=== Claws single-rollout comparison (unbranched, fixed seeds) ===")
    print(f"{'seed':<16}" + "".join(f"{label:>{col_w}}" for label in label_results))
    for i, seed in enumerate(seeds):
        row = f"{seed:<16}"
        for results in label_results.values():
            r = results[i]
            if r is None:
                cell = "FAILED (see log)"
            else:
                floor_str = (f"act{r['final_act']}f{r['final_floor']}"
                             if r["final_floor"] is not None else "no-combat")
                cell = f"{r['outcome']} {floor_str} {r['combats_won']}/{r['combats_total']}"
            row += f"{cell:>{col_w}}"
        print(row)
    print()
    for label, results in label_results.items():
        ok = [r for r in results if r is not None]
        wins = sum(1 for r in ok if r["outcome"] == "won")
        n_failed = len(results) - len(ok)
        if ok:
            avg_won = sum(r["combats_won"] for r in ok) / len(ok)
            suffix = f", {n_failed} failed" if n_failed else ""
            print(f"{label:<12} {wins}/{len(ok)} run(s) won, avg {avg_won:.1f} combats won/run{suffix}")
        else:
            print(f"{label:<12} all {n_failed} run(s) failed")
    print()


def _run_driver(args) -> None:
    checkpoints = dict(kv.split("=", 1) for kv in args.checkpoints)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmpdir = tempfile.mkdtemp(prefix="claws_eval_driver_")

    jobs = []  # (label, seed, proc, out_path, log_path, log_file)
    for label, path in checkpoints.items():
        env = dict(os.environ)
        env["CLAWS_CHECKPOINT"] = path
        for seed in args.seeds:
            tag = f"{label}__{seed}"
            out_path = os.path.join(tmpdir, f"{tag}.json")
            log_path = os.path.join(tmpdir, f"{tag}.log")
            cmd = [
                sys.executable, __file__, "--worker",
                "--label", label, "--character", args.character,
                "--ascension", str(args.ascension), "--out", out_path,
                "--seeds", seed,
            ]
            if args.no_combat_dirichlet_noise:
                cmd.append("--no-combat-dirichlet-noise")
            log_file = open(log_path, "w")
            proc = subprocess.Popen(cmd, env=env, cwd=script_dir, stdout=log_file, stderr=subprocess.STDOUT)
            jobs.append((label, seed, proc, out_path, log_path, log_file))

    print(f"--- Running {len(jobs)} single-line rollout(s) in parallel "
          f"({len(checkpoints)} checkpoint(s) x {len(args.seeds)} seed(s)) ---")

    seed_index = {seed: i for i, seed in enumerate(args.seeds)}
    label_results: dict[str, list[dict | None]] = {label: [None] * len(args.seeds) for label in checkpoints}
    failed_logs = []
    for label, seed, proc, out_path, log_path, log_file in jobs:
        ret = proc.wait()
        log_file.close()
        if ret != 0:
            failed_logs.append((label, seed, log_path))
            continue
        with open(out_path) as f:
            (result,) = json.load(f)
        label_results[label][seed_index[seed]] = result
        print(f"[{label}] seed={seed} -> {result['outcome']} "
              f"(act {result['final_act']} floor {result['final_floor']}, "
              f"{result['combats_won']}/{result['combats_total']} combats won)")

    if failed_logs:
        print("\nFailed rollout(s) -- logs kept for inspection:")
        for label, seed, log_path in failed_logs:
            print(f"  [{label}] seed={seed}: {log_path}")
    else:
        shutil.rmtree(tmpdir, ignore_errors=True)

    _print_comparison(label_results, args.seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoints", nargs="+", default=[],
                         help="driver mode: label=path pairs, e.g. "
                              "current=claws.pt previous=claws_prev.pt untrained=claws_untrained.pt")
    parser.add_argument("--label", default=None, help="worker mode only")
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--ascension", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out", default=None, help="worker mode only: where to write this checkpoint's results JSON")
    parser.add_argument("--no-combat-dirichlet-noise", action="store_true",
                         help="sample straight from each checkpoint's own softmax, with no root "
                              "Dirichlet noise mixed in -- for a comparison that isn't muddied by "
                              "per-run exploration noise (default: noise enabled, same as "
                              "self-play training)")
    args = parser.parse_args()

    if args.worker:
        _run_worker(args)
    else:
        if not args.checkpoints:
            raise SystemExit("--checkpoints label=path [label=path ...] is required")
        _run_driver(args)
