"""Card-reward meta-decision branching, generating training data for `choices_paradox` (see
agents/choices_paradox/model/README.md's Design section). Every eligible
card_reward decision along the main line branches into one chain per legal option, each played
forward to natural death (a real loss, not a fixed horizon) using the SAME turn-branch-and-adopt
combat quality the main line itself gets -- not a single noisy rollout. card_select's remove/
upgrade purposes still resolve via choices_paradox/policy.py's existing heuristic, unbranched,
until Phase 2.

Shares its worker pool and combat turn-branching with facility_claws.py rather than duplicating
either -- the main line here still plays combat exactly like facility_claws.py's main line does
(reuses facility_claws._resolve_combat unchanged), and each option's own evaluation chain reuses
the SAME _resolve_combat function again for its own combats, dispatching BranchJob turn-branches
to the shared branch_worker_loop exactly the way the main line does. Unlike Phase 1, there is no
separate job type dispatched to the branch-worker pool for meta-decisions at all -- an option's
whole multi-combat chain runs inline in the main worker process (see _evaluate_option_chain).

Options run CONCURRENTLY, each on its own thread (_resolve_meta_decision), because they are
provably independent once the divergence happens: each chain forks its own Sts2Session (its own
OS subprocess, see env.py) from the same `prefix`, never touches the live main-line session, and
writes only to files keyed by its own option_index, so nothing about one option's chain can
observe or corrupt another's. The one thing that DID serialize them originally was
_resolve_combat's own _wait_for_turn_result, which needs a private results_queue with exactly one
waiter in flight -- sharing the issuing main worker's single queue across concurrent chains would
mean one chain's waiter could steal (and silently drop, per its "not in pending" filter) a result
meant for a sibling chain's pending set. Fixed by moving queue selection out of a fixed
main_worker_index -> queue table and into the job itself (branch_worker_loop now reports back to
whatever queue rode along with the job -- see facility_common.py's docstring there): each option
chain here is handed its own fresh Queue() (from the Facility's shared multiprocessing.Manager(),
passed down as queue_manager -- NOT a plain mp.Queue(), which can't be shipped through job_queue
to an already-running branch worker on spawn-based multiprocessing, and NOT a Manager built
locally either, since a main worker process is daemonic and daemonic processes can't start their
own child processes -- see main_worker_loop's queue_manager docstring for both) instead of the
shared one, so concurrent chains never share a waiter. Threads (not processes) are enough despite
the GIL because each chain is I/O-bound -- blocked on its own subprocess's pipes or on
Queue.get() waiting for branch workers, not on local CPU work -- the same reason the
branch-worker pool itself buys real wall-clock parallelism from separate OS processes rather than
from Python-level concurrency.

Design (Change of strategy from Phase 1's M-replicate, fixed-horizon-then-bootstrap scheme):
- One chain per legal option, no replicate averaging (M=1, by design -- see the README's
  "Original meaning of replicate" discussion; the noise that used to justify M replicates on a raw
  single-turn rollout is largely absorbed by combat's own turn-branch-and-adopt quality now that
  every combat inside a chain gets that treatment too, not a single Dirichlet-sampled turn).
- Each option-branch commits its card, then plays forward using the ordinary self-play policy
  (Golden Compass for map_select, Claws' Dirichlet-noised policy for combat, the existing
  choices_paradox tier-list heuristic for anything else) -- nothing else is allowed to diverge
  between sibling option-branches, so any difference in outcome is attributable to the one choice
  being compared, same principle Phase 1 already established.
- A native checkpoint is taken before every combat entry along the chain (mirroring
  main_worker_loop's own select_map_node checkpointing), enabling BranchJob's existing fast-load
  path for each of that chain's own combats. Confirmed safe to chain repeatedly -- not just
  faithful for one hop off a pristine session -- by
  debug_verify_chained_checkpoint_fidelity.py (13 hops across 4 independent seeds, zero drift;
  act-boundary crossing not yet directly exercised). The divergence itself is reconstructed
  faster too, but NOT via a fresh checkpoint taken at the pending card_reward decision itself --
  confirmed by hand that doesn't work: RunSimulator.SaveCheckpoint rolls back to a pre-room map
  state whenever the current room isn't a MapRoom (TryRollbackSerializedSaveToPreRoom, in
  sts2-cli's RunSimulator.SaveLoad.cs), so a checkpoint taken mid-card_reward reloads PAST it,
  at map_select, with no pending-reward signal left to recover -- the save format has no
  representation of "mid-room"/"room just finished" state outside MapRoom at all, so there is
  nothing to poll for. Instead, every option-chain reuses the checkpoint main_worker_loop already
  took right before the combat that produced this card_reward (last_combat_checkpoint_path, see
  its docstring), and replays just that ONE combat's own tail of actions via
  load_and_resume_combat to land back on the reward -- same fast-path-vs-full-replay trick
  facility_claws.py's own combat-internal turns already rely on, and the only way this checkpoint
  format supports resuming "mid-room" state at all. Replay cost (of that one combat, not the
  whole run) is now paid once per option per divergence rather than growing with total run depth
  the way a full open_from_checkpoint(prefix) replay would -- and since every option now runs
  concurrently, that's N replays in flight at once per divergence, so keeping each one cheap
  matters more than it used to.
- No fixed horizon, no TD-bootstrap off Claws's value head -- a chain plays until it actually
  loses. Value formula: _WIN_BONUS, plus that combat's own _combat_value (the same HP-fraction/
  boss-shaped quality-of-win term facility_common.py already uses elsewhere), for every combat WON
  along the chain, summed. _WIN_BONUS started at 1.0 and was dropped to 0.5 after the first live
  smoke test: at 1.0, "how many combats did this chain survive" (an integer count) dominated
  "how well did this pick's presence shape each fight" (_combat_value, typically ~0.1-0.5) --
  three genuinely different card picks landed within ~0.2 of each other while skipping landed
  ~1.1 lower, i.e. the signal that's actually supposed to differentiate options was getting
  drowned out by combat-count noise. Retune again if trained output still looks survival-count-
  dominated rather than pick-quality-dominated. No terminal -1 for the eventual loss -- every
  chain loses eventually by
  construction, so a flat penalty would just be a constant materialize.py's own group-baseline
  subtraction (_group_baseline) already cancels out; a chain that dies after 2 combats naturally
  scores lower than one that dies after 8 without it. Boss/normal combats are weighted equally for
  now (no extra credit for a boss clear) -- an accepted v1 simplification, not a considered
  decision to keep them equal forever.
- No recursive meta-branching for any card_reward a chain encounters past its own initial commit
  -- resolved by a single self-play pick like any other decision, same as main_worker_loop's own
  behavior whenever meta_resolver isn't invoked. Combinatorial cost is exactly why (options^depth
  branching), and this was a deliberate choice, not a placeholder.
- The trial that actually continues to the next divergence does NOT adopt whichever option scored
  best. It rewinds to the pre-divergence state (the live session never left it -- every option's
  chain runs on its OWN forked session, never grafted onto the main line) and commits via
  ChoicesParadoxNet's own softmax-plus-Dirichlet-noise prediction instead (see
  _choose_meta_option_claws) -- keeps the real trial's trajectory on-policy rather than either
  blind-random or coupled to whichever option an under-trained net happened to explore best this
  round, same "own predict + Dirichlet noise, to avoid a blind spot" reasoning combat's own
  adoption already follows.

Output format is deliberately unchanged from Phase 1 (role="branch", branch_kind="meta", one
value attached to the commit action via recorder.mark_step()/set_value()) -- one file per option
instead of per (option, replicate) is exactly the M=1 degenerate case _group_baseline's own "grand
mean equals per-option-mean-of-equal-size-groups" math already handles, so materialize.py needs no
changes to consume this scheme's output.
"""
from __future__ import annotations

import concurrent.futures
import functools
import importlib.util
import os
import sys
from pathlib import Path

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from env import Sts2Session

from checkpoint import BranchJob
from recorder import CompactRecorder
from policy import choose_action
from facility_common import (
    CombatOutcome, Facility, FacilityConfig, _combat_value, add_common_cli_args,
    branch_worker_loop, load_resume, main_worker_loop,
)
from facility_claws import _resolve_combat, _run_branch_job

# Loaded by file path (not sys.path + `import model`) for the same reason policy.py loads
# choices_paradox/policy.py and claws/model's ClawsNet this way: this package's model.py lives
# under a sibling package with no __init__.py-based package structure to import through
# normally, and doing it by path avoids any risk of colliding with another module also named
# `model` elsewhere on sys.path (e.g. claws/model/model.py, if that ever gets imported in the
# same process -- policy.py's get_claws_net can load ClawsNet in this very process).
_model_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "model" / "model.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_model", _model_path)
assert _spec is not None and _spec.loader is not None
_cp_model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cp_model  # dataclasses' own class-processing looks itself up via
# sys.modules[cls.__module__] during exec_module -- module_from_spec alone doesn't register it
_spec.loader.exec_module(_cp_model)
legal_options = _cp_model.legal_options

_choices_paradox_policy_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "policy.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_policy_module", _choices_paradox_policy_path)
assert _spec is not None and _spec.loader is not None
_choices_paradox_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_choices_paradox_policy)

# Same Dirichlet convention as policy.py's own _choose_combat_action_claws (root noise mixed into
# the softmax before sampling, concentration adapted to how many options are on offer) -- kept as
# a local copy rather than importing policy.py's underscore-prefixed constants across modules.
_DIRICHLET_EPS = 0.15
_DIRICHLET_ALPHA_SCALE = 10.0

# Flat per-combat-won contribution to an option-chain's accumulated value, on top of that combat's
# own _combat_value -- see this module's docstring for why 0.5 (not the original 1.0): at 1.0,
# raw survival depth swamped the actual pick-quality signal _combat_value is meant to carry.
_WIN_BONUS = 0.5

_CHOICES_PARADOX_CHECKPOINT = os.environ.get("CHOICES_PARADOX_CHECKPOINT")
_choices_paradox_net = None  # lazy-loaded once per main-worker process, on first divergence


def _load_choices_paradox_net():
    global _choices_paradox_net
    if _choices_paradox_net is not None or not _CHOICES_PARADOX_CHECKPOINT:
        return _choices_paradox_net
    if not os.path.exists(_CHOICES_PARADOX_CHECKPOINT):
        return None  # round 1, or CHOICES_PARADOX_CHECKPOINT unset by training_choices.sh yet
    _choices_paradox_net = _cp_model.ChoicesParadoxNet.from_checkpoint(_CHOICES_PARADOX_CHECKPOINT)
    return _choices_paradox_net


def _choose_meta_option_claws(decision, decision_kind: str, options: list[tuple[str, dict]]) -> int:
    """Picks which option the REAL trial actually commits to -- ChoicesParadoxNet's own softmax
    over `options`, mixed with root Dirichlet noise, sampled (not argmax). Deliberately NOT the
    best-scoring option from this divergence's own chain evaluations (see this module's docstring)
    -- falls back to the existing tier-list heuristic (via choose_action, ignoring the chain
    values entirely) when no checkpoint is loaded yet, same "noisy but harmless" round-1 fallback
    Claws's own combat picker already follows."""
    import torch

    net = _load_choices_paradox_net()
    if net is None:
        # Round 1 (no checkpoint yet) -- the same tier-list heuristic policy.py's own
        # _choose_generic_action delegates to for card_reward, called directly here rather than
        # through choose_action (which would need a live session this function doesn't have one
        # to spare -- decision.cards/.can_skip/.player are all this heuristic actually reads).
        index = _choices_paradox_policy.choose_card_reward(
            decision.cards, decision.can_skip, decision.player.deck_size or 0
        )
        for i, (opt_name, opt_args) in enumerate(options):
            if index is None:
                if opt_name == "skip_card_reward":
                    return i
            elif opt_name == "select_card_reward" and opt_args.get("card_index") == index:
                return i
        return 0  # defensive -- the heuristic's pick should always be one of `options`

    _, logits, _ = net.forward(decision, decision_kind)
    probs = torch.softmax(logits, dim=-1)
    alpha = _DIRICHLET_ALPHA_SCALE / len(options)
    noise = torch.distributions.Dirichlet(torch.full((len(options),), alpha)).sample()
    probs = (1 - _DIRICHLET_EPS) * probs + _DIRICHLET_EPS * noise
    return int(torch.multinomial(probs, 1).item())


def _evaluate_option_chain(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, character: str, seed: str,
    ascension: int, main_id: str, decision_kind: str, divergence_index: int, option_index: int,
    prefix: list[dict], last_combat_checkpoint_path: str | None, last_combat_entry_index: int | None,
    commit_name: str, commit_args: dict, stop_event,
) -> float | None:
    """Plays one card_reward option all the way to natural death, combat by combat, reusing
    facility_claws._resolve_combat for each combat's own turn-branch-and-adopt quality (the SAME
    shared branch-worker pool the main line uses) rather than a single noisy rollout. Checkpoints
    before every combat entry it reaches -- each new checkpoint descends from wherever this
    option's own reconstructed continuation currently is, confirmed safe to chain by
    debug_verify_chained_checkpoint_fidelity.py, not just faithful for one hop off a pristine
    session. Terminates on the first loss; returns the accumulated value (see this module's
    docstring for the formula), or None if this chain errored before ever reaching a terminal
    state.

    Runs on its OWN forked Sts2Session -- never touches or is grafted onto the live main-line
    session the caller holds, mirroring Phase 1's own "no replicate's downstream play is ever
    grafted onto the main line" invariant and sidestepping _resolve_combat's documented
    load_save-into-fresh-process RNG desync class by the same construction. Reconstructed via
    load_and_resume_combat(last_combat_checkpoint_path, ...) -- replaying just the tail of actions
    for the ONE combat that produced this card_reward decision -- rather than open_from_checkpoint
    replaying the entire run from StartRun, or a fresh checkpoint taken directly at the pending
    card_reward decision (confirmed NOT to work: RunSimulator's save/load_save doesn't preserve a
    pending decision screen -- see main_worker_loop's last_combat_checkpoint_path docstring for
    both). Falls back to the full open_from_checkpoint(prefix) replay when no combat checkpoint
    was available (last_combat_checkpoint_path is None -- vanishingly rare, see that same
    docstring). Either way, the reconstructed session has no memory of anything before the
    checkpoint/replay point, so `prefix` is spliced back onto session.action_log afterward --
    needed purely so CompactRecorder's start_step=len(prefix) indexing (set below) lines up with
    the action_log this chain's own recorder.save() is eventually handed, same reason
    facility_claws.py's own combat-checkpoint fast path re-prepends its pre-combat prefix after
    load_and_resume_combat.

    No recursive meta-branching for any card_reward this chain encounters past its own initial
    commit -- resolved via a single choose_action pick, same as main_worker_loop's own behavior
    whenever meta_resolver isn't invoked."""
    recorder = CompactRecorder({
        "character": character, "seed": seed, "ascension": ascension, "main_id": main_id,
        "role": "branch", "branch_kind": "meta", "decision_kind": decision_kind,
        "divergence_index": divergence_index, "option_index": option_index,
        "prefix_len": len(prefix),
    }, start_step=len(prefix))
    # Qualifies main_id for anything downstream of THIS chain's own combats -- specifically the
    # BranchJob/branch_id namespace _resolve_combat's shared turn-branching machinery builds (see
    # checkpoint.py's docstring: branch_id relies on combat_index being monotonic and never reset
    # for the whole trial to stay collision-free, which was true in facility_claws.py's original
    # single-main-line design). Each option-chain here resets its OWN local combat_index to 0 and
    # multiple chains for the same divergence run concurrently (see this module's docstring) --
    # with the bare main_id, two sibling chains both reaching their own combat_index=1 at the same
    # act/floor (guaranteed right after a shared divergence) mint IDENTICAL branch_ids, and
    # _run_branch_job's `f"{cfg.raw_dir}/{job.branch_id}.json"` write silently clobbers one
    # chain's turn-branch data with another's. Mirrors the divergence_index/option_index already
    # folded into this function's own select_map_node checkpoint names below and its final
    # branch_id at the bottom of this function.
    branch_main_id = f"{main_id}_m{divergence_index:03d}_o{option_index:02d}"
    session = Sts2Session(logger=recorder)
    checkpoint_dir = os.path.join(cfg.raw_dir, "checkpoints")
    checkpoint_paths: list[str] = []
    combat_checkpoint_path = None
    total_value = 0.0
    pending_purpose = None
    outcome = None
    try:
        try:
            if last_combat_checkpoint_path is not None and last_combat_entry_index is not None:
                entry_action = prefix[last_combat_entry_index]
                in_combat_actions = prefix[last_combat_entry_index + 1:]
                session.load_and_resume_combat(
                    last_combat_checkpoint_path, entry_action, in_combat_actions
                )
            else:
                session.open_from_checkpoint(character, seed, ascension, prefix)
            session.action_log = list(prefix)
            commit_step = recorder.mark_step()
            decision = session.step(commit_name, **commit_args)
            combat_index = 0

            while decision.decision != "game_over":
                if decision.decision == "combat_play":
                    combat_index += 1
                    session, decision, combat_outcome = _resolve_combat(
                        job_queue, results_queue, main_worker_index, cfg, recorder, session,
                        decision, character, seed, branch_main_id, combat_index, stop_event,
                        combat_checkpoint_path,
                    )
                    combat_checkpoint_path = None
                    if combat_outcome is not None:  # "lost" or "error" -- this chain ends here
                        outcome = combat_outcome
                        break
                    total_value += _WIN_BONUS + _combat_value(
                        CombatOutcome(decision.player.hp, decision.player.max_hp)
                    )
                    continue

                name, kwargs, policy, pending_purpose = choose_action(decision, session, pending_purpose)
                if policy is not None:
                    recorder.attach_policy(policy)
                if name == "select_map_node":
                    # Unique per (trial, divergence, option, combat) -- never reused, same
                    # straggler/overwrite-race reasoning as main_worker_loop's own checkpoint
                    # naming (see checkpoint.py's docstring): this chain runs sequentially with
                    # its own combats, but a straggler branch worker from a PRIOR combat in this
                    # same chain could still be reading an older file after this one moves on.
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"{branch_main_id}_c{combat_index + 1:03d}.json",
                    )
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    session.save(checkpoint_path)
                    checkpoint_paths.append(checkpoint_path)
                decision = session.step(name, **kwargs)
                if name == "select_map_node" and decision.decision == "combat_play":
                    combat_checkpoint_path = checkpoint_path

            if outcome is None:
                outcome = "won" if decision.victory else "lost"  # reached game_over directly
                # (e.g. beat the final boss) rather than via a _resolve_combat "lost" -- rare but
                # not impossible, and outcome=="won" here still correctly skips set_value below
                # since total_value has already accumulated every combat this chain actually won.
        except Exception as e:
            recorder.log_error(f"{type(e).__name__}: {e}")
            outcome = "error"

        if outcome != "error":
            recorder.set_value(commit_step, total_value)
        try:
            recorder.save(f"{cfg.raw_dir}/{branch_main_id}.json", session.action_log, outcome=outcome)
        except Exception as e:
            recorder.log_error(f"save failed: {type(e).__name__}: {e}")
    finally:
        session.close()
        # Same "safe to reclaim once this chain concludes" reasoning as main_worker_loop's own
        # trial-end checkpoint cleanup -- any straggler branch worker still reading one of these
        # is already past its own _wait_for_turn_result timeout, so its result would be discarded
        # as stale anyway.
        for path in checkpoint_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    return None if outcome == "error" else total_value


def _resolve_meta_decision(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, recorder: CompactRecorder,
    session, decision, character: str, seed: str, main_id: str, decision_kind: str,
    divergence_index: int, stop_event, queue_manager,
    last_combat_checkpoint_path: str | None, last_combat_entry_index: int | None,
):
    """Evaluates every legal option of `decision` (a live, pydantic-typed card_reward decision on
    the main Sts2Session) by playing each to natural death via _evaluate_option_chain, all options
    CONCURRENTLY (one thread each -- see this module's docstring for why that's safe: each chain
    forks its own Sts2Session/subprocess and gets its own private results_queue, so nothing shared
    can race), then commits for real via ChoicesParadoxNet's own prediction
    (_choose_meta_option_claws) -- not necessarily whichever option scored best. Falls back to the
    plain heuristic pick (what choose_action would have done anyway) if fewer than 2 options exist
    to compare, or if every chain errored -- forward progress on the main line matters more than a
    missed training signal for this one decision.

    last_combat_checkpoint_path/last_combat_entry_index (from main_worker_loop, see its docstring):
    the native checkpoint for the combat that produced this card_reward, and that combat's entry
    action's position in `prefix` below. Passed straight through to every option-chain, which
    reconstructs its own starting state by replaying just that one combat's tail of actions on top
    of it (_evaluate_option_chain's own load_and_resume_combat call) -- deliberately NOT a fresh
    checkpoint taken here at the pending card_reward decision itself: confirmed by hand that
    doesn't work (RunSimulator.SaveCheckpoint rolls back to a pre-room map state whenever the
    current room isn't a MapRoom -- see TryRollbackSerializedSaveToPreRoom in sts2-cli's
    RunSimulator.SaveLoad.cs -- so a checkpoint taken mid-card_reward reloads past it, at
    map_select, with no pending-reward signal left to recover; there's nothing to poll because the
    save format has no representation of "mid-room"/"room just finished" state outside MapRoom at
    all). Reusing the pre-combat checkpoint main_worker_loop already takes is the same trick
    facility_claws.py's own combat-internal fast path relies on for the exact same reason."""
    options = legal_options(decision, decision_kind)
    if len(options) < 2:
        if not options:
            return decision  # defensive -- card_reward always offers skip_card_reward at least
        name, args = options[0]
        return session.step(name, **args)

    prefix = list(session.action_log)
    # One thread and one fresh manager-backed Queue() per option -- NOT this main worker's own
    # shared results_queue, which stays reserved for the main line's own (unbranched,
    # single-waiter) combat resolution. A dedicated queue per chain is what makes concurrent
    # chains safe: see branch_worker_loop's docstring on why sharing one would silently drop
    # results. queue_manager.Queue(), not a plain mp.Queue() -- a raw Queue created this late
    # can't be shipped to the already-running branch pool (see branch_worker_loop's docstring),
    # and this main worker process is itself daemonic so it can't build its OWN Manager to work
    # around that (see main_worker_loop's queue_manager docstring) -- queue_manager is the
    # Facility's own manager, built in its non-daemonic process and handed down at this worker's
    # spawn, reused here across every divergence this worker resolves.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(options)) as pool:
        futures = [
            pool.submit(
                _evaluate_option_chain,
                job_queue, queue_manager.Queue(), main_worker_index, cfg, character, seed,
                cfg.ascension, main_id, decision_kind, divergence_index, option_index, prefix,
                last_combat_checkpoint_path, last_combat_entry_index, name, args, stop_event,
            )
            for option_index, (name, args) in enumerate(options)
        ]
        values: list[float | None] = [f.result() for f in futures]

    # choices_paradox/model/materialize.py's materialize_main_trial pulls the main line's own
    # adopted meta-decision back out as a value-regression example, the same way facility_claws's
    # _absorb_actions attaches an adopted turn-branch's policy/value onto the main line -- it only
    # checks "policy" in entry as a presence marker (never reads its per-option content), so a None
    # value for an errored option's chain is harmless here.
    recorder.attach_policy([
        {"action": a, "args": kw, "value": v} for (a, kw), v in zip(options, values)
    ])

    if not any(v is not None for v in values):
        recorder.log_error(
            f"meta-decision {divergence_index} ({decision_kind}): every option's chain errored "
            f"-- falling back to the heuristic pick"
        )
        name, args, policy, _ = choose_action(decision, session)
        if policy is not None:
            recorder.attach_policy(policy)
        return session.step(name, **args)

    adopted_index = _choose_meta_option_claws(decision, decision_kind, options)
    adopted_value = values[adopted_index]
    if adopted_value is not None:
        recorder.attach_value(adopted_value)
    name, args = options[adopted_index]
    return session.step(name, **args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Whispering Earring self-play facility (card_reward meta-branching, "
                     "for choices_paradox) -- combat is still turn-branched exactly like "
                     "facility_claws.py's main line, sharing the same branch-worker pool"
    )
    add_common_cli_args(parser)
    parser.add_argument("--branches-per-turn", type=int, default=3,
                         help="combat turn-branching, same knob and default as facility_claws.py "
                              "-- both the main line and every option-chain's own combats use it")
    parser.add_argument("--meta-branches-per-trial", type=int, default=None,
                         help="cap on how many card_reward decisions per trial actually spawn "
                              "chains (default: unlimited -- branch every one, matching combat's "
                              "own always-branch default)")
    args = parser.parse_args()

    resume = load_resume(args.resume_branch_file) if args.resume_branch_file else None

    facility = Facility(
        FacilityConfig(
            characters=args.characters,
            n_branch_workers=args.branch_workers,
            branches_per_turn=args.branches_per_turn,
            n_main_workers=args.main_workers,
            n_main_trials=args.trials,
            raw_dir=args.raw_dir,
            verbose=args.verbose,
            meta_branches_per_trial=args.meta_branches_per_trial,
        ),
        main_worker_fn=functools.partial(
            main_worker_loop, combat_resolver=_resolve_combat, meta_resolver=_resolve_meta_decision,
        ),
        branch_worker_fn=functools.partial(
            branch_worker_loop, handlers={BranchJob: _run_branch_job},
        ),
        resume=resume,
    )
    facility.run()
