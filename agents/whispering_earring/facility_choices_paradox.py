"""Card-reward meta-decision branching, generating training data for `choices_paradox` (see
agents/choices_paradox/model/README.md's Design section). Every eligible card_reward decision
along the main line branches into one chain per legal option, each played forward to natural death
(a real loss, not a fixed horizon) -- not a single noisy rollout, but not the main line's own
turn-branch-and-adopt combat quality either (see below on why). card_select's remove/upgrade
purposes still resolve via choices_paradox/policy.py's existing heuristic, unbranched, until
Phase 2.

Each option-chain runs UNBRANCHED (no per-turn forking of its own combats -- see below) on its own
`Sts2BufferSession` (buffer_env.py -- in-process CLR calls via pythonnet) held open for the
chain's entire natural-death walk. Chains are dispatched as `MetaOptionJob`s (checkpoint.py) onto
the shared branch-worker pool -- the SAME pool facility_claws.py's combat-turn branching uses,
via the plain (low-priority) `job_queue` rather than combat's `job_queue_priority` -- and the main
line does NOT wait for them: `_resolve_meta_decision` commits to an option immediately
(`_choose_meta_option_claws`, never a function of the chains' own values -- see below) and moves
on, while every option's chain plays out asynchronously at whatever pace the pool can sustain. See
`_run_meta_option_job`'s docstring for the branch-worker-side mechanics, and main_worker_loop's
`pending_meta_jobs`/`checkpoint_refcount` for how a trial avoids deleting a checkpoint a
still-in-flight chain job needs.

This design point (whole chains as pool jobs) is DIFFERENT from, and doesn't reintroduce the
problems of, an earlier design that turn-branched every chain's OWN combat through the pool too
(sibling options concurrent, one BranchJob-style dispatch per combat turn per chain):

- Compute cost: facility_claws.py's `_resolve_turn` closes and respawns a real `dotnet run` OS
  subprocess on EVERY combat turn, replaying the whole combat's actions into it over JSON. The
  main line pays that once per turn, total; the old per-turn-branched design paid it once per turn
  PER CHAIN, and a single divergence spawns ~3-4 chains, each playing to natural death (so replay
  cost grows with how many actions a chain has already accumulated) -- measured contributor, not
  hypothetical: one option-chain two combats and ten turns in had already accumulated 123 actions
  to replay on every one of those resyncs. Dispatching one job per WHOLE CHAIN avoids this
  entirely: each `MetaOptionJob` holds its own `Sts2BufferSession` open, in-process on whichever
  branch worker runs it, for the chain's entire walk -- no subprocess spawn or resync at any point,
  same as before, just relocated to a branch-worker process instead of the main worker's own.
- Thread safety: the old per-turn-branched design's concurrency was safe only because each chain's
  Sts2Session was its own OS subprocess with its own results_queue -- provably independent.
  Sts2BufferSession has no such isolation WITHIN one process: every session in a process shares the
  one CoreCLR clr_bridge.load_clr() loads into it, and debug_repro_branch_divergence.py already
  caught RunSimulator's save/profile state racing across merely-concurrent *processes* sharing only
  an unlocked on-disk profile path -- threads sharing the same in-process CLR directly would be at
  least as exposed. Dispatching one MetaOptionJob per chain to the branch-worker pool sidesteps
  this too: each branch worker is its own OS process with its own independently-loaded CoreCLR
  (clr_bridge.py's `load_clr()` cache is a process-global, not shared across processes) -- the SAME
  isolation the pool already relies on for combat-turn BranchJobs, just applied to a coarser unit
  of work (one whole chain, not one turn).

The combat-quality trade this makes explicit: each chain's own play is exactly what
`choose_action`'s Dirichlet-noised policy picks (combat included, via Claws's own net over the
buffer protocol, the same path branch workers already use), on-policy but unbranched -- noisier
per chain than the main line's turn-branch-and-adopt quality. Accepted because the option
comparison this module trains on only needs to be unbiased on average, and chain volume should
average out per-chain noise the same way materialize.py's group-baseline subtraction already
discounts other per-chain variance -- trading combat-quality-per-chain for far more chains is the
better use of the same compute budget here (unlike the main line, whose own trajectory and Claws's
turn-branch training data both still want the higher-quality adopted play).

Design:
- One chain per legal option, no replicate averaging (M=1 by design -- see the README's "Original
  meaning of replicate"; the noise that used to justify M replicates on a raw single-turn rollout
  is instead absorbed by chain volume).
- Each option-branch commits its card, then plays forward using the ordinary self-play policy --
  nothing else is allowed to diverge between siblings, so any outcome difference is attributable
  to the one choice being compared. Enforced via a shared walk_seed (drawn once per divergence in
  _resolve_meta_decision, common to every sibling): reseeds Python's global `random` and torch's
  global RNG identically right after each sibling commits, so policy draws (map routing, Dirichlet-
  sampled cards, tie-breaks) are bit-identical for as long as game state hasn't actually diverged --
  "common random numbers," standard variance reduction for a paired counterfactual comparison. This
  can't and doesn't force identical downstream content (reward offers, monster picks) once real
  divergence begins -- that's the engine's own internal RNG drifting once siblings take different
  actions, and that drift IS the causal signal being measured. Reseeding the GLOBAL state (rather
  than threading an explicit Generator) is safe because each sibling's MetaOptionJob runs its own
  branch-worker OS process with its own independent Python/torch global state -- no OTHER chain's
  RNG to clobber, since there's nothing shared to clobber it in. (Within one such process, a
  branch worker only ever runs one job at a time, and _run_meta_option_job restores unpredictable
  global state before returning -- see its docstring -- so the NEXT job that process picks up,
  whether another chain or an unrelated combat BranchJob, never inherits a walk_seed-correlated
  stream either.)
- The divergence is reconstructed via the pre-combat native checkpoint main_worker_loop already
  took before the combat that produced this card_reward (last_combat_checkpoint_path), replaying
  just that one combat's tail of actions via load_and_resume_combat to land back on the reward --
  NOT a fresh checkpoint taken at the pending card_reward decision itself: confirmed by hand that
  doesn't work (RunSimulator.SaveCheckpoint rolls back to a pre-room map state whenever the current
  room isn't a MapRoom -- TryRollbackSerializedSaveToPreRoom, sts2-cli's RunSimulator.SaveLoad.cs --
  so a checkpoint taken mid-card_reward reloads PAST it, at map_select, with no pending-reward
  signal left; the save format has no representation of "mid-room" state outside MapRoom at all).
  This is the chain's ONLY reconstruction -- it never re-checkpoints itself again, just keeps
  stepping its live session forward to natural death.
- No fixed horizon, no TD-bootstrap off Claws's value head -- a chain plays until it actually
  loses. Value = _WIN_BONUS + that combat's own _combat_value (facility_common.py's HP-fraction/
  boss-shaped quality-of-win term), summed per combat won -- except is_last_of_act is always False
  for a chain's own combats: that flag comes from the main line's live pydantic
  `context.room_type` in the original design (see checkpoint.py's docstring on why the buffer
  protocol alone can't recover it), and a chain has no pydantic Decision to read that from. Accepted
  simplification: a chain-internal boss win scores with the ordinary HP-fraction formula instead of
  the boss-shaped one.
- _WIN_BONUS started at 1.0, dropped to 0.5 after the first live smoke test: at 1.0, combat-count
  survival dominated pick-quality (_combat_value, typically ~0.1-0.5) -- three genuinely different
  picks landed within ~0.2 of each other while skipping landed ~1.1 lower. Retune again if trained
  output still looks survival-count-dominated. No terminal -1 for the eventual loss -- every chain
  loses eventually by construction, so a flat penalty would just be a constant materialize.py's own
  group-baseline subtraction already cancels out.
- No recursive meta-branching for any card_reward a chain encounters past its own initial commit --
  resolved by a single self-play pick like any other decision. Combinatorial cost (options^depth) is
  why, and it's a deliberate choice, not a placeholder.
- The trial that continues to the next divergence does NOT adopt whichever option scored best --
  in fact it can't: since chains now run asynchronously on the branch-worker pool, no chain's value
  is even known yet at commit time. It rewinds to the pre-divergence state (the live session never
  left it -- each option's chain runs on its OWN forked session, now on its OWN branch-worker
  process) and commits via ChoicesParadoxNet's own softmax-plus-Dirichlet-noise prediction instead
  (_choose_meta_option_claws) -- keeps the real trial on-policy rather than coupled to whichever
  option an under-trained net happened to explore best this round.
- The main line's own recorder gets the adopted option's value attached OPPORTUNISTICALLY, not at
  commit time: `_resolve_meta_decision` returns `(decision, pending_meta_jobs)`, and
  main_worker_loop applies `recorder.set_value(commit_step, value)` whenever that specific job's
  result shows up on meta_results_queue (a non-blocking drain after every subsequent decision, plus
  one bounded drain at trial end -- see main_worker_loop's docstring). If the adopted chain's job
  hasn't reported back by the time the trial ends and its grace period elapses, that one step
  simply has no "value" in the saved JSON -- a soft loss of one training example, not an error;
  every option's own {branch_main_id}.json file (see below) is unaffected either way.

Output format is unchanged from Phase 1 (role="branch", branch_kind="meta", one value attached to
the commit action) -- one file per option instead of per (option, replicate) is exactly the M=1
degenerate case _group_baseline's math already handles, so materialize.py needs no changes.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import random
import sys
from pathlib import Path

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from buffer_env import Sts2BufferSession

from checkpoint import BranchJob, MetaOptionJob
from recorder import CompactRecorder
from policy import choose_action
from facility_common import (
    CombatOutcome, Facility, FacilityConfig, _combat_value, _play_turn_only,
    add_common_cli_args, branch_worker_loop, load_resume, main_worker_loop,
)
from facility_claws import _resolve_combat, _run_branch_job

# Loaded by file path (not sys.path + `import model`) for the same reason policy.py loads
# choices_paradox/policy.py and claws/model's ClawsNet this way: this package's model.py lives
# under a sibling package with no __init__.py-based package structure to import through
# normally, and doing it by path avoids colliding with another module also named `model`
# elsewhere on sys.path (e.g. claws/model/model.py).
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
# own _combat_value -- see this module's docstring for why 0.5, not the original 1.0.
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
    -- falls back to the existing tier-list heuristic (ignoring the chain values entirely) when no
    checkpoint is loaded yet, same "noisy but harmless" round-1 fallback Claws's own combat picker
    already follows."""
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


def _run_meta_option_job(cfg: FacilityConfig, job: MetaOptionJob, results_queue) -> None:
    """branch_worker_loop's MetaOptionJob handler -- plays one card_reward option all the way to
    natural death, combat by combat, on ONE in-process Sts2BufferSession held open for the chain's
    entire walk (see this module's docstring for why running whole chains as separate branch-worker
    processes doesn't reintroduce the per-turn subprocess cost or CLR thread-safety hazard an
    earlier, different pool-based design was reverted for).

    job.walk_seed: the SAME value for every sibling option of this divergence (drawn once by
    _resolve_meta_decision) -- reseeds Python's global `random` and torch's global RNG right after
    committing (see below), giving common-random-numbers variance reduction (see this module's
    docstring). Only removes noise from wherever siblings haven't actually diverged yet; once a
    chain's own state differs (different cards change what's drawn/playable), the streams desync
    from there, which is the causal signal being measured, not something to fight. Restored to
    unpredictable state before this function returns -- this is a persistent branch-worker process
    that goes on to run other, unrelated jobs (other divergences' chains, or combat BranchJobs)
    next, and leaving global random/torch state walk_seed-correlated would leak into whichever job
    runs here after this one.

    Runs on its own Sts2BufferSession -- never touches or is grafted onto the main-line session.
    Reconstructed via load_and_resume_combat(job.last_combat_checkpoint_path, ...) -- replaying
    just the tail of actions for the ONE combat that produced this card_reward -- rather than a
    full open_from_checkpoint(job.prefix) replay from StartRun (see this module's docstring on why
    a fresh checkpoint at the card_reward itself doesn't work). Falls back to the full replay when
    no combat checkpoint was available (job.last_combat_checkpoint_path is None -- vanishingly
    rare). Either way the reconstructed session has no memory of anything before the checkpoint/
    replay point, so job.prefix is spliced back onto session.action_log afterward, needed purely so
    CompactRecorder's start_step=len(prefix) indexing lines up with the action_log this chain's own
    recorder.save() is eventually handed. This is the chain's only reconstruction -- nothing
    re-checkpoints it again after this point.

    No recursive meta-branching for any card_reward this chain encounters past its own initial
    commit -- resolved via a single choose_action pick, same as main_worker_loop's own behavior
    whenever meta_resolver isn't invoked.

    Always reports back on results_queue, even on failure (value=None) -- main_worker_loop's
    pending_meta_jobs tracking would otherwise wait out the full trial-end drain timeout for this
    meta_id and never reclaim its checkpoint refcount."""
    import torch

    recorder = CompactRecorder({
        "character": job.character, "seed": job.seed, "ascension": job.ascension,
        "main_id": job.main_id, "role": "branch", "branch_kind": "meta",
        "decision_kind": job.decision_kind, "divergence_index": job.divergence_index,
        "option_index": job.option_index, "prefix_len": len(job.prefix),
    }, start_step=len(job.prefix))
    session = Sts2BufferSession(logger=recorder)
    total_value = 0.0
    pending_purpose = None
    outcome = None
    try:
        try:
            if job.last_combat_checkpoint_path is not None and job.last_combat_entry_index is not None:
                entry_action = job.prefix[job.last_combat_entry_index]
                in_combat_actions = job.prefix[job.last_combat_entry_index + 1:]
                decision = session.load_and_resume_combat(
                    job.last_combat_checkpoint_path, entry_action, in_combat_actions
                )
            else:
                decision = session.open_from_checkpoint(
                    job.character, job.seed, job.ascension, job.prefix
                )
            session.action_log = list(job.prefix)
            commit_step = recorder.mark_step()
            decision = session.step(job.commit_name, **job.commit_args)
            # Reseed AFTER committing (the commit itself makes no random.*/torch.* draws) so
            # every sibling option's own post-commit policy draws start from the identical
            # stream -- see this function's docstring on walk_seed.
            random.seed(job.walk_seed)
            torch.manual_seed(job.walk_seed)

            while decision.decision != "game_over":
                if decision.decision == "combat_play":
                    # Unbranched: one turn at a time on this same session via
                    # facility_common._play_turn_only, looped until the combat itself resolves.
                    # bypass_reward_on_win=False keeps the terminal card_reward real so it can be
                    # resolved via choose_action below (see _play_turn_only's own docstring).
                    combat_hp_before, combat_max_hp = decision.player_hp, decision.player_max_hp
                    turn_outcome = "continue"
                    while turn_outcome == "continue":
                        turn_outcome, decision = _play_turn_only(
                            session, decision, combat_hp_before, combat_max_hp,
                            bypass_reward_on_win=False,
                        )
                    if turn_outcome == "lost":
                        outcome = "lost"
                        break
                    total_value += _WIN_BONUS + _combat_value(
                        CombatOutcome(decision.player_hp, decision.player_max_hp)
                    )
                    continue

                # card_reward_dirichlet_noise=False: any card_reward this chain hits past its own
                # initial commit would otherwise pick up extra sampling variance on top of the
                # option-chain comparison this whole job exists to measure -- see choose_action's
                # docstring on why that's unwanted here even though combat's own noise (left on,
                # via _play_turn_only above) is accepted variance for a chain's play.
                name, kwargs, policy, pending_purpose = choose_action(
                    decision, session, pending_purpose, card_reward_dirichlet_noise=False,
                )
                if policy is not None:
                    recorder.attach_policy(policy)
                decision = session.step(name, **kwargs)

            if outcome is None:
                # Reached game_over directly (e.g. beat the final boss) rather than via a combat
                # "lost" -- rare but not impossible; total_value has already accumulated every
                # combat this chain actually won.
                outcome = "won" if decision.victory else "lost"
        except Exception as e:
            recorder.log_error(f"{type(e).__name__}: {e}")
            outcome = "error"

        if outcome != "error":
            recorder.set_value(commit_step, total_value)
        try:
            recorder.save(f"{cfg.raw_dir}/{job.meta_id}.json", session.action_log, outcome=outcome)
        except Exception as e:
            recorder.log_error(f"save failed: {type(e).__name__}: {e}")
    finally:
        session.close()
        # See this function's docstring on job.walk_seed -- must run regardless of outcome, so a
        # crashed/errored chain doesn't leave this worker process walk_seed-correlated either.
        random.seed(os.urandom(16))
        torch.seed()
        results_queue.put((job.meta_id, None if outcome == "error" else total_value))


def _resolve_meta_decision(
    job_queue, results_queue, meta_results_queue, main_worker_index: int, cfg: FacilityConfig,
    recorder: CompactRecorder, session, decision, character: str, seed: str, main_id: str,
    decision_kind: str, divergence_index: int, stop_event, queue_manager,
    last_combat_checkpoint_path: str | None, last_combat_entry_index: int | None,
):
    """Commits to a card_reward option immediately via ChoicesParadoxNet's own prediction
    (_choose_meta_option_claws) -- never a function of the options' own chain values, which aren't
    even known yet at this point (see this module's docstring) -- then dispatches one
    MetaOptionJob per legal option onto the shared branch-worker pool (job_queue) to evaluate them
    asynchronously; the main line does not wait for any of it. Falls back to the plain heuristic
    pick (what choose_action would have done anyway, via _choose_meta_option_claws's own round-1
    path) if fewer than 2 options exist to compare -- nothing to dispatch either way.

    Returns (decision, pending_meta_jobs): pending_meta_jobs is a list of (meta_id,
    checkpoint_path, is_adopted, commit_step) tuples for main_worker_loop to track until each job's
    result arrives on meta_results_queue -- see its docstring for how those get applied
    (recorder.set_value for the adopted option once known, and checkpoint refcounting so
    last_combat_checkpoint_path isn't deleted out from under a still-queued/running job).

    job_queue/results_queue/main_worker_index/stop_event/queue_manager are unused by this
    function's own body -- kept only because main_worker_loop's meta_resolver contract calls every
    meta_resolver with this same positional signature (see its docstring); facility_claws.py's
    combat_resolver still uses job_queue_priority/results_queue/stop_event for the branch-worker
    pool the main line's own combat uses.

    last_combat_checkpoint_path/last_combat_entry_index (from main_worker_loop, see its docstring):
    passed straight through to every option-chain job, which reconstructs its own starting state
    from it (MetaOptionJob/_run_meta_option_job's own load_and_resume_combat call, see its
    docstring and this module's docstring for why a fresh checkpoint at the card_reward itself
    doesn't work)."""
    options = legal_options(decision, decision_kind)
    if len(options) < 2:
        if not options:
            return decision, []  # defensive -- card_reward always offers skip_card_reward at least
        name, args = options[0]
        return session.step(name, **args), []

    prefix = list(session.action_log)
    # One shared walk_seed for every sibling option of THIS divergence (common random numbers --
    # see _run_meta_option_job's docstring). Drawn fresh per divergence, not per trial: a fixed
    # cross-divergence seed would make every divergence's chains replay identical "random" choices
    # too, which isn't wanted -- only siblings of the SAME divergence should share one.
    walk_seed = random.getrandbits(64)

    adopted_index = _choose_meta_option_claws(decision, decision_kind, options)

    # choices_paradox/model/materialize.py's materialize_main_trial pulls the main line's own
    # adopted meta-decision back out as a value-regression example, the same way facility_claws's
    # _absorb_actions attaches an adopted turn-branch's policy/value onto the main line -- it only
    # checks "policy" in entry as a presence marker (never reads its per-option content), so a
    # placeholder with no numeric values yet (filled in later via recorder.set_value once the
    # adopted chain's own job reports back -- see main_worker_loop) is a harmless stand-in for what
    # this used to carry synchronously.
    commit_step = recorder.mark_step()
    recorder.attach_policy([{"action": a, "args": kw, "value": None} for a, kw in options])
    name, args = options[adopted_index]
    decision = session.step(name, **args)

    pending_meta_jobs = []
    for option_index, (opt_name, opt_args) in enumerate(options):
        job = MetaOptionJob(
            main_id=main_id, character=character, seed=seed, ascension=cfg.ascension,
            decision_kind=decision_kind, divergence_index=divergence_index,
            option_index=option_index, prefix=prefix,
            last_combat_checkpoint_path=last_combat_checkpoint_path,
            last_combat_entry_index=last_combat_entry_index,
            commit_name=opt_name, commit_args=opt_args, walk_seed=walk_seed,
        )
        # The dispatched-to queue travels with the job itself, not looked up from a table -- see
        # branch_worker_loop's docstring. Lands on the plain (low-priority) job_queue so this
        # trial's own combat turn-branches (job_queue_priority) are never stuck behind it.
        job_queue.put((job, meta_results_queue))
        pending_meta_jobs.append(
            (job.meta_id, last_combat_checkpoint_path, option_index == adopted_index, commit_step)
        )

    return decision, pending_meta_jobs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Whispering Earring self-play facility (card_reward meta-branching, "
                     "for choices_paradox) -- the main line's own combat is still turn-branched "
                     "exactly like facility_claws.py's main line, sharing its branch-worker pool; "
                     "each card_reward option's own evaluation chain plays unbranched and is "
                     "dispatched to that same pool asynchronously, without the main line waiting "
                     "on it (see this module's docstring)"
    )
    add_common_cli_args(parser)
    parser.add_argument("--branches-per-turn", type=int, default=3,
                         help="combat turn-branching, same knob and default as facility_claws.py "
                              "-- only the main line's own combat uses it; option-chains play "
                              "unbranched (see this module's docstring)")
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
            branch_worker_loop,
            handlers={BranchJob: _run_branch_job, MetaOptionJob: _run_meta_option_job},
        ),
        resume=resume,
    )
    facility.run()
