"""Turns whispering_earring trial files into tokenized training examples

Every branch spawned by whispering_earring's turn-level branch-and-adopt (see
../../whispering_earring/facility_claws.py) replays an identical shared prefix before diverging on
its own first action, then plays out the rest of that turn unbranched (facility_claws.py's
_play_turn_only) under one shared set of per-stream REINFORCE advantages (the branch's end-of-turn
value vs. its group's baseline, one baseline per reward stream -- see _group_baselines). Scoring
only ever used that first divergent action
(comment left here in case a from-scratch rewrite is ever worth it: a branch's other siblings'
later actions live in states nothing else in this pipeline ever visits, since recorder.py
deliberately never persists per-step state and this codebase's native save/load only checkpoints
at room boundaries, not mid-combat -- forking each sibling's own continuation from the shared
divergence point would mean replaying its prefix from scratch, exactly the O(L^2) blowup
materialize_main_and_groups's single-walk optimization exists to avoid).

What IS cheap: whichever sibling was actually adopted into the main line has its WHOLE turn's
states already visited by this trial's own sequential replay (facility_claws.py's _absorb_actions
copies an adopted branch's actions verbatim into the main line). materialize_main_and_groups
scores that one sibling's later actions too (see _find_adopted_member), not just its first --
train.py's per-example policy_loss is -advantage*log_prob(a_t), so emitting one example per
(state, a_t) along that sibling's turn and letting them land anywhere across training batches
sums to advantage*sum_t(log_prob(a_t)) for the whole trajectory, the standard multi-step REINFORCE
objective, without train.py itself needing to know a trajectory boundary exists or this module
paying for any replay it wasn't already doing.

Usage:
    conda activate sts2
    python materialize.py ../../whispering_earring/data/raw/*.json --out dataset.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see model.py's top -- two OpenMP
# runtimes (conda's llvm-openmp + torch's bundled copy) otherwise abort on import

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

import sts2_path  # noqa: E402,F401  (sets sys.path to sts2-cli/python before the imports below)
from buffer_env import Sts2BufferSession  # noqa: E402
from installation import ensure_setup  # noqa: E402

from model import (  # noqa: E402
    GlobalVocab, POLICY_STREAMS, legal_options_from_buffer, tokenize_from_buffer,
)

# Saved as plain dicts of builtins/tensors, not TokenBatch/dataclass instances -- torch.save
# pickles a class by its import path, and TokenBatch would resolve to "__main__.TokenBatch" when
# this script is run directly, unloadable from any other module. Plain dicts sidestep that.
Example = dict[str, Any]

# n-step/TD(lambda) propagation constants -- tri_head_design.md's Propagation section flags the
# concrete value as needing a real sweep later; these are a reasonable starting point, not tuned.
_TD_GAMMA = 0.99
_TD_LAMBDA = 0.9

_ALL_STREAMS = (*POLICY_STREAMS, "win")


def _lambda_returns(
    rewards: list[float], bootstrap: list[float | None], gamma: float, lam: float
) -> list[float]:
    """Backward TD(lambda) over one already-fully-known reward sequence (tri_head_design.md's
    Propagation section): `rewards[t]` is earned during turn t (0-indexed, chronological, last
    index is the sequence's real end -- not a truncation), `bootstrap[t]` is a value estimate for
    the state right after turn t, or None where no such estimate is available/needed.

    G[T] = rewards[T]
    G[t] = rewards[t] + gamma * ((1-lam)*bootstrap[t] + lam*G[t+1])   if bootstrap[t] is not None
    G[t] = rewards[t] + gamma * G[t+1]                                if bootstrap[t] is None

    The `bootstrap is None` branch degrades to a plain discounted sum over the turns actually
    known -- correct (not a fallback approximation) whenever the sequence already reaches the
    real end, which every caller here guarantees (see this module's docstring on branch trials
    only ever covering one real turn, and main-line trials always reaching a real outcome)."""
    n = len(rewards)
    g = [0.0] * n
    g[-1] = rewards[-1]
    for t in range(n - 2, -1, -1):
        b = bootstrap[t]
        if b is not None:
            g[t] = rewards[t] + gamma * ((1 - lam) * b + lam * g[t + 1])
        else:
            g[t] = rewards[t] + gamma * g[t + 1]
    return g


@dataclass
class TurnTargets:
    """_turn_value_targets' per-action-index output. `values` is the 4 value-head regression
    targets (`{"aggro","defense","scale","win"}`). `terminal_bonus` is the RAW graded win/lose
    outcome (facility_common.py's `_combat_value` on a win, `-1.0` on a loss) for action indices
    belonging to a combat's own genuinely-final turn, else None -- feeds train.py's extra
    terminal-only gate REINFORCE term (on top of, not instead of, the regular per-turn advantage
    every decision gets)."""
    values: dict[str, float]
    terminal_bonus: float | None


def _turn_value_targets(
    actions: list[dict], gamma: float = _TD_GAMMA, lam: float = _TD_LAMBDA
) -> dict[int, TurnTargets]:
    """Every combat_play-turn action's index (into `actions`) -> its TurnTargets, built from
    recorder.py's per-turn tri-head reward dicts. Pure JSON -- no engine replay needed, `actions`
    can be a whole trial's action log or just one branch member's own post-divergence actions.

    recorder.py's end_turn() backfills the IDENTICAL reward dict onto every action of one turn.
    "value" isn't unique to combat_play turns though -- a meta/map job's own commit
    (attach_value()/set_value(), e.g. golden_compass's select_map_node reward) carries a plain
    float "value" too, and can sit right next to a combat's own turns in one trial's action log --
    so a maximal contiguous run of DICT-valued actions (not just "value"-bearing ones) is exactly
    one combat, and within it a maximal contiguous sub-run sharing the identical dict is exactly
    one turn.

    value_aggro's target is enemy_term's own return (offense reward), value_defense's is
    hp_term's, value_scale's is scale_term's -- see tri_head_design.md's Architecture section for
    the stream/head naming. hp_term/enemy_term/scale_term have no per-stream value estimate
    recorded at self-play time (those heads are new), so their returns are a real discounted sum
    to the combat's actual end (see _lambda_returns's docstring). win_value DOES have a real
    per-turn estimate already recorded (win_value_after, the self-play net's own live reading --
    today's existing "TD-bootstrapped" value, just now blended over multiple steps instead of 1)."""
    result: dict[int, TurnTargets] = {}
    i, n = 0, len(actions)
    while i < n:
        if not isinstance(actions[i].get("value"), dict):
            i += 1
            continue
        turns: list[tuple[dict, list[int]]] = []  # (reward dict, action indices) per turn
        while i < n and isinstance(actions[i].get("value"), dict):
            reward = actions[i]["value"]
            if turns and turns[-1][0] == reward:
                turns[-1][1].append(i)
            else:
                turns.append((reward, [i]))
            i += 1

        hp = [r["hp_term"] for r, _ in turns]
        enemy = [r["enemy_term"] for r, _ in turns]
        scale = [r["scale_term"] for r, _ in turns]
        win_after = [r["win_value_after"] for r, _ in turns]

        g_hp = _lambda_returns(hp, [None] * len(hp), gamma, lam)
        g_enemy = _lambda_returns(enemy, [None] * len(enemy), gamma, lam)
        g_scale = _lambda_returns(scale, [None] * len(scale), gamma, lam)
        # win reward is 0 every non-terminal turn (the terminal turn's own win_value_after IS the
        # real terminal outcome); bootstrap at turn t is that SAME turn's own win_value_after --
        # exactly what a 1-step target would already use, generalized to blend further turns in.
        win_rewards = [0.0] * (len(turns) - 1) + [win_after[-1]]
        win_bootstrap: list[float | None] = win_after[:-1] + [None]
        g_win = _lambda_returns(win_rewards, win_bootstrap, gamma, lam)

        # A combat's own final turn is "genuinely terminal" (as opposed to a branch trial's one
        # recorded turn simply running out of file) when either this scan stopped because the
        # NEXT action is a real, different decision (i < n -- the trial kept logging past this
        # combat, so it really ended in a win) or win_value_after is exactly the -1.0 sentinel
        # _play_turn_only writes on a loss. Heuristic, not a persisted "outcome" field
        # (recorder.py/facility_common.py are deliberately unchanged by this restart) -- the one
        # gap is a branch trial whose single turn happens to end in a win right at its own file's
        # end with no follow-up action recorded, which under-fires the bonus rather than
        # mis-firing it.
        is_terminal = i < n or win_after[-1] == -1.0

        for turn_i, (_, idxs) in enumerate(turns):
            values = {
                "aggro": g_enemy[turn_i], "defense": g_hp[turn_i], "scale": g_scale[turn_i],
                "win": g_win[turn_i],
            }
            bonus = win_after[-1] if (is_terminal and turn_i == len(turns) - 1) else None
            for idx in idxs:
                result[idx] = TurnTargets(values=values, terminal_bonus=bonus)
    return result


def _same_option(option: tuple[str, dict], recorded: dict) -> bool:
    return option[0] == recorded["action"] and option[1] == recorded.get("args", {})


def _head_of(entry: dict) -> str | None:
    """Which of POLICY_STREAMS the self-play sampler actually used for this recorded action
    (policy.py's _choose_combat_action_claws's hard/stochastic gating -- see tri_head_design.md's
    Training section) -- needed to route train.py's per-head/gate loss to the right head. None
    for a turn with no recorded policy (a non-policy_point placeholder) or one recorded before
    this field existed (the old flat-list `policy` format) -- callers skip a policy_real example
    in that case rather than mis-attribute credit to the wrong head."""
    policy = entry.get("policy")
    return policy.get("head") if isinstance(policy, dict) else None


def _option_index(options: list[tuple[str, dict]], entry: dict) -> int | None:
    """Which slot in the model's own legal_options_from_buffer() matches the action a trial
    actually recorded"""
    for i, option in enumerate(options):
        if _same_option(option, entry):
            return i
    return None


def _norm_action(entry: dict) -> tuple[str, tuple]:
    """Hashable key for an {action, args} pair, used both to key the shared prefix a branch
    trial replayed and to check how many distinct first actions its group's siblings sampled."""
    return (entry["action"], tuple(sorted(entry.get("args", {}).items())))


@dataclass
class _Group:
    """Everything materialize_group() needs for one (main_id, shared-prefix) group of sibling
    branch trials, captured in _scan_trials's single read pass so no branch file ever needs to
    be opened a second time. `members` holds each sibling's own post-divergence action entries
    (`[{action, args, value}, ...]`) -- the branch's ENTIRE remaining turn, not just its first
    novel action, so materialize_group()/materialize_main_and_groups() can score every action the
    branch took under its own per-stream advantages instead of only its first (see this module's
    docstring on `sum(log_prob(a_t))`). `end_turn()`'s whole-range backfill (recorder.py) means
    every entry in a member carries the same `value` -- reading it off entry 0 is enough for the
    baseline."""
    main_id: str
    character: str
    seed: str
    ascension: int
    prefix_actions: list[dict]
    members: list[list[dict]] = field(default_factory=list)


def _group_baselines(group: _Group) -> dict[str, float] | None:
    """The REINFORCE baseline for this group's turn-start decision, one per reward stream --
    each head's own policy loss is trained off ITS OWN stream's advantage (`targets[name] -
    baselines[name]`), not a single shared win-derived number: `policy_aggro` should be judged
    against "how much enemy damage did this action lead to, relative to what this group's other
    candidate actions averaged," not against the group's overall win outcome. Only the GATE
    (which strategy to trust) is trained off the `win` baseline here, since arbitrating between
    strategies by overall outcome IS the gate's job -- see tri_head_design.md's Training section.
    Without this split, hard-gating alone only gives each head distinct training DATA (whichever
    states the gate routes to it); every head would still be chasing the identical `win` signal,
    so nothing stops all three from converging toward the same generic "do whatever wins" policy
    rather than a genuine, recognizable offense/defense/scale specialization."""
    subkeys = {_norm_action(m[0]) for m in group.members}
    if len(subkeys) < 2:
        return None
    per_member_targets = [_turn_value_targets(m)[0].values for m in group.members]
    return {
        name: sum(t[name] for t in per_member_targets) / len(per_member_targets)
        for name in _ALL_STREAMS
    }


def _read_trial(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _scan_trials(paths: list[str]) -> tuple[dict[str, str], dict[tuple, _Group]]:
    """Single read pass over every trial file: classifies each as main-line or branch (grouped by the exact shared prefix
    it replayed before diverging). The reads themselves run on a thread pool. Returns
    (main_id -> path for every non-errored main-line trial, branch groups) -- main_id comes from
    each main-line file's own recorded field (see facility_common.py's main_worker_loop, which
    writes it as f"{character}_{seed}" -- same value every branch of that trial also carries),
    read for free here since _read_trial already loaded the whole file."""
    main_id_to_path: dict[str, str] = {}
    groups: dict[tuple, _Group] = {}

    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4)) as pool:
        trials = pool.map(_read_trial, paths)
        for path, trial in zip(paths, trials):
            _scan_one(path, trial, main_id_to_path, groups)

    return main_id_to_path, groups


def _scan_one(path: str, trial: dict, main_id_to_path: dict[str, str], groups: dict[tuple, _Group]) -> None:
    if trial.get("role") != "branch":
        if trial.get("error") is None and trial.get("main_id") is not None:
            main_id_to_path[trial["main_id"]] = path
        return
    if trial.get("branch_kind") in ("meta", "map"):
        return  # choices_paradox's MetaOptionJob / golden_compass's MapOptionJob files (different
        # shape -- see agents/choices_paradox/model/materialize.py and
        # agents/golden_compass/model/materialize.py, which are what actually read these); only
        # relevant if this ever points at a raw-dir shared with facility_choices_paradox.py. A
        # MapOptionJob's own first post-divergence step is select_map_node with a plain float
        # value (facility_common._apply_golden_compass_rewards), not a combat_play turn's tri-head
        # reward dict -- letting one through here breaks _turn_value_targets's index-0 assumption
        # (KeyError: 0), since it only skips non-dict-valued actions when scanning for turns.
    if trial.get("error") is not None:
        return

    prefix_len = trial.get("prefix_len")
    actions = trial.get("actions", [])
    if prefix_len is None or len(actions) <= prefix_len:
        return
    steps = actions[prefix_len:]  # the branch's whole post-divergence turn, not just its first
    # action -- a branch trial only ever plays one turn (facility_claws.py's _play_turn_only), so
    # this IS that turn's complete action sequence.
    if "value" not in steps[0]:
        return  # turn never resolved (branch errored mid-turn) -- no realized outcome

    prefix_actions = actions[:prefix_len]
    prefix_key = (trial["main_id"], tuple(_norm_action(a) for a in prefix_actions))
    group = groups.get(prefix_key)
    if group is None:
        group = _Group(
            main_id=trial["main_id"], character=trial["character"], seed=trial["seed"],
            ascension=trial.get("ascension", 0), prefix_actions=prefix_actions,
        )
        groups[prefix_key] = group
    group.members.append(steps)


def materialize_group(
    group: _Group, gvocab: GlobalVocab, session: Sts2BufferSession
) -> list[Example]:
    """Replays this group's shared prefix exactly once, tokenizes the resulting turn-start
    decision once, and emits one policy_real example per sibling directly from its own recorded
    first action + value. Only used as a fallback for a branch group whose main-line file is
    missing/errored (see main()) -- when the main file IS available,
    materialize_main_and_groups() below both reaches the same checkpoint without replaying this
    group's prefix from scratch AND (unlike this fallback) can score one sibling's WHOLE turn, not
    just its first action -- see that function's docstring for why this path can't do the same:
    with no main-line file, there's no way to tell which sibling would have been adopted, and
    replaying every sibling's own remaining turn here would mean re-walking this group's prefix
    once per sibling (this codebase's native save/load only checkpoints at room boundaries, not
    mid-combat -- see BufferSession.SaveCheckpoint in sts2-cli -- so there's no cheap fork point
    to replay each sibling's continuation from instead)."""
    baselines = _group_baselines(group)
    if baselines is None:
        return []

    decision = session.start(group.character, group.seed, group.ascension)
    for entry in group.prefix_actions:
        decision = session.step(entry["action"], **entry.get("args", {}))
    if decision.decision != "combat_play":
        return []  # defensive -- a turn-start checkpoint should always land here

    options = legal_options_from_buffer(decision)
    batch = tokenize_from_buffer(decision, gvocab)

    examples: list[Example] = []
    for member in group.members:
        first = member[0]
        action_index = _option_index(options, first)
        head = _head_of(first)
        if action_index is None or head is None:
            continue
        tt = _turn_value_targets(member)[0]
        examples.append({
            "type_ids": batch.type_ids,
            "global_ids": batch.global_ids,
            "features": batch.features,
            "player_pos": batch.player_pos,
            "hand_pos": batch.hand_pos,
            "enemy_pos": batch.enemy_pos,
            "potion_pos": batch.potion_pos,
            "options": options,
            "value_targets": tt.values,
            "action_index": action_index,
            "head": head,
            # Own-stream advantage per head, not one shared win-derived number -- see
            # _group_baselines' docstring.
            "advantages": {name: tt.values[name] - baselines[name] for name in _ALL_STREAMS},
            # No wider main-line context here (no main-line file exists for this fallback path --
            # see this function's docstring), so terminal detection degrades to the isolated,
            # loss-only case (_turn_value_targets' is_terminal note).
            "terminal_bonus": tt.terminal_bonus,
            "policy_real": True,
        })
    return examples


def _find_adopted_member(actions: list[dict], boundary: int, members: list[list[dict]]) -> list[dict] | None:
    """Whichever member of a branch group is an exact match of the trial's OWN action log
    starting at `boundary` -- guaranteed to exist (and be unique in practice) when that branch was
    the one adopted into the main line, since facility_claws.py's _absorb_actions copies an
    adopted branch's whole turn verbatim into the main recorder. Lets
    materialize_main_and_groups score that one sibling's WHOLE turn using states its own
    sequential walk visits anyway -- no extra engine replay, unlike every OTHER (non-adopted)
    sibling, whose later states genuinely aren't available without forking the live session (see
    materialize_group's docstring for why that fork isn't cheap here)."""
    for member in members:
        end = boundary + len(member)
        if end > len(actions):
            continue
        if all(_norm_action(actions[boundary + k]) == _norm_action(member[k]) for k in range(len(member))):
            return member
    return None


def materialize_main_and_groups(
    path: str, own_groups: list[_Group], gvocab: GlobalVocab, session: Sts2BufferSession
) -> list[Example]:
    """Single sequential replay of one main-line trial, emitting its own per-turn
    policy_real=False examples (same as this function used to do as materialize_main_trial) and,
    at each shared-prefix boundary one of its OWN sibling branch groups diverged from, that
    group's policy_real=True examples -- one per sibling's first action (same as materialize_group,
    without re-replaying from action 0), PLUS every remaining action of whichever sibling was
    actually adopted into this trial's own line (see _find_adopted_member and this module's
    docstring on `sum(log_prob(a_t))`): those later actions are states this walk was already going
    to visit for its own is_policy_point examples, so scoring them against the group's real
    advantage instead of the flat placeholder costs nothing extra -- `extension_map` is what
    carries a group's baseline forward from its boundary to those later positions.

    The prefix-reuse trick works because whispering_earring's turn-level branch-and-adopt
    (facility_claws.py) commits each turn's action to the main line before the next turn is even
    branched -- so every branch group spawned off this trial replayed a prefix that is guaranteed
    to be an EXACT prefix of this trial's own action log (verified against real data, 0 mismatches
    across 840 branch files). Before this, each of a trial's branch groups independently replayed
    its own prefix from scratch, so a trial with branch points every few turns paid for the sum of
    every prefix length instead of just the trial's own length -- on one real ~126-action trial
    with 28 branch groups, that was 1759 replayed steps instead of 126, a ~15x blowup that scales
    O(L^2) in trial length rather than O(L)."""
    trial = json.loads(Path(path).read_text())
    if trial.get("error") is not None:
        return []  # aborted trial -- no value signal, per ../README.md's Data section

    groups_by_len: dict[int, list[_Group]] = defaultdict(list)
    baselines: dict[int, dict[str, float]] = {}
    for g in own_groups:
        group_baselines = _group_baselines(g)
        if group_baselines is None:
            continue  # same guard materialize_group applies per-group
        groups_by_len[len(g.prefix_actions)].append(g)
        baselines[id(g)] = group_baselines

    actions = trial["actions"]
    # Pure JSON, no engine replay -- every combat_play turn's TurnTargets, keyed by action index
    # into `actions`. The main line always reaches a real terminal outcome for every combat it
    # plays, so this is a real (not estimated) lambda-return over each combat's whole turn
    # sequence -- see _turn_value_targets's docstring. Full-trial context also means this is the
    # only place terminal_bonus can be detected for a genuine WIN (see _turn_value_targets'
    # is_terminal note) -- reused below for the adopted sibling's own continuation.
    main_targets = _turn_value_targets(actions)
    examples: list[Example] = []
    decision = session.start(trial["character"], trial["seed"], trial.get("ascension", 0))

    # position -> (recorded entry, value_targets, advantages, terminal_bonus) for an adopted
    # sibling's 2nd-and-later actions, populated when its group's boundary is reached below and
    # consumed as the walk passes through each later position.
    extension_map: dict[int, tuple[dict, dict[str, float], dict[str, float], float | None]] = {}

    for i in range(len(actions) + 1):
        # i == len(actions) is a final sentinel pass -- covers the (unlikely but possible) case
        # of a branch group whose prefix is the trial's ENTIRE action log -- without stepping
        # anything further.
        entry = actions[i] if i < len(actions) else None
        extra = extension_map.pop(i, None)
        is_policy_point = (
            entry is not None and decision.decision == "combat_play" and "policy" in entry
        )
        groups_here = groups_by_len.get(i, [])
        if is_policy_point or groups_here or extra is not None:
            options = legal_options_from_buffer(decision)
            batch = tokenize_from_buffer(decision, gvocab)
            scored_by_extension = False
            if extra is not None and decision.decision == "combat_play":
                extra_entry, extra_values, extra_advantages, extra_bonus = extra
                action_index = _option_index(options, extra_entry)
                extra_head = _head_of(extra_entry)
                if action_index is not None and extra_head is not None:
                    examples.append({
                        "type_ids": batch.type_ids,
                        "global_ids": batch.global_ids,
                        "features": batch.features,
                        "player_pos": batch.player_pos,
                        "hand_pos": batch.hand_pos,
                        "enemy_pos": batch.enemy_pos,
                        "potion_pos": batch.potion_pos,
                        "options": options,
                        "value_targets": extra_values,
                        "action_index": action_index,
                        "head": extra_head,
                        "advantages": extra_advantages,
                        "terminal_bonus": extra_bonus,
                        "policy_real": True,
                    })
                    scored_by_extension = True
            if is_policy_point and not scored_by_extension:
                # already scored with its real advantage via extension_map above -- the flat
                # placeholder would only dilute that signal with a duplicate, zero-advantage copy
                tt = main_targets.get(i)
                examples.append({
                    "type_ids": batch.type_ids,
                    "global_ids": batch.global_ids,
                    "features": batch.features,
                    "player_pos": batch.player_pos,
                    "hand_pos": batch.hand_pos,
                    "enemy_pos": batch.enemy_pos,
                    "potion_pos": batch.potion_pos,
                    "options": options,
                    "value_targets": tt.values if tt is not None else {
                        "aggro": 0.0, "defense": 0.0, "scale": 0.0, "win": 0.0
                    },
                    "action_index": None,
                    "advantages": {name: 0.0 for name in _ALL_STREAMS},
                    "terminal_bonus": None,
                    "policy_real": False,
                })
            if decision.decision == "combat_play":  # defensive -- see materialize_group's guard
                for g in groups_here:
                    group_baselines = baselines[id(g)]
                    adopted = _find_adopted_member(actions, i, g.members)
                    for member in g.members:
                        first = member[0]
                        action_index = _option_index(options, first)
                        head = _head_of(first)
                        if action_index is None or head is None:
                            continue
                        tt = _turn_value_targets(member)[0]
                        # The adopted sibling's later states genuinely live in this trial's own
                        # action log (see _find_adopted_member's docstring), so its terminal
                        # detection can use main_targets' full-trial context (real win detection,
                        # not just the isolated loss-only case) -- every OTHER sibling's fate
                        # beyond this one turn is never actually known, so it keeps the isolated
                        # per-member bonus.
                        bonus = (
                            main_targets[i].terminal_bonus
                            if member is adopted and i in main_targets else tt.terminal_bonus
                        )
                        examples.append({
                            "type_ids": batch.type_ids,
                            "global_ids": batch.global_ids,
                            "features": batch.features,
                            "player_pos": batch.player_pos,
                            "hand_pos": batch.hand_pos,
                            "enemy_pos": batch.enemy_pos,
                            "potion_pos": batch.potion_pos,
                            "options": options,
                            "value_targets": tt.values,
                            "action_index": action_index,
                            "head": head,
                            "advantages": {
                                name: tt.values[name] - group_baselines[name] for name in _ALL_STREAMS
                            },
                            "terminal_bonus": bonus,
                            "policy_real": True,
                        })
                    if adopted is not None and len(adopted) > 1:
                        tt = _turn_value_targets(adopted)[0]
                        advantages = {
                            name: tt.values[name] - group_baselines[name] for name in _ALL_STREAMS
                        }
                        for k in range(1, len(adopted)):
                            idx = i + k
                            bonus = main_targets[idx].terminal_bonus if idx in main_targets else None
                            extension_map[idx] = (adopted[k], tt.values, advantages, bonus)
        if entry is None:
            break
        decision = session.step(entry["action"], **entry.get("args", {}))

    return examples


_gvocab: GlobalVocab | None = None
_session: Sts2BufferSession | None = None


def _silence_worker_stderr() -> None:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)


def _init_worker() -> None:
    _silence_worker_stderr()
    global _gvocab, _session
    _gvocab = GlobalVocab(EntityVocab.load())
    _session = Sts2BufferSession()


def _materialize_merged_task(
    path: str, own_groups: list[_Group]
) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None  # set by _init_worker before any task runs
    label = path if not own_groups else f"{path} (+{len(own_groups)} branch group(s))"
    try:
        examples = materialize_main_and_groups(path, own_groups, _gvocab, _session)
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"
    return label, examples, None


def _materialize_group_task(group: _Group) -> tuple[str, list[Example] | None, str | None]:
    assert _gvocab is not None and _session is not None
    label = f"group({group.main_id}, {len(group.prefix_actions)}-action prefix, {len(group.members)} branches)"
    try:
        examples = materialize_group(group, _gvocab, _session)
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"
    return label, examples, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trials", nargs="+",
        help="trial JSON files, or directories to glob *.json from (data/raw/*.json) -- "
             "directories let a huge data/raw/ be passed as one argument instead of a shell-"
             "expanded file list, which can exceed the OS's exec argument-length limit",
    )
    parser.add_argument("--out", default="dataset.pt")
    parser.add_argument(
        "--jobs", "-j", type=int, default=os.cpu_count(),
        help="parallel worker processes, each loading its own in-process sts2-cli CLR session "
             "(default: cpu count)",
    )
    args = parser.parse_args()

    trial_paths: list[str] = []
    for trial in args.trials:
        if os.path.isdir(trial):
            trial_paths.extend(sorted(str(p) for p in Path(trial).glob("*.json")))
        else:
            trial_paths.append(trial)

    # Once, here, in the parent -- not per worker, so a first-time DLL copy/build (see
    # ensure_setup) can't race across several worker processes hitting it at once.
    ensure_setup()

    # Single read pass (cheap -- no CLR session needed, just reading the files) that both
    # separates out main-line trials and groups every branch trial by its shared prefix -- see
    # _scan_trials's docstring.
    main_id_to_path, groups = _scan_trials(trial_paths)
    real_groups = {k: g for k, g in groups.items() if _group_baselines(g) is not None}
    print(f"{len(main_id_to_path)} main-line trial(s), {len(groups)} branch group(s) "
          f"({len(real_groups)} with real cross-branch signal)")

    # Every group replays a prefix of its OWN main trial's action log (branch-and-adopt commits
    # each turn before the next is even branched -- see materialize_main_and_groups's docstring),
    # so route each group through one shared walk of its main file instead of replaying its
    # prefix from scratch. A group only falls back to the old from-scratch replay when its main
    # file is missing/errored (excluded from main_id_to_path above).
    groups_by_main_id: dict[str, list[_Group]] = defaultdict(list)
    for group in groups.values():
        groups_by_main_id[group.main_id].append(group)
    orphan_groups = [
        g for main_id, gs in groups_by_main_id.items() if main_id not in main_id_to_path for g in gs
    ]
    if orphan_groups:
        print(f"{len(orphan_groups)} branch group(s) have no matching main-line file -- "
              f"replaying independently from action 0")

    all_examples: list[Example] = []
    per_file_counts: list[tuple[str, int]] = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool:
        futures = [
            pool.submit(_materialize_merged_task, path, groups_by_main_id.get(main_id, []))
            for main_id, path in main_id_to_path.items()
        ]
        futures += [pool.submit(_materialize_group_task, group) for group in orphan_groups]
        for future in as_completed(futures):
            label, examples, error = future.result()
            if error is not None:
                print(f"skip {label}: {error}")
                continue
            assert examples is not None
            if examples:
                per_file_counts.append((label, len(examples)))
            all_examples.extend(examples)

    _print_truncated_counts(per_file_counts)
    _print_dataset_summary(all_examples)
    torch.save(all_examples, args.out)
    print(f"wrote {len(all_examples)} examples to {args.out}")


def _print_dataset_summary(examples: list[Example]) -> None:
    """First-run sanity check for the tri-head value_targets/advantage a full materialize pass
    just produced -- catches, without a manual debug session, the degenerate case a
    checkpoint-less self-play round hits (see facility_common.py's _win_value_of warning): every
    continuing turn's win_value_after is a flat 0.0, so any sibling group scored entirely over
    non-terminal turns gets baseline==value and a real but useless advantage of exactly 0.0."""
    if not examples:
        print("dataset summary: 0 examples -- nothing to report")
        return
    real = [ex for ex in examples if ex.get("policy_real", False)]
    nonzero_adv = [ex for ex in real if any(v != 0.0 for v in ex["advantages"].values())]
    print(
        f"dataset summary: {len(examples)} examples total, {len(real)} policy_real "
        f"({len(nonzero_adv)} with nonzero advantage)"
    )
    if real and not nonzero_adv:
        print(
            "  WARNING: every policy_real example has advantage==0.0 on every stream -- the "
            "policy loss and gate will get no real gradient this round. Likely cause: no "
            "CLAWS_CHECKPOINT was loaded during self-play (see facility_common.py's "
            "_win_value_of warning), so win_value_after was a flat 0.0 on every non-combat-ending "
            "turn and every sibling comparison over such a turn produced baseline==value. "
            "Expected for a checkpoint-less round 1."
        )
    for name in _ALL_STREAMS:
        vals = [ex["value_targets"][name] for ex in examples]
        lo, hi = min(vals), max(vals)
        mean = sum(vals) / len(vals)
        nonzero = sum(1 for v in vals if v != 0.0)
        print(
            f"  value_targets[{name}]: mean={mean:+.4f} min={lo:+.4f} max={hi:+.4f} "
            f"({nonzero}/{len(vals)} nonzero)"
        )
    if real:
        n_terminal = sum(1 for ex in real if ex.get("terminal_bonus") is not None)
        print(f"  terminal_bonus present on {n_terminal}/{len(real)} policy_real examples")
    if real:
        # Hard/stochastic gating (policy.py's _choose_combat_action_claws) means self-play itself
        # decides which head generates each policy_real example -- if one head is barely
        # represented here, train.py's per-head loss has barely any examples to learn that head
        # from regardless of how training goes (see tri_head_design.md's Training section).
        head_counts = {name: 0 for name in POLICY_STREAMS}
        no_head = 0
        for ex in real:
            head = ex.get("head")
            if head in head_counts:
                head_counts[head] += 1
            else:
                no_head += 1
        counts_str = " ".join(f"{name}={n}" for name, n in head_counts.items())
        print(f"  head usage among policy_real examples: {counts_str} (no_head={no_head})")


def _print_truncated_counts(per_file_counts: list[tuple[str, int]], head_tail: int = 5) -> None:
    """Print "{label}: N examples" per file, but only the first/last `head_tail` when there are
    many -- with hundreds of raw trial files the full per-file listing just floods the terminal."""
    total = len(per_file_counts)
    if total <= 2 * head_tail:
        for label, n in per_file_counts:
            print(f"{label}: {n} examples")
        return
    for label, n in per_file_counts[:head_tail]:
        print(f"{label}: {n} examples")
    print(f"... ({total - 2 * head_tail} more files) ...")
    for label, n in per_file_counts[-head_tail:]:
        print(f"{label}: {n} examples")


if __name__ == "__main__":
    main()
