"""Per-round + cumulative stats for whispering_earring's choices_paradox meta-branch trials (see
facility_choices_paradox.py) -- sibling to stats.py, reusing its generic main-line floor/outcome/
seed tables and history-log machinery as-is (same main-line file shape: role="main", "combats",
"outcome" -- a card_reward divergence doesn't change how a run's own floor progress is recorded).
Only the choices_paradox-specific pieces are new here:

- Meta-decision adoption quality: for every resolved divergence, how the option actually adopted
  onto the main line (via _choose_meta_option_claws) compares to whichever option's own chain
  scored best in evaluation -- pulled straight from the main line's own attached policy/value (see
  facility_choices_paradox._resolve_meta_decision), no replay needed since the raw values are
  already stored. "adopted == best" trending toward 1.0 over rounds is the signal that the net's
  own prediction is converging toward what evaluation actually found, which is the whole point of
  this facility's decoupled-adoption design (see that module's docstring).
- Card pick stats, name-resolved: same problem stats.py's own card-stats table solves for combat
  (see its docstring) -- select_card_reward's card_index is only meaningful against whichever
  cards were actually offered at that exact decision, and the recorder never stores card identity,
  only the index. Recovering names needs one replay per distinct divergence (grouped by shared
  prefix, reusing choices_paradox/model/materialize.py's own _scan_trials -- the M=1-per-option
  branch-file shape that module already groups) to decode the offered cards, then two tallies
  against those names: n_evaluated/avg_value (every option-chain's own realized value, from the
  branch files) and n_adopted (how often the real trial actually took that card, from a second,
  full-trial replay of the main-line files -- mirrors choices_paradox/model/materialize.py's own
  materialize_main_and_groups replay shape).

Usage:
    python stats_choices.py [--raw-dir data/raw_choices] [--history-file PATH] [--round N]
                             [--jobs N] [--no-card-stats]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from stats import (
    _append_history, _fmt_rate, _floor_stats, _load_trials, _print_floor_table, _print_outcomes,
    _print_seed_table, _print_value_summary, _read_history, _seed_rows, _value_summary,
)

cp_materialize = cp_model = bl = Sts2BufferSession = Sts2Session = decode_token = ensure_setup = None
# see _import_choices_stats_deps -- only the card-stats replay path needs torch/the CLR bridge


def _import_choices_stats_deps() -> None:
    """Same lazy-import reasoning as stats.py's own _import_card_stats_deps -- --no-card-stats
    runs shouldn't have to pay for torch/the CLR bridge. Deliberately loaded from
    choices_paradox/model/, not claws/model/ -- a different package's materialize.py/model.py
    (different tokenizer, different _Group shape), never both in the same process (see
    facility_choices_paradox.py's own comment on why that matters)."""
    global cp_materialize, cp_model, bl, Sts2BufferSession, Sts2Session, decode_token, ensure_setup
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "choices_paradox" / "model"))
    import materialize as _cp_materialize  # _scan_trials -- shared-prefix branch grouping
    import model as _cp_model  # legal_options_from_buffer

    import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
    import buffer_layout as _bl
    from buffer_env import Sts2BufferSession as _Sts2BufferSession, decode_token as _decode_token
    from env import Sts2Session as _Sts2Session  # rest_site/event_choice name resolution only --
    # see _rest_event_group_task's docstring for why these two need the JSON protocol, not buffer
    from installation import ensure_setup as _ensure_setup

    cp_materialize, cp_model = _cp_materialize, _cp_model
    bl, Sts2BufferSession, Sts2Session = _bl, _Sts2BufferSession, _Sts2Session
    decode_token, ensure_setup = _decode_token, _ensure_setup


# ---------------------------------------------------------------------------
# Meta-decision adoption quality (no replay needed -- values are already in the JSON)
# ---------------------------------------------------------------------------

def _meta_decision_rows(main_trials: list[dict]) -> list[dict]:
    """One row per resolved card_reward divergence this round -- distinguishes a meta-decision's
    own value-shaped policy ({"action","args","value"} per option) from combat's prob-shaped one
    ({"action","args","prob"}) by checking the first entry's own keys."""
    rows = []
    for trial in main_trials:
        for entry in trial.get("actions", []):
            policy = entry.get("policy")
            if not isinstance(policy, list) or not policy or "value" not in policy[0]:
                continue
            values = [p["value"] for p in policy if p.get("value") is not None]
            if len(values) < 2:
                continue
            adopted_value = entry.get("value")
            best_value = max(values)
            rows.append({
                "main_id": trial.get("main_id"), "n_options": len(policy),
                "adopted_action": entry.get("action"), "adopted_value": adopted_value,
                "best_value": best_value, "worst_value": min(values),
                "adopted_is_best": adopted_value is not None and adopted_value >= best_value - 1e-9,
            })
    return rows


def _print_meta_decision_summary(rows: list[dict], title: str) -> None:
    print(f"\n-- {title}: meta-decision adoption ({len(rows)} divergence(s)) -- "
          f"'adopted==best' trending toward 1.0 is the net's own prediction converging toward "
          f"what evaluation actually found --")
    if not rows:
        print("no resolved meta-decisions found.")
        return
    n_best = sum(1 for r in rows if r["adopted_is_best"])
    avg_options = sum(r["n_options"] for r in rows) / len(rows)
    avg_spread = sum(r["best_value"] - r["worst_value"] for r in rows) / len(rows)
    adopted_values = [r["adopted_value"] for r in rows if r["adopted_value"] is not None]
    avg_adopted = sum(adopted_values) / len(adopted_values) if adopted_values else None
    print(f"avg options compared: {avg_options:.2f}   avg value spread (best-worst): {avg_spread:.3f}")
    print(f"adopted == best-scoring option: {n_best}/{len(rows)} ({_fmt_rate(n_best, len(rows))})")
    print(f"avg adopted value: {avg_adopted:.3f}" if avg_adopted is not None else "avg adopted value: n/a")


# ---------------------------------------------------------------------------
# Card pick stats, name-resolved (needs replay -- see module docstring)
# ---------------------------------------------------------------------------

def _decode_option_names(decision, options: list[tuple[str, dict]]) -> list[str]:
    """One decoded card name per legal_options_from_buffer(...) entry, same order -- "<skip>" for
    the skip option (no card of its own), decode_token(card_token) for every select_card_reward.
    Handles both buffer decision kinds a "card_reward" decision_kind can actually be (see
    choices_paradox/model/model.py's legal_options_from_buffer docstring for why)."""
    arr = decision.arr
    cards = dict(
        bl.event_card_reward_cards(arr) if decision.decision == "event_card_reward"
        else bl.reward_cards(arr)
    )
    names = []
    for action, args in options:
        if action == "skip_card_reward":
            names.append("<skip>")
            continue
        slot = cards.get(args["card_index"])
        names.append((decode_token(bl.card_token(slot)) or "?") if slot is not None else "?")
    return names


def _decode_rest_event_names(decision, decision_kind: str, options: list[tuple[str, dict]]) -> list[str]:
    """One name per legal_options(...) entry for a rest_site/event_choice decision -- "<leave_room>"
    for that fixed non-option action, otherwise the JSON protocol's own per-option identity: a
    RestOption's option_id (e.g. "REST_SITE_HEAL") or an EventOption's text_key, prefixed with the
    event's name since text_key alone isn't guaranteed unique across different events. See
    _rest_event_group_task's docstring for why this needs `decision` to come from Sts2Session
    (JSON protocol), not the buffer-protocol decision _decode_option_names above works from."""
    if decision_kind == "rest_site":
        by_index = {o.index: o.option_id for o in decision.options}
    else:
        by_index = {o.index: f"{decision.event_name}: {o.text_key}" for o in decision.options}
    names = []
    for action, args in options:
        names.append("<leave_room>" if action == "leave_room" else by_index.get(args["option_index"], "?"))
    return names


_stats_session: Sts2BufferSession | None = None
_json_stats_session: Sts2Session | None = None  # lazy per worker -- most rounds have zero
# rest_site/event_choice divergences, so don't pay for a dotnet subprocess unless one's needed


def _get_json_stats_session() -> Sts2Session:
    global _json_stats_session
    if _json_stats_session is None:
        _json_stats_session = Sts2Session()
    return _json_stats_session


def _rest_event_group_task(group) -> tuple[str, dict[str, list], dict[str, list], int]:
    """JSON-protocol counterpart to _card_value_group_task, for rest_site/event_choice groups
    only. The buffer protocol used everywhere else in this facility (see facility_choices_paradox.
    _choose_meta_option_claws's own comment on this) carries no per-option identity for either
    kind -- buffer_layout.rest_options()/event_options() expose only (present, is_enabled)/
    is_locked flags, nothing like the card token that lets _decode_option_names name a card. The
    JSON protocol's RestOption/EventOption models do carry a real id, so this replays the group's
    shared prefix through a real dotnet subprocess (Sts2Session) instead of the in-process CLR
    (Sts2BufferSession) the rest of this module uses, just to reach that one decision and read its
    options' names off. Slower per group, but only run for the (typically few) rest_site/
    event_choice divergences, not every trial -- see _rest_event_pick_stats."""
    global _json_stats_session
    session = _get_json_stats_session()
    counts: dict[str, list] = defaultdict(lambda: [0, 0.0])
    adv_counts: dict[str, list] = defaultdict(lambda: [0, 0.0])
    try:
        decision = session.start(group.character, group.seed, group.ascension)
        for entry in group.prefix_actions:
            decision = session.step(entry["action"], **entry.get("args", {}))
        if decision.decision != group.decision_kind:
            return group.decision_kind, {}, {}, 1

        baseline = cp_materialize._group_baseline(group)
        options = cp_model.legal_options(decision, group.decision_kind)
        names = _decode_rest_event_names(decision, group.decision_kind, options)

        for member in group.members:
            idx = cp_materialize._option_index(options, member)
            name = names[idx] if idx is not None else "?"
            if "value" in member:
                row = counts[name]
                row[0] += 1
                row[1] += member["value"]
                if baseline is not None:
                    adv_row = adv_counts[name]
                    adv_row[0] += 1
                    adv_row[1] += member["value"] - baseline
        return group.decision_kind, dict(counts), dict(adv_counts), 0
    except Exception:
        _json_stats_session = None  # subprocess likely dead/wedged -- force a fresh one next task
        return group.decision_kind, {}, {}, 1


def _init_choices_stats_worker() -> None:
    _import_choices_stats_deps()
    # Sts2BufferSession loads Sts2Headless's CLR in-process, and its [SIM]/[INFO] logging writes
    # unconditionally to this process's real fd 2 -- silence it, same as stats.py's own worker.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    global _stats_session
    _stats_session = Sts2BufferSession()


def _card_value_group_task(group) -> tuple[dict[str, list], dict[str, list], int]:
    """Replays one meta-branch group's shared prefix once, decodes the offered cards' names, then
    tallies every sibling option's own realized value against the card it committed to -- both
    raw and against this SAME group's own skip-option value (materialize.py's _group_baseline,
    the same v_i - v0 causal baseline train.py's policy loss trains on -- see that module's
    docstring for why skip, not the group mean, is the right reference). Returns (name ->
    [n_evaluated, value_sum], name -> [n_evaluated, advantage_sum] (only for groups with a real
    skip baseline), 1-if-skipped-else-0)."""
    assert _stats_session is not None
    counts: dict[str, list] = defaultdict(lambda: [0, 0.0])
    adv_counts: dict[str, list] = defaultdict(lambda: [0, 0.0])
    try:
        decision = _stats_session.start(group.character, group.seed, group.ascension)
        for entry in group.prefix_actions:
            decision = _stats_session.step(entry["action"], **entry.get("args", {}))
    except Exception:
        return {}, {}, 1  # prefix replay failed -- skip this group, don't kill the round's stats
    if decision.decision not in ("card_reward", "event_card_reward"):
        return {}, {}, 1

    baseline = cp_materialize._group_baseline(group)  # None if skip's own chain errored/missing,
    # or fewer than 2 distinct options were observed -- same guard materialize.py itself uses.

    options = cp_model.legal_options_from_buffer(decision, group.decision_kind)
    names = _decode_option_names(decision, options)

    for member in group.members:
        idx = cp_materialize._option_index(options, member)
        name = names[idx] if idx is not None else "?"
        if "value" in member:
            row = counts[name]
            row[0] += 1
            row[1] += member["value"]
            if baseline is not None:
                adv_row = adv_counts[name]
                adv_row[0] += 1
                adv_row[1] += member["value"] - baseline
    return dict(counts), dict(adv_counts), 0


def _card_adoption_trial_task(path: str) -> tuple[dict[str, int], int]:
    """Full sequential replay of one main-line trial (mirrors choices_paradox/model/materialize.py's
    own materialize_main_and_groups), tallying which card NAME each resolved meta-decision actually
    adopted -- needs the live card_reward/event_card_reward state at that exact point to decode
    it, same reason the value-side replay above does."""
    assert _stats_session is not None
    try:
        trial = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return {}, 1
    if trial.get("error") is not None:
        return {}, 1
    counts: dict[str, int] = defaultdict(int)
    try:
        decision = _stats_session.start(trial["character"], trial["seed"], trial.get("ascension", 0))
        for entry in trial["actions"]:
            policy = entry.get("policy")
            if (decision.decision in ("card_reward", "event_card_reward")
                    and policy and "value" in policy[0]):
                options = cp_model.legal_options_from_buffer(decision, "card_reward")
                names = _decode_option_names(decision, options)
                idx = cp_materialize._option_index(options, entry)
                counts[names[idx] if idx is not None else "?"] += 1
            decision = _stats_session.step(entry["action"], **entry.get("args", {}))
    except Exception:
        return dict(counts), 1
    return dict(counts), 0


_CARD_DECISION_KINDS = frozenset({"card_reward", "event_card_reward"})
_REST_EVENT_DECISION_KINDS = frozenset({"rest_site", "event_choice"})


def _pick_stats(raw_dir: Path, jobs: int) -> tuple[dict, dict[str, list], dict[str, list], int, int]:
    """Returns (per_kind, name -> [n_evaluated, value_sum, n_adopted] for card_reward/
    event_card_reward only, name -> [n_evaluated, advantage_sum] for the same, groups+trials
    skipped, groups+trials total). `per_kind` is {"rest_site": (total, adv_total), "event_choice":
    (total, adv_total)} -- name -> [n_evaluated, value_sum] each, no adoption count (see
    _rest_event_group_task's docstring: these two need a slow JSON-protocol replay per group, and
    adding trial-level adoption counts on top would mean a second full JSON-protocol replay of
    every main-line trial, not just the divergent groups -- skipped as not worth that cost yet).
    Card pick stats are the pre-existing name-resolved table; empty/zero if there's nothing to
    replay yet (e.g. round 1 before any branch has finished)."""
    _import_choices_stats_deps()
    paths = [str(p) for p in raw_dir.glob("*.json")]
    try:
        main_paths, groups = cp_materialize._scan_trials(paths)
    except Exception as e:
        print(f"warning: card-stats grouping pass failed ({type(e).__name__}: {e}); "
              "skipping card/rest/event stats for this round")
        return {}, {}, {}, 0, 0

    card_groups = [g for g in groups.values() if g.decision_kind in _CARD_DECISION_KINDS]
    rest_groups = [g for g in groups.values() if g.decision_kind == "rest_site"]
    event_groups = [g for g in groups.values() if g.decision_kind == "event_choice"]

    total: dict[str, list] = defaultdict(lambda: [0, 0.0, 0])
    adv_total: dict[str, list] = defaultdict(lambda: [0, 0.0])
    rest_total: dict[str, list] = defaultdict(lambda: [0, 0.0])
    rest_adv_total: dict[str, list] = defaultdict(lambda: [0, 0.0])
    event_total: dict[str, list] = defaultdict(lambda: [0, 0.0])
    event_adv_total: dict[str, list] = defaultdict(lambda: [0, 0.0])
    skipped = 0
    n_units = len(groups) + len(main_paths)
    if n_units == 0:
        return {}, {}, {}, 0, 0

    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_choices_stats_worker) as pool:
        value_futures = [pool.submit(_card_value_group_task, g) for g in card_groups]
        rest_event_futures = [
            pool.submit(_rest_event_group_task, g) for g in (*rest_groups, *event_groups)
        ]
        adoption_futures = [pool.submit(_card_adoption_trial_task, p) for p in main_paths]
        for fut in as_completed(value_futures):
            counts, adv_counts, skip = fut.result()
            skipped += skip
            for name, (n, val_sum) in counts.items():
                row = total[name]
                row[0] += n
                row[1] += val_sum
            for name, (n, adv_sum) in adv_counts.items():
                adv_row = adv_total[name]
                adv_row[0] += n
                adv_row[1] += adv_sum
        for fut in as_completed(rest_event_futures):
            decision_kind, counts, adv_counts, skip = fut.result()
            skipped += skip
            dst, adv_dst = (
                (rest_total, rest_adv_total) if decision_kind == "rest_site"
                else (event_total, event_adv_total)
            )
            for name, (n, val_sum) in counts.items():
                row = dst[name]
                row[0] += n
                row[1] += val_sum
            for name, (n, adv_sum) in adv_counts.items():
                adv_row = adv_dst[name]
                adv_row[0] += n
                adv_row[1] += adv_sum
        for fut in as_completed(adoption_futures):
            counts, skip = fut.result()
            skipped += skip
            for name, n in counts.items():
                total[name][2] += n
    per_kind = {
        "rest_site": (dict(rest_total), dict(rest_adv_total)),
        "event_choice": (dict(event_total), dict(event_adv_total)),
    }
    return per_kind, dict(total), dict(adv_total), skipped, n_units


def _print_card_value_table(total: dict[str, list], adv_total: dict[str, list], title: str) -> None:
    print(f"\n-- {title}: card pick value, name-resolved -- 'avg_value' raw realized value from "
          f"every option-chain that committed to this card; 'avg_vs_skip' = v_i - v0 against that "
          f"same draft's own skip-option value (materialize.py's causal baseline -- see its "
          f"docstring), the actual signal the policy trains on, positive means better than not "
          f"picking anything; 'n_adopted' from how often the real trial actually took it --")
    if not total:
        print("no card value data found.")
        return
    rows = []
    for name, (n, val_sum, n_adopted) in total.items():
        adv_n, adv_sum = adv_total.get(name, (0, 0.0))
        vs_skip = f"{adv_sum / adv_n:+.3f}" if adv_n else "n/a"
        rows.append((name, n, val_sum / n if n else 0.0, vs_skip, n_adopted))
    rows.sort(key=lambda r: -r[2])
    print(f"{'card':<28} {'n_evaluated':>11} {'avg_value':>10} {'avg_vs_skip':>12} {'n_adopted':>10}")
    for name, n, avg, vs_skip, n_adopted in rows:
        print(f"{name:<28} {n:>11} {avg:>10.3f} {vs_skip:>12} {n_adopted:>10}")


def _print_option_value_table(
    total: dict[str, list], adv_total: dict[str, list], title: str, name_header: str,
) -> None:
    """Same shape as _print_card_value_table, minus the n_adopted column -- rest_site/event_choice
    stats only come from group replays (see _pick_stats), no trial-level adoption replay."""
    print(f"\n-- {title}: {name_header} value, name-resolved -- 'avg_value' raw realized value "
          f"from every option-chain that committed to this option; 'avg_vs_skip' = v_i - v0 "
          f"against that same draft's own skip-option value (materialize.py's causal baseline), "
          f"positive means better than not picking anything --")
    if not total:
        print(f"no {name_header} value data found.")
        return
    rows = []
    for name, (n, val_sum) in total.items():
        adv_n, adv_sum = adv_total.get(name, (0, 0.0))
        vs_skip = f"{adv_sum / adv_n:+.3f}" if adv_n else "n/a"
        rows.append((name, n, val_sum / n if n else 0.0, vs_skip))
    rows.sort(key=lambda r: -r[2])
    print(f"{name_header:<28} {'n_evaluated':>11} {'avg_value':>10} {'avg_vs_skip':>12}")
    for name, n, avg, vs_skip in rows:
        print(f"{name:<28} {n:>11} {avg:>10.3f} {vs_skip:>12}")


def _print_card_coverage_table(coverage: dict[str, list], n_rounds_total: int, title: str) -> None:
    """Cross-round -- unlike _print_card_value_table, this is safe to accumulate across rounds
    since it's just presence counts, not a situational value (see _aggregate's comment)."""
    print(f"\n-- {title}: card pool coverage across {n_rounds_total} round(s) -- how many rounds "
          f"each card has actually been evaluated in at least once, so you know how much of the "
          f"pool self-play has covered (NOT an averaged value -- that's situational, see 'This "
          f"round' card table for this round's own) --")
    if not coverage:
        print("no card coverage data found.")
        return
    rows = sorted(coverage.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    print(f"{len(rows)} distinct card(s) seen across {n_rounds_total} round(s)")
    print(f"{'card':<28} {'n_rounds':>9} {'n_evaluated_total':>18}")
    for name, (n_rounds, n_total) in rows:
        print(f"{name:<28} {n_rounds:>9} {n_total:>18}")


# ---------------------------------------------------------------------------
# History log: append-only JSONL, one line per stats_choices.py invocation
# ---------------------------------------------------------------------------

def _aggregate(records: list[dict]) -> dict:
    per_floor: dict[str, dict] = defaultdict(
        lambda: {"combats": 0, "won": 0, "lost": 0, "combat_idx_sum": 0}
    )
    reached_floors: list = []
    outcomes: dict[str, int] = defaultdict(int)
    card_coverage: dict[str, list] = defaultdict(lambda: [0, 0])  # [n_rounds, n_evaluated_total]
    value_n = value_sum = value_sumsq = 0.0
    value_min = value_max = None
    n_main = n_branch = n_branch_errors = n_divergences = n_adopted_best = 0
    gen_seconds_total = 0.0

    for rec in records:
        for k, v in rec.get("per_floor", {}).items():
            row = per_floor[k]
            row["combats"] += v.get("combats", 0)
            row["won"] += v.get("won", 0)
            row["lost"] += v.get("lost", 0)
            row["combat_idx_sum"] += v.get("combat_idx_sum", 0)
        reached_floors.extend(rec.get("reached_floors", []))
        for k, v in rec.get("outcomes", {}).items():
            outcomes[k] += v
        # Deliberately NOT accumulating card_stats/card_adv_stats VALUES across rounds here --
        # unlike floor/outcome/adoption stats, a card's value is situational (deck/act/context-
        # dependent), so averaging its evaluations across many different rounds/drafts doesn't
        # mean anything (_print_card_value_table is only ever called with a single round's own
        # card_total/card_adv_total -- see main()'s "This round" call). Presence is fine to
        # accumulate though -- "how many rounds has this card even been evaluated in" is a plain
        # count, not a situational value, and is what tells you how much of the card pool
        # self-play has actually covered.
        for name in rec.get("card_stats", {}):
            n, _val_sum, _n_adopted = rec["card_stats"][name]
            row = card_coverage[name]
            row[0] += 1
            row[1] += n
        vs = rec.get("adopted_value_summary") or {}
        value_n += vs.get("n", 0)
        value_sum += vs.get("sum", 0.0)
        value_sumsq += vs.get("sumsq", 0.0)
        if vs.get("min") is not None:
            value_min = vs["min"] if value_min is None else min(value_min, vs["min"])
        if vs.get("max") is not None:
            value_max = vs["max"] if value_max is None else max(value_max, vs["max"])
        n_main += rec.get("n_main_trials", 0)
        n_branch += rec.get("n_branch_trials", 0)
        n_branch_errors += rec.get("n_branch_errors", 0)
        n_divergences += rec.get("n_divergences", 0)
        n_adopted_best += rec.get("n_adopted_best", 0)
        if rec.get("gen_wall_seconds"):
            gen_seconds_total += rec["gen_wall_seconds"]

    return {
        "per_floor": {k: dict(v) for k, v in per_floor.items()},
        "reached_floors": reached_floors,
        "outcomes": dict(outcomes),
        "card_coverage": dict(card_coverage),
        "adopted_value_summary": {"n": value_n, "sum": value_sum, "sumsq": value_sumsq,
                                   "min": value_min, "max": value_max},
        "n_main_trials": n_main, "n_branch_trials": n_branch, "n_branch_errors": n_branch_errors,
        "n_divergences": n_divergences, "n_adopted_best": n_adopted_best,
        "gen_wall_seconds_total": gen_seconds_total, "n_rounds": len(records),
    }


def _print_trend(records: list[dict], limit: int = 40) -> None:
    if not records:
        return
    shown = records[-limit:]
    print(f"\n-- Round trend (last {len(shown)} of {len(records)}) --")
    print(f"{'Round':>5} {'Timestamp':<20} {'Ckpt':>4} {'Divergences':>11} {'Adopt=Best':>10} "
          f"{'AvgAdoptVal':>11} {'Trials/hr':>10}")
    for rec in shown:
        n_div = rec.get("n_divergences", 0)
        n_best = rec.get("n_adopted_best", 0)
        vs = rec.get("adopted_value_summary") or {}
        avg_value = f"{vs['sum'] / vs['n']:.3f}" if vs.get("n") else "n/a"
        wall = rec.get("gen_wall_seconds")
        n_trials = rec.get("n_main_trials", 0) + rec.get("n_branch_trials", 0)
        rate = f"{n_trials / wall * 3600:.1f}" if wall else "n/a"
        ts = str(rec.get("timestamp", "?"))[:19]
        ckpt = "yes" if rec.get("choices_paradox_checkpoint") else "no"
        print(f"{rec.get('round', '?'):>5} {ts:<20} {ckpt:>4} {n_div:>11} "
              f"{_fmt_rate(n_best, n_div):>10} {avg_value:>11} {rate:>10}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-round + cumulative stats for whispering_earring's choices_paradox trials")
    parser.add_argument("--raw-dir", default="data/raw_choices",
                         help="directory of this round's trial JSON files (default: "
                              "'data/raw_choices', resolved against the process's cwd if relative)")
    parser.add_argument("--history-file", default=None,
                         help="append-only JSONL log (default: <raw-dir's parent>/"
                              "stats_history_choices.jsonl, so it survives --raw-dir being wiped)")
    parser.add_argument("--round", type=int, default=None,
                         help="round number to tag this entry with (default: one past the "
                              "highest round already in --history-file, or 1 if empty/missing)")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 4,
                         help="parallel worker processes for the card-stats replay pass "
                              "(default: cpu count)")
    parser.add_argument("--no-card-stats", action="store_true",
                         help="skip the replay-based, name-resolved card/rest_site/event_choice "
                              "tables (faster; floor/outcome/adoption stats only, still logged "
                              "to history)")
    parser.add_argument("--history-only", action="store_true",
                         help="just pretty-print --history-file's cumulative Summary/trend "
                              "tables and exit -- doesn't touch --raw-dir or append a new round")
    parser.add_argument("--trend-limit", type=int, default=40)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    history_file = (Path(args.history_file) if args.history_file
                     else raw_dir.parent / "stats_history_choices.jsonl")

    if args.history_only:
        history = _read_history(history_file)
        if not history:
            print(f"no history found in {history_file}")
            return
        cumulative = _aggregate(history)
        print(f"=== Summary: cumulative across {cumulative['n_rounds']} round(s) "
              f"in {history_file} ===")
        _print_floor_table(cumulative["per_floor"], cumulative["reached_floors"], "Summary")
        _print_outcomes(cumulative["outcomes"], "Summary")
        print(f"\n-- Summary: meta-decision adoption "
              f"({cumulative['n_divergences']} divergence(s)) --")
        print(f"adopted == best-scoring option: {cumulative['n_adopted_best']}/"
              f"{cumulative['n_divergences']} "
              f"({_fmt_rate(cumulative['n_adopted_best'], cumulative['n_divergences'])})")
        _print_value_summary(cumulative["adopted_value_summary"], "Summary adopted value")
        _print_card_coverage_table(cumulative["card_coverage"], cumulative["n_rounds"], "Summary")
        if cumulative["n_branch_errors"]:
            print(f"Summary branch trial errors: {cumulative['n_branch_errors']}/"
                  f"{cumulative['n_branch_trials']}")
        if cumulative["gen_wall_seconds_total"]:
            total_trials = cumulative["n_main_trials"] + cumulative["n_branch_trials"]
            rate = total_trials / cumulative["gen_wall_seconds_total"] * 3600
            print(f"lifetime generation throughput: {total_trials} trial(s) over "
                  f"{cumulative['gen_wall_seconds_total'] / 3600:.2f}h of generation time "
                  f"(~{rate:.1f} trials/hr)")
        _print_trend(history, limit=args.trend_limit)
        return

    main_trials, n_branch, n_branch_errors, gen_wall = _load_trials(raw_dir)
    floor = _floor_stats(main_trials)
    meta_rows = _meta_decision_rows(main_trials)

    print(f"=== This round: {len(main_trials)} main-line, {n_branch} branch trial file(s) "
          f"in {raw_dir} ===")
    if gen_wall:
        print(f"generation wall time (raw file mtime span): {gen_wall:.0f}s")
    if n_branch_errors:
        print(f"branch trial errors: {n_branch_errors}/{n_branch}")

    _print_floor_table(floor["per_floor"], floor["reached_floors"], "This round")
    _print_outcomes(floor["outcomes"], "This round")
    _print_seed_table(_seed_rows(main_trials), "This round")
    _print_meta_decision_summary(meta_rows, "This round")

    card_total: dict[str, list] = {}
    card_adv_total: dict[str, list] = {}
    rest_total: dict[str, list] = {}
    rest_adv_total: dict[str, list] = {}
    event_total: dict[str, list] = {}
    event_adv_total: dict[str, list] = {}
    skipped = n_units = 0
    if not args.no_card_stats:
        _import_choices_stats_deps()
        ensure_setup()
        per_kind, card_total, card_adv_total, skipped, n_units = _pick_stats(raw_dir, args.jobs)
        rest_total, rest_adv_total = per_kind.get("rest_site", ({}, {}))
        event_total, event_adv_total = per_kind.get("event_choice", ({}, {}))
        _print_card_value_table(card_total, card_adv_total, "This round")
        _print_option_value_table(rest_total, rest_adv_total, "This round", "rest_site option")
        _print_option_value_table(event_total, event_adv_total, "This round", "event option")
        if n_units:
            print(f"(replayed {n_units - skipped}/{n_units} group(s)/trial(s) for card/rest/"
                  f"event identity; {skipped} skipped)")

    history = _read_history(history_file)
    round_num = (args.round if args.round is not None
                 else max((r.get("round", 0) for r in history), default=0) + 1)
    now = datetime.now().astimezone()

    adopted_values = [r["adopted_value"] for r in meta_rows if r["adopted_value"] is not None]
    record = {
        "round": round_num,
        "timestamp": now.isoformat(timespec="seconds"),
        "timestamp_unix": now.timestamp(),
        "raw_dir": str(raw_dir),
        "choices_paradox_checkpoint": os.environ.get("CHOICES_PARADOX_CHECKPOINT") or None,
        "n_main_trials": len(main_trials),
        "n_branch_trials": n_branch,
        "n_branch_errors": n_branch_errors,
        "gen_wall_seconds": gen_wall,
        "outcomes": floor["outcomes"],
        "per_floor": floor["per_floor"],
        "reached_floors": floor["reached_floors"],
        "n_divergences": len(meta_rows),
        "n_adopted_best": sum(1 for r in meta_rows if r["adopted_is_best"]),
        "adopted_value_summary": _value_summary(adopted_values),
        "card_stats": card_total,
        "card_adv_stats": card_adv_total,
        "rest_stats": rest_total,
        "rest_adv_stats": rest_adv_total,
        "event_stats": event_total,
        "event_adv_stats": event_adv_total,
        "card_groups_total": n_units,
        "card_groups_skipped": skipped,
    }
    _append_history(history_file, record)
    print(f"\n(appended round {round_num} to {history_file})")

    all_records = history + [record]
    cumulative = _aggregate(all_records)
    print(f"\n=== Summary: cumulative across {cumulative['n_rounds']} round(s) "
          f"in {history_file} ===")
    _print_floor_table(cumulative["per_floor"], cumulative["reached_floors"], "Summary")
    _print_outcomes(cumulative["outcomes"], "Summary")
    print(f"\n-- Summary: meta-decision adoption ({cumulative['n_divergences']} divergence(s)) --")
    print(f"adopted == best-scoring option: {cumulative['n_adopted_best']}/"
          f"{cumulative['n_divergences']} "
          f"({_fmt_rate(cumulative['n_adopted_best'], cumulative['n_divergences'])})")
    _print_value_summary(cumulative["adopted_value_summary"], "Summary adopted value")
    _print_card_coverage_table(cumulative["card_coverage"], cumulative["n_rounds"], "Summary")
    if cumulative["n_branch_errors"]:
        print(f"Summary branch trial errors: {cumulative['n_branch_errors']}/"
              f"{cumulative['n_branch_trials']}")
    if cumulative["gen_wall_seconds_total"]:
        total_trials = cumulative["n_main_trials"] + cumulative["n_branch_trials"]
        rate = total_trials / cumulative["gen_wall_seconds_total"] * 3600
        print(f"lifetime generation throughput: {total_trials} trial(s) over "
              f"{cumulative['gen_wall_seconds_total'] / 3600:.2f}h of generation time "
              f"(~{rate:.1f} trials/hr)")

    _print_trend(all_records, limit=args.trend_limit)


if __name__ == "__main__":
    main()
