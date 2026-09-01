"""Per-round + cumulative stats for whispering_earring main-line and branch trials.

training.sh wipes data/raw/*.json at the start of every round (see its "Cleaning up old raw
data" step), so this script can only ever see one round's worth of trial files at a time --
whatever's currently sitting in --raw-dir. To make a full-day training.sh run legible (was it
improving, or flat, at 3am?), every invocation:

  1. Computes "This round" stats from --raw-dir's current contents.
  2. Appends one JSON line describing this round to --history-file (default:
     <raw-dir's parent>/stats_history.jsonl -- a sibling of raw/, so it survives raw/ being
     wiped). Each line carries enough raw counts (not just derived rates) to be re-aggregated
     later, plus a wall-clock timestamp.
  3. Reads the whole history file back and prints "Summary" tables that aggregate every round
     recorded so far, plus a per-round trend table (win rate / avg value / throughput over
     time) -- this is the "is it improving or doing nothing" view.

Terminology (one main-line trial == one "seed" == one attempted STS2 run):

- "combat": one resolved encounter (regular fight, elite, or boss) within a seed's run --
  see facility.py's _resolve_combat, called once per encounter as the seed's main line steps
  through the map. Not a whole run, and not a single turn within a fight.
- "floor": the map floor (within an act) the game was on when that combat happened
  (decision.context.floor at the time -- floors don't repeat within a run, they only
  increase, but not every floor has a combat: rest sites, events, and shops don't).
- "combat #" / combat_index: this combat's 1-indexed position within its own seed's run (the
  2nd combat this seed fought is combat_index=2, wherever on the map it happened) -- this is
  what answers "how many combats had this seed already fought by the time it reached floor
  F?", since floor number alone conflates two runs that took different paths to get there.

Two kinds of per-round data feed this:

- Floor/outcome stats: (act, floor, outcome) is recorded live by facility.py's _resolve_combat
  via CompactRecorder.record_combat() -- see recorder.py -- so no replay is needed for these;
  only main-line (role == "main") files carry a "combats" list and a final "outcome".
- Card offer/pick stats (which card got offered vs. actually played, and at what average
  softmax probability): this needs the *card identity* behind each action's card_index, which
  the recorder never stores (see recorder.py's docstring -- state is deliberately never stored,
  only replayed). Getting names back means replaying each distinct turn-start decision once,
  the same shared-prefix-grouped approach materialize.py and debug_card_bias.py use, reused
  here via materialize._scan_trials so the grouping logic isn't duplicated a third time.

Usage:
    python stats.py [--raw-dir data/raw] [--history-file PATH] [--round N] [--jobs N]
                     [--no-card-stats]
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

materialize = bl = Sts2BufferSession = decode_token = ensure_setup = None  # see _import_card_stats_deps


def _import_card_stats_deps() -> None:
    """Only the card offer/pick replay path (_card_stats/_init_card_worker) needs torch
    (via materialize) and the CLR bridge (via buffer_env) -- --history-only and --no-card-stats
    runs shouldn't have to pay for or even have those installed, so these are imported lazily
    instead of at module load. Cheap to call more than once (sys.modules caches it)."""
    global materialize, bl, Sts2BufferSession, decode_token, ensure_setup
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claws" / "model"))
    import materialize as _materialize  # _scan_trials -- shared-prefix branch grouping, reused as-is

    import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
    import buffer_layout as _bl
    from buffer_env import Sts2BufferSession as _Sts2BufferSession, decode_token as _decode_token
    from installation import ensure_setup as _ensure_setup

    materialize, bl = _materialize, _bl
    Sts2BufferSession, decode_token, ensure_setup = _Sts2BufferSession, _decode_token, _ensure_setup


# ---------------------------------------------------------------------------
# Loading this round's trial files
# ---------------------------------------------------------------------------

def _load_trials(raw_dir: Path) -> tuple[list[dict], int, int]:
    """Single read pass. Main-role trials are kept in full (small in number, needed for
    floor/outcome stats). Branch trials are only counted here -- _card_stats below re-reads
    them anyway via materialize._scan_trials to build shared-prefix groups, so keeping a full
    second in-memory copy here would just double memory for files that can run into the
    thousands per round."""
    main_trials = []
    n_branch = 0
    n_branch_errors = 0
    for path in sorted(raw_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # e.g. a file still being written when this ran mid-round
        role = data.get("role")
        if role == "main":
            # No timestamp is recorded inside the trial JSON itself (recorder.py writes the
            # file once, at trial end, with no wall-clock field) -- the file's mtime is the
            # only "when was this played" signal available, so stash it for _seed_rows.
            data["_file_mtime"] = path.stat().st_mtime
            main_trials.append(data)
        elif role == "branch":
            n_branch += 1
            if data.get("error") is not None:
                n_branch_errors += 1
    return main_trials, n_branch, n_branch_errors


def _mtime_span(raw_dir: Path) -> float | None:
    """Rough generation wall-clock time for this round: span between the oldest and newest
    trial file's mtime. Needs no changes to facility.py/recorder.py (neither stamps timing
    today) -- the filesystem already knows when each file was written."""
    mtimes = [p.stat().st_mtime for p in raw_dir.glob("*.json")]
    return (max(mtimes) - min(mtimes)) if mtimes else None


# ---------------------------------------------------------------------------
# Floor / outcome stats
# ---------------------------------------------------------------------------

def _floor_stats(main_trials: list[dict]) -> dict:
    per_floor: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"combats": 0, "won": 0, "lost": 0, "combat_idx_sum": 0}
    )
    max_floor_per_run: dict[str, tuple[int, int]] = {}

    for trial in main_trials:
        main_id = trial.get("main_id")
        for combat in trial.get("combats", []):
            key = (combat["act"], combat["floor"])
            per_floor[key]["combats"] += 1
            per_floor[key][combat["outcome"]] += 1
            # combat_index is 1-indexed within *this* trial's own run -- summed here so the
            # table below can show, per floor, the average number of combats a seed had
            # already fought by the time it got there (see module docstring's Terminology).
            per_floor[key]["combat_idx_sum"] += combat["combat_index"]
            if main_id not in max_floor_per_run or key > max_floor_per_run[main_id]:
                max_floor_per_run[main_id] = key

    outcomes: dict[str, int] = defaultdict(int)
    for trial in main_trials:
        outcomes[trial.get("outcome", "unknown")] += 1

    return {
        "per_floor": {f"{a},{f}": dict(v) for (a, f), v in per_floor.items()},
        "reached_floors": [list(v) for v in max_floor_per_run.values()],
        "outcomes": dict(outcomes),
    }


def _fmt_rate(num: int, den: int) -> str:
    return f"{num / den:.3f}" if den else "n/a"


def _print_floor_table(per_floor: dict[str, dict], reached_floors: list, title: str) -> None:
    print(f"\n-- {title}: floors -- 'AvgCombat#' is, on average, how many combats a seed had "
          f"already fought (this one included) by the time it fought at that floor --")
    if not per_floor:
        print("no combat stats found (no main-line files with a recorded 'combats' list yet).")
        return
    reached = sorted(tuple(x) for x in reached_floors)
    total_runs = len(reached)
    keys = sorted(tuple(int(x) for x in k.split(",")) for k in per_floor)

    print(f"{'Act':>3} {'Floor':>5} {'Runs Reached':>12} {'Combats':>7} {'Won':>5} {'Lost':>5} "
          f"{'AvgCombat#':>10}")
    for act, floor in keys:
        stats = per_floor[f"{act},{floor}"]
        runs_reached = sum(1 for f in reached if f >= (act, floor))
        n_combats = stats.get("combats", 0)
        idx_sum = stats.get("combat_idx_sum", 0)
        # combat_index is always >= 1, so idx_sum == 0 here can only mean this bucket was
        # aggregated from pre-upgrade history lines that don't carry the field yet.
        avg_idx = f"{idx_sum / n_combats:.1f}" if n_combats and idx_sum else "n/a"
        print(f"{act:>3} {floor:>5} {runs_reached:>12} {n_combats:>7} "
              f"{stats.get('won', 0):>5} {stats.get('lost', 0):>5} {avg_idx:>10}")
    print(f"{total_runs} run(s) with combat data")


def _print_outcomes(outcomes: dict[str, int], title: str) -> None:
    won, lost = outcomes.get("won", 0), outcomes.get("lost", 0)
    line = f"{title} outcomes: " + (", ".join(f"{k}={v}" for k, v in sorted(outcomes.items()))
                                     or "none")
    line += f"  (win_rate={_fmt_rate(won, won + lost)})"
    print(line)


def _seed_rows(main_trials: list[dict]) -> list[dict]:
    """One row per seed (== per main-line trial) -- this round's raw material for the
    aggregate floor/outcome tables above, but shown per-run since e.g. two seeds reaching
    "act 1 floor 9" via 6 vs. 9 combats is lost once they're blended into one bucket. Doesn't
    survive into --history-file: raw_dir's per-seed files get wiped every round (see module
    docstring), so this view only ever covers "this round"."""
    rows = []
    for trial in main_trials:
        combats = trial.get("combats", [])
        reached = max(((c["act"], c["floor"]) for c in combats), default=None)
        rows.append({
            "seed": trial.get("seed", "?"),
            "character": trial.get("character", "?"),
            "n_combats": len(combats),
            "n_won": sum(1 for c in combats if c["outcome"] == "won"),
            "n_lost": sum(1 for c in combats if c["outcome"] == "lost"),
            "reached": reached,
            "outcome": trial.get("outcome", "unknown"),
            "mtime": trial.get("_file_mtime"),
        })
    rows.sort(key=lambda r: (r["reached"] or (0, 0)), reverse=True)
    return rows


def _print_seed_table(rows: list[dict], title: str) -> None:
    print(f"\n-- {title}: per-seed -- one row per attempted run "
          f"('Played' is the trial file's mtime, i.e. when it finished writing) --")
    if not rows:
        print("no main-line trials found.")
        return
    print(f"{'Seed':<20} {'Character':<10} {'Combats':>7} {'Won':>5} {'Lost':>5} "
          f"{'Reached':>9} {'Outcome':>9} {'Played':>19}")
    for r in rows:
        reached = f"{r['reached'][0]},{r['reached'][1]}" if r["reached"] else "n/a"
        played = (datetime.fromtimestamp(r["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
                  if r["mtime"] else "n/a")
        print(f"{r['seed']:<20} {r['character']:<10} {r['n_combats']:>7} {r['n_won']:>5} "
              f"{r['n_lost']:>5} {reached:>9} {r['outcome']:>9} {played:>19}")


# ---------------------------------------------------------------------------
# Card offer/pick stats (needs replay to recover card identity -- see module docstring)
# ---------------------------------------------------------------------------

def _option_name(action: str, args: dict, hand_names: dict[int, str]) -> str:
    if action == "play_card":
        return hand_names.get(args["card_index"], "?")
    if action == "use_potion":
        return "<potion>"
    return "<end_turn>"


_card_session: Sts2BufferSession | None = None


def _init_card_worker() -> None:
    # Fresh subprocess (ProcessPoolExecutor spawns) -- doesn't inherit the parent's lazy import.
    _import_card_stats_deps()
    # Sts2BufferSession loads Sts2Headless's CLR in-process, and its [SIM]/[INFO] logging
    # writes unconditionally to this process's real fd 2 -- silence it the same way
    # materialize.py's workers do.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    global _card_session
    _card_session = Sts2BufferSession()


def _card_stats_group_task(group) -> tuple[dict[str, list], list[float], int]:
    """Replays one turn-branch group's shared prefix exactly once, decodes the resulting hand's
    card names, then tallies every sibling branch's *offered* options (its logged `policy`) and
    *picked* option (its own recorded action) against those names. Returns
    (name -> [n_offered, n_picked, prob_mass], list of each sibling's `value`, 1-if-skipped-else-0).
    """
    assert _card_session is not None
    counts: dict[str, list] = defaultdict(lambda: [0, 0, 0.0])
    values: list[float] = []
    try:
        decision = _card_session.start(group.character, group.seed, group.ascension)
        for entry in group.prefix_actions:
            decision = _card_session.step(entry["action"], **entry.get("args", {}))
    except Exception:
        return {}, [], 1  # prefix replay failed -- skip this group, don't kill the round's stats
    if decision.decision != "combat_play":
        return {}, [], 1

    arr = decision.arr
    hand_names = {idx: (decode_token(bl.card_token(slot, 0)) or "?")
                  for idx, slot in bl.hand_cards(arr)}

    for member in group.members:
        policy = member.get("policy")
        if policy:
            for opt in policy:
                row = counts[_option_name(opt["action"], opt["args"], hand_names)]
                row[0] += 1
                row[2] += opt["prob"]
        counts[_option_name(member["action"], member.get("args", {}), hand_names)][1] += 1
        if "value" in member:
            values.append(member["value"])

    return {k: v for k, v in counts.items()}, values, 0


def _card_stats(raw_dir: Path, jobs: int) -> tuple[dict[str, list], list[float], int, int]:
    """Returns (name -> [n_offered, n_picked, prob_mass], all turn-values seen, groups skipped,
    total groups). Empty/zero if there's nothing to replay yet (e.g. round 1 before any branch
    has finished)."""
    _import_card_stats_deps()
    paths = [str(p) for p in raw_dir.glob("*.json")]
    try:
        _, groups = materialize._scan_trials(paths)
    except Exception as e:
        print(f"warning: card-stats grouping pass failed ({type(e).__name__}: {e}); "
              "skipping card stats for this round")
        return {}, [], 0, 0
    if not groups:
        return {}, [], 0, 0

    total: dict[str, list] = defaultdict(lambda: [0, 0, 0.0])
    values: list[float] = []
    skipped = 0
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_card_worker) as pool:
        futures = [pool.submit(_card_stats_group_task, g) for g in groups.values()]
        for fut in as_completed(futures):
            counts, vals, skip = fut.result()
            skipped += skip
            values.extend(vals)
            for name, (n_off, n_pick, mass) in counts.items():
                row = total[name]
                row[0] += n_off
                row[1] += n_pick
                row[2] += mass
    return dict(total), values, skipped, len(groups)


def _print_card_table(total: dict[str, list], title: str) -> None:
    print(f"\n-- {title}: card offer/pick bias --")
    if not total:
        print("no policy/card data found.")
        return
    rows = []
    for name, (n_off, n_pick, mass) in total.items():
        rows.append((name, n_off, n_pick, mass / n_off if n_off else 0.0))
    rows.sort(key=lambda r: -r[3])
    print(f"{'card':<24} {'n_offered':>10} {'n_picked':>10} {'avg_prob':>10}")
    for name, n, p, avg in rows:
        print(f"{name:<24} {n:>10} {p:>10} {avg:>10.4f}")


def _value_summary(values: list[float]) -> dict:
    """Compact sufficient statistics (n/sum/sumsq/min/max), not the raw list -- history entries
    accumulate over a full day, and per-branch value lists would otherwise dominate the file's
    size for no benefit (mean/stdev/min/max are all these combine into anyway)."""
    if not values:
        return {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}
    return {
        "n": len(values), "sum": sum(values), "sumsq": sum(v * v for v in values),
        "min": min(values), "max": max(values),
    }


def _print_value_summary(vs: dict, title: str) -> None:
    n = vs.get("n", 0)
    if not n:
        print(f"{title}: no turn values recorded.")
        return
    mean = vs["sum"] / n
    variance = max(vs["sumsq"] / n - mean * mean, 0.0)
    print(f"{title}: n={n} mean={mean:.4f} stdev={variance ** 0.5:.4f} "
          f"min={vs['min']:.4f} max={vs['max']:.4f}")


# ---------------------------------------------------------------------------
# History log: append-only JSONL, one line per stats.py invocation (== one training.sh round)
# ---------------------------------------------------------------------------

def _read_history(history_file: Path) -> list[dict]:
    if not history_file.exists():
        return []
    records = []
    with open(history_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn last line from a killed previous run
    return records


def _append_history(history_file: Path, record: dict) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def _aggregate(records: list[dict]) -> dict:
    per_floor: dict[str, dict] = defaultdict(
        lambda: {"combats": 0, "won": 0, "lost": 0, "combat_idx_sum": 0}
    )
    reached_floors: list = []
    outcomes: dict[str, int] = defaultdict(int)
    card_stats: dict[str, list] = defaultdict(lambda: [0, 0, 0.0])
    value_n = value_sum = value_sumsq = 0.0
    value_min = value_max = None
    n_main = n_branch = n_branch_errors = 0
    gen_seconds_total = 0.0

    for rec in records:
        for k, v in rec.get("per_floor", {}).items():
            row = per_floor[k]
            row["combats"] += v.get("combats", 0)
            row["won"] += v.get("won", 0)
            row["lost"] += v.get("lost", 0)
            row["combat_idx_sum"] += v.get("combat_idx_sum", 0)  # absent in pre-upgrade history lines -> 0
        reached_floors.extend(rec.get("reached_floors", []))
        for k, v in rec.get("outcomes", {}).items():
            outcomes[k] += v
        for name, (n_off, n_pick, mass) in rec.get("card_stats", {}).items():
            row = card_stats[name]
            row[0] += n_off
            row[1] += n_pick
            row[2] += mass
        vs = rec.get("value_summary") or {}
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
        if rec.get("gen_wall_seconds"):
            gen_seconds_total += rec["gen_wall_seconds"]

    return {
        "per_floor": {k: dict(v) for k, v in per_floor.items()},
        "reached_floors": reached_floors,
        "outcomes": dict(outcomes),
        "card_stats": dict(card_stats),
        "value_summary": {"n": value_n, "sum": value_sum, "sumsq": value_sumsq,
                           "min": value_min, "max": value_max},
        "n_main_trials": n_main, "n_branch_trials": n_branch, "n_branch_errors": n_branch_errors,
        "gen_wall_seconds_total": gen_seconds_total,
        "n_rounds": len(records),
    }


def _print_trend(records: list[dict], limit: int = 40) -> None:
    if not records:
        return
    shown = records[-limit:]
    print(f"\n-- Round trend (last {len(shown)} of {len(records)}) "
          f"-- use this to see whether it's improving or flat --")
    print(f"{'Round':>5} {'Timestamp':<20} {'Ckpt':>4} {'Trials':>7} {'WinRate':>7} "
          f"{'AvgValue':>9} {'Trials/hr':>10}")
    for rec in shown:
        outcomes = rec.get("outcomes", {})
        won, lost = outcomes.get("won", 0), outcomes.get("lost", 0)
        vs = rec.get("value_summary") or {}
        avg_value = f"{vs['sum'] / vs['n']:.4f}" if vs.get("n") else "n/a"
        wall = rec.get("gen_wall_seconds")
        n_trials = rec.get("n_main_trials", 0) + rec.get("n_branch_trials", 0)
        rate = f"{n_trials / wall * 3600:.1f}" if wall else "n/a"
        ts = str(rec.get("timestamp", "?"))[:19]
        ckpt = "yes" if rec.get("claws_checkpoint") else "no"
        print(f"{rec.get('round', '?'):>5} {ts:<20} {ckpt:>4} {n_trials:>7} "
              f"{_fmt_rate(won, won + lost):>7} {avg_value:>9} {rate:>10}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-round + cumulative stats for whispering_earring trials")
    parser.add_argument("--raw-dir", default="data/raw",
                         help="directory of this round's trial JSON files (default: 'data/raw', "
                              "resolved against the process's cwd if relative)")
    parser.add_argument("--history-file", default=None,
                         help="append-only JSONL log this round's stats get added to, and that "
                              "cumulative Summary/trend tables are built from (default: "
                              "<raw-dir's parent>/stats_history.jsonl, so it survives --raw-dir "
                              "being wiped between rounds)")
    parser.add_argument("--round", type=int, default=None,
                         help="round number to tag this entry with (default: one past the "
                              "highest round already in --history-file, or 1 if empty/missing)")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 4,
                         help="parallel worker processes for the card-stats replay pass "
                              "(default: cpu count)")
    parser.add_argument("--no-card-stats", action="store_true",
                         help="skip the replay-based card offer/pick table (faster; floor/"
                              "outcome stats only, still logged to history)")
    parser.add_argument("--history-only", action="store_true",
                         help="just pretty-print --history-file's cumulative Summary/trend "
                              "tables and exit -- doesn't touch --raw-dir or append a new round. "
                              "For casually checking progress (e.g. mid-run) without the "
                              "raw_dir's current contents or a card-stats replay pass")
    parser.add_argument("--trend-limit", type=int, default=40,
                         help="how many of the most recent rounds to show in the trend table "
                              "(default: 40)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    history_file = (Path(args.history_file) if args.history_file
                     else raw_dir.parent / "stats_history.jsonl")

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
        _print_card_table(cumulative["card_stats"], "Summary")
        _print_value_summary(cumulative["value_summary"], "Summary turn value")
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

    main_trials, n_branch, n_branch_errors = _load_trials(raw_dir)
    floor = _floor_stats(main_trials)
    gen_wall = _mtime_span(raw_dir)

    print(f"=== This round: {len(main_trials)} main-line, {n_branch} branch trial file(s) "
          f"in {raw_dir} ===")
    if gen_wall:
        print(f"generation wall time (raw file mtime span): {gen_wall:.0f}s")
    if n_branch_errors:
        print(f"branch trial errors: {n_branch_errors}/{n_branch}")

    _print_floor_table(floor["per_floor"], floor["reached_floors"], "This round")
    _print_outcomes(floor["outcomes"], "This round")
    _print_seed_table(_seed_rows(main_trials), "This round")

    card_total: dict[str, list] = {}
    values: list[float] = []
    n_groups = skipped = 0
    if not args.no_card_stats:
        _import_card_stats_deps()
        ensure_setup()
        card_total, values, skipped, n_groups = _card_stats(raw_dir, args.jobs)
        _print_card_table(card_total, "This round")
        _print_value_summary(_value_summary(values), "This round turn value")
        if n_groups:
            print(f"(replayed {n_groups - skipped}/{n_groups} turn-branch groups for card "
                  f"identity; {skipped} skipped)")

    history = _read_history(history_file)
    round_num = (args.round if args.round is not None
                 else max((r.get("round", 0) for r in history), default=0) + 1)
    now = datetime.now().astimezone()

    record = {
        "round": round_num,
        "timestamp": now.isoformat(timespec="seconds"),
        "timestamp_unix": now.timestamp(),
        "raw_dir": str(raw_dir),
        "claws_checkpoint": os.environ.get("CLAWS_CHECKPOINT") or None,
        "n_main_trials": len(main_trials),
        "n_branch_trials": n_branch,
        "n_branch_errors": n_branch_errors,
        "gen_wall_seconds": gen_wall,
        "outcomes": floor["outcomes"],
        "per_floor": floor["per_floor"],
        "reached_floors": floor["reached_floors"],
        "card_stats": card_total,
        "card_groups_total": n_groups,
        "card_groups_skipped": skipped,
        "value_summary": _value_summary(values),
    }
    _append_history(history_file, record)
    print(f"\n(appended round {round_num} to {history_file})")

    all_records = history + [record]
    cumulative = _aggregate(all_records)
    print(f"\n=== Summary: cumulative across {cumulative['n_rounds']} round(s) "
          f"in {history_file} ===")
    _print_floor_table(cumulative["per_floor"], cumulative["reached_floors"], "Summary")
    _print_outcomes(cumulative["outcomes"], "Summary")
    if not args.no_card_stats:
        _print_card_table(cumulative["card_stats"], "Summary")
        _print_value_summary(cumulative["value_summary"], "Summary turn value")
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
