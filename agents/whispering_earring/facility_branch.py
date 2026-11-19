"""Card-reward branching: for each fork point (a card_reward decision of a scout run) play K continuations of EVERY
option (each offered card, and skip) for `horizon` combats, with the draw pile reshuffled at each combat's start,
and average the outcome per option. One file per fork point in raw_dir: the encoded RunState, the options, every
continuation's metrics and each option's mean.

    python facility_branch.py --trials 10 --forks-per-scout 2 --k 4 --horizon 3 --branch-workers 8 --raw-dir <dir>

--trials is the number of scout runs. Nothing here is replayed after a shuffle (see branch_jobs.py)."""
from __future__ import annotations

import json
import os
import random
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)

from branch_jobs import BranchJob, MapBranchJob, ScoutJob
from checkpoint import Checkpoint
from facility_common import FacilityConfig, NewFacility, add_common_cli_args
from utils import new_run

METRICS = ("died", "hp_lost_share", "hp_frac_end", "floors", "combats")


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


class FacilityBranch(NewFacility):
    def __init__(self, cfg: FacilityConfig, forks_per_scout: int, k: int | None, horizon: int, shuffle: bool, save_dir: str,
                 k_per_unique: float = 2.0, k_min: int = 6, k_max: int = 20, map_forks_per_scout: int = 5, map_k: int = 8):
        super().__init__(cfg)
        self.map_forks_per_scout, self.map_k = map_forks_per_scout, map_k
        self.forks_per_scout, self.k, self.horizon, self.shuffle = forks_per_scout, k, horizon, shuffle
        self.k_per_unique, self.k_min, self.k_max = k_per_unique, k_min, k_max
        self.save_dir = save_dir
        self._scouts_started = 0
        self._scouts_open: set[str] = set()
        self._branches_open = 0
        self._forks: dict[str, dict] = {}  # fork_id -> {"meta": ..., "results": {option_key: [metrics]}, "left": n}
        self._done_forks = 0
        self._errors = 0
        self._dropped = 0

    def _k_for(self, fork: dict) -> int:
        """Continuations per option at one fork point. A fixed --k wins; otherwise it scales with the number of distinct
        cards in the deck once the pick is made (an early deck of Strike / Defend / Bash gives almost the same hands
        however it is shuffled, so it needs few samples), clamped to [k_min, k_max]. The same K for every option of a
        fork, so the options stay paired."""
        if self.k:
            return self.k
        unique = len({c[0] for c in fork["state"]["deck"]}) + 1  # + the card being taken
        return max(self.k_min, min(self.k_max, round(self.k_per_unique * unique)))

    def _more_scouts(self) -> bool:
        return self._scouts_started < (self.cfg.n_main_trials or 0)

    def _submit_scouts(self) -> None:
        while self._more_scouts() and len(self._scouts_open) < self.cfg.n_branch_workers:
            self._scouts_started += 1
            character = random.choice(self.cfg.characters)
            seed, main_id = new_run(character, self.cfg.seeds, self._scouts_started - 1)
            job = ScoutJob(Checkpoint.fresh(main_id, character, seed, self.cfg.ascension), self.save_dir)
            self._scouts_open.add(job.uid())
            self.job_queue.put(job)

    def _on_scout(self, uid: str, result) -> None:
        self._scouts_open.discard(uid)
        forks = result.auxiliary_data.get("forks") or []
        self._submit_map_forks(result)
        for why in result.auxiliary_data.get("dropped") or []:
            self.live.scroll(f"dropped fork {result.main_id}_{why}")
            self._dropped += 1
        if result.auxiliary_data.get("error") or not forks:
            self._errors += bool(result.auxiliary_data.get("error"))
            return
        for fork in random.sample(forks, min(self.forks_per_scout, len(forks))):
            fork_id = f"{result.main_id}_f{fork['index']:02d}"
            options: list[int | None] = [i for i, _cid, _up in fork["cards"]] + ([None] if fork["can_skip"] else [])
            k = self._k_for(fork)
            meta = {key: fork[key] for key in ("floor", "cards", "can_skip", "state")}
            meta.update(fork_id=fork_id, k=k, horizon=self.horizon, shuffle=self.shuffle, main_id=result.main_id)
            self._forks[fork_id] = {"meta": meta, "results": {}, "left": len(options) * k}
            for opt in options:
                for rep in range(k):
                    self.job_queue.put(BranchJob(fork["checkpoint"], fork_id, opt, rep, self.horizon, self.shuffle, fork["script"]))
                    self._branches_open += 1

    def _submit_map_forks(self, result) -> None:
        """Route branching: a random few of the scout's map decisions, every reachable node played `map_k` times."""
        map_forks = result.auxiliary_data.get("map_forks") or []
        for fork in random.sample(map_forks, min(self.map_forks_per_scout, len(map_forks))):
            fork_id = f"{result.main_id}_m{fork['index']:02d}"
            meta = {key: fork[key] for key in ("floor", "nodes", "state", "afterstates")}
            meta.update(kind="map", fork_id=fork_id, k=self.map_k, horizon=self.horizon, shuffle=self.shuffle, main_id=result.main_id)
            self._forks[fork_id] = {"meta": meta, "results": {}, "left": len(fork["nodes"]) * self.map_k}
            for node in fork["nodes"]:
                for rep in range(self.map_k):
                    self.job_queue.put(MapBranchJob(fork["checkpoint"], fork_id, node, rep, self.horizon, self.shuffle))
                    self._branches_open += 1

    def _on_branch(self, uid: str, result) -> None:
        self._branches_open -= 1
        aux = result.auxiliary_data
        if aux.get("error"):
            self._errors += 1
            fork = self._forks[uid.rsplit("_o", 1)[0]]  # uid = {fork_id}_o{option}_k{rep}
        else:
            fork = self._forks[aux["fork_id"]]
            key = "skip" if aux["option"] is None else str(aux["option"])
            fork["results"].setdefault(key, []).append({"k": aux["k"], **aux["metrics"], "taken": aux.get("taken")})
        fork["left"] -= 1
        if fork["left"] == 0 and fork["results"]:
            self._write(fork)

    def _write(self, fork: dict) -> None:
        options = {}
        for key, rows in fork["results"].items():
            options[key] = {"n": len(rows), "runs": rows, **{f"mean_{m}": _mean([float(r[m]) for r in rows]) for m in METRICS}}
        out = {**fork["meta"], "options": options}
        os.makedirs(self.cfg.raw_dir, exist_ok=True)
        with open(os.path.join(self.cfg.raw_dir, f"{fork['meta']['fork_id']}.json"), "w") as f:
            json.dump(out, f)
        self._done_forks += 1

    def run(self) -> None:
        self.start()
        try:
            self._submit_scouts()
            while self._scouts_open or self._branches_open or self._more_scouts():
                try:
                    uid, result = self.result_queue.get(timeout=self.cfg.stall_timeout_s)
                except Empty:
                    self.live.scroll(f"No result for {self.cfg.stall_timeout_s:.0f}s -- treating the pool as wedged.")
                    break
                if uid in self._scouts_open:
                    self._on_scout(uid, result)
                    self._submit_scouts()
                else:
                    self._on_branch(uid, result)
                self.status(f"scouts {self._scouts_started}/{self.cfg.n_main_trials} | forks done {self._done_forks}/{len(self._forks)} "
                            f"| branches open {self._branches_open} | errors {self._errors} | dropped forks {self._dropped}")
        except KeyboardInterrupt:
            self.live.scroll("Stopping facility...")
        finally:
            self.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Card-reward branching with shuffled continuations")
    add_common_cli_args(parser)
    parser.add_argument("--forks-per-scout", type=int, default=99, help="fork points sampled from each scout run (default: all of act 1's card rewards)")
    parser.add_argument("--k", type=int, default=None, help="continuations per option; default: scaled by the deck's unique cards")
    parser.add_argument("--k-per-unique", type=float, default=2.0, help="K = this x (unique cards after the pick), clamped")
    parser.add_argument("--k-min", type=int, default=6)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--map-forks-per-scout", type=int, default=5, help="map decisions branched per scout run (0 = none)")
    parser.add_argument("--map-k", type=int, default=8, help="continuations per reachable node at a map fork")
    parser.add_argument("--horizon", type=int, default=6, help="most combats each continuation plays (metrics are cumulative per combat)")
    parser.add_argument("--no-shuffle", action="store_true", help="do not reshuffle the draw pile (control)")
    parser.add_argument("--save-dir", default=None, help="native saves of the scouts (default: <raw-dir>/saves)")
    args = parser.parse_args()
    FacilityBranch(
        FacilityConfig(characters=args.characters, ascension=args.ascension, seeds=args.seeds,
                       n_branch_workers=args.branch_workers, n_main_trials=args.trials,
                       raw_dir=args.raw_dir, verbose=args.verbose),
        args.forks_per_scout, args.k, args.horizon, not args.no_shuffle,
        args.save_dir or os.path.join(args.raw_dir, "saves"),
        args.k_per_unique, args.k_min, args.k_max, args.map_forks_per_scout, args.map_k,
    ).run()
