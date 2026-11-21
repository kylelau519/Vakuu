"""One Golden Compass data job on spire-zero: a whole run played at REAL HP (nothing healed) with Claws in combat and the compass
deciding map, card rewards, rests, shops and card prompts, logging the RunState before every macro decision for the hazard net to
train on. The counterpart of whispering_earring's RunJob; writes raw_dir/{main_id}.json in the format golden_compass/materialize.py
reads (outcome, end_act/end_floor, states, combats)."""
from __future__ import annotations

import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "whispering_earring"), os.path.join(_HERE, "..", "golden_compass")):
    if p not in sys.path:
        sys.path.insert(0, p)
import sts2_path  # noqa: F401,E402
import policy as P  # noqa: E402
import run_state as rs  # noqa: E402

from spire import claws_job as J, macro  # noqa: E402
from spire import tables  # noqa: E402,F401
from spire.actions import random_pick, to_action  # noqa: E402
from spire.branch_job import FORCED, _offered, _sig, _write_clone  # noqa: E402
from spire.host import HostError, SpireHost  # noqa: E402
from spire.run_encode import act_floor, encode_run, to_host_action  # noqa: E402

MAX_COMBAT_TURNS = 20
MAX_RUN_DECISIONS = 6000
MAX_SAME_DECISION = 25


class _Session:
    """What policy.choose_action reads off a session."""

    def __init__(self, character: str, ascension: int):
        self.character, self.ascension = character, ascension


class CompassRun:
    def __init__(self, save: str | None, main_id: str, asc: int, raw_dir: str, character: str | None = None, seed: int = 1,
                 fork_dir: str | None = None, card_forks: int = 0, map_forks: int = 0):
        """`fork_dir` with card_forks / map_forks > 0: the run is hosted in SL mode and card_forks card rewards and map_forks route
        choices of it (picked at random when the run is over) are kept as FORK POINTS for facility_branch.py --points: fork_dir/points/
        {main_id}.json plus a clone file each in fork_dir/clones. Without them the run is hosted in NoSL (faster, nothing to fork)."""
        self.save, self.main_id, self.asc, self.raw_dir, self.character, self.seed = save, main_id, asc, raw_dir, character, seed
        self.fork_dir, self.card_forks, self.map_forks = fork_dir, card_forks, map_forks
        self.rng = random.Random(J.zlib_hash(main_id))

    def run(self) -> dict:
        J._load()
        J.seed_everything(J.zlib_hash(self.main_id))
        macro.reset()
        t0 = time.time()
        states: list[dict] = []
        combats: list[dict] = []
        outcome, error, capped = None, None, None
        floor = floor_offset = 0
        act, acts_seen = 0, 1
        forking = bool(self.fork_dir) and (self.card_forks > 0 or self.map_forks > 0)
        fork_card: list[dict] = []
        fork_map: list[dict] = []
        macro_log: list[tuple] = []
        host = SpireHost.start(save=self.save, character=self.character, seed=self.seed, asc=self.asc, env="SL" if forking else "NoSL")
        try:
            d = host.decision
            ch = d.obs["run"]["character"]
            session = _Session(ch, self.asc)
            act = d.obs["run"]["act"]
            purpose = None
            same, last_sig, n = 0, None, 0
            while d is not None and n < MAX_RUN_DECISIONS:
                n += 1
                run = d.obs.get("run") or {}
                if run:
                    if run["act"] != act:  # the first decision of a new act: floor still holds the old act's last
                        floor_offset += floor
                        act, acts_seen = run["act"], acts_seen + 1
                    floor = act_floor(run)
                if d.surface == "combat" or (d.surface == "card_choice" and "combat" in d.obs["screen"]):
                    d, stalled = self._combat(host, d, combats)
                    if stalled:
                        outcome, capped = "censored", "a combat hit the turn cap"
                        break
                    continue
                sig = (d.surface, json.dumps(d.legal), run.get("hp"), run.get("gold"))
                same = same + 1 if sig == last_sig else 0
                last_sig = sig
                if same >= MAX_SAME_DECISION:
                    outcome, capped = "censored", f"{d.surface} unchanged for {same} actions"
                    break
                dec = encode_run(d, ch)
                if dec is None:  # a screen the agents never decide: rewards, treasure, proceed
                    action = macro.choose(d, self.rng, repeat=same)
                    if d.surface in FORCED:
                        macro_log.append((d.surface, action))
                    d = host.act(action)
                    continue
                state = rs.encode_from_buffer(dec.arr, session.character, session.ascension, dec.remaining)
                if d.surface == "event" and same >= 3 and "proceed" in d.legal:
                    name, kwargs, policy = "leave_room", {}, None
                else:
                    name, kwargs, policy, purpose = P.choose_action(dec, session, purpose)
                states.append({"decision": dec.decision, "action": name, "state": state.to_json()})
                faced = [host.faced.act, sorted(host.faced.seen)]
                if forking and d.surface == "card_reward" and host.checkpoint is not None:
                    fork_card.append({"index": len(fork_card), "floor": floor, "cards": _offered(d), "script_start": len(macro_log),
                                      "can_skip": "card_alt:Skip" in d.legal, "state": state.to_json(), "sig": _sig(d),
                                      "checkpoint": host.checkpoint, "trace": list(d.trace or []), "faced": faced})
                if name == "select_map_node":
                    afters = rs.map_afterstates(dec.arr, state)
                    for key, entered in afters:
                        if key == (kwargs["col"], kwargs["row"]):
                            states.append({"decision": "map_entered", "action": name, "state": entered.to_json()})
                    if forking and host.checkpoint is not None:
                        legal_nodes = {f"{t['col']}_{t['row']}" for t in d.obs["screen"]["travelable"]}
                        after = {f"{c}_{r}": st.to_json() for (c, r), st in afters if f"{c}_{r}" in legal_nodes}
                        if len(after) >= 2:
                            fork_map.append({"index": len(fork_map), "floor": floor, "sig": _sig(d), "nodes": list(after), "state": state.to_json(),
                                             "afterstates": after, "checkpoint": host.checkpoint, "trace": list(d.trace or []), "faced": faced})
                action = to_host_action(name, kwargs, dec, d)
                if d.surface in FORCED:
                    macro_log.append((d.surface, action))
                if d.surface == "card_reward" and action.startswith("card_alt:"):
                    macro.mark_skipped(d)
                d = host.act(action)
            if outcome is None:
                if d is None:
                    outcome = "won" if host.end and host.end.get("won") else "lost"
                else:
                    outcome, capped = "censored", f"{n} decisions in one run"
        except HostError as e:
            error = f"host: {e}"
        except Exception as e:  # noqa: BLE001
            import traceback
            where = " < ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(traceback.extract_tb(e.__traceback__)[-4:]))
            error = f"{type(e).__name__}: {e} [{where}]"
        finally:
            host.close()
        data = {
            "main_id": self.main_id, "role": "run", "character": self.character, "seed": self.seed, "ascension": self.asc,
            "outcome": outcome or "error", "error": error, "end_floor": floor, "end_act": act + 1, "acts_reached": acts_seen,
            "total_floor": floor_offset + floor, "states": states, "combats": combats, "seconds": round(time.time() - t0, 1),
        }
        if capped:
            data["capped"] = capped
        if forking and (fork_card or fork_map):
            self._write_forks(fork_card, fork_map, macro_log)
        os.makedirs(self.raw_dir, exist_ok=True)
        tmp = os.path.join(self.raw_dir, f".{self.main_id}.json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(self.raw_dir, f"{self.main_id}.json"))
        return data

    def _write_forks(self, fork_card: list[dict], fork_map: list[dict], macro_log: list) -> None:
        """Keeps card_forks / map_forks of the run's fork points (random picks), with the clone file of each."""
        rng = random.Random(J.zlib_hash(f"{self.main_id}|forks"))
        cards = rng.sample(fork_card, min(self.card_forks, len(fork_card)))
        maps = rng.sample(fork_map, min(self.map_forks, len(fork_map)))
        for f in cards:
            f["script"] = [list(x) for x in macro_log[f.pop("script_start"):]]
        for kind, lst in (("c", cards), ("m", maps)):
            for f in lst:
                f["clone_path"] = os.path.abspath(os.path.join(self.fork_dir, "clones", f"{self.main_id}_{kind}{f['index']:02d}.json"))
                _write_clone(f["clone_path"], f.pop("checkpoint"), f.pop("trace"))
        os.makedirs(os.path.join(self.fork_dir, "points"), exist_ok=True)
        path = os.path.join(self.fork_dir, "points", f"{self.main_id}.json")
        with open(path + ".tmp", "w") as fh:
            json.dump({"main_id": self.main_id, "forks": cards, "map_forks": maps}, fh)
        os.replace(path + ".tmp", path)

    def _combat(self, host: SpireHost, d, combats: list[dict]):
        """One combat at real HP with Claws. -> (the first decision after it or None, stalled)."""
        run = d.obs["run"]
        obs0 = d.obs["screen"]["combat"]
        ctx = J.run_context(run)
        hp_in, max_hp = obs0["player"]["hp"], obs0["player"]["max_hp"]
        floor, act = act_floor(run), run["act"]
        boss = str(obs0.get("encounter", "")).endswith("_BOSS")
        turns, last_round = 0, None
        while d is not None and (d.surface == "combat" or (d.surface == "card_choice" and "combat" in d.obs["screen"])):
            if d.surface == "card_choice":
                ch = d.obs["screen"]["choice"]
                d = host.act(random_pick(ch["min"], ch["max"], len(ch["options"]), self.rng))
                continue
            obs = d.obs["screen"]["combat"]
            if obs["round"] != last_round:
                last_round, turns = obs["round"], turns + 1
                if turns > MAX_COMBAT_TURNS:
                    host.abort("stalled combat")
                    combats.append({"floor": floor, "hp_in": hp_in, "hp_out": hp_in, "max_hp": max_hp, "turns": turns, "boss": boss})
                    return None, True
            _dec, options, idx, _p, _v = J.decide(obs, d.legal, ctx)
            d = host.act(to_action(*options[idx], obs, d.legal))
        hp_out = d.obs["run"]["hp"] if d is not None else 0
        combats.append({"floor": floor, "hp_in": hp_in, "hp_out": hp_out, "max_hp": max_hp, "turns": turns, "boss": boss})
        return d, False
