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
    def __init__(self, save: str | None, main_id: str, asc: int, raw_dir: str, character: str | None = None, seed: int = 1):
        self.save, self.main_id, self.asc, self.raw_dir, self.character, self.seed = save, main_id, asc, raw_dir, character, seed
        self.rng = random.Random(J.zlib_hash(main_id))

    def run(self) -> dict:
        J._load()
        J.seed_everything(J.zlib_hash(self.main_id))
        t0 = time.time()
        states: list[dict] = []
        combats: list[dict] = []
        outcome, error, capped = None, None, None
        floor = floor_offset = 0
        act, acts_seen = 0, 1
        host = SpireHost.start(save=self.save, character=self.character, seed=self.seed, asc=self.asc)
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
                    d = host.act(macro.choose(d, self.rng, repeat=same))
                    continue
                state = rs.encode_from_buffer(dec.arr, session.character, session.ascension)
                if d.surface == "event" and same >= 3 and "proceed" in d.legal:
                    name, kwargs, policy = "leave_room", {}, None
                else:
                    name, kwargs, policy, purpose = P.choose_action(dec, session, purpose)
                states.append({"decision": dec.decision, "action": name, "state": state.to_json()})
                if name == "select_map_node":
                    for key, entered in rs.map_afterstates(dec.arr, state):
                        if key == (kwargs["col"], kwargs["row"]):
                            states.append({"decision": "map_entered", "action": name, "state": entered.to_json()})
                d = host.act(to_host_action(name, kwargs, dec, d))
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
        os.makedirs(self.raw_dir, exist_ok=True)
        tmp = os.path.join(self.raw_dir, f".{self.main_id}.json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(self.raw_dir, f"{self.main_id}.json"))
        return data

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
