"""Branch evaluation on spire-zero (the counterpart of whispering_earring/branch_jobs.py): what one out-of-combat choice costs.

ScoutRun    plays one run (Claws in combat, the compass deciding the rest, real HP) in the host's SL mode, and at every card reward
            (and every map choice with 2+ nodes) records a FORK POINT: the host's latest node checkpoint plus the steps since it, which
            replay exactly to that decision (spire-zero's own clone mechanism, checked by `run --check`). The scout's later route /
            event / rest choices are kept so a branch can repeat them.
BranchRun   starts a host at a fork point (`run --from-clone`), takes ONE option (a card, skip, or a map node), then plays on for
            `horizon` combats with the draw pile reshuffled at every combat's start (host command shuffle:N; the hand and the pile's
            contents are kept). Nothing after the shuffle is ever replayed. Repetition k of every option uses the same numbers
            (fork id, k, combat ordinal), so the options share what luck they can. Card branches repeat the scout's route / event / rest
            choices, buy nothing and skip later rewards, so only the pick differs; map branches let the policy play everything.

Same output as the old facility, so golden_compass/train.py --branch-dir reads it unchanged."""
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
import run_state as rs  # noqa: E402

from spire import claws_job as J, macro  # noqa: E402
from spire.actions import random_pick, to_action  # noqa: E402
from spire.host import HostError, SpireHost  # noqa: E402
from spire.macro_player import MacroPlayer  # noqa: E402
from spire.run_encode import act_floor, encode_run  # noqa: E402

FINAL_FLOOR = 18  # hazard_net.FINAL_FLOOR: a run past act 1's boss counts as censored here
MAX_COMBAT_TURNS = 20
FORCED = ("map", "event", "rest")  # decisions a card branch repeats from the scout instead of deciding


def _in_combat(d) -> bool:
    return d.surface == "combat" or (d.surface == "card_choice" and "combat" in d.obs["screen"])


def _offered(d) -> list[list]:
    """[[card_index, card_id, upgraded]] of a card_reward screen."""
    return [[i, c["id"].rstrip("+"), 1 if c.get("up", 0) > 0 or c["id"].endswith("+") else 0] for i, c in enumerate(d.obs["screen"]["cards"])]


def _sig(d) -> list:
    """What must match after a replay: the surface, floor, HP and (on a card reward) the cards on offer."""
    run = d.obs["run"]
    return [d.surface, act_floor(run), run["hp"], run["max_hp"], _offered(d) if d.surface == "card_reward" else None]


def _write_clone(path: str, checkpoint: str, trace: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"checkpoint": checkpoint, "trace": trace}, f)
    os.replace(tmp, path)


class ScoutRun:
    def __init__(self, main_id: str, asc: int, character: str, seed: int, clone_dir: str, max_floor: int = 17):
        self.main_id, self.asc, self.character, self.seed, self.clone_dir, self.max_floor = main_id, asc, character, seed, clone_dir, max_floor
        self.rng = random.Random(J.zlib_hash(main_id))

    def run(self) -> dict:
        J._load()
        J.seed_everything(J.zlib_hash(self.main_id))
        player = MacroPlayer(self.rng, self.asc)
        forks: list[dict] = []
        map_forks: list[dict] = []
        macro_log: list[str] = []
        outcome, error, floor = None, None, 0
        host = SpireHost.start(character=self.character, seed=self.seed, asc=self.asc, env="SL")
        try:
            d = host.decision
            ctx = None
            same, last, n = 0, None, 0
            while d is not None and n < 4000:
                n += 1
                run = d.obs.get("run") or {}
                if run:
                    floor = act_floor(run)
                    if run["act"] != 0 or floor > self.max_floor:
                        outcome = "stopped"
                        break
                if _in_combat(d):
                    if d.surface == "card_choice":
                        ch = d.obs["screen"]["choice"]
                        d = host.act(random_pick(ch["min"], ch["max"], len(ch["options"]), self.rng))
                    else:
                        obs = d.obs["screen"]["combat"]
                        _dec, options, idx, _p, _v = J.decide(obs, d.legal, J.run_context(run))
                        d = host.act(to_action(*options[idx], obs, d.legal))
                    continue
                sig = (d.surface, json.dumps(d.legal), run.get("hp"), run.get("gold"))
                same = same + 1 if sig == last else 0
                last = sig
                if same >= 25:
                    outcome = "stopped"
                    break
                dec = encode_run(d)
                if d.surface == "card_reward" and host.checkpoint is not None and dec is not None:
                    state = rs.encode_from_buffer(dec.arr, run["character"], self.asc)
                    forks.append({"index": len(forks), "floor": floor, "cards": _offered(d), "script_start": len(macro_log),
                                  "can_skip": "card_alt:Skip" in d.legal, "state": state.to_json(), "sig": _sig(d),
                                  "checkpoint": host.checkpoint, "trace": list(d.trace or [])})
                action = player.act(d, same)
                if d.surface == "map" and host.checkpoint is not None and dec is not None:
                    base = rs.encode_from_buffer(dec.arr, run["character"], self.asc)
                    # the nodes the buffer can represent (it keeps 3 children per node: an Ancient with 4 exits shows 3)
                    after = {f"{c}_{r}": st.to_json() for (c, r), st in rs.map_afterstates(dec.arr, base)}
                    legal_nodes = {f"{t['col']}_{t['row']}" for t in d.obs["screen"]["travelable"]}
                    after = {k: v for k, v in after.items() if k in legal_nodes}
                    if len(after) >= 2:
                        map_forks.append({"index": len(map_forks), "floor": floor, "sig": _sig(d), "nodes": list(after), "state": base.to_json(),
                                          "afterstates": after, "checkpoint": host.checkpoint, "trace": list(d.trace or [])})
                if d.surface in FORCED:
                    macro_log.append((d.surface, action))
                d = host.act(action)
            if outcome is None:
                outcome = ("won" if host.end and host.end.get("won") else "lost") if d is None else "stopped"
        except HostError as e:
            error = f"host: {e}"
        except Exception as e:  # noqa: BLE001
            import traceback
            where = " < ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(traceback.extract_tb(e.__traceback__)[-4:]))
            error = f"{type(e).__name__}: {e} [{where}]"
        finally:
            host.close()
        for f in forks:
            f["script"] = [list(x) for x in macro_log[f.pop("script_start"):]]
        for kind, lst in (("c", forks), ("m", map_forks)):
            for f in lst:
                f["clone_path"] = os.path.abspath(os.path.join(self.clone_dir, f"{self.main_id}_{kind}{f['index']:02d}.json"))
                _write_clone(f["clone_path"], f.pop("checkpoint"), f.pop("trace"))
        return {"main_id": self.main_id, "outcome": outcome, "error": error, "forks": forks, "map_forks": map_forks, "floor": floor}


class BranchRun:
    """One continuation: fork point `fork_id` (a card fork, or a map fork when `kind == "map"`), option, repetition k."""

    def __init__(self, fork_id: str, clone_path: str, sig: list, option, k: int, asc: int, horizon: int = 6, shuffle: bool = True,
                 script: list | None = None, kind: str = "card"):
        self.fork_id, self.clone_path, self.sig, self.option, self.k, self.asc = fork_id, clone_path, sig, option, k, asc
        self.horizon, self.shuffle, self.script, self.kind = horizon, shuffle, script or [], kind
        self.rng = random.Random(J.zlib_hash(f"{fork_id}|k{k}"))

    def run(self) -> dict:
        J._load()
        pair_key = f"{self.fork_id}|k{self.k}"  # the same for every option: the shared random numbers
        J.seed_everything(J.zlib_hash(pair_key))
        player = MacroPlayer(self.rng, self.asc)
        host = SpireHost.start(clone=self.clone_path, asc=self.asc, env="NoSL")
        combats: list[dict] = []
        taken: list = []
        outcome, error = None, None
        try:
            d = host.decision
            if d is None or _sig(d) != self.sig:
                raise HostError(f"fork point replayed to a different state: {_sig(d) if d else None} != {self.sig}")
            run0 = d.obs["run"]
            floor0, act0, max_hp0 = act_floor(run0), run0["act"], run0["max_hp"]
            d = host.act(self._take_option(d))
            queue = {s: [a for (sc, a) in self.script if sc == s] for s in FORCED}
            guard_same, last = 0, None
            while d is not None and outcome is None:
                run = d.obs["run"]
                if len(combats) >= self.horizon or run["act"] != act0:
                    outcome = "horizon"
                    break
                if _in_combat(d):
                    d, died, capped = self._combat(host, d, combats, pair_key)
                    if capped:
                        outcome = "censored"
                    elif died:
                        outcome = "lost"
                    continue
                sig = (d.surface, json.dumps(d.legal), run["hp"], run["gold"])
                guard_same = guard_same + 1 if sig == last else 0
                last = sig
                if guard_same >= 25:
                    outcome = "censored"
                    break
                forced = self._forced(d, queue)
                if forced is not None:
                    taken.append([d.surface, forced])
                    d = host.act(forced)
                    continue
                d = host.act(player.act(d, guard_same))
            if outcome is None:  # the host ended the run
                outcome = "won" if host.end and host.end.get("won") else "lost"
            last_run = d.obs["run"] if d is not None else None
            last_hp = last_run["hp"] if last_run else 0
            last_floor = act_floor(last_run) if last_run else (combats[-1]["floor"] if combats else floor0)
            act_now = last_run["act"] if last_run else act0
        except HostError as e:
            error = f"host: {e}"
        except Exception as e:  # noqa: BLE001
            import traceback
            where = " < ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(traceback.extract_tb(e.__traceback__)[-4:]))
            error = f"{type(e).__name__}: {e} [{where}]"
        finally:
            host.close()
        if error:
            return {"fork_id": self.fork_id, "option": self.option, "k": self.k, "error": error}
        died = outcome == "lost"
        cleared = outcome == "horizon" and act_now != act0  # past act 1's boss: the floor restarts and HP is restored
        lost = sum(c["hp_in"] - c["hp_out"] for c in combats)
        if died:
            hp_end, floors = 0, (combats[-1]["floor"] - floor0 if combats else 0)
        elif cleared:
            hp_end, floors = (combats[-1]["hp_out"] if combats else last_hp), FINAL_FLOOR - floor0
        else:
            hp_end, floors = last_hp, max(last_floor - floor0, 0)
        cum_lost, cum_died, hp_after, run_lost = [], [], [], 0
        cum_lost_abs = []
        for c in combats:
            run_lost += c["hp_in"] - c["hp_out"]
            cum_lost_abs.append(run_lost)
            cum_lost.append(run_lost / max(max_hp0, 1))
            hp_after.append(c["hp_out"] / max(max_hp0, 1))
            cum_died.append(float(c["hp_out"] <= 0))
        pad = self.horizon - len(combats)
        if pad > 0:  # a run that ended early (or never reached a combat) keeps its last value
            cum_lost += [cum_lost[-1] if cum_lost else 0.0] * pad
            cum_lost_abs += [cum_lost_abs[-1] if cum_lost_abs else 0] * pad
            hp_after += [hp_after[-1] if hp_after else last_hp / max(max_hp0, 1)] * pad
            cum_died += [cum_died[-1] if cum_died else float(died)] * pad
        metrics = {"outcome": "cleared_act" if cleared else outcome, "died": died, "combats": len(combats), "floors": floors,
                   "hp_frac_end": hp_end / max(max_hp0, 1), "hp_lost_share": lost / max(max_hp0, 1),
                   "hp_lost": lost, "max_hp0": max_hp0, "cum_lost_abs": cum_lost_abs,
                   "cum_lost": cum_lost, "cum_died": cum_died, "hp_after": hp_after, "ended_at": len(combats)}
        return {"fork_id": self.fork_id, "option": self.option, "k": self.k, "metrics": metrics, "taken": taken, "error": None}

    def _take_option(self, d) -> str:
        if self.kind == "map":
            col, row = (int(x) for x in self.option.split("_"))
            return f"map:{col},{row}"
        if self.option is None:
            return "card_alt:Skip"
        return f"card:{self.option}"

    def _forced(self, d, queue) -> str | None:
        """The action a card branch must take instead of the policy's: the scout's next route / event / rest choice (when still
        legal), nothing bought in a shop, a later card reward skipped. None: let the policy decide. Map branches force nothing."""
        if self.kind == "map":
            return None
        if d.surface == "card_reward" and "card_alt:Skip" in d.legal:
            macro.mark_skipped(d)
            return "card_alt:Skip"
        if d.surface == "shop":
            return "leave" if "leave" in d.legal else None
        if d.surface in queue and queue[d.surface]:
            action = queue[d.surface].pop(0)
            return action if action in d.legal else None
        return None

    def _combat(self, host: SpireHost, d, combats: list[dict], pair_key: str):
        """One combat at real HP with Claws, the draw pile shuffled at its start. -> (the decision after it, died, capped)."""
        run = d.obs["run"]
        ctx = J.run_context(run)
        hp_in, floor = d.obs["screen"]["combat"]["player"]["hp"] if d.surface == "combat" else run["hp"], act_floor(run)
        if self.shuffle and d.surface == "combat":
            host.shuffle(J.zlib_hash(f"{pair_key}|c{len(combats)}") & 0x7FFFFFFF)
        turns, last_round = 0, None
        while d is not None and _in_combat(d):
            if d.surface == "card_choice":
                ch = d.obs["screen"]["choice"]
                d = host.act(random_pick(ch["min"], ch["max"], len(ch["options"]), self.rng))
                continue
            obs = d.obs["screen"]["combat"]
            if obs["round"] != last_round:
                last_round, turns = obs["round"], turns + 1
                if turns > MAX_COMBAT_TURNS:
                    combats.append({"floor": floor, "hp_in": hp_in, "hp_out": obs["player"]["hp"], "turns": turns})
                    return d, False, True
            _dec, options, idx, _p, _v = J.decide(obs, d.legal, ctx)
            d = host.act(to_action(*options[idx], obs, d.legal))
        died = d is None
        combats.append({"floor": floor, "hp_in": hp_in, "hp_out": 0 if died else d.obs["run"]["hp"], "turns": turns})
        return d, died, False
