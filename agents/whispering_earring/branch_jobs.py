"""Jobs for facility_branch.py: card-reward branching with shuffled continuations.

ScoutJob   plays one run with the live policy (real HP, like RunJob) and records every card_reward decision as a
           fork point: a Checkpoint (native save at the last map node + the actions since) that reloads exactly there.
BranchJob  reloads a fork point (deterministic replay -- nothing has been shuffled yet), takes ONE option (a card or
           skip), then plays forward live for `horizon` combats with the draw pile shuffled at the start of every
           combat (Sts2BufferSession.randomize_draw_pile). The continuation is never saved or replayed, so the
           "cannot replay after a reshuffle" problem does not arise: only the out-of-combat outcome is kept.

Pairing: branch k of every option at one fork point seeds the policy sampling and each combat's shuffle from
(fork_id, k, combat ordinal) -- the same numbers for every option -- so the options share what luck they can.
"""
from __future__ import annotations

import os
import random
import zlib

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
import buffer_layout as bl
from buffer_env import decode_token
from checkpoint import Checkpoint
from jobs import (Job, JobResult, MAX_ACTIONS_PER_TURN, MAX_COMBAT_TURNS, _StuckGuard, _seed_everything, zlib_hash)
from policy import choose_action
import run_state as rs

FINAL_FLOOR = 18  # hazard_net.FINAL_FLOOR: a run past act 1's boss counts as censored here


FORCED = ("map_select", "event_choice", "rest_site")  # decisions a branch repeats from the scout instead of deciding


def offered_cards(decision) -> list[tuple[int, str, int]]:
    """[(card_index, card_id, upgraded)] of a card_reward decision."""
    return [(i, decode_token(bl.card_token(slot)), int(slot[bl.CARD_UPGRADED])) for i, slot in bl.reward_cards(decision.arr)]


def _sig(decision) -> list:
    """What must match after a reload: the decision, floor, HP and the cards on offer."""
    return [decision.decision, decision.floor, decision.player_hp, decision.player_max_hp,
            [list(c) for c in offered_cards(decision)] if decision.decision == "card_reward" else None]


class ScoutJob(Job):
    """Plays a run with the policy, taking a native save before every select_map_node, and returns the fork points:
    one per card_reward decision (not event_card_reward), as dicts with a Checkpoint reloading to just before the
    pick, the offered cards, whether skipping is allowed, the floor, and the encoded RunState."""
    role = "scout"

    def __init__(self, checkpoint: Checkpoint, save_dir: str, max_floor: int = 17):
        super().__init__(checkpoint)
        self.save_dir = save_dir
        self.max_floor = max_floor  # stop once the run is past this floor (act 1's boss is floor 17)

    def execute(self, session) -> JobResult:
        decision = self.begin(session)
        purpose = None
        save_path, save_index = None, 0
        forks: list[dict] = []
        map_forks: list[dict] = []  # one per map_select with 2+ reachable nodes (MapBranchJob)
        macro_log: list[dict] = []  # the route / event / rest choices in order, so a branch can be forced to repeat them
        guard = _StuckGuard()
        outcome = None
        while outcome is None:
            if decision.decision == "game_over":
                outcome = "won" if decision.victory else "lost"
                break
            if decision.act != 1 or decision.floor > self.max_floor or guard.tripped(decision):
                outcome = "stopped"
                break
            if decision.decision == "card_reward" and save_path is not None:
                cards = offered_cards(decision)
                state = rs.encode_from_buffer(decision.arr, session.character, session.ascension)
                forks.append({
                    "index": len(forks), "floor": decision.floor, "cards": cards, "script_start": len(macro_log),
                    "can_skip": bool(bl.can_skip_reward(decision.arr)), "state": state.to_json(),
                    "sig": _sig(decision),
                    "checkpoint": self.checkpoint.__class__(
                        main_id=self.main_id, character=self.checkpoint.character, seed=self.checkpoint.seed,
                        ascension=self.checkpoint.ascension, actions=list(session.action_log),
                        save_path=save_path, save_index=save_index),
                })
            name, kwargs, policy, purpose = choose_action(decision, session, purpose)
            if policy is not None:
                session.logger.attach_policy(policy)
            if decision.decision in FORCED:
                macro_log.append({"decision": decision.decision, "action": name, "args": dict(kwargs)})
            if name == "select_map_node":  # before the step: a save cannot rewind the room-entry RNG (see Checkpoint)
                save_index = len(session.action_log)
                save_path = os.path.abspath(os.path.join(self.save_dir, f"{self.main_id}_a{save_index:04d}.json"))
                os.makedirs(self.save_dir, exist_ok=True)
                session.save(save_path)
                nodes = rs.map_options(decision.arr)
                if decision.decision == "map_select" and len(nodes) >= 2:
                    base = rs.encode_from_buffer(decision.arr, session.character, session.ascension)
                    map_forks.append({
                        "index": len(map_forks), "floor": decision.floor, "sig": _sig(decision),
                        "nodes": [f"{c}_{r}" for c, r in nodes], "state": base.to_json(),
                        # what each option leads to, encoded now: the graph is not in the training file
                        "afterstates": {f"{c}_{r}": st.to_json() for (c, r), st in rs.map_afterstates(decision.arr, base)},
                        "checkpoint": self.checkpoint.__class__(
                            main_id=self.main_id, character=self.checkpoint.character, seed=self.checkpoint.seed,
                            ascension=self.checkpoint.ascension, actions=list(session.action_log),
                            save_path=save_path, save_index=save_index),
                    })
            decision = session.step(name, **kwargs)
        for fork in forks:  # what the scout chose after each fork point, for the branches to repeat
            fork["script"] = macro_log[fork.pop("script_start"):]
        forks, dropped = self._verified(forks)
        map_forks, map_dropped = self._verified(map_forks, "map_select")
        return self.result(session, outcome, forks=forks, map_forks=map_forks, end_floor=decision.floor,
                           dropped=dropped + map_dropped)

    @staticmethod
    def _verified(forks: list[dict], kind: str = "card_reward") -> tuple[list[dict], list[str]]:
        """Reloads every fork point in a second session and keeps only those that come back byte-identical to what the
        scout saw. (A fork point that reloads to a different state would make every branch from it fail or mislead.)"""
        from buffer_env import Sts2BufferSession
        kept, dropped = [], []
        check = Sts2BufferSession()
        try:
            for fork in forks:
                try:
                    d = fork["checkpoint"].reload(check)
                    same = d.decision == kind and _sig(d) == fork["sig"]
                    why = None if same else f"reloaded to {d.decision}: {_sig(d)} != {fork['sig']}"
                except Exception as e:
                    why = f"{type(e).__name__}: {str(e)[:80]}"
                (kept if why is None else dropped).append(fork if why is None else f"f{fork['index']:02d} floor {fork['floor']}: {why}")
        finally:
            check.close()
        return kept, dropped


class BranchJob(Job):
    """One continuation: fork point `fork_id`, option `option` (a card_index, or None to skip), repetition `k`."""
    role = "branch"

    def __init__(self, checkpoint: Checkpoint, fork_id: str, option: int | None, k: int, horizon: int = 6,
                 shuffle: bool = True, script: list[dict] | None = None):
        super().__init__(checkpoint)
        self.script = script or []  # the scout's route / event / rest choices after the fork, repeated in order
        self.fork_id = fork_id
        self.option = option
        self.k = k
        self.horizon = horizon
        self.shuffle = shuffle

    def uid(self) -> str:
        return f"{self.fork_id}_o{'skip' if self.option is None else self.option}_k{self.k:03d}"

    def execute(self, session) -> JobResult:
        pair_key = f"{self.fork_id}|k{self.k}"  # the same for every option: the shared random numbers
        _seed_everything(zlib_hash(pair_key))
        decision = self.begin(session)
        floor0, act0, max_hp0 = decision.floor, decision.act, decision.player_max_hp
        decision = self._take_option(session, decision)

        purpose = None
        guard = _StuckGuard()
        combats: list[dict] = []
        outcome = None
        taken: list = []  # the forced choices actually made, to check every option followed the same path
        queue = {d: [e for e in self.script if e["decision"] == d] for d in FORCED}
        while outcome is None:
            if decision.decision == "game_over":
                outcome = "won" if decision.victory else "lost"
                break
            if len(combats) >= self.horizon or decision.act != act0:
                outcome = "horizon"
                break
            if decision.decision == "combat_play":
                decision, died, capped = self._combat(session, decision, combats, guard, pair_key)
                if capped:
                    outcome = "censored"
                elif died:
                    outcome = "lost"
                continue
            if guard.tripped(decision):
                outcome = "censored"
                break
            forced = self._forced(decision, queue)
            if forced is not None:
                taken.append([decision.decision, forced[0], forced[1]])
                decision = session.step(forced[0], **forced[1])
                continue
            name, kwargs, policy, purpose = choose_action(decision, session, purpose)
            if policy is not None:
                session.logger.attach_policy(policy)
            decision = session.step(name, **kwargs)

        died = outcome == "lost"
        cleared = outcome == "horizon" and decision.act != act0  # past act 1's boss: the buffer's floor restarts and HP is restored
        lost = sum(c["hp_in"] - c["hp_out"] for c in combats)
        if died:
            hp_end, floors = 0, (combats[-1]["floor"] - floor0 if combats else 0)
        elif cleared:
            hp_end, floors = (combats[-1]["hp_out"] if combats else decision.player_hp), FINAL_FLOOR - floor0
        else:
            hp_end, floors = decision.player_hp, max(decision.floor - floor0, 0)
        # Cumulative after every combat, so one long run also answers every shorter horizon (padded with the final
        # value when the run ended earlier -- died, cleared the act or was cut off; `ended_at` says when).
        cum_lost, cum_died, hp_after = [], [], []
        run_lost = 0
        for c in combats:
            run_lost += c["hp_in"] - c["hp_out"]
            cum_lost.append(run_lost / max(max_hp0, 1))
            hp_after.append(c["hp_out"] / max(max_hp0, 1))
            cum_died.append(float(c["hp_out"] <= 0))
        pad = self.horizon - len(combats)
        if pad > 0:  # a run that ended early (or never reached a combat) keeps its last value
            cum_lost += [cum_lost[-1] if cum_lost else 0.0] * pad
            hp_after += [hp_after[-1] if hp_after else decision.player_hp / max(max_hp0, 1)] * pad
            cum_died += [cum_died[-1] if cum_died else float(died)] * pad
        metrics = {
            "outcome": "cleared_act" if cleared else outcome, "died": died, "combats": len(combats), "floors": floors,
            "hp_frac_end": hp_end / max(max_hp0, 1),
            "hp_lost_share": lost / max(max_hp0, 1),  # HP the combats cost, as a share of max HP (death counts in full)
            "cum_lost": cum_lost, "cum_died": cum_died, "hp_after": hp_after, "ended_at": len(combats),
        }
        return self.result(session, outcome, fork_id=self.fork_id, option=self.option, k=self.k, metrics=metrics, combats=combats, taken=taken)

    def _take_option(self, session, decision):
        assert decision.decision == "card_reward", f"fork point reloaded to {decision.decision!r}"
        return session.step("skip_card_reward") if self.option is None else session.step("select_card_reward", card_index=self.option)

    @staticmethod
    def _forced(decision, queue) -> tuple[str, dict] | None:
        """The action a branch must take instead of the policy's, or None to let the policy decide: the scout's next
        route / event / rest choice (map nodes only if still reachable), no purchases in a shop, and no new card pick
        (a later card reward is skipped, so a branch can never fork again)."""
        dec = decision.decision
        if dec == "card_reward" and bl.can_skip_reward(decision.arr):
            return "skip_card_reward", {}
        if dec == "shop":
            return "leave_room", {}
        if dec in queue and queue[dec]:
            entry = queue[dec].pop(0)
            if entry["action"] == "select_map_node" and (entry["args"]["col"], entry["args"]["row"]) not in rs.map_options(decision.arr):
                return None
            return entry["action"], entry["args"]
        return None

    def _combat(self, session, decision, combats, guard, pair_key):
        """Plays one combat to its end, with the draw pile shuffled at its start.
        -> (decision after it, died, capped by the stuck guard)"""
        if self.shuffle:
            session.randomize_draw_pile(decision, seed=zlib_hash(f"{pair_key}|c{len(combats)}") & 0x7FFFFFFF)
        hp_in, floor = decision.player_hp, decision.floor
        turns = actions_this_turn = 0
        capped = None
        while decision.decision in ("combat_play", "card_select"):
            capped = guard.tripped(decision)
            if capped is None and actions_this_turn >= MAX_ACTIONS_PER_TURN:
                capped = "too many actions in one turn"
            if capped:
                break
            name, kwargs, policy, _p = choose_action(decision, session)
            if policy is not None:
                session.logger.attach_policy(policy)
            try:
                decision = session.step(name, **kwargs)
            except Exception as e:  # say where it happened: which combat, turn and action
                raise RuntimeError(f"{type(e).__name__}: {e} [combat {len(combats)} floor {floor} turn {turns} "
                                   f"action {name} {kwargs} shuffled={self.shuffle} actions_this_turn {actions_this_turn}]") from e
            actions_this_turn += 1
            if name == "end_turn":
                turns += 1
                actions_this_turn = 0
                if turns >= MAX_COMBAT_TURNS and decision.decision == "combat_play":
                    capped = "combat stalled at the turn cap"
                    break
        died = decision.decision == "game_over"
        combats.append({"floor": floor, "hp_in": hp_in, "hp_out": 0 if died else decision.player_hp, "turns": turns})
        return decision, died, capped


class MapBranchJob(BranchJob):
    """One continuation of a map fork: reloads the scout's save at a map_select, enters node `option` ("col_row"), then
    lets the live policy play everything (route, rooms, rewards, shops) for `horizon` combats, the draw pile shuffled at
    each combat's start. What it measures is how that node does under the policy that will be deployed, so nothing
    is forced after the first step. Paired like the card branches: repetition k seeds every option alike."""

    def uid(self) -> str:
        return f"{self.fork_id}_o{self.option}_k{self.k:03d}"

    def _take_option(self, session, decision):
        assert decision.decision == "map_select", f"map fork point reloaded to {decision.decision!r}"
        col, row = (int(x) for x in self.option.split("_"))
        assert (col, row) in rs.map_options(decision.arr), f"node {self.option} not reachable"
        return session.step("select_map_node", col=col, row=row)

    @staticmethod
    def _forced(decision, queue) -> None:
        return None
