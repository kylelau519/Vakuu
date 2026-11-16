from __future__ import annotations

import os
import random
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
import buffer_layout as bl
from buffer_env import Sts2BufferSession, BufferDecision
from checkpoint import Checkpoint
from policy import choose_action, choose_generic_action, get_claws_net
import run_state as rs  # golden_compass/run_state.py; policy.py puts its directory on sys.path
from recorder import CompactRecorder

MAX_COMBAT_TURNS = 20  # a combat that outlasts this is "stalled": HP loss alone gives no reason to finish one
# RunJob's backstops against a run that never ends (MAX_COMBAT_TURNS only counts end_turns, so it can't see
# any of these): actions within one turn, the same decision coming back unchanged, and a run's total length.
MAX_ACTIONS_PER_TURN = 200
MAX_SAME_DECISION = 25
MAX_RUN_ACTIONS = 4000


@dataclass
class JobResult:
    main_id: str
    outcome: str
    action_log: list[dict]  # full history since StartRun, including whatever the checkpoint skipped replaying
    record: list[dict] = field(default_factory=list)  # action_log + the job's own policy per step
    auxiliary_data: dict = field(default_factory=dict)


class Job(ABC):
    """A unit of work a worker runs against its session. Every job starts from a Checkpoint --
    fresh start, full replay, or native-save fast-load are all the same call (see begin())."""
    role: str = "main"  # CompactRecorder meta "role"

    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint
        self.recorder: CompactRecorder | None = None

    @property
    def main_id(self) -> str:
        return self.checkpoint.main_id

    def uid(self) -> str:
        return self.main_id

    def begin(self, session: Sts2BufferSession) -> BufferDecision:
        """Attaches a fresh recorder to the session and brings it to this job's checkpoint."""
        cp = self.checkpoint
        self.recorder = CompactRecorder(
            {
                "character": cp.character, "seed": cp.seed, "ascension": cp.ascension,
                "main_id": cp.main_id, "role": self.role,
            },
            start_step=len(cp.actions),
        )
        session.logger = self.recorder
        return cp.reload(session)

    def result(self, session, outcome: str, **auxiliary) -> JobResult:
        log = list(session.action_log)
        return JobResult(
            main_id=self.main_id, outcome=outcome, action_log=log,
            record=self.recorder.annotated_actions(log), auxiliary_data=auxiliary,
        )

    # Must not return None; the worker reports a raised exception as an "error" result.
    @abstractmethod
    def execute(self, session: Sts2BufferSession) -> JobResult:
        pass


# --- combat helpers shared by MainlineJob and ForkJob -------------------------------------------

def _play_turn(session: Sts2BufferSession, decision: BufferDecision) -> BufferDecision:
    """Plays from a turn's start until the next turn starts (a combat_play after end_turn) or the
    combat ends (card_reward on a win, game_over on a loss)."""
    ended_turn = False
    while decision.decision in ("combat_play", "card_select"):
        name, kwargs, policy, _purpose = choose_action(decision, session)
        if policy is not None:
            session.logger.attach_policy(policy)
        decision = session.step(name, **kwargs)
        if name == "end_turn":
            ended_turn = True
        if ended_turn and decision.decision == "combat_play":
            break
    assert decision.decision in ("combat_play", "card_reward", "game_over"), (
        f"turn ended on unexpected decision {decision.decision!r} -- would silently score "
        "this turn wrong if treated as resolved"
    )
    return decision


def _value(decision: BufferDecision) -> float | None:
    """The claws net's V(state), or None without a CLAWS_CHECKPOINT."""
    net = get_claws_net()
    if net is None:
        return None
    import torch
    with torch.no_grad():
        _, _, value = net.forward(decision)
    return float(value.item())


def _with_returns(turns: list[dict]) -> list[dict]:
    """Adds `g` to every turn: the HP actually lost from that turn's start to the end of the combat."""
    g = 0
    for turn in reversed(turns):
        g += turn["loss"]
        turn["g"] = g
    return turns


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
    except ImportError:
        pass


# --- jobs ---------------------------------------------------------------------------------------

class TrialJob(Job):
    """A whole run played with the generic policy (uniform-random combat; heuristics for map and
    cards), no HP tricks. With raw_dir it saves raw_dir/{main_id}.json in the trial format
    materialize.py reads. With save_dir it also takes a native save at every map_select and returns
    each as a Checkpoint in auxiliary_data["checkpoints"] -- for pipeline checks and resume tests."""
    role = "main"

    def __init__(self, checkpoint: Checkpoint, raw_dir: str | None = None, save_dir: str | None = None):
        super().__init__(checkpoint)
        self.raw_dir = raw_dir
        self.save_dir = save_dir

    def execute(self, session: Sts2BufferSession) -> JobResult:
        decision = self.begin(session)
        purpose = None
        checkpoints: list[Checkpoint] = []
        while decision.decision != "game_over":
            if self.save_dir is not None and decision.decision == "map_select":
                # Named by action index, so every save is unique -- a later save can never
                # overwrite one a job is still waiting to load.
                save_path = os.path.join(self.save_dir, f"{self.main_id}_a{len(session.action_log):04d}.json")
                os.makedirs(self.save_dir, exist_ok=True)
                session.save(save_path)
                checkpoints.append(self.checkpoint.saved_from(session, save_path))
            name, kwargs, purpose, policy = choose_generic_action(decision, purpose)
            if policy is not None:
                session.logger.attach_policy(policy)
            decision = session.step(name, **kwargs)

        outcome = "won" if decision.victory else "lost"
        if self.raw_dir is not None:
            self.recorder.save(os.path.join(self.raw_dir, f"{self.main_id}.json"), session.action_log, outcome=outcome)
        return self.result(session, outcome, checkpoints=checkpoints)


class MainlineJob(Job):
    """A whole run played with the policy, with every combat measured from FULL HP: at the combat's
    first decision it records the HP the run really has, heals to full, plays the combat, then
    deducts the HP the combat cost from that recorded HP. So V (expected HP loss) never has to
    absorb "and then you die"; the run dies only if the deduction takes it to 0 (or the player dies
    at full HP). Both HP changes are set_hp entries in the action log, so every reload reproduces them.

    Takes a native save before every select_map_node, so each combat turn's checkpoint is
    save + replay of that one combat. Returns the per-combat records (auxiliary_data["combats"]) a
    facility needs to fork every turn; with raw_dir it also writes the trial file."""
    role = "main"

    def __init__(self, checkpoint: Checkpoint, save_dir: str, raw_dir: str | None = None,
                 keep_room_checkpoints: bool = False):
        super().__init__(checkpoint)
        self.save_dir = save_dir
        self.raw_dir = raw_dir
        self.keep_room_checkpoints = keep_room_checkpoints  # debug scripts: a Checkpoint per room entry

    def execute(self, session: Sts2BufferSession) -> JobResult:
        decision = self.begin(session)
        purpose = None
        save_path, save_index = self.checkpoint.save_path, self.checkpoint.save_index
        combats: list[dict] = []
        room_checkpoints: list[Checkpoint] = []
        outcome = None

        while outcome is None:
            if decision.decision == "game_over":
                outcome = "won" if decision.victory else "lost"
            elif decision.decision == "combat_play":
                decision, outcome = self._play_combat(session, decision, combats, save_path, save_index)
            else:
                name, kwargs, policy, purpose = choose_action(decision, session, purpose)
                if policy is not None:
                    session.logger.attach_policy(policy)
                if name == "select_map_node":
                    # Before the step, not after: encounter selection consumes RNG at room entry and a
                    # save can't rewind it (see Checkpoint). Named by action index, so each is unique.
                    save_index = len(session.action_log)
                    save_path = os.path.join(self.save_dir, f"{self.main_id}_a{save_index:04d}.json")
                    os.makedirs(self.save_dir, exist_ok=True)
                    session.save(save_path)
                    if self.keep_room_checkpoints:
                        room_checkpoints.append(self.checkpoint.saved_from(session, save_path))
                decision = session.step(name, **kwargs)

        if self.raw_dir is not None:
            self.recorder.save(os.path.join(self.raw_dir, f"{self.main_id}.json"), session.action_log, outcome=outcome)
        return self.result(session, outcome, combats=combats, room_checkpoints=room_checkpoints)

    def _play_combat(self, session, decision, combats: list[dict], save_path, save_index):
        """Returns (decision after the combat, run outcome or None to keep going)."""
        entry_hp, max_hp = decision.player_hp, decision.player_max_hp
        potions_before = len(list(bl.player_potions(decision.arr)))
        combat = {
            "combat_index": len(combats) + 1, "act": decision.act, "floor": decision.floor,
            "enemies": [e for e in decision.get_enemies_ids() if e],
            "entry_hp": entry_hp, "max_hp": max_hp, "save_path": save_path, "save_index": save_index,
        }

        decision = session.set_player_hp(max_hp)
        turns: list[dict] = []
        while True:
            turn = {"start_index": len(session.action_log), "v": _value(decision)}
            hp_before = decision.player_hp
            decision = _play_turn(session, decision)
            died = decision.decision == "game_over"
            turn["loss"] = hp_before if died else hp_before - decision.player_hp
            turns.append(turn)
            if decision.decision != "combat_play" or len(turns) >= MAX_COMBAT_TURNS:
                break
        combat["turns"] = _with_returns(turns)
        combat["loss"] = sum(t["loss"] for t in turns)

        if decision.decision == "combat_play":  # stalled: still going at the turn cap
            combat.update(outcome="stalled", died=False)
            self._record(combat, combats)
            return decision, "lost"
        if decision.decision == "game_over":  # died with a full HP bar to spend
            combat.update(outcome="lost", died=True)
            self._record(combat, combats)
            return decision, "lost"

        combat["potions_used"] = max(0, potions_before - len(list(bl.player_potions(decision.arr))))
        new_hp = entry_hp - combat["loss"]
        # outcome is the combat itself, played from full HP; `died` is whether the RUN survives it
        combat.update(outcome="won", died=new_hp <= 0, exit_hp=new_hp if new_hp > 0 else None)
        self._record(combat, combats)
        if new_hp <= 0:  # the run didn't have the HP this combat cost
            return decision, "lost"
        return session.set_player_hp(new_hp), None

    def _record(self, combat: dict, combats: list[dict]) -> None:
        combats.append(combat)
        self.recorder.record_combat(
            combat["act"], combat["floor"], combat["combat_index"], combat["outcome"],
            **{k: v for k, v in combat.items() if k not in ("act", "floor", "combat_index", "outcome")},
        )


MACRO_DECISIONS = frozenset({"map_select", "card_reward", "event_card_reward", "shop", "rest_site", "event_choice"})


class _StuckGuard:
    """Notices a run that has stopped making progress: the same decision (byte-identical buffer) returned
    MAX_SAME_DECISION times in a row, or MAX_RUN_ACTIONS actions in total."""

    def __init__(self):
        self.steps = 0
        self._last: int | None = None
        self._same = 0

    def tripped(self, decision) -> str | None:
        self.steps += 1
        crc = zlib.crc32(decision.arr.tobytes())
        self._same = self._same + 1 if crc == self._last else 0
        self._last = crc
        if self._same >= MAX_SAME_DECISION:
            return f"{decision.decision} unchanged for {self._same} actions in a row"
        if self.steps >= MAX_RUN_ACTIONS:
            return f"{self.steps} actions in one run"
        return None


class RunJob(Job):
    """A whole run played with the policy at REAL HP (nothing is healed or reset), recording what
    golden_compass's hazard net trains on -- the opposite of MainlineJob, which heals to full before
    every combat so the run can never die of attrition.

    At every macro decision it logs the RunState (golden_compass/run_state.py) the player is
    looking at, and for a map move also the afterstate of the node it chose ("map_entered": the
    state the net is asked about when it picks a path). Per combat it logs the HP going in and out.
    The run file holds those plus the outcome and the floor the run ended on:

        outcome "lost"      died on end_floor            (hazard label 1 there, 0 before)
        outcome "won"       cleared the game             (hazard label 0 everywhere)
        outcome "censored"  stopped alive on end_floor: a combat hit MAX_COMBAT_TURNS, or the run stopped
                            making progress (_StuckGuard; the file then carries `capped` and the last
                            actions as `tail` so the cause can be found)

    Always starts fresh -- there is nothing here worth checkpointing yet. Writes
    raw_dir/{main_id}.json (role "run") when raw_dir is given."""
    role = "run"

    def __init__(self, checkpoint: Checkpoint, raw_dir: str | None = None):
        super().__init__(checkpoint)
        self.raw_dir = raw_dir

    def execute(self, session: Sts2BufferSession) -> JobResult:
        decision = self.begin(session)
        purpose = None
        states: list[dict] = []
        combats: list[dict] = []
        floor = decision.floor
        outcome = None
        guard = _StuckGuard()
        capped: str | None = None

        while outcome is None:
            if decision.decision == "game_over":
                outcome = "won" if decision.victory else "lost"
                break
            floor = decision.floor  # at game_over the buffer's floor is the room the run died in -- keep the last live one
            if decision.decision == "combat_play":
                decision, stalled, capped = self._play_combat(session, decision, combats, guard)
                if stalled:
                    outcome = "censored"
                continue
            capped = guard.tripped(decision)
            if capped:
                outcome = "censored"
                break

            state = None
            if decision.decision in MACRO_DECISIONS:
                state = rs.encode_from_buffer(decision.arr, session.character, session.ascension)
            name, kwargs, policy, purpose = choose_action(decision, session, purpose)
            if policy is not None:
                session.logger.attach_policy(policy)
            if state is not None:
                states.append({"decision": decision.decision, "action": name, "state": state.to_json()})
                if name == "select_map_node":
                    for key, entered in rs.map_afterstates(decision.arr, state):
                        if key == (kwargs["col"], kwargs["row"]):
                            states.append({"decision": "map_entered", "action": name, "state": entered.to_json()})
            decision = session.step(name, **kwargs)

        self.recorder.meta.update(end_floor=floor, states=states)
        if capped:
            self.recorder.meta.update(capped=capped, tail=session.action_log[-12:])
        if self.raw_dir is not None:
            self.recorder.save(os.path.join(self.raw_dir, f"{self.main_id}.json"), session.action_log, outcome=outcome)
        return self.result(session, outcome, end_floor=floor, n_states=len(states), combats=combats, capped=capped)

    def _play_combat(self, session, decision, combats: list[dict], guard: _StuckGuard):
        """Plays one combat to its end. -> (decision after it, whether it was stopped as stalled, why it was
        cut off by the guard or None)."""
        entry_hp, max_hp, floor, act = decision.player_hp, decision.player_max_hp, decision.floor, decision.act
        turns = actions_this_turn = 0
        capped = None
        while decision.decision in ("combat_play", "card_select"):
            capped = guard.tripped(decision)
            if capped is None and actions_this_turn >= MAX_ACTIONS_PER_TURN:
                capped = f"{actions_this_turn} actions in one combat turn"
            if capped:
                break
            name, kwargs, policy, _purpose = choose_action(decision, session)
            if policy is not None:
                session.logger.attach_policy(policy)
            decision = session.step(name, **kwargs)
            actions_this_turn += 1
            if name == "end_turn":
                turns += 1
                actions_this_turn = 0
                if turns >= MAX_COMBAT_TURNS and decision.decision == "combat_play":
                    break

        stalled = decision.decision in ("combat_play", "card_select")
        died = decision.decision == "game_over"
        outcome = "stalled" if stalled else "lost" if died else "won"
        combat = {
            "floor": floor, "hp_in": entry_hp, "hp_out": 0 if died else decision.player_hp,
            "max_hp": max_hp, "turns": turns,
        }
        combats.append(combat)
        self.recorder.record_combat(act, floor, len(combats), outcome, **{k: v for k, v in combat.items() if k != "floor"})
        return decision, stalled, capped


class ForkJob(Job):
    """One alternative continuation of a mainline combat: reloads the mainline's checkpoint at the
    start of turn `turn_index` (full HP, mid-combat) and plays the policy to the end of the combat,
    recording each turn's V(state), HP lost and the HP actually lost from there to the end. Writes
    raw_dir/{main_id}_c..t.._f...json (role "fork"). Fork `fork_index` of a turn always samples the
    same way (seeded), so a fork is reproducible."""
    role = "fork"

    def __init__(self, checkpoint: Checkpoint, combat_index: int, turn_index: int, fork_index: int,
                 raw_dir: str | None = None):
        super().__init__(checkpoint)
        self.combat_index = combat_index
        self.turn_index = turn_index
        self.fork_index = fork_index
        self.raw_dir = raw_dir

    def uid(self) -> str:
        return f"{self.main_id}_c{self.combat_index:03d}t{self.turn_index:02d}_f{self.fork_index:03d}"

    def execute(self, session: Sts2BufferSession) -> JobResult:
        _seed_everything(zlib_hash(self.uid()))
        decision = self.begin(session)
        turns: list[dict] = []
        while True:
            turn = {"start_index": len(session.action_log), "v": _value(decision)}
            hp_before = decision.player_hp
            decision = _play_turn(session, decision)
            died = decision.decision == "game_over"
            turn["loss"] = hp_before if died else hp_before - decision.player_hp
            turns.append(turn)
            if decision.decision != "combat_play" or len(turns) >= MAX_COMBAT_TURNS:
                break

        outcome = {"card_reward": "won", "game_over": "lost"}.get(decision.decision, "stalled")
        _with_returns(turns)
        if self.raw_dir is not None:
            self.recorder.meta.update(
                combat_index=self.combat_index, turn_index=self.turn_index, fork_index=self.fork_index,
                prefix_len=len(self.checkpoint.actions), turns=turns,
            )
            self.recorder.save(os.path.join(self.raw_dir, f"{self.uid()}.json"), session.action_log, outcome=outcome)
        return self.result(session, outcome, turns=turns)


def zlib_hash(text: str) -> int:
    """A stable seed from a string (Python's hash() is randomized per process, so forks wouldn't reproduce)."""
    import zlib
    return zlib.crc32(text.encode())
