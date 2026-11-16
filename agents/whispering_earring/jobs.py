from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from buffer_env import Sts2BufferSession, BufferDecision
from env import Sts2Session
from checkpoint import Checkpoint
from policy import choose_action, choose_generic_action, get_claws_net
from recorder import CompactRecorder
from utils import BOSS_HP_WEIGHT, HP_WEIGHT, bypass_card_reward, combat_value

TEST_CHECKPOINT_DIR = "data/test_checkpoints"


# This should have enough information to create a save .json, resume, and materialize for training.
@dataclass
class JobResult:
    main_id: str
    outcome: str
    action_log: list[dict]  # full history since StartRun, including whatever the checkpoint skipped replaying
    checkpoint_path: str | None = None
    decision: BufferDecision | None = None
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

    def begin(self, session: Sts2BufferSession | Sts2Session) -> BufferDecision:
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

    # Returns a JobResult that is useful to spawn new jobs. Must not return None; the worker
    # reports a raised exception as an "error" result.
    @abstractmethod
    def execute(self, session: Sts2BufferSession | Sts2Session) -> JobResult:
        pass


class TrialJob(Job):
    """The main line of one run. Skeleton: the combat/map handlers are the next piece of the refactor."""
    role = "main"

    def __init__(self, checkpoint: Checkpoint, combat_index: int = 0):
        super().__init__(checkpoint)
        self.combat_index = combat_index

    def execute(self, session: Sts2BufferSession | Sts2Session) -> JobResult:
        decision = self.begin(session)

        while decision.decision != "game_over":
            if decision.decision == "combat_play":
                decision = self._resolve_combat_play(session, decision)
            elif decision.decision == "select_map_node":
                decision = self._resolve_select_map_node(session, decision)
            else:
                raise NotImplementedError(f"TrialJob can't resolve {decision.decision!r} yet")
        return JobResult(
            main_id=self.main_id,
            outcome="won" if decision.victory else "lost",
            action_log=session.action_log,
            decision=decision,
        )

    def _resolve_combat_play(self, session, decision: BufferDecision) -> BufferDecision:
        self.combat_index += 1
        raise NotImplementedError

    def _resolve_select_map_node(self, session, decision: BufferDecision) -> BufferDecision:
        raise NotImplementedError


class CombatTurnJob(Job):
    """Plays exactly one combat turn from the checkpoint and scores it."""
    role = "branch"

    def __init__(self, checkpoint: Checkpoint, combat_index: int):
        super().__init__(checkpoint)
        self.combat_index = combat_index

    def uid(self) -> str:
        return f"{self.main_id}_combat_{self.combat_index}"

    def execute(self, session: Sts2BufferSession | Sts2Session) -> JobResult:
        decision = self.begin(session)
        hp_before_turn = decision.player_hp
        combat_max_hp = decision.player_max_hp
        is_last_of_act = decision.is_last_of_act()

        enemies_alive = decision.get_enemies_alive_mask()
        enemies_hp_before_turn = decision.get_enemies_hp()
        enemies_max_hp = decision.get_enemies_max_hp()

        session.logger.begin_turn()
        ended_turn = False

        while decision.decision in ("combat_play", "card_select"):
            name, kwargs, policy, _purpose = choose_action(decision, session)
            if policy is not None:
                session.logger.attach_policy(policy)
            decision = session.step(name, **kwargs)
            if name == "end_turn":
                ended_turn = True
            if ended_turn and decision.decision == "combat_play":
                break  # new round started -- this turn is fully resolved, combat continues

        assert decision.decision in ("combat_play", "card_reward", "game_over"), (
            f"turn ended on unexpected decision {decision.decision!r} -- would silently score "
            "this turn wrong if treated as resolved"
        )

        if decision.decision == "card_reward":
            bypass_card_reward(session)
            outcome = "won"
            turn_value = min(1.0, combat_value(decision, is_last_of_act) + 1.0)
        elif decision.decision == "game_over":
            outcome = "lost"
            turn_value = -1.0
        else:
            outcome = "continue"
            v_net = 0.0
            net = get_claws_net()
            if net is not None:
                _, _, value_t = net.forward(decision)
                v_net = float(value_t.item())
            hp_term = (
                -(hp_before_turn - decision.player_hp) / combat_max_hp if combat_max_hp else 0.0
            )
            enemy_term = 0.0
            if enemies_alive.any():
                d_enemies_hp = enemies_hp_before_turn - decision.get_enemies_hp()
                enemy_term = d_enemies_hp.sum() / enemies_max_hp.sum()
            hp_weight = BOSS_HP_WEIGHT if is_last_of_act else HP_WEIGHT
            turn_value = hp_weight * (hp_term + enemy_term) + v_net

        session.logger.end_turn(turn_value)

        return JobResult(
            main_id=self.main_id,
            action_log=session.action_log,
            outcome=outcome,
            decision=decision,
        )


class WholeCombatJob(Job):
    """Plays a whole combat from the checkpoint (a room-entry save is the usual start)."""
    role = "branch"

    def __init__(self, checkpoint: Checkpoint, combat_index: int):
        super().__init__(checkpoint)
        self.combat_index = combat_index

    def uid(self) -> str:
        return f"{self.main_id}_wholecombat_{self.combat_index}"

    def execute(self, session: Sts2BufferSession | Sts2Session) -> JobResult:
        decision = self.begin(session)
        purpose = None
        while decision.decision in ("combat_play", "card_select"):
            name, kwargs, purpose, policy = choose_generic_action(decision, purpose)
            if policy is not None:
                session.logger.attach_policy(policy)
            decision = session.step(name, **kwargs)

        assert decision.decision in ("card_reward", "game_over"), (
            f"combat ended on unexpected decision {decision.decision!r} -- would silently score "
            "this combat wrong if treated as resolved"
        )

        return JobResult(
            main_id=self.main_id,
            action_log=session.action_log,
            outcome="won" if decision.decision == "card_reward" else "lost",
            decision=decision,
        )


class TestJob(Job):
    """Plays a whole run with the network-free generic policy, saving a native checkpoint at every
    map_select. Each save comes back as a Checkpoint in auxiliary_data["checkpoints"], so any later
    job can resume from it."""
    role = "test"
    __test__ = False  # not a pytest class

    def execute(self, session: Sts2BufferSession) -> JobResult:
        decision = self.begin(session)
        purpose = None
        room_index = 0
        in_turn = False
        hp_before_turn = 0
        enemies_hp_before_turn = 0.0
        turns: list[dict] = []  # per-combat-turn aux stats, not a training value -- see JobResult.auxiliary_data
        checkpoints: list[Checkpoint] = []

        while decision.decision != "game_over":
            if decision.decision == "map_select":
                room_index += 1
                os.makedirs(TEST_CHECKPOINT_DIR, exist_ok=True)
                # room_index, not floor: unique per save even across acts, so a later save can
                # never overwrite one a job is still waiting to load.
                save_path = os.path.join(TEST_CHECKPOINT_DIR, f"{self.main_id}_r{room_index:03d}.json")
                session.save(save_path)
                checkpoints.append(self.checkpoint.saved_from(session, save_path))

            if decision.decision == "combat_play" and not in_turn:
                in_turn = True
                hp_before_turn = decision.player_hp
                enemies_hp_before_turn = float(decision.get_enemies_hp().sum())

            name, kwargs, purpose, policy = choose_generic_action(decision, purpose)
            if policy is not None:
                session.logger.attach_policy(policy)
            decision = session.step(name, **kwargs)

            if in_turn and (name == "end_turn" or decision.decision in ("card_reward", "game_over")):
                turns.append({
                    "damage_taken": max(0, hp_before_turn - decision.player_hp),
                    "damage_done": max(0.0, enemies_hp_before_turn - float(decision.get_enemies_hp().sum())),
                })
                in_turn = False

        return JobResult(
            main_id=self.main_id,
            action_log=session.action_log,
            outcome="won" if decision.victory else "lost",
            decision=decision,
            auxiliary_data={"turns": turns, "checkpoints": checkpoints},
        )
