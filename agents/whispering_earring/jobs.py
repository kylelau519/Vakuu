from __future__ import annotations
from dataclasses import dataclass, field
from buffer_env import Sts2BufferSession, Sts2Session, BufferDecision
import buffer_layout as bl
from checkpoint import BasicRunInfo, FastLoadInfo
from recorder import CompactRecorder
from policy import choose_action, get_claws_net
from utils import *
import multiprocessing as mp

from abc import ABC, abstractmethod

# Atomic unit once a run is started: a single combat turn, or a whole combat.
class Job(ABC):
    '''A job that can be executed by a worker.'''
    basic_run_info: BasicRunInfo
    result_queue: mp.Queue
    
    # This should return a result of some kind that is useful to spawn new jobs...
    @abstractmethod
    def execute(self, session: Sts2BufferSession | Sts2Session):
        pass
    
    def wrap_result(self):
        pass
    
    def emit_result(self):
        result = self.wrap_result()
        if result is not None:
            self.result_queue.put(result) 


# This should have enough information to create a save .json, resume, and materialize for training. 
@dataclass
class JobResult:
    main_id: str
    outcome: str
    action_log: list[dict]  # in case checkpoint does not include all actions.
    checkpoint_path: str | None = None
    decision: BufferDecision | None = None
    auxiliary_data: dict = field(default_factory=dict)
    
class TrialJob(Job):
    def __init__(self, basic_run_info: BasicRunInfo, result_queue: mp.Queue, checkpoint: dict | None = None):
        self.basic_run_info = basic_run_info
        self.checkpoint = checkpoint
        self.combat_index = 0
        self.result_queue = result_queue
    
    def execute(self, session: Sts2BufferSession | Sts2Session):
        decision = self._init_start(session)
        
        while decision.decision != "game_over":
            if decision.decision == "combat_play":
                self._resolve_combat_play()
            elif decision.decision == "select_map_node":
                self._resolve_select_map_node()
        return ("won" if decision.victory else "lost"), decision

    def _init_start(self, session: Sts2BufferSession | Sts2Session):
        if self.checkpoint is not None:
            decision = session.open_from_checkpoint(
                self.basic_run_info.character, self.basic_run_info.seed, self.basic_run_info.ascension, self.checkpoint["actions"]
            )
            self.combat_index = self.checkpoint["combat_index"]
        else:
            decision = session.start(self.basic_run_info.character, self.basic_run_info.seed, self.basic_run_info.ascension)
            self.combat_index = 0
        return decision
    
    def _resolve_combat_play(self):
        self.combat_index += 1
        
    def _resolve_select_map_node(self):
        pass
    
    def create_recorder(self):
        return CompactRecorder({
            "character": self.basic_run_info.character,
            "seed": self.basic_run_info.seed,
            "ascension": self.basic_run_info.ascension,
            "main_id": self.basic_run_info.main_id,
            "role": "main",
        })

class CombatTurnJob(Job):
    def __init__(self, run_info: FastLoadInfo | BasicRunInfo, result_queue: mp.Queue, combat_index: int, checkpoint: dict):
        self.run_info = run_info
        self.combat_index = combat_index
        self.checkpoint = checkpoint
        self.result_queue = result_queue
    
    def execute(self, session: Sts2BufferSession | Sts2Session):
        decision = self._init_start(session)
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
                break
            if ended_turn and decision.decision == "combat_play":
                break  # new round started -- this turn is fully resolved, combat continues
            
        assert decision.decision in ("combat_play", "card_reward", "game_over"), (
            f"turn ended on unexpected decision {decision.decision!r} -- would silently score "
            "this turn wrong if treated as resolved"
        )

        if decision.decision == "card_reward":
            bypass_card_reward(session)
            outcome = "won"
            
            turn_value = min(1.0, combat_value( decision) + 1.0)
        elif decision.decision == "game_over":
            outcome = "lost"
            turn_value = -1.0
            lost_to = decision.get_enemy_ids()[0] if decision.get_enemy_ids() else None
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
        
        result = JobResult(
            main_id=self.run_info.main_id,
            action_log=session.logger.action_log,
            outcome=outcome,
            decision=decision,
            checkpoint_path=None
        )
        return result
        # return outcome, decision
            
        # put in play turn only logic
    
    def _init_start(self, session: Sts2BufferSession | Sts2Session) -> BufferDecision:
        # checkpoint must exist for combat turn.
        if isinstance(self.run_info, FastLoadInfo):
            entry_action = self.checkpoint["actions"][self.run_info.combat_start_index]
            in_combat_actions = self.checkpoint["actions"][self.run_info.combat_start_index + 1:]
            decision = session.load_and_resume_combat(
                self.run_info.checkpoint_path, entry_action, in_combat_actions
            )
        else:
            decision = session.open_from_checkpoint(
                self.run_info.character, self.run_info.seed, self.run_info.ascension, self.checkpoint["actions"]
            )
        return decision
    
    def uid(self) -> str:
        return f"{self.run_info.main_id}_combat_{self.combat_index}"

class WholeCombatJob(Job):
    def __init__(self, run_info: FastLoadInfo | BasicRunInfo, result_queue: mp.Queue, combat_index: int, checkpoint: dict):
        self.run_info = run_info
        self.combat_index = combat_index
        self.checkpoint = checkpoint
        self.result_queue = result_queue
    
    def execute(self, session: Sts2BufferSession | Sts2Session):
        decision = self._init_start(session)
        while decision.decision in ("combat_play", "card_select"):
            name, kwargs, policy, _purpose = choose_action(decision, session)
            if policy is not None:
                session.logger.attach_policy(policy)
            decision = session.step(name, **kwargs)
        
        assert decision.decision in ("card_reward", "game_over"), (
            f"combat ended on unexpected decision {decision.decision!r} -- would silently score "
            "this combat wrong if treated as resolved"
        )
        
        if decision.decision == "card_reward":
            # bypass_card_reward(session)
            outcome = "won"
        else:
            outcome = "lost"
        
        result = JobResult(
            main_id=self.run_info.main_id,
            action_log=session.logger.action_log,
            outcome=outcome,
            decision=decision,
            checkpoint_path=None
        )
        return result
    

    def _init_start(self, session: Sts2BufferSession | Sts2Session) -> BufferDecision:
        # checkpoint must exist for combat turn.
        if isinstance(self.run_info, FastLoadInfo):
            entry_action = self.checkpoint["actions"][self.run_info.combat_start_index]
            in_combat_actions = self.checkpoint["actions"][self.run_info.combat_start_index + 1:]
            decision = session.load_and_resume_combat(
                self.run_info.checkpoint_path, entry_action, in_combat_actions
            )
        else:
            decision = session.open_from_checkpoint(
                self.run_info.character, self.run_info.seed, self.run_info.ascension, self.checkpoint["actions"]
            )
        return decision