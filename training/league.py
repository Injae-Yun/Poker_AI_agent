"""
league.py — League Training 관리자

AgentLeague:
  - 4개의 Main NFSPAgent (다양한 하이퍼파라미터 및 초기 전략 편향)
  - 2개의 Exploiter Agent (RuleAgent — 취약점 착취)
  - Frozen Snapshots (주기적으로 동결된 에이전트 가중치)
  - 매칭 구성: 현재 에이전트 + 랜덤 스냅샷 조합
  - 에이전트별 성과 추적 (chip delta 기준 ELO 근사)

설계 원칙:
  - 스냅샷은 ActorNetwork 가중치만 저장 (메모리 효율)
  - 스냅샷 에이전트는 학습하지 않음 (inference only)
  - Exploiter는 매 게임 2개 중 1개 랜덤 포함
"""

import os
import copy
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from agents.base_agent import BaseAgent
from agents.rule_agent import RuleAgent
from training.nfsp import NFSPAgent
from models.actor import ActorNetwork, ACTION_DIM
from utils.state_encoder import STATE_DIM


# ══════════════════════════════════════════════════════════
#  FrozenAgent — 학습하지 않는 스냅샷 에이전트
# ══════════════════════════════════════════════════════════

class FrozenAgent(BaseAgent):
    """
    NFSPAgent의 스냅샷으로부터 생성된 읽기 전용 에이전트.
    ActorNetwork 가중치를 복사해 inference-only로 동작합니다.
    """

    def __init__(
        self,
        player_id:   int,
        actor:       ActorNetwork,
        name:        str = "Frozen",
        initial_stack: int = 1000,
        big_blind:   int = 10,
    ):
        super().__init__(player_id, name)
        from utils.state_encoder import StateEncoder
        from utils.opponent_profiler import OpponentProfiler
        from reward.reward_shaper import build_valid_mask, action_idx_to_decision

        self._actor   = actor
        self._encoder = StateEncoder(initial_stack=initial_stack, big_blind=big_blind)
        self._profiler = OpponentProfiler()
        self._build_valid_mask   = build_valid_mask
        self._action_idx_to_decision = action_idx_to_decision

    def declare_action(self, obs) -> Dict[str, Any]:
        from utils.state_encoder import StateEncoder
        state      = self._encoder.encode(obs, self._profiler)
        valid_mask = self._build_valid_mask(obs)
        action_idx = self._actor.greedy_action(state, valid_mask)
        decision   = self._action_idx_to_decision(action_idx, obs)
        self._record_action(decision['action'])
        return decision

    def on_game_start(self, config) -> None:
        from utils.state_encoder import StateEncoder
        self._encoder = StateEncoder(
            initial_stack=config.initial_stack,
            big_blind=config.big_blind,
        )

    def on_round_end(self, result: Dict[str, Any]) -> None:
        super().on_round_end(result)
        for entry in result.get('betting_history', []):
            if isinstance(entry, dict):
                pid    = entry['player_id']
                act    = entry['action']
                street = entry['street']
                amt    = entry['amount']
            else:
                pid    = entry.player_id
                act    = entry.action
                street = entry.street
                amt    = entry.amount
            if pid != self.player_id:
                self._profiler.on_action(pid, act, street, amt, 0, is_blind=(act=='blind'))


# ══════════════════════════════════════════════════════════
#  Snapshot — 단일 스냅샷 항목
# ══════════════════════════════════════════════════════════

@dataclass
class Snapshot:
    """게임 N회 시점의 에이전트 상태 스냅샷"""
    step:         int
    source_id:    int           # 원본 에이전트 player_id
    actor_state:  List[Dict]    # Sequential.state_dict() 직렬화 데이터
    eta:          float = 0.1
    avg_chip_delta: float = 0.0


# ══════════════════════════════════════════════════════════
#  AgentLeague
# ══════════════════════════════════════════════════════════

# Main 에이전트별 하이퍼파라미터 (전략 다양성 초기화)
MAIN_AGENT_CONFIGS = [
    # (이름접두사, eta, epsilon, entropy_coef, actor_lr, critic_lr, 전략편향)
    dict(name="NFSP_Tight",    eta=0.10, epsilon=0.05, entropy_coef=0.01, actor_lr=2e-4, critic_lr=8e-4),
    dict(name="NFSP_Loose",    eta=0.10, epsilon=0.15, entropy_coef=0.03, actor_lr=4e-4, critic_lr=1.2e-3),
    dict(name="NFSP_Balanced", eta=0.15, epsilon=0.10, entropy_coef=0.02, actor_lr=3e-4, critic_lr=1e-3),
    dict(name="NFSP_Aggr",     eta=0.20, epsilon=0.08, entropy_coef=0.02, actor_lr=3e-4, critic_lr=1e-3),
]


class AgentLeague:
    """
    League Training 관리자.

    에이전트 구성:
      - main_agents[0..3] : NFSPAgent × 4 (다양한 초기 편향)
      - exploiters[0..1]  : RuleAgent × 2 (취약점 착취용)
      - snapshots         : 과거 에이전트 스냅샷 풀

    매칭:
      - 4개 슬롯 = main + exploiter + snapshot 조합으로 채움
      - 슬롯 구성: main(최소 1) + exploiter(0~1) + snapshot(나머지)
    """

    def __init__(
        self,
        initial_stack:    int   = 1000,
        small_blind:      int   = 5,
        big_blind:        int   = 10,
        snapshot_every:   int   = 10,      # N 게임마다 스냅샷 저장
        max_snapshots:    int   = 20,      # 유지할 최대 스냅샷 수
        exploiter_ratio:  float = 0.5,     # 게임당 exploiter 포함 확률
        seed:             int   = 0,
    ):
        self.initial_stack   = initial_stack
        self.small_blind     = small_blind
        self.big_blind       = big_blind
        self.snapshot_every  = snapshot_every
        self.max_snapshots   = max_snapshots
        self.exploiter_ratio = exploiter_ratio
        self._rng            = np.random.default_rng(seed)

        # Main 에이전트 생성
        self.main_agents: List[NFSPAgent] = []
        for i, cfg in enumerate(MAIN_AGENT_CONFIGS):
            agent = NFSPAgent(
                player_id    = i,
                name         = cfg['name'],
                eta          = cfg['eta'],
                epsilon      = cfg['epsilon'],
                entropy_coef = cfg['entropy_coef'],
                actor_lr     = cfg['actor_lr'],
                critic_lr    = cfg['critic_lr'],
                initial_stack= initial_stack,
                big_blind    = big_blind,
                seed         = seed + i * 37,
            )
            self.main_agents.append(agent)

        # Exploiter 에이전트 생성 (RuleAgent 2개)
        self.exploiters: List[RuleAgent] = [
            RuleAgent(player_id=0, name="Exploiter0", seed=seed + 200),
            RuleAgent(player_id=0, name="Exploiter1", seed=seed + 300),
        ]

        # 스냅샷 풀
        self._snapshots: List[Snapshot] = []

        # 성과 추적 (player_id → 최근 게임 chip_delta 리스트)
        self._perf: Dict[int, List[float]] = {i: [] for i in range(4)}

        # 게임 카운터
        self._game_count = 0

    # ══════════════════════════════════════════════════════
    #  매칭 생성
    # ══════════════════════════════════════════════════════

    def create_matchup(self) -> List[BaseAgent]:
        """
        4인 게임 매칭 구성.

        전략:
          슬롯 0~3을 채우는 방식:
            - main_agents 전원 참가 (슬롯 0~3 = main_agents × 4)
            - exploiter_ratio 확률로 슬롯 중 하나를 RuleAgent로 교체
            - 스냅샷이 있으면 슬롯 하나를 랜덤 스냅샷으로 교체

        Returns:
            List[BaseAgent] length 4 (player_id 재할당됨)
        """
        # main_agents 4개 복사 (player_id 재할당)
        agents = []
        for i, agent in enumerate(self.main_agents):
            agent.player_id = i
            agents.append(agent)

        # exploiter 교체
        if self._rng.random() < self.exploiter_ratio and self._snapshots:
            slot = int(self._rng.integers(1, 4))   # 슬롯 1~3 중 하나를 교체
            exploiter = copy.copy(self._rng.choice(self.exploiters))  # type: ignore
            exploiter.player_id = slot
            agents[slot] = exploiter

        # 스냅샷 교체 (exploiter와 다른 슬롯)
        if self._snapshots and len(self._snapshots) >= 2:
            snap = self._snapshots[int(self._rng.integers(0, len(self._snapshots)))]
            actor = ActorNetwork()
            actor.net.load_state_dict(snap.actor_state)

            # exploiter가 교체한 슬롯은 피함
            occupied = {a.player_id for a in agents if not isinstance(a, NFSPAgent)}
            available = [i for i in range(1, 4) if i not in occupied]
            if available:
                slot = int(self._rng.choice(available))
                agents[slot] = FrozenAgent(
                    player_id    = slot,
                    actor        = actor,
                    name         = f"Snap{snap.step}",
                    initial_stack= self.initial_stack,
                    big_blind    = self.big_blind,
                )

        return agents

    # ══════════════════════════════════════════════════════
    #  스냅샷 관리
    # ══════════════════════════════════════════════════════

    def maybe_snapshot(self, game_idx: int) -> bool:
        """게임 카운터 기준 스냅샷 저장 여부 판단 및 저장"""
        self._game_count = game_idx
        if (game_idx + 1) % self.snapshot_every != 0:
            return False

        for agent in self.main_agents:
            snap = Snapshot(
                step         = game_idx + 1,
                source_id    = agent.player_id,
                actor_state  = agent._actor.net.state_dict(),
                eta          = agent.eta,
                avg_chip_delta = float(np.mean(self._perf[agent.player_id][-10:]))
                                 if self._perf[agent.player_id] else 0.0,
            )
            self._snapshots.append(snap)

        # 최대 스냅샷 수 유지
        if len(self._snapshots) > self.max_snapshots * 4:
            self._snapshots = self._snapshots[-(self.max_snapshots * 4):]

        return True

    # ══════════════════════════════════════════════════════
    #  성과 기록
    # ══════════════════════════════════════════════════════

    def record_result(self, final_stacks: Dict[int, int]) -> None:
        """게임 결과를 성과 추적에 기록"""
        for i in range(4):
            if i in final_stacks:
                delta = final_stacks[i] - self.initial_stack
                self._perf[i].append(float(delta))
                # 최근 100 게임만 유지
                if len(self._perf[i]) > 100:
                    self._perf[i] = self._perf[i][-100:]

    def get_stats(self) -> Dict[str, Any]:
        """에이전트별 성과 통계 반환"""
        stats = {}
        for i, agent in enumerate(self.main_agents):
            hist = self._perf[i]
            stats[agent.name] = {
                'player_id':         i,
                'avg_chip_delta':    float(np.mean(hist)) if hist else 0.0,
                'std_chip_delta':    float(np.std(hist))  if hist else 0.0,
                'win_rate':          sum(1 for d in hist if d > 0) / max(len(hist), 1),
                'games_played':      len(hist),
                'reservoir_size':    agent.reservoir_size,
                'eta':               agent.eta,
                'sl_updates':        len(agent.sl_stats),
            }
        return stats

    # ══════════════════════════════════════════════════════
    #  체크포인트 저장 / 로드
    # ══════════════════════════════════════════════════════

    def save(self, checkpoint_dir: str, step: int) -> None:
        """리그 전체 체크포인트 저장"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        for agent in self.main_agents:
            agent.save(checkpoint_dir, step)

        # 스냅샷 메타 저장
        meta = [{
            'step':            s.step,
            'source_id':       s.source_id,
            'eta':             s.eta,
            'avg_chip_delta':  s.avg_chip_delta,
        } for s in self._snapshots[-20:]]  # 최근 20개만
        with open(os.path.join(checkpoint_dir, f"league_meta_{step}.json"), 'w') as f:
            json.dump({'snapshots': meta, 'game_count': self._game_count}, f, indent=2)

    def load(self, checkpoint_dir: str, step: int) -> None:
        """체크포인트 로드"""
        for agent in self.main_agents:
            agent.load(checkpoint_dir, step)
