"""
rl_agent.py — Actor-Critic (A2C) RL 에이전트

Phase 2: Advantage Function 기반 보상으로 학습
  - ActorNetwork  : 정책 (action distribution)
  - CriticNetwork : 가치 함수 V(s)
  - RewardShaper  : 결정 품질 + 결과 놀람 보상
  - StateEncoder  : AgentObservation → 98차원 상태 벡터
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from agents.base_agent import BaseAgent
from engine.state import AgentObservation
from models.actor import ActorNetwork
from models.critic import CriticNetwork
from reward.reward_shaper import RewardShaper, action_idx_to_decision, build_valid_mask
from utils.state_encoder import StateEncoder
from utils.opponent_profiler import OpponentProfiler


@dataclass
class Transition:
    """단일 결정 스텝의 전환 정보"""
    state:      np.ndarray        # (STATE_DIM,)
    action_idx: int
    log_prob:   float
    value:      float             # V(s) at decision time
    reward:     float = 0.0       # on_round_end에서 채워짐
    done:       bool  = False
    valid_mask: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=bool))


class RLAgent(BaseAgent):
    """
    Actor-Critic RL 에이전트.
    핸드 단위로 에피소드 버퍼를 수집하고 업데이트합니다.
    """

    def __init__(
        self,
        player_id:     int,
        name:          str   = "RLAgent",
        actor_lr:      float = 3e-4,
        critic_lr:     float = 1e-3,
        entropy_coef:  float = 0.02,
        epsilon:       float = 0.10,   # ε-greedy 탐색 비율 (Trainer가 외부에서 제어)
        update_every:  int   = 1,      # N 핸드마다 업데이트
        initial_stack: int   = 1000,
        big_blind:     int   = 10,
        seed:          Optional[int] = None,
    ):
        super().__init__(player_id, name)

        self.entropy_coef = entropy_coef
        self.epsilon      = epsilon
        self.update_every = update_every

        if seed is not None:
            np.random.seed(seed)

        # 네트워크
        self._actor  = ActorNetwork(lr=actor_lr)
        self._critic = CriticNetwork(lr=critic_lr)

        # 보상 설계
        self._shaper = RewardShaper()

        # 상태 인코더 (on_game_start에서 config 값으로 재초기화)
        self._encoder = StateEncoder(initial_stack=initial_stack, big_blind=big_blind)

        # 상대 프로파일러
        self._profiler = OpponentProfiler()

        # 에피소드 버퍼 (update 후 clear)
        self._episode_buffer: List[Transition] = []

        # 핸드 단위 임시 버퍼
        self._hand_transitions: List[Transition] = []
        self._hand_baseline_vs: List[float]      = []

        # 업데이트 스케줄
        self._hands_since_update = 0

        # 학습 통계
        self.train_stats: List[Dict] = []

    # ══════════════════════════════════════════════════════
    #  BaseAgent 콜백
    # ══════════════════════════════════════════════════════

    def on_game_start(self, config) -> None:
        """게임 설정으로 StateEncoder 재초기화"""
        self._encoder = StateEncoder(
            initial_stack=config.initial_stack,
            big_blind=config.big_blind,
        )
        # 프로파일러는 게임 간 지속 (누적 통계 유지)

    def on_round_start(self, round_num: int) -> None:
        super().on_round_start(round_num)
        self._shaper.reset()
        self._hand_transitions = []
        self._hand_baseline_vs = []

    def declare_action(self, obs: AgentObservation) -> Dict[str, Any]:
        """상태 인코딩 → Actor 샘플링 → 액션 반환"""
        state      = self._encoder.encode(obs, self._profiler)
        valid_mask = build_valid_mask(obs)
        value      = self._critic.value(state)

        action_idx, log_prob, entropy = self._actor.sample_action(
            state, valid_mask, self.epsilon
        )
        decision = action_idx_to_decision(action_idx, obs)

        # 전환 기록 (reward는 on_round_end에서 채움)
        self._hand_transitions.append(Transition(
            state      = state,
            action_idx = action_idx,
            log_prob   = log_prob,
            value      = value,
            reward     = 0.0,
            done       = False,
            valid_mask = valid_mask,
        ))
        self._hand_baseline_vs.append(value)

        # RewardShaper에 결정 품질 즉시 계산 (outcome_surprise는 대기)
        self._shaper.on_action(obs, decision['action'], decision['amount'], value)

        self._record_action(decision['action'])
        return decision

    def on_round_end(self, result: Dict[str, Any]) -> None:
        super().on_round_end(result)

        # 핸드 결과로 보상 확정
        chip_delta    = result.get('chip_delta', 0)
        final_rewards = self._shaper.on_hand_end(chip_delta, self._hand_baseline_vs)

        # 전환에 보상 기입
        for i, t in enumerate(self._hand_transitions):
            t.reward = final_rewards[i] if i < len(final_rewards) else 0.0
            t.done   = (i == len(self._hand_transitions) - 1)

        # 에피소드 버퍼에 추가
        self._episode_buffer.extend(self._hand_transitions)

        # 상대 프로파일 업데이트 (rule_agent.py 패턴과 동일)
        self._update_profiler(result)

        # 업데이트 트리거
        self._hands_since_update += 1
        if self._hands_since_update >= self.update_every and self._episode_buffer:
            actor_loss, critic_loss = self._update()
            self.train_stats.append({
                'actor_loss':  actor_loss,
                'critic_loss': critic_loss,
            })
            self._hands_since_update = 0

    def on_game_end(self) -> None:
        # 게임 종료 시 잔여 버퍼 플러시
        if self._episode_buffer:
            self._update()
            self._hands_since_update = 0

    # ══════════════════════════════════════════════════════
    #  내부 메서드
    # ══════════════════════════════════════════════════════

    def _update(self) -> Tuple[float, float]:
        """Actor-Critic 업데이트 한 스텝. 버퍼 clear 후 (actor_loss, critic_loss) 반환."""
        buf = self._episode_buffer

        states      = np.array([t.state      for t in buf], dtype=np.float32)
        action_idxs = np.array([t.action_idx for t in buf], dtype=np.int32)
        rewards     = np.array([t.reward     for t in buf], dtype=np.float32)
        dones       = np.array([t.done       for t in buf], dtype=bool)
        valid_masks = np.array([t.valid_mask for t in buf], dtype=bool)

        returns, advantages = self._critic.compute_advantages(states, rewards, dones)
        actor_loss  = self._actor.update(
            states, action_idxs, advantages, valid_masks, self.entropy_coef
        )
        critic_loss = self._critic.update(states, returns)

        self._episode_buffer.clear()
        return float(actor_loss), float(critic_loss)

    def _update_profiler(self, result: Dict[str, Any]) -> None:
        """on_round_end 결과로 상대 프로파일러 업데이트"""
        for entry in result.get('betting_history', []):
            # BettingAction 객체 또는 dict 모두 지원
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
                self._profiler.on_action(
                    player_id = pid,
                    action    = act,
                    street    = street,
                    amount    = amt,
                    pot       = 0,
                    is_blind  = (act == 'blind'),
                )

        if result.get('showdown'):
            for pid in result.get('player_hands', {}).keys():
                if pid != self.player_id:
                    won = pid in result.get('winners', [])
                    self._profiler.on_showdown(pid, won)

    # ══════════════════════════════════════════════════════
    #  체크포인트
    # ══════════════════════════════════════════════════════

    def save(self, checkpoint_dir: str, step: int) -> None:
        """Actor/Critic 가중치를 checkpoint_dir에 저장"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._actor.save(os.path.join(checkpoint_dir, f"actor_step_{step}.json"))
        self._critic.save(os.path.join(checkpoint_dir, f"critic_step_{step}.json"))

    def load(self, checkpoint_dir: str, step: int) -> None:
        """Actor/Critic 가중치 로드"""
        self._actor.load(os.path.join(checkpoint_dir, f"actor_step_{step}.json"))
        self._critic.load(os.path.join(checkpoint_dir, f"critic_step_{step}.json"))
