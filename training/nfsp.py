"""
nfsp.py — Neural Fictitious Self-Play (NFSP) 에이전트

구성요소:
  ReservoirBuffer   : 과거 (state, action) 이력을 균등 확률로 보존하는 버퍼
  NFSPAgent         : Best Response(A2C) + Average Policy(SL) 혼합 에이전트
    - η 확률로 Best Response(BR) 정책 사용
    - (1-η) 확률로 Average Strategy Policy(ASP) 사용
    - ReservoirBuffer 에 BR 정책의 (state, action) 쌍 저장
    - 주기적으로 ASP 네트워크를 Supervised Learning 으로 업데이트

참고:
  Lanctot et al. (2017), "A Unified Game-Theoretic Approach to Multiagent RL"
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from agents.base_agent import BaseAgent
from agents.rl_agent import RLAgent, Transition
from engine.state import AgentObservation
from models.actor import ActorNetwork, ACTION_DIM
from models.critic import CriticNetwork
from models.nn import Sequential, Linear, ReLU, LayerNorm, Adam, Softmax
from reward.reward_shaper import RewardShaper, action_idx_to_decision, build_valid_mask
from utils.state_encoder import StateEncoder, STATE_DIM
from utils.opponent_profiler import OpponentProfiler


# ══════════════════════════════════════════════════════════
#  ReservoirBuffer
# ══════════════════════════════════════════════════════════

class ReservoirBuffer:
    """
    Reservoir Sampling 기반 경험 버퍼.

    버퍼가 가득 찰 때 이전 데이터를 균등한 확률로 교체합니다.
    이를 통해 오래된 경험도 남아있어 평균 전략을 더 잘 근사합니다.
    """

    def __init__(self, capacity: int = 100_000, seed: Optional[int] = None):
        self.capacity = capacity
        self._rng     = np.random.default_rng(seed)
        self._buf: List[Tuple[np.ndarray, int]] = []   # (state, action_idx)
        self._total_added = 0   # 삽입 시도 누적 횟수

    def add(self, state: np.ndarray, action_idx: int) -> None:
        """O(1) Reservoir Sampling 삽입"""
        self._total_added += 1
        if len(self._buf) < self.capacity:
            self._buf.append((state.copy(), int(action_idx)))
        else:
            idx = int(self._rng.integers(0, self._total_added))
            if idx < self.capacity:
                self._buf[idx] = (state.copy(), int(action_idx))

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        n개 샘플 반환.
        Returns:
            states  : (n, STATE_DIM) float32
            actions : (n,)          int64
        """
        n = min(n, len(self._buf))
        if n == 0:
            return np.zeros((0, STATE_DIM), dtype=np.float32), np.zeros(0, dtype=np.int64)
        indices = self._rng.choice(len(self._buf), n, replace=False)
        states  = np.array([self._buf[i][0] for i in indices], dtype=np.float32)
        actions = np.array([self._buf[i][1] for i in indices], dtype=np.int64)
        return states, actions

    @property
    def size(self) -> int:
        return len(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════
#  Average Strategy Policy (ASP) 네트워크 — SL 업데이트 전용
# ══════════════════════════════════════════════════════════

def build_asp_network(state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> Sequential:
    """
    Average Strategy Policy 네트워크 구조 (BR 과 동일):
      FC(256) → LayerNorm → ReLU → FC(128) → ReLU → FC(action_dim) → Softmax
    """
    return Sequential([
        Linear(state_dim, 256),
        LayerNorm(256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, action_dim),
        Softmax(),
    ])


class AveragePolicyNetwork:
    """
    Average Strategy Policy 네트워크.
    Reservoir Buffer에서 샘플링한 (state, action) 쌍으로
    지도학습(cross-entropy) 방식으로 업데이트됩니다.
    """

    def __init__(
        self,
        state_dim:  int   = STATE_DIM,
        action_dim: int   = ACTION_DIM,
        lr:         float = 1e-3,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.net        = build_asp_network(state_dim, action_dim)
        self.optimizer  = Adam(lr=lr, clip=1.0)

    def probs(self, state: np.ndarray) -> np.ndarray:
        return self.net.forward(state.astype(np.float32))

    def sample_action(
        self,
        state:      np.ndarray,
        valid_mask: np.ndarray,
    ) -> int:
        """합법 액션 마스크 고려 greedy 샘플링"""
        p = self.probs(state) * valid_mask
        total = p.sum()
        if total < 1e-8:
            valid_idxs = np.where(valid_mask)[0]
            return int(np.random.choice(valid_idxs))
        p /= total
        return int(np.random.choice(len(p), p=p))

    def update(
        self,
        states:  np.ndarray,    # (T, STATE_DIM)
        actions: np.ndarray,    # (T,) int — 타겟 액션 인덱스
    ) -> float:
        """
        Cross-entropy 지도학습 업데이트.
        Returns: 평균 cross-entropy loss
        """
        T          = len(states)
        total_loss = 0.0

        for t in range(T):
            probs = self.net.forward(states[t].astype(np.float32))
            a     = int(actions[t])

            # Cross-entropy loss: -log p(a)
            log_p = np.log(probs[a] + 1e-8)
            total_loss -= log_p

            # 그래디언트: softmax cross-entropy
            grad = probs.copy()
            grad[a] -= 1.0

            self.net.backward(grad)
            self.optimizer.step(self.net)

        return total_loss / max(T, 1)

    def save(self, path: str) -> None:
        self.net.save(path)

    def load(self, path: str) -> None:
        self.net.load(path)

    def copy_from(self, other: 'AveragePolicyNetwork') -> None:
        self.net.copy_weights_from(other.net)


# ══════════════════════════════════════════════════════════
#  NFSPAgent
# ══════════════════════════════════════════════════════════

class NFSPAgent(RLAgent):
    """
    Neural Fictitious Self-Play 에이전트.

    BR(Best Response) 정책과 ASP(Average Strategy Policy)를 혼합합니다.
      - η 확률: BR(A2C) 정책으로 액션 선택 → Reservoir Buffer에 저장
      - (1-η) 확률: ASP(SL) 정책으로 액션 선택 (저장 안 함)
      - BR 업데이트: 기존 A2C 방식 (RLAgent 상속)
      - ASP 업데이트: N 핸드마다 Reservoir에서 샘플링해 cross-entropy 업데이트

    파라미터:
      eta               : BR 사용 확률 (기본 0.1 — 논문 권장값)
      reservoir_capacity: Reservoir 최대 크기
      sl_batch_size     : SL 업데이트 배치 크기
      sl_update_every   : N 핸드마다 SL 업데이트
    """

    def __init__(
        self,
        player_id:            int,
        name:                 str   = "NFSPAgent",
        eta:                  float = 0.1,
        actor_lr:             float = 3e-4,
        critic_lr:            float = 1e-3,
        asp_lr:               float = 5e-4,
        entropy_coef:         float = 0.02,
        epsilon:              float = 0.06,
        update_every:         int   = 1,
        reservoir_capacity:   int   = 100_000,
        sl_batch_size:        int   = 128,
        sl_update_every:      int   = 5,
        initial_stack:        int   = 1000,
        big_blind:            int   = 10,
        seed:                 Optional[int] = None,
    ):
        super().__init__(
            player_id    = player_id,
            name         = name,
            actor_lr     = actor_lr,
            critic_lr    = critic_lr,
            entropy_coef = entropy_coef,
            epsilon      = epsilon,
            update_every = update_every,
            initial_stack= initial_stack,
            big_blind    = big_blind,
            seed         = seed,
        )

        self.eta            = eta
        self.sl_batch_size  = sl_batch_size
        self.sl_update_every = sl_update_every

        # Average Strategy Policy 네트워크
        self._asp = AveragePolicyNetwork(lr=asp_lr)

        # Reservoir Buffer
        self._reservoir = ReservoirBuffer(
            capacity = reservoir_capacity,
            seed     = seed,
        )

        # SL 업데이트 카운터
        self._hands_since_sl = 0

        # 현재 핸드에서 BR 모드 여부
        self._using_br: bool = True

        # SL 학습 통계
        self.sl_stats: List[Dict] = []

    # ══════════════════════════════════════════════════════
    #  콜백 오버라이드
    # ══════════════════════════════════════════════════════

    def on_round_start(self, round_num: int) -> None:
        """핸드 시작 시 BR / ASP 모드 결정"""
        super().on_round_start(round_num)
        self._using_br = (np.random.random() < self.eta)

    def declare_action(self, obs: AgentObservation) -> Dict[str, Any]:
        """
        η 확률로 BR, (1-η) 확률로 ASP 사용.
        BR 사용 시 (state, action)을 Reservoir에 저장합니다.
        """
        state      = self._encoder.encode(obs, self._profiler)
        valid_mask = build_valid_mask(obs)

        if self._using_br:
            # ── Best Response (A2C) ──────────────────────
            value                    = self._critic.value(state)
            action_idx, log_prob, _  = self._actor.sample_action(state, valid_mask, self.epsilon)
            decision                 = action_idx_to_decision(action_idx, obs)

            # 전환 기록 (on_round_end에서 보상 채움)
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
            self._shaper.on_action(obs, decision['action'], decision['amount'], value)

            # Reservoir Buffer에 BR 결정 저장
            self._reservoir.add(state, action_idx)

        else:
            # ── Average Strategy Policy (SL) ─────────────
            action_idx = self._asp.sample_action(state, valid_mask)
            decision   = action_idx_to_decision(action_idx, obs)
            # ASP 모드: 에피소드 버퍼에 추가하지 않음 (BR 업데이트에서 제외)

        self._record_action(decision['action'])
        return decision

    def on_round_end(self, result: Dict[str, Any]) -> None:
        """핸드 종료 후 BR 업데이트 + 주기적 SL 업데이트"""
        # BR 모드였을 때만 에피소드 버퍼에 데이터가 쌓임
        if self._using_br:
            super().on_round_end(result)
        else:
            # ASP 모드: 상대 프로파일만 업데이트
            self._update_profiler(result)

        # SL 업데이트 스케줄
        self._hands_since_sl += 1
        if (self._hands_since_sl >= self.sl_update_every and
                self._reservoir.size >= self.sl_batch_size):
            sl_loss = self._update_asp()
            self.sl_stats.append({'sl_loss': sl_loss})
            self._hands_since_sl = 0

    # ══════════════════════════════════════════════════════
    #  내부 메서드
    # ══════════════════════════════════════════════════════

    def _update_asp(self) -> float:
        """Reservoir Buffer에서 샘플링 → ASP 지도학습 업데이트"""
        states, actions = self._reservoir.sample(self.sl_batch_size)
        return self._asp.update(states, actions)

    # ── 체크포인트 ──────────────────────────────────────────
    def save(self, checkpoint_dir: str, step: int) -> None:
        """BR(Actor/Critic) + ASP 가중치 저장"""
        super().save(checkpoint_dir, step)
        self._asp.save(os.path.join(checkpoint_dir, f"asp_{self.player_id}_step_{step}.json"))

    def load(self, checkpoint_dir: str, step: int) -> None:
        """BR(Actor/Critic) + ASP 가중치 로드"""
        super().load(checkpoint_dir, step)
        asp_path = os.path.join(checkpoint_dir, f"asp_{self.player_id}_step_{step}.json")
        if os.path.exists(asp_path):
            self._asp.load(asp_path)

    def copy_br_to_asp(self) -> None:
        """BR(Actor) 가중치를 ASP에 복사 (초기화 옵션)"""
        self._asp.net.copy_weights_from(self._actor.net)

    @property
    def reservoir_size(self) -> int:
        return self._reservoir.size
