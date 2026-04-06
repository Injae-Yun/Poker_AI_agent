"""
actor.py — 정책 네트워크 (Actor)

입력: 상태 벡터 (STATE_DIM)
출력: 각 액션의 확률 분포 (ACTION_DIM)

액션 공간:
  0: fold
  1: check / call
  2: raise × 0.5 pot
  3: raise × 1.0 pot
  4: raise × 1.5 pot
  5: raise × 2.0 pot
  6: all-in
"""

import numpy as np
from typing import Tuple, List
from models.nn import Sequential, Linear, ReLU, LayerNorm, Adam, Softmax

STATE_DIM  = 98
ACTION_DIM = 7

# 레이즈 사이즈 매핑 (팟 대비 배율)
RAISE_SIZES = [0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 999.0]  # 0,1은 fold/call, 6은 all-in


def build_actor(state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> Sequential:
    """
    Actor 네트워크 구조:
      FC(256) → LayerNorm → ReLU
      FC(128) → ReLU
      FC(action_dim) → Softmax
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


class ActorNetwork:
    """
    정책 네트워크 래퍼.
    행동 선택, log-probability, 엔트로피를 제공합니다.
    """

    def __init__(
        self,
        state_dim:  int   = STATE_DIM,
        action_dim: int   = ACTION_DIM,
        lr:         float = 3e-4,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.net        = build_actor(state_dim, action_dim)
        self.optimizer  = Adam(lr=lr, clip=0.5)
        self._last_probs: np.ndarray = np.ones(action_dim) / action_dim

    # ── 추론 ───────────────────────────────────────────────
    def probs(self, state: np.ndarray) -> np.ndarray:
        """상태 벡터 → 액션 확률 분포"""
        p = self.net.forward(state.astype(np.float32))
        self._last_probs = p
        return p

    def sample_action(
        self,
        state:      np.ndarray,
        valid_mask: np.ndarray,       # (ACTION_DIM,) bool, 합법 액션만 True
        epsilon:    float = 0.0,      # ε-greedy 탐색
    ) -> Tuple[int, float, float]:
        """
        합법 액션 마스크를 고려하여 액션을 샘플링합니다.

        Returns:
            (action_idx, log_prob, entropy)
        """
        p = self.probs(state)

        # 비합법 액션 마스킹
        masked = p * valid_mask
        total  = masked.sum()
        if total < 1e-8:
            masked = valid_mask.astype(float)
            total  = masked.sum()
        masked /= total

        # ε-greedy 탐색
        if epsilon > 0 and np.random.random() < epsilon:
            valid_indices = np.where(valid_mask)[0]
            action_idx    = np.random.choice(valid_indices)
        else:
            action_idx = int(np.random.choice(len(masked), p=masked))

        log_prob = float(np.log(masked[action_idx] + 1e-8))
        entropy  = float(-np.sum(masked * np.log(masked + 1e-8)))

        return action_idx, log_prob, entropy

    def greedy_action(
        self,
        state:      np.ndarray,
        valid_mask: np.ndarray,
    ) -> int:
        """그리디 액션 (평가 시 사용)"""
        p = self.probs(state) * valid_mask
        if p.sum() < 1e-8:
            return int(np.argmax(valid_mask))
        return int(np.argmax(p))

    # ── 업데이트 ───────────────────────────────────────────
    def update(
        self,
        states:       np.ndarray,      # (T, STATE_DIM)
        action_idxs:  np.ndarray,      # (T,)
        advantages:   np.ndarray,      # (T,)
        valid_masks:  np.ndarray,      # (T, ACTION_DIM)
        entropy_coef: float = 0.01,
    ) -> float:
        """
        Policy Gradient 업데이트.
        Returns: 평균 actor loss
        """
        T = len(states)
        total_loss = 0.0

        for t in range(T):
            p = self.net.forward(states[t])

            # 마스킹
            masked = p * valid_masks[t]
            s = masked.sum()
            if s < 1e-8:
                continue
            masked /= s

            a_idx = action_idxs[t]
            adv   = advantages[t]

            log_p   = np.log(masked[a_idx] + 1e-8)
            entropy = float(-np.sum(masked * np.log(masked + 1e-8)))

            # 손실 = -log_prob × advantage - entropy_coef × entropy
            loss = -(log_p * adv) - entropy_coef * entropy
            total_loss += loss

            # 그래디언트: softmax output에 대한 손실 기울기
            grad_softmax = masked.copy()
            grad_softmax[a_idx] -= 1.0   # cross-entropy gradient trick
            grad_softmax = grad_softmax * (-adv) - entropy_coef * (-np.log(masked + 1e-8) - 1)

            self.net.backward(grad_softmax)
            self.optimizer.step(self.net)

        return total_loss / max(T, 1)

    # ── 직렬화 ────────────────────────────────────────────
    def save(self, path: str) -> None:
        self.net.save(path)

    def load(self, path: str) -> None:
        self.net.load(path)

    def copy_from(self, other: 'ActorNetwork') -> None:
        self.net.copy_weights_from(other.net)
