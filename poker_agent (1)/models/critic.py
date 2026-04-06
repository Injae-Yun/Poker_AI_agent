"""
critic.py — 가치 네트워크 (Critic)

입력: 상태 벡터 (STATE_DIM)
출력: V(s) — 현재 상태의 기대 보상 (스칼라)

역할:
  - Advantage 계산의 베이스라인: A(s,a) = Q(s,a) - V(s)
  - 폴드 포함 모든 액션의 결정 품질을 올바르게 평가
"""

import numpy as np
from typing import Tuple
from models.nn import Sequential, Linear, ReLU, LayerNorm, Adam, mse_loss

STATE_DIM = 98


def build_critic(state_dim: int = STATE_DIM) -> Sequential:
    """
    Critic 네트워크 구조:
      FC(256) → LayerNorm → ReLU
      FC(128) → ReLU
      FC(1)
    """
    return Sequential([
        Linear(state_dim, 256),
        LayerNorm(256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 1),
    ])


class CriticNetwork:
    """
    가치 네트워크 래퍼.
    V(s) 추정과 MSE 기반 업데이트를 제공합니다.
    """

    def __init__(
        self,
        state_dim: int   = STATE_DIM,
        lr:        float = 1e-3,
        gamma:     float = 0.99,
    ):
        self.state_dim = state_dim
        self.gamma     = gamma
        self.net       = build_critic(state_dim)
        self.optimizer = Adam(lr=lr, clip=1.0)

    # ── 추론 ───────────────────────────────────────────────
    def value(self, state: np.ndarray) -> float:
        """V(s) 추정"""
        v = self.net.forward(state.astype(np.float32))
        return float(v.flatten()[0])

    def value_batch(self, states: np.ndarray) -> np.ndarray:
        """배치 V(s) 추정 — (T, STATE_DIM) → (T,)"""
        return np.array([self.value(s) for s in states])

    # ── Advantage 계산 ─────────────────────────────────────
    def compute_advantages(
        self,
        states:   np.ndarray,    # (T, STATE_DIM)
        rewards:  np.ndarray,    # (T,)
        dones:    np.ndarray,    # (T,) bool — 에피소드 종료 여부
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Returns + Advantage 계산.

        Returns:
            (returns, advantages)
            returns    : (T,) 할인 누적 보상
            advantages : (T,) A(s,a) = returns - V(s) (정규화됨)
        """
        T       = len(rewards)
        values  = self.value_batch(states)
        returns = np.zeros(T)

        # 뒤에서 앞으로 할인 누적 보상 계산
        G = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                G = 0.0
            G = rewards[t] + self.gamma * G
            returns[t] = G

        advantages = returns - values

        # Advantage 정규화 (학습 안정성)
        if len(advantages) > 1:
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / std

        return returns, advantages

    # ── 업데이트 ───────────────────────────────────────────
    def update(
        self,
        states:  np.ndarray,    # (T, STATE_DIM)
        targets: np.ndarray,    # (T,) — Monte Carlo returns
    ) -> float:
        """
        MSE 손실로 Critic 업데이트.
        Returns: 평균 critic loss
        """
        total_loss = 0.0

        for t in range(len(states)):
            v_pred = self.net.forward(states[t].astype(np.float32))
            target = np.array([targets[t]], dtype=np.float32)

            loss, grad = mse_loss(v_pred, target)
            total_loss += loss

            self.net.backward(grad)
            self.optimizer.step(self.net)

        return total_loss / max(len(states), 1)

    # ── 직렬화 ────────────────────────────────────────────
    def save(self, path: str) -> None:
        self.net.save(path)

    def load(self, path: str) -> None:
        self.net.load(path)

    def copy_from(self, other: 'CriticNetwork') -> None:
        self.net.copy_weights_from(other.net)
