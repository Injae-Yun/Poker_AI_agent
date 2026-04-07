"""
poker_net.py — PyTorch 기반 듀얼 브랜치 포커 네트워크

아키텍처 (Phase 2.5 개선):

  정적 피처 (104차원)                GRU 브랜치
  [홀카드 34 + 보드 19 +            베팅 히스토리 시퀀스
   핸드강도 14 + 컨텍스트 9 +        (T × 15차원)
   경제지표 6 + 상대통계 12 +           ↓
   스트리트공격성 4 + 잔여덱 6]     GRU(hidden=64, layers=2)
          ↓                              ↓
   MLP(256→256→128)             last hidden (64차원)
          └──────────── concat(192) ────┘
                              ↓
                         FC(64) + ReLU
                        ↙            ↘
              Actor head(7)      Critic head(1)
              (action logits)    (V(s) value)

Phase 2 대비 개선:
  - numpy → PyTorch (autograd, GPU 지원)
  - 히든 레이어 2→3개 (더 깊은 표현력)
  - GRU 브랜치 추가 (상대 행동 시계열 학습)
  - 상태 벡터 98→104차원 (잔여 덱 정보)
  - LayerNorm 2군데 적용으로 학습 안정성 향상
  - Orthogonal 초기화로 초기 수렴 개선
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ── 전역 상수 ──────────────────────────────────────────────
STATE_DIM   = 104   # 정적 상태 벡터 (98 + 잔여덱 6)
SEQ_DIM     = 15    # 시퀀스 스텝당 특징 수
MAX_SEQ_LEN = 32    # 최대 시퀀스 길이 (초과분은 최신 것만 보존)
GRU_HIDDEN  = 64    # GRU 히든 사이즈
GRU_LAYERS  = 2     # GRU 레이어 수
ACTION_DIM  = 7     # 액션 공간: fold/call/raise×0.5/1.0/1.5/2.0/allin

# 레이즈 사이즈 (팟 대비 배율) — reward_shaper와 일치
RAISE_SIZES = [0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 999.0]


class PokerNet(nn.Module):
    """
    MLP + GRU 듀얼 브랜치 공유 트렁크.

    Actor 헤드와 Critic 헤드를 공유 피처 위에 올립니다.
    forward() 한 번으로 action_logits + value 모두 계산합니다.
    """

    def __init__(
        self,
        state_dim:  int   = STATE_DIM,
        seq_dim:    int   = SEQ_DIM,
        gru_hidden: int   = GRU_HIDDEN,
        gru_layers: int   = GRU_LAYERS,
        action_dim: int   = ACTION_DIM,
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.state_dim  = state_dim
        self.seq_dim    = seq_dim
        self.gru_hidden = gru_hidden
        self.action_dim = action_dim

        # ── MLP 브랜치 (정적 피처) ──────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # ── GRU 브랜치 (베팅 시퀀스) ───────────────────────
        self.gru = nn.GRU(
            input_size  = seq_dim,
            hidden_size = gru_hidden,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )
        self.gru_norm = nn.LayerNorm(gru_hidden)

        # ── 병합 레이어 (MLP 128 + GRU 64 = 192) ──────────
        self.merge = nn.Sequential(
            nn.Linear(128 + gru_hidden, 64),
            nn.ReLU(),
        )

        # ── Actor 헤드 (action logits) ─────────────────────
        self.actor_head = nn.Linear(64, action_dim)

        # ── Critic 헤드 (V(s) 스칼라) ──────────────────────
        self.critic_head = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal 초기화 — 초기 수렴 안정성 향상"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 출력 헤드는 작은 gain으로 초기화 (확률 분포가 초기에 균등하게)
                gain = 0.01 if m.out_features in (self.action_dim, 1) else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)

    # ══════════════════════════════════════════════════════
    #  Forward
    # ══════════════════════════════════════════════════════

    def forward(
        self,
        state:   torch.Tensor,                   # (B, STATE_DIM)
        seq:     torch.Tensor,                   # (B, T, SEQ_DIM)
        seq_len: Optional[torch.Tensor] = None,  # (B,) 실제 유효 길이
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits : (B, ACTION_DIM)  — softmax 전 로짓
            value  : (B, 1)           — V(s)
        """
        # MLP 브랜치
        mlp_out  = self.mlp(state)                  # (B, 128)

        # GRU 브랜치
        gru_out, _ = self.gru(seq)                   # (B, T, GRU_HIDDEN)
        if seq_len is not None:
            # 실제 마지막 유효 스텝의 hidden state 추출
            idx      = (seq_len - 1).clamp(min=0).long()
            gru_last = gru_out[torch.arange(gru_out.size(0)), idx]
        else:
            gru_last = gru_out[:, -1, :]             # (B, GRU_HIDDEN)
        gru_last = self.gru_norm(gru_last)

        # 병합
        merged   = torch.cat([mlp_out, gru_last], dim=-1)   # (B, 192)
        features = self.merge(merged)                         # (B, 64)

        logits = self.actor_head(features)                    # (B, ACTION_DIM)
        value  = self.critic_head(features)                   # (B, 1)
        return logits, value

    # ══════════════════════════════════════════════════════
    #  편의 메서드
    # ══════════════════════════════════════════════════════

    @torch.no_grad()
    def actor_probs(
        self,
        state:      torch.Tensor,           # (B, STATE_DIM)
        seq:        torch.Tensor,           # (B, T, SEQ_DIM)
        valid_mask: torch.Tensor,           # (B, ACTION_DIM) bool
        seq_len:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """마스킹된 softmax 확률 반환 (B, ACTION_DIM)"""
        logits, _ = self.forward(state, seq, seq_len)
        logits = logits.masked_fill(~valid_mask, float('-inf'))
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def get_value(
        self,
        state:   torch.Tensor,
        seq:     torch.Tensor,
        seq_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """V(s) 반환 (B, 1)"""
        _, value = self.forward(state, seq, seq_len)
        return value

    # ══════════════════════════════════════════════════════
    #  1개 샘플 편의 메서드 (declare_action용)
    # ══════════════════════════════════════════════════════

    @torch.no_grad()
    def single_probs(
        self,
        state:      torch.Tensor,   # (STATE_DIM,)
        seq:        torch.Tensor,   # (T, SEQ_DIM)
        valid_mask: torch.Tensor,   # (ACTION_DIM,) bool
        seq_len:    int,
    ) -> torch.Tensor:
        """단일 샘플 확률 벡터 (ACTION_DIM,)"""
        s = state.unsqueeze(0)
        q = seq.unsqueeze(0)
        m = valid_mask.unsqueeze(0)
        l = torch.tensor([seq_len])
        return self.actor_probs(s, q, m, l).squeeze(0)

    @torch.no_grad()
    def single_value(
        self,
        state:   torch.Tensor,   # (STATE_DIM,)
        seq:     torch.Tensor,   # (T, SEQ_DIM)
        seq_len: int,
    ) -> float:
        """단일 샘플 V(s) 스칼라"""
        s = state.unsqueeze(0)
        q = seq.unsqueeze(0)
        l = torch.tensor([seq_len])
        return self.get_value(s, q, l).item()

    # ══════════════════════════════════════════════════════
    #  체크포인트
    # ══════════════════════════════════════════════════════

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = 'cpu') -> None:
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    def copy_from(self, other: 'PokerNet') -> None:
        """다른 PokerNet의 가중치를 복사 (FrozenAgent 스냅샷용)"""
        self.load_state_dict(other.state_dict())


def make_optimizer(net: PokerNet, lr: float = 3e-4) -> torch.optim.Adam:
    return torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
