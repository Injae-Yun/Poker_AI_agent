"""
nfsp.py — Neural Fictitious Self-Play (NFSP) 에이전트  (Phase 2.5 PyTorch 전환)

구성요소:
  ReservoirBuffer  : 과거 (state, seq, seq_len, action) 이력을 균등 확률로 보존
  NFSPAgent        : Best Response(A2C) + Average Policy(SL) 혼합 에이전트
    - η 확률로 Best Response(BR) 정책 사용 → Reservoir에 저장
    - (1-η) 확률로 Average Strategy Policy(ASP) 사용 (저장 안 함)
    - ASP 네트워크: PokerNet (actor head 전용 사용) → PyTorch cross-entropy SL

참고:
  Lanctot et al. (2017), "A Unified Game-Theoretic Approach to Multiagent RL"
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from agents.rl_agent import RLAgent, Transition
from engine.state import AgentObservation
from models.poker_net import PokerNet, make_optimizer, ACTION_DIM, MAX_SEQ_LEN, SEQ_DIM
from models.seq_encoder import encode_betting_history
from reward.reward_shaper import action_idx_to_decision, build_valid_mask
from utils.state_encoder import STATE_DIM


# ══════════════════════════════════════════════════════════
#  ReservoirBuffer  (Phase 2.5: seq / seq_len 추가)
# ══════════════════════════════════════════════════════════

class ReservoirBuffer:
    """
    Reservoir Sampling 기반 경험 버퍼.

    각 항목: (state, seq, seq_len, action_idx)
      state    : np.ndarray (STATE_DIM,)
      seq      : np.ndarray (MAX_SEQ_LEN, SEQ_DIM)
      seq_len  : int
      action_idx: int
    """

    def __init__(self, capacity: int = 100_000, seed: Optional[int] = None):
        self.capacity     = capacity
        self._rng         = np.random.default_rng(seed)
        self._buf: List[Tuple] = []
        self._total_added = 0

    def add(
        self,
        state:      np.ndarray,
        seq:        np.ndarray,
        seq_len:    int,
        action_idx: int,
    ) -> None:
        """O(1) Reservoir Sampling 삽입"""
        item = (state.copy(), seq.copy(), int(seq_len), int(action_idx))
        self._total_added += 1
        if len(self._buf) < self.capacity:
            self._buf.append(item)
        else:
            idx = int(self._rng.integers(0, self._total_added))
            if idx < self.capacity:
                self._buf[idx] = item

    def sample(self, n: int) -> Tuple:
        """
        n개 샘플 반환.
        Returns:
            states   : (n, STATE_DIM)        float32
            seqs     : (n, MAX_SEQ_LEN, SEQ_DIM) float32
            seq_lens : (n,)                  int64
            actions  : (n,)                  int64
        """
        n = min(n, len(self._buf))
        empty_state = np.zeros((0, STATE_DIM),             dtype=np.float32)
        empty_seq   = np.zeros((0, MAX_SEQ_LEN, SEQ_DIM), dtype=np.float32)
        empty_slen  = np.zeros(0,                          dtype=np.int64)
        empty_act   = np.zeros(0,                          dtype=np.int64)

        if n == 0:
            return empty_state, empty_seq, empty_slen, empty_act

        indices  = self._rng.choice(len(self._buf), n, replace=False)
        states   = np.array([self._buf[i][0] for i in indices], dtype=np.float32)
        seqs     = np.array([self._buf[i][1] for i in indices], dtype=np.float32)
        seq_lens = np.array([self._buf[i][2] for i in indices], dtype=np.int64)
        actions  = np.array([self._buf[i][3] for i in indices], dtype=np.int64)
        return states, seqs, seq_lens, actions

    @property
    def size(self) -> int:
        return len(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════
#  NFSPAgent
# ══════════════════════════════════════════════════════════

class NFSPAgent(RLAgent):
    """
    Neural Fictitious Self-Play 에이전트 (PyTorch 기반).

    - BR 정책: 부모 RLAgent의 PokerNet (net)
    - ASP 정책: 별도 PokerNet (asp_net, actor head만 사용)
    - η 확률로 BR, (1-η) 확률로 ASP 선택

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
        lr:                   float = 3e-4,
        asp_lr:               float = 5e-4,
        entropy_coef:         float = 0.02,
        epsilon:              float = 0.06,
        gamma:                float = 0.99,
        update_every:         int   = 1,
        reservoir_capacity:   int   = 100_000,
        sl_batch_size:        int   = 128,
        sl_update_every:      int   = 5,
        initial_stack:        int   = 1000,
        big_blind:            int   = 10,
        device:               str   = 'cpu',
        seed:                 Optional[int] = None,
    ):
        super().__init__(
            player_id     = player_id,
            name          = name,
            lr            = lr,
            entropy_coef  = entropy_coef,
            epsilon       = epsilon,
            gamma         = gamma,
            update_every  = update_every,
            initial_stack = initial_stack,
            big_blind     = big_blind,
            device        = device,
            seed          = seed,
        )

        self.eta             = eta
        self.sl_batch_size   = sl_batch_size
        self.sl_update_every = sl_update_every

        # ── ASP 네트워크 (PokerNet — actor head만 사용) ─────
        self.asp_net  = PokerNet().to(device)
        self.asp_optim = make_optimizer(self.asp_net, lr=asp_lr)

        # ── Reservoir Buffer ────────────────────────────────
        self._reservoir = ReservoirBuffer(capacity=reservoir_capacity, seed=seed)

        # ── 모드 및 카운터 ──────────────────────────────────
        self._using_br:        bool = True
        self._hands_since_sl:  int  = 0

        # ── SL 학습 통계 ────────────────────────────────────
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
        η 확률로 BR(RLAgent), (1-η) 확률로 ASP 사용.
        BR 사용 시 (state, seq, seq_len, action)을 Reservoir에 저장합니다.
        """
        if self._using_br:
            # ── Best Response: 부모 RLAgent 로직 그대로 ───────
            decision = super().declare_action(obs)

            # 마지막으로 저장된 Transition에서 seq 정보 추출 → Reservoir 저장
            if self._hand_transitions:
                t = self._hand_transitions[-1]
                self._reservoir.add(t.state, t.seq, t.seq_len, t.action_idx)
        else:
            # ── Average Strategy Policy (ASP) ─────────────────
            state = self._encoder.encode(obs, self._profiler, self._deck_tracker)
            seq, seq_len = encode_betting_history(
                obs.betting_history,
                self_id=self.player_id,
                initial_stack=self.initial_stack,
            )
            valid_mask = build_valid_mask(obs)

            state_t  = torch.tensor(state,      dtype=torch.float32, device=self.device).unsqueeze(0)
            seq_t    = torch.tensor(seq,        dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t   = torch.tensor(valid_mask, dtype=torch.bool,    device=self.device).unsqueeze(0)
            slen_t   = torch.tensor([seq_len],  dtype=torch.long,    device=self.device)

            with torch.no_grad():
                logits, _ = self.asp_net(state_t, seq_t, slen_t)
                masked    = logits.masked_fill(~mask_t, float('-inf'))
                probs     = F.softmax(masked, dim=-1)
                action_idx = int(torch.multinomial(probs, 1).item())

            decision = action_idx_to_decision(action_idx, obs)
            # ASP 모드: 에피소드 버퍼에 추가하지 않음 (BR 업데이트 제외)

        self._record_action(decision['action'])
        return decision

    def on_round_end(self, result: Dict[str, Any]) -> None:
        """핸드 종료 후 BR 업데이트 + 주기적 SL 업데이트"""
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
    #  ASP SL 업데이트
    # ══════════════════════════════════════════════════════

    def _update_asp(self) -> float:
        """Reservoir에서 샘플링 → ASP cross-entropy SL 업데이트"""
        states, seqs, seq_lens, actions = self._reservoir.sample(self.sl_batch_size)

        states_t   = torch.tensor(states,   dtype=torch.float32, device=self.device)
        seqs_t     = torch.tensor(seqs,     dtype=torch.float32, device=self.device)
        slen_t     = torch.tensor(seq_lens, dtype=torch.long,    device=self.device)
        actions_t  = torch.tensor(actions,  dtype=torch.long,    device=self.device)

        self.asp_net.train()
        logits, _ = self.asp_net(states_t, seqs_t, slen_t)
        loss      = F.cross_entropy(logits, actions_t)

        self.asp_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.asp_net.parameters(), 0.5)
        self.asp_optim.step()
        self.asp_net.eval()

        return loss.item()

    # ══════════════════════════════════════════════════════
    #  체크포인트  (.pth 포맷)
    # ══════════════════════════════════════════════════════

    def save(self, checkpoint_dir: str, step: int) -> None:
        """BR(net) + ASP(asp_net) 가중치 저장"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        # BR 네트워크 (부모 RLAgent 방식)
        net_path = os.path.join(checkpoint_dir, f"net_P{self.player_id}_step_{step}.pth")
        self.net.save(net_path)
        # ASP 네트워크
        asp_path = os.path.join(checkpoint_dir, f"asp_P{self.player_id}_step_{step}.pth")
        self.asp_net.save(asp_path)

    def load(self, checkpoint_dir: str, step: int) -> None:
        """BR(net) + ASP(asp_net) 가중치 로드"""
        net_path = os.path.join(checkpoint_dir, f"net_P{self.player_id}_step_{step}.pth")
        self.net.load(net_path, device=self.device)
        asp_path = os.path.join(checkpoint_dir, f"asp_P{self.player_id}_step_{step}.pth")
        if os.path.exists(asp_path):
            self.asp_net.load(asp_path, device=self.device)

    def copy_br_to_asp(self) -> None:
        """BR 가중치를 ASP에 복사 (초기화 옵션)"""
        self.asp_net.copy_from(self.net)

    @property
    def reservoir_size(self) -> int:
        return self._reservoir.size
