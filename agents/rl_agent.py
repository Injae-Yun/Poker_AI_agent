"""
rl_agent.py — Actor-Critic (A2C) RL 에이전트  (Phase 2.5 PyTorch 전환)

Phase 2.5 변경 사항:
  - numpy ActorNetwork / CriticNetwork → PyTorch PokerNet (단일 공유 트렁크)
  - 상태 벡터 98 → 104차원 (잔여 덱 정보 추가)
  - 베팅 히스토리 시퀀스 인코딩 추가 (GRU 입력)
  - Transition에 seq / seq_len 필드 추가
  - _update() : torch 텐서 + F.log_softmax + MSE 손실 + 그래디언트 클리핑
  - 체크포인트 : .pth 포맷 (net_P{id}_step_{N}.pth)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from agents.base_agent import BaseAgent
from engine.state import AgentObservation
from models.poker_net import PokerNet, make_optimizer, ACTION_DIM, MAX_SEQ_LEN, SEQ_DIM
from models.seq_encoder import encode_betting_history, INITIAL_STACK
from reward.reward_shaper import RewardShaper, action_idx_to_decision, build_valid_mask
from utils.state_encoder import StateEncoder
from utils.opponent_profiler import OpponentProfiler
from utils.hand_evaluator import equity_by_street


# ── 하이퍼파라미터 기본값 ─────────────────────────────────
GAMMA        = 0.99    # 할인율
ENTROPY_COEF = 0.02    # 엔트로피 보너스
CLIP_GRAD    = 0.5     # 그래디언트 클리핑
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class Transition:
    """단일 결정 스텝의 전환 정보"""
    state:      np.ndarray        # (STATE_DIM,)  float32
    seq:        np.ndarray        # (MAX_SEQ_LEN, SEQ_DIM) float32
    seq_len:    int               # 실제 유효 시퀀스 길이
    action_idx: int
    log_prob:   float
    value:      float             # V(s) at decision time
    reward:     float = 0.0       # on_round_end에서 채워짐
    done:       bool  = False
    valid_mask: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIM, dtype=bool))


class RLAgent(BaseAgent):
    """
    PyTorch 기반 Actor-Critic RL 에이전트.
    PokerNet (MLP + GRU 듀얼 브랜치) 하나로 정책과 가치 함수를 공유합니다.
    핸드 단위로 에피소드 버퍼를 수집하고 업데이트합니다.
    """

    def __init__(
        self,
        player_id:     int,
        name:          str   = "RLAgent",
        lr:            float = 3e-4,
        entropy_coef:  float = ENTROPY_COEF,
        epsilon:       float = 0.10,   # ε-greedy 탐색 비율
        gamma:         float = GAMMA,
        clip_grad:     float = CLIP_GRAD,
        update_every:  int   = 1,      # N 핸드마다 업데이트
        initial_stack: int   = 1000,
        big_blind:     int   = 10,
        device:        str   = DEVICE,
        seed:          Optional[int] = None,
    ):
        super().__init__(player_id, name)

        self.entropy_coef = entropy_coef
        self.epsilon      = epsilon
        self.gamma        = gamma
        self.clip_grad    = clip_grad
        self.update_every = update_every
        self.device       = device
        self.initial_stack = initial_stack

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ── 네트워크 + 옵티마이저 ─────────────────────────
        self.net  = PokerNet().to(device)
        self.optim = make_optimizer(self.net, lr=lr)

        # ── 보상 설계 ──────────────────────────────────────
        self._shaper = RewardShaper(bb_scale=initial_stack / 100)

        # ── 상태 인코더 ────────────────────────────────────
        self._encoder = StateEncoder(initial_stack=initial_stack, big_blind=big_blind)

        # ── 상대 프로파일러 ────────────────────────────────
        self._profiler = OpponentProfiler()

        # ── 버퍼 ──────────────────────────────────────────
        self._episode_buffer:   List[Transition] = []
        self._hand_transitions: List[Transition] = []
        self._hand_baseline_vs: List[float]      = []

        # ── 덱 트래커 (on_round_start에서 주입) ───────────
        self._deck_tracker = None

        # ── 업데이트 스케줄 ────────────────────────────────
        self._hands_since_update = 0

        # ── 학습 통계 ─────────────────────────────────────
        self.train_stats: List[Dict] = []

    # ══════════════════════════════════════════════════════
    #  BaseAgent 콜백
    # ══════════════════════════════════════════════════════

    def on_game_start(self, config) -> None:
        self._encoder = StateEncoder(
            initial_stack=config.initial_stack,
            big_blind=config.big_blind,
        )
        self.initial_stack = config.initial_stack
        self._shaper = RewardShaper(bb_scale=config.initial_stack / 100)

    def on_round_start(self, round_num: int) -> None:
        super().on_round_start(round_num)
        self._shaper.reset()
        self._hand_transitions = []
        self._hand_baseline_vs = []

    def set_deck_tracker(self, deck_tracker) -> None:
        """게임 엔진에서 DeckTracker를 주입할 때 사용"""
        self._deck_tracker = deck_tracker

    def declare_action(self, obs: AgentObservation) -> Dict[str, Any]:
        """상태 인코딩 → 정책 샘플링 → 액션 반환"""
        # equity 1회만 계산 (StateEncoder + RewardShaper 공유 → MC 중복 제거)
        num_opp = max(1, obs.active_players - 1)
        equity = equity_by_street(obs.hole_cards, obs.community_cards, num_opp)

        # 상태 벡터 (104차원) — 계산된 equity 주입
        state = self._encoder.encode(obs, self._profiler, self._deck_tracker,
                                     precomputed_equity=equity)

        # 시퀀스 인코딩 (베팅 히스토리 → GRU 입력)
        seq, seq_len = encode_betting_history(
            obs.betting_history,
            self_id=self.player_id,
            initial_stack=self.initial_stack,
        )

        # 유효 액션 마스크
        valid_mask = build_valid_mask(obs)   # np.ndarray (ACTION_DIM,) bool

        # 추론 (no_grad)
        state_t  = torch.tensor(state,      dtype=torch.float32, device=self.device).unsqueeze(0)
        seq_t    = torch.tensor(seq,        dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t   = torch.tensor(valid_mask, dtype=torch.bool,    device=self.device).unsqueeze(0)
        slen_t   = torch.tensor([seq_len],  dtype=torch.long,    device=self.device)

        with torch.no_grad():
            logits, value_t = self.net(state_t, seq_t, slen_t)
            value = value_t.item()

            # ε-greedy
            if np.random.random() < self.epsilon:
                valid_idxs = np.where(valid_mask)[0]
                action_idx = int(np.random.choice(valid_idxs))
                # log_prob 계산 (선택한 액션에 대한 실제 log prob)
                masked_logits = logits.masked_fill(~mask_t, float('-inf'))
                log_probs = F.log_softmax(masked_logits, dim=-1)
                log_prob = log_probs[0, action_idx].item()
            else:
                masked_logits = logits.masked_fill(~mask_t, float('-inf'))
                probs     = F.softmax(masked_logits, dim=-1)
                log_probs = F.log_softmax(masked_logits, dim=-1)
                action_idx = int(torch.multinomial(probs, 1).item())
                log_prob   = log_probs[0, action_idx].item()

        decision = action_idx_to_decision(action_idx, obs)

        # 전환 기록
        self._hand_transitions.append(Transition(
            state      = state,
            seq        = seq,
            seq_len    = seq_len,
            action_idx = action_idx,
            log_prob   = log_prob,
            value      = value,
            reward     = 0.0,
            done       = False,
            valid_mask = valid_mask,
        ))
        self._hand_baseline_vs.append(value)

        self._shaper.on_action(obs, decision['action'], decision['amount'], value,
                               precomputed_equity=equity)
        self._record_action(decision['action'])
        return decision

    def on_round_end(self, result: Dict[str, Any]) -> None:
        super().on_round_end(result)

        chip_delta    = result.get('chip_delta', 0)
        final_rewards = self._shaper.on_hand_end(chip_delta, self._hand_baseline_vs)

        for i, t in enumerate(self._hand_transitions):
            t.reward = final_rewards[i] if i < len(final_rewards) else 0.0
            t.done   = (i == len(self._hand_transitions) - 1)

        self._episode_buffer.extend(self._hand_transitions)
        self._update_profiler(result)

        self._hands_since_update += 1
        if self._hands_since_update >= self.update_every and self._episode_buffer:
            stats = self._update()
            self.train_stats.append(stats)
            self._hands_since_update = 0

    def on_game_end(self) -> None:
        if self._episode_buffer:
            self._update()
            self._hands_since_update = 0

    # ══════════════════════════════════════════════════════
    #  내부 메서드
    # ══════════════════════════════════════════════════════

    def _update(self) -> Dict[str, float]:
        """Actor-Critic 업데이트 (PyTorch). 버퍼 clear 후 손실 딕셔너리 반환."""
        buf = self._episode_buffer
        B   = len(buf)

        # ── numpy → 텐서 배치 ─────────────────────────────
        states      = torch.tensor(np.array([t.state      for t in buf]),
                                   dtype=torch.float32, device=self.device)
        seqs        = torch.tensor(np.array([t.seq        for t in buf]),
                                   dtype=torch.float32, device=self.device)
        seq_lens    = torch.tensor([t.seq_len    for t in buf],
                                   dtype=torch.long,    device=self.device)
        action_idxs = torch.tensor([t.action_idx for t in buf],
                                   dtype=torch.long,    device=self.device)
        rewards     = torch.tensor([t.reward     for t in buf],
                                   dtype=torch.float32, device=self.device)
        dones       = torch.tensor([t.done       for t in buf],
                                   dtype=torch.float32, device=self.device)
        valid_masks = torch.tensor(np.array([t.valid_mask for t in buf]),
                                   dtype=torch.bool,    device=self.device)

        # ── Discounted Returns 계산 ───────────────────────
        returns = self._compute_returns(rewards, dones)  # (B,)

        # ── Forward pass ──────────────────────────────────
        logits, values = self.net(states, seqs, seq_lens)  # (B,7), (B,1)
        values = values.squeeze(-1)                         # (B,)

        # ── Advantages ───────────────────────────────────
        advantages = (returns - values.detach())
        # 정규화 (분산이 0이면 skip)
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Actor 손실 (Policy Gradient + Entropy) ────────
        masked_logits = logits.masked_fill(~valid_masks, float('-inf'))
        log_probs_all = F.log_softmax(masked_logits, dim=-1)       # (B, ACTION_DIM)
        log_probs_act = log_probs_all.gather(1, action_idxs.unsqueeze(1)).squeeze(1)  # (B,)

        probs_all  = F.softmax(masked_logits, dim=-1)
        entropy    = -(probs_all * log_probs_all.clamp(min=-1e9)).sum(dim=-1).mean()

        actor_loss  = -(log_probs_act * advantages).mean() - self.entropy_coef * entropy

        # ── Critic 손실 (MSE) ─────────────────────────────
        critic_loss = F.mse_loss(values, returns)

        # ── 통합 손실 + 역전파 ────────────────────────────
        total_loss = actor_loss + 0.5 * critic_loss
        self.optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)
        self.optim.step()

        self._episode_buffer.clear()
        return {
            'actor_loss':  actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy':     entropy.item(),
        }

    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Monte-Carlo discounted returns (역방향 누적)"""
        B = rewards.size(0)
        returns = torch.zeros(B, dtype=torch.float32, device=self.device)
        R = 0.0
        for t in reversed(range(B)):
            R = rewards[t].item() + self.gamma * R * (1.0 - dones[t].item())
            returns[t] = R
        return returns

    def _update_profiler(self, result: Dict[str, Any]) -> None:
        for entry in result.get('betting_history', []):
            if isinstance(entry, dict):
                pid, act = entry['player_id'], entry['action']
                street, amt = entry['street'], entry['amount']
            else:
                pid, act = entry.player_id, entry.action
                street, amt = entry.street, entry.amount

            if pid != self.player_id:
                self._profiler.on_action(
                    player_id=pid, action=act,
                    street=street, amount=amt,
                    pot=0, is_blind=(act == 'blind'),
                )

        if result.get('showdown'):
            for pid in result.get('player_hands', {}).keys():
                if pid != self.player_id:
                    won = pid in result.get('winners', [])
                    self._profiler.on_showdown(pid, won)

    # ══════════════════════════════════════════════════════
    #  체크포인트  (.pth 포맷)
    # ══════════════════════════════════════════════════════

    def save(self, checkpoint_dir: str, step: int) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"net_P{self.player_id}_step_{step}.pth")
        self.net.save(path)

    def load(self, checkpoint_dir: str, step: int) -> None:
        path = os.path.join(checkpoint_dir, f"net_P{self.player_id}_step_{step}.pth")
        self.net.load(path, device=self.device)
