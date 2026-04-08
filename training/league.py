"""
league.py — League Training 관리자  (Phase 2.5 PyTorch 전환)

변경 사항:
  - FrozenAgent : ActorNetwork → PokerNet (actor head만 사용, no_grad 추론)
  - Snapshot    : actor_state(numpy dict) → net_state_dict (torch state_dict)
  - AgentLeague : NFSPAgent 생성 시 actor_lr/critic_lr → lr 인자로 통일
                  maybe_snapshot() : agent._actor → agent.net
                  save() : 에이전트별 파일명 (net_P{i}_step_{N}.pth / asp_P{i}_step_{N}.pth)
                           — BUG FIX: 모든 에이전트가 같은 파일을 덮어쓰던 문제 해결

설계 원칙:
  - 스냅샷은 PokerNet 전체 state_dict를 in-memory로 저장 (파일 저장 없음)
  - 스냅샷 에이전트는 학습하지 않음 (inference only)
  - Exploiter는 매 게임 RuleAgent로 포함 (exploiter_ratio 확률)
"""

import os
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from agents.base_agent import BaseAgent
from agents.rule_agent import RuleAgent
from training.nfsp import NFSPAgent
from models.poker_net import PokerNet, ACTION_DIM
from models.seq_encoder import encode_betting_history
from utils.state_encoder import StateEncoder, STATE_DIM
from utils.opponent_profiler import OpponentProfiler
from reward.reward_shaper import build_valid_mask, action_idx_to_decision


# ══════════════════════════════════════════════════════════
#  FrozenAgent — 학습하지 않는 스냅샷 에이전트
# ══════════════════════════════════════════════════════════

class FrozenAgent(BaseAgent):
    """
    NFSPAgent의 스냅샷으로부터 생성된 읽기 전용 에이전트.
    PokerNet 가중치를 복사해 inference-only로 동작합니다.
    """

    def __init__(
        self,
        player_id:     int,
        net:           PokerNet,
        name:          str = "Frozen",
        initial_stack: int = 1000,
        big_blind:     int = 10,
        device:        str = DEVICE,
    ):
        super().__init__(player_id, name)

        self._net     = net
        self._device  = device
        self._encoder = StateEncoder(initial_stack=initial_stack, big_blind=big_blind)
        self._profiler = OpponentProfiler()
        self.initial_stack = initial_stack

    def declare_action(self, obs) -> Dict[str, Any]:
        state = self._encoder.encode(obs, self._profiler)
        seq, seq_len = encode_betting_history(
            obs.betting_history,
            self_id=self.player_id,
            initial_stack=self.initial_stack,
        )
        valid_mask = build_valid_mask(obs)

        state_t = torch.tensor(state,      dtype=torch.float32, device=self._device).unsqueeze(0)
        seq_t   = torch.tensor(seq,        dtype=torch.float32, device=self._device).unsqueeze(0)
        mask_t  = torch.tensor(valid_mask, dtype=torch.bool,    device=self._device).unsqueeze(0)
        slen_t  = torch.tensor([seq_len],  dtype=torch.long,    device=self._device)

        probs = self._net.actor_probs(state_t, seq_t, mask_t, slen_t)
        action_idx = int(probs.argmax(dim=-1).item())
        decision = action_idx_to_decision(action_idx, obs)
        self._record_action(decision['action'])
        return decision

    def on_game_start(self, config) -> None:
        self._encoder = StateEncoder(
            initial_stack=config.initial_stack,
            big_blind=config.big_blind,
        )
        self.initial_stack = config.initial_stack

    def on_round_end(self, result: Dict[str, Any]) -> None:
        super().on_round_end(result)
        for entry in result.get('betting_history', []):
            if isinstance(entry, dict):
                pid, act = entry['player_id'], entry['action']
                street, amt = entry['street'], entry['amount']
            else:
                pid, act = entry.player_id, entry.action
                street, amt = entry.street, entry.amount
            if pid != self.player_id:
                self._profiler.on_action(pid, act, street, amt, 0, is_blind=(act == 'blind'))


# ══════════════════════════════════════════════════════════
#  Snapshot
# ══════════════════════════════════════════════════════════

@dataclass
class Snapshot:
    """게임 N회 시점의 에이전트 PokerNet 스냅샷"""
    step:            int
    source_id:       int
    net_state_dict:  Dict       # PokerNet.state_dict()  (in-memory)
    eta:             float = 0.1
    avg_chip_delta:  float = 0.0


# ══════════════════════════════════════════════════════════
#  Main 에이전트 구성 (전략 다양성)
# ══════════════════════════════════════════════════════════

MAIN_AGENT_CONFIGS = [
    dict(name="NFSP_Tight",    eta=0.10, epsilon=0.05, entropy_coef=0.01, lr=2e-4,  asp_lr=4e-4),
    dict(name="NFSP_Loose",    eta=0.10, epsilon=0.15, entropy_coef=0.03, lr=4e-4,  asp_lr=8e-4),
    dict(name="NFSP_Balanced", eta=0.15, epsilon=0.10, entropy_coef=0.02, lr=3e-4,  asp_lr=5e-4),
    dict(name="NFSP_Aggr",     eta=0.20, epsilon=0.08, entropy_coef=0.02, lr=3e-4,  asp_lr=5e-4),
]


# ══════════════════════════════════════════════════════════
#  AgentLeague
# ══════════════════════════════════════════════════════════

class AgentLeague:
    """
    League Training 관리자.

    에이전트 구성:
      - main_agents[0..3] : NFSPAgent × 4 (다양한 초기 편향)
      - exploiters[0..1]  : RuleAgent × 2 (취약점 착취용)
      - snapshots         : 과거 PokerNet 스냅샷 풀

    매칭:
      - 4개 슬롯 = main + exploiter + snapshot 조합으로 채움
    """

    def __init__(
        self,
        initial_stack:    int   = 10000,
        small_blind:      int   = 5,
        big_blind:        int   = 10,
        snapshot_every:   int   = 10,
        max_snapshots:    int   = 20,
        exploiter_ratio:  float = 0.5,
        device:           str   = 'cpu',
        seed:             int   = 0,
    ):
        self.initial_stack   = initial_stack
        self.small_blind     = small_blind
        self.big_blind       = big_blind
        self.snapshot_every  = snapshot_every
        self.max_snapshots   = max_snapshots
        self.exploiter_ratio = exploiter_ratio
        self.device          = device
        self._rng            = np.random.default_rng(seed)

        # ── Main 에이전트 생성 ──────────────────────────────
        self.main_agents: List[NFSPAgent] = []
        for i, cfg in enumerate(MAIN_AGENT_CONFIGS):
            agent = NFSPAgent(
                player_id     = i,
                name          = cfg['name'],
                eta           = cfg['eta'],
                epsilon       = cfg['epsilon'],
                entropy_coef  = cfg['entropy_coef'],
                lr            = cfg['lr'],
                asp_lr        = cfg['asp_lr'],
                initial_stack = initial_stack,
                big_blind     = big_blind,
                device        = device,
                seed          = seed + i * 37,
            )
            self.main_agents.append(agent)

        # ── Exploiter 에이전트 (RuleAgent × 2) ─────────────
        self.exploiters: List[RuleAgent] = [
            RuleAgent(player_id=0, name="Exploiter0", seed=seed + 200),
            RuleAgent(player_id=0, name="Exploiter1", seed=seed + 300),
        ]

        # ── 스냅샷 풀 ──────────────────────────────────────
        self._snapshots: List[Snapshot] = []

        # ── 성과 추적 ──────────────────────────────────────
        self._perf: Dict[int, List[float]] = {i: [] for i in range(4)}
        self._game_count = 0

    # ══════════════════════════════════════════════════════
    #  매칭 생성
    # ══════════════════════════════════════════════════════

    def create_matchup(self) -> List[BaseAgent]:
        """4인 게임 매칭 구성."""
        # main_agents 4개 (player_id 재할당)
        agents: List[BaseAgent] = []
        for i, agent in enumerate(self.main_agents):
            agent.player_id = i
            agents.append(agent)

        # exploiter 슬롯 교체
        if self._rng.random() < self.exploiter_ratio:
            slot = int(self._rng.integers(1, 4))
            exploiter = self._rng.choice(self.exploiters)  # type: ignore
            exp_copy = copy.copy(exploiter)
            exp_copy.player_id = slot
            agents[slot] = exp_copy

        # 스냅샷 슬롯 교체 (exploiter와 다른 슬롯)
        if len(self._snapshots) >= 2:
            snap = self._snapshots[int(self._rng.integers(0, len(self._snapshots)))]

            # FrozenAgent에 PokerNet 복원
            net = PokerNet().to(self.device)
            net.load_state_dict(snap.net_state_dict)
            net.eval()

            occupied = {a.player_id for a in agents if not isinstance(a, NFSPAgent)}
            available = [i for i in range(1, 4) if i not in occupied]
            if available:
                slot = int(self._rng.choice(available))
                agents[slot] = FrozenAgent(
                    player_id     = slot,
                    net           = net,
                    name          = f"Snap{snap.step}",
                    initial_stack = self.initial_stack,
                    big_blind     = self.big_blind,
                    device        = self.device,
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
            # PokerNet (BR 네트워크) 가중치 복사 — CPU 텐서로 저장
            state_dict = {k: v.cpu().clone() for k, v in agent.net.state_dict().items()}
            snap = Snapshot(
                step           = game_idx + 1,
                source_id      = agent.player_id,
                net_state_dict = state_dict,
                eta            = agent.eta,
                avg_chip_delta = float(np.mean(self._perf[agent.player_id][-10:]))
                                 if self._perf[agent.player_id] else 0.0,
            )
            self._snapshots.append(snap)

        # 최대 스냅샷 수 유지 (에이전트 4명 × max_snapshots)
        limit = self.max_snapshots * 4
        if len(self._snapshots) > limit:
            self._snapshots = self._snapshots[-limit:]

        return True

    # ══════════════════════════════════════════════════════
    #  성과 기록
    # ══════════════════════════════════════════════════════

    def record_result(self, final_stacks: Dict[int, int]) -> None:
        for i in range(4):
            if i in final_stacks:
                delta = final_stacks[i] - self.initial_stack
                self._perf[i].append(float(delta))
                if len(self._perf[i]) > 100:
                    self._perf[i] = self._perf[i][-100:]

    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for i, agent in enumerate(self.main_agents):
            hist = self._perf[i]
            stats[agent.name] = {
                'player_id':      i,
                'avg_chip_delta': float(np.mean(hist)) if hist else 0.0,
                'std_chip_delta': float(np.std(hist))  if hist else 0.0,
                'win_rate':       sum(1 for d in hist if d > 0) / max(len(hist), 1),
                'games_played':   len(hist),
                'reservoir_size': agent.reservoir_size,
                'eta':            agent.eta,
                'sl_updates':     len(agent.sl_stats),
            }
        return stats

    # ══════════════════════════════════════════════════════
    #  체크포인트 (BUG FIX: 에이전트별 개별 파일명)
    # ══════════════════════════════════════════════════════

    def save(self, checkpoint_dir: str, step: int) -> None:
        """
        리그 전체 체크포인트 저장.

        파일 구조:
          net_P{i}_step_{step}.pth   — agent i의 BR(PokerNet)
          asp_P{i}_step_{step}.pth   — agent i의 ASP(PokerNet)
          league_meta_{step}.json    — 스냅샷 메타 + 게임 카운터
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        for agent in self.main_agents:
            agent.save(checkpoint_dir, step)   # → net_P{i}_step_{step}.pth + asp_P{i}_...

        meta = [{
            'step':           s.step,
            'source_id':      s.source_id,
            'eta':            s.eta,
            'avg_chip_delta': s.avg_chip_delta,
        } for s in self._snapshots[-20:]]
        with open(os.path.join(checkpoint_dir, f"league_meta_{step}.json"), 'w') as f:
            json.dump({'snapshots': meta, 'game_count': self._game_count}, f, indent=2)

    def load(self, checkpoint_dir: str, step: int) -> None:
        """체크포인트 로드 (각 에이전트 개별 로드)"""
        for agent in self.main_agents:
            agent.load(checkpoint_dir, step)   # → net_P{i}_step_{step}.pth + asp_P{i}_...
