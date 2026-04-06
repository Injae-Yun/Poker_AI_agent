"""
state.py — 게임 상태 표현

PlayerState : 개별 플레이어의 공개 상태
GameState   : 특정 에이전트에게 전달되는 관찰(observation)
              → 에이전트는 자신의 홀 카드만 볼 수 있음
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from engine.card import Card


# ── 베팅 액션 상수 ─────────────────────────────────────────
ACTION_FOLD  = 'fold'
ACTION_CALL  = 'call'
ACTION_RAISE = 'raise'
ACTION_CHECK = 'check'

STREETS = ['preflop', 'flop', 'turn', 'river']


@dataclass
class BettingAction:
    """단일 베팅 액션 기록"""
    player_id:  int
    street:     str
    action:     str         # fold / call / raise / check
    amount:     int         # 해당 액션으로 추가로 낸 칩 (fold/check = 0)
    pot_at_action: int      # 액션 시점의 팟 크기
    stack_at_action: int    # 액션 시점의 본인 스택


@dataclass
class PlayerState:
    """
    에이전트에게 공개되는 상대 플레이어 정보.
    홀 카드는 포함하지 않습니다.
    """
    player_id:    int
    stack:        int
    is_active:    bool      # 폴드하지 않은 상태
    is_all_in:    bool
    bet_this_street: int    # 현재 스트리트에서 낸 총 금액
    total_bet:    int       # 현재 핸드에서 낸 총 금액


@dataclass
class AgentObservation:
    """
    각 에이전트가 의사결정 시 받는 관찰 정보.
    정보 은닉 원칙: hole_cards는 해당 에이전트 본인의 카드만 포함.
    """
    # ── 본인 정보 ──────────────────────────────────────────
    player_id:      int
    hole_cards:     List[Card]          # 본인의 홀 카드 2장
    stack:          int
    position:       int                 # 딜러 기준 포지션 (0=딜러, 1=SB, 2=BB, ...)

    # ── 공개 정보 ──────────────────────────────────────────
    community_cards: List[Card]         # 현재까지 공개된 커뮤니티 카드
    pot:            int
    street:         str                 # preflop / flop / turn / river
    round_num:      int

    # ── 베팅 정보 ──────────────────────────────────────────
    call_amount:    int                 # 콜하려면 추가로 내야 하는 금액
    min_raise:      int                 # 최소 레이즈 금액 (추가분)
    max_raise:      int                 # 최대 레이즈 금액 (올인 금액)
    raises_this_street: int             # 현 스트리트 레이즈 횟수 (상한 계산용)

    # ── 상대 플레이어 공개 상태 ────────────────────────────
    players:        List[PlayerState]   # 모든 플레이어의 공개 상태

    # ── 배팅 히스토리 ──────────────────────────────────────
    betting_history: List[BettingAction] = field(default_factory=list)

    # ── 유틸리티 ──────────────────────────────────────────
    @property
    def active_players(self) -> int:
        return sum(1 for p in self.players if p.is_active)

    @property
    def valid_actions(self) -> List[Dict[str, Any]]:
        """
        현재 상황에서 가능한 액션 목록.
        각 항목: {'action': str, 'amount': int or None}
        """
        actions = []

        # 폴드는 항상 가능
        actions.append({'action': ACTION_FOLD, 'amount': 0})

        if self.call_amount == 0:
            # 콜 금액이 0이면 체크 가능
            actions.append({'action': ACTION_CHECK, 'amount': 0})
        else:
            # 콜
            call_amt = min(self.call_amount, self.stack)
            actions.append({'action': ACTION_CALL, 'amount': call_amt})

        # 레이즈 (스택이 충분할 때)
        if self.stack > self.call_amount and self.min_raise <= self.max_raise:
            actions.append({
                'action': ACTION_RAISE,
                'amount': self.min_raise,   # 기본값; 에이전트가 amount를 조정 가능
                'min_amount': self.min_raise,
                'max_amount': self.max_raise,
            })

        return actions
