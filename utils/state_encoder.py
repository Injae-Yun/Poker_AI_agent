"""
state_encoder.py — AgentObservation → 104차원 상태 벡터 변환

인덱스 구조 (합계 = 104):
  [ 0: 34] 홀카드 인코딩  (카드1 rank×13 + suit×4 + 카드2 rank×13 + suit×4)
  [34: 53] 보드 인코딩    (rank 존재×13 + suit 비율×4 + pair/trips 플래그×2)
  [53: 67] 핸드 강도      (equity×1 + preflop_strength×1 + hand_rank×10 + suited/paired×2)
  [67: 76] 컨텍스트       (street×4 + position×4 + active_players×1)
  [76: 82] 경제 지표      (my_stack + pot + pot_odds + call_norm + raises + stack_bb)
  [82: 94] 상대 통계      (3명 × 4 = vpip, pfr, af_norm, est_strength)
  [94: 98] 스트리트 공격성 (4 스트리트 × 1)
  [98:104] 잔여 덱 정보   (suit_remaining×4 + high_card_remaining×1 + total_remaining×1)

Phase 2.5: 98 → 104차원 (잔여 덱 정보 6차원 추가)
"""

import numpy as np
from typing import List, Optional

from engine.state import AgentObservation
from engine.hand_eval import evaluate_hand, HAND_RANK
from utils.hand_evaluator import preflop_strength, equity_by_street
from utils.opponent_profiler import OpponentProfiler


RANKS  = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS  = ['s', 'h', 'd', 'c']
STREETS = ['preflop', 'flop', 'turn', 'river']

# 하이카드로 간주하는 랭크 (A, K, Q, J, T) — 총 5×4=20장
_HIGH_RANKS = {'A', 'K', 'Q', 'J', 'T'}

STATE_DIM = 104   # Phase 2.5: 98 → 104

# HAND_RANK: HIGH_CARD=1 … ROYAL_FLUSH=10 → 인덱스 0~9
_HAND_RANK_ORDER = [
    'HIGH_CARD', 'ONE_PAIR', 'TWO_PAIR', 'THREE_OF_A_KIND',
    'STRAIGHT', 'FLUSH', 'FULL_HOUSE', 'FOUR_OF_A_KIND',
    'STRAIGHT_FLUSH', 'ROYAL_FLUSH',
]


class StateEncoder:
    """
    AgentObservation을 (104,) float32 numpy 배열로 변환합니다.
    모든 값은 [0, 1] 범위로 정규화됩니다.

    Phase 2.5: DeckTracker를 옵셔널로 받아 잔여 덱 정보(6차원)를 추가.
    deck_tracker=None이면 0.5 사전확률로 채움.
    """

    def __init__(self, initial_stack: int = 1000, big_blind: int = 10):
        self.initial_stack = initial_stack
        self.big_blind = big_blind

    # ══════════════════════════════════════════════════════
    #  공개 메서드
    # ══════════════════════════════════════════════════════

    def encode(
        self,
        obs: AgentObservation,
        profiler: Optional[OpponentProfiler] = None,
        deck_tracker=None,   # Optional[DeckTracker]  — 순환 임포트 방지를 위해 타입 힌트 생략
    ) -> np.ndarray:
        """AgentObservation → (104,) float32"""
        parts = [
            self._encode_hole_cards(obs.hole_cards),          # 34
            self._encode_board(obs.community_cards),           # 19
            self._encode_hand_strength(obs),                   # 14
            self._encode_context(obs),                         # 9
            self._encode_economics(obs),                       # 6
            self._encode_opponents(obs, profiler),             # 12
            self._encode_street_aggression(obs),               # 4
            self._encode_remaining_deck(deck_tracker),         # 6
        ]
        vec = np.concatenate(parts).astype(np.float32)

        assert vec.shape == (STATE_DIM,), f"shape 오류: {vec.shape}"
        assert not np.isnan(vec).any(), "NaN 발생"
        assert not np.isinf(vec).any(), "Inf 발생"
        return vec

    # ══════════════════════════════════════════════════════
    #  내부 헬퍼
    # ══════════════════════════════════════════════════════

    def _encode_card(self, card) -> np.ndarray:
        """카드 한 장 → (17,) [rank×13 + suit×4]"""
        vec = np.zeros(17, dtype=np.float32)
        if card is None:
            return vec
        rank_idx = RANKS.index(card.rank)
        suit_idx = SUITS.index(card.suit)
        vec[rank_idx] = 1.0
        vec[13 + suit_idx] = 1.0
        return vec

    def _encode_hole_cards(self, hole_cards) -> np.ndarray:
        """홀카드 2장 → (34,), rank 내림차순 정렬"""
        sorted_cards = sorted(hole_cards, key=lambda c: RANKS.index(c.rank), reverse=True)
        c1 = sorted_cards[0] if len(sorted_cards) > 0 else None
        c2 = sorted_cards[1] if len(sorted_cards) > 1 else None
        return np.concatenate([self._encode_card(c1), self._encode_card(c2)])

    def _encode_board(self, community_cards) -> np.ndarray:
        """커뮤니티 카드 → (19,) [rank 존재×13 + suit 비율×4 + pair/trips×2]"""
        vec = np.zeros(19, dtype=np.float32)
        if not community_cards:
            return vec

        rank_counts = {}
        suit_counts = {}
        for card in community_cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

        # rank 존재 여부 (0:13)
        for i, r in enumerate(RANKS):
            if rank_counts.get(r, 0) > 0:
                vec[i] = 1.0

        # suit 비율 (13:17)
        for i, s in enumerate(SUITS):
            vec[13 + i] = suit_counts.get(s, 0) / 5.0

        # pair/trips 플래그 (17:19)
        max_count = max(rank_counts.values()) if rank_counts else 0
        vec[17] = 1.0 if max_count >= 2 else 0.0
        vec[18] = 1.0 if max_count >= 3 else 0.0

        return vec

    def _encode_hand_strength(self, obs: AgentObservation) -> np.ndarray:
        """핸드 강도 → (14,) [equity + preflop_str + hand_rank×10 + suited + paired]"""
        vec = np.zeros(14, dtype=np.float32)
        num_opp = max(1, obs.active_players - 1)

        # equity (0)
        equity = equity_by_street(obs.hole_cards, obs.community_cards, num_opp)
        vec[0] = float(np.clip(equity, 0.0, 1.0))

        # preflop strength (1)
        vec[1] = float(np.clip(preflop_strength(obs.hole_cards), 0.0, 1.0))

        # 현재 핸드 랭크 one-hot (2:12) — 플롭 이후만
        if len(obs.community_cards) >= 3:
            all_cards = obs.hole_cards + obs.community_cards
            score, _, _ = evaluate_hand(all_cards)
            rank_idx = score - 1  # 1~10 → 0~9
            if 0 <= rank_idx < 10:
                vec[2 + rank_idx] = 1.0

        # suited flag (12)
        if len(obs.hole_cards) == 2:
            vec[12] = 1.0 if obs.hole_cards[0].suit == obs.hole_cards[1].suit else 0.0
            # paired flag (13)
            vec[13] = 1.0 if obs.hole_cards[0].rank == obs.hole_cards[1].rank else 0.0

        return vec

    def _encode_context(self, obs: AgentObservation) -> np.ndarray:
        """컨텍스트 → (9,) [street×4 + position×4 + active_players]"""
        vec = np.zeros(9, dtype=np.float32)

        # street one-hot (0:4)
        if obs.street in STREETS:
            vec[STREETS.index(obs.street)] = 1.0

        # position one-hot (4:8) — 0=딜러, 1=SB, 2=BB, 3=기타
        pos = min(obs.position, 3)
        vec[4 + pos] = 1.0

        # active players (8)
        vec[8] = obs.active_players / 4.0

        return vec

    def _encode_economics(self, obs: AgentObservation) -> np.ndarray:
        """경제 지표 → (6,)"""
        vec = np.zeros(6, dtype=np.float32)

        # 내 스택 정규화 (0): initial_stack 기준, 최대 3배까지 허용 후 /3
        stack_ratio = obs.stack / max(self.initial_stack, 1)
        vec[0] = float(np.clip(stack_ratio / 3.0, 0.0, 1.0))

        # 팟 정규화 (1)
        max_pot = 4 * self.initial_stack
        vec[1] = float(np.clip(obs.pot / max(max_pot, 1), 0.0, 1.0))

        # 팟 오즈 (2)
        denom = obs.pot + obs.call_amount + 1
        vec[2] = float(obs.call_amount / denom)

        # 콜 금액 정규화 (3)
        vec[3] = float(np.clip(obs.call_amount / max(obs.stack + 1, 1), 0.0, 1.0))

        # 이번 스트리트 레이즈 횟수 (4)
        vec[4] = obs.raises_this_street / 4.0

        # 스택 / BB 비율 (5)
        vec[5] = float(np.clip(obs.stack / max(self.big_blind * 100, 1), 0.0, 1.0))

        return vec

    def _encode_opponents(
        self,
        obs: AgentObservation,
        profiler: Optional[OpponentProfiler],
    ) -> np.ndarray:
        """상대 통계 → (12,) [3명 × (vpip, pfr, af_norm, est_strength)]"""
        vec = np.zeros(12, dtype=np.float32)

        # player_id 오름차순으로 상대 3명 슬롯 할당
        opp_ids = sorted(
            [p.player_id for p in obs.players if p.player_id != obs.player_id]
        )

        for slot, pid in enumerate(opp_ids[:3]):
            base = slot * 4
            if profiler is not None:
                profile = profiler.get_profile(pid)
                af = profile.af
                af_norm = af / (af + 1.0)  # [0, 1) 정규화
                vec[base + 0] = float(np.clip(profile.vpip, 0.0, 1.0))
                vec[base + 1] = float(np.clip(profile.pfr,  0.0, 1.0))
                vec[base + 2] = float(np.clip(af_norm,      0.0, 1.0))
                vec[base + 3] = float(np.clip(profile.estimated_hand_strength, 0.0, 1.0))
            else:
                # profiler 없을 때 사전확률
                vec[base + 0] = 0.25  # vpip 사전확률
                vec[base + 1] = 0.15  # pfr 사전확률
                vec[base + 2] = 0.5   # af_norm 중립
                vec[base + 3] = 0.5   # est_strength 중립

        return vec

    def _encode_remaining_deck(self, deck_tracker) -> np.ndarray:
        """
        잔여 덱 정보 → (6,)

        [0..3] suit_remaining : 각 수트별 잔여 카드 수 / 13.0
        [4]    high_remaining  : 하이카드(A/K/Q/J/T) 잔여 수 / 20.0
        [5]    total_remaining : 전체 잔여 수 / 52.0
        """
        vec = np.zeros(6, dtype=np.float32)

        if deck_tracker is None:
            # 사전확률 (아직 카드를 보지 못한 상태)
            vec[0:4] = 1.0    # 13/13 → 1.0
            vec[4]   = 1.0    # 20/20 → 1.0
            vec[5]   = 1.0    # 52/52 → 1.0
            return vec

        remaining = deck_tracker.remaining
        if not remaining:
            return vec

        suit_counts = {s: 0 for s in SUITS}
        high_count  = 0
        for card in remaining:
            suit_counts[card.suit] += 1
            if card.rank in _HIGH_RANKS:
                high_count += 1

        for i, s in enumerate(SUITS):
            vec[i] = suit_counts[s] / 13.0

        vec[4] = high_count / 20.0
        vec[5] = len(remaining) / 52.0

        return vec

    def _encode_street_aggression(self, obs: AgentObservation) -> np.ndarray:
        """스트리트별 공격성 → (4,) [각 스트리트의 raise 비율]"""
        vec = np.zeros(4, dtype=np.float32)

        street_counts = {s: {'total': 0, 'raises': 0} for s in STREETS}
        for entry in obs.betting_history:
            street = entry.street if hasattr(entry, 'street') else entry.get('street', '')
            action = entry.action if hasattr(entry, 'action') else entry.get('action', '')
            if street in street_counts:
                street_counts[street]['total'] += 1
                if action == 'raise':
                    street_counts[street]['raises'] += 1

        for i, s in enumerate(STREETS):
            total = street_counts[s]['total']
            if total > 0:
                vec[i] = street_counts[s]['raises'] / total

        return vec
