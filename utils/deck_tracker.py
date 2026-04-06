"""
deck_tracker.py — 잔여 덱 및 아웃츠 추적기

공개된 카드를 제거한 잔여 덱을 관리하고,
아웃츠(Outs) 계산 및 Rule of 4/2 근사값을 제공합니다.
"""

from typing import List, Set
from engine.card import Card, SUITS, RANKS


class DeckTracker:
    """
    공개된 카드를 제거한 잔여 덱을 추적합니다.

    각 라운드 시작 시 reset()을 호출하고,
    카드가 공개될 때마다 mark_seen()을 호출합니다.
    """

    def __init__(self):
        self._seen: Set[str] = set()   # 'Ah', 'Ks' 형태
        self._all_cards: List[Card] = [
            Card(r, s) for s in SUITS for r in RANKS
        ]

    # ── 초기화 / 업데이트 ──────────────────────────────────
    def reset(self) -> None:
        """새 라운드 시작 시 호출 — 잔여 덱을 52장으로 리셋"""
        self._seen.clear()

    def mark_seen(self, cards: List[Card]) -> None:
        """공개된 카드를 잔여 덱에서 제거"""
        for c in cards:
            self._seen.add(repr(c))

    # ── 잔여 덱 조회 ───────────────────────────────────────
    @property
    def remaining(self) -> List[Card]:
        """현재 잔여 덱 카드 목록"""
        return [c for c in self._all_cards if repr(c) not in self._seen]

    @property
    def remaining_count(self) -> int:
        return len(self.remaining)

    # ── 아웃츠 / 확률 계산 ────────────────────────────────
    def count_outs(
        self,
        hole_cards:      List[Card],
        community_cards: List[Card],
        target_rank:     int,          # 목표 핸드 카테고리 (1~10)
    ) -> int:
        """
        잔여 덱에서 뽑았을 때 target_rank 이상의 핸드를 만드는
        카드(아웃츠)의 수를 반환합니다.
        """
        from engine.hand_eval import evaluate_hand

        current_score, _, _ = evaluate_hand(hole_cards + community_cards)
        outs = 0

        for card in self.remaining:
            if repr(card) in {repr(c) for c in hole_cards + community_cards}:
                continue
            new_score, _, _ = evaluate_hand(hole_cards + community_cards + [card])
            if new_score >= target_rank and new_score > current_score:
                outs += 1

        return outs

    def rule_of_x(self, outs: int, cards_to_come: int) -> float:
        """
        Rule of 4/2: 아웃츠 기반 간이 승률 추정
        cards_to_come=2 → outs × 4%  (턴+리버)
        cards_to_come=1 → outs × 2%  (리버만)
        """
        multiplier = 4 if cards_to_come >= 2 else 2
        return min(1.0, outs * multiplier / 100)

    def equity_by_rule_of_x(
        self,
        hole_cards:      List[Card],
        community_cards: List[Card],
        target_rank:     int,
        cards_to_come:   int,
    ) -> float:
        """아웃츠 계산 + Rule of X 를 결합한 간이 에퀴티"""
        outs = self.count_outs(hole_cards, community_cards, target_rank)
        return self.rule_of_x(outs, cards_to_come)
