"""
hand_eval.py — 7장 카드 핸드 평가기

7장(홀 카드 2 + 커뮤니티 5) 중 최선의 5장 조합을 찾아
비교 가능한 정수 점수를 반환합니다.

점수가 높을수록 강한 핸드입니다.
"""

from itertools import combinations
from typing import List, Tuple

from engine.card import Card, RANK_VALUE


# ── 핸드 카테고리 (높을수록 강함) ──────────────────────────
HAND_RANK = {
    'HIGH_CARD':       1,
    'ONE_PAIR':        2,
    'TWO_PAIR':        3,
    'THREE_OF_A_KIND': 4,
    'STRAIGHT':        5,
    'FLUSH':           6,
    'FULL_HOUSE':      7,
    'FOUR_OF_A_KIND':  8,
    'STRAIGHT_FLUSH':  9,
    'ROYAL_FLUSH':     10,
}

HAND_NAME = {v: k for k, v in HAND_RANK.items()}


def evaluate_hand(cards: List[Card]) -> Tuple[int, List[int], str]:
    """
    최대 7장의 카드에서 최선의 5장 조합을 평가합니다.

    Returns:
        (score, tiebreakers, hand_name)
        score       : 핸드 카테고리 점수 (1~10)
        tiebreakers : 동점 비교용 rank value 리스트 (높은 카드 순)
        hand_name   : 핸드 이름 문자열
    """
    assert 2 <= len(cards) <= 7, f"카드 수는 2~7장이어야 합니다. (받은 수: {len(cards)})"

    best_score = -1
    best_tb    = []
    best_name  = ''

    for combo in combinations(cards, min(5, len(cards))):
        score, tb, name = _eval5(list(combo))
        if (score, tb) > (best_score, best_tb):
            best_score = score
            best_tb    = tb
            best_name  = name

    return best_score, best_tb, best_name


def compare_hands(cards_a: List[Card], cards_b: List[Card]) -> int:
    """
    두 핸드를 비교합니다.
    Returns: 1 (a 승), -1 (b 승), 0 (무승부)
    """
    sa, ta, _ = evaluate_hand(cards_a)
    sb, tb, _ = evaluate_hand(cards_b)
    if (sa, ta) > (sb, tb):
        return 1
    if (sa, ta) < (sb, tb):
        return -1
    return 0


# ── 5장 평가 내부 함수 ─────────────────────────────────────

def _eval5(cards: List[Card]) -> Tuple[int, List[int], str]:
    assert len(cards) == 5
    values = sorted([c.value for c in cards], reverse=True)
    suits  = [c.suit for c in cards]
    ranks  = [c.rank for c in cards]

    is_flush    = len(set(suits)) == 1
    is_straight, straight_high = _check_straight(values)

    # 카운트
    from collections import Counter
    cnt = Counter(values)
    freq = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    groups = [f for _, f in freq]   # 빈도 내림차순
    group_vals = [v for v, _ in freq]

    # ── 카테고리 판별 ──────────────────────────────────────
    if is_straight and is_flush:
        if straight_high == 12:   # A high
            return HAND_RANK['ROYAL_FLUSH'],    [straight_high], 'ROYAL_FLUSH'
        return     HAND_RANK['STRAIGHT_FLUSH'], [straight_high], 'STRAIGHT_FLUSH'

    if groups[0] == 4:
        return HAND_RANK['FOUR_OF_A_KIND'], group_vals, 'FOUR_OF_A_KIND'

    if groups[0] == 3 and groups[1] == 2:
        return HAND_RANK['FULL_HOUSE'], group_vals, 'FULL_HOUSE'

    if is_flush:
        return HAND_RANK['FLUSH'], values, 'FLUSH'

    if is_straight:
        return HAND_RANK['STRAIGHT'], [straight_high], 'STRAIGHT'

    if groups[0] == 3:
        return HAND_RANK['THREE_OF_A_KIND'], group_vals, 'THREE_OF_A_KIND'

    if groups[0] == 2 and groups[1] == 2:
        return HAND_RANK['TWO_PAIR'], group_vals, 'TWO_PAIR'

    if groups[0] == 2:
        return HAND_RANK['ONE_PAIR'], group_vals, 'ONE_PAIR'

    return HAND_RANK['HIGH_CARD'], values, 'HIGH_CARD'


def _check_straight(values: List[int]) -> Tuple[bool, int]:
    """
    5장의 rank value 리스트에서 스트레이트 여부와 최고 카드를 반환.
    A-2-3-4-5 (wheel) 처리 포함.
    """
    unique = sorted(set(values), reverse=True)
    if len(unique) < 5:
        return False, -1

    # 일반 스트레이트
    if unique[0] - unique[4] == 4:
        return True, unique[0]

    # Wheel: A-2-3-4-5 (A=12, 2=0, 3=1, 4=2, 5=3)
    if unique == [12, 3, 2, 1, 0]:
        return True, 3   # 5-high straight

    return False, -1
