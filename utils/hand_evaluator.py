"""
hand_evaluator.py — 몬테카를로 에퀴티 계산기

Monte Carlo 시뮬레이션으로 현재 핸드의 승률(에퀴티)을 추정합니다.
프리플롭 핸드 강도 룩업 테이블도 포함합니다.
"""

import random
from typing import List, Dict, Tuple, Optional

from engine.card import Card, SUITS, RANKS
from engine.hand_eval import evaluate_hand


# ── 프리플롭 핸드 그룹 (169가지 → 그룹 분류) ────────────────
# 슈티드(s): 같은 무늬, 오프수트(o): 다른 무늬
# 값 범위: 0.0 ~ 1.0 (높을수록 강한 핸드)

_PREFLOP_STRENGTH: Dict[str, float] = {
    # Pocket Pairs (내림차순)
    'AA': 0.85, 'KK': 0.82, 'QQ': 0.80, 'JJ': 0.77, 'TT': 0.75,
    '99': 0.72, '88': 0.69, '77': 0.66, '66': 0.63, '55': 0.60,
    '44': 0.57, '33': 0.54, '22': 0.50,
    # Suited connectors & broadway (suited)
    'AKs': 0.80, 'AQs': 0.77, 'AJs': 0.75, 'ATs': 0.73,
    'KQs': 0.72, 'KJs': 0.70, 'KTs': 0.68,
    'QJs': 0.68, 'QTs': 0.66,
    'JTs': 0.65, 'T9s': 0.62, '98s': 0.60, '87s': 0.58,
    '76s': 0.56, '65s': 0.54, '54s': 0.52,
    'A9s': 0.68, 'A8s': 0.66, 'A7s': 0.64, 'A6s': 0.62,
    'A5s': 0.63, 'A4s': 0.61, 'A3s': 0.59, 'A2s': 0.57,
    'K9s': 0.62, 'Q9s': 0.58, 'J9s': 0.58,
    # Offsuit
    'AKo': 0.75, 'AQo': 0.72, 'AJo': 0.70, 'ATo': 0.68,
    'KQo': 0.67, 'KJo': 0.65, 'KTo': 0.63,
    'QJo': 0.63, 'QTo': 0.61,
    'JTo': 0.60,
    'A9o': 0.62, 'A8o': 0.60, 'A7o': 0.58,
    'K9o': 0.56, 'K8o': 0.54,
}
_PREFLOP_DEFAULT = 0.38   # 테이블에 없는 핸드의 기본값


def preflop_strength(hole_cards: List[Card]) -> float:
    """
    2장 홀 카드의 프리플롭 상대적 강도를 반환합니다. (0.0~1.0)
    포켓 페어, 수티드, 오프수트를 구분합니다.
    """
    assert len(hole_cards) == 2
    c1, c2 = sorted(hole_cards, reverse=True)   # 높은 카드가 먼저

    r1, r2 = c1.rank, c2.rank
    suited  = c1.suit == c2.suit

    if r1 == r2:
        key = r1 + r2                            # 'AA', 'KK' …
    elif suited:
        key = r1 + r2 + 's'                      # 'AKs' …
    else:
        key = r1 + r2 + 'o'                      # 'AKo' …

    return _PREFLOP_STRENGTH.get(key, _PREFLOP_DEFAULT)


# ══════════════════════════════════════════════════════════
#  Monte Carlo 에퀴티 계산
# ══════════════════════════════════════════════════════════

def monte_carlo_equity(
    hole_cards:      List[Card],
    community_cards: List[Card],
    num_opponents:   int,
    simulations:     int = 500,
    seed:            Optional[int] = None,
) -> float:
    """
    Monte Carlo 시뮬레이션으로 승률(에퀴티)을 추정합니다.

    Args:
        hole_cards      : 본인의 홀 카드 2장
        community_cards : 현재 공개된 커뮤니티 카드 (0~5장)
        num_opponents   : 활성 상대 플레이어 수
        simulations     : 시뮬레이션 횟수 (정확도 vs 속도)
        seed            : 재현 가능성

    Returns:
        float: 0.0~1.0 사이의 승률 (무승부는 0.5로 처리)
    """
    if num_opponents <= 0:
        return 1.0

    rng         = random.Random(seed)
    seen        = set(repr(c) for c in hole_cards + community_cards)
    all_cards   = [
        Card(r, s) for s in SUITS for r in RANKS
        if Card(r, s).__repr__() not in seen
    ]

    streets_left = 5 - len(community_cards)
    wins         = 0
    ties         = 0

    for _ in range(simulations):
        rng.shuffle(all_cards)
        idx = 0

        # 상대 홀 카드 딜
        opp_hands = []
        for _ in range(num_opponents):
            opp_hands.append(all_cards[idx:idx+2])
            idx += 2

        # 남은 커뮤니티 카드 채우기
        runout = all_cards[idx: idx + streets_left]
        board  = community_cards + runout

        # 본인 핸드 평가
        my_score, my_tb, _ = evaluate_hand(hole_cards + board)

        # 상대 핸드 평가 및 비교
        best_opp = (-1, [])
        for opp in opp_hands:
            s, tb, _ = evaluate_hand(opp + board)
            if (s, tb) > best_opp:
                best_opp = (s, tb)

        if (my_score, my_tb) > best_opp:
            wins += 1
        elif (my_score, my_tb) == best_opp:
            ties += 1

    return (wins + ties * 0.5) / simulations


def equity_by_street(
    hole_cards:      List[Card],
    community_cards: List[Card],
    num_opponents:   int,
    simulations:     int = 300,
) -> float:
    """
    현재 스트리트에 맞는 시뮬레이션 횟수를 자동 조정하여
    에퀴티를 반환합니다.
    프리플롭은 룩업 테이블로 빠르게 처리합니다.
    """
    n_community = len(community_cards)

    if n_community == 0:
        # 프리플롭: 룩업 테이블 사용 (빠름)
        base = preflop_strength(hole_cards)
        # 상대 수에 따라 보정 (상대가 많을수록 승률 감소)
        penalty = 0.05 * (num_opponents - 1)
        return max(0.05, base - penalty)

    # 플롭 이후: Monte Carlo
    # 커뮤니티가 많을수록 불확실성이 낮아 시뮬 횟수 줄임
    sims = {3: simulations, 4: simulations // 2, 5: simulations // 4}
    n_sims = sims.get(n_community, simulations)

    return monte_carlo_equity(
        hole_cards, community_cards, num_opponents, n_sims
    )
