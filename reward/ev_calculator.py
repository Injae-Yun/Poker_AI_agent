"""
ev_calculator.py — EV(기댓값) 계산기

매몰 비용을 제외한 순수 미래 기댓값을 계산합니다.

  EV(fold)  = 0                          (매몰 비용 무시)
  EV(call)  = equity × pot_after - (1-equity) × call_amount
  EV(raise) = fold_equity × pot + (1-fold_equity) × (equity × new_pot - (1-equity) × raise_total)

핵심 원칙:
  - 이미 낸 칩(팟에 들어간 금액)은 매몰 비용 → EV 계산에서 제외
  - 팟 오즈(Pot Odds) = call / (pot + call) → 에퀴티와 비교
  - EV > 0 이면 수익 기대 액션
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from engine.card import Card
from engine.state import AgentObservation
from utils.hand_evaluator import equity_by_street


@dataclass
class ActionEV:
    """각 액션의 기댓값 및 관련 정보"""
    fold_ev:      float
    call_ev:      float
    raise_ev:     float          # 최적 레이즈 사이즈 기준
    best_raise:   int            # 권장 레이즈 금액 (추가분)
    equity:       float
    pot_odds:     float          # call / (pot + call)
    has_pot_odds: bool           # equity > pot_odds
    best_action:  str            # 기댓값 최대 액션


class EVCalculator:
    """
    관찰 정보(AgentObservation)를 바탕으로 각 액션의 EV를 계산합니다.
    """

    def __init__(self, simulations: int = 400):
        self.simulations = simulations

    # ── 메인 계산 ──────────────────────────────────────────
    def calculate(
        self,
        obs:            AgentObservation,
        fold_probability_by_raise: Optional[Dict[int, float]] = None,
    ) -> ActionEV:
        """
        현재 관찰을 기반으로 모든 액션의 EV를 계산합니다.

        Args:
            obs: 에이전트 관찰 정보
            fold_probability_by_raise: {raise_amount: fold_prob} 상대 폴드 확률
                None이면 기본값(팟 크기 기반 추정)을 사용합니다.
        """
        num_opponents = obs.active_players - 1

        # ── 에퀴티 계산 ────────────────────────────────────
        equity = equity_by_street(
            hole_cards      = obs.hole_cards,
            community_cards = obs.community_cards,
            num_opponents   = max(1, num_opponents),
            simulations     = self.simulations,
        )

        # ── 팟 오즈 ────────────────────────────────────────
        total_pot_if_call = obs.pot + obs.call_amount
        pot_odds = (
            obs.call_amount / total_pot_if_call
            if total_pot_if_call > 0 else 0.0
        )

        # ── EV(fold) ───────────────────────────────────────
        # 매몰 비용 제외 → 항상 0
        fold_ev = 0.0

        # ── EV(call / check) ───────────────────────────────
        if obs.call_amount == 0:
            # 체크: 공짜로 다음 카드 봄
            call_ev = equity * obs.pot
        else:
            call_ev = self._ev_call(equity, obs.pot, obs.call_amount)

        # ── EV(raise) — 최적 레이즈 크기 탐색 ───────────────
        raise_ev   = fold_ev    # 레이즈 불가면 폴드와 동일
        best_raise = 0

        if obs.min_raise <= obs.max_raise and obs.stack > obs.call_amount:
            raise_ev, best_raise = self._best_raise_ev(
                equity                     = equity,
                pot                        = obs.pot,
                call_amount                = obs.call_amount,
                min_raise                  = obs.min_raise,
                max_raise                  = obs.max_raise,
                num_opponents              = num_opponents,
                fold_probability_by_raise  = fold_probability_by_raise,
            )

        # ── 최선 액션 결정 ─────────────────────────────────
        evs = {'fold': fold_ev, 'call': call_ev, 'raise': raise_ev}
        best_action = max(evs, key=lambda a: evs[a])

        # call/check가 거의 같으면 콜 선택
        if abs(call_ev - raise_ev) < 1.0 and call_ev >= fold_ev:
            best_action = 'call'

        return ActionEV(
            fold_ev      = fold_ev,
            call_ev      = call_ev,
            raise_ev     = raise_ev,
            best_raise   = best_raise,
            equity       = equity,
            pot_odds     = pot_odds,
            has_pot_odds = equity > pot_odds,
            best_action  = best_action,
        )

    # ── 내부 계산 메서드 ───────────────────────────────────

    def _ev_call(self, equity: float, pot: int, call_amount: int) -> float:
        """
        EV(call) = equity × (pot + call) - (1 - equity) × call

        pot      : 현재 팟 (내가 낼 콜 금액 포함 전)
        call_amount: 내가 추가로 내야 하는 금액
        """
        pot_after = pot + call_amount
        return equity * pot_after - (1 - equity) * call_amount

    def _estimate_fold_prob(
        self,
        raise_total: int,   # 내가 내는 총 추가 금액
        pot:         int,
        num_opponents: int,
        fold_probability_by_raise: Optional[Dict[int, float]],
    ) -> float:
        """
        상대가 레이즈에 폴드할 확률을 추정합니다.
        fold_probability_by_raise가 없으면 팟 대비 레이즈 크기로 추정.
        """
        if fold_probability_by_raise and raise_total in fold_probability_by_raise:
            return fold_probability_by_raise[raise_total]

        if pot == 0:
            return 0.3

        # 팟 대비 레이즈 비율이 클수록 폴드 유도 높음
        ratio = raise_total / pot
        base_fold = min(0.7, 0.2 + ratio * 0.25)

        # 상대가 많을수록 누군가는 콜할 가능성 증가
        multi_opponent_penalty = 0.1 * (num_opponents - 1)
        return max(0.05, base_fold - multi_opponent_penalty)

    def _ev_raise(
        self,
        equity:       float,
        pot:          int,
        call_amount:  int,
        raise_extra:  int,    # call 초과 추가 금액
        fold_prob:    float,
    ) -> float:
        """
        EV(raise) = fold_eq × pot
                  + (1 - fold_eq) × [equity × new_pot - (1-equity) × total_put_in]

        raise_extra : call_amount 위에 추가로 올리는 금액
        total_put_in: 내가 이번에 내는 전체 금액 (call + raise_extra)
        """
        total_put_in = call_amount + raise_extra
        new_pot      = pot + total_put_in

        # 상대가 콜했을 때 내 기댓값
        ev_if_called = equity * new_pot - (1 - equity) * total_put_in
        # 상대가 폴드했을 때 내 기댓값 (팟 획득)
        ev_if_folded = pot   # 현재 팟 획득

        return fold_prob * ev_if_folded + (1 - fold_prob) * ev_if_called

    def _best_raise_ev(
        self,
        equity:       float,
        pot:          int,
        call_amount:  int,
        min_raise:    int,
        max_raise:    int,
        num_opponents: int,
        fold_probability_by_raise: Optional[Dict[int, float]],
    ):
        """
        가능한 레이즈 크기 중 EV가 가장 높은 (ev, raise_amount) 반환.
        min ~ max 사이를 5개 포인트로 샘플링합니다.
        """
        # 탐색할 레이즈 금액 후보 (추가분)
        candidates = self._raise_candidates(min_raise, max_raise)

        best_ev     = -float('inf')
        best_amount = min_raise

        for extra in candidates:
            total_extra = call_amount + extra
            fold_prob   = self._estimate_fold_prob(
                total_extra, pot, num_opponents, fold_probability_by_raise
            )
            ev = self._ev_raise(equity, pot, call_amount, extra, fold_prob)
            if ev > best_ev:
                best_ev     = ev
                best_amount = extra

        return best_ev, best_amount

    @staticmethod
    def _raise_candidates(min_r: int, max_r: int) -> List[int]:
        """min ~ max 사이에서 의미 있는 레이즈 후보를 반환합니다."""
        if min_r >= max_r:
            return [min_r]

        span = max_r - min_r
        # 항상 min, 0.33팟, 0.5팟, 0.75팟, 팟, max 포인트 포함
        ratios    = [0.0, 0.33, 0.5, 0.75, 1.0]
        candidates = sorted(set(
            [min_r, max_r] + [min_r + int(r * span) for r in ratios]
        ))
        return [c for c in candidates if min_r <= c <= max_r]

    # ── 편의 메서드 ────────────────────────────────────────
    def pot_odds(self, obs: AgentObservation) -> float:
        """팟 오즈만 빠르게 계산"""
        total = obs.pot + obs.call_amount
        return obs.call_amount / total if total > 0 else 0.0

    def should_call(self, obs: AgentObservation, equity: float) -> bool:
        """팟 오즈 기준 콜 여부"""
        return equity > self.pot_odds(obs)
