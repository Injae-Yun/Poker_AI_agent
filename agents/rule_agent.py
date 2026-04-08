"""
rule_agent.py — 룰 기반 에이전트 (Phase 1)

의사결정 우선순위:
  1. EV 계산 (팟 오즈 + 에퀴티) → 기본 행동 결정
  2. 상대 프로파일 → 레이즈 사이즈 / 블러프 빈도 조정
  3. 포지션 보정 → 포지션이 좋을수록 공격적

블러프 조건:
  - 폴드 에퀴티가 충분히 높고 (상대가 약해 보이고)
  - 포지션이 유리하고 (후포지션)
  - 스택이 충분한 경우
  - 매 핸드 최대 1번, 일정 확률로만 실행
"""

import random
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from engine.state import (
    AgentObservation,
    ACTION_FOLD, ACTION_CALL, ACTION_RAISE, ACTION_CHECK,
)
from reward.ev_calculator import EVCalculator
from utils.hand_evaluator import preflop_strength, equity_by_street
from utils.opponent_profiler import OpponentProfiler


# ── 전략 임계값 ────────────────────────────────────────────
PREFLOP_RAISE_THRESHOLD  = 0.65    # 이 이상 강도면 레이즈
PREFLOP_CALL_THRESHOLD   = 0.45    # 이 이상이면 콜 (미만이면 폴드)
EQUITY_FOLD_THRESHOLD    = 0.20    # 에퀴티 이 이하이면 팟 오즈 무관 폴드
BLUFF_BASE_PROB          = 0.12    # 블러프 기본 확률
POSITION_RAISE_BONUS     = 0.05    # 후포지션 공격성 보너스


class RuleAgent(BaseAgent):
    """팟 오즈 + 에퀴티 + 상대 모델 기반 룰 에이전트"""

    def __init__(
        self,
        player_id: int,
        name:      str = "RuleAgent",
        seed:      Optional[int] = None,
        verbose:   bool = False,
    ):
        super().__init__(player_id, name)
        self._ev_calc   = EVCalculator(simulations=100)
        self._profiler  = OpponentProfiler()
        self._rng       = random.Random(seed)
        self._verbose   = verbose

        # 라운드별 상태
        self._bluffed_this_hand = False
        self._num_players       = 4

    # ══════════════════════════════════════════════════════
    #  핵심 의사결정
    # ══════════════════════════════════════════════════════

    def declare_action(self, obs: AgentObservation) -> Dict[str, Any]:
        self._num_players = len(obs.players)

        # 상대 폴드 확률 맵 구성
        fold_prob_map = self._build_fold_prob_map(obs)

        # EV 계산
        ev = self._ev_calc.calculate(obs, fold_prob_map)

        # ── 의사결정 트리 ──────────────────────────────────
        action, amount = self._decide(obs, ev)

        # 유효성 검증 (잘못된 액션은 폴드로 대체)
        valid_names = {v['action'] for v in obs.valid_actions}
        if action not in valid_names:
            if ACTION_CHECK in valid_names:
                action, amount = ACTION_CHECK, 0
            else:
                action, amount = ACTION_FOLD, 0

        self._record_action(action)

        if self._verbose:
            self._log(obs, ev, action, amount)

        return {'action': action, 'amount': amount}

    def _decide(self, obs: AgentObservation, ev) -> tuple:
        """
        메인 의사결정 로직.
        Returns: (action, amount)
        """
        street    = obs.street
        equity    = ev.equity
        valid     = {v['action'] for v in obs.valid_actions}

        # ── 1. 에퀴티가 너무 낮으면 무조건 폴드 ──────────────
        if equity < EQUITY_FOLD_THRESHOLD and obs.call_amount > 0:
            return ACTION_FOLD, 0

        # ── 2. 포지션 보정값 ──────────────────────────────────
        pos_bonus = self._position_bonus(obs)

        # ── 3. 스트리트별 전략 ────────────────────────────────
        if street == 'preflop':
            return self._decide_preflop(obs, ev, pos_bonus, valid)
        else:
            return self._decide_postflop(obs, ev, pos_bonus, valid)

    # ── 프리플롭 전략 ──────────────────────────────────────
    def _decide_preflop(self, obs, ev, pos_bonus, valid):
        strength = preflop_strength(obs.hole_cards)
        effective_strength = strength + pos_bonus

        # 강한 핸드 → 레이즈
        if effective_strength >= PREFLOP_RAISE_THRESHOLD:
            if ACTION_RAISE in valid:
                amount = self._sizing_preflop(obs, strength)
                return ACTION_RAISE, amount
            elif ACTION_CALL in valid:
                return ACTION_CALL, obs.call_amount

        # 중간 핸드 → 콜 (팟 오즈 충족 여부 확인)
        if effective_strength >= PREFLOP_CALL_THRESHOLD:
            if ev.has_pot_odds or obs.call_amount == 0:
                if ACTION_CALL in valid:
                    return ACTION_CALL, obs.call_amount
                elif ACTION_CHECK in valid:
                    return ACTION_CHECK, 0

        # 약한 핸드 → 체크 가능하면 체크, 아니면 폴드
        if ACTION_CHECK in valid:
            return ACTION_CHECK, 0
        return ACTION_FOLD, 0

    # ── 포스트플롭 전략 ────────────────────────────────────
    def _decide_postflop(self, obs, ev, pos_bonus, valid):
        equity = ev.equity

        # 강한 핸드 → 밸류 레이즈
        if equity >= 0.65 + pos_bonus:
            if ACTION_RAISE in valid and ev.raise_ev > ev.call_ev:
                amount = self._sizing_postflop(obs, equity)
                return ACTION_RAISE, amount
            elif ACTION_CALL in valid:
                return ACTION_CALL, obs.call_amount
            elif ACTION_CHECK in valid:
                return ACTION_CHECK, 0

        # 중간 핸드 → EV 기반 결정
        if equity >= 0.40:
            if ev.has_pot_odds or obs.call_amount == 0:
                # 블러프 레이즈 시도 여부 확인
                if self._should_bluff(obs, ev, valid):
                    amount = self._sizing_bluff(obs)
                    self._bluffed_this_hand = True
                    return ACTION_RAISE, amount
                if ACTION_CALL in valid:
                    return ACTION_CALL, obs.call_amount
                elif ACTION_CHECK in valid:
                    return ACTION_CHECK, 0

        # 팟 오즈 충족하는 낮은 에퀴티 → 콜/체크
        if ev.has_pot_odds and obs.call_amount > 0:
            if ACTION_CALL in valid:
                return ACTION_CALL, obs.call_amount

        # 체크 가능하면 체크
        if ACTION_CHECK in valid:
            return ACTION_CHECK, 0

        return ACTION_FOLD, 0

    # ══════════════════════════════════════════════════════
    #  베팅 사이즈 결정
    # ══════════════════════════════════════════════════════

    def _sizing_preflop(self, obs: AgentObservation, strength: float) -> int:
        """
        프리플롭 레이즈 사이즈.
        표준: 3BB + 콜러 수 × 1BB
        강한 핸드(AA, KK)는 4BB
        """
        bb       = obs.min_raise  # min_raise ≈ 1BB 단위
        callers  = sum(
            1 for a in obs.betting_history
            if a.street == 'preflop' and a.action == 'call'
        )
        base_size = 4 if strength >= 0.80 else 3
        amount    = base_size * (obs.min_raise) + callers * obs.min_raise
        return min(amount, obs.max_raise)

    def _sizing_postflop(self, obs: AgentObservation, equity: float) -> int:
        """
        포스트플롭 밸류 베팅 사이즈.
        강할수록 팟 대비 비율 높임 (0.5팟 ~ 팟 사이즈).
        """
        pot = obs.pot
        if equity >= 0.80:
            ratio = 0.9    # 팟 사이즈 벳
        elif equity >= 0.65:
            ratio = 0.65   # 2/3 팟
        else:
            ratio = 0.5    # 하프 팟

        amount = int(pot * ratio)
        return min(max(amount, obs.min_raise), obs.max_raise)

    def _sizing_bluff(self, obs: AgentObservation) -> int:
        """블러프 사이즈: 팟의 0.6~0.75 (폴드 에퀴티 극대화)"""
        pot    = obs.pot
        amount = int(pot * self._rng.uniform(0.6, 0.75))
        return min(max(amount, obs.min_raise), obs.max_raise)

    # ══════════════════════════════════════════════════════
    #  블러프 조건
    # ══════════════════════════════════════════════════════

    def _should_bluff(self, obs: AgentObservation, ev, valid: set) -> bool:
        """
        블러프 시도 여부를 결정합니다.
        조건:
          1. 아직 이 핸드에서 블러프 안 했음
          2. ACTION_RAISE가 가능
          3. 후포지션 (포지션 인덱스가 큰 쪽 = 늦은 포지션)
          4. 상대들이 전반적으로 약해 보임 (타이트-패시브)
          5. 스택이 팟의 3배 이상 (리스크 관리)
          6. 일정 확률 (BLUFF_BASE_PROB + 포지션 보정)
        """
        if self._bluffed_this_hand:
            return False
        if ACTION_RAISE not in valid:
            return False

        # 후포지션 여부 (포지션 인덱스 높을수록 후포지션)
        num_players  = obs.active_players
        late_pos     = obs.position >= num_players // 2

        # 상대 강도 추정
        opp_strength = self._avg_opponent_strength(obs)
        weak_opponents = opp_strength < 0.5

        # 스택 충분 여부
        sufficient_stack = obs.stack >= obs.pot * 3

        if not (late_pos and weak_opponents and sufficient_stack):
            return False

        # 확률적 블러프
        bluff_prob = BLUFF_BASE_PROB + (0.05 if late_pos else 0.0)
        return self._rng.random() < bluff_prob

    # ══════════════════════════════════════════════════════
    #  포지션 및 상대 모델 유틸리티
    # ══════════════════════════════════════════════════════

    def _position_bonus(self, obs: AgentObservation) -> float:
        """
        후포지션일수록 양의 보너스 (공격성 증가).
        딜러 포지션(=0)이 가장 유리 → 가장 큰 보너스.
        """
        n    = obs.active_players
        pos  = obs.position
        # 포지션 0(딜러)이 가장 유리, BB(2)가 가장 불리
        # 후포지션: position이 크면 클수록 유리
        rel  = pos / max(n - 1, 1)   # 0=SB(불리) ~ 1=딜러(유리)
        return rel * POSITION_RAISE_BONUS

    def _build_fold_prob_map(self, obs: AgentObservation) -> Dict[int, float]:
        """
        각 레이즈 금액에 대한 평균 폴드 확률 맵을 구성합니다.
        """
        active_opps = [
            p for p in obs.players
            if p.player_id != self.player_id and p.is_active
        ]
        if not active_opps:
            return {}

        fold_map = {}
        for extra in EVCalculator._raise_candidates(obs.min_raise, obs.max_raise):
            total_extra = obs.call_amount + extra
            ratio       = total_extra / obs.pot if obs.pot > 0 else 1.0
            avg_fold    = sum(
                self._profiler.estimate_fold_probability(
                    p.player_id, ratio, obs.street
                )
                for p in active_opps
            ) / len(active_opps)
            fold_map[extra] = avg_fold

        return fold_map

    def _avg_opponent_strength(self, obs: AgentObservation) -> float:
        """현재 활성 상대들의 평균 추정 핸드 강도"""
        opps = [
            p for p in obs.players
            if p.player_id != self.player_id and p.is_active
        ]
        if not opps:
            return 0.5
        return sum(
            self._profiler.get_profile(p.player_id).estimated_hand_strength
            for p in opps
        ) / len(opps)

    # ══════════════════════════════════════════════════════
    #  콜백 오버라이드
    # ══════════════════════════════════════════════════════

    def on_round_start(self, round_num: int) -> None:
        super().on_round_start(round_num)
        self._bluffed_this_hand = False

    def on_round_end(self, result: dict) -> None:
        super().on_round_end(result)
        # 상대 프로파일 업데이트
        # betting_history는 BettingAction 객체 또는 dict 둘 다 허용
        for entry in result.get('betting_history', []):
            if isinstance(entry, dict):
                pid    = entry['player_id']
                act    = entry['action']
                street = entry['street']
                amt    = entry['amount']
            else:
                pid    = entry.player_id
                act    = entry.action
                street = entry.street
                amt    = entry.amount

            if pid != self.player_id:
                self._profiler.on_action(
                    player_id = pid,
                    action    = act,
                    street    = street,
                    amount    = amt,
                    pot       = 0,
                    is_blind  = (act == 'blind'),
                )
        # 쇼다운 결과 업데이트
        if result.get('showdown'):
            for pid, _ in result.get('player_hands', {}).items():
                if pid != self.player_id:
                    won = pid in result.get('winners', [])
                    self._profiler.on_showdown(pid, won)

    # ── 디버그 로그 ────────────────────────────────────────
    def _log(self, obs, ev, action, amount):
        print(
            f"[{self.name}] {obs.street:<8} "
            f"equity={ev.equity:.2f}  pot_odds={ev.pot_odds:.2f}  "
            f"EV(f/c/r)={ev.fold_ev:.1f}/{ev.call_ev:.1f}/{ev.raise_ev:.1f}  "
            f"→ {action.upper()} {amount if action==ACTION_RAISE else ''}"
        )
