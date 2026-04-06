"""
reward_shaper.py — Advantage Function 기반 보상 설계

핵심 원칙:
  - 매몰 비용 제외: EV(fold) = 0 (이미 낸 칩은 무시)
  - 결정 품질 측정: A(s,a) = EV(선택한 액션) - EV(최선 액션)
  - 결과 놀람 측정: chip_delta - V(s)  (Critic 추정 대비 실제 결과)
  - 최종 보상: α × decision_quality + β × outcome_surprise

폴드가 최선일 때:
  EV(fold)=0, EV(call)=-10  → V(s)≈0  → A(fold)=0  (중립) ✓
  잘못된 콜 시: A(call)=-10-0=-10 (패널티) ✓
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from engine.state import AgentObservation, ACTION_FOLD, ACTION_CALL, ACTION_CHECK, ACTION_RAISE
from reward.ev_calculator import EVCalculator, ActionEV
from models.actor import RAISE_SIZES


# 보상 가중치
ALPHA = 0.7   # 결정 품질 비중
BETA  = 0.3   # 결과 놀람 비중

# 보상 정규화를 위한 빅블라인드 기준값 (스케일 조정)
BB_SCALE = 10.0


@dataclass
class StepReward:
    """단일 결정 스텝의 보상 분해"""
    decision_quality: float   # EV(선택) - EV(최선)
    outcome_surprise: float   # chip_delta - V(s)  (핸드 종료 후 채움)
    total:            float   # 최종 보상
    ev:               ActionEV
    action_taken:     str
    chosen_ev:        float


class RewardShaper:
    """
    각 의사결정 시점의 보상을 계산합니다.
    핸드 종료 후 outcome_surprise를 채워 최종 보상을 확정합니다.
    """

    def __init__(
        self,
        ev_simulations: int   = 300,
        alpha:          float = ALPHA,
        beta:           float = BETA,
        bb_scale:       float = BB_SCALE,
    ):
        self._ev_calc  = EVCalculator(simulations=ev_simulations)
        self.alpha     = alpha
        self.beta      = beta
        self.bb_scale  = bb_scale

        # 핸드 진행 중 결정 버퍼
        self._pending: list = []   # List[StepReward]

    # ══════════════════════════════════════════════════════
    #  핸드 진행 중 호출
    # ══════════════════════════════════════════════════════

    def on_action(
        self,
        obs:         AgentObservation,
        action:      str,
        amount:      int,
        baseline_v:  float,            # Critic의 V(s) 추정값
    ) -> StepReward:
        """
        액션 직후 호출.
        EV 기반 결정 품질을 즉시 계산하고, 버퍼에 저장합니다.
        outcome_surprise는 핸드 종료 후 채워집니다.
        """
        ev = self._ev_calc.calculate(obs)

        # 선택한 액션의 EV
        if action == ACTION_FOLD:
            chosen_ev = ev.fold_ev
        elif action in (ACTION_CALL, ACTION_CHECK):
            chosen_ev = ev.call_ev
        else:  # RAISE
            chosen_ev = ev.raise_ev

        # 결정 품질: 최선 EV와의 차이 (BB 단위로 정규화)
        best_ev = max(ev.fold_ev, ev.call_ev, ev.raise_ev)
        decision_quality = (chosen_ev - best_ev) / self.bb_scale

        step = StepReward(
            decision_quality = decision_quality,
            outcome_surprise = 0.0,   # 핸드 종료 후 채움
            total            = 0.0,   # 핸드 종료 후 확정
            ev               = ev,
            action_taken     = action,
            chosen_ev        = chosen_ev,
        )
        self._pending.append(step)
        return step

    # ══════════════════════════════════════════════════════
    #  핸드 종료 후 호출
    # ══════════════════════════════════════════════════════

    def on_hand_end(
        self,
        chip_delta:  int,              # 이번 핸드의 실제 칩 변화
        baseline_vs: list,             # 각 결정 시점의 V(s) (리스트)
    ) -> list:
        """
        핸드 종료 시 호출.
        결과 놀람을 채워 최종 보상을 확정하고 보상 리스트를 반환합니다.

        Returns:
            List[float] — 각 결정 스텝의 최종 보상
        """
        rewards = []
        n       = len(self._pending)

        # 핸드 전체 칩 변화를 각 결정에 균등 배분
        # (마지막 액션에 몰아주면 분산이 너무 큼)
        per_step_delta = (chip_delta / self.bb_scale) / max(n, 1)

        for i, step in enumerate(self._pending):
            v_s = baseline_vs[i] if i < len(baseline_vs) else 0.0
            step.outcome_surprise = per_step_delta - v_s
            step.total = (
                self.alpha * step.decision_quality +
                self.beta  * step.outcome_surprise
            )
            rewards.append(step.total)

        self._pending.clear()
        return rewards

    def reset(self) -> None:
        """핸드 시작 시 버퍼 초기화"""
        self._pending.clear()


# ══════════════════════════════════════════════════════════
#  액션 인덱스 → 실제 amount 변환
# ══════════════════════════════════════════════════════════

def action_idx_to_decision(
    action_idx: int,
    obs:        AgentObservation,
) -> dict:
    """
    Actor 네트워크의 출력 인덱스(0~6)를 실제 게임 액션으로 변환합니다.

    인덱스:
      0: fold
      1: check/call
      2: raise 0.5 pot
      3: raise 1.0 pot
      4: raise 1.5 pot
      5: raise 2.0 pot
      6: all-in
    """
    valid_names = {v['action'] for v in obs.valid_actions}

    if action_idx == 0:
        return {'action': ACTION_FOLD, 'amount': 0}

    if action_idx == 1:
        if ACTION_CHECK in valid_names:
            return {'action': ACTION_CHECK, 'amount': 0}
        return {'action': ACTION_CALL, 'amount': obs.call_amount}

    # 레이즈 시도
    if ACTION_RAISE in valid_names:
        ratio  = RAISE_SIZES[action_idx]   # 팟 대비 배율
        if action_idx == 6:
            amount = obs.max_raise         # all-in
        else:
            amount = max(obs.min_raise, int(obs.pot * ratio))
            amount = min(amount, obs.max_raise)
        return {'action': ACTION_RAISE, 'amount': amount}

    # 레이즈 불가 → 콜/체크로 대체
    if ACTION_CALL in valid_names:
        return {'action': ACTION_CALL, 'amount': obs.call_amount}
    if ACTION_CHECK in valid_names:
        return {'action': ACTION_CHECK, 'amount': 0}
    return {'action': ACTION_FOLD, 'amount': 0}


def build_valid_mask(obs: AgentObservation) -> np.ndarray:
    """
    현재 합법 액션에 대한 ACTION_DIM 크기의 마스크를 반환합니다.
    """
    from models.actor import ACTION_DIM
    mask        = np.zeros(ACTION_DIM, dtype=bool)
    valid_names = {v['action'] for v in obs.valid_actions}

    mask[0] = ACTION_FOLD  in valid_names
    mask[1] = (ACTION_CHECK in valid_names) or (ACTION_CALL in valid_names)

    if ACTION_RAISE in valid_names:
        mask[2:7] = True   # 레이즈 가능하면 모든 사이즈 허용

    # 최소한 하나는 True 보장
    if not mask.any():
        mask[0] = True

    return mask
