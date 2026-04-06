"""
opponent_profiler.py — 상대 플레이어 통계 프로파일러

각 상대의 배팅 패턴을 누적하여 다음 통계를 추적합니다.

  VPIP  (Voluntarily Put $ In Pot): 자발적 베팅 참여율
  PFR   (Pre-Flop Raise)          : 프리플롭 레이즈 빈도
  AF    (Aggression Factor)       : 공격성 지수 = (레이즈+벳) / 콜
  WTSD  (Went To ShowDown)        : 쇼다운 참여율
  WSD   (Won at ShowDown)         : 쇼다운 승률

스타일 분류:
  타이트(Tight)  : VPIP < 25%
  루즈(Loose)    : VPIP >= 25%
  패시브(Passive): AF < 1.0
  어그레시브(AG) : AF >= 1.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PlayerProfile:
    player_id: int

    # ── 누적 카운터 ───────────────────────────────────────
    hands_seen:       int = 0

    # VPIP 관련
    vpip_opportunities: int = 0   # 자발적 참여 기회 (BB 제외)
    vpip_count:         int = 0   # 실제 참여 횟수

    # PFR 관련
    pfr_opportunities:  int = 0
    pfr_count:          int = 0

    # AF 관련 (공격/수비 횟수)
    aggressive_actions: int = 0   # 레이즈 + 벳
    passive_actions:    int = 0   # 콜

    # 쇼다운 관련
    showdowns:          int = 0
    showdown_wins:      int = 0

    # 배팅 사이즈 통계
    avg_raise_size:     float = 0.0
    raise_size_samples: int   = 0

    # ── 파생 통계 (프로퍼티) ──────────────────────────────
    @property
    def vpip(self) -> float:
        if self.vpip_opportunities == 0:
            return 0.25  # 사전확률
        return self.vpip_count / self.vpip_opportunities

    @property
    def pfr(self) -> float:
        if self.pfr_opportunities == 0:
            return 0.15
        return self.pfr_count / self.pfr_opportunities

    @property
    def af(self) -> float:
        """Aggression Factor: (레이즈+벳) / 콜. 높을수록 공격적."""
        if self.passive_actions == 0:
            return float(self.aggressive_actions) if self.aggressive_actions > 0 else 1.0
        return self.aggressive_actions / self.passive_actions

    @property
    def wtsd(self) -> float:
        if self.hands_seen == 0:
            return 0.25
        return self.showdowns / self.hands_seen

    @property
    def wsd(self) -> float:
        if self.showdowns == 0:
            return 0.5
        return self.showdown_wins / self.showdowns

    # ── 스타일 분류 ───────────────────────────────────────
    @property
    def is_tight(self) -> bool:
        return self.vpip < 0.25

    @property
    def is_aggressive(self) -> bool:
        return self.af >= 1.0

    @property
    def style(self) -> str:
        t = "Tight" if self.is_tight else "Loose"
        a = "Aggressive" if self.is_aggressive else "Passive"
        return f"{t}-{a}"

    # ── 핸드 레인지 추정 ──────────────────────────────────
    @property
    def estimated_hand_strength(self) -> float:
        """
        상대가 현재 들고 있을 핸드의 추정 강도 (0~1).
        VPIP와 PFR을 바탕으로 베이지안 추정.
        """
        # 타이트할수록, PFR이 높을수록 강한 핸드 보유 경향
        base = 0.5
        tight_bonus    = (0.25 - self.vpip) * 0.4 if self.is_tight else 0.0
        aggress_bonus  = min(0.1, (self.af - 1.0) * 0.05) if self.is_aggressive else 0.0
        return min(0.9, max(0.1, base + tight_bonus + aggress_bonus))

    def update_raise_size(self, size: float) -> None:
        n = self.raise_size_samples
        self.avg_raise_size = (self.avg_raise_size * n + size) / (n + 1)
        self.raise_size_samples += 1

    def to_dict(self) -> dict:
        return {
            'player_id':      self.player_id,
            'style':          self.style,
            'vpip':           round(self.vpip, 3),
            'pfr':            round(self.pfr, 3),
            'af':             round(self.af, 3),
            'wtsd':           round(self.wtsd, 3),
            'wsd':            round(self.wsd, 3),
            'hands_seen':     self.hands_seen,
            'est_strength':   round(self.estimated_hand_strength, 3),
        }


class OpponentProfiler:
    """
    여러 상대의 프로파일을 관리합니다.
    에이전트 인스턴스마다 독립적으로 생성됩니다.
    """

    def __init__(self):
        self._profiles: Dict[int, PlayerProfile] = {}

    def get_profile(self, player_id: int) -> PlayerProfile:
        if player_id not in self._profiles:
            self._profiles[player_id] = PlayerProfile(player_id)
        return self._profiles[player_id]

    # ── 이벤트 업데이트 ───────────────────────────────────
    def on_round_start(self, all_player_ids: List[int]) -> None:
        for pid in all_player_ids:
            p = self.get_profile(pid)
            p.hands_seen += 1

    def on_action(
        self,
        player_id:  int,
        action:     str,       # 'fold' | 'call' | 'raise' | 'check'
        street:     str,       # 'preflop' | 'flop' | 'turn' | 'river'
        amount:     int,
        pot:        int,
        is_blind:   bool = False,
    ) -> None:
        """
        상대 액션 발생 시 호출.
        블라인드(is_blind=True)는 자발적 참여로 카운트하지 않음.
        """
        p = self.get_profile(player_id)

        if is_blind:
            return

        # VPIP: 자발적으로 팟에 돈을 넣는 모든 행위
        if action in ('call', 'raise'):
            p.vpip_opportunities += 1
            p.vpip_count += 1
        elif action == 'fold':
            p.vpip_opportunities += 1   # 기회는 있었음

        # PFR
        if street == 'preflop' and action == 'raise':
            p.pfr_opportunities += 1
            p.pfr_count += 1
        elif street == 'preflop' and action in ('call', 'fold'):
            p.pfr_opportunities += 1

        # AF
        if action == 'raise':
            p.aggressive_actions += 1
            if pot > 0 and amount > 0:
                p.update_raise_size(amount / pot)
        elif action == 'call':
            p.passive_actions += 1

    def on_showdown(self, player_id: int, won: bool) -> None:
        p = self.get_profile(player_id)
        p.showdowns += 1
        if won:
            p.showdown_wins += 1

    # ── 전략적 조회 ───────────────────────────────────────
    def estimate_fold_probability(
        self,
        player_id:   int,
        raise_size:  float,   # 레이즈 금액 / 팟 크기
        street:      str,
    ) -> float:
        """
        특정 상대가 해당 레이즈 사이즈에 폴드할 확률 추정.
        타이트-패시브 상대일수록, 레이즈가 클수록 폴드 확률 높음.
        """
        p = self.get_profile(player_id)

        # 기본 폴드 확률: VPIP의 역수
        base_fold  = 1.0 - p.vpip

        # 레이즈 크기 조정: 팟 대비 클수록 폴드 유도 증가
        size_bonus = min(0.3, raise_size * 0.2)

        # 공격성 조정: 어그레시브 플레이어는 폴드 덜 함
        ag_penalty = 0.1 if p.is_aggressive else 0.0

        return min(0.85, max(0.05, base_fold + size_bonus - ag_penalty))

    def all_profiles(self) -> List[dict]:
        return [p.to_dict() for p in self._profiles.values()]
