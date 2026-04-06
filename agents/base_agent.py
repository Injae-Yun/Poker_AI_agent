"""
base_agent.py — 에이전트 추상 기반 클래스

모든 에이전트는 BaseAgent를 상속하고
declare_action() 메서드를 구현해야 합니다.

정보 은닉 원칙:
  - 에이전트는 AgentObservation 에 담긴 정보만 사용 가능
  - 다른 에이전트 인스턴스에 직접 접근 금지
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from engine.state import AgentObservation, ACTION_FOLD, ACTION_CALL, ACTION_RAISE, ACTION_CHECK


class BaseAgent(ABC):
    """포커 에이전트 추상 기반 클래스"""

    def __init__(self, player_id: int, name: str = "Agent"):
        self.player_id = player_id
        self.name      = name

        # 에이전트가 자체적으로 유지하는 통계 (게임 간 지속)
        self._hands_played = 0
        self._total_reward = 0.0
        self._action_counts = {ACTION_FOLD: 0, ACTION_CALL: 0,
                               ACTION_RAISE: 0, ACTION_CHECK: 0}

    # ── 필수 구현 메서드 ───────────────────────────────────
    @abstractmethod
    def declare_action(self, obs: AgentObservation) -> Dict[str, Any]:
        """
        현재 관찰(obs)을 바탕으로 액션을 결정합니다.

        Returns:
            {'action': str, 'amount': int}
            action : 'fold' | 'call' | 'raise' | 'check'
            amount : raise인 경우 추가로 낼 금액 (call_amount 초과분)
                     fold/call/check는 0 또는 자동 계산

        주의:
            반드시 obs.valid_actions에 포함된 액션만 반환해야 합니다.
        """

    # ── 선택적 콜백 메서드 ─────────────────────────────────
    def on_game_start(self, config) -> None:
        """게임 시작 시 호출 (설정 정보 수신)"""
        pass

    def on_round_start(self, round_num: int) -> None:
        """각 핸드 시작 시 호출"""
        self._hands_played += 1

    def on_round_end(self, result: Dict[str, Any]) -> None:
        """
        각 핸드 종료 시 호출.
        result: {
            'winners': List[int],          # 승자 player_id 목록
            'pot': int,                    # 총 팟 크기
            'chip_delta': int,             # 본인의 칩 변화량
            'showdown': bool,              # 쇼다운 여부
            'community_cards': List[Card],
            'player_hands': Dict[int, List[Card]],  # 쇼다운 시 공개 핸드
        }
        """
        self._total_reward += result.get('chip_delta', 0)

    def on_game_end(self) -> None:
        """게임 전체 종료 시 호출"""
        pass

    # ── 유틸리티 ──────────────────────────────────────────
    def _record_action(self, action: str) -> None:
        self._action_counts[action] = self._action_counts.get(action, 0) + 1

    @property
    def stats(self) -> Dict[str, Any]:
        total_actions = sum(self._action_counts.values()) or 1
        return {
            'player_id':    self.player_id,
            'name':         self.name,
            'hands_played': self._hands_played,
            'total_reward': self._total_reward,
            'avg_reward':   self._total_reward / max(self._hands_played, 1),
            'fold_rate':    self._action_counts[ACTION_FOLD]  / total_actions,
            'call_rate':    self._action_counts[ACTION_CALL]  / total_actions,
            'raise_rate':   self._action_counts[ACTION_RAISE] / total_actions,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.player_id}, name={self.name!r})"
