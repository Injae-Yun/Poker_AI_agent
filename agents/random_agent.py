"""
random_agent.py — 랜덤 에이전트

합법 액션 중 무작위로 선택합니다.
레이즈 시 금액도 min~max 사이에서 무작위 선택.

Phase 0 베이스라인 및 테스트 용도.
"""

import random
from typing import Dict, Any

from agents.base_agent import BaseAgent
from engine.state import AgentObservation, ACTION_FOLD, ACTION_CALL, ACTION_RAISE, ACTION_CHECK


class RandomAgent(BaseAgent):
    """완전 무작위 에이전트 — 베이스라인"""

    def __init__(self, player_id: int, name: str = "Random", seed: int = None):
        super().__init__(player_id, name)
        self._rng = random.Random(seed)

    def declare_action(self, obs: AgentObservation) -> Dict[str, Any]:
        valid = obs.valid_actions
        chosen = self._rng.choice(valid)

        action = chosen['action']
        amount = chosen.get('amount', 0)

        # 레이즈 시 min~max 사이 무작위 금액
        if action == ACTION_RAISE:
            min_a = chosen['min_amount']
            max_a = chosen['max_amount']
            amount = self._rng.randint(min_a, max_a)

        self._record_action(action)
        return {'action': action, 'amount': amount}
