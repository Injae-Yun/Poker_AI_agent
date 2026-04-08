"""
config.py — 게임 설정 옵션

GameConfig: 게임 타입, 플레이어 수, 블라인드 구조 등을 관리합니다.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GameConfig:
    # ── 게임 기본 설정 ────────────────────────────────────
    game_type:     str = "texas_holdem"    # 향후 확장 예정
    num_players:   int = 4                 # 현재 4인 고정
    initial_stack: int = 1000             # 초기 칩

    # ── 블라인드 구조 ─────────────────────────────────────
    small_blind:   int = 5
    big_blind:     int = 10

    # ── 진행 설정 ─────────────────────────────────────────
    max_rounds:    int = 100              # 총 게임 라운드 수
    max_raises_per_street: int = 4        # 한 스트리트 최대 레이즈 횟수

    # ── 기타 ──────────────────────────────────────────────
    seed:          Optional[int] = None   # 재현 가능성을 위한 시드
    verbose:       bool = True            # 콘솔 출력 여부

    def __post_init__(self):
        assert self.num_players >= 2,       "플레이어는 2명 이상이어야 합니다."
        assert self.big_blind > 0,          "빅 블라인드는 0보다 커야 합니다."
        assert self.small_blind > 0,        "스몰 블라인드는 0보다 커야 합니다."
        assert self.small_blind < self.big_blind, "스몰 블라인드는 빅 블라인드보다 작아야 합니다."
        assert self.initial_stack >= self.big_blind * 10, \
            "초기 스택은 빅 블라인드의 10배 이상이어야 합니다."

    @property
    def ante(self) -> int:
        return 0   # 앤티 없음 (향후 옵션 추가 가능)
