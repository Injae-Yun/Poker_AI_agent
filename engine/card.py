"""
card.py — 카드 및 덱 표현

Card: 단일 카드 (suit + rank)
Deck: 52장 표준 덱, 셔플 및 딜 기능
"""

import random
from typing import List, Optional


SUITS = ['s', 'h', 'd', 'c']          # spade, heart, diamond, club
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}  # '2'=0 ... 'A'=12


class Card:
    """단일 카드를 표현합니다."""

    __slots__ = ('rank', 'suit', '_value')

    def __init__(self, rank: str, suit: str):
        assert rank in RANK_VALUE, f"Invalid rank: {rank}"
        assert suit in SUITS,      f"Invalid suit: {suit}"
        self.rank   = rank
        self.suit   = suit
        self._value = RANK_VALUE[rank]

    # ── 표현 ──────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    # ── 비교 (rank만 비교) ────────────────────────────────
    def __lt__(self, other) -> bool:
        return self._value < other._value

    @property
    def value(self) -> int:
        """숫자 값 (2=0 ... A=12)"""
        return self._value

    @classmethod
    def from_str(cls, s: str) -> 'Card':
        """'Ah', 'Ts', '2c' 형태의 문자열에서 Card 생성"""
        assert len(s) == 2, f"Card string must be 2 chars, got: {s!r}"
        return cls(s[0], s[1])


class Deck:
    """52장 표준 덱"""

    def __init__(self, seed: Optional[int] = None):
        self._rng   = random.Random(seed)
        self._cards = self._build()
        self._dealt: List[Card] = []

    # ── 내부 ──────────────────────────────────────────────
    @staticmethod
    def _build() -> List[Card]:
        return [Card(r, s) for s in SUITS for r in RANKS]

    def reset(self) -> None:
        """덱을 52장으로 재구성하고 셔플합니다."""
        self._cards = self._build()
        self._dealt.clear()
        self._rng.shuffle(self._cards)

    def shuffle(self) -> None:
        """현재 남은 카드만 셔플합니다 (reset과 구분)."""
        self._rng.shuffle(self._cards)
        self._dealt.clear()

    def deal(self, n: int = 1) -> List[Card]:
        if n > len(self._cards):
            raise RuntimeError(f"덱에 카드가 부족합니다. (요청: {n}, 남은: {len(self._cards)})")
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        self._dealt.extend(dealt)
        return dealt

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    def burn(self) -> None:
        """번 카드 (버림)"""
        self.deal(1)

    # ── 조회 ──────────────────────────────────────────────
    @property
    def remaining(self) -> int:
        return len(self._cards)

    @property
    def dealt_cards(self) -> List[Card]:
        return list(self._dealt)

    def __repr__(self) -> str:
        return f"Deck(remaining={self.remaining})"
