"""
game.py — 텍사스 홀덤 게임 루프

TexasHoldemGame: 설정에 따라 N 라운드를 진행하고 결과를 반환합니다.

구조:
  game.run()
    └─ _play_round()  (핸드 1회)
         ├─ _deal_hole_cards()
         ├─ _betting_round('preflop')
         ├─ _deal_community('flop', 3)
         ├─ _betting_round('flop')
         ├─ _deal_community('turn', 1)
         ├─ _betting_round('turn')
         ├─ _deal_community('river', 1)
         ├─ _betting_round('river')
         └─ _showdown()
"""

import random
from typing import List, Dict, Any, Optional

from engine.card import Card, Deck
from engine.config import GameConfig
from engine.hand_eval import evaluate_hand
from engine.state import (
    AgentObservation, PlayerState, BettingAction,
    ACTION_FOLD, ACTION_CALL, ACTION_RAISE, ACTION_CHECK,
    STREETS,
)
from agents.base_agent import BaseAgent


class _Player:
    """게임 내부에서 사용하는 플레이어 상태 (에이전트와 분리)"""

    def __init__(self, agent: BaseAgent, stack: int):
        self.agent          = agent
        self.player_id      = agent.player_id
        self.stack          = stack
        self.hole_cards: List[Card] = []

        # 라운드별 초기화 필드
        self.is_active      = True    # 폴드하지 않음
        self.is_all_in      = False
        self.bet_this_street = 0
        self.total_bet      = 0       # 이번 핸드 총 투자금

    def reset_for_round(self):
        self.hole_cards      = []
        self.is_active       = True
        self.is_all_in       = False
        self.bet_this_street = 0
        self.total_bet       = 0

    def reset_for_street(self):
        self.bet_this_street = 0

    def place_bet(self, amount: int) -> int:
        """실제 베팅 처리. 스택 초과 시 올인. 실제 낸 금액 반환."""
        actual = min(amount, self.stack)
        self.stack          -= actual
        self.bet_this_street += actual
        self.total_bet      += actual
        if self.stack == 0:
            self.is_all_in = True
        return actual

    def to_public_state(self) -> PlayerState:
        return PlayerState(
            player_id        = self.player_id,
            stack            = self.stack,
            is_active        = self.is_active,
            is_all_in        = self.is_all_in,
            bet_this_street  = self.bet_this_street,
            total_bet        = self.total_bet,
        )


class TexasHoldemGame:
    """텍사스 홀덤 게임 진행기"""

    def __init__(self, agents: List[BaseAgent], config: GameConfig):
        assert len(agents) == config.num_players, \
            f"에이전트 수({len(agents)})가 설정 플레이어 수({config.num_players})와 다릅니다."

        self.config  = config
        self._rng    = random.Random(config.seed)
        self._deck   = Deck(seed=config.seed)

        # 플레이어 객체 생성
        self._players: List[_Player] = [
            _Player(agent, config.initial_stack) for agent in agents
        ]

        self._dealer_idx  = 0        # 딜러 버튼 포지션
        self._round_num   = 0
        self._round_logs: List[Dict[str, Any]] = []

        # 에이전트에게 게임 시작 알림
        for p in self._players:
            p.agent.on_game_start(config)

    # ══════════════════════════════════════════════════════
    #  공개 API
    # ══════════════════════════════════════════════════════

    def run(self) -> Dict[str, Any]:
        """전체 게임 실행. 결과 딕셔너리 반환."""
        for _ in range(self.config.max_rounds):
            if self._count_active_players_with_chips() < 2:
                break
            self._play_round()
            self._dealer_idx = (self._dealer_idx + 1) % len(self._players)

        return self._build_game_result()

    # ══════════════════════════════════════════════════════
    #  라운드 진행
    # ══════════════════════════════════════════════════════

    def _play_round(self):
        self._round_num += 1

        # 초기화 — 매 라운드 덱을 52장으로 완전 재구성
        self._deck.reset()
        community_cards: List[Card] = []
        betting_history: List[BettingAction] = []
        pot = 0

        # 스택이 있는 플레이어만 참여
        for p in self._players:
            if p.stack > 0:
                p.reset_for_round()
            else:
                p.is_active = False

        active = self._active_players()

        # 라운드 시작 알림
        for p in active:
            p.agent.on_round_start(self._round_num)

        # ── 블라인드 ───────────────────────────────────────
        n = len(active)
        sb_idx = (self._dealer_idx + 1) % len(self._players)
        bb_idx = (self._dealer_idx + 2) % len(self._players)

        # 실제 블라인드를 낼 수 있는 플레이어 찾기
        sb_player = self._get_next_active_from(sb_idx)
        bb_player = self._get_next_active_from(bb_idx, skip=sb_player)

        sb_paid = sb_player.place_bet(self.config.small_blind)
        bb_paid = bb_player.place_bet(self.config.big_blind)
        pot += sb_paid + bb_paid

        betting_history.append(BettingAction(
            sb_player.player_id, 'preflop', 'blind', sb_paid, pot, sb_player.stack))
        betting_history.append(BettingAction(
            bb_player.player_id, 'preflop', 'blind', bb_paid, pot, bb_player.stack))

        # ── 홀 카드 딜 ─────────────────────────────────────
        for p in active:
            p.hole_cards = self._deck.deal(2)

        # ── 프리플롭 베팅 ──────────────────────────────────
        current_bet = self.config.big_blind
        pot, betting_history = self._betting_round(
            street          = 'preflop',
            community_cards = community_cards,
            betting_history = betting_history,
            pot             = pot,
            current_bet     = current_bet,
            first_actor_idx = (bb_idx + 1) % len(self._players),
        )

        if self._count_active_in_list(active) <= 1:
            return self._end_round(active, pot, community_cards, betting_history, False)

        # ── 플롭 ──────────────────────────────────────────
        self._deck.burn()
        community_cards.extend(self._deck.deal(3))
        for p in active:
            p.reset_for_street()

        pot, betting_history = self._betting_round(
            street          = 'flop',
            community_cards = community_cards,
            betting_history = betting_history,
            pot             = pot,
            current_bet     = 0,
            first_actor_idx = (self._dealer_idx + 1) % len(self._players),
        )

        if self._count_active_in_list(active) <= 1:
            return self._end_round(active, pot, community_cards, betting_history, False)

        # ── 턴 ────────────────────────────────────────────
        self._deck.burn()
        community_cards.extend(self._deck.deal(1))
        for p in active:
            p.reset_for_street()

        pot, betting_history = self._betting_round(
            street          = 'turn',
            community_cards = community_cards,
            betting_history = betting_history,
            pot             = pot,
            current_bet     = 0,
            first_actor_idx = (self._dealer_idx + 1) % len(self._players),
        )

        if self._count_active_in_list(active) <= 1:
            return self._end_round(active, pot, community_cards, betting_history, False)

        # ── 리버 ──────────────────────────────────────────
        self._deck.burn()
        community_cards.extend(self._deck.deal(1))
        for p in active:
            p.reset_for_street()

        pot, betting_history = self._betting_round(
            street          = 'river',
            community_cards = community_cards,
            betting_history = betting_history,
            pot             = pot,
            current_bet     = 0,
            first_actor_idx = (self._dealer_idx + 1) % len(self._players),
        )

        if self._count_active_in_list(active) <= 1:
            return self._end_round(active, pot, community_cards, betting_history, False)

        # ── 쇼다운 ────────────────────────────────────────
        return self._end_round(active, pot, community_cards, betting_history, True)

    # ══════════════════════════════════════════════════════
    #  베팅 라운드
    # ══════════════════════════════════════════════════════

    def _betting_round(
        self,
        street:          str,
        community_cards: List[Card],
        betting_history: List[BettingAction],
        pot:             int,
        current_bet:     int,
        first_actor_idx: int,
    ):
        """
        한 스트리트의 베팅 라운드를 완료합니다.
        Returns: (updated_pot, updated_betting_history)
        """
        players         = self._players
        n               = len(players)
        raises_this_st  = 0

        # 액션해야 할 플레이어 순서 결정
        # 각 플레이어가 현재 베팅 금액에 맞출 때까지 반복
        acted           = set()   # 현재 배팅 수준에서 액션한 플레이어
        last_raiser     = None

        idx = first_actor_idx
        max_iterations  = n * (self.config.max_raises_per_street + 2)
        iteration       = 0

        while iteration < max_iterations:
            iteration += 1
            p = players[idx % n]

            # 폴드했거나 올인인 플레이어 스킵
            if not p.is_active or p.is_all_in:
                idx += 1
                continue

            # 현재 베팅 라운드가 끝났는지 확인
            # (모든 활성 플레이어가 동일 금액을 낸 경우)
            still_to_act = [
                q for q in players
                if q.is_active and not q.is_all_in
                   and q.bet_this_street < current_bet
            ]

            # 아무도 더 낼 필요가 없고, 현재 플레이어도 맞췄으면 종료
            if (not still_to_act and p in acted
                    and p.bet_this_street >= current_bet):
                break

            # 콜 금액 계산
            call_amount = max(0, current_bet - p.bet_this_street)

            # 최소/최대 레이즈 계산
            min_raise = current_bet + self.config.big_blind  # 현재 베팅 + BB
            max_raise = p.stack + p.bet_this_street          # 올인

            # 관찰 생성 및 에이전트 호출
            obs = self._build_observation(
                player          = p,
                street          = street,
                community_cards = community_cards,
                pot             = pot,
                call_amount     = call_amount,
                min_raise       = max(0, min_raise - p.bet_this_street),
                max_raise       = max(0, max_raise - p.bet_this_street),
                raises_this_street = raises_this_st,
                betting_history = betting_history,
            )

            decision = p.agent.declare_action(obs)
            action   = decision['action']
            amount   = int(decision.get('amount', 0))

            # ── 액션 처리 ──────────────────────────────────
            if action == ACTION_FOLD:
                p.is_active = False
                actual_paid = 0

            elif action in (ACTION_CALL, ACTION_CHECK):
                actual_paid = p.place_bet(call_amount)
                pot += actual_paid

            elif action == ACTION_RAISE:
                # 레이즈 금액 = call_amount + 추가 올리는 금액
                total_to_put = call_amount + max(
                    amount,
                    self.config.big_blind
                )
                total_to_put = min(total_to_put, p.stack)  # 올인 한도
                actual_paid  = p.place_bet(total_to_put)
                pot         += actual_paid
                new_bet      = p.bet_this_street
                if new_bet > current_bet:
                    current_bet  = new_bet
                    raises_this_st += 1
                    last_raiser  = p.player_id
                    acted        = {p.player_id}   # 레이즈 후 다른 플레이어 재액션 필요
            else:
                # 알 수 없는 액션은 폴드로 처리
                p.is_active = False
                actual_paid = 0
                action      = ACTION_FOLD

            acted.add(p.player_id)

            # 히스토리 기록
            betting_history.append(BettingAction(
                player_id       = p.player_id,
                street          = street,
                action          = action,
                amount          = actual_paid,
                pot_at_action   = pot,
                stack_at_action = p.stack,
            ))

            # 활성 플레이어가 1명만 남으면 즉시 종료
            if self._count_active_in_list(self._players) <= 1:
                break

            idx += 1

        return pot, betting_history

    # ══════════════════════════════════════════════════════
    #  라운드 종료 및 쇼다운
    # ══════════════════════════════════════════════════════

    def _end_round(
        self,
        players:         List[_Player],
        pot:             int,
        community_cards: List[Card],
        betting_history: List[BettingAction],
        showdown:        bool,
    ):
        active_players = [p for p in players if p.is_active]

        # ── 승자 결정 ──────────────────────────────────────
        if len(active_players) == 1:
            winners = [active_players[0]]
        else:
            # 쇼다운: 핸드 강도 비교
            winners = self._determine_winners(active_players, community_cards)

        # ── 팟 분배 ────────────────────────────────────────
        share = pot // len(winners)
        remainder = pot % len(winners)
        for i, w in enumerate(winners):
            w.stack += share + (1 if i == 0 else 0) * remainder

        winner_ids  = [w.player_id for w in winners]
        player_hands = {
            p.player_id: p.hole_cards for p in active_players
        } if showdown else {}

        # ── 에이전트 on_round_end 호출 ────────────────────
        stacks_before = {p.player_id: p.stack - (share if p in winners else 0)
                         for p in players}

        for p in players:
            chip_delta = p.stack - (self.config.initial_stack
                         if self._round_num == 1
                         else self._get_prev_stack(p.player_id))
            # 실질적인 델타: 이번 핸드에서의 변화
            hand_delta = -p.total_bet + (share if p in winners else 0) + \
                         (remainder if (winners and p == winners[0]) else 0)

            result = {
                'round_num':      self._round_num,
                'winners':        winner_ids,
                'pot':            pot,
                'chip_delta':     hand_delta,
                'showdown':       showdown,
                'community_cards': community_cards,
                'player_hands':   player_hands,
                'my_stack':       p.stack,
                'betting_history': betting_history,
            }
            p.agent.on_round_end(result)

        # ── 로그 저장 ──────────────────────────────────────
        log = {
            'round_num':       self._round_num,
            'winners':         winner_ids,
            'pot':             pot,
            'showdown':        showdown,
            'community_cards': [str(c) for c in community_cards],
            'player_hands': {
                pid: [str(c) for c in cards]
                for pid, cards in player_hands.items()
            },
            'stacks': {p.player_id: p.stack for p in self._players},
            'betting_history': [
                {
                    'player_id': a.player_id,
                    'street':    a.street,
                    'action':    a.action,
                    'amount':    a.amount,
                }
                for a in betting_history
            ],
        }
        self._round_logs.append(log)

        if self.config.verbose:
            self._print_round_summary(log)

    def _determine_winners(
        self, players: List[_Player], community: List[Card]
    ) -> List[_Player]:
        """핸드 강도 비교로 승자(들)를 결정합니다."""
        best_score = (-1, [])
        winners    = []

        for p in players:
            all_cards = p.hole_cards + community
            score, tb, _ = evaluate_hand(all_cards)
            if (score, tb) > best_score:
                best_score = (score, tb)
                winners    = [p]
            elif (score, tb) == best_score:
                winners.append(p)

        return winners

    # ══════════════════════════════════════════════════════
    #  관찰 생성
    # ══════════════════════════════════════════════════════

    def _build_observation(
        self,
        player:          _Player,
        street:          str,
        community_cards: List[Card],
        pot:             int,
        call_amount:     int,
        min_raise:       int,
        max_raise:       int,
        raises_this_street: int,
        betting_history: List[BettingAction],
    ) -> AgentObservation:

        # 포지션: 딜러 기준 상대 위치
        position = (self._players.index(player) - self._dealer_idx) % len(self._players)

        return AgentObservation(
            player_id       = player.player_id,
            hole_cards      = list(player.hole_cards),
            stack           = player.stack,
            position        = position,
            community_cards = list(community_cards),
            pot             = pot,
            street          = street,
            round_num       = self._round_num,
            call_amount     = call_amount,
            min_raise       = min_raise,
            max_raise       = max_raise,
            raises_this_street = raises_this_street,
            players         = [p.to_public_state() for p in self._players],
            betting_history = list(betting_history),
        )

    # ══════════════════════════════════════════════════════
    #  유틸리티
    # ══════════════════════════════════════════════════════

    def _active_players(self) -> List[_Player]:
        return [p for p in self._players if p.stack > 0]

    def _count_active_players_with_chips(self) -> int:
        return sum(1 for p in self._players if p.stack > 0)

    def _count_active_in_list(self, players: List[_Player]) -> int:
        return sum(1 for p in players if p.is_active)

    def _get_next_active_from(self, start: int, skip: Optional[_Player] = None) -> _Player:
        n = len(self._players)
        for i in range(n):
            p = self._players[(start + i) % n]
            if p.stack > 0 and p != skip:
                return p
        raise RuntimeError("활성 플레이어를 찾을 수 없습니다.")

    def _get_prev_stack(self, player_id: int) -> int:
        """이전 라운드 스택 (로그에서 조회)"""
        if len(self._round_logs) < 2:
            return self.config.initial_stack
        prev_log = self._round_logs[-2]
        return prev_log['stacks'].get(player_id, self.config.initial_stack)

    def _build_game_result(self) -> Dict[str, Any]:
        return {
            'rounds_played': self._round_num,
            'final_stacks':  {p.player_id: p.stack for p in self._players},
            'agent_stats':   {p.player_id: p.agent.stats for p in self._players},
            'round_logs':    self._round_logs,
        }

    def _print_round_summary(self, log: Dict[str, Any]):
        winners_str = ', '.join(f"P{w}" for w in log['winners'])
        stacks_str  = '  '.join(
            f"P{pid}:{s}" for pid, s in sorted(log['stacks'].items())
        )
        community_str = ' '.join(log['community_cards']) or '(없음)'
        print(
            f"[Round {log['round_num']:>3}] "
            f"Winner: {winners_str:<8} "
            f"Pot: {log['pot']:>5}  "
            f"Community: {community_str:<17}  "
            f"Stacks: {stacks_str}"
        )
