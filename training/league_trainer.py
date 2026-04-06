"""
league_trainer.py — Phase 3 League Training 학습 루프

구성:
  - AgentLeague (4×NFSPAgent + 2×RuleAgent + Snapshots) 생성
  - 매 게임: 매칭 구성 → 게임 실행 → 학습 업데이트
  - 주기적 스냅샷 저장 및 착취가능성 측정
  - 전략 다양성 지표 (VPIP, PFR, AF) 모니터링

성공 기준:
  - 4개 에이전트가 서로 다른 전략 프로파일(VPIP, PFR, AF)을 보임
  - Exploitability가 학습 진행에 따라 감소
  - 한 에이전트가 동일 전략으로 나머지를 지속 지배하지 않음
"""

import os
import csv
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

from engine.config import GameConfig
from engine.game import TexasHoldemGame
from training.league import AgentLeague
from evaluation.exploitability import ExploitabilityMeasurer


@dataclass
class LeagueTrainingConfig:
    num_games:           int   = 300
    rounds_per_game:     int   = 200
    initial_stack:       int   = 1000
    small_blind:         int   = 5
    big_blind:           int   = 10
    snapshot_every:      int   = 15    # N게임마다 스냅샷
    exploit_every:       int   = 30    # N게임마다 착취가능성 측정
    checkpoint_every:    int   = 30    # N게임마다 체크포인트
    log_every:           int   = 10    # N게임마다 콘솔 출력
    exploit_games:       int   = 15    # 착취가능성 측정용 게임 수
    exploit_rounds:      int   = 80    # 착취가능성 측정용 라운드 수
    exploiter_ratio:     float = 0.4   # 게임당 exploiter 포함 확률
    max_snapshots:       int   = 20    # 유지할 최대 스냅샷 수 (에이전트당)
    checkpoint_dir:      str   = "checkpoints/league"
    log_dir:             str   = "logs/league"
    seed:                int   = 42


class LeagueTrainer:
    """Phase 3 League Training 루프 관리자"""

    def __init__(self, config: Optional[LeagueTrainingConfig] = None):
        self.config = config or LeagueTrainingConfig()
        self._metrics: List[Dict] = []
        self._exploit_history: List[Dict] = []
        self._league: Optional[AgentLeague] = None

    # ══════════════════════════════════════════════════════
    #  메인 학습 루프
    # ══════════════════════════════════════════════════════

    def train(self) -> AgentLeague:
        """League Training 실행. 완료된 AgentLeague를 반환합니다."""
        cfg = self.config
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        # 리그 초기화
        self._league = AgentLeague(
            initial_stack   = cfg.initial_stack,
            small_blind     = cfg.small_blind,
            big_blind       = cfg.big_blind,
            snapshot_every  = cfg.snapshot_every,
            max_snapshots   = cfg.max_snapshots,
            exploiter_ratio = cfg.exploiter_ratio,
            seed            = cfg.seed,
        )

        # 착취가능성 측정기
        measurer = ExploitabilityMeasurer(
            games         = cfg.exploit_games,
            rounds        = cfg.exploit_rounds,
            initial_stack = cfg.initial_stack,
            seed          = cfg.seed + 50000,
        )

        print(f"\n{'='*60}")
        print(f"Phase 3 League Training 시작")
        print(f"  에이전트: {[a.name for a in self._league.main_agents]}")
        print(f"  총 게임: {cfg.num_games} × {cfg.rounds_per_game}라운드")
        print(f"{'='*60}\n")

        for game_idx in range(cfg.num_games):

            # ── 매칭 구성 & 게임 실행 ────────────────────────
            agents = self._league.create_matchup()
            game_config = GameConfig(
                num_players   = 4,
                initial_stack = cfg.initial_stack,
                small_blind   = cfg.small_blind,
                big_blind     = cfg.big_blind,
                max_rounds    = cfg.rounds_per_game,
                seed          = cfg.seed + game_idx,
                verbose       = False,
            )
            result = TexasHoldemGame(agents, game_config).run()

            # ── 결과 기록 ─────────────────────────────────────
            self._league.record_result(result['final_stacks'])

            # ── 스냅샷 ────────────────────────────────────────
            snapshotted = self._league.maybe_snapshot(game_idx)

            # ── 메트릭 수집 ───────────────────────────────────
            metrics = self._collect_metrics(game_idx, result)
            self._metrics.append(metrics)

            # ── 로그 출력 ─────────────────────────────────────
            if (game_idx + 1) % cfg.log_every == 0:
                self._log(game_idx + 1)
                if snapshotted:
                    print(f"  📸 스냅샷 저장 (game {game_idx+1})")

            # ── 착취가능성 측정 ───────────────────────────────
            if (game_idx + 1) % cfg.exploit_every == 0:
                self._measure_exploitability(game_idx + 1, measurer)

            # ── 체크포인트 ────────────────────────────────────
            if (game_idx + 1) % cfg.checkpoint_every == 0:
                self._league.save(cfg.checkpoint_dir, game_idx + 1)
                print(f"  💾 체크포인트 저장 (step={game_idx+1})")

        # 최종 처리
        self._league.save(cfg.checkpoint_dir, cfg.num_games)
        self._save_metrics_csv()
        self._save_exploit_csv()
        self._print_final_summary()

        return self._league

    # ══════════════════════════════════════════════════════
    #  내부 헬퍼
    # ══════════════════════════════════════════════════════

    def _collect_metrics(self, game_idx: int, result: Dict) -> Dict:
        """게임 결과로 메트릭 딕셔너리 생성"""
        cfg          = self.config
        final_stacks = result['final_stacks']
        agent_stats  = result['agent_stats']

        row: Dict[str, Any] = {'game': game_idx + 1}

        for i, agent in enumerate(self._league.main_agents):
            stack = final_stacks.get(i, cfg.initial_stack)
            delta = stack - cfg.initial_stack
            stats = agent_stats.get(i, {})

            # 학습 손실 (최근 것)
            actor_loss  = agent.train_stats[-1]['actor_loss']  if agent.train_stats else 0.0
            critic_loss = agent.train_stats[-1]['critic_loss'] if agent.train_stats else 0.0
            sl_loss     = agent.sl_stats[-1]['sl_loss']        if agent.sl_stats    else 0.0

            row[f'p{i}_delta']       = delta
            row[f'p{i}_fold_rate']   = round(stats.get('fold_rate',  0.0), 3)
            row[f'p{i}_call_rate']   = round(stats.get('call_rate',  0.0), 3)
            row[f'p{i}_raise_rate']  = round(stats.get('raise_rate', 0.0), 3)
            row[f'p{i}_actor_loss']  = round(actor_loss,  4)
            row[f'p{i}_critic_loss'] = round(critic_loss, 4)
            row[f'p{i}_sl_loss']     = round(sl_loss, 4)
            row[f'p{i}_reservoir']   = agent.reservoir_size

        return row

    def _measure_exploitability(
        self,
        game_num:  int,
        measurer: 'ExploitabilityMeasurer',
    ) -> None:
        """착취가능성 측정 및 기록"""
        results = measurer.measure_league(self._league.main_agents)

        row: Dict[str, Any] = {'game': game_num}
        for name, r in results.items():
            row[f'{name}_exploit']    = round(r['exploitability'],   1)
            row[f'{name}_chip_delta'] = round(r['agent_chip_delta'], 1)
            row[f'{name}_win_rate']   = round(r['win_rate'], 3)
        self._exploit_history.append(row)

        # 콘솔 출력
        print(f"\n  [착취가능성 측정 @ game {game_num}]")
        for name, r in results.items():
            print(
                f"    {name:<16}  "
                f"exploit={r['exploitability']:+.1f}  "
                f"chip_delta={r['agent_chip_delta']:+.1f}  "
                f"win_rate={r['win_rate']:.1%}"
            )
        print()

    def _log(self, game_num: int) -> None:
        """롤링 window 로그 출력"""
        window = self._metrics[-self.config.log_every:]
        print(f"[Game {game_num:>4}/{self.config.num_games}]")

        for i, agent in enumerate(self._league.main_agents):
            deltas    = [m[f'p{i}_delta']      for m in window]
            fold_r    = [m[f'p{i}_fold_rate']  for m in window]
            raise_r   = [m[f'p{i}_raise_rate'] for m in window]
            actor_l   = self._metrics[-1][f'p{i}_actor_loss']
            sl_l      = self._metrics[-1][f'p{i}_sl_loss']

            print(
                f"  {agent.name:<16}"
                f"  avg_Δ={np.mean(deltas):+.1f}"
                f"  fold={np.mean(fold_r):.1%}"
                f"  raise={np.mean(raise_r):.1%}"
                f"  actor_loss={actor_l:.4f}"
                f"  sl_loss={sl_l:.4f}"
                f"  reservoir={agent.reservoir_size}"
            )

    def _save_metrics_csv(self) -> None:
        if not self._metrics:
            return
        path = os.path.join(self.config.log_dir, "league_metrics.csv")
        _write_csv(path, self._metrics)
        print(f"  📈 메트릭 저장: {path}")

    def _save_exploit_csv(self) -> None:
        if not self._exploit_history:
            return
        path = os.path.join(self.config.log_dir, "exploitability.csv")
        _write_csv(path, self._exploit_history)
        print(f"  📉 착취가능성 저장: {path}")

    def _print_final_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"League Training 완료 — {self.config.num_games}게임")
        print(f"{'='*60}")

        stats = self._league.get_stats()
        print(f"\n{'에이전트':<16} {'avg_Δ':>8} {'win%':>7} {'η':>5} {'reservoir':>10}")
        print("-" * 55)
        for name, s in stats.items():
            print(
                f"{name:<16}"
                f"  {s['avg_chip_delta']:>+8.1f}"
                f"  {s['win_rate']:>6.1%}"
                f"  {s['eta']:>5.2f}"
                f"  {s['reservoir_size']:>10,}"
            )

        if self._exploit_history:
            last = self._exploit_history[-1]
            print(f"\n[최종 착취가능성]")
            for i, agent in enumerate(self._league.main_agents):
                k = f"{agent.name}_exploit"
                if k in last:
                    print(f"  {agent.name}: {last[k]:+.1f}")

        print(f"\n체크포인트: {self.config.checkpoint_dir}")
        print(f"로그:       {self.config.log_dir}")


# ══════════════════════════════════════════════════════════
#  유틸리티
# ══════════════════════════════════════════════════════════

def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
