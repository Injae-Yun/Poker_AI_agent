"""
trainer.py — Phase 2 학습 루프

RLAgent(seat 0) vs RuleAgent × 3 구성으로 학습합니다.
- epsilon 감쇠 스케줄 (게임당 ×0.98)
- N게임마다 체크포인트 저장
- 롤링 평균으로 수렴 신호 모니터링
- Best model selection: 롤링 chip_delta 개선 시 best_net_P0.pth 갱신
- Early stopping: patience 게임 이상 개선 없으면 조기 종료 (min_games 이후)
"""

import os
import csv
import shutil
from dataclasses import dataclass
from typing import List, Dict, Optional

from engine.config import GameConfig
from engine.game import TexasHoldemGame
from agents.rl_agent import RLAgent
from agents.rule_agent import RuleAgent


@dataclass
class TrainingConfig:
    num_games:        int   = 200
    rounds_per_game:  int   = 200
    initial_stack:    int   = 10000
    small_blind:      int   = 5
    big_blind:        int   = 10
    checkpoint_every: int   = 20    # N게임마다 체크포인트
    log_every:        int   = 10    # N게임마다 콘솔 출력
    epsilon_start:    float = 0.20
    epsilon_end:      float = 0.02
    epsilon_decay:    float = 0.98  # 게임당 곱셈 감쇠
    entropy_coef:     float = 0.02
    update_every:     int   = 1     # 핸드당 업데이트
    checkpoint_dir:   str   = "checkpoints"
    log_dir:          str   = "logs/training"
    seed:             int   = 42
    # ── Early Stop / Best Model ────────────────────────────
    early_stop_patience: int  = 40   # 개선 없으면 N게임 후 중단 (0 = 비활성)
    min_games:           int  = 60   # early stop 검사 시작 게임 수
    eval_window:         int  = 20   # 지표 평가에 사용할 롤링 윈도우 크기


class Trainer:
    """Phase 2 학습 루프 관리자"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self._metrics: List[Dict] = []
        self._rl_agent: Optional[RLAgent] = None

        # best model tracking
        self._best_metric:    float = -float('inf')   # chip_delta 롤링 평균 (높을수록 좋음)
        self._no_improve_cnt: int   = 0
        self._best_step:      int   = 0

    def train(self) -> RLAgent:
        """학습 실행. 완료된 RLAgent를 반환합니다."""
        cfg = self.config
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        self._rl_agent = RLAgent(
            player_id    = 0,
            name         = "RLAgent",
            lr           = 3e-4,
            entropy_coef = cfg.entropy_coef,
            epsilon      = cfg.epsilon_start,
            update_every = cfg.update_every,
            initial_stack= cfg.initial_stack,
            big_blind    = cfg.big_blind,
        )

        epsilon = cfg.epsilon_start
        stopped_early = False

        for game_idx in range(cfg.num_games):
            agents = self._build_agents(game_idx, epsilon)
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

            metrics = self._collect_metrics(game_idx, result, epsilon)
            self._metrics.append(metrics)

            # epsilon 감쇠
            epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)
            self._rl_agent.epsilon = epsilon

            # 로그 출력
            if (game_idx + 1) % cfg.log_every == 0:
                self._log(game_idx + 1, epsilon)

            # 체크포인트 저장
            if (game_idx + 1) % cfg.checkpoint_every == 0:
                self._rl_agent.save(cfg.checkpoint_dir, game_idx + 1)
                print(f"  [체크포인트 저장] step={game_idx + 1}")

            # best model 갱신 + early stop 판정
            if (game_idx + 1) >= cfg.min_games:
                improved = self._update_best(game_idx + 1)
                if not improved and cfg.early_stop_patience > 0:
                    if self._no_improve_cnt >= cfg.early_stop_patience:
                        print(
                            f"\n[Early Stop] {cfg.early_stop_patience}게임 동안 개선 없음 "
                            f"(game {game_idx + 1}/{cfg.num_games})"
                            f"  best step={self._best_step}"
                            f"  best chip_delta={self._best_metric:+.1f}"
                        )
                        stopped_early = True
                        break

        # 최종 처리
        final_step = self._metrics[-1]['game'] if self._metrics else cfg.num_games
        if not stopped_early:
            self._rl_agent.save(cfg.checkpoint_dir, final_step)
        self._save_metrics_csv()

        total_games = len(self._metrics)
        print(f"\n학습 완료. 총 {total_games}게임 × {cfg.rounds_per_game}라운드")
        print(f"  Best step={self._best_step}  best chip_delta={self._best_metric:+.1f}")
        print(f"  Best model: {cfg.checkpoint_dir}/best_net_P0.pth")
        return self._rl_agent

    # ══════════════════════════════════════════════════════
    #  Best Model / Early Stop
    # ══════════════════════════════════════════════════════

    def _update_best(self, game_num: int) -> bool:
        """
        롤링 윈도우 chip_delta 평균으로 best model 판정.
        개선됐으면 True 반환 + best checkpoint 갱신.
        개선 안 됐으면 False + no_improve_cnt 증가.
        """
        cfg    = self.config
        window = self._metrics[-cfg.eval_window:]
        avg    = sum(m['rl_chip_delta'] for m in window) / len(window)

        if avg > self._best_metric:
            self._best_metric    = avg
            self._best_step      = game_num
            self._no_improve_cnt = 0
            # best model 저장 (현재 체크포인트를 best로 복사)
            src = os.path.join(cfg.checkpoint_dir, f"net_P0_step_{game_num}.pth")
            dst = os.path.join(cfg.checkpoint_dir, "best_net_P0.pth")
            # 아직 이 step 체크포인트가 없으면 임시 저장
            if not os.path.exists(src):
                self._rl_agent.save(cfg.checkpoint_dir, game_num)
            shutil.copy2(src, dst)
            print(
                f"  [Best Model] step={game_num}"
                f"  chip_delta(avg{cfg.eval_window})={avg:+.1f}"
                f"  -> best_net_P0.pth"
            )
            return True
        else:
            self._no_improve_cnt += 1
            return False

    # ══════════════════════════════════════════════════════
    #  내부 헬퍼
    # ══════════════════════════════════════════════════════

    def _build_agents(self, game_idx: int, epsilon: float) -> list:
        """seat 0: RLAgent, seat 1-3: RuleAgent"""
        self._rl_agent.epsilon = epsilon
        agents = [self._rl_agent]
        for i in range(1, 4):
            agents.append(RuleAgent(
                player_id = i,
                name      = f"Rule{i}",
                seed      = self.config.seed + game_idx * 100 + i,
            ))
        return agents

    def _collect_metrics(
        self,
        game_idx: int,
        result: Dict,
        epsilon: float,
    ) -> Dict:
        cfg = self.config
        final_stacks = result['final_stacks']
        agent_stats  = result['agent_stats']

        rl_stack    = final_stacks.get(0, cfg.initial_stack)
        rule_stacks = [final_stacks.get(i, cfg.initial_stack) for i in range(1, 4)]
        avg_rule    = sum(rule_stacks) / 3.0

        rl_stats = agent_stats.get(0, {})

        train_stats   = self._rl_agent.train_stats
        recent_actor  = train_stats[-1]['actor_loss']  if train_stats else 0.0
        recent_critic = train_stats[-1]['critic_loss'] if train_stats else 0.0

        return {
            'game':          game_idx + 1,
            'epsilon':       round(epsilon, 4),
            'rl_stack':      rl_stack,
            'rl_chip_delta': rl_stack - cfg.initial_stack,
            'rl_vs_rule':    round(rl_stack - avg_rule, 1),
            'rl_fold_rate':  round(rl_stats.get('fold_rate', 0.0), 3),
            'rl_call_rate':  round(rl_stats.get('call_rate', 0.0), 3),
            'rl_raise_rate': round(rl_stats.get('raise_rate', 0.0), 3),
            'actor_loss':    round(recent_actor,  4),
            'critic_loss':   round(recent_critic, 4),
        }

    def _log(self, game_num: int, epsilon: float) -> None:
        """롤링 20게임 평균 출력"""
        window    = self._metrics[-20:]
        avg_delta = sum(m['rl_chip_delta'] for m in window) / len(window)
        avg_raise = sum(m['rl_raise_rate'] for m in window) / len(window)
        avg_fold  = sum(m['rl_fold_rate']  for m in window) / len(window)
        last      = self._metrics[-1]

        print(
            f"[Game {game_num:>4}/{self.config.num_games}]"
            f"  e={epsilon:.3f}"
            f"  chip_delta(20avg)={avg_delta:+.1f}"
            f"  vs_rule={last['rl_vs_rule']:+.1f}"
            f"  fold={avg_fold:.1%}"
            f"  raise={avg_raise:.1%}"
            f"  actor_loss={last['actor_loss']:.4f}"
            f"  critic_loss={last['critic_loss']:.4f}"
            f"  no_improve={self._no_improve_cnt}"
        )

    def _save_metrics_csv(self) -> None:
        if not self._metrics:
            return
        path = os.path.join(self.config.log_dir, "training_metrics.csv")
        fieldnames = list(self._metrics[0].keys())
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._metrics)
        print(f"  메트릭 저장: {path}")
