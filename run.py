"""
run.py — 게임/학습/평가 통합 실행 진입점

사용법:
  # ── 게임 실행 ──────────────────────────────────────────
  python run.py                                       # 기본: rule vs rnd×3, 100라운드
  python run.py --mode single --agents rule,rnd,rnd,rnd --rounds 200 --verbose
  python run.py --mode test   --agents rule,rnd,rnd,rnd --games 50 --rounds 200

  에이전트 종류 (--agents):
    rule    RuleAgent (룰 기반)
    rnd     RandomAgent (랜덤)
    rl      RLAgent (Phase 2, 체크포인트 필요 시 --checkpoint 지정)
    nfsp    NFSPAgent (Phase 3)

  예: rule vs rl(학습된) 비교
  python run.py --mode test --agents rule,rl,rnd,rnd --checkpoint checkpoints --games 30

  # ── 학습 ──────────────────────────────────────────────
  python run.py --mode train2                         # Phase 2 RL 학습
  python run.py --mode train2 --games 300 --rounds 200

  python run.py --mode train3                         # Phase 3 League Training
  python run.py --mode train3 --games 300 --rounds 200

  # ── 평가 ──────────────────────────────────────────────
  python run.py --mode exploit                        # 착취가능성 측정 (fresh NFSPAgent)
  python run.py --mode exploit --exploit-games 20 --rounds 100

저장 결과:
  logs/games/          ← 단일 게임 로그 (JSON + JSONL)
  logs/reports/        ← 마크다운 리포트
  logs/league/         ← League Training 메트릭 CSV
  checkpoints/         ← Phase 2 체크포인트
  checkpoints/league/  ← Phase 3 체크포인트
"""

import argparse
import sys
from datetime import datetime


# ══════════════════════════════════════════════════════════
#  인자 파서
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Poker Agent Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", default="single",
                   choices=["single", "test", "train2", "train3", "exploit"],
                   help=(
                       "single : 단일 게임 실행\n"
                       "test   : 다중 게임 비교 리포트\n"
                       "train2 : Phase 2 RL (A2C) 학습\n"
                       "train3 : Phase 3 League Training (NFSP)\n"
                       "exploit: 착취가능성 측정"
                   ))
    p.add_argument("--rounds",  type=int, default=100,
                   help="라운드 수 (기본: 100)")
    p.add_argument("--stack",   type=int, default=10000,
                   help="초기 칩 (기본: 10000)")
    p.add_argument("--games",   type=int, default=50,
                   help="test/train 모드 게임 수 (기본: 50)")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--agents",  type=str, default="rule,rnd,rnd,rnd",
                   help="에이전트 조합 (rule/rnd/rl/nfsp, 쉼표 구분)")
    p.add_argument("--verbose", action="store_true",
                   help="라운드별 상세 출력")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="rl/nfsp 에이전트 체크포인트 디렉토리")
    p.add_argument("--checkpoint-step", type=int, default=None,
                   help="로드할 체크포인트 스텝 번호 (없으면 자동 탐색)")
    p.add_argument("--exploit-games", type=int, default=20,
                   help="착취가능성 측정 게임 수 (기본: 20)")
    p.add_argument("--resume-step", type=int, default=0,
                   help="train3 재개 시 로드할 체크포인트 스텝 (기본: 0=처음부터)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════
#  에이전트 팩토리
# ══════════════════════════════════════════════════════════

def make_agents(agent_spec: str, seed_base: int = 0,
                stack: int = 1000, big_blind: int = 10,
                checkpoint: str = None, checkpoint_step: int = None):
    """
    agent_spec 문자열에서 에이전트 목록을 생성합니다.

    지원 타입:
      rule  — RuleAgent
      rnd   — RandomAgent
      rl    — RLAgent (체크포인트 있으면 로드)
      nfsp  — NFSPAgent (체크포인트 있으면 로드)
    """
    from agents.random_agent import RandomAgent
    from agents.rule_agent   import RuleAgent
    from agents.rl_agent     import RLAgent
    from training.nfsp       import NFSPAgent

    specs  = [s.strip().lower() for s in agent_spec.split(",")]
    agents = []
    rl_idx = nfsp_idx = 0

    for i, spec in enumerate(specs):
        if spec == "rule":
            agents.append(RuleAgent(player_id=i, name=f"Rule{i}", seed=seed_base + i))

        elif spec == "rnd":
            agents.append(RandomAgent(player_id=i, name=f"Rnd{i}", seed=seed_base + i * 7))

        elif spec == "rl":
            agent = RLAgent(
                player_id    = i,
                name         = f"RL{rl_idx}",
                initial_stack= stack,
                big_blind    = big_blind,
                seed         = seed_base + i,
            )
            if checkpoint:
                step = checkpoint_step or _find_latest_step(checkpoint, "actor")
                if step:
                    agent.load(checkpoint, step)
                    print(f"  [RLAgent] 체크포인트 로드: {checkpoint}/step={step}")
            agents.append(agent)
            rl_idx += 1

        elif spec == "nfsp":
            agent = NFSPAgent(
                player_id    = i,
                name         = f"NFSP{nfsp_idx}",
                initial_stack= stack,
                big_blind    = big_blind,
                seed         = seed_base + i,
            )
            if checkpoint:
                step = checkpoint_step or _find_latest_step(checkpoint, "actor")
                if step:
                    agent.load(checkpoint, step)
                    print(f"  [NFSPAgent] 체크포인트 로드: {checkpoint}/step={step}")
            agents.append(agent)
            nfsp_idx += 1

        else:
            print(f"  [경고] 알 수 없는 에이전트 타입 '{spec}' → RandomAgent로 대체")
            agents.append(RandomAgent(player_id=i, name=f"Rnd{i}", seed=seed_base + i))

    return agents


def _find_latest_step(checkpoint_dir: str, prefix: str) -> int:
    """체크포인트 디렉토리에서 가장 최신 스텝 번호 반환"""
    import os, re
    if not os.path.isdir(checkpoint_dir):
        return None
    steps = []
    for f in os.listdir(checkpoint_dir):
        m = re.match(rf"{prefix}_step_(\d+)\.json", f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


# ══════════════════════════════════════════════════════════
#  모드별 실행 함수
# ══════════════════════════════════════════════════════════

# ── single ─────────────────────────────────────────────────

def run_single(args):
    from engine.config   import GameConfig
    from engine.game     import TexasHoldemGame, _config_to_dict
    from utils.logger    import GameLogger
    from utils.reporter  import GameReporter

    config = GameConfig(
        num_players   = len(args.agents.split(",")),
        initial_stack = args.stack,
        small_blind   = 5,
        big_blind     = 10,
        max_rounds    = args.rounds,
        seed          = args.seed,
        verbose       = args.verbose,
    )
    agents      = make_agents(args.agents, args.seed, args.stack, 10,
                               args.checkpoint, args.checkpoint_step)
    agent_names = {a.player_id: a.name for a in agents}
    session_id  = f"single_{datetime.now().strftime('%H%M%S')}"

    logger = GameLogger(
        session_id  = session_id,
        config_dict = _config_to_dict(config),
        agent_names = {str(k): v for k, v in agent_names.items()},
    )

    print(f"[run:single] {args.rounds}라운드  에이전트: {args.agents}  세션: {session_id}")
    result = TexasHoldemGame(agents, config, logger=logger).run()

    reporter    = GameReporter()
    report_path = reporter.from_result(
        game_result = result,
        round_logs  = result["round_logs"],
        config_dict = _config_to_dict(config),
        agent_names = agent_names,
        title       = session_id,
    )

    _print_summary(result, agent_names, config.initial_stack)
    print(f"\n게임 로그  : {logger.json_path}")
    print(f"리포트    : {report_path}")


# ── test ───────────────────────────────────────────────────

def run_test(args):
    from engine.config   import GameConfig
    from engine.game     import TexasHoldemGame, _config_to_dict
    from utils.reporter  import SessionReporter

    agent_spec  = args.agents
    session_rep = SessionReporter()
    num_games   = args.games
    agent_names = {}

    config_dict = {
        "game_type":     "texas_holdem",
        "num_players":   len(agent_spec.split(",")),
        "initial_stack": args.stack,
        "small_blind":   5,
        "big_blind":     10,
        "max_rounds":    args.rounds,
    }

    print(f"[run:test] {num_games}게임 × {args.rounds}라운드  에이전트: {agent_spec}")

    for game_idx in range(num_games):
        config = GameConfig(
            num_players   = config_dict["num_players"],
            initial_stack = args.stack,
            small_blind   = 5,
            big_blind     = 10,
            max_rounds    = args.rounds,
            seed          = args.seed + game_idx,
            verbose       = False,
        )
        agents = make_agents(agent_spec, args.seed + game_idx, args.stack, 10,
                              args.checkpoint, args.checkpoint_step)

        if game_idx == 0:
            agent_names = {a.player_id: a.name for a in agents}

        result = TexasHoldemGame(agents, config).run()
        session_rep.add_game(
            game_idx     = game_idx,
            final_stacks = result["final_stacks"],
            agent_stats  = result["agent_stats"],
            init_stack   = args.stack,
        )

        sys.stdout.write(
            f"\r  진행: {game_idx+1}/{num_games}"
            f"  P0({agent_names.get(0,'?')}) 스택: {result['final_stacks'].get(0, 0):>5}"
        )
        sys.stdout.flush()

    print()

    title       = f"test_{datetime.now().strftime('%H%M%S')}"
    report_path = session_rep.save_report(title, agent_names, config_dict)
    csv_path    = session_rep.save_csv(title)

    print(f"\n비교 리포트 : {report_path}")
    print(f"CSV 데이터  : {csv_path}")


# ── train2 ─────────────────────────────────────────────────

def run_train2(args):
    from training.trainer import Trainer, TrainingConfig

    cfg = TrainingConfig(
        num_games        = args.games,
        rounds_per_game  = args.rounds,
        initial_stack    = args.stack,
        seed             = args.seed,
    )

    print(f"[run:train2] Phase 2 RL 학습")
    print(f"  {cfg.num_games}게임 × {cfg.rounds_per_game}라운드")
    print(f"  ε: {cfg.epsilon_start} → {cfg.epsilon_end}  (감쇠: ×{cfg.epsilon_decay}/game)")
    print(f"  체크포인트: {cfg.checkpoint_dir}/")
    print(f"  로그: {cfg.log_dir}/\n")

    agent = Trainer(cfg).train()

    print(f"\n[학습 완료]")
    print(f"  최종 train_stats 수: {len(agent.train_stats)}")
    if agent.train_stats:
        last = agent.train_stats[-1]
        print(f"  마지막 actor_loss={last['actor_loss']:.4f}  critic_loss={last['critic_loss']:.4f}")


# ── train3 ─────────────────────────────────────────────────

def run_train3(args):
    from training.league_trainer import LeagueTrainer, LeagueTrainingConfig

    cfg = LeagueTrainingConfig(
        num_games       = args.games,
        rounds_per_game = args.rounds,
        initial_stack   = args.stack,
        seed            = args.seed,
        resume_step     = args.resume_step,
    )

    print(f"[run:train3] Phase 3 League Training (NFSP)")
    print(f"  {cfg.num_games}게임 × {cfg.rounds_per_game}라운드")
    print(f"  스냅샷: {cfg.snapshot_every}게임마다")
    print(f"  착취가능성 측정: {cfg.exploit_every}게임마다 ({cfg.exploit_games}게임)")
    print(f"  체크포인트: {cfg.checkpoint_dir}/\n")

    LeagueTrainer(cfg).train()


# ── exploit ────────────────────────────────────────────────

def run_exploit(args):
    from training.nfsp             import NFSPAgent
    from evaluation.exploitability import ExploitabilityMeasurer

    print(f"[run:exploit] 착취가능성 측정")
    print(f"  측정 게임: {args.exploit_games}게임 × {args.rounds}라운드")

    # 에이전트 준비
    agents_to_measure = []
    specs = [s.strip().lower() for s in args.agents.split(",")]

    for i, spec in enumerate(specs):
        if spec == "nfsp":
            agent = NFSPAgent(player_id=0, name=f"NFSP{i}",
                              initial_stack=args.stack, seed=args.seed + i)
            if args.checkpoint:
                step = args.checkpoint_step or _find_latest_step(args.checkpoint, "actor")
                if step:
                    agent.load(args.checkpoint, step)
                    print(f"  NFSP{i}: 체크포인트 step={step} 로드")
            agents_to_measure.append(agent)

    # 에이전트 지정 없으면 신규 NFSPAgent 4개
    if not agents_to_measure:
        for i in range(4):
            agents_to_measure.append(
                NFSPAgent(player_id=0, name=f"NFSP_{i}", initial_stack=args.stack,
                          seed=args.seed + i)
            )

    measurer = ExploitabilityMeasurer(
        games         = args.exploit_games,
        rounds        = args.rounds,
        initial_stack = args.stack,
        seed          = args.seed + 9999,
    )

    print(f"\n{'에이전트':<16} {'착취가능성':>10} {'chip_delta':>12} {'win_rate':>10}")
    print("-" * 52)

    for agent in agents_to_measure:
        result = measurer.measure(agent)
        print(
            f"{agent.name:<16}"
            f"  {result['exploitability']:>+10.1f}"
            f"  {result['agent_chip_delta']:>+12.1f}"
            f"  {result['win_rate']:>9.1%}"
        )

    print()
    print("* 착취가능성(exploit): RuleAgent가 해당 에이전트를 상대로 얻는 평균 chip/game")
    print("  → 낮을수록 착취하기 어려운 전략 (GTO에 가까움)")


# ══════════════════════════════════════════════════════════
#  공통 출력 헬퍼
# ══════════════════════════════════════════════════════════

def _print_summary(result, agent_names, init_stack):
    print("\n" + "=" * 58)
    print(f"{'P':<3} {'이름':<12} {'최종':>7} {'변화':>8} {'수익률':>8}")
    print("-" * 58)
    for pid, stack in sorted(result["final_stacks"].items(),
                              key=lambda x: x[1], reverse=True):
        name  = agent_names.get(pid, f"P{pid}")
        delta = stack - init_stack
        roi   = delta / init_stack * 100
        print(f"P{pid:<2} {name:<12} {stack:>7,} {delta:>+8,} {roi:>+7.1f}%")
    total = sum(result["final_stacks"].values())
    print("=" * 58)
    expected = init_stack * len(result["final_stacks"])
    print(f"총 칩: {total:,}  ({'OK' if total == expected else f'ERR (expected {expected:,})'})")


# ══════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    dispatch = {
        "single":  run_single,
        "test":    run_test,
        "train2":  run_train2,
        "train3":  run_train3,
        "exploit": run_exploit,
    }
    dispatch[args.mode](args)
