"""
run.py — 게임/테스트 실행 진입점

사용법:
  python run.py                          # 기본: RuleAgent 1 vs Random 3, 100라운드
  python run.py --mode test              # 검증 모드: 50게임 × 200라운드 비교 리포트
  python run.py --rounds 500 --verbose   # 라운드 수 지정 + 상세 출력
  python run.py --agents rule,rule,rnd,rnd  # 에이전트 조합 지정
  python run.py --mode single --rounds 50 --seed 42

저장 결과:
  logs/games/     ← 게임별 JSON + JSONL
  logs/reports/   ← 마크다운 리포트
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Poker Agent Runner")
    p.add_argument("--mode",    default="single",
                   choices=["single", "test"],
                   help="single: 단일 게임, test: 다중 게임 비교")
    p.add_argument("--rounds",  type=int,   default=100,
                   help="라운드 수 (기본: 100)")
    p.add_argument("--stack",   type=int,   default=1000,
                   help="초기 스택 (기본: 1000)")
    p.add_argument("--games",   type=int,   default=50,
                   help="test 모드 게임 수 (기본: 50)")
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--agents",  type=str,
                   default="rule,rnd,rnd,rnd",
                   help="에이전트 조합 (쉼표 구분, rule/rnd)")
    p.add_argument("--verbose", action="store_true",
                   help="라운드별 출력")
    return p.parse_args()


def make_agents(agent_spec: str, seed_base: int = 0):
    from agents.random_agent import RandomAgent
    from agents.rule_agent   import RuleAgent

    specs  = [s.strip().lower() for s in agent_spec.split(",")]
    agents = []
    for i, spec in enumerate(specs):
        if spec == "rule":
            agents.append(RuleAgent(player_id=i, name=f"Rule{i}", seed=seed_base+i))
        else:
            agents.append(RandomAgent(player_id=i, name=f"Rnd{i}", seed=seed_base+i*7))
    return agents


# ══════════════════════════════════════════════════════════
#  단일 게임 모드
# ══════════════════════════════════════════════════════════

def run_single(args):
    from engine.config import GameConfig
    from engine.game   import TexasHoldemGame, _config_to_dict
    from utils.logger  import GameLogger
    from utils.reporter import GameReporter

    config = GameConfig(
        num_players   = len(args.agents.split(",")),
        initial_stack = args.stack,
        small_blind   = 5,
        big_blind     = 10,
        max_rounds    = args.rounds,
        seed          = args.seed,
        verbose       = args.verbose,
    )
    agents     = make_agents(args.agents, args.seed)
    agent_names = {a.player_id: a.name for a in agents}
    session_id  = f"single_{datetime.now().strftime('%H%M%S')}"

    logger = GameLogger(
        session_id  = session_id,
        config_dict = _config_to_dict(config),
        agent_names = {str(k): v for k, v in agent_names.items()},
    )

    print(f"[run] 단일 게임 시작 — {args.rounds}라운드  세션: {session_id}")
    result = TexasHoldemGame(agents, config, logger=logger).run()

    # 리포트 생성
    reporter  = GameReporter()
    report_path = reporter.from_result(
        game_result = result,
        round_logs  = result["round_logs"],
        config_dict = _config_to_dict(config),
        agent_names = agent_names,
        title       = session_id,
    )

    _print_summary(result, agent_names, config.initial_stack)
    print(f"\n📄 게임 로그  : {logger.json_path}")
    print(f"📊 리포트    : {report_path}")


# ══════════════════════════════════════════════════════════
#  테스트(다중 게임 비교) 모드
# ══════════════════════════════════════════════════════════

def run_test(args):
    from engine.config  import GameConfig
    from engine.game    import TexasHoldemGame, _config_to_dict
    from utils.reporter import SessionReporter

    agent_spec  = args.agents
    agent_names = {}
    session_rep = SessionReporter()
    num_games   = args.games

    config_dict = {
        "game_type":     "texas_holdem",
        "num_players":   len(agent_spec.split(",")),
        "initial_stack": args.stack,
        "small_blind":   5,
        "big_blind":     10,
        "max_rounds":    args.rounds,
    }

    print(f"[run] 비교 테스트 — {num_games}게임 × {args.rounds}라운드")
    print(f"      에이전트: {agent_spec}")

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
        agents = make_agents(agent_spec, args.seed + game_idx)

        if game_idx == 0:
            agent_names = {a.player_id: a.name for a in agents}

        result = TexasHoldemGame(agents, config).run()

        session_rep.add_game(
            game_idx     = game_idx,
            final_stacks = result["final_stacks"],
            agent_stats  = result["agent_stats"],
            init_stack   = args.stack,
        )

        # 진행 표시
        sys.stdout.write(
            f"\r  진행: {game_idx+1}/{num_games}  "
            f"P0 스택: {result['final_stacks'].get(0, 0):>5}"
        )
        sys.stdout.flush()

    print()

    title       = f"test_{datetime.now().strftime('%H%M%S')}"
    report_path = session_rep.save_report(title, agent_names, config_dict)
    csv_path    = session_rep.save_csv(title)

    print(f"\n📊 비교 리포트 : {report_path}")
    print(f"📈 CSV 데이터  : {csv_path}")


# ══════════════════════════════════════════════════════════
#  공통 출력
# ══════════════════════════════════════════════════════════

def _print_summary(result, agent_names, init_stack):
    print("\n" + "=" * 55)
    print(f"{'Player':<8} {'이름':<10} {'최종':>7} {'변화':>7} {'수익률':>7}")
    print("-" * 55)
    for pid, stack in sorted(result["final_stacks"].items(),
                              key=lambda x: x[1], reverse=True):
        name  = agent_names.get(pid, f"P{pid}")
        delta = stack - init_stack
        roi   = delta / init_stack * 100
        print(f"P{pid:<7} {name:<10} {stack:>7,} {delta:>+7,} {roi:>+6.1f}%")
    total = sum(result["final_stacks"].values())
    print("=" * 55)
    print(f"총 칩: {total:,}  ({'OK' if total == init_stack * len(result['final_stacks']) else 'ERR'})")


# ══════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "single":
        run_single(args)
    elif args.mode == "test":
        run_test(args)
