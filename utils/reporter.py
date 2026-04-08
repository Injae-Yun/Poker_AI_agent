"""
reporter.py — 게임/학습 결과 리포트 생성기

GameReporter : 단일 게임 결과를 마크다운 리포트로 저장
SessionReporter : 여러 게임(세션)을 종합 비교 리포트로 저장

저장 경로: logs/reports/YYYYMMDD_HHMMSS_{title}.md
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_DIR = Path(__file__).parent.parent / "logs"


class GameReporter:
    """
    단일 게임 JSON 로그를 읽어 마크다운 리포트를 생성합니다.

    포함 내용:
      - 게임 설정 요약
      - 라운드별 진행 개요 (팟, 승자, 스택 변화)
      - 에이전트별 통계 (칩 변화, 폴드/콜/레이즈 비율)
      - 핵심 핸드 하이라이트 (팟이 가장 큰 상위 5핸드)
      - 스트리트별 액션 분포
    """

    def __init__(self, report_dir: Optional[Path] = None):
        self._report_dir = Path(report_dir) if report_dir else LOG_DIR / "reports"
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def from_json(self, json_path: Path) -> Path:
        """JSON 로그 파일에서 리포트를 생성합니다."""
        with open(json_path, encoding="utf-8") as f:
            log = json.load(f)
        return self._generate(log, json_path.stem)

    def from_result(
        self,
        game_result: Dict[str, Any],
        round_logs:  List[Dict[str, Any]],
        config_dict: Dict[str, Any],
        agent_names: Dict[int, str],
        title:       str = "game",
    ) -> Path:
        """game.run() 결과 딕셔너리에서 직접 리포트를 생성합니다."""
        log = {
            "meta": {
                "session_id": title,
                "start_time": datetime.now().isoformat(),
                "config":     config_dict,
                "agents":     {str(k): v for k, v in agent_names.items()},
            },
            "rounds":  round_logs,
            "summary": {
                "rounds_played": game_result["rounds_played"],
                "final_stacks":  {str(k): v for k, v in game_result["final_stacks"].items()},
                "agent_stats":   {str(k): v for k, v in game_result["agent_stats"].items()},
            },
        }
        return self._generate(log, title)

    # ══════════════════════════════════════════════════════
    #  마크다운 생성
    # ══════════════════════════════════════════════════════

    def _generate(self, log: Dict, title: str) -> Path:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = self._report_dir / f"{ts}_{title}.md"

        meta    = log.get("meta", {})
        rounds  = log.get("rounds", [])
        summary = log.get("summary", {})
        config  = meta.get("config", {})
        agents  = meta.get("agents", {})

        lines = []
        _h  = lambda t, n=1: lines.append(f"{'#'*n} {t}")
        _p  = lambda t="":    lines.append(t)
        _hr = lambda:         lines.append("---")

        # ── 헤더 ───────────────────────────────────────────
        _h(f"포커 게임 리포트 — {meta.get('session_id', title)}")
        _p(f"> 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        _p(f"> 게임 시작: {meta.get('start_time', 'N/A')[:19]}")
        _hr()

        # ── 게임 설정 ──────────────────────────────────────
        _h("1. 게임 설정", 2)
        _p(f"| 항목 | 값 |")
        _p(f"|------|-----|")
        _p(f"| 게임 타입 | {config.get('game_type', 'texas_holdem')} |")
        _p(f"| 플레이어 수 | {config.get('num_players', 4)} |")
        _p(f"| 초기 스택 | {config.get('initial_stack', 0):,} chips |")
        _p(f"| 블라인드 | SB {config.get('small_blind', 5)} / BB {config.get('big_blind', 10)} |")
        _p(f"| 총 라운드 | {summary.get('rounds_played', len(rounds))} |")
        _p(f"| 경과 시간 | {summary.get('elapsed_s', 'N/A')} 초 |")
        _p()

        # ── 에이전트 정보 ──────────────────────────────────
        _h("2. 에이전트 등록", 2)
        _p("| Player | 이름 | 타입 |")
        _p("|--------|------|------|")
        for pid, name in sorted(agents.items(), key=lambda x: int(x[0])):
            agent_type = "RuleAgent" if "Rule" in name else ("RandomAgent" if "Rnd" in name or "Random" in name else "Agent")
            _p(f"| P{pid} | {name} | {agent_type} |")
        _p()

        # ── 최종 결과 ──────────────────────────────────────
        _h("3. 최종 결과", 2)
        final_stacks  = {int(k): v for k, v in summary.get("final_stacks", {}).items()}
        agent_stats   = {int(k): v for k, v in summary.get("agent_stats", {}).items()}
        init_stack    = config.get("initial_stack", 1000)
        total_chips   = sum(final_stacks.values())

        _p("| Player | 이름 | 최종 스택 | 칩 변화 | 수익률 |")
        _p("|--------|------|----------|---------|-------|")
        sorted_pids = sorted(final_stacks.items(), key=lambda x: x[1], reverse=True)
        for rank, (pid, stack) in enumerate(sorted_pids, 1):
            name   = agents.get(str(pid), f"P{pid}")
            delta  = stack - init_stack
            roi    = delta / init_stack * 100
            medal  = ["🥇","🥈","🥉",""][min(rank-1, 3)]
            _p(f"| {medal} P{pid} | {name} | {stack:,} | {delta:+,} | {roi:+.1f}% |")
        _p()
        _p(f"**총 칩 합계**: {total_chips:,}  "
           f"({'✅ 보존됨' if total_chips == init_stack * config.get('num_players', 4) else '⚠️ 불일치'})")
        _p()

        # ── 에이전트 통계 ──────────────────────────────────
        _h("4. 에이전트 전략 통계", 2)
        _p("| Player | 이름 | 핸드 수 | 폴드율 | 콜율 | 레이즈율 | 평균 수익/핸드 |")
        _p("|--------|------|--------|-------|------|---------|-------------|")
        for pid in sorted(agent_stats.keys()):
            st    = agent_stats[pid]
            name  = agents.get(str(pid), f"P{pid}")
            hands = st.get("hands_played", 0) or 1
            avg_r = st.get("avg_reward", 0)
            _p(f"| P{pid} | {name} | {st.get('hands_played',0)} | "
               f"{st.get('fold_rate',0):.1%} | "
               f"{st.get('call_rate',0):.1%} | "
               f"{st.get('raise_rate',0):.1%} | "
               f"{avg_r:+.1f} |")
        _p()

        # ── 스트리트별 액션 분포 ───────────────────────────
        _h("5. 스트리트별 액션 분포 (전체)", 2)
        street_actions = self._aggregate_street_actions(rounds)
        _p("| 스트리트 | 폴드 | 체크 | 콜 | 레이즈 | 합계 |")
        _p("|---------|------|------|-----|-------|------|")
        for street in ["preflop", "flop", "turn", "river"]:
            d     = street_actions.get(street, {})
            total = sum(d.values()) or 1
            fold  = d.get("fold",  0)
            check = d.get("check", 0)
            call  = d.get("call",  0)
            raise_= d.get("raise", 0)
            blind = d.get("blind", 0)
            act_t = total - blind
            _p(f"| {street} | {fold}({fold/act_t:.0%}) | {check}({check/act_t:.0%}) | "
               f"{call}({call/act_t:.0%}) | {raise_}({raise_/act_t:.0%}) | {act_t} |")
        _p()

        # ── 하이라이트 핸드 (팟 상위 5) ──────────────────
        _h("6. 하이라이트 핸드 (팟 기준 상위 5)", 2)
        top_rounds = sorted(
            [r for r in rounds if isinstance(r, dict) and r.get("pot", 0) > 0],
            key=lambda r: r.get("pot", 0),
            reverse=True
        )[:5]

        for i, rnd in enumerate(top_rounds, 1):
            winners     = rnd.get("winners", [])
            pot         = rnd.get("pot", 0)
            community   = " ".join(rnd.get("community_cards", [])) or "(없음)"
            winner_str  = ", ".join(f"P{w}" for w in winners)
            _p(f"**#{i} Round {rnd.get('round_num', '?')}** — "
               f"팟: {pot:,} chips  |  승자: {winner_str}  |  "
               f"커뮤니티: `{community}`")

            # 핸드 공개 정보
            ph = rnd.get("player_hands", {})
            if ph:
                for pid, cards in ph.items():
                    name = agents.get(str(pid), f"P{pid}")
                    _p(f"  - {name}: `{' '.join(cards)}`")

            # 해당 라운드 주요 액션
            history = rnd.get("betting_history", [])
            key_acts = [a for a in history if isinstance(a, dict) and a.get("action") in ("raise",)]
            if key_acts:
                for a in key_acts[:3]:
                    name = agents.get(str(a["player_id"]), f"P{a['player_id']}")
                    _p(f"  - {name} {a['street']} RAISE {a['amount']}")
            _p()

        # ── 라운드별 스택 추이 ─────────────────────────────
        _h("7. 라운드별 스택 추이 (10라운드 간격)", 2)
        if rounds:
            pids = sorted(final_stacks.keys())
            header_cols = " | ".join(f"P{p}" for p in pids)
            _p(f"| 라운드 | {header_cols} |")
            _p(f"|--------|{'|'.join(['------']*len(pids))}|")

            step = max(1, len(rounds) // 20)
            sampled = rounds[::step]
            if rounds[-1] not in sampled:
                sampled.append(rounds[-1])

            for rnd in sampled:
                if not isinstance(rnd, dict):
                    continue
                stacks = rnd.get("stacks", {})
                rnum   = rnd.get("round_num", "?")
                cols   = " | ".join(
                    f"{stacks.get(str(p), stacks.get(p, 0)):,}" for p in pids
                )
                _p(f"| {rnum} | {cols} |")
        _p()

        # ── 푸터 ──────────────────────────────────────────
        _hr()
        _p(f"*자동 생성 — Texas Hold'em Poker Agent*")

        # 파일 저장
        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return outpath

    # ── 집계 유틸리티 ─────────────────────────────────────
    @staticmethod
    def _aggregate_street_actions(rounds: List[Dict]) -> Dict[str, Dict[str, int]]:
        from collections import defaultdict
        result = defaultdict(lambda: defaultdict(int))
        for rnd in rounds:
            if not isinstance(rnd, dict):
                continue
            for act in rnd.get("betting_history", []):
                if isinstance(act, dict):
                    street = act.get("street", "unknown")
                    action = act.get("action", "unknown")
                    result[street][action] += 1
        return result


# ══════════════════════════════════════════════════════════
#  세션(다중 게임) 비교 리포트
# ══════════════════════════════════════════════════════════

class SessionReporter:
    """
    여러 게임 결과를 종합하여 에이전트 비교 리포트를 생성합니다.
    주로 Phase 1 검증처럼 N 게임 × M 라운드 실험에 사용합니다.
    """

    def __init__(self, report_dir: Optional[Path] = None):
        self._report_dir = Path(report_dir) if report_dir else LOG_DIR / "reports"
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict] = []

    def add_game(
        self,
        game_idx:    int,
        final_stacks: Dict[int, int],
        agent_stats:  Dict[int, Dict],
        init_stack:   int,
    ) -> None:
        """게임 한 판의 결과를 기록합니다."""
        self._records.append({
            "game_idx":    game_idx,
            "final_stacks": final_stacks,
            "agent_stats":  agent_stats,
            "init_stack":   init_stack,
        })

    def save_report(
        self,
        title:       str,
        agent_names: Dict[int, str],
        config_dict: Dict[str, Any],
    ) -> Path:
        """종합 비교 리포트를 마크다운으로 저장합니다."""
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = self._report_dir / f"{ts}_{title}.md"

        lines = []
        _h  = lambda t, n=1: lines.append(f"{'#'*n} {t}")
        _p  = lambda t="":    lines.append(t)
        _hr = lambda:         lines.append("---")

        num_games  = len(self._records)
        init_stack = config_dict.get("initial_stack", 1000)
        num_rounds = config_dict.get("max_rounds", 0)
        pids       = sorted(agent_names.keys())

        _h(f"에이전트 비교 리포트 — {title}")
        _p(f"> 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        _p(f"> 실험 규모: {num_games}게임 × {num_rounds}라운드  |  초기 스택: {init_stack:,}")
        _hr()

        # ── 에이전트 등록 ──────────────────────────────────
        _h("1. 에이전트 구성", 2)
        for pid in pids:
            _p(f"- **P{pid}**: {agent_names[pid]}")
        _p()

        # ── 평균 칩 변화 ───────────────────────────────────
        _h("2. 칩 변화 통계", 2)

        deltas  = {pid: [] for pid in pids}
        winners = {pid: 0  for pid in pids}

        for rec in self._records:
            fs = rec["final_stacks"]
            max_stack = max(fs.values())
            for pid in pids:
                delta = fs.get(pid, 0) - init_stack
                deltas[pid].append(delta)
            for pid, stack in fs.items():
                if stack == max_stack:
                    winners[int(pid)] = winners.get(int(pid), 0) + 1
                    break

        _p("| Player | 이름 | 평균 칩 변화 | 최대 | 최소 | 수익 게임 수 | 최다 보유율 |")
        _p("|--------|------|------------|------|------|-----------|-----------|")
        for pid in pids:
            d    = deltas[pid]
            avg  = sum(d) / len(d) if d else 0
            mx   = max(d) if d else 0
            mn   = min(d) if d else 0
            pos  = sum(1 for x in d if x > 0)
            name = agent_names[pid]
            win_rate = winners.get(pid, 0) / num_games
            _p(f"| P{pid} | {name} | {avg:+,.1f} | {mx:+,} | {mn:+,} | "
               f"{pos}/{num_games} ({pos/num_games:.0%}) | {win_rate:.1%} |")
        _p()

        # ── 전략 통계 평균 ─────────────────────────────────
        _h("3. 전략 통계 (전체 게임 평균)", 2)
        _p("| Player | 이름 | 폴드율 | 콜율 | 레이즈율 | 평균 수익/핸드 |")
        _p("|--------|------|-------|------|---------|-------------|")
        for pid in pids:
            fold_rates  = []
            call_rates  = []
            raise_rates = []
            avg_rewards = []
            for rec in self._records:
                st = rec["agent_stats"].get(pid, rec["agent_stats"].get(str(pid), {}))
                if st:
                    fold_rates.append(st.get("fold_rate", 0))
                    call_rates.append(st.get("call_rate", 0))
                    raise_rates.append(st.get("raise_rate", 0))
                    avg_rewards.append(st.get("avg_reward", 0))
            avg_f = sum(fold_rates)  / len(fold_rates)  if fold_rates  else 0
            avg_c = sum(call_rates)  / len(call_rates)  if call_rates  else 0
            avg_r = sum(raise_rates) / len(raise_rates) if raise_rates else 0
            avg_rew = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
            _p(f"| P{pid} | {agent_names[pid]} | {avg_f:.1%} | {avg_c:.1%} | "
               f"{avg_r:.1%} | {avg_rew:+.2f} |")
        _p()

        # ── 결론 ──────────────────────────────────────────
        _h("4. 결론", 2)
        best_pid = max(pids, key=lambda p: sum(deltas[p]) / len(deltas[p]) if deltas[p] else 0)
        best_avg = sum(deltas[best_pid]) / len(deltas[best_pid])
        _p(f"- **최고 성능 에이전트**: P{best_pid} ({agent_names[best_pid]})  "
           f"— 평균 칩 변화 **{best_avg:+,.1f}**")

        # 비교쌍 차이
        if len(pids) >= 2:
            for i, pid_a in enumerate(pids):
                for pid_b in pids[i+1:]:
                    avg_a = sum(deltas[pid_a]) / len(deltas[pid_a]) if deltas[pid_a] else 0
                    avg_b = sum(deltas[pid_b]) / len(deltas[pid_b]) if deltas[pid_b] else 0
                    diff  = avg_a - avg_b
                    _p(f"- P{pid_a} vs P{pid_b}: 평균 차이 **{diff:+,.1f}** chips")
        _p()

        _hr()
        _p(f"*자동 생성 — Texas Hold'em Poker Agent*")

        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return outpath

    def save_csv(self, title: str) -> Path:
        """원시 데이터를 CSV로도 저장합니다."""
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = self._report_dir / f"{ts}_{title}.csv"
        if not self._records:
            return outpath

        pids = sorted(self._records[0]["final_stacks"].keys())
        init = self._records[0]["init_stack"]

        with open(outpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["game_idx"] + [f"P{p}_delta" for p in pids] + [f"P{p}_fold" for p in pids]
            writer.writerow(header)
            for rec in self._records:
                row  = [rec["game_idx"]]
                row += [rec["final_stacks"].get(p, 0) - init for p in pids]
                row += [
                    round(rec["agent_stats"].get(p, rec["agent_stats"].get(str(p), {})).get("fold_rate", 0), 3)
                    for p in pids
                ]
                writer.writerow(row)
        return outpath
