"""
logger.py — 게임 세션 로거

게임 진행 중 발생하는 모든 이벤트를 구조화된 JSON으로 저장합니다.

저장 경로:
  logs/games/YYYYMMDD_HHMMSS_{session_id}.json   ← 전체 게임 로그
  logs/games/YYYYMMDD_HHMMSS_{session_id}.jsonl  ← 라운드별 스트리밍 로그

JSON 구조:
  {
    "meta": { session_id, start_time, config, agents },
    "rounds": [ { round_num, winners, pot, stacks, actions, community_cards, ... } ],
    "summary": { total_rounds, final_stacks, agent_stats }
  }
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_DIR = Path(__file__).parent.parent / "logs"


class GameLogger:
    """
    게임 세션 전체를 로깅합니다.

    사용법:
        logger = GameLogger(session_id="phase1_test", config=config)
        logger.log_round(round_log)       # 매 라운드 호출
        logger.finalize(game_result)      # 게임 종료 시 호출
    """

    def __init__(
        self,
        session_id:  str,
        config_dict: Dict[str, Any],
        agent_names: Dict[int, str],
        log_dir:     Optional[Path] = None,
    ):
        self.session_id  = session_id
        self._start_time = time.time()
        self._start_dt   = datetime.now()

        self._log_dir = Path(log_dir) if log_dir else LOG_DIR / "games"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # 파일명: YYYYMMDD_HHMMSS_session_id
        ts       = self._start_dt.strftime("%Y%m%d_%H%M%S")
        stem     = f"{ts}_{session_id}"
        self._json_path  = self._log_dir / f"{stem}.json"
        self._jsonl_path = self._log_dir / f"{stem}.jsonl"

        # JSONL: 라운드별 스트리밍 저장 (큰 게임에서도 메모리 효율적)
        self._jsonl_file = open(self._jsonl_path, "w", encoding="utf-8")

        # 메타 데이터
        self._meta = {
            "session_id":  session_id,
            "start_time":  self._start_dt.isoformat(),
            "config":      config_dict,
            "agents":      agent_names,
        }
        self._rounds:    List[Dict] = []
        self._finalized: bool       = False

        # 메타 정보를 JSONL 첫 줄에 기록
        self._write_jsonl({"type": "meta", **self._meta})

    # ══════════════════════════════════════════════════════
    #  라운드 로깅
    # ══════════════════════════════════════════════════════

    def log_round(self, round_log: Dict[str, Any]) -> None:
        """매 라운드 종료 시 호출합니다."""
        enriched = {
            "type":       "round",
            "elapsed_s":  round(time.time() - self._start_time, 2),
            **round_log,
        }
        self._rounds.append(enriched)
        self._write_jsonl(enriched)

    # ══════════════════════════════════════════════════════
    #  게임 종료
    # ══════════════════════════════════════════════════════

    def finalize(self, game_result: Dict[str, Any]) -> Path:
        """
        게임 종료 시 호출합니다.
        전체 로그를 JSON 파일로 저장하고 경로를 반환합니다.
        """
        if self._finalized:
            return self._json_path

        end_dt    = datetime.now()
        elapsed   = time.time() - self._start_time

        summary = {
            "end_time":      end_dt.isoformat(),
            "elapsed_s":     round(elapsed, 2),
            "rounds_played": game_result.get("rounds_played", len(self._rounds)),
            "final_stacks":  game_result.get("final_stacks", {}),
            "agent_stats":   game_result.get("agent_stats", {}),
        }

        full_log = {
            "meta":    self._meta,
            "rounds":  self._rounds,
            "summary": summary,
        }

        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(full_log, f, ensure_ascii=False, indent=2, default=str)

        self._write_jsonl({"type": "summary", **summary})
        self._jsonl_file.close()
        self._finalized = True

        return self._json_path

    # ══════════════════════════════════════════════════════
    #  내부 유틸리티
    # ══════════════════════════════════════════════════════

    def _write_jsonl(self, data: Dict) -> None:
        line = json.dumps(data, ensure_ascii=False, default=str)
        self._jsonl_file.write(line + "\n")
        self._jsonl_file.flush()

    @property
    def json_path(self) -> Path:
        return self._json_path

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    def __del__(self):
        if not self._finalized and not self._jsonl_file.closed:
            self._jsonl_file.close()


# ══════════════════════════════════════════════════════════
#  학습 진행 로거
# ══════════════════════════════════════════════════════════

class TrainingLogger:
    """
    Phase 2+ RL 학습 진행 상황을 CSV로 저장합니다.

    저장 경로: logs/training/{session_id}/progress.csv

    열 구성:
      episode, avg_reward, win_rate, fold_rate, raise_rate,
      actor_loss, critic_loss, entropy, elapsed_s
    """

    HEADER = [
        "episode", "avg_reward", "win_rate",
        "fold_rate", "call_rate", "raise_rate",
        "actor_loss", "critic_loss", "entropy",
        "exploitability", "elapsed_s",
    ]

    def __init__(self, session_id: str, log_dir: Optional[Path] = None):
        self.session_id  = session_id
        self._start_time = time.time()

        self._log_dir = Path(log_dir) if log_dir else LOG_DIR / "training" / session_id
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self._log_dir / "progress.csv"

        # 헤더 작성 (파일이 없을 때만)
        if not self._csv_path.exists():
            with open(self._csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(self.HEADER) + "\n")

        self._csv_file = open(self._csv_path, "a", encoding="utf-8")

    def log_episode(
        self,
        episode:        int,
        avg_reward:     float,
        win_rate:       float,
        fold_rate:      float       = 0.0,
        call_rate:      float       = 0.0,
        raise_rate:     float       = 0.0,
        actor_loss:     float       = 0.0,
        critic_loss:    float       = 0.0,
        entropy:        float       = 0.0,
        exploitability: float       = 0.0,
    ) -> None:
        elapsed = round(time.time() - self._start_time, 2)
        row = [
            episode,
            round(avg_reward,     4),
            round(win_rate,       4),
            round(fold_rate,      4),
            round(call_rate,      4),
            round(raise_rate,     4),
            round(actor_loss,     6),
            round(critic_loss,    6),
            round(entropy,        6),
            round(exploitability, 6),
            elapsed,
        ]
        self._csv_file.write(",".join(map(str, row)) + "\n")
        self._csv_file.flush()

    def save_checkpoint(self, episode: int, data: Dict[str, Any]) -> Path:
        """에이전트 체크포인트를 JSON으로 저장합니다."""
        path = self._log_dir / f"checkpoint_ep{episode:06d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return path

    def close(self) -> None:
        if not self._csv_file.closed:
            self._csv_file.close()

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def __del__(self):
        self.close()
