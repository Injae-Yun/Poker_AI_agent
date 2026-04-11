# Texas Hold'em Poker Agent

> 텍사스 홀덤 포커에서 높은 승률을 목표로 하는 멀티에이전트 AI 시스템

---

## 학습 현황 리포트 (2026-04-09 기준)

### 실험 개요

| 항목 | 내용 |
|------|------|
| 학습 알고리즘 | NFSP (Neural Fictitious Self-Play) + A2C |
| 모델 구조 | PokerNet: MLP(104→256→128) + GRU(hidden=64, 2-layer), 7 actions |
| 학습 설정 | League Training 4×NFSPAgent + 2×RuleAgent exploiter + snapshot pool |
| 누적 학습량 | v2: 400 games × 200 rounds = 약 80,000 핸드 (v1: 500 games) |
| 초기 스택 | 10,000 칩 (BB=10, SB=5) |
| 하드웨어 | RTX 5060 Ti (CUDA 12.8 nightly) |

### 기준선 성능 (비학습 에이전트)

> 실험: 50게임 × 100라운드, 초기 스택 10,000

| 에이전트 | 평균 칩 변화 | 수익 게임율 | 1위 비율 | 레이즈율 |
|---------|------------|-----------|---------|---------|
| RuleAgent | **+10,000** | 50% | 50% | 36.3% |
| RandomAgent | -3,000 ~ -5,800 | 12~24% | 12~24% | ~28% |

### NFSP 학습 결과 비교

#### v1 (step 300, 하이퍼파라미터 미조정)
| 에이전트 | 평균 칩 변화 | 1위 비율 | 레이즈율 | ASP fold율 |
|---------|------------|---------|---------|----------|
| **NFSP v1** | -7,333 | 6.7% | **74.8%** | 8.8% |
| RuleAgent | +12,667 | 56.7% | 23% | — |

#### v2 (step 400, Phase 4-0 적용) — Phase 4-0 수정 사항
- Returns 정규화 + Advantage 클리핑(±5)
- Critic 손실: MSE → Huber, 가중치 0.5→0.25
- entropy_coef 0.01~0.03 → 0.003~0.005
- 레이즈율 50% 초과 시 보상 패널티

| 에이전트 | 평균 칩 변화 (vs 1Rule+2Rnd) | 1위 비율 | ASP fold율 | ASP all-in율 |
|---------|--------------------------|---------|----------|------------|
| **NFSP v2** | **-3,333** | **16.7%** | **25.7%** | **8.2%** |
| NFSP v1 | -6,000 | 10.0% | 8.8% | 4.7% |

> 3×RuleAgent 상대로는 v1과 동일(-7,333, 6.7%) — 3명의 강한 상대 대비 아직 부족

### 착취가능성 (Exploitability) 추이

> 낮을수록 GTO에 가까운 전략. RuleAgent 3명이 대상 에이전트를 상대로 얻는 평균 칩/게임.

**v1 (하이퍼파라미터 미조정):**

| game | NFSP_Tight | NFSP_Loose | NFSP_Balanced | NFSP_Aggr |
|------|-----------|-----------|--------------|---------|
| 30 | +1,556 | +1,556 | +1,556 | +1,556 |
| 150 | +2,444 | +2,444 | +1,556 | +3,333 |

**v2 (Phase 4-0 수정 후):**

| game | NFSP_Tight | NFSP_Loose | NFSP_Balanced | NFSP_Aggr |
|------|-----------|-----------|--------------|---------|
| 30 | +2,444 | +1,556 | +1,556 | **+142** |
| 90 | **+351** | +1,556 | **+148** | **+144** |
| 120 | +3,333 | +1,556 | **-1,111** | +473 |

- v2 초반(game 30)부터 일부 에이전트 착취가능성 급감 (Aggr: 1556→142)
- game 90에서 세 에이전트 1000 이하 동시 달성
- game 120 Balanced **-1,111**: RuleAgent 상대로 평균 수익 달성

### 분석

**개선된 점:**
- ASP 정책 분포 정상화 (fold 8.8%→25.7%, all-in 4.7%→8.2%)
- 착취가능성 초기 수렴 속도 크게 향상
- 1Rule+2Rnd 상대 win_rate: 6.7%→16.7%

**남은 과제 (Phase 4-1 이후):**
- BR 네트워크 여전히 uniform 분포 (71.9% 레이즈) — 더 많은 학습 필요
- OpponentModel 통합: 상대 배팅 패턴 → 핸드 레인지 추론으로 GTO/Exploit 전환
- 착취가능성 후반 불안정 해소를 위한 추가 학습

### 체크포인트 위치

```
checkpoints/league/
├── best_net_P{0-3}.pth      # 각 에이전트 BR 네트워크 best
├── best_asp_P{0-3}.pth      # 각 에이전트 ASP 네트워크 best
├── net/asp_P{i}_step_N.pth  # 스텝별 체크포인트 (5~400, v2)
└── (league_v1/)             # v1 백업 (step 30~500)
```

---

## 프로젝트 개요

카드 카운팅, 덱 추적, 상대 배팅 패턴 분석, 핸드 에퀴티 기반 의사결정을 결합한 포커 AI 에이전트를 구현합니다.
현재 구현 대상은 **텍사스 홀덤 4인 플레이**이며, 이후 다양한 포커 변형 규칙으로 확장 가능한 구조로 설계됩니다.

---

## 주요 특징

- **독립 에이전트**: 4개의 플레이어는 서로 정보를 공유하지 않는 독립 에이전트로 구현
- **Advantage Function 기반 보상**: 단순 칩 델타가 아닌, 결정 품질(EV 대비 실제 선택)을 보상으로 사용
- **전략 다양성 보장**: League Training + NFSP 구조로 전략 동질화 방지
- **상대 모델링**: 배팅 패턴 추적(VPIP, PFR, AF)을 통한 실시간 상대 프로파일링
- **옵션 선택 구조**: 게임 타입, 플레이어 수, 블라인드 구조 등을 설정 가능

---

## 기술 스택

| 구성 요소 | 선택 기술 |
|-----------|-----------|
| 게임 엔진 | 자체 구현 (Pure Python + numpy) |
| 강화학습 | PyTorch Actor-Critic (A2C), MLP+GRU 듀얼 브랜치 |
| 전략 다양성 | NFSP + League Training |
| 에퀴티 계산 | Monte Carlo Simulation |
| 언어 | Python 3.10+ / PyTorch |

---

## 프로젝트 구조

```
poker_agent/
├── README.md
├── PLAN.md                    # 개발 계획
├── requirements.txt
│
├── engine/                    # 게임 엔진 래퍼
│   ├── game.py                # 게임 루프 및 설정
│   ├── state.py               # 게임 상태 표현
│   └── config.py              # 게임 옵션 설정
│
├── agents/                    # 에이전트 구현
│   ├── base_agent.py          # 에이전트 추상 클래스
│   ├── random_agent.py        # 랜덤 에이전트 (베이스라인)
│   ├── rule_agent.py          # 룰 기반 에이전트 (Phase 1)
│   └── rl_agent.py            # PyTorch Actor-Critic 에이전트 (Phase 2.5~)
│
├── models/                    # 신경망 모델
│   ├── poker_net.py           # PokerNet: MLP+GRU 듀얼 브랜치 (Phase 2.5)
│   ├── seq_encoder.py         # 베팅 히스토리 → 시퀀스 텐서 (GRU 입력)
│   ├── actor.py               # (레거시) numpy Actor 네트워크
│   ├── critic.py              # (레거시) numpy Critic 네트워크
│   └── nn.py                  # (레거시) numpy 자체 구현 레이어
│
├── reward/                    # 보상 설계
│   ├── ev_calculator.py       # EV 계산 (팟 오즈, 에퀴티)
│   └── reward_shaper.py       # Advantage Function 기반 보상
│
├── training/                  # 학습 파이프라인
│   ├── trainer.py             # Phase 2 학습 루프
│   ├── league.py              # League Training 관리 (Phase 2.5 PyTorch)
│   ├── league_trainer.py      # Phase 3 리그 학습 루프
│   └── nfsp.py                # NFSP 에이전트 (Phase 2.5 PyTorch)
│
├── evaluation/                # 평가 및 분석
│   ├── evaluator.py           # 에이전트 성능 평가
│   └── exploitability.py      # 착취가능성 측정
│
└── utils/
    ├── hand_evaluator.py      # 핸드 강도 평가
    ├── deck_tracker.py        # 덱/카드 추적
    ├── state_encoder.py       # AgentObservation → 104차원 상태 벡터 (Phase 2.5)
    ├── opponent_profiler.py   # VPIP/PFR/AF 상대 통계
    └── stats.py               # 통계 집계
```

---

## 빠른 시작

```bash
# 단일 게임 실행 (RuleAgent 1 vs Random 3)
python run.py --mode single --rounds 100

# 비교 테스트 (50게임 × 200라운드)
python run.py --mode test --games 50 --rounds 200

# Phase 2 — RL 에이전트 학습
python -c "
from training.trainer import Trainer, TrainingConfig
Trainer(TrainingConfig(num_games=200, rounds_per_game=200)).train()
"

# Phase 3 — League Training (NFSP)
python -c "
from training.league_trainer import LeagueTrainer, LeagueTrainingConfig
LeagueTrainer(LeagueTrainingConfig(num_games=300, rounds_per_game=200)).train()
"
```

---

## 게임 옵션

```python
GameConfig(
    game_type   = "texas_holdem",   # 게임 타입 (향후 확장 예정)
    num_players = 4,                # 플레이어 수 (2~9)
    initial_stack = 1000,           # 초기 칩
    small_blind   = 5,
    big_blind     = 10,
    max_rounds    = 100,
)
```

---

## 핵심 설계 원칙

**1. 정보 은닉 (Information Hiding)**
각 에이전트는 자신의 홀 카드, 커뮤니티 카드, 배팅 히스토리만 관찰합니다. 상대의 홀 카드는 접근 불가합니다.

**2. Advantage Function 보상**
폴드 포함 모든 액션의 보상은 `A(s,a) = Q(s,a) - V(s)` 로 계산합니다. 매몰 비용을 제외한 순수 미래 기댓값 기반으로 결정 품질을 평가합니다.

**3. 전략 다양성**
League Training으로 4개 에이전트가 서로 다른 전략을 유지합니다. 전략 동질화, 패배 회피 성향, 단순 족보 기반 플레이를 방지합니다.

---

## 참고 자료

- [PyPokerEngine 공식 문서](https://ishikota.github.io/PyPokerEngine/)
- [RLCard Toolkit](https://rlcard.org/)
- [Pluribus 논문 — Superhuman AI for multiplayer poker (Science, 2019)](https://www.science.org/doi/10.1126/science.aay2400)
- [텍사스 홀덤 규칙 (나무위키)](https://namu.wiki/w/%ED%85%8D%EC%82%AC%EC%8A%A4%20%ED%99%80%EB%8D%A4)
- [선행 연구 조사 보고서](./포커_에이전트_선행연구_보고서.docx)
