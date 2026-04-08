# Texas Hold'em Poker Agent

> 텍사스 홀덤 포커에서 높은 승률을 목표로 하는 멀티에이전트 AI 시스템

---

## 학습 현황 리포트 (2026-04-08 기준)

### 실험 개요

| 항목 | 내용 |
|------|------|
| 학습 알고리즘 | NFSP (Neural Fictitious Self-Play) + A2C |
| 모델 구조 | PokerNet: MLP(104→256→128) + GRU(hidden=64, 2-layer), 7 actions |
| 학습 설정 | League Training 4×NFSPAgent + 2×RuleAgent exploiter + snapshot pool |
| 누적 학습량 | 500 games × 200 rounds = 약 100,000 핸드 |
| 초기 스택 | 10,000 칩 (BB=10, SB=5) |
| 하드웨어 | RTX 5060 Ti (CUDA 12.8 nightly) |

### 기준선 성능 (비학습 에이전트)

> 실험: 50게임 × 100라운드, 초기 스택 10,000

| 에이전트 | 평균 칩 변화 | 수익 게임율 | 1위 비율 | 레이즈율 |
|---------|------------|-----------|---------|---------|
| RuleAgent | **+10,000** | 50% | 50% | 36.3% |
| RandomAgent | -3,000 ~ -5,800 | 12~24% | 12~24% | ~28% |

### NFSP 학습 결과 (vs RuleAgent + 2×RandomAgent, 30게임)

> 체크포인트 step=300 (게임 300회 학습 후)

| 에이전트 | 평균 칩 변화 | 수익 게임율 | 1위 비율 | 레이즈율 |
|---------|------------|-----------|---------|---------|
| **NFSP (step 300)** | -7,333 | 7% | 6.7% | **74.8%** |
| RuleAgent | +12,667 | 57% | 56.7% | 23.0% |
| RandomAgent | -3,000 ~ -1,900 | 17~20% | 17~20% | ~29% |

### 착취가능성 (Exploitability) 추이

> 낮을수록 GTO에 가까운 전략. RuleAgent 3명이 대상 에이전트를 상대로 얻는 평균 칩/게임.

| game | NFSP_Tight | NFSP_Loose | NFSP_Balanced | NFSP_Aggr |
|------|-----------|-----------|--------------|---------|
| 30 (초기) | +1,556 | +1,556 | +1,556 | +1,556 |
| 330 | +2,444 | **-222** | +2,444 | +1,556 |
| 390 | **+667** | +2,444 | +2,444 | +2,444 |
| 450 | +1,556 | +1,556 | +1,556 | +2,444 |

- `NFSP_Loose` game 330에서 exploit **-222** 달성 (RuleAgent 상대로 이김)
- `NFSP_Tight` game 390에서 **+667** 기록 (초기 대비 57% 개선)

### 분석 및 한계

**현재 문제: 정책 붕괴 (Policy Collapse)**

NFSP 에이전트의 레이즈율이 74.8%까지 치솟아 RandomAgent보다 낮은 성능을 보입니다.
- 원인: A2C Best Response 네트워크가 "항상 올인" 전략으로 수렴 (국소 최적)
- ASP 네트워크의 SL 학습은 game 165부터 정상 작동 (sl_loss 1.9 → 0.35 수렴)
- 그러나 reservoir에 쌓인 BR 경험 자체가 이미 편향된 상태

**다음 단계 (Phase 4 예정)**

- 레이즈 액션 빈도 패널티를 보상에 적용 (과도한 레이즈 억제)
- ε-greedy 탐색 범위 재설계
- 추가 리그 학습 또는 하이퍼파라미터 조정

### 체크포인트 위치

```
checkpoints/league/
├── best_net_P{0-3}.pth      # 각 에이전트 BR 네트워크 best
├── best_asp_P{0-3}.pth      # 각 에이전트 ASP 네트워크 best
└── net/asp_P{i}_step_N.pth  # 스텝별 체크포인트 (30~500)
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
