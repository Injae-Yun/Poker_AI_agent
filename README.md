# Texas Hold'em Poker Agent

> 텍사스 홀덤 포커에서 높은 승률을 목표로 하는 멀티에이전트 AI 시스템

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
| 게임 엔진 | [PyPokerEngine](https://github.com/ishikota/PyPokerEngine) |
| 강화학습 | PyTorch + Actor-Critic (A2C) |
| 에퀴티 계산 | Monte Carlo Simulation |
| 실험 추적 | (추후 결정) |
| 언어 | Python 3.10+ |

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
│   └── rl_agent.py            # 강화학습 에이전트 (Phase 2~3)
│
├── models/                    # 신경망 모델
│   ├── actor.py               # 정책 네트워크 (Actor)
│   ├── critic.py              # 가치 네트워크 (Critic)
│   └── opponent_model.py      # 상대 모델링 네트워크
│
├── reward/                    # 보상 설계
│   ├── ev_calculator.py       # EV 계산 (팟 오즈, 에퀴티)
│   └── reward_shaper.py       # Advantage Function 기반 보상
│
├── training/                  # 학습 파이프라인
│   ├── trainer.py             # 기본 학습 루프
│   ├── league.py              # League Training 관리
│   └── nfsp.py                # Neural Fictitious Self-Play
│
├── evaluation/                # 평가 및 분석
│   ├── evaluator.py           # 에이전트 성능 평가
│   └── exploitability.py      # 착취가능성 측정
│
└── utils/
    ├── hand_evaluator.py      # 핸드 강도 평가
    ├── deck_tracker.py        # 덱/카드 추적
    └── stats.py               # 통계 집계
```

---

## 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# 랜덤 에이전트 간 테스트 게임 실행
python -m engine.game --players 4 --mode random

# 룰 기반 에이전트 학습 (Phase 1)
python -m training.trainer --agent rule --episodes 1000

# RL 에이전트 학습 (Phase 2)
python -m training.trainer --agent rl --episodes 100000
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
