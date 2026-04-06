# 개발 계획 (PLAN.md)

> Texas Hold'em Poker Agent — 단계별 구현 로드맵

---

## 개발 철학

- **점진적 구현**: 랜덤 → 룰 기반 → RL 순서로 단계를 밟아 각 단계를 검증하며 진행
- **측정 가능한 진행**: 각 Phase마다 명확한 성공 기준을 두고, 달성 후 다음 단계 진행
- **문제 격리**: 게임 엔진, 보상 설계, 학습 알고리즘을 독립 모듈로 분리하여 디버깅 용이성 확보

---

## Phase 0 — 환경 세팅 및 기반 구조

**목표**: PyPokerEngine 기반 게임 루프가 정상 동작하는 것을 확인

### 작업 목록

- [ ] 의존성 설치 및 환경 구성
  - `PyPokerEngine`, `torch`, `numpy`, `treys` (핸드 평가 라이브러리)
- [ ] `GameConfig` 클래스 구현
  - 게임 타입, 플레이어 수, 블라인드 구조 등 옵션화
- [ ] `BaseAgent` 추상 클래스 정의
  - `declare_action(valid_actions, hole_cards, round_state)` 인터페이스
  - 에이전트 간 정보 공유 차단 구조 확인
- [ ] `RandomAgent` 구현 (합법 액션 중 무작위 선택)
- [ ] 4인 대전 게임 루프 실행 테스트
- [ ] 게임 상태 로깅 구조 구현 (배팅 히스토리, 칩 변화 등)

### 성공 기준
- 4인 랜덤 에이전트가 100 라운드를 오류 없이 완주
- 게임 상태(팟, 스택, 커뮤니티 카드, 배팅 히스토리)가 정확히 기록됨

---

## Phase 1 — 룰 기반 에이전트 (Rule-Based Agent)

**목표**: 팟 오즈와 핸드 에퀴티를 기반으로 한 기본 전략 에이전트 구현

### 작업 목록

- [ ] `HandEvaluator` 구현
  - `treys` 라이브러리로 핸드 강도(Hand Rank) 계산
  - 프리플롭 핸드 강도 룩업 테이블 (169가지 핸드)
- [ ] `DeckTracker` 구현
  - 공개된 카드를 제거한 잔여 덱 추적
  - 아웃츠(Outs) 계산 유틸리티
- [ ] `EVCalculator` 구현
  - 몬테카를로 시뮬레이션 기반 에퀴티 계산
  - `EV(fold)`, `EV(call)`, `EV(raise)` 계산 (매몰 비용 제외)
  - 팟 오즈(Pot Odds) 계산
- [ ] `RuleAgent` 구현
  - 프리플롭: 핸드 강도 + 포지션 기반 전략 테이블
  - 플롭/턴/리버: 에퀴티 vs 팟 오즈 비교 의사결정
  - 기본 블러프 로직 (포지션, 팟 크기 조건부)
- [ ] `OpponentProfiler` 구현 (기초)
  - 상대별 VPIP, PFR, AF 통계 누적
  - 스타일 분류: 타이트/루즈 × 패시브/어그레시브

### 성공 기준
- `RuleAgent` 4개가 `RandomAgent` 4개를 상대로 통계적으로 유의미한 수익 달성
- 폴드 빈도가 합리적 범위 내 (프리플롭 기준 30~60%)
- 팟 오즈 미충족 상황에서 올바르게 폴드하는 것 확인

---

## Phase 2 — 보상 설계 및 Actor-Critic 에이전트

**목표**: Advantage Function 기반 보상으로 RL 에이전트 학습

### 작업 목록

- [ ] `RewardShaper` 구현
  - `A(s,a) = Q(s,a) - V(s)` 기반 보상 계산
  - 결정 품질 보상 (즉시): `EV(선택한 액션) - EV(최선 액션)`
  - 결과 놀람 보상 (지연): `실제 칩 결과 - V(s)`
  - 가중 합산: `reward = 0.7 × decision_quality + 0.3 × outcome_surprise`

- [ ] 상태 표현 벡터 설계
  - 홀 카드 인코딩 (슈트 × 숫자)
  - 커뮤니티 카드 인코딩
  - 핸드 에퀴티 (정규화)
  - 팟 오즈 (정규화)
  - 스택 크기 비율 (내 스택 / 총 스택)
  - 포지션 원핫 인코딩
  - 배팅 라운드 원핫 인코딩
  - 상대 프로파일 통계 (VPIP, PFR, AF per opponent)

- [ ] `ActorNetwork` 구현
  - 입력: 상태 벡터
  - 출력: 액션 확률 분포 [fold, call, raise_small, raise_medium, raise_large, all_in]
  - 구조: FC(256) → ReLU → FC(128) → ReLU → FC(6) → Softmax

- [ ] `CriticNetwork` 구현
  - 입력: 상태 벡터
  - 출력: V(s) (스칼라)
  - 구조: FC(256) → ReLU → FC(128) → ReLU → FC(1)

- [ ] `RLAgent` (A2C) 구현
  - Actor-Critic 업데이트 루프
  - `RewardShaper`와 연동
  - Entropy 보너스 (탐색 촉진)

- [ ] 학습 루프 구현 (`trainer.py`)
  - 에피소드 단위 학습
  - 체크포인트 저장 (N 에피소드마다)
  - 학습 곡선 로깅 (평균 보상, 폴드 비율, 승률)

### 성공 기준
- `RLAgent`가 `RuleAgent`를 상대로 수렴 (손실 곡선 안정화)
- 폴드 / 콜 / 레이즈 비율이 상황에 따라 다르게 나타남
- 강한 핸드와 약한 핸드에서 다른 배팅 행동 확인

---

## Phase 3 — 전략 다양성 보장 (League Training + NFSP)

**목표**: 전략 동질화 및 패배 회피 성향 방지

### 작업 목록

- [ ] `ReservoirBuffer` 구현
  - 과거 행동 이력을 일정 확률로 보존하는 샘플링 버퍼
- [ ] `NFSPAgent` 구현
  - Best Response Network (DQN) + Average Policy Network (Supervised)
  - η(eta) 파라미터로 두 정책 간 전환 비율 조정
- [ ] `AgentLeague` 구현
  - Main agents (4개) + Exploiter agents (2개) + Frozen snapshots
  - N 에피소드마다 현재 에이전트를 스냅샷으로 동결 저장
  - 매칭 시 현재 에이전트 + 랜덤 스냅샷 조합으로 대전 구성
- [ ] `ExploitabilityMeasurer` 구현
  - 각 에이전트를 최적 반응 에이전트(Best Response)가 얼마나 착취할 수 있는지 측정
  - 주기적으로 측정하여 학습 품질 모니터링
- [ ] 에이전트 다양성 초기화
  - 에이전트별 다른 하이퍼파라미터 및 랜덤 시드 적용
  - 초기 전략 편향 설정 (타이트/루즈, 어그레시브/패시브)

### 성공 기준
- 4개 에이전트가 서로 다른 전략 프로파일(VPIP, PFR, AF)을 보임
- Exploitability가 학습 진행에 따라 감소
- 한 에이전트가 동일한 전략으로 나머지 에이전트를 지속적으로 지배하지 않음

---

## Phase 4 — 상대 모델링 고도화

**목표**: 상대 배팅 패턴 학습 및 동적 전략 전환 (GTO ↔ Exploit)

### 작업 목록

- [ ] `OpponentModel` 신경망 구현
  - 입력: 상대의 배팅 히스토리 시퀀스
  - 출력: 상대 핸드 레인지 분포
- [ ] GTO ↔ Exploit 전환 로직
  - 상대 모델 신뢰도가 충분할 때 익스플로잇 전략으로 전환
  - 블러프 탐지 및 대응 전략
- [ ] 배팅 사이즈 최적화
  - 상대 폴드 확률 추정 기반 최적 레이즈 사이즈 계산

### 성공 기준
- 약한 상대(랜덤, 룰 기반)에 대해 GTO 에이전트 대비 높은 수익률 달성
- 블러프 빈도가 높은 상대에 대한 콜 빈도 상승 확인

---

## Phase 5 — 평가 및 정리

**목표**: 전체 시스템 성능 평가 및 확장 구조 정비

### 작업 목록

- [ ] 에이전트 토너먼트 평가
  - Random / Rule / RL-Phase2 / RL-Phase3 / RL-Phase4 간 리그전
  - mbb/hand (milli-big-blind per hand) 단위 수익률 비교
- [ ] 시각화 대시보드
  - 학습 곡선, 에이전트별 전략 프로파일, Exploitability 추이
- [ ] 게임 타입 옵션 확장 준비
  - `GameConfig`에 Omaha, Short Deck 등 추가 가능한 구조 검토
- [ ] 코드 정리 및 문서화

---

## 기술 결정 사항 (Decision Log)

| 항목 | 결정 | 이유 |
|------|------|------|
| 게임 엔진 | PyPokerEngine | 에이전트 인터페이스 단순, RL 연동 용이 |
| 핸드 평가 | treys 라이브러리 | 빠른 핸드 랭킹 계산 |
| 에퀴티 계산 | Monte Carlo (500회 샘플링) | 실시간 성능과 정확도의 균형 |
| 보상 설계 | Advantage Function (A2C) | 폴드 포함 모든 액션의 결정 품질 측정 |
| 전략 다양성 | League Training + NFSP | 전략 동질화, 패배 회피, 족보 기반 플레이 방지 |
| 플레이어 수 | 4인 (현재) | 프로젝트 요구사항; 옵션으로 확장 예정 |

---

## 마일스톤 요약

```
Phase 0  환경 세팅        → 게임 루프 정상 동작
Phase 1  룰 기반 에이전트  → RandomAgent 대비 유의미한 수익
Phase 2  RL 에이전트       → RuleAgent 대비 수렴
Phase 3  전략 다양성       → Exploitability 감소 확인
Phase 4  상대 모델링       → GTO 대비 Exploit 수익 개선
Phase 5  평가 및 정리      → 전체 시스템 토너먼트 비교
```
