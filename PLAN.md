# 개발 계획 (PLAN.md)

> Texas Hold'em Poker Agent — 단계별 구현 로드맵

---

## 개발 철학

- **점진적 구현**: 랜덤 → 룰 기반 → RL 순서로 단계를 밟아 각 단계를 검증하며 진행
- **측정 가능한 진행**: 각 Phase마다 명확한 성공 기준을 두고, 달성 후 다음 단계 진행
- **문제 격리**: 게임 엔진, 보상 설계, 학습 알고리즘을 독립 모듈로 분리하여 디버깅 용이성 확보

---

## Phase 0 — 환경 세팅 및 기반 구조 ✅ DONE

**목표**: 순수 Python + numpy 기반 게임 루프 정상 동작 확인
> pip 외부 연결 불가로 PyPokerEngine 대신 직접 구현. treys 없이 자체 핸드 평가기 작성.

### 구현 파일
- `engine/card.py`       — Card, Deck (reset/deal/burn)
- `engine/hand_eval.py`  — 7장 핸드 평가 (Royal~High Card, Wheel Straight)
- `engine/config.py`     — GameConfig (옵션 구조)
- `engine/state.py`      — AgentObservation, PlayerState (정보 은닉)
- `engine/game.py`       — 텍사스 홀덤 풀 게임 루프
- `agents/base_agent.py` — 추상 클래스 (declare_action 인터페이스)
- `agents/random_agent.py` — 랜덤 에이전트 (베이스라인)

### 검증 결과
- ✅ 4인 랜덤 에이전트 20게임 × 500라운드 오류 없이 완주
- ✅ 칩 보존 ALL PASS (20 게임, 칩 생성/소멸 없음)
- ✅ 핸드 평가 11개 케이스 ALL PASS (Wheel Straight 포함)

---

## Phase 1 — 룰 기반 에이전트 (Rule-Based Agent) ✅ DONE

**목표**: 팟 오즈와 핸드 에퀴티를 기반으로 한 기본 전략 에이전트 구현

### 구현 파일
- `utils/hand_evaluator.py`   — Monte Carlo 에퀴티 계산 + 프리플롭 룩업 테이블 (169핸드)
- `utils/deck_tracker.py`     — 잔여 덱 추적, 아웃츠 계산, Rule of 4/2
- `utils/opponent_profiler.py`— VPIP/PFR/AF 통계 추적, 스타일 분류 (Tight/Loose × Passive/Aggressive)
- `reward/ev_calculator.py`   — EV(fold)=0, EV(call), EV(raise) 계산 (매몰 비용 제외)
- `agents/rule_agent.py`      — EV 기반 의사결정, 포지션 보정, 블러프 로직

### 검증 결과 (50게임 × 200라운드)
- ✅ RuleAgent 평균 칩 변화: **+840** (vs Random **-280**), 차이 +1120 chips/game
- ✅ RuleAgent 최다 칩 보유 비율: 46% (랜덤 기대값 25%)
- ✅ 프리플롭 폴드율 24% (베팅 없는 체크 포함 시 합리적 범위)
- ✅ 팟 오즈 미충족 시 폴드 동작 확인
- ✅ 강한 핸드(strength≥0.65) 레이즈 동작 확인

### 설계 메모
- 체크(check)와 폴드(fold)는 별도 카운트: 전체 액션의 52%가 체크 (공짜 카드)
- 블러프: 후포지션 + 약한 상대 + 스택 충분 조건부 12% 확률 실행

---

## Phase 2 — 보상 설계 및 Actor-Critic 에이전트 ✅ DONE

**목표**: Advantage Function 기반 보상으로 RL 에이전트 학습

### 구현 파일
- `reward/reward_shaper.py`  — `RewardShaper` (A(s,a) = Q(s,a) - V(s), α=0.7 × 결정품질 + β=0.3 × 결과놀람)
- `utils/state_encoder.py`   — 98차원 상태 벡터 (홀카드 34 + 보드 19 + 핸드강도 14 + 컨텍스트 9 + 경제 6 + 상대통계 12 + 스트리트공격성 4)
- `models/actor.py`          — `ActorNetwork` FC(256)→LayerNorm→ReLU→FC(128)→ReLU→FC(7)→Softmax, 액션 7종
- `models/critic.py`         — `CriticNetwork` FC(256)→LayerNorm→ReLU→FC(128)→ReLU→FC(1)
- `models/nn.py`             — numpy 순수 구현 (Linear, ReLU, LayerNorm, Softmax, Adam)
- `agents/rl_agent.py`       — `RLAgent` (A2C), ε-greedy 탐색, 에피소드 버퍼, 핸드당 업데이트
- `training/trainer.py`      — Phase 2 학습 루프 (ε 감쇠, 체크포인트, CSV 로깅)

### 검증 결과
- ✅ RLAgent 학습 루프 오류 없이 동작 (3게임 × 50라운드 테스트)
- ✅ Actor/Critic 손실 정상 출력 및 체크포인트 저장 확인
- ✅ ε-greedy 탐색 + 합법 액션 마스킹 정상 동작

### 설계 메모
- torch 미사용: numpy 기반 자체 역전파 구현
- 액션 공간: fold(0), check/call(1), raise×0.5/1.0/1.5/2.0 pot(2-5), all-in(6) = 7종
- 보상 정규화: BB(=10) 단위로 스케일 조정

---

## Phase 2.5 — PyTorch 전환 및 아키텍처 고도화 ✅ DONE

**목표**: numpy 기반 신경망 → PyTorch 전환 + 듀얼 브랜치(MLP+GRU) 아키텍처 도입

### 배경
- 기존 Phase 2/3는 자체 numpy 역전파로 구현 (학습 효율 제한)
- PyTorch 환경 확보 후 전환 결정
- 상대 행동 시계열 정보를 GRU로 추적해 모델 고도화

### 신규 / 변경 파일

**`models/poker_net.py`** (신규)
  - `PokerNet` : MLP + GRU 듀얼 브랜치 공유 트렁크
    - MLP : Linear(104,256)+LayerNorm→Linear(256,256)+LayerNorm→Linear(256,128)
    - GRU : input=15, hidden=64, layers=2, dropout=0.1
    - Merge: Linear(192,64) → Actor head Linear(64,7) + Critic head Linear(64,1)
  - Orthogonal 초기화, `make_optimizer()` (Adam lr=3e-4, eps=1e-5)
  - `save()`/`load()` : `.pth` 포맷

**`models/seq_encoder.py`** (신규)
  - 베팅 히스토리 → GRU 입력 텐서 변환 (SEQ_DIM=15, MAX_SEQ_LEN=32)
  - 스텝당 인코딩: is_self(1) + opp_slot_onehot(3) + action_onehot(5) + street_onehot(4) + amount_ratio(1) + stack_depth(1) = 15차원

**`utils/state_encoder.py`** (변경)
  - STATE_DIM : 98 → 104 (잔여 덱 6차원 추가)
  - `_encode_remaining_deck()`: suit_remaining×4 + high_card_remaining×1 + total_remaining×1
  - `encode()`: `deck_tracker` 옵셔널 파라미터 추가

**`agents/rl_agent.py`** (전면 재작성)
  - ActorNetwork + CriticNetwork → 단일 `PokerNet`
  - `Transition`: `seq`, `seq_len` 필드 추가
  - `_update()`: PyTorch tensors + F.log_softmax + MSE + 그래디언트 클리핑(0.5)
  - 체크포인트: `net_P{id}_step_{N}.pth`

**`training/nfsp.py`** (전면 재작성)
  - `ReservoirBuffer`: (state, seq, seq_len, action_idx) 저장
  - `NFSPAgent`: ASP 네트워크를 PokerNet으로 교체, PyTorch cross-entropy SL 업데이트
  - 체크포인트: `net_P{id}_step_{N}.pth` + `asp_P{id}_step_{N}.pth`

**`training/league.py`** (전면 재작성)
  - `FrozenAgent`: ActorNetwork → PokerNet (actor_probs()로 추론)
  - `Snapshot`: actor_state(numpy) → net_state_dict(torch)
  - `maybe_snapshot()`: `agent._actor.net` → `agent.net`
  - **BUG FIX**: 모든 에이전트가 같은 파일명으로 덮어쓰던 문제 해결 → 에이전트별 개별 파일명

### 아키텍처 개선 근거

| 개선 항목 | 이전 | 이후 | 효과 |
|-----------|------|------|------|
| 학습 프레임워크 | 자체 numpy 역전파 | PyTorch autograd | GPU 지원, 안정적 그래디언트 |
| 상태 차원 | 98 | 104 | 잔여 덱 정보 학습 가능 |
| 시퀀스 모델링 | 없음 | GRU(hidden=64,layers=2) | 상대 행동 변화 추적 |
| 히든 레이어 | 2개 | 3개 (깊이 +1) | 더 높은 표현력 |
| 정규화 | 없음 | LayerNorm ×2 | 학습 안정성 향상 |
| 초기화 | 기본(Xavier) | Orthogonal | 초기 수렴 개선 |
| 체크포인트 | .json (numpy dict) | .pth (torch state_dict) | 표준 포맷, 에이전트별 개별 파일 |

---

## Phase 3 — 전략 다양성 보장 (League Training + NFSP) ✅ DONE

**목표**: 전략 동질화 및 패배 회피 성향 방지

### 구현 파일
- `training/nfsp.py`             — `ReservoirBuffer` (Reservoir Sampling 기반 균등 보존) + `NFSPAgent` (BR+ASP 혼합)
- `training/league.py`           — `AgentLeague` (Main×4 + Exploiter×2 + FrozenSnapshot 관리)
- `training/league_trainer.py`   — Phase 3 학습 루프 (snapshot, exploit 측정, CSV 로깅)
- `evaluation/exploitability.py` — `ExploitabilityMeasurer` (RuleAgent Best Response로 착취 측정)

### 설계 세부

**NFSPAgent (training/nfsp.py)**
  - `ReservoirBuffer(capacity=100k)` — 과거 BR 결정 (state, action) 균등 보존
  - `AveragePolicyNetwork` — Supervised cross-entropy 로 평균 전략 학습
  - η 확률로 BR(A2C) 사용 → Reservoir에 저장
  - (1-η) 확률로 ASP(SL) 사용 → Reservoir에 저장 안 함
  - SL 업데이트: N 핸드마다 Reservoir 배치 샘플링 → cross-entropy 업데이트

**AgentLeague (training/league.py)**
  - Main agents × 4: NFSP_Tight(η=0.10), NFSP_Loose(η=0.10), NFSP_Balanced(η=0.15), NFSP_Aggr(η=0.20)
  - Exploiter agents × 2: RuleAgent (취약점 착취 역할)
  - `FrozenAgent`: ActorNetwork 가중치 복사본으로 inference-only 동작
  - `maybe_snapshot()`: N게임마다 자동 스냅샷 저장

**ExploitabilityMeasurer (evaluation/exploitability.py)**
  - 대상 에이전트를 FrozenAgent로 동결 후 RuleAgent 3명과 대전
  - `exploit = RuleAgent 평균 chip delta`  낮을수록 착취에 강함

### 검증 결과 (5게임 × 50라운드 단기 테스트)
  - ✅ NFSPAgent 4종 정상 학습 루프 동작
  - ✅ ReservoirBuffer 크기 누적 확인
  - ✅ ExploitabilityMeasurer 정상 측정 (초기 미학습 상태 exploit≈+333)
  - ✅ snapshot 저장 및 FrozenAgent 대전 정상
  - ✅ League metrics / exploitability CSV 저장 확인

### 설계 메모
  - η 작을수록 ASP 중심 → GTO 수렴 안정적, η 클수록 BR 중심 → 빠른 전략 다양성
  - Reservoir capacity=100k: 학습 후반에도 초기 전략 견본 유지 가능
  - Exploitability 초기값 ≈ +333 (chip/game): 학습 후 감소 추세 관찰 필요

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
