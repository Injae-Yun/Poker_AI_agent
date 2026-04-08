# Poker Agent Architecture

Texas Hold'em 4인 No-Limit 환경에서 동작하는 강화학습 에이전트의 구조 문서입니다.

---

## 1. 전체 구조 개요

```
AgentObservation
       |
       +---> StateEncoder (104-dim)  ----+
       |                                  |
       +---> SeqEncoder  (32x15-dim) ----+
                                          |
                                     PokerNet
                                    /         \
                               Actor Head   Critic Head
                              (7 logits)     (V(s))
                                    |
                              action_idx_to_decision()
                                    |
                              Game Engine
```

---

## 2. 입력: 상태 표현

### 2-1. 정적 상태 벡터 — `StateEncoder` (104차원)

`utils/state_encoder.py`

| 인덱스 | 크기 | 그룹 | 설명 |
|--------|------|------|------|
| 0:34 | 34 | 홀카드 | 카드 2장 각각 rank one-hot(13) + suit one-hot(4) = 17×2. rank 내림차순 정렬 |
| 34:53 | 19 | 보드 | rank 존재 여부(13) + suit 비율(4) + pair/trips 플래그(2) |
| 53:67 | 14 | 핸드 강도 | equity(1) + preflop_strength(1) + 현재 핸드 rank one-hot(10) + suited/paired(2) |
| 67:76 | 9 | 컨텍스트 | street one-hot(4) + position one-hot(4) + active_players/4(1) |
| 76:82 | 6 | 경제 지표 | stack/initial(1) + pot/max_pot(1) + pot_odds(1) + call_norm(1) + raises/4(1) + stack_bb(1) |
| 82:94 | 12 | 상대 통계 | 3명 x [VPIP, PFR, AF_norm, est_strength] — OpponentProfiler 누적값 |
| 94:98 | 4 | 스트리트 공격성 | 각 스트리트별 raise/(total actions) 비율 |
| 98:104 | 6 | 잔여 덱 | suit별 잔여 수/13(4) + 하이카드 잔여/20(1) + 전체 잔여/52(1) |

모든 값은 [0, 1]로 정규화. `encode()` 말미에 shape==(104,), NaN/Inf 없음 assert.

### 2-2. 시퀀스 입력 — `SeqEncoder` (32 x 15차원)

`models/seq_encoder.py`

베팅 히스토리를 GRU 입력용 시퀀스로 변환합니다. 최근 32 스텝을 보존하고 zero-padding.

| 차원 | 크기 | 설명 |
|------|------|------|
| [0] | 1 | is_self: 내가 한 액션이면 1 |
| [1:4] | 3 | 상대 슬롯 one-hot (등장 순서 기준 최대 3명) |
| [4:9] | 5 | action one-hot: fold / call / check / raise / blind |
| [9:13] | 4 | street one-hot: preflop / flop / turn / river |
| [13] | 1 | amount_ratio: amount / pot_at_action (0~1 클리핑) |
| [14] | 1 | stack_depth: stack_at_action / initial_stack |

---

## 3. 네트워크: `PokerNet`

`models/poker_net.py`

MLP와 GRU 두 브랜치를 병합한 공유 트렁크에 Actor/Critic 헤드를 올립니다.

```
정적 피처 (104)              시퀀스 피처 (32 x 15)
      |                              |
  Linear(104, 256)            GRU(input=15, hidden=64,
  LayerNorm + ReLU                layers=2, dropout=0.1)
  Linear(256, 256)                  |
  LayerNorm + ReLU          마지막 유효 hidden (64)
  Linear(256, 128)                LayerNorm
  ReLU                              |
      |____________ concat(192) ____|
                        |
                  Linear(192, 64)
                      ReLU
                   /          \
         Linear(64, 7)    Linear(64, 1)
          Actor Head        Critic Head
         (action logits)      (V(s))
```

| 항목 | 값 |
|------|-----|
| 정적 입력 차원 | 104 |
| 시퀀스 입력 차원 | 15 |
| 최대 시퀀스 길이 | 32 |
| GRU hidden | 64, layers=2 |
| MLP 출력 | 128 |
| 병합 후 | 192 -> 64 |
| 액션 공간 | 7 |
| 가중치 초기화 | Orthogonal (헤드는 gain=0.01) |
| 파라미터 수 | 약 220k |

---

## 4. 액션 공간 (7종)

`models/poker_net.py — RAISE_SIZES`

| 인덱스 | 액션 | 금액 |
|--------|------|------|
| 0 | fold | 0 |
| 1 | check / call | call_amount |
| 2 | raise 0.5x pot | max(min_raise, pot × 0.5) |
| 3 | raise 1.0x pot | max(min_raise, pot × 1.0) |
| 4 | raise 1.5x pot | max(min_raise, pot × 1.5) |
| 5 | raise 2.0x pot | max(min_raise, pot × 2.0) |
| 6 | all-in | max_raise |

합법 액션 마스킹(`build_valid_mask`)을 거친 뒤 softmax -> 샘플링.

---

## 5. 보상 설계 — `RewardShaper`

`reward/reward_shaper.py`

```
R(s,a) = alpha * decision_quality + beta * outcome_surprise

decision_quality = (EV(선택한 액션) - EV(최선 액션)) / bb_scale
outcome_surprise = chip_delta / bb_scale / N_steps - V(s)

alpha  = 0.7
beta   = 0.3
bb_scale = initial_stack / 100   (stack=10000 -> bb_scale=100)
```

- `EV(fold) = 0` (매몰 비용 제외)
- `EV(call) = equity × (pot + call) - (1-equity) × call`
- `EV(raise)` = fold_prob × pot + (1-fold_prob) × (equity × new_pot - (1-equity) × total_put_in)
- `chip_delta`는 핸드 내 N개 결정에 균등 배분

---

## 6. 학습 알고리즘 — `RLAgent` (A2C)

`agents/rl_agent.py`

### 업데이트 루프 (핸드 단위)

```
declare_action() 호출마다:
  state  = StateEncoder.encode()          # (104,)
  seq    = encode_betting_history()       # (32, 15)
  logits, value = PokerNet(state, seq)
  action_idx ~ softmax(masked_logits)     # epsilon-greedy

on_round_end() 호출 시:
  RewardShaper.on_hand_end() -> rewards[]
  episode_buffer에 Transition 누적

update_every 핸드마다:
  returns    = discounted_returns(rewards, gamma=0.99)
  advantages = returns - values.detach()
  advantages = normalize(advantages)

  actor_loss  = -mean(log_prob * advantage) - entropy_coef * entropy
  critic_loss = MSE(values, returns)
  total_loss  = actor_loss + 0.5 * critic_loss

  Adam(lr=3e-4).step()
  grad_clip = 0.5
```

---

## 7. NFSP 에이전트 — `NFSPAgent`

`training/nfsp.py`

`RLAgent`를 상속하고 Best Response(BR)와 Average Strategy Policy(ASP)를 혼합합니다.

```
핸드 시작 시:
  eta 확률 -> BR 모드 (RLAgent 정책)
              -> Reservoir에 (state, seq, action) 저장
  (1-eta) 확률 -> ASP 모드 (asp_net 정책)
                  -> Reservoir에 저장하지 않음

sl_update_every 핸드마다:
  Reservoir에서 128개 샘플링
  cross_entropy(asp_net(state, seq), action) 최소화
```

| 에이전트 | eta | epsilon | 특징 |
|----------|-----|---------|------|
| NFSP_Tight | 0.10 | 0.05 | 낮은 탐색, 타이트 수렴 |
| NFSP_Loose | 0.10 | 0.15 | 높은 탐색, 루즈 스타일 |
| NFSP_Balanced | 0.15 | 0.10 | 균형 |
| NFSP_Aggr | 0.20 | 0.08 | BR 비중 높음, 공격적 |

### ReservoirBuffer

`capacity=100,000`. Reservoir Sampling으로 과거 전략을 균등 확률로 유지.
저장 항목: `(state: 104, seq: 32×15, seq_len: int, action_idx: int)`

---

## 8. League Training — `AgentLeague`

`training/league.py`

```
4인 게임 슬롯 구성:
  [0..3] main_agents (NFSPAgent x4) 기본 배치
  확률 0.4 -> 랜덤 슬롯 1개를 RuleAgent(Exploiter)로 교체
  스냅샷 2개 이상 -> 추가 슬롯을 FrozenAgent(과거 스냅샷)으로 교체

snapshot_every(15)게임마다:
  각 에이전트 PokerNet state_dict -> in-memory Snapshot 저장
  최대 20 x 4 = 80개 유지 (초과 시 오래된 것 삭제)
```

### FrozenAgent

PokerNet 가중치 복사본으로 inference-only 동작. 학습 없음.
`actor_probs()` -> argmax로 결정적 행동.

---

## 9. 착취가능성 측정 — `ExploitabilityMeasurer`

`evaluation/exploitability.py`

```
대상 에이전트 -> FrozenAgent 복사
나머지 3자리: RuleAgent (Best Response 근사)
N게임 실행 후:
  exploitability = RuleAgent 평균 chip_delta  (낮을수록 GTO에 가까움)
```

`exploit_every(30)`게임마다 자동 측정. 결과는 `logs/league/exploitability.csv`.

---

## 10. 상대 모델링 — `OpponentProfiler`

`utils/opponent_profiler.py`

베팅 히스토리에서 상대별 통계를 누적합니다. 게임 간 지속(프로파일러 미초기화).

| 통계 | 설명 | 사전확률 |
|------|------|---------|
| VPIP | 자발적 팟 참여율 | 0.25 |
| PFR | 프리플롭 레이즈율 | 0.15 |
| AF | Aggression Factor = (raise+bet)/call | 1.0 |
| WTSD | 쇼다운 참여율 | 0.25 |
| WSD | 쇼다운 승률 | 0.50 |

`estimated_hand_strength`: VPIP/AF 기반 베이지안 추정 (0.1~0.9).

---

## 11. 체크포인트 구조

```
checkpoints/
  net_P0_step_N.pth          Phase 2 RLAgent BR 가중치
  best_net_P0.pth            Phase 2 best model

checkpoints/league/
  net_P{i}_step_N.pth        NFSPAgent i의 BR 가중치
  asp_P{i}_step_N.pth        NFSPAgent i의 ASP 가중치
  league_meta_N.json         스냅샷 메타 + 게임 카운터
  best_net_P{i}.pth          NFSPAgent i의 best model

logs/training/
  training_metrics.csv       Phase 2 게임별 지표

logs/league/
  league_metrics.csv         Phase 3 게임별 에이전트 지표
  exploitability.csv         착취가능성 측정 이력
```

---

## 12. 알려진 제약 / 개선 예정

| 항목 | 현재 | 개선 방향 |
|------|------|---------|
| 프레임워크 | PyTorch CPU only | CUDA 옵션 추가 예정 |
| MC 시뮬레이션 | 액션당 900회 (중복 2회) | 결과 공유로 1회로 축소 예정 |
| seq_encoder INITIAL_STACK | 1000 하드코딩 | initial_stack 파라미터화 예정 |
| Reservoir capacity | 100,000 | 메모리 부족 시 20,000으로 축소 예정 |
| Early stopping | trainer.py 적용 완료 | league_trainer.py 적용 예정 |
