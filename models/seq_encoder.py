"""
seq_encoder.py — 베팅 히스토리 시퀀스 인코더

BettingHistory(List[BettingAction])를 GRU 입력용 텐서로 변환합니다.

각 스텝 인코딩 (SEQ_DIM = 15):
  [0]      is_self          : 내가 한 액션인지 (1차원)
  [1..3]   opp_slot_onehot  : 상대 슬롯 (3차원, max 3명)
  [4..8]   action_onehot    : fold/call/check/raise/blind (5차원)
  [9..12]  street_onehot    : preflop/flop/turn/river (4차원)
  [13]     amount_ratio     : amount / pot_at_action (클리핑, 1차원)
  [14]     stack_depth      : stack_at_action / initial_stack (1차원)

총 = 1 + 3 + 5 + 4 + 1 + 1 = 15차원 ✓
"""

import numpy as np
import torch
from typing import List, Optional

from engine.state import BettingAction, STREETS

SEQ_DIM     = 15
MAX_SEQ_LEN = 32

# 액션 → 인덱스
_ACTION2IDX = {
    'fold':  0,
    'call':  1,
    'check': 2,
    'raise': 3,
    # 'blind' 또는 기타 알 수 없는 액션 → 4
}
_STREET2IDX = {s: i for i, s in enumerate(STREETS)}   # preflop=0..river=3

INITIAL_STACK = 1000   # config와 일치시킴 (정규화 기준)


def encode_betting_history(
    history:   List[BettingAction],
    self_id:   int,
    max_len:   int = MAX_SEQ_LEN,
    initial_stack: int = INITIAL_STACK,
) -> tuple:
    """
    베팅 히스토리를 (seq_tensor, seq_len) 형태로 반환합니다.

    Args:
        history:       BettingAction 리스트 (시간 순서)
        self_id:       현재 에이전트의 player_id
        max_len:       최대 스텝 수 (초과 시 최신 것만 보존)
        initial_stack: 스택 정규화 기준값

    Returns:
        seq    : np.ndarray  (max_len, SEQ_DIM), float32, zero-padded
        seq_len: int, 실제 유효 스텝 수 (0 ~ max_len)
    """
    # 최신 max_len 스텝만 보존
    steps = history[-max_len:] if len(history) > max_len else history
    seq_len = len(steps)

    seq = np.zeros((max_len, SEQ_DIM), dtype=np.float32)

    # 유일한 상대 player_id 목록 (self_id 제외, 등장 순서 정렬)
    opp_ids: List[int] = []
    for a in steps:
        if a.player_id != self_id and a.player_id not in opp_ids:
            opp_ids.append(a.player_id)

    for t, action in enumerate(steps):
        vec = seq[t]

        # [0] is_self
        vec[0] = 1.0 if action.player_id == self_id else 0.0

        # [1..3] opp_slot_onehot (상대방만, 최대 3슬롯)
        if action.player_id != self_id:
            slot = opp_ids.index(action.player_id)
            if slot < 3:
                vec[1 + slot] = 1.0

        # [4..8] action_onehot
        act_idx = _ACTION2IDX.get(action.action, 4)   # 4 = blind/기타
        vec[4 + act_idx] = 1.0

        # [9..12] street_onehot
        street_idx = _STREET2IDX.get(action.street, 0)
        vec[9 + street_idx] = 1.0

        # [13] amount_ratio = amount / (pot_at_action + 1e-6)  (0~3 클리핑)
        if action.pot_at_action > 0:
            ratio = action.amount / action.pot_at_action
        else:
            ratio = 0.0
        vec[13] = min(ratio, 3.0) / 3.0   # [0, 1] 정규화

        # [14] stack_depth = stack / initial_stack  (0~1 클리핑)
        vec[14] = min(action.stack_at_action / max(initial_stack, 1), 1.0)

    return seq, seq_len


def history_to_tensor(
    history:   List[BettingAction],
    self_id:   int,
    device:    str = 'cpu',
    max_len:   int = MAX_SEQ_LEN,
    initial_stack: int = INITIAL_STACK,
) -> tuple:
    """
    단일 샘플용 텐서 변환.

    Returns:
        seq_t    : torch.Tensor (1, max_len, SEQ_DIM)  — batch dim 포함
        seq_len_t: torch.Tensor (1,)                   — 유효 길이
    """
    seq, seq_len = encode_betting_history(history, self_id, max_len, initial_stack)
    seq_t     = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
    seq_len_t = torch.tensor([seq_len], dtype=torch.long, device=device)
    return seq_t, seq_len_t


def batch_to_tensor(
    seqs:     np.ndarray,   # (B, max_len, SEQ_DIM)
    seq_lens: np.ndarray,   # (B,)
    device:   str = 'cpu',
) -> tuple:
    """
    학습용 배치 텐서 변환.

    Returns:
        seq_t    : torch.Tensor (B, max_len, SEQ_DIM)
        seq_len_t: torch.Tensor (B,)
    """
    seq_t     = torch.tensor(seqs,     dtype=torch.float32, device=device)
    seq_len_t = torch.tensor(seq_lens, dtype=torch.long,    device=device)
    return seq_t, seq_len_t
