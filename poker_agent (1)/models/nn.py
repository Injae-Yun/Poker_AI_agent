"""
nn.py — numpy 기반 신경망 엔진

torch 없이 순수 numpy로 구현한 feedforward network.
- Linear, ReLU, Softmax, LayerNorm
- Adam 옵티마이저 (per-parameter 상태 유지)
- 직렬화(save/load) 지원

사용 예:
    net = Sequential([Linear(98, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 7)])
    logits = net.forward(state)
    net.backward(grad_output)
    adam.step(net)
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# ══════════════════════════════════════════════════════════
#  레이어 기반 클래스
# ══════════════════════════════════════════════════════════

class Layer:
    """모든 레이어의 추상 기반"""
    trainable = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> Dict[str, np.ndarray]:
        return {}

    def grads(self) -> Dict[str, np.ndarray]:
        return {}


class Linear(Layer):
    """완전 연결 레이어  y = xW + b"""
    trainable = True

    def __init__(self, in_dim: int, out_dim: int, init_scale: float = None):
        # He 초기화 (ReLU 이후에 적합)
        scale = init_scale or np.sqrt(2.0 / in_dim)
        self.W  = np.random.randn(in_dim, out_dim) * scale
        self.b  = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: (batch, out_dim) 또는 (out_dim,)
        x = self._x
        if x.ndim == 1:
            self.dW = np.outer(x, grad)
            self.db = grad.copy()
        else:
            self.dW = x.T @ grad
            self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self) -> Dict[str, np.ndarray]:
        return {'W': self.W, 'b': self.b}

    def grads(self) -> Dict[str, np.ndarray]:
        return {'W': self.dW, 'b': self.db}


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1 - self._out ** 2)


class Softmax(Layer):
    """수치 안정적 Softmax"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        self._out = e / e.sum(axis=-1, keepdims=True)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Jacobian 없이 cross-entropy와 함께 쓸 때 간소화
        s = self._out
        return s * (grad - (grad * s).sum(axis=-1, keepdims=True))


class LayerNorm(Layer):
    """레이어 정규화 — 학습 안정성 개선"""
    trainable = True

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = np.ones(dim)
        self.beta  = np.zeros(dim)
        self.eps   = eps
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x    = x
        self._mean = x.mean(axis=-1, keepdims=True)
        self._var  = x.var(axis=-1, keepdims=True)
        self._xn   = (x - self._mean) / np.sqrt(self._var + self.eps)
        return self.gamma * self._xn + self.beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        N    = self._x.shape[-1]
        xn   = self._xn
        std  = np.sqrt(self._var + self.eps)
        self.dgamma = (grad * xn).sum(axis=0) if grad.ndim > 1 else grad * xn
        self.dbeta  = grad.sum(axis=0) if grad.ndim > 1 else grad.copy()
        dxn  = grad * self.gamma
        dvar = (-0.5 * dxn * (self._x - self._mean) * (self._var + self.eps)**(-1.5)).sum(axis=-1, keepdims=True)
        dmean = (-dxn / std).sum(axis=-1, keepdims=True) + dvar * (-2 * (self._x - self._mean)).mean(axis=-1, keepdims=True)
        return dxn / std + dvar * 2 * (self._x - self._mean) / N + dmean / N

    def params(self):
        return {'gamma': self.gamma, 'beta': self.beta}

    def grads(self):
        return {'gamma': self.dgamma, 'beta': self.dbeta}


# ══════════════════════════════════════════════════════════
#  Sequential 네트워크
# ══════════════════════════════════════════════════════════

class Sequential:
    """레이어를 순서대로 연결한 네트워크"""

    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def trainable_layers(self) -> List[Layer]:
        return [l for l in self.layers if l.trainable]

    # ── 직렬화 ────────────────────────────────────────────
    def state_dict(self) -> List[Dict]:
        result = []
        for l in self.layers:
            if l.trainable:
                result.append({k: v.tolist() for k, v in l.params().items()})
            else:
                result.append({})
        return result

    def load_state_dict(self, state: List[Dict]) -> None:
        si = 0
        for l in self.layers:
            if l.trainable:
                for k, v in state[si].items():
                    setattr(l, k, np.array(v))
                si += 1
            else:
                si += 1

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.state_dict(), f)

    def load(self, path: str) -> None:
        with open(path) as f:
            self.load_state_dict(json.load(f))

    def copy_weights_from(self, other: 'Sequential') -> None:
        """다른 네트워크의 가중치를 복사"""
        for l_self, l_other in zip(self.trainable_layers(), other.trainable_layers()):
            for k in l_self.params():
                setattr(l_self, k, getattr(l_other, k).copy())


# ══════════════════════════════════════════════════════════
#  Adam 옵티마이저
# ══════════════════════════════════════════════════════════

class Adam:
    """
    Adam 옵티마이저
    네트워크 전체의 파라미터를 한 번에 업데이트합니다.
    """

    def __init__(
        self,
        lr:      float = 3e-4,
        beta1:   float = 0.9,
        beta2:   float = 0.999,
        eps:     float = 1e-8,
        clip:    float = 1.0,      # 그래디언트 클리핑 (L2 norm)
    ):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.clip  = clip
        self.t     = 0
        self._m: Dict[int, Dict[str, np.ndarray]] = {}
        self._v: Dict[int, Dict[str, np.ndarray]] = {}

    def step(self, net: Sequential) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t

        for i, layer in enumerate(net.trainable_layers()):
            if i not in self._m:
                self._m[i] = {k: np.zeros_like(v) for k, v in layer.params().items()}
                self._v[i] = {k: np.zeros_like(v) for k, v in layer.params().items()}

            g_all = layer.grads()

            # 그래디언트 클리핑
            if self.clip > 0:
                total_norm = np.sqrt(sum((g ** 2).sum() for g in g_all.values()))
                if total_norm > self.clip:
                    scale = self.clip / (total_norm + 1e-8)
                    g_all = {k: v * scale for k, v in g_all.items()}

            for k, g in g_all.items():
                self._m[i][k] = b1 * self._m[i][k] + (1 - b1) * g
                self._v[i][k] = b2 * self._v[i][k] + (1 - b2) * g ** 2

                m_hat = self._m[i][k] / bc1
                v_hat = self._v[i][k] / bc2
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

                param = getattr(layer, k)
                param -= update

    def reset(self) -> None:
        self.t   = 0
        self._m.clear()
        self._v.clear()


# ══════════════════════════════════════════════════════════
#  손실 함수
# ══════════════════════════════════════════════════════════

def mse_loss(pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    """MSE 손실 + 그래디언트"""
    diff = pred - target
    loss = float((diff ** 2).mean())
    grad = 2 * diff / diff.size
    return loss, grad


def policy_gradient_loss(
    log_probs:   np.ndarray,   # (T,) 선택된 액션의 log probability
    advantages:  np.ndarray,   # (T,) advantage 값
    entropy:     np.ndarray,   # (T,) 엔트로피 (탐색 장려)
    entropy_coef: float = 0.01,
) -> Tuple[float, np.ndarray]:
    """
    Policy Gradient (REINFORCE) 손실
    L = -log_prob × advantage - entropy_coef × entropy
    """
    pg_loss   = -(log_probs * advantages).mean()
    ent_loss  = -entropy_coef * entropy.mean()
    total     = pg_loss + ent_loss
    grad      = -(advantages / len(advantages))
    return float(total), grad
