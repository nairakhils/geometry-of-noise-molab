"""Tiny 2-layer MLP with tanh, pure numpy. Forward + backward + Adam.

Designed so the training script in scripts/train_score_mlp.py can fit a
score model from scratch with no framework dependency, and so the inline
helpers cell in notebooks/walkthrough.py can carry the same API for
WASM.
"""

from __future__ import annotations

import numpy as np


class TinyMLP:
    """Two-layer MLP, tanh activations, Adam optimizer state inline."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_in, d_hidden)) * (2 / d_in) ** 0.5
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.standard_normal((d_hidden, d_out)) * (2 / d_hidden) ** 0.5
        self.b2 = np.zeros(d_out)
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._step = 0

    def _params(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        self._x = x
        self._h_pre = x @ self.W1 + self.b1
        self._h = np.tanh(self._h_pre)
        self._y = self._h @ self.W2 + self.b2
        return self._y

    def backward(self, dy):
        dW2 = self._h.T @ dy
        db2 = dy.sum(0)
        dh = dy @ self.W2.T
        dh_pre = dh * (1.0 - self._h ** 2)
        dW1 = self._x.T @ dh_pre
        db1 = dh_pre.sum(0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def adam_step(self, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._step += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            m_hat = self._m[k] / (1 - b1 ** self._step)
            v_hat = self._v[k] / (1 - b2 ** self._step)
            getattr(self, k)[...] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def jacobian_at(self, x0, h: float = 1e-4):
        """Effective input-output Jacobian via central differences at x0."""
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        d_in = x0.shape[0]
        eye = np.eye(d_in) * h
        x_plus = x0 + eye
        x_minus = x0 - eye
        y_plus = self.forward(x_plus)
        y_minus = self.forward(x_minus)
        # y has shape (d_in, d_out). J[i, j] = dy_i/dx_j.
        return ((y_plus - y_minus).T) / (2.0 * h)


def train_score_mlp(
    d_in: int,
    d_out: int,
    sample_fn,
    n_steps: int = 2000,
    batch: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    d_hidden: int = 64,
):
    """Generic trainer. `sample_fn(rng, batch)` returns (x_batch, y_batch)."""
    net = TinyMLP(d_in, d_out, d_hidden=d_hidden, seed=seed)
    rng = np.random.default_rng(seed + 1)
    losses = np.empty(n_steps, dtype=np.float64)
    for step in range(n_steps):
        x, y = sample_fn(rng, batch)
        pred = net.forward(x)
        diff = pred - y
        # loss = (1/batch) * sum_i ||pred_i - y_i||^2  (sum over output dims,
        # mean over batch). Matches dy = 2*diff/batch.
        losses[step] = (diff * diff).sum(axis=1).mean()
        dy = 2.0 * diff / batch
        grads = net.backward(dy)
        net.adam_step(grads, lr=lr)
    return net, losses
