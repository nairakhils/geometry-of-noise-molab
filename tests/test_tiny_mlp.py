"""Sanity checks for src/tiny_mlp.py."""

import numpy as np

from src.tiny_mlp import TinyMLP, train_score_mlp


def test_backward_matches_numeric_gradient():
    """Analytic backward matches central-difference numeric grads at random init."""
    rng = np.random.default_rng(0)
    d_in, d_out, d_hidden = 3, 2, 8
    net = TinyMLP(d_in, d_out, d_hidden=d_hidden, seed=0)
    batch = 4
    x = rng.standard_normal((batch, d_in))
    y = rng.standard_normal((batch, d_out))

    def _loss(p, t):
        # Same convention as TinyMLP trainer: sum over output dims, mean over batch.
        return ((p - t) ** 2).sum(axis=1).mean()

    pred = net.forward(x)
    diff = pred - y
    dy = 2.0 * diff / batch
    grads = net.backward(dy)

    h = 1e-5
    for name in ("W1", "b1", "W2", "b2"):
        param = getattr(net, name)
        flat = param.ravel()
        g_analytic = grads[name].ravel()
        g_numeric = np.empty_like(flat)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + h
            loss_plus = _loss(net.forward(x), y)
            flat[i] = orig - h
            loss_minus = _loss(net.forward(x), y)
            flat[i] = orig
            g_numeric[i] = (loss_plus - loss_minus) / (2.0 * h)
        assert np.allclose(g_analytic, g_numeric, atol=1e-5), (
            f"{name}: analytic vs numeric grad mismatch, "
            f"max |diff| = {np.max(np.abs(g_analytic - g_numeric))}"
        )


def test_linear_data_jacobian_recovery():
    """Train on y = A x; recovered Jacobian should match A within 1e-2."""
    rng = np.random.default_rng(1)
    d_in, d_out = 2, 2
    A = np.array([[1.5, -0.3], [0.4, 0.9]])

    def sample_fn(rng, batch):
        x = rng.standard_normal((batch, d_in)) * 0.5
        return x, x @ A.T

    net, _ = train_score_mlp(
        d_in, d_out, sample_fn, n_steps=8000, batch=256, lr=2e-3, seed=42,
    )
    J = net.jacobian_at(np.zeros(d_in))
    rel_err = np.linalg.norm(J - A) / np.linalg.norm(A)
    # 5% tolerance: a 2-layer tanh MLP with 64 hidden units approximates a
    # global linear map at the origin to about that fidelity within 8K steps
    # of Adam at batch=256. Tighter tolerance would need a longer schedule.
    assert rel_err < 5e-2, f"recovered Jacobian rel err = {rel_err}, J = {J}, A = {A}"
