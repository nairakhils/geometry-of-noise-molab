"""Train tiny MLP score models from scratch and dump results to npz.

Three sweeps:
  (i)   per-t MLPs: one model per (t, parameterization) pair, trained on
        x ~ N(0, Sigma) with Sigma = diag([2.0, 0.5]).
  (ii)  global MLPs: one model per parameterization, conditioned on
        (u, t) and trained with t sampled uniformly in [0.05, 0.95].
  (iii) t_min sweep: one global MLP per (parameterization, t_min) pair,
        with t sampled uniformly in [t_min, 0.95]. The noise loss
        explodes as t_min -> 0; the velocity loss stays bounded.

All trainings together run in ~25 s on one CPU. No framework deps.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schedules import a_of_t, adot_of_t, b_of_t, bdot_of_t
from src.tiny_mlp import TinyMLP, train_score_mlp


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SIGMA_2D = np.diag([2.0, 0.5])
T_PER_T = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.95])
T_MIN_SWEEP = np.array([0.5, 0.1, 0.05, 0.01, 0.005])
N_STEPS = 5000
N_STEPS_GLOBAL = 8000
N_STEPS_SWEEP = 3000
BATCH = 256
DRIFT_EVAL_BATCH = 4096


# ---------- closed-form A matrices, lifted from scripts/linear_score_fit.py ----

def exact_A_eps(t, Sigma):
    a = float(a_of_t(t)); b = float(b_of_t(t))
    M = a * a * Sigma + b * b * np.eye(Sigma.shape[0])
    return b * np.linalg.inv(M)


def exact_A_v(t, Sigma):
    a = float(a_of_t(t)); b = float(b_of_t(t))
    a_dot = float(adot_of_t(t)); b_dot = float(bdot_of_t(t))
    d = Sigma.shape[0]
    M = a * a * Sigma + b * b * np.eye(d)
    return (a_dot * a * Sigma + b_dot * b * np.eye(d)) @ np.linalg.inv(M)


def _sample_pair(rng, batch, Sigma, t):
    a = float(a_of_t(t)); b = float(b_of_t(t))
    a_dot = float(adot_of_t(t)); b_dot = float(bdot_of_t(t))
    d = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    x = rng.standard_normal((batch, d)) @ L.T
    eps = rng.standard_normal((batch, d))
    u = a * x + b * eps
    v = a_dot * x + b_dot * eps
    return u, eps, v


# ---------- (i) per-t models -----------------------------------------------

def _train_per_t(t, Sigma, target_kind, n_steps, batch, seed):
    def sample_fn(rng, B):
        u, eps, v = _sample_pair(rng, B, Sigma, t)
        return u, eps if target_kind == "eps" else v

    return train_score_mlp(
        d_in=Sigma.shape[0], d_out=Sigma.shape[0], sample_fn=sample_fn,
        n_steps=n_steps, batch=batch, lr=2e-3, seed=seed,
    )


def per_t_results(Sigma, t_values, n_steps, batch, seed_base=0):
    n_t = len(t_values)
    d = Sigma.shape[0]
    loss_eps = np.empty((n_t, n_steps))
    loss_v = np.empty((n_t, n_steps))
    rel_err_eps = np.empty(n_t)
    rel_err_v = np.empty(n_t)
    for i, t in enumerate(t_values):
        net_eps, lc_eps = _train_per_t(float(t), Sigma, "eps", n_steps, batch, seed_base + 10 * i)
        loss_eps[i] = lc_eps
        J_eps = net_eps.jacobian_at(np.zeros(d))
        A_eps = exact_A_eps(float(t), Sigma)
        rel_err_eps[i] = np.linalg.norm(J_eps - A_eps) / np.linalg.norm(A_eps)

        net_v, lc_v = _train_per_t(float(t), Sigma, "v", n_steps, batch, seed_base + 10 * i + 5)
        loss_v[i] = lc_v
        J_v = net_v.jacobian_at(np.zeros(d))
        A_v = exact_A_v(float(t), Sigma)
        rel_err_v[i] = np.linalg.norm(J_v - A_v) / np.linalg.norm(A_v)

    return loss_eps, loss_v, rel_err_eps, rel_err_v


# ---------- (ii) global conditioned MLP ------------------------------------

def _train_global(Sigma, target_kind, t_min, t_max, n_steps, batch, seed):
    d = Sigma.shape[0]

    def sample_fn(rng, B):
        t_b = rng.uniform(t_min, t_max, size=B)
        a = a_of_t(t_b); b = b_of_t(t_b)
        a_dot = adot_of_t(t_b); b_dot = bdot_of_t(t_b)
        L = np.linalg.cholesky(Sigma)
        x = rng.standard_normal((B, d)) @ L.T
        eps = rng.standard_normal((B, d))
        u = a[:, None] * x + b[:, None] * eps
        target = eps if target_kind == "eps" else (a_dot[:, None] * x + b_dot[:, None] * eps)
        # Concatenate (u, t) as input.
        return np.concatenate([u, t_b[:, None]], axis=1), target

    return train_score_mlp(
        d_in=d + 1, d_out=d, sample_fn=sample_fn,
        n_steps=n_steps, batch=batch, lr=2e-3, seed=seed,
    )


def _global_rel_err_at_t(net, t, Sigma, target_kind):
    """Effective Jacobian wrt u at u=0 for the conditioned model, evaluated at t."""
    d = Sigma.shape[0]
    h = 1e-4
    J = np.zeros((d, d))
    base = np.zeros(d + 1); base[-1] = t
    for i in range(d):
        xp = base.copy(); xp[i] += h
        xm = base.copy(); xm[i] -= h
        J[:, i] = (net.forward(xp)[0] - net.forward(xm)[0]) / (2.0 * h)
    A = exact_A_eps(t, Sigma) if target_kind == "eps" else exact_A_v(t, Sigma)
    return float(np.linalg.norm(J - A) / np.linalg.norm(A))


# ---------- main -----------------------------------------------------------

def main():
    t0 = time.time()

    print("(i) per-t MLPs ...")
    loss_eps, loss_v, rel_err_eps, rel_err_v = per_t_results(
        SIGMA_2D, T_PER_T, N_STEPS, BATCH, seed_base=0,
    )
    print(f"  rel_err_eps median = {np.median(rel_err_eps):.3e}")
    print(f"  rel_err_v   median = {np.median(rel_err_v):.3e}")

    print("(ii) global conditioned MLPs ...")
    net_g_eps, lc_g_eps = _train_global(
        SIGMA_2D, "eps", t_min=0.05, t_max=0.95,
        n_steps=N_STEPS_GLOBAL, batch=BATCH, seed=100,
    )
    net_g_v, lc_g_v = _train_global(
        SIGMA_2D, "v", t_min=0.05, t_max=0.95,
        n_steps=N_STEPS_GLOBAL, batch=BATCH, seed=200,
    )
    # Median rel-err at the mid-grid t values.
    g_rel_err_eps = np.median([
        _global_rel_err_at_t(net_g_eps, float(t), SIGMA_2D, "eps") for t in T_PER_T
    ])
    g_rel_err_v = np.median([
        _global_rel_err_at_t(net_g_v, float(t), SIGMA_2D, "v") for t in T_PER_T
    ])
    print(f"  global eps median rel err = {g_rel_err_eps:.3e}")
    print(f"  global v   median rel err = {g_rel_err_v:.3e}")

    print("(iii) t_min stability sweep (drift-weighted error) ...")
    n_sweep = len(T_MIN_SWEEP)
    sweep_loss_eps = np.empty(n_sweep)
    sweep_loss_v = np.empty(n_sweep)
    sweep_drift_eps = np.empty(n_sweep)
    sweep_drift_v = np.empty(n_sweep)
    rng_eval = np.random.default_rng(999)
    d = SIGMA_2D.shape[0]
    L = np.linalg.cholesky(SIGMA_2D)
    for j, t_min in enumerate(T_MIN_SWEEP):
        net_eps_j, lc_eps = _train_global(
            SIGMA_2D, "eps", t_min=float(t_min), t_max=0.95,
            n_steps=N_STEPS_SWEEP, batch=BATCH, seed=300 + j,
        )
        net_v_j, lc_v = _train_global(
            SIGMA_2D, "v", t_min=float(t_min), t_max=0.95,
            n_steps=N_STEPS_SWEEP, batch=BATCH, seed=400 + j,
        )
        sweep_loss_eps[j] = float(lc_eps[-100:].mean())
        sweep_loss_v[j] = float(lc_v[-100:].mean())

        # Gain-weighted drift error on a fresh evaluation batch:
        # noise gain = |b_dot/b| = 1/t (Eq. 66 envelope), velocity gain = 1.
        t_b = rng_eval.uniform(float(t_min), 0.95, size=DRIFT_EVAL_BATCH)
        a_b = a_of_t(t_b); b_b = b_of_t(t_b)
        a_dot_b = adot_of_t(t_b); b_dot_b = bdot_of_t(t_b)
        x_b = rng_eval.standard_normal((DRIFT_EVAL_BATCH, d)) @ L.T
        eps_b = rng_eval.standard_normal((DRIFT_EVAL_BATCH, d))
        u_b = a_b[:, None] * x_b + b_b[:, None] * eps_b
        target_eps = eps_b
        target_v = a_dot_b[:, None] * x_b + b_dot_b[:, None] * eps_b
        net_in = np.concatenate([u_b, t_b[:, None]], axis=1)

        pred_eps = net_eps_j.forward(net_in)
        gain_eps = np.abs(b_dot_b / b_b)  # 1/t under FM
        sweep_drift_eps[j] = float(
            (gain_eps * np.linalg.norm(pred_eps - target_eps, axis=1)).mean()
        )

        pred_v = net_v_j.forward(net_in)
        sweep_drift_v[j] = float(np.linalg.norm(pred_v - target_v, axis=1).mean())

        print(
            f"  t_min={t_min:.4f}  eps loss={sweep_loss_eps[j]:.3e}  "
            f"v loss={sweep_loss_v[j]:.3e}  "
            f"eps drift={sweep_drift_eps[j]:.3e}  v drift={sweep_drift_v[j]:.3e}"
        )

    payload = dict(
        t_per_t=T_PER_T,
        loss_curves_eps_per_t=loss_eps.astype(np.float32),
        loss_curves_v_per_t=loss_v.astype(np.float32),
        rel_err_eps_per_t=rel_err_eps,
        rel_err_v_per_t=rel_err_v,
        global_loss_eps=lc_g_eps.astype(np.float32),
        global_loss_v=lc_g_v.astype(np.float32),
        global_rel_err_eps=np.asarray(g_rel_err_eps),
        global_rel_err_v=np.asarray(g_rel_err_v),
        t_min_sweep=T_MIN_SWEEP,
        t_min_sweep_loss_eps=sweep_loss_eps,
        t_min_sweep_loss_v=sweep_loss_v,
        t_min_sweep_drift_eps=sweep_drift_eps,
        t_min_sweep_drift_v=sweep_drift_v,
        n_steps=np.asarray(N_STEPS),
        n_steps_global=np.asarray(N_STEPS_GLOBAL),
        n_steps_sweep=np.asarray(N_STEPS_SWEEP),
    )
    out = DATA_DIR / "tiny_mlp_results.npz"
    np.savez(out, **payload)
    elapsed = time.time() - t0
    size_kb = out.stat().st_size / 1024
    print(f"[tiny_mlp_results] {elapsed:.1f}s   {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
