"""Closed-form sanity check: OLS recovers the analytic Wiener matrix.

For x ~ N(0, Sigma) and eps ~ N(0, I) with u_t = a(t) x + b(t) eps, the
optimal noise-prediction estimator is linear in u:

    E[eps | u, t] = b(t) M(t)^{-1} u                       (matrix A_eps(t))

The Salimans-Ho velocity v = a(t) eps - b(t) x has

    E[v | u, t] = a(t) b(t) (I - Sigma) M(t)^{-1} u        (matrix A_v(t))

(by the joint-Gaussian formula Cov(v, u) M^{-1} u; cross-covariance
Cov(v, u) = a*b*(I - Sigma)).

If the forward process, the schedule, and the conditional-mean derivation in
src/exact_affine.py all agree with each other, OLS on enough (u, target)
pairs converges to A_exact up to sampling noise of order 1/sqrt(N). This
script records the empirical agreement.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schedules import a_of_t, b_of_t


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 0
SIGMA_2D = np.diag([2.0, 0.5])
T_VALUES = np.linspace(0.05, 0.95, 10)
N_SAMPLES = 200_000


def exact_A_eps(t: float, Sigma: np.ndarray) -> np.ndarray:
    a = float(a_of_t(t))
    b = float(b_of_t(t))
    d = Sigma.shape[0]
    M = a * a * Sigma + b * b * np.eye(d)
    return b * np.linalg.inv(M)


def exact_A_v(t: float, Sigma: np.ndarray) -> np.ndarray:
    a = float(a_of_t(t))
    b = float(b_of_t(t))
    d = Sigma.shape[0]
    M = a * a * Sigma + b * b * np.eye(d)
    return a * b * (np.eye(d) - Sigma) @ np.linalg.inv(M)


def fit_one(
    t: float, Sigma: np.ndarray, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    a = float(a_of_t(t))
    b = float(b_of_t(t))
    d = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    x = rng.standard_normal((n_samples, d)) @ L.T
    eps = rng.standard_normal((n_samples, d))
    u = a * x + b * eps
    v = a * eps - b * x

    coef_eps, *_ = np.linalg.lstsq(u, eps, rcond=None)
    coef_v, *_ = np.linalg.lstsq(u, v, rcond=None)
    return coef_eps.T, coef_v.T


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    d = SIGMA_2D.shape[0]
    n_t = len(T_VALUES)

    A_learned_eps = np.empty((n_t, d, d))
    A_exact_eps = np.empty_like(A_learned_eps)
    A_learned_v = np.empty_like(A_learned_eps)
    A_exact_v = np.empty_like(A_learned_eps)

    for i, t in enumerate(T_VALUES):
        A_le, A_lv = fit_one(t, SIGMA_2D, N_SAMPLES, rng)
        A_learned_eps[i] = A_le
        A_learned_v[i] = A_lv
        A_exact_eps[i] = exact_A_eps(t, SIGMA_2D)
        A_exact_v[i] = exact_A_v(t, SIGMA_2D)

    rel_err_eps = np.linalg.norm(A_learned_eps - A_exact_eps, axis=(1, 2)) / np.maximum(
        np.linalg.norm(A_exact_eps, axis=(1, 2)), 1e-12
    )
    rel_err_v = np.linalg.norm(A_learned_v - A_exact_v, axis=(1, 2)) / np.maximum(
        np.linalg.norm(A_exact_v, axis=(1, 2)), 1e-12
    )

    payload = dict(
        t_values=T_VALUES,
        A_learned_eps=A_learned_eps,
        A_exact_eps=A_exact_eps,
        A_learned_v=A_learned_v,
        A_exact_v=A_exact_v,
        rel_err_eps=rel_err_eps,
        rel_err_v=rel_err_v,
        n_samples=np.asarray(N_SAMPLES),
    )
    np.savez(DATA_DIR / "linear_score_fit.npz", **payload)

    elapsed = time.time() - t0
    size_mb = sum(np.asarray(v).nbytes for v in payload.values()) / 1e6
    print(f"[linear_score_fit     ] {elapsed:6.2f}s   {size_mb:7.3f} MB   N={N_SAMPLES} per t")
    print(f"  eps rel err   min={rel_err_eps.min():.2e}   median={np.median(rel_err_eps):.2e}   max={rel_err_eps.max():.2e}")
    print(f"  v   rel err   min={rel_err_v.min():.2e}   median={np.median(rel_err_v):.2e}   max={rel_err_v.max():.2e}")


if __name__ == "__main__":
    main()
