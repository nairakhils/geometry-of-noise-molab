"""Closed-form sanity check: OLS recovers the analytic Wiener matrix.

For x ~ N(0, Sigma) and eps ~ N(0, I) with u_t = a(t) x + b(t) eps, the
optimal noise-prediction estimator is linear in u:

    E[eps | u, t] = b(t) M(t)^{-1} u                       (matrix A_eps(t))

The paper's velocity v = du/dt = a_dot(t) x + b_dot(t) eps has

    E[v | u, t] = (a_dot(t) a(t) Sigma + b_dot(t) b(t) I) M(t)^{-1} u
                                                            (matrix A_v(t))

(direct combination of the conditional means, since the joint of (x, eps, u)
is Gaussian and the linear map is just the differentiated forward process).
On the FM linear schedule (a_dot=-1, b_dot=1) this simplifies to
A_v = (b I - a Sigma) M^{-1} = (t I - (1-t) Sigma) M^{-1}.

This script sweeps the sample size N across {1e3, 1e4, 1e5, 1e6}. The
relative Frobenius error of the OLS estimate vs the closed form is
expected to scale as 1/sqrt(N) per Gaussian linear-regression theory, so
each decade in N should shift the curve by sqrt(10) on a semilog plot.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schedules import a_of_t, adot_of_t, b_of_t, bdot_of_t


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 0
SIGMA_2D = np.diag([2.0, 0.5])
T_VALUES = np.linspace(0.05, 0.95, 10)
N_SAMPLES_LIST = [1_000, 10_000, 100_000, 1_000_000]
RUNTIME_BUDGET_S = 90.0


def exact_A_eps(t: float, Sigma: np.ndarray) -> np.ndarray:
    a = float(a_of_t(t))
    b = float(b_of_t(t))
    d = Sigma.shape[0]
    M = a * a * Sigma + b * b * np.eye(d)
    return b * np.linalg.inv(M)


def exact_A_v(t: float, Sigma: np.ndarray) -> np.ndarray:
    """A_v(t) = (a_dot * a * Sigma + b_dot * b * I) M(t)^{-1}  (paper convention)."""
    a = float(a_of_t(t))
    b = float(b_of_t(t))
    a_dot = float(adot_of_t(t))
    b_dot = float(bdot_of_t(t))
    d = Sigma.shape[0]
    M = a * a * Sigma + b * b * np.eye(d)
    return (a_dot * a * Sigma + b_dot * b * np.eye(d)) @ np.linalg.inv(M)


def fit_one(
    t: float, Sigma: np.ndarray, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    a = float(a_of_t(t))
    b = float(b_of_t(t))
    a_dot = float(adot_of_t(t))
    b_dot = float(bdot_of_t(t))
    d = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    x = rng.standard_normal((n_samples, d)) @ L.T
    eps = rng.standard_normal((n_samples, d))
    u = a * x + b * eps
    v = a_dot * x + b_dot * eps  # paper convention v = du/dt

    coef_eps, *_ = np.linalg.lstsq(u, eps, rcond=None)
    coef_v, *_ = np.linalg.lstsq(u, v, rcond=None)
    return coef_eps.T, coef_v.T


def _sweep_one_N(N: int, Sigma: np.ndarray, rng: np.random.Generator):
    """Return (rel_err_eps[t], rel_err_v[t]) at sample size N."""
    n_t = len(T_VALUES)
    d = Sigma.shape[0]
    rel_eps = np.empty(n_t)
    rel_v = np.empty(n_t)
    for i, t in enumerate(T_VALUES):
        A_le, A_lv = fit_one(t, Sigma, N, rng)
        A_ee = exact_A_eps(t, Sigma)
        A_ev = exact_A_v(t, Sigma)
        rel_eps[i] = np.linalg.norm(A_le - A_ee) / max(np.linalg.norm(A_ee), 1e-12)
        rel_v[i] = np.linalg.norm(A_lv - A_ev) / max(np.linalg.norm(A_ev), 1e-12)
    return rel_eps, rel_v


def main() -> None:
    t_total = time.time()
    rng = np.random.default_rng(SEED)
    n_t = len(T_VALUES)
    n_list = list(N_SAMPLES_LIST)

    rel_err_eps_grid = np.empty((len(n_list), n_t))
    rel_err_v_grid = np.empty((len(n_list), n_t))
    per_N_runtime = np.empty(len(n_list))

    used_n = []
    for j, N in enumerate(n_list):
        t_n = time.time()
        if time.time() - t_total + 1.5 * (per_N_runtime[j - 1] if j > 0 else 0.0) > RUNTIME_BUDGET_S:
            print(f"[linear_score_fit     ] dropping N={N} to stay under {RUNTIME_BUDGET_S:.0f}s budget")
            break
        rel_e, rel_v = _sweep_one_N(N, SIGMA_2D, rng)
        rel_err_eps_grid[j] = rel_e
        rel_err_v_grid[j] = rel_v
        per_N_runtime[j] = time.time() - t_n
        used_n.append(N)
        print(
            f"  N={N:>9d}   {per_N_runtime[j]:5.2f}s   "
            f"eps median={np.median(rel_e):.2e}   v median={np.median(rel_v):.2e}"
        )

    used_n_arr = np.asarray(used_n, dtype=np.int64)
    payload = dict(
        t_values=T_VALUES,
        n_samples_list=used_n_arr,
        rel_err_eps_grid=rel_err_eps_grid[: len(used_n)],
        rel_err_v_grid=rel_err_v_grid[: len(used_n)],
    )
    np.savez(DATA_DIR / "linear_score_fit.npz", **payload)

    elapsed = time.time() - t_total
    size_mb = sum(np.asarray(v).nbytes for v in payload.values()) / 1e6
    print(f"[linear_score_fit     ] {elapsed:6.2f}s   {size_mb:7.3f} MB   N sweep over {used_n}")


if __name__ == "__main__":
    main()
