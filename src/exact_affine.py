"""Exact closed-form affine-noising quantities for x ~ N(0, Sigma).

For x ~ N(0, Sigma) and eps ~ N(0, I) independent in R^d, the noisy
observation u_t = a(t) x + b(t) eps satisfies

    u_t | t ~ N(0, M(t)),    M(t) = a(t)^2 Sigma + b(t)^2 I.

Conditional posteriors (joint Gaussian + Schur complement):

    E[x   | u, t] = a(t)  Sigma  M(t)^{-1} u,
    E[eps | u, t] = b(t)         M(t)^{-1} u.

Marginal density and energy via discretization on a t-grid (paper Eqs. 1, 8):

    log p(u)        = logsumexp_i [ log p(u | t_i) + log w_i ],
    E_marg(u)       = - log p(u),
    grad E_marg(u)  = E_{t | u}[ M(t)^{-1} u ]                (Eq. 11),
    p(t_i | u)      proportional to p(u | t_i) * w_i.

Here w_i = prior_t[i] * dt_i are the discretized integration weights.

All u arguments accept arbitrary leading axes; the trailing axis is the d-axis.
Sigma is shape (d, d), symmetric positive definite. Matrix arithmetic is done
in Sigma's eigenbasis once per call to keep everything diagonal and avoid
explicit per-coordinate loops.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .schedules import a_of_t, b_of_t, adot_of_t, bdot_of_t


_LOG2PI = float(np.log(2.0 * np.pi))


def _eig_sigma(Sigma: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (D, V) so Sigma = V diag(D) V^T, with V orthogonal."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    D, V = np.linalg.eigh(Sigma)
    return D, V


def _grid_weights(t_grid: NDArray[np.float64], prior_t: NDArray[np.float64]) -> NDArray[np.float64]:
    """w_i = prior_t[i] * (local spacing). Trapezoidal on uniform grids."""
    dt = np.gradient(t_grid)
    return prior_t * dt


def marginal_cov(t: ArrayLike, Sigma: ArrayLike) -> NDArray[np.float64]:
    """M(t) = a(t)^2 Sigma + b(t)^2 I_d. Returns (..., d, d)."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    d = Sigma.shape[-1]
    a = a_of_t(t)
    b = b_of_t(t)
    a2 = (a * a)[..., None, None]
    b2 = (b * b)[..., None, None]
    return a2 * Sigma + b2 * np.eye(d)


def log_p_u_given_t(u: ArrayLike, t: ArrayLike, Sigma: ArrayLike) -> NDArray[np.float64]:
    """log N(u; 0, M(t)). u shape (..., d), t scalar."""
    u = np.asarray(u, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    d = Sigma.shape[-1]
    D, V = _eig_sigma(Sigma)
    a = float(np.asarray(a_of_t(t)))
    b = float(np.asarray(b_of_t(t)))
    m_eig = a * a * D + b * b
    y = u @ V
    quad = np.einsum("...j,j->...", y * y, 1.0 / m_eig)
    log_det = float(np.sum(np.log(m_eig)))
    return -0.5 * quad - 0.5 * log_det - 0.5 * d * _LOG2PI


def _log_p_u_given_t_on_grid(
    u: NDArray[np.float64],
    t_grid: NDArray[np.float64],
    Sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """log p(u | t_i) for every t_i in t_grid. Output shape (..., T)."""
    d = Sigma.shape[-1]
    D, V = _eig_sigma(Sigma)
    a = a_of_t(t_grid)
    b = b_of_t(t_grid)
    m_eig = (a * a)[:, None] * D[None, :] + (b * b)[:, None]
    y = u @ V
    quad = np.einsum("...j,tj->...t", y * y, 1.0 / m_eig)
    log_det = np.sum(np.log(m_eig), axis=-1)
    return -0.5 * quad - 0.5 * log_det - 0.5 * d * _LOG2PI


def log_p_u(
    u: ArrayLike,
    t_grid: ArrayLike,
    prior_t: ArrayLike,
    Sigma: ArrayLike,
) -> NDArray[np.float64]:
    """log of the marginal mixture int p(u|t) p(t) dt, by logsumexp on the grid."""
    u = np.asarray(u, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    prior_t = np.asarray(prior_t, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)

    log_w = np.log(np.clip(_grid_weights(t_grid, prior_t), 1e-300, None))
    log_pu_t = _log_p_u_given_t_on_grid(u, t_grid, Sigma)
    return logsumexp(log_pu_t + log_w, axis=-1)


def E_marg(
    u: ArrayLike,
    t_grid: ArrayLike,
    prior_t: ArrayLike,
    Sigma: ArrayLike,
) -> NDArray[np.float64]:
    """Marginal energy E_marg(u) = -log p(u), discretized on t_grid."""
    return -log_p_u(u, t_grid, prior_t, Sigma)


def posterior_t_given_u(
    u: ArrayLike,
    t_grid: ArrayLike,
    prior_t: ArrayLike,
    Sigma: ArrayLike,
) -> NDArray[np.float64]:
    """p(t_i | u) on the discrete grid, normalized to sum to 1 along the t axis."""
    u = np.asarray(u, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    prior_t = np.asarray(prior_t, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)

    log_w = np.log(np.clip(_grid_weights(t_grid, prior_t), 1e-300, None))
    log_joint = _log_p_u_given_t_on_grid(u, t_grid, Sigma) + log_w
    return np.exp(log_joint - logsumexp(log_joint, axis=-1, keepdims=True))


def grad_E_marg_analytic(
    u: ArrayLike,
    t_grid: ArrayLike,
    prior_t: ArrayLike,
    Sigma: ArrayLike,
) -> NDArray[np.float64]:
    """grad E_marg(u) = E_{t|u}[ M(t)^{-1} u ]  (Tweedie-style, paper Eq. 11)."""
    u = np.asarray(u, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    D, V = _eig_sigma(Sigma)

    posterior = posterior_t_given_u(u, t_grid, prior_t, Sigma)
    a = a_of_t(np.asarray(t_grid, dtype=np.float64))
    b = b_of_t(np.asarray(t_grid, dtype=np.float64))
    m_eig = (a * a)[:, None] * D[None, :] + (b * b)[:, None]

    y = u @ V
    y_over_m = y[..., None, :] / m_eig
    avg_y = np.einsum("...t,...td->...d", posterior, y_over_m)
    return avg_y @ V.T


def grad_E_marg_numeric(
    u: ArrayLike,
    t_grid: ArrayLike,
    prior_t: ArrayLike,
    Sigma: ArrayLike,
    h: float = 1e-4,
) -> NDArray[np.float64]:
    """Central-difference gradient of E_marg, vectorized over coordinates."""
    u = np.asarray(u, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    d = Sigma.shape[-1]
    eye_h = np.eye(d) * h
    u_plus = u[..., None, :] + eye_h
    u_minus = u[..., None, :] - eye_h
    return (E_marg(u_plus, t_grid, prior_t, Sigma) - E_marg(u_minus, t_grid, prior_t, Sigma)) / (2.0 * h)


def conditional_mean_x_given_u_t(u: ArrayLike, t: ArrayLike, Sigma: ArrayLike) -> NDArray[np.float64]:
    """E[x | u, t] = a(t) Sigma M(t)^{-1} u."""
    u = np.asarray(u, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    a = float(np.asarray(a_of_t(t)))
    b = float(np.asarray(b_of_t(t)))
    D, V = _eig_sigma(Sigma)
    m_eig = a * a * D + b * b
    y = u @ V
    return a * ((D * y / m_eig) @ V.T)


def conditional_mean_eps_given_u_t(u: ArrayLike, t: ArrayLike, Sigma: ArrayLike) -> NDArray[np.float64]:
    """E[eps | u, t] = b(t) M(t)^{-1} u."""
    u = np.asarray(u, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    a = float(np.asarray(a_of_t(t)))
    b = float(np.asarray(b_of_t(t)))
    D, V = _eig_sigma(Sigma)
    m_eig = a * a * D + b * b
    y = u @ V
    return b * ((y / m_eig) @ V.T)


def velocity_target_paper(u: ArrayLike, t: ArrayLike, Sigma: ArrayLike) -> NDArray[np.float64]:
    """E[v | u, t] for v = du/dt under the affine schedule (paper Eq. 61).

    With u_t = a(t) x + b(t) eps, the time derivative is
    v = a_dot(t) x + b_dot(t) eps. Taking the posterior mean given u, t:

        E[v | u, t] = a_dot(t) * E[x | u, t] + b_dot(t) * E[eps | u, t].

    For FM-linear (a_dot = -1, b_dot = 1) this collapses to E[eps - x | u, t].
    This is the velocity object the paper's stability theorem is about.
    """
    a_dot = float(np.asarray(adot_of_t(t)))
    b_dot = float(np.asarray(bdot_of_t(t)))
    x_hat = conditional_mean_x_given_u_t(u, t, Sigma)
    eps_hat = conditional_mean_eps_given_u_t(u, t, Sigma)
    return a_dot * x_hat + b_dot * eps_hat


def velocity_target_SH(u: ArrayLike, t: ArrayLike, Sigma: ArrayLike) -> NDArray[np.float64]:
    """E[v_SH | u, t] for v_SH = alpha(t) eps - sigma(t) x (Salimans & Ho, ICLR 2022).

    Kept for reference. NOT what the paper's stability theorem analyzes; that
    role is played by `velocity_target_paper`. With alpha = a, sigma = b
    (FM-linear), this gives E[v_SH] = a * E[eps] - b * E[x].
    """
    a = float(np.asarray(a_of_t(t)))
    b = float(np.asarray(b_of_t(t)))
    x_hat = conditional_mean_x_given_u_t(u, t, Sigma)
    eps_hat = conditional_mean_eps_given_u_t(u, t, Sigma)
    return a * eps_hat - b * x_hat


# Default velocity convention: the paper's v = du/dt.
velocity_target = velocity_target_paper


def effective_gain_noise_pred(t: ArrayLike) -> NDArray[np.float64]:
    """|b_dot(t) / b(t)|: the 1/b-type prefactor in front of the Jensen Gap (Eq. 66).

    Diverges as t -> 0 for any polynomial b(t) ~ t^k with k > 0. For FM
    (b = t) this is 1/t = 1/b. The Drift Perturbation Error for noise
    prediction inherits this divergence regardless of schedule.
    """
    t = np.asarray(t, dtype=np.float64)
    b = b_of_t(t)
    bdot = bdot_of_t(t)
    return np.abs(bdot / np.where(b > 0, b, np.nan))


def effective_gain_velocity_pred(t: ArrayLike) -> NDArray[np.float64]:
    """nu(t) = 1 for velocity prediction (Eq. 70: ad - bc = 1 makes the gain unity)."""
    t = np.asarray(t, dtype=np.float64)
    return np.ones_like(t)


def jensen_gap(
    u: ArrayLike,
    t_grid: ArrayLike,
    prior_t: ArrayLike,
    Sigma: ArrayLike,
    param: str = "eps",
) -> NDArray[np.float64]:
    """Posterior-uncertainty gap.

    param = 'eps':
        AM-HM convexity gap of b(.) under the posterior:
            E_{tau|u}[b(tau)] * E_{tau|u}[1/b(tau)] - 1.
        Non-negative by Jensen, zero iff p(tau|u) is a Dirac. This is the
        bracketed Jensen Gap of paper Eq. 66; multiplying by the divergent
        prefactor |b_dot/b| produces the noise-prediction Drift Perturbation
        Error.
    param = 'v':
        Posterior dispersion of the velocity target,
            E_{tau|u}[ ||v_tau*(u) - E_{tau|u}[v_tau*(u)]||^2 ].
        Bounded near the data manifold; the velocity-prediction analog of
        the Jensen gap.
    """
    u = np.asarray(u, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    posterior = posterior_t_given_u(u, t_grid, prior_t, Sigma)

    if param == "eps":
        b_vals = b_of_t(t_grid)
        b_safe = np.clip(b_vals, 1e-12, None)
        E_b = np.einsum("...t,t->...", posterior, b_vals)
        E_inv_b = np.einsum("...t,t->...", posterior, 1.0 / b_safe)
        return np.abs(E_b * E_inv_b - 1.0)

    if param == "v":
        D, V = _eig_sigma(Sigma)
        a = a_of_t(t_grid)
        b = b_of_t(t_grid)
        a_dot = adot_of_t(t_grid)
        b_dot = bdot_of_t(t_grid)
        m_eig = (a * a)[:, None] * D[None, :] + (b * b)[:, None]
        y = u @ V
        # paper convention: E[v|u] = a_dot E[x|u] + b_dot E[eps|u]
        # in eigenbasis: per-mode coefficient = (a_dot * a * D + b_dot * b) / m
        v_coef = (a_dot * a)[:, None] * D[None, :] + (b_dot * b)[:, None]
        v_y = v_coef * (y[..., None, :] / m_eig)
        v_orig = np.einsum("...td,jd->...tj", v_y, V)
        v_mean = np.einsum("...t,...td->...d", posterior, v_orig)
        diff = v_orig - v_mean[..., None, :]
        sq = np.einsum("...td,...td->...t", diff, diff)
        return np.einsum("...t,...t->...", posterior, sq)

    raise ValueError(f"param must be 'eps' or 'v', got {param!r}")
