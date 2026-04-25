import numpy as np
import pytest
from scipy.stats import multivariate_normal

from src.exact_affine import (
    E_marg,
    conditional_mean_eps_given_u_t,
    conditional_mean_x_given_u_t,
    grad_E_marg_analytic,
    grad_E_marg_numeric,
    log_p_u_given_t,
    marginal_cov,
    posterior_t_given_u,
    velocity_target,
)
from src.schedules import a_of_t, b_of_t


SIGMA = np.diag([2.0, 0.5])
T_GRID = np.linspace(0.05, 0.95, 91)
PRIOR = np.full_like(T_GRID, 1.0 / (T_GRID[-1] - T_GRID[0]))


@pytest.mark.parametrize("t", [0.1, 0.5, 0.9])
def test_marginal_cov_symmetric_psd(t):
    M = marginal_cov(t, SIGMA)
    assert np.allclose(M, M.T, atol=1e-12)
    eig = np.linalg.eigvalsh(M)
    assert (eig > 0).all()


@pytest.mark.parametrize("t", [0.1, 0.4, 0.8])
def test_log_p_u_given_t_matches_scipy(t):
    rng = np.random.default_rng(0)
    M = marginal_cov(t, SIGMA)
    rv = multivariate_normal(mean=np.zeros(2), cov=M)
    for _ in range(5):
        u = rng.multivariate_normal(np.zeros(2), M)
        ours = float(log_p_u_given_t(u, t, SIGMA))
        ref = float(rv.logpdf(u))
        assert np.isclose(ours, ref, atol=1e-10), f"t={t}: {ours} vs {ref}"


def test_posterior_sums_to_one():
    rng = np.random.default_rng(1)
    for _ in range(5):
        u = rng.standard_normal(2)
        post = posterior_t_given_u(u, T_GRID, PRIOR, SIGMA)
        assert np.isclose(post.sum(), 1.0, atol=1e-10)


def test_grad_E_marg_analytic_matches_numeric():
    rng = np.random.default_rng(2)
    for _ in range(20):
        u = rng.standard_normal(2) * 0.5
        ga = grad_E_marg_analytic(u, T_GRID, PRIOR, SIGMA)
        gn = grad_E_marg_numeric(u, T_GRID, PRIOR, SIGMA, h=1e-4)
        assert np.allclose(ga, gn, atol=1e-4), f"u={u}: analytic={ga} numeric={gn}"


def test_conditional_mean_x_b_to_zero():
    """As b -> 0, M -> a^2 Sigma so E[x|u,t] = a Sigma (a^2 Sigma)^{-1} u = u/a."""
    u = np.array([1.3, -0.7])
    t_small = 1e-3
    a = float(a_of_t(t_small))
    expected = u / a
    got = conditional_mean_x_given_u_t(u, t_small, SIGMA)
    assert np.allclose(got, expected, atol=1e-3)


def test_conditional_mean_eps_a_to_zero():
    """As a -> 0, M -> b^2 I so E[eps|u,t] = b (b^2 I)^{-1} u = u/b."""
    u = np.array([1.3, -0.7])
    t_big = 1.0 - 1e-3
    b = float(b_of_t(t_big))
    expected = u / b
    got = conditional_mean_eps_given_u_t(u, t_big, SIGMA)
    assert np.allclose(got, expected, atol=1e-3)


def test_velocity_target_linear_in_u():
    """Conditional means are linear in u, so velocity_target is too."""
    rng = np.random.default_rng(3)
    t = 0.4
    u1 = rng.standard_normal(2)
    u2 = rng.standard_normal(2)
    alpha, beta = 1.7, -0.3
    v1 = velocity_target(u1, t, SIGMA)
    v2 = velocity_target(u2, t, SIGMA)
    v_combo = velocity_target(alpha * u1 + beta * u2, t, SIGMA)
    assert np.allclose(v_combo, alpha * v1 + beta * v2, atol=1e-10)
