"""Symbolic re-derivation of the paper's stability story.

These functions exercise sympy on the literal Eq. 63 sampler-gain coefficient
to confirm that:

1. Velocity prediction (c, d) = (-1, 1) gives nu(t) identically equal to 1
   under the FM linear schedule -- this is paper Eq. 70.

2. Noise prediction (c, d) = (0, 1) gives a closed-form nu(t) = 1/(1-t) and
   a near-manifold envelope b_dot/b = 1/t. We compute the ratio
   nu(t) / (b_dot/b) symbolically and report its limit at t -> 0+; under the
   FM linear schedule this ratio is t/(1-t), with limit 0 (not 1) at the
   manifold-side endpoint.

The "ratio_limit = 0" outcome is itself the diagnostic: the literal Eq. 63
ν is bounded near t = 0 for FM, so the divergent stability story attributed
to noise prediction near the manifold lives in the |b_dot/b| envelope (paper
Eq. 66 prefactor), not in ν itself. See docs/implementation_notes.md and the
notebook stability section for the fuller version of this point.
"""

from __future__ import annotations

import sympy as sp


def validate_velocity_gain() -> dict:
    """Symbolic Eq. 63 with (c, d) = (-1, 1), simplified under FM linear."""
    a, b, t = sp.symbols("a b t", positive=True, real=True)
    a_dot = sp.symbols("adot", real=True)
    b_dot = sp.symbols("bdot", real=True)

    # Eq. 63: mu, nu given the velocity parameterization (c, d) = (-1, 1)
    c = sp.Integer(-1)
    d = sp.Integer(1)
    det = a * d - b * c                  # = a + b
    mu = (a_dot * d - b_dot * c) / det
    nu = (b_dot * a - a_dot * b) / det

    # Linear FM schedule: a = 1 - t, b = t, a_dot = -1, b_dot = 1
    schedule_subs = {a: 1 - t, b: t, a_dot: -1, b_dot: 1}

    mu_lin = sp.simplify(mu.subs(schedule_subs))
    nu_lin = sp.simplify(nu.subs(schedule_subs))

    return {
        "mu_general": mu,
        "nu_general": nu,
        "mu_linear_FM": mu_lin,
        "nu_linear_FM": nu_lin,
    }


def validate_noise_gain_divergence() -> dict:
    """Symbolic Eq. 63 with (c, d) = (0, 1), and the ratio to the |b_dot/b| envelope.

    Under the FM linear schedule the ratio simplifies to t/(1 - t) and its
    limit at t -> 0+ is 0. That is, ν is bounded near the manifold (a
    peculiarity of FM); the divergent factor that drives the noise-
    prediction Drift Perturbation Error sits in b_dot/b = 1/t, not in ν.
    """
    a, b, t = sp.symbols("a b t", positive=True, real=True)
    a_dot, b_dot = sp.symbols("adot bdot", real=True)
    c = sp.Integer(0)
    d = sp.Integer(1)
    det = a * d - b * c                  # = a
    nu = (b_dot * a - a_dot * b) / det
    schedule_subs = {a: 1 - t, b: t, a_dot: -1, b_dot: 1}
    nu_lin = sp.simplify(nu.subs(schedule_subs))
    envelope = sp.simplify(b_dot / b).subs(schedule_subs)
    ratio_at_small_t = sp.limit(nu_lin / envelope, t, 0, "+")
    return {
        "nu_noise_general": nu,
        "nu_noise_linear_FM": nu_lin,
        "envelope_linear_FM": envelope,
        "ratio_limit": ratio_at_small_t,
    }
