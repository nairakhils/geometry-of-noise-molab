"""Symbolic re-derivation of the paper's stability identities.

Velocity prediction (c, d) = (-1, 1) under FM linear: nu(t) simplifies
identically to 1 (paper Eq. 70). This is the credibility lock.

Noise prediction (c, d) = (0, 1) under FM linear: nu(t) = 1/(1-t),
envelope b_dot/b = 1/t, ratio nu/envelope = t/(1-t) -> 0 at t -> 0+.
Per Phase 11's notebook prose, the literal ν for noise prediction is
bounded near the manifold under FM linear; the divergence that drives
the Drift Perturbation Error lives in the envelope b_dot/b, not in ν.
The ratio_limit = 0 outcome is the symbolic confirmation of that point;
the user's spec named '1' as the expected ratio_limit, but a tight
envelope-vs-nu match would require a schedule like VP-SDE where ν
itself diverges. We assert what the symbolic computation actually
returns.
"""

import sympy as sp

from src.sympy_validation import (
    validate_noise_gain_divergence,
    validate_velocity_gain,
)


def test_velocity_nu_linear_FM_is_one():
    res = validate_velocity_gain()
    assert sp.simplify(res["nu_linear_FM"] - 1) == 0, (
        f"expected nu_linear_FM = 1 (paper Eq. 70), got {res['nu_linear_FM']}"
    )


def test_noise_ratio_limit_is_zero_under_FM():
    """Under FM linear nu_noise(t) is bounded near t=0 while the envelope
    diverges, so ratio nu/envelope -> 0. (User spec wrote '1'; that holds for
    schedules where nu itself diverges, e.g. VP-SDE, not for FM linear.)
    """
    res = validate_noise_gain_divergence()
    assert res["ratio_limit"] == 0, (
        f"expected ratio_limit = 0 under FM linear, got {res['ratio_limit']}"
    )
    # And confirm the closed forms
    t = sp.symbols("t", positive=True, real=True)
    assert sp.simplify(res["nu_noise_linear_FM"] - 1 / (1 - t)) == 0
    assert sp.simplify(res["envelope_linear_FM"] - 1 / t) == 0
