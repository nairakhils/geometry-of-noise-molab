"""Affine noising schedule.

The paper (arXiv:2602.18428, Eq. 2) parameterizes the forward process as

    u_t = a(t) x + b(t) eps,    eps ~ N(0, I),  t in [0, 1].

We use the FM-linear specialization (Table 1, Flow Matching row, p. 4):

    a(t) = 1 - t,
    b(t) = t.

This is the schedule against which the closed-form near-manifold expressions
in Appendix E (Eqs. 59, 60, 67-70) are written. It also produces a constant
ad - bc = 1 for the velocity parameterization (c = -1, d = 1), giving the
bounded sampler-gain v(t) = 1 of Eq. 70. See docs/paper_summary.md (i).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def a_of_t(t: ArrayLike) -> NDArray[np.float64]:
    """Signal coefficient a(t) = 1 - t."""
    t = np.asarray(t, dtype=np.float64)
    return 1.0 - t


def b_of_t(t: ArrayLike) -> NDArray[np.float64]:
    """Noise coefficient b(t) = t."""
    t = np.asarray(t, dtype=np.float64)
    return t


def adot_of_t(t: ArrayLike) -> NDArray[np.float64]:
    """da/dt = -1."""
    t = np.asarray(t, dtype=np.float64)
    return -np.ones_like(t)


def bdot_of_t(t: ArrayLike) -> NDArray[np.float64]:
    """db/dt = 1."""
    t = np.asarray(t, dtype=np.float64)
    return np.ones_like(t)


# DDPM/EDM-style aliases. For the FM-linear schedule we adopt here,
# alpha_t = a(t) and sigma_t = b(t) coincide with paper's notation in Table 1.
alpha_of_t = a_of_t
sigma_of_t = b_of_t
