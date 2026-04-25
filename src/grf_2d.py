"""2D Gaussian random fields with isotropic power spectrum P(k) ~ k^{-n_s}.

This is the substrate for the Fourier-mode shrinkage extension (see
docs/paper_summary.md, "Notebook scope" / "Extension"). The covariance is
diagonal in the Fourier basis; each Fourier mode independently realizes the
affine-noising geometry of the paper.

Conventions:
- Use np.fft.fftfreq(N) * N to get integer wave numbers in {-N/2, ..., N/2-1}.
- kmag is the radial wavenumber in those units; the k=0 mode is suppressed
  (zero-mean field) to avoid the pole in P(k) ~ k^{-n_s}.
- Fields are real by Hermitian symmetry of the Fourier draw.
"""

import numpy as np
from numpy.typing import NDArray

from .schedules import a_of_t, adot_of_t, b_of_t, bdot_of_t


def build_k_grid(N: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (kx, ky, kmag) with shapes (N, N), in integer-wavenumber units."""
    k1d = np.fft.fftfreq(N) * N
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky)
    return kx, ky, kmag


def build_grf_covariance_diag(N: int, n_s: float) -> NDArray[np.float64]:
    """Diagonal of the GRF covariance in Fourier basis: sigma_k^2 = k^{-n_s} for k>0, 0 at k=0."""
    _, _, kmag = build_k_grid(N)
    sigma2 = np.zeros_like(kmag)
    nz = kmag > 0
    sigma2[nz] = kmag[nz] ** (-n_s)
    return sigma2


def sample_grf_batch(
    B: int,
    N: int,
    n_s: float,
    rng: np.random.Generator,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Draw B real GRFs of size NxN by FFT-domain filtering.

    With `normalize=True` (default), each field is rescaled to unit empirical
    real-space variance. That is convenient for visualization and for the
    PSD-slope check, but it distorts the per-mode Fourier variance relative
    to the textbook `sigma_k^2 = k^{-n_s}` and is therefore incompatible
    with the autonomous reverse flow in `exact_reverse_trajectory` (which
    derives `c_k(t)` from the textbook per-mode variance).

    For the flow demo, pass `normalize=False`: the field then has per-mode
    `E[|fft(x)(k)|^2] = N^2 * sigma_k^2` and the reverse step is consistent.
    """
    sigma2 = build_grf_covariance_diag(N, n_s)
    white = rng.standard_normal((B, N, N))
    W = np.fft.fft2(white)
    F = W * np.sqrt(sigma2)[None, :, :]
    field = np.fft.ifft2(F).real
    if normalize:
        var = field.var(axis=(1, 2), keepdims=True)
        field = field / np.sqrt(np.where(var > 0, var, 1.0))
    return field


def measure_psd_radial(
    field: NDArray[np.float64], n_bins: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Radially-averaged periodogram on log-spaced bins. field shape (..., N, N)."""
    field = np.asarray(field, dtype=np.float64)
    N = field.shape[-1]
    F = np.fft.fft2(field) / N
    power = (F.conj() * F).real
    power_mean = power.reshape(-1, N, N).mean(axis=0)
    _, _, kmag = build_k_grid(N)

    if n_bins is None:
        n_bins = max(8, int(np.log2(N // 2)) * 4 + 1)
    edges = np.logspace(0.0, np.log10(N / 2.0), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    bin_idx = np.digitize(kmag.ravel(), edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    counts = np.bincount(bin_idx[valid], minlength=n_bins)
    sums = np.bincount(bin_idx[valid], weights=power_mean.ravel()[valid], minlength=n_bins)
    P_k = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return centers, P_k


def half_power_cutoff(t, n_s: float, schedule=None) -> NDArray[np.float64]:
    """Radial wavenumber k_c at which the per-mode signal-fraction W drops to 1/2.

    With W(k, t) = a^2 sigma_k^2 / (a^2 sigma_k^2 + b^2) and
    sigma_k^2 = k^{-n_s}, setting W = 1/2 yields a^2 sigma_k^2 = b^2, hence

        k_c(t, n_s) = (a(t) / b(t)) ** (2 / n_s).

    Diverges to +inf as b(t) -> 0 (every mode preserved at zero noise) and
    falls to 0 as a(t) -> 0 (no mode preserved when the signal is gone).
    """
    t = np.asarray(t, dtype=np.float64)
    if schedule is None:
        a = a_of_t(t)
        b = b_of_t(t)
    else:
        a, b = schedule(t)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    safe_b = np.where(b > 0, b, np.finfo(np.float64).tiny)
    return np.where(b > 0, (a / safe_b) ** (2.0 / float(n_s)), np.inf)


def exact_shrinkage_per_mode(N: int, n_s: float, t: float, schedule=None) -> NDArray[np.float64]:
    """Per-mode Wiener signal-fraction W(k, t) = a^2 sigma_k^2 / (a^2 sigma_k^2 + b^2).

    This is the diagonal of (a^2 Sigma) M(t)^{-1} in the Fourier basis -- the
    signal-fraction of the Wiener filter that maps x -> E[x | u, t]. By
    construction it lies in [0, 1] for every k and every t with (a, b) >= 0.

    NOTE: the related quantity a sigma_k^2 / (a^2 sigma_k^2 + b^2), which maps
    u_k -> E[x | u, t]_k, can exceed 1 for FM-style schedules with a^2 + b^2 < 1.
    The bounded form implemented here is the one relevant to the geometric-
    stability story (it is the per-mode analog of the conformal preconditioner).
    See docs/implementation_notes.md.
    """
    sigma2 = build_grf_covariance_diag(N, n_s)
    if schedule is None:
        a = float(a_of_t(t))
        b = float(b_of_t(t))
    else:
        a, b = schedule(t)
    num = (a * a) * sigma2
    den = num + (b * b)
    den = np.where(den > 0, den, 1.0)
    return num / den


# ============================================================================
# Forward / exact reverse-time flow on a GRF.
#
# Forward: u_t(k) = a(t) X(k) + b(t) Eps(k), with Eps the FFT of real-space
# white noise (Hermitian-symmetric by construction, so the iFFT is real).
#
# Reverse: per Fourier mode the autonomous velocity v = du/dt evaluated at the
# conditional mean is a closed-form scalar,
#
#     E[V(k) | U(k), t] = c_k(t) U(k),
#     c_k(t) = (a_dot(t) a(t) sigma_k^2 + b_dot(t) b(t)) / (a^2 sigma_k^2 + b^2),
#
# so each Euler step in reverse time multiplies U(k) by  1 + dt * c_k(t).
# The per-mode multiplier depends only on |k|, so the iFFT stays real.
# ============================================================================


def _make_hermitian_noise(N: int, rng: np.random.Generator) -> NDArray[np.complex128]:
    """Fourier coefficients with the Hermitian symmetry of a real NxN field.

    Implementation: draw real-space white noise N(0, I_{NxN}) and FFT it.
    The FFT of a real array is automatically Hermitian-symmetric, and the
    per-mode complex variance is N^2.
    """
    return np.fft.fft2(rng.standard_normal((N, N)))


def forward_corrupt(
    field_clean: NDArray[np.float64],
    t: float,
    schedule=None,
    rng: np.random.Generator | None = None,
    eps_real: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply the affine forward step in Fourier basis: U(k) = a X(k) + b Eps(k).

    If eps_real is provided, use it; otherwise draw fresh white noise from rng.
    Returns the real-space field at noise level t.
    """
    field_clean = np.asarray(field_clean, dtype=np.float64)
    N = field_clean.shape[-1]
    if schedule is None:
        a = float(a_of_t(t)); b = float(b_of_t(t))
    else:
        a, b = schedule(t)

    if eps_real is None:
        if rng is None:
            raise ValueError("forward_corrupt needs either eps_real or rng")
        eps_real = rng.standard_normal((N, N))
    else:
        eps_real = np.asarray(eps_real, dtype=np.float64)

    X_k = np.fft.fft2(field_clean)
    Eps_k = np.fft.fft2(eps_real)
    U_k = a * X_k + b * Eps_k
    return np.fft.ifft2(U_k).real


def _per_mode_velocity_factor(
    t: float,
    n_s: float,
    N: int,
    schedule=None,
    sigma2: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """c_k(t) per Fourier mode for the paper-convention velocity v = du/dt.

    If `sigma2` is supplied, it is used directly as the per-mode signal
    variance (must broadcast against the (N, N) Fourier grid). Otherwise
    `build_grf_covariance_diag(N, n_s)` is called.
    """
    if schedule is None:
        a = float(a_of_t(t)); b = float(b_of_t(t))
        a_dot = float(adot_of_t(t)); b_dot = float(bdot_of_t(t))
    else:
        a, b = schedule(t)
        # schedule is expected to return (a, b) only; derivatives default to FM
        a_dot, b_dot = -1.0, 1.0
    if sigma2 is None:
        sigma2 = build_grf_covariance_diag(N, n_s)
    m_k = a * a * sigma2 + b * b
    safe_m = np.where(m_k > 0, m_k, 1.0)
    return (a_dot * a * sigma2 + b_dot * b) / safe_m


def reverse_step_exact(
    field_t: NDArray[np.float64],
    t: float,
    t_next: float,
    n_s: float,
    schedule=None,
    sigma2: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """One forward-Euler reverse-time step: U(t_next; k) = U(t; k) (1 + dt c_k(t)).

    `sigma2` overrides `build_grf_covariance_diag(N, n_s)` if supplied.
    """
    field_t = np.asarray(field_t, dtype=np.float64)
    N = field_t.shape[-1]
    c_k = _per_mode_velocity_factor(t, n_s, N, schedule=schedule, sigma2=sigma2)
    multiplier = 1.0 + (t_next - t) * c_k
    U_k = np.fft.fft2(field_t)
    U_next = U_k * multiplier
    return np.fft.ifft2(U_next).real


def exact_reverse_trajectory(
    field_t_start: NDArray[np.float64],
    t_start: float,
    t_end: float,
    n_steps: int,
    n_s: float,
    schedule=None,
    sigma2: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reverse-flow trajectory on a log-spaced grid from t_start down to t_end.

    Returns (t_values, fields) with shapes (n_steps + 1,) and
    (n_steps + 1, N, N). `sigma2` is forwarded to every step.
    """
    if t_end >= t_start:
        raise ValueError("t_end must be strictly less than t_start for reverse flow")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    field_t_start = np.asarray(field_t_start, dtype=np.float64)
    N = field_t_start.shape[-1]
    t_values = np.logspace(np.log10(t_start), np.log10(t_end), n_steps + 1)
    fields = np.empty((n_steps + 1, N, N), dtype=np.float64)
    fields[0] = field_t_start
    for i in range(n_steps):
        fields[i + 1] = reverse_step_exact(
            fields[i], float(t_values[i]), float(t_values[i + 1]),
            n_s=n_s, schedule=schedule, sigma2=sigma2,
        )
    return t_values, fields

