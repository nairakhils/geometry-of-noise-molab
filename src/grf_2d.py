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

from .schedules import a_of_t, b_of_t


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


def sample_grf_batch(B: int, N: int, n_s: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Draw B real GRFs of size NxN by FFT-domain filtering, normalize to unit variance."""
    sigma2 = build_grf_covariance_diag(N, n_s)
    white = rng.standard_normal((B, N, N))
    W = np.fft.fft2(white)
    F = W * np.sqrt(sigma2)[None, :, :]
    field = np.fft.ifft2(F).real
    var = field.var(axis=(1, 2), keepdims=True)
    return field / np.sqrt(np.where(var > 0, var, 1.0))


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
