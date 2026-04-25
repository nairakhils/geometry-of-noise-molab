import numpy as np
import pytest

from src.grf_2d import (
    build_grf_covariance_diag,
    build_k_grid,
    exact_shrinkage_per_mode,
    measure_psd_radial,
    sample_grf_batch,
)


def test_grf_generator_slope_at_N64():
    """N=64, 2000 fields, n_s=2: log-log slope on the middle decade should be near -2.

    Tolerance loose because radial-bin variance is non-trivial even with B=2000.
    """
    rng = np.random.default_rng(42)
    fields = sample_grf_batch(B=2000, N=64, n_s=2, rng=rng)
    centers, P = measure_psd_radial(fields, n_bins=20)
    mask = (centers >= 3.0) & (centers <= 16.0) & (P > 0)
    assert mask.sum() >= 4
    slope, _ = np.polyfit(np.log(centers[mask]), np.log(P[mask]), 1)
    assert abs(slope - (-2.0)) < 0.15, f"slope={slope}, expected near -2"


def test_grf_samples_real():
    """Hermitian symmetry of the Fourier draw forces the inverse FFT to be real."""
    N = 16
    sigma2 = build_grf_covariance_diag(N, n_s=2)
    rng = np.random.default_rng(1)
    white = rng.standard_normal((4, N, N))
    F = np.fft.fft2(white) * np.sqrt(sigma2)[None, :, :]
    field_complex = np.fft.ifft2(F)
    assert np.max(np.abs(field_complex.imag)) < 1e-10


def test_grf_unit_variance_per_field():
    rng = np.random.default_rng(7)
    fields = sample_grf_batch(B=64, N=32, n_s=2, rng=rng)
    var = fields.var(axis=(1, 2))
    assert np.allclose(var, 1.0, rtol=0.05)


@pytest.mark.parametrize("t", [0.05, 0.25, 0.5, 0.75, 0.95])
@pytest.mark.parametrize("n_s", [1, 2, 3])
def test_shrinkage_in_unit_interval(t, n_s):
    W = exact_shrinkage_per_mode(N=32, n_s=n_s, t=t)
    assert (W >= 0.0).all()
    assert (W <= 1.0).all()
