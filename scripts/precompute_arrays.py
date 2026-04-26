"""Precompute all arrays for notebooks/walkthrough.py.

Keeps the marimo notebook startup fast on molab and makes static GitHub
previews self-contained. All randomness is seeded; rerunning produces
byte-identical .npz files.

Target wall-clock: < 2 minutes on CPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.exact_affine import (
    E_marg,
    _circle_centers,
    conformal_factor_circle,
    denoiser_discrete,
    grad_E_marg_analytic,
    jensen_gap,
    marginal_cov,
    posterior_t_given_u,
)
from src.grf_2d import (
    build_grf_covariance_diag,
    build_k_grid,
    exact_reverse_trajectory,
    exact_shrinkage_per_mode,
    forward_corrupt,
    half_power_cutoff,
    measure_psd_radial,
    sample_grf_batch,
)
from src.schedules import a_of_t, adot_of_t, b_of_t, bdot_of_t


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 0
SIGMA_2D = np.diag([2.0, 0.5])
T_GRID = np.linspace(0.05, 0.95, 91)
PRIOR_T = np.full_like(T_GRID, 1.0 / (T_GRID[-1] - T_GRID[0]))


def _report(name: str, t0: float, payload: dict[str, np.ndarray]) -> None:
    elapsed = time.time() - t0
    size_mb = sum(np.asarray(v).nbytes for v in payload.values()) / 1e6
    print(f"[{name:<22}] {elapsed:6.2f}s   {size_mb:7.2f} MB   {len(payload)} arrays")


def conformal_factor_velocity(
    u: np.ndarray, t_grid: np.ndarray, prior_t: np.ndarray, Sigma: np.ndarray
) -> np.ndarray:
    """lambda_bar(u) for the velocity parameterization (c=-1, d=1).

    lambda(t) = (b/a) * (d*a - c*b) = (b/a)(a + b) = b + b^2/a   [Eq. 15].
    Posterior-average over t_grid yields the conformal factor used to
    precondition grad E_marg into a bounded autonomous field (see
    docs/paper_summary.md item d).
    """
    posterior = posterior_t_given_u(u, t_grid, prior_t, Sigma)
    a = a_of_t(t_grid)
    b = b_of_t(t_grid)
    lam_t = b + (b * b) / np.where(a > 0, a, np.nan)
    return np.einsum("...t,t->...", posterior, lam_t)


def precompute_energy_landscape() -> None:
    t0 = time.time()
    res = 120
    xs = np.linspace(-3.0, 3.0, res)
    ys = np.linspace(-3.0, 3.0, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    u_grid = np.stack([X, Y], axis=-1)
    flat = u_grid.reshape(-1, 2)

    E = E_marg(flat, T_GRID, PRIOR_T, SIGMA_2D).reshape(res, res)
    G = grad_E_marg_analytic(flat, T_GRID, PRIOR_T, SIGMA_2D).reshape(res, res, 2)
    lam = conformal_factor_velocity(flat, T_GRID, PRIOR_T, SIGMA_2D).reshape(res, res)
    G_pre = lam[..., None] * G

    payload = dict(
        u_grid=u_grid.astype(np.float32),
        E_marg_grid=E.astype(np.float32),
        grad_raw_grid=G.astype(np.float32),
        grad_preconditioned_grid=G_pre.astype(np.float32),
        conformal_factor_grid=lam.astype(np.float32),
        t_grid=T_GRID,
        prior_t=PRIOR_T,
    )
    np.savez(DATA_DIR / "energy_landscape_2d.npz", **payload)
    _report("energy_landscape_2d", t0, payload)


def precompute_stability_curves(rng: np.random.Generator) -> None:
    t0 = time.time()
    t_values = np.linspace(0.05, 0.95, 300)
    n_samples = 500

    gain_noise = 1.0 / b_of_t(t_values)
    gain_velocity = np.ones_like(t_values)
    gain_envelope = 1.0 / b_of_t(t_values)

    # Literal Eq. 63 sampler-gain coefficient: nu(t) = (b_dot a - a_dot b) / (a d - b c)
    _a = a_of_t(t_values); _b = b_of_t(t_values)
    _ad = adot_of_t(t_values); _bd = bdot_of_t(t_values)
    _num = _bd * _a - _ad * _b
    # noise prediction: (c, d) = (0, 1)  =>  ad - bc = a
    gain_nu_literal_noise = _num / _a
    # velocity prediction (paper conv.): (c, d) = (-1, 1)  =>  ad - bc = a + b
    gain_nu_literal_velocity = _num / (_a + _b)

    jensen_eps = np.empty_like(t_values)
    jensen_v = np.empty_like(t_values)

    for i, t in enumerate(t_values):
        M = marginal_cov(t, SIGMA_2D)
        L = np.linalg.cholesky(M)
        eps = rng.standard_normal((n_samples, 2))
        u_batch = eps @ L.T
        jensen_eps[i] = jensen_gap(u_batch, T_GRID, PRIOR_T, SIGMA_2D, param="eps").mean()
        jensen_v[i] = jensen_gap(u_batch, T_GRID, PRIOR_T, SIGMA_2D, param="v").mean()

    drift_noise = gain_noise * jensen_eps
    drift_velocity = gain_velocity * jensen_v

    payload = dict(
        t_values=t_values,
        gain_noise_pred=gain_noise,
        gain_velocity_pred=gain_velocity,
        gain_envelope_analytic=gain_envelope,
        gain_nu_literal_noise=gain_nu_literal_noise,
        gain_nu_literal_velocity=gain_nu_literal_velocity,
        jensen_gap_noise_pred=jensen_eps,
        jensen_gap_velocity_pred=jensen_v,
        drift_error_noise_pred=drift_noise,
        drift_error_velocity_pred=drift_velocity,
        n_samples_per_t=np.asarray(n_samples),
    )
    np.savez(DATA_DIR / "stability_curves.npz", **payload)
    _report("stability_curves", t0, payload)


def precompute_grf_gallery(rng: np.random.Generator) -> None:
    t0 = time.time()
    N = 32
    n_fields = 16
    payload: dict[str, np.ndarray] = {}

    for n_s in (1.0, 2.0, 3.0):
        sub_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        # generate more than 16 to get a stable PSD measurement, save 16
        many = sample_grf_batch(B=512, N=N, n_s=n_s, rng=sub_rng)
        gallery = many[:n_fields]
        centers, P = measure_psd_radial(many, n_bins=20)
        theo = np.where(centers > 0, centers ** (-n_s), 0.0)
        scale = P[centers > 0][0] / theo[centers > 0][0] if (theo > 0).any() else 1.0
        theo = theo * scale

        tag = f"ns{int(n_s)}"
        payload[f"samples_{tag}"] = gallery.astype(np.float32)
        payload[f"measured_psd_{tag}"] = P
        payload[f"theoretical_psd_{tag}"] = theo
        payload[f"psd_centers_{tag}"] = centers

    np.savez(DATA_DIR / "grf_gallery.npz", **payload)
    _report("grf_gallery", t0, payload)


def precompute_singular_gradient() -> None:
    """Per-t slice of grad E_marg on a 4-corner discrete prior with small jitter.

    Demonstrates the paper's central pathology: the integrand of grad E_marg
    diverges as 1/b(t)^2 at every t-slice for probes off the data, while the
    conformal preconditioning lambda(t) trims this to 1/b(t).
    """
    t0 = time.time()
    centers = np.array(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=np.float64
    )
    jitter = 1e-4
    # Upper bound 0.999 (not 1.0) avoids the FM-linear singularity at a(1) = 0,
    # which would put a NaN in lambda(t) = b + b^2/a at the endpoint.
    t_axis = np.logspace(-3.0, np.log10(0.999), 100)
    u_probe = np.array(
        [
            [1.0, 1.0],   # at a center
            [1.0, 0.0],   # midpoint between two centers
            [0.0, 0.0],   # center of the square
            [1.5, 0.0],   # slightly outside the right edge
            [3.0, 3.0],   # far outside
        ],
        dtype=np.float64,
    )
    probe_labels = np.array(
        [
            "at a center (1, 1)",
            "between two centers (1, 0)",
            "center of square (0, 0)",
            "outside edge (1.5, 0)",
            "far outside (3, 3)",
        ]
    )

    a_vals = a_of_t(t_axis)
    b_vals = b_of_t(t_axis)
    sigma2 = a_vals * a_vals * jitter + b_vals * b_vals          # (T,)
    lam_t = b_vals + (b_vals * b_vals) / np.where(a_vals > 0, a_vals, np.nan)

    # denoiser_discrete with leading axis (P,) of probes and t-axis (T,).
    # For each (probe, t) we get D shape (P, T, d).
    D = denoiser_discrete(u_probe, t_axis, centers, jitter)       # (P, T, d)
    integrand = (u_probe[:, None, :] - a_vals[None, :, None] * D) / sigma2[None, :, None]
    raw_grad_norm = np.linalg.norm(integrand, axis=-1)            # (P, T)

    # Per-t conformal factor squared: lambda(t)^2 = (b + b^2/a)^2.
    # Near small t lambda ~ b so lambda^2 ~ b^2, the rate that exactly cancels
    # the 1/b^2 of raw_grad_norm. Broadcast to (P, T) for shape consistency
    # with the other curves; the value is the same for every probe.
    P = u_probe.shape[0]
    lambda_bar_curves = np.broadcast_to((lam_t * lam_t)[None, :], (P, len(t_axis))).copy()
    preconditioned_grad_norm = lambda_bar_curves * raw_grad_norm   # (P, T)

    envelope_inv_b_squared = 1.0 / (b_vals * b_vals)
    envelope_b_squared = b_vals * b_vals

    payload = dict(
        centers=centers,
        jitter=np.asarray(jitter),
        t_axis=t_axis,
        u_probe=u_probe,
        probe_labels=probe_labels,
        raw_grad_norm=raw_grad_norm,
        lambda_bar_curves=lambda_bar_curves,
        preconditioned_grad_norm=preconditioned_grad_norm,
        envelope_inv_b_squared=envelope_inv_b_squared,
        envelope_b_squared=envelope_b_squared,
        lambda_t=lam_t,
        # Backward-compatible aliases used by the Phase 9 figure cell:
        raw_norm=raw_grad_norm,
        preconditioned_norm=np.abs(lam_t)[None, :] * raw_grad_norm,
    )
    np.savez(DATA_DIR / "singular_gradient.npz", **payload)
    _report("singular_gradient", t0, payload)


def precompute_circle_manifold() -> None:
    """Same per-t curves as precompute_singular_gradient, but the prior is
    a uniform distribution on a unit circle in R^2 instead of 4 corners.

    Demonstrates that the singular-gradient / conformal-cancellation story
    extends from a discrete-set prior to a smooth 1D manifold: the integrand
    still inherits a 1/b(t)^2 blow-up off the manifold, and lambda(t)^2 still
    matches the b^2 rate, so their product stays bounded.
    """
    t0 = time.time()
    radius = 1.0
    n_anchors = 256
    jitter = 1e-6
    centers = _circle_centers(radius, n_anchors)
    t_axis = np.logspace(-3.0, np.log10(0.999), 100)
    u_probe = np.array(
        [
            [1.0, 0.0],     # on the circle
            [0.7, 0.0],     # slightly inside
            [0.0, 0.0],     # at center
            [1.3, 0.0],     # slightly outside
            [3.0, 3.0],     # far outside
        ],
        dtype=np.float64,
    )
    probe_labels = np.array(
        [
            "on circle (1, 0)",
            "inside (0.7, 0)",
            "at center (0, 0)",
            "outside (1.3, 0)",
            "far outside (3, 3)",
        ]
    )

    a_vals = a_of_t(t_axis)
    b_vals = b_of_t(t_axis)
    sigma2 = a_vals * a_vals * jitter + b_vals * b_vals
    lam_t = b_vals + (b_vals * b_vals) / np.where(a_vals > 0, a_vals, np.nan)

    D = denoiser_discrete(u_probe, t_axis, centers, jitter)            # (P, T, d)
    integrand = (u_probe[:, None, :] - a_vals[None, :, None] * D) / sigma2[None, :, None]
    raw_grad_norm = np.linalg.norm(integrand, axis=-1)                  # (P, T)

    P = u_probe.shape[0]
    lambda_bar_curves = np.broadcast_to((lam_t * lam_t)[None, :], (P, len(t_axis))).copy()
    preconditioned_grad_norm = lambda_bar_curves * raw_grad_norm        # (P, T)

    envelope_inv_b_squared = 1.0 / (b_vals * b_vals)
    envelope_b_squared = b_vals * b_vals

    payload = dict(
        centers=centers,
        radius=np.asarray(radius),
        n_anchors=np.asarray(n_anchors),
        jitter=np.asarray(jitter),
        t_axis=t_axis,
        u_probe=u_probe,
        probe_labels=probe_labels,
        raw_grad_norm=raw_grad_norm,
        lambda_bar_curves=lambda_bar_curves,
        preconditioned_grad_norm=preconditioned_grad_norm,
        envelope_inv_b_squared=envelope_inv_b_squared,
        envelope_b_squared=envelope_b_squared,
        lambda_t=lam_t,
    )
    np.savez(DATA_DIR / "circle_manifold.npz", **payload)
    _report("circle_manifold", t0, payload)


def precompute_grf_flow_strip(rng: np.random.Generator) -> None:
    """One clean GRF; one shared eps draw used at four forward t values; an
    exact reverse trajectory from t=0.8 down to t=1e-3 with 50 log-spaced
    Euler steps; a per-t subsample of the trajectory matching the four
    forward t values for the bottom row of the figure strip.

    We rescale sigma_k^2 by K = N^2 / sum(k^{-n_s}) so the GRF has unit real-
    space variance in expectation AND per-mode |fft(x)(k)|^2 / N^2 = K * k^{-n_s}.
    The same rescaled sigma_k^2 is passed into exact_reverse_trajectory so the
    autonomous reverse flow uses the same per-mode signal variance the field
    actually has. With this, signal and noise (eps_real ~ N(0, I)) are on the
    same scale and the forward / reverse demo is visually meaningful.
    """
    t0 = time.time()
    N = 32
    n_s = 2.0
    forward_t = np.array([0.05, 0.2, 0.5, 0.8])
    n_steps = 50

    # rescaled per-mode signal variance so the GRF has unit real-space variance
    sigma2_raw = build_grf_covariance_diag(N, n_s)
    K_scale = (N * N) / sigma2_raw.sum()
    sigma2 = K_scale * sigma2_raw

    # generate clean GRF directly with the rescaled per-mode variance
    white = rng.standard_normal((N, N))
    F = np.fft.fft2(white) * np.sqrt(sigma2)
    clean = np.fft.ifft2(F).real

    eps_real = np.random.default_rng(SEED + 100).standard_normal((N, N))
    forward_fields = np.stack(
        [forward_corrupt(clean, t=float(t), eps_real=eps_real) for t in forward_t],
        axis=0,
    )

    t_traj, traj = exact_reverse_trajectory(
        forward_fields[-1], t_start=float(forward_t[-1]), t_end=1e-3,
        n_steps=n_steps, n_s=n_s, sigma2=sigma2,
    )

    # Forward corruption at every t along the reverse trajectory, sharing the
    # same eps_real -- gives the live widget a paired forward/reverse view at
    # every step without inflating the npz size noticeably.
    forward_at_traj = np.stack(
        [forward_corrupt(clean, t=float(t), eps_real=eps_real) for t in t_traj],
        axis=0,
    )

    # Pull the trajectory frames whose t most closely match forward_t.
    idx = np.array([int(np.argmin(np.abs(t_traj - float(t)))) for t in forward_t])
    reverse_strip_t = t_traj[idx]
    reverse_strip = traj[idx]

    payload = dict(
        clean_field=clean.astype(np.float32),
        forward_t_values=forward_t,
        forward_fields=forward_fields.astype(np.float32),
        reverse_t_values=t_traj,
        reverse_trajectory=traj.astype(np.float32),
        reverse_strip_t_values=reverse_strip_t,
        reverse_strip=reverse_strip.astype(np.float32),
        forward_at_traj_t=forward_at_traj.astype(np.float32),
        n_s=np.asarray(n_s),
        N=np.asarray(N),
        n_steps=np.asarray(n_steps),
    )
    np.savez(DATA_DIR / "grf_flow_strip.npz", **payload)
    _report("grf_flow_strip", t0, payload)


def precompute_shrinkage_heatmap() -> None:
    t0 = time.time()
    N = 64
    t_values = np.logspace(-3.0, np.log10(0.999), 20)
    _, _, kmag = build_k_grid(N)

    # Linear k-bins of unit width: [i, i+1) for i = 1..N/2-1. Every bin
    # contains at least the integer wavenumber i, so no empties to mask.
    edges = np.arange(1, N // 2 + 1, dtype=float)
    n_bins = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(kmag.ravel(), edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    counts = np.bincount(bin_idx[valid], minlength=n_bins)

    payload: dict[str, np.ndarray] = dict(
        t_values=t_values,
        kmag=kmag.astype(np.float32),
        radial_centers=centers,
    )

    for n_s in (1.0, 2.0, 3.0):
        stack = np.stack(
            [exact_shrinkage_per_mode(N=N, n_s=n_s, t=float(t)) for t in t_values],
            axis=0,
        )
        flat = stack.reshape(stack.shape[0], -1)
        # vectorized radial average via bincount per t (loop over T=20 only)
        sums = np.stack(
            [np.bincount(bin_idx[valid], weights=row[valid], minlength=n_bins) for row in flat],
            axis=0,
        )
        radial = sums / np.maximum(counts, 1)

        tag = f"ns{int(n_s)}"
        payload[f"shrinkage_{tag}"] = stack.astype(np.float32)
        payload[f"shrinkage_radial_{tag}"] = radial.astype(np.float32)

    # Half-power cutoff k_c(t, n_s) = (a/b)^(2/n_s). Stack (T, 3) for n_s in (1, 2, 3).
    k_c_curves = np.stack(
        [half_power_cutoff(t_values, n_s) for n_s in (1.0, 2.0, 3.0)],
        axis=1,
    )
    payload["k_c_curves"] = k_c_curves
    payload["k_c_n_s_list"] = np.asarray([1.0, 2.0, 3.0])

    np.savez(DATA_DIR / "shrinkage_heatmap.npz", **payload)
    _report("shrinkage_heatmap", t0, payload)


def write_data_readme() -> None:
    text = """# data/

Precomputed arrays for `notebooks/walkthrough.py`. All files are produced by
`scripts/precompute_arrays.py` with `seed=0`. Reruns are deterministic.

## singular_gradient.npz   (4 corner data, jitter = 1e-4)
- `centers`                        (4, 2)      data atoms at the corners of [-1, 1]^2
- `jitter`                         ()          1e-4
- `t_axis`                         (100,)      log-spaced [1e-3, 1.0]
- `u_probe`                        (5, 2)      probe points (at center, between two,
                                                center of square, outside edge, far)
- `probe_labels`                   (5,) U      human-readable labels for each probe
- `raw_grad_norm`                  (5, 100)    || (u - a D_t*) / sigma^2 ||  per (probe, t)
- `lambda_bar_curves`              (5, 100)    lambda(t)^2 broadcast across probes
- `preconditioned_grad_norm`       (5, 100)    lambda_bar_curves * raw_grad_norm  (bounded)
- `envelope_inv_b_squared`         (100,)      1 / b(t)^2 reference curve
- `envelope_b_squared`             (100,)      b(t)^2 reference curve
- `lambda_t`                       (100,)      conformal factor lambda(t) = b + b^2/a (paper Eq. 15)
- `raw_norm`, `preconditioned_norm`             backward-compatible aliases

## energy_landscape_2d.npz   (Sigma = diag([2.0, 0.5]))
- `u_grid`                     (120, 120, 2)  query points on [-3, 3]^2
- `E_marg_grid`                (120, 120)     marginal energy E_marg(u) (Eq. 8)
- `grad_raw_grid`              (120, 120, 2)  grad E_marg(u) (Eq. 11) -- singular near data
- `grad_preconditioned_grid`   (120, 120, 2)  lambda_bar(u) * grad E_marg(u) -- bounded
- `conformal_factor_grid`      (120, 120)     lambda_bar(u) for velocity parameterization
- `t_grid`, `prior_t`          (91,) each     integration grid (uniform [0.05, 0.95])

## stability_curves.npz
Curves on a dense 300-point t grid (linear in [0.05, 0.95]).
- `t_values`                       (300,)
- `gain_noise_pred`                (300,)   |b_dot/b| = 1/t for FM
- `gain_velocity_pred`             (300,)   identically 1 (Eq. 70)
- `gain_envelope_analytic`         (300,)   1/b(t) reference curve (Eq. 66 prefactor)
- `gain_nu_literal_noise`          (300,)   literal nu(t) from Eq. 63 with (c, d) = (0, 1)
- `gain_nu_literal_velocity`       (300,)   literal nu(t) from Eq. 63 with (c, d) = (-1, 1)
- `jensen_gap_noise_pred`          (300,)   E_u[ |E[b]E[1/b]-1| ], 500 u ~ N(0,M(t)) per t
- `jensen_gap_velocity_pred`       (300,)   posterior dispersion of v_tau*(u), same u's
- `drift_error_noise_pred`         (300,)   gain * gap (paper Eq. 22 product)
- `drift_error_velocity_pred`      (300,)   gain * gap
- `n_samples_per_t`                ()       500

## grf_gallery.npz   (N = 32)
For each n_s in {1, 2, 3}:
- `samples_ns{1,2,3}`              (16, 32, 32)  sample fields, unit-variance
- `measured_psd_ns{1,2,3}`         (n_bins,)     radial PSD averaged over 512 fields
- `theoretical_psd_ns{1,2,3}`      (n_bins,)     k^(-n_s), rescaled to overlay
- `psd_centers_ns{1,2,3}`          (n_bins,)     log-spaced bin centers

## grf_flow_strip.npz   (n_s = 2, N = 32, seed = 3)
- `clean_field`                    (32, 32)    one GRF sample
- `forward_t_values`               (4,)        [0.05, 0.2, 0.5, 0.8]
- `forward_fields`                 (4, 32, 32) corruptions of `clean_field`,
                                                shared eps draw across t values
- `reverse_t_values`               (51,)       log-spaced from 0.8 down to 1e-3
- `reverse_trajectory`             (51, 32, 32)  exact-flow fields
- `reverse_strip_t_values`         (4,)        sub-sampled t indices matching forward
- `reverse_strip`                  (4, 32, 32) trajectory frames at those indices
- `forward_at_traj_t`              (51, 32, 32) forward fields sampled at the
                                                  reverse-trajectory t values,
                                                  using the same shared eps draw
- `n_s`, `N`, `n_steps`            metadata

## shrinkage_heatmap.npz   (N = 64)
- `t_values`                       (20,)        log-spaced [1e-3, ~1]
- `radial_centers`                 (31,)        k-bin centers (linear width 1)
- `kmag`                           (64, 64)     wavenumber magnitudes
For each n_s in {1, 2, 3}:
- `shrinkage_ns{1,2,3}`            (20, 64, 64) Wiener signal fraction a^2 sigma^2 / (a^2 sigma^2 + b^2)
- `shrinkage_radial_ns{1,2,3}`     (20, 31)     radially averaged per t
- `k_c_curves`                     (20, 3)      half-power cutoff k_c(t, n_s) for n_s in (1, 2, 3)
- `k_c_n_s_list`                   (3,)         the n_s values matching k_c_curves columns

## linear_score_fit.npz   (produced by scripts/linear_score_fit.py, not this script)
Sanity check: OLS recovers the closed-form Wiener matrix on N(0, Sigma) data.
- `t_values`                       (10,)       linear in [0.05, 0.95]
- `A_learned_eps`                  (10, 2, 2)  OLS-fit linear estimator of eps from u
- `A_exact_eps`                    (10, 2, 2)  b(t) M(t)^{-1}  (closed form)
- `A_learned_v`                    (10, 2, 2)  OLS-fit estimator of velocity from u
- `A_exact_v`                      (10, 2, 2)  a(t) b(t) (I - Sigma) M(t)^{-1}
- `rel_err_eps`, `rel_err_v`       (10,)       Frobenius |A_OLS - A_exact| / |A_exact|
- `n_samples`                      ()          N (per t) used for the OLS fit
"""
    (DATA_DIR / "README.md").write_text(text)


def main() -> None:
    t_total = time.time()
    rng = np.random.default_rng(SEED)

    precompute_singular_gradient()
    precompute_circle_manifold()
    precompute_energy_landscape()
    precompute_stability_curves(np.random.default_rng(SEED + 1))
    precompute_grf_gallery(np.random.default_rng(SEED + 2))
    precompute_grf_flow_strip(np.random.default_rng(SEED + 3))
    precompute_shrinkage_heatmap()
    write_data_readme()

    print(f"[{'total':<22}] {time.time() - t_total:6.2f}s")


if __name__ == "__main__":
    main()
