# data/

Precomputed arrays for `notebooks/walkthrough.py`. All files are produced by
`scripts/precompute_arrays.py` with `seed=0`. Reruns are deterministic.

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
- `gain_envelope_analytic`         (300,)   1/b(t) reference curve
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

## shrinkage_heatmap.npz   (N = 64)
- `t_values`                       (20,)        log-spaced [1e-3, ~1]
- `radial_centers`                 (31,)        k-bin centers (linear width 1)
- `kmag`                           (64, 64)     wavenumber magnitudes
For each n_s in {1, 2, 3}:
- `shrinkage_ns{1,2,3}`            (20, 64, 64) Wiener signal fraction a^2 sigma^2 / (a^2 sigma^2 + b^2)
- `shrinkage_radial_ns{1,2,3}`     (20, 31)     radially averaged per t

## linear_score_fit.npz   (produced by scripts/linear_score_fit.py, not this script)
Sanity check: OLS recovers the closed-form Wiener matrix on N(0, Sigma) data.
- `t_values`                       (10,)       linear in [0.05, 0.95]
- `A_learned_eps`                  (10, 2, 2)  OLS-fit linear estimator of eps from u
- `A_exact_eps`                    (10, 2, 2)  b(t) M(t)^{-1}  (closed form)
- `A_learned_v`                    (10, 2, 2)  OLS-fit estimator of velocity from u
- `A_exact_v`                      (10, 2, 2)  a(t) b(t) (I - Sigma) M(t)^{-1}
- `rel_err_eps`, `rel_err_v`       (10,)       Frobenius |A_OLS - A_exact| / |A_exact|
- `n_samples`                      ()          N (per t) used for the OLS fit
