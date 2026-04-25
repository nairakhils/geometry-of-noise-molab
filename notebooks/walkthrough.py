import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # The Geometry of Noise: a closed-form walkthrough

    Sahraee-Ardakan, Delbracio & Milanfar (arXiv:2602.18428, February 2026)
    show that an autonomous (noise-blind) generative model implicitly performs
    Riemannian gradient flow on a marginal energy whose Euclidean gradient is
    singular at every clean datum, and that a posterior-averaged conformal
    factor cancels the singularity at exactly the rate it grows. We reproduce
    that analytic story in closed form for $x \sim \mathcal{N}(0, \Sigma)$ in
    two dimensions and add a Fourier-mode picture for isotropic 2D Gaussian
    random fields that the paper does not draw.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The marginal energy $E_{\text{marg}}(u) = -\log \int p(u\mid t)\,p(t)\,dt$
    has a $1/b(t)^2$ singularity at every clean data point, so the raw
    gradient diverges as the noise level shrinks. The figure below evaluates
    that integrand on four discrete data points $\{(\pm 1, \pm 1)\}$ at
    five fixed probes: the raw norm tracks the $1/b^2$ envelope, while the
    paper's conformal factor $\lambda(t) = b + b^2/a$ trims the divergence.
    The original-claim part of this notebook is a Fourier-mode shrinkage
    picture for isotropic Gaussian random fields, presented later.
    """)
    return


@app.cell
def _(plt, singular_gradient):
    _t = singular_gradient["t_axis"]
    _raw = singular_gradient["raw_norm"]
    _pre = singular_gradient["preconditioned_norm"]
    _env = singular_gradient["envelope_inv_b_squared"]
    _labels = list(singular_gradient["probe_labels"])

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)

    for _i, _lab in enumerate(_labels):
        _y = _raw[_i]
        _mask = _y > 0
        _axes[0].loglog(_t[_mask], _y[_mask], label=_lab, linewidth=1.4)
    _axes[0].loglog(_t, _env, "k--", alpha=0.55, linewidth=1.0,
                    label=r"$1 / b(t)^2$ envelope")
    _axes[0].set_xlabel("t"); _axes[0].set_ylabel(r"$\|(u - a\,D_t^*) / \sigma^2\|$")
    _axes[0].set_title("raw integrand of grad E_marg")
    _axes[0].legend(fontsize=8, loc="upper right")

    for _i, _lab in enumerate(_labels):
        _y = _pre[_i]
        _mask = _y > 0
        _axes[1].loglog(_t[_mask], _y[_mask], label=_lab, linewidth=1.4)
    _axes[1].set_xlabel("t"); _axes[1].set_ylabel(r"$|\lambda(t)| \cdot \|(u - a\,D_t^*) / \sigma^2\|$")
    _axes[1].set_title("preconditioned by conformal factor")
    _axes[1].legend(fontsize=8, loc="upper right")

    _fig.suptitle("Singular gradient (left) and its conformal cancellation (right)",
                  y=1.02, fontsize=12)
    _fig.text(0.5, -0.02,
              "4 discrete data points at the corners of [-1, 1]^2, jitter = 1e-4."
              "  Raw integrand diverges as 1 / b(t)^2 near each datum;"
              "  preconditioning by lambda(t) = b + b^2/a removes one order of t.",
              ha="center", fontsize=8.5, style="italic")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    import sys as _sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(REPO_ROOT))

    from src import exact_affine, grf_2d, schedules

    return REPO_ROOT, mo, np, plt


@app.cell
def _(plt):
    _RC = {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "image.cmap": "magma",
    }
    plt.rcParams.update(_RC)
    return


@app.cell
def _(REPO_ROOT, np):
    _DATA_DIR = REPO_ROOT / "data"

    def _load_npz(name):
        with np.load(_DATA_DIR / name) as _z:
            return {_k: np.asarray(_z[_k]) for _k in _z.files}

    energy = _load_npz("energy_landscape_2d.npz")
    stability = _load_npz("stability_curves.npz")
    gallery = _load_npz("grf_gallery.npz")
    shrinkage = _load_npz("shrinkage_heatmap.npz")
    linear_fit = _load_npz("linear_score_fit.npz")
    singular_gradient = _load_npz("singular_gradient.npz")
    return energy, gallery, linear_fit, shrinkage, singular_gradient, stability


@app.cell
def _(mo):
    mo.md(r"""
    ## Paper claim vs notebook scope

    **Paper result.** The marginal energy
    $E_{\text{marg}}(u) = -\log \int p(u\mid t)\,p(t)\,dt$ has a $1/b(t)^2$
    singularity at every data point (Eqs. 11, 12). The autonomous field
    $f^*(u)$ that an unconditional network learns is structurally a natural
    gradient $\bar\lambda(u)\,\nabla E_{\text{marg}}(u)$ on a conformal metric
    $g(u) = 1/\bar\lambda(u)$, plus a transport correction that vanishes in
    the regimes the paper analyzes (Eqs. 14, 16, 18). The conformal factor
    $\bar\lambda(u)$ shrinks at the rate $b(t)$ shrinks near the manifold,
    so the product $\bar\lambda(u) \nabla E_{\text{marg}}(u)$ stays bounded.

    **Notebook result.** We compute $E_{\text{marg}}$, $\nabla E_{\text{marg}}$,
    the conformal factor $\bar\lambda(u)$, and the resulting preconditioned
    field on a $120\times 120$ grid for $x \sim \mathcal{N}(0, \Sigma)$ with
    $\Sigma = \mathrm{diag}([2.0,\,0.5])$. Every quantity is closed form;
    nothing is fitted. A separate OLS sanity check certifies the conditional-
    mean derivation against samples without any neural network.

    **Extension.** For an isotropic 2D Gaussian random field with
    $P(k) \propto k^{-n_s}$, the covariance is diagonal in the Fourier basis,
    so the analysis applies mode by mode. We plot the per-mode Wiener
    signal-fraction $W(k, t) = a^2 \sigma_k^2 / (a^2 \sigma_k^2 + b^2)$ as
    both a heatmap and 1D slices.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Same geometry on smooth Gaussian data for comparison: with no Dirac
    atoms in $p(x)$, neither field is singular, but the conformal
    preconditioner still reshapes the flow.
    """)
    return


@app.cell
def _(energy, np, plt):
    _u = energy["u_grid"]
    _E = energy["E_marg_grid"]
    _Gr = energy["grad_raw_grid"]
    _Gp = energy["grad_preconditioned_grid"]
    _extent = [
        float(_u[0, 0, 0]),
        float(_u[-1, 0, 0]),
        float(_u[0, 0, 1]),
        float(_u[0, -1, 1]),
    ]

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.6))

    _im = _axes[0].imshow(_E.T, origin="lower", extent=_extent, cmap="viridis")
    _axes[0].streamplot(
        _u[:, :, 0].T, _u[:, :, 1].T,
        _Gr[..., 0].T, _Gr[..., 1].T,
        density=1.1, color="white", linewidth=0.7, arrowsize=0.7,
    )
    _axes[0].set_title("E_marg(u) + raw gradient streamlines")
    _axes[0].set_xlabel("u_1"); _axes[0].set_ylabel("u_2")
    plt.colorbar(_im, ax=_axes[0], label="E_marg")

    _norm = np.linalg.norm(_Gp, axis=-1)
    _axes[1].streamplot(
        _u[:, :, 0].T, _u[:, :, 1].T,
        _Gp[..., 0].T, _Gp[..., 1].T,
        density=1.1, color=_norm.T, cmap="magma", linewidth=0.9,
    )
    _axes[1].set_title("preconditioned autonomous field f*(u) = lambda_bar * grad E_marg")
    _axes[1].set_xlabel("u_1"); _axes[1].set_ylabel("u_2")
    _axes[1].set_xlim(_extent[0], _extent[1])
    _axes[1].set_ylim(_extent[2], _extent[3])

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Geometry of the marginal energy

    **Paper result.** Treating the noise level $t$ as a random variable with
    prior $p(t)$ converts the family of conditional Gaussians $p(u \mid t)$
    into a single mixture density $p(u) = \int p(u\mid t)\,p(t)\,dt$. Tweedie's
    identity gives a closed-form gradient (Eq. 11):

    $$\nabla_u E_{\text{marg}}(u) = \mathbb{E}_{t \mid u}\!\left[
        \frac{u - a(t)\,D_t^*(u)}{b(t)^2} \right].$$

    The $1/b(t)^2$ kernel diverges as $t \to 0$, so $\|\nabla E_{\text{marg}}\|$
    is unbounded at every clean datum.

    **Notebook result.** The left panel of the figure above shows
    $E_{\text{marg}}$ for $\Sigma = \mathrm{diag}([2.0,\,0.5])$ with the raw
    Euclidean streamlines overlaid. The right panel shows the same field after
    multiplication by the posterior-averaged conformal factor
    $\bar\lambda(u) = \mathbb{E}_{t \mid u}[\lambda(t)]$ (paper Eq. 15). Both
    fields are bounded on this grid because the data distribution we picked,
    $x \sim \mathcal{N}(0, \Sigma)$, is itself smooth: the marginal $p(u)$ is
    a continuous mixture of Gaussians with no Dirac atoms, so
    $\nabla E_{\text{marg}}$ has no singularity. What the figure does show is
    the *shape* of the conformal preconditioning: $\bar\lambda(u)$ rescales
    the field non-uniformly, suppressing it more in directions where the
    posterior $p(t \mid u)$ concentrates on small $t$. The singular-gradient
    regime requires data on a discrete set or low-dimensional manifold; we
    do not visualize that case here (see "Limits" below).

    The conformal preconditioner $\bar\lambda(u)$ plays a role analogous to
    a softened Green's function: it carries a built-in length scale
    $\sim b(t)$ that suppresses the singular kernel of $\nabla E_{\text{marg}}$
    at the same rate the singularity grows.
    """)
    return


@app.cell
def _(mo, shrinkage):
    n_t = len(shrinkage["t_values"])
    t_slider = mo.ui.slider(
        start=0, stop=n_t - 1, step=1, value=n_t // 2, label="t index"
    )
    t_slider
    return (t_slider,)


@app.cell
def _(plt, shrinkage, t_slider):
    _idx = int(t_slider.value)
    _t_val = float(shrinkage["t_values"][_idx])
    _centers = shrinkage["radial_centers"]

    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    for _ns, _marker in zip([1, 2, 3], ["o", "s", "^"]):
        _rad = shrinkage[f"shrinkage_radial_ns{_ns}"][_idx]
        _ax.plot(_centers, _rad, _marker + "-", label=f"n_s = {_ns}")
    _ax.set_xscale("log")
    _ax.set_xlabel("k (radial wavenumber)")
    _ax.set_ylabel("Wiener signal-fraction W(k, t)")
    _ax.set_ylim(-0.02, 1.02)
    _ax.set_title(f"per-mode shrinkage at t = {_t_val:.4f}")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the raw gradient is singular

    **Paper result.** Equation 11 expresses $\nabla_u E_{\text{marg}}(u)$ as a
    posterior expectation of $(u - a\,D_t^*)/b^2$. Near a clean datum $x_k$
    the posterior $p(t \mid u)$ concentrates on small $t$ (paper's Lemmas 5
    and 6, conditional on codimension $D - d > 2$), where $b(t) \to 0$. The
    integrand inherits the full $1/b^2$ blow-up, so
    $\|\nabla E_{\text{marg}}\| \to \infty$ on the support of the data
    (Eq. 12).

    **Notebook implication.** For data on a discrete set or low-dimensional
    manifold, a sampler that follows the raw gradient becomes stiff in
    finite arithmetic; truncating $t$ at some $t_{\min} > 0$ converts the
    analytic singularity into a Hessian whose eigenvalues scale as
    $1/t_{\min}^2$. The preconditioned field circumvents this without
    truncation. Our $x \sim \mathcal{N}(0, \Sigma)$ setup does not put a
    Dirac atom anywhere, so we cannot reproduce the divergence directly; the
    implication is theoretical, supported by the paper's Eqs. 12 and 66 and
    by the stability curves below.
    """)
    return


@app.cell
def _(plt, stability):
    _t = stability["t_values"]
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.2))

    _axes[0].loglog(_t, stability["gain_nu_literal_noise"],
                    label=r"noise: $\nu(t) = 1/(1-t)$  ($c=0,\,d=1$)")
    _axes[0].loglog(_t, stability["gain_nu_literal_velocity"],
                    label=r"velocity: $\nu(t) = 1$  ($c=-1,\,d=1$)")
    _axes[0].loglog(_t, stability["gain_envelope_analytic"], "--",
                    alpha=0.55, label=r"$|\dot b / b|$ envelope (Eq. 66)")
    _axes[0].set_xlabel("t"); _axes[0].set_ylabel(r"$\nu(t)$  /  envelope")
    _axes[0].set_title(r"(a) sampler gain coefficient $\nu(t)$ vs near-manifold envelope")
    _axes[0].legend(fontsize=8)

    _axes[1].loglog(_t, stability["jensen_gap_noise_pred"], label="noise (eps form)")
    _axes[1].loglog(_t, stability["jensen_gap_velocity_pred"], label="velocity (v form)")
    _axes[1].set_xlabel("t"); _axes[1].set_ylabel("posterior gap")
    _axes[1].set_title("(b) Jensen / posterior gap")
    _axes[1].legend()

    _axes[2].loglog(_t, stability["drift_error_noise_pred"], label="noise: diverges")
    _axes[2].loglog(_t, stability["drift_error_velocity_pred"], label="velocity: bounded")
    _axes[2].set_xlabel("t"); _axes[2].set_ylabel("|nu| * gap")
    _axes[2].set_title("(c) drift perturbation error")
    _axes[2].legend()

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Parameterization stability: why velocity wins

    **Paper result.** The unified sampler ODE is
    $du/dt = \mu(t)\,u + \nu(t)\,f^*(u)$ with closed-form coefficients
    (Eq. 63):

    $$\mu(t) = \frac{\dot a\,d - \dot b\,c}{a\,d - b\,c},\qquad
      \nu(t) = \frac{\dot b\,a - \dot a\,b}{a\,d - b\,c}.$$

    For noise prediction, $(c, d) = (0, 1)$, the relevant prefactor near the
    manifold is $\dot b / b$, which diverges as $1/b(t)$ (Eq. 66). For
    velocity prediction, $(c, d) = (-1, 1)$, the determinant $a d - b c = 1$
    forces a constant gain $\nu(t) = 1$ (Eq. 70).

    **Notebook result.** We compute the noise-prediction Jensen Gap
    $|E_{\tau \mid u}[b(\tau)]\,E_{\tau \mid u}[1/b(\tau)] - 1|$ and the
    velocity dispersion $E_{\tau \mid u}[\|v_\tau^* - E[v]\|^2]$ on $500$
    samples per $t$, and multiply by the near-manifold envelope
    $|\dot b / b|$ for noise prediction and by $1$ for velocity prediction.
    Panel (a) shows the literal Eq. 63 coefficient for both
    parameterizations alongside the near-manifold envelope $|\dot b / b|$
    from Eq. 66; the noise-prediction coefficient is unbounded under FM
    linear (it grows as $1/(1-t)$ and diverges at $t \to 1$, where
    $a \to 0$), the envelope diverges at $t \to 0$ where the noise level
    collapses, and the velocity coefficient is identically 1. Panel (c)
    shows the product of envelope and gap: the noise curve diverges as
    $t \to 0$, the velocity curve stays $\mathcal{O}(10^{-2})$. The Jensen
    Gap (panel b) does not vanish on its own near $t \to 0$; the divergence
    sits in the envelope, as the paper predicts.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Sanity check: OLS recovers the closed-form Wiener matrix

    **Notebook result.** As an independent check on the conditional-mean
    derivation, we fit a linear estimator $A(t)$ to $(u_t, \text{target})$
    pairs by ordinary least squares and compare to the closed forms

    $$A_{\epsilon}(t) = b(t)\,M(t)^{-1},\qquad
      A_v(t) = \bigl(\dot a(t)\,a(t)\,\Sigma + \dot b(t)\,b(t)\,I\bigr)
                M(t)^{-1},$$

    where the velocity target is $v = \dot u = \dot a(t)\,x + \dot b(t)\,
    \epsilon$ (paper Eq. 61). On the FM linear schedule
    $(\dot a, \dot b) = (-1, 1)$, so $A_v(t) = (b(t)\,I - a(t)\,\Sigma)\,
    M(t)^{-1}$. We sweep $N \in \{10^3, 10^4, 10^5, 10^6\}$ and plot the
    relative Frobenius error per $t$ for each. The error scales as
    $1/\sqrt{N}$ as Gaussian linear-regression theory predicts, so each
    decade in $N$ shifts the curve by $\sqrt{10}$ on a semilog plot.
    The vertical ladder is the diagnostic: it confirms that the forward
    process, the schedule, the schedule derivatives, and the conditional-
    mean derivation in `src/exact_affine.py` are mutually consistent. No
    neural network is fitted; OLS on a closed-form linear estimator is the
    only learning step in this notebook. The full numerics are recorded in
    `docs/implementation_notes.md`.
    """)
    return


@app.cell
def _(linear_fit, plt):
    _t = linear_fit["t_values"]
    _Ns = linear_fit["n_samples_list"]
    _eps_grid = linear_fit["rel_err_eps_grid"]
    _v_grid = linear_fit["rel_err_v_grid"]

    _fig, _ax = plt.subplots(figsize=(8.5, 5.2))
    _cmap = plt.get_cmap("viridis")
    for _j, _N in enumerate(_Ns):
        _color = _cmap(_j / max(len(_Ns) - 1, 1))
        _ax.semilogy(_t, _eps_grid[_j], "o-", color=_color, linewidth=1.6,
                     label=f"eps, N = {int(_N):>7d}")
        _ax.semilogy(_t, _v_grid[_j], "s--", color=_color, linewidth=1.2,
                     alpha=0.85, label=f"v,   N = {int(_N):>7d}")
        _ax.axhline(float(_N) ** -0.5, color=_color, ls=":", alpha=0.55, linewidth=0.9)

    _ax.set_xlabel("t")
    _ax.set_ylabel(r"$\|A_{\mathrm{OLS}} - A_{\mathrm{exact}}\|_F / \|A_{\mathrm{exact}}\|_F$")
    _ax.set_title("OLS recovery of the closed-form Wiener matrix vs N")
    _ax.legend(ncol=2, fontsize=8, loc="upper right")
    _fig.text(
        0.5, -0.02,
        "Each decade in N shifts the error floor by a factor of ~sqrt(10). "
        "The vertical ladder is the diagnostic, not any single number. "
        "Dotted horizontal lines mark 1/sqrt(N) per N.",
        ha="center", fontsize=8.5, style="italic",
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Extension: Fourier-mode shrinkage for Gaussian random fields

    **Extension.** For an isotropic 2D Gaussian random field with isotropic
    power $\sigma_k^2 \propto k^{-n_s}$ and $\sigma_0^2 = 0$, the covariance
    is diagonal in the Fourier basis. Each mode $k$ then independently
    realizes the affine-noising geometry of the paper, with the same scalar
    schedule $a(t), b(t)$ acting per mode. We picked Gaussian random fields
    as the substrate because they make the mode-by-mode Wiener picture both
    exact and easy to plot; the paper itself sticks to discrete data sets and
    does not draw this picture.

    **Extension result (gallery and PSD).** The gallery below shows sample
    fields for $n_s \in \{1, 2, 3\}$. As $n_s$ grows the spectrum reddens
    and the texture coarsens. The PSD overlay validates the generator: the
    measured radial slope agrees with the analytic $-n_s$ within $\pm 0.06$
    at $B = 512$ samples per $n_s$ (numbers in `docs/implementation_notes.md`).

    **Extension result (shrinkage).** The interactive slider above and the
    static heatmap below both show the per-mode Wiener signal-fraction
    $W(k, t) = a^2 \sigma_k^2 / (a^2 \sigma_k^2 + b^2)$. As $t \to 0$,
    $W \to 1$ for every mode, recovering the data; at intermediate $t$ the
    high-$k$ modes fall toward $0$ first because their signal-to-noise ratio
    drops fastest. This is the per-mode picture of the same Wiener
    preconditioning that $\bar\lambda(u)$ implements globally.
    """)
    return


@app.cell
def _(gallery, plt):
    _fig, _axes = plt.subplots(3, 6, figsize=(13, 6.5))
    for _row, _ns in enumerate([1, 2, 3]):
        _samples = gallery[f"samples_ns{_ns}"]
        _show = _samples[: min(6, _samples.shape[0])]
        _vmax = float(max(abs(_show.min()), abs(_show.max())))
        for _col in range(6):
            _ax = _axes[_row, _col]
            if _col < _show.shape[0]:
                _ax.imshow(_show[_col], cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
            _ax.set_xticks([]); _ax.set_yticks([])
            _ax.grid(False)
            if _col == 0:
                _ax.set_ylabel(f"n_s = {_ns}", fontsize=11)
    _fig.suptitle("2D Gaussian random fields: sample gallery", y=0.995)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(gallery, plt):
    _fig, _ax = plt.subplots(figsize=(7.5, 4.6))
    _markers = {1: "o", 2: "s", 3: "^"}
    _colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
    for _ns in [1, 2, 3]:
        _c = gallery[f"psd_centers_ns{_ns}"]
        _m = gallery[f"measured_psd_ns{_ns}"]
        _th = gallery[f"theoretical_psd_ns{_ns}"]
        _sel = (_c > 0) & (_m > 0) & (_th > 0)
        _ax.loglog(_c[_sel], _m[_sel], _markers[_ns], color=_colors[_ns],
                   label=f"measured n_s = {_ns}")
        _ax.loglog(_c[_sel], _th[_sel], "-", color=_colors[_ns], alpha=0.55,
                   label=f"k^(-{_ns})")
    _ax.set_xlabel("k"); _ax.set_ylabel("P(k)")
    _ax.set_title("radial PSD: measured (512 fields per n_s) vs. theoretical")
    _ax.legend(ncol=2, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(plt, shrinkage):
    _centers = shrinkage["radial_centers"]
    _t_vals = shrinkage["t_values"]
    _rad = shrinkage["shrinkage_radial_ns2"]

    _fig = plt.figure(figsize=(13, 4.6))
    _gs = _fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1])

    _ax0 = _fig.add_subplot(_gs[0, 0])
    _im = _ax0.imshow(
        _rad, aspect="auto", origin="lower", cmap="magma",
        extent=[float(_centers[0]), float(_centers[-1]),
                float(_t_vals[0]), float(_t_vals[-1])],
        vmin=0, vmax=1,
    )
    _ax0.set_yscale("log")
    _ax0.set_xlabel("k"); _ax0.set_ylabel("t")
    _ax0.set_title("W(k, t) for n_s = 2")
    plt.colorbar(_im, ax=_ax0, label="W")

    _ax1 = _fig.add_subplot(_gs[0, 1])
    for _i in [3, 10, 16]:
        _ax1.plot(_centers, _rad[_i], "o-", label=f"t = {float(_t_vals[_i]):.3f}")
    _ax1.set_xlabel("k"); _ax1.set_ylabel("W")
    _ax1.set_xscale("log"); _ax1.set_ylim(-0.02, 1.02)
    _ax1.set_title("slice at fixed t")
    _ax1.legend(fontsize=8)

    _ax2 = _fig.add_subplot(_gs[0, 2])
    for _j in [0, 5, 15, 25]:
        if _j < _rad.shape[1]:
            _ax2.plot(_t_vals, _rad[:, _j], "o-",
                      label=f"k = {float(_centers[_j]):.1f}")
    _ax2.set_xlabel("t"); _ax2.set_ylabel("W")
    _ax2.set_xscale("log"); _ax2.set_ylim(-0.02, 1.02)
    _ax2.set_title("slice at fixed k")
    _ax2.legend(fontsize=8)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Limits and what this notebook does not claim

    Four constraints bound the analytic story above.

    **1. We do not reproduce the image-generation experiments.** The paper's
    Section 7 trains DDPM and Flow-Matching variants on CIFAR-10, SVHN, and
    Fashion-MNIST. We do not retrain those U-Nets, and we do not reproduce
    Figures 2-4 or Table 3. Our claims live entirely in the closed-form
    regime.

    **2. The exact story assumes linear-Gaussian data.** Every conditional,
    posterior, and score in this notebook is closed form precisely because
    $x \sim \mathcal{N}(0, \Sigma)$. The paper's near-manifold analysis
    (Appendix E, Lemmas 5-6) covers discrete data sets and smooth manifolds
    too, but we did not implement either path.

    **3. We do not train a neural score network.** The OLS sanity check
    above fits a *linear* estimator whose closed form is already known; that
    is the only learning step in this notebook. Statements about
    parameterization stability are derived from the closed form, not measured
    on a trained model.

    **4. The GRF extension is isotropic only.** We assume
    $\sigma_k^2 = k^{-n_s}$ depends on $\|k\|$ alone, which keeps the
    covariance diagonal in the Fourier basis. Anisotropic or non-stationary
    covariances induce mode coupling that we did not implement; the per-mode
    Wiener picture would no longer collapse to a single 2D heatmap.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## References

    [1] Sahraee-Ardakan, M., Delbracio, M., Milanfar, P. *The Geometry of
    Noise: Why Diffusion Models Don't Need Noise Conditioning.*
    arXiv:2602.18428v1, Google, February 2026. The paper this notebook
    reproduces.

    [2] Sun, Q., Jiang, Z., Zhao, H., He, K. *Is noise conditioning necessary
    for denoising generative models?* arXiv:2502.13129, 2025. Source of the
    unified affine-schedule formulation $u_t = a(t) x + b(t) \epsilon$ used
    throughout.

    [3] Lipman, Y., Chen, R. T. Q., Ben-Hamou, H., Nickel, M., Le, M.
    *Flow Matching for Generative Modeling.* arXiv:2210.02747, 2023. Source
    of the linear schedule $a(t) = 1 - t,\,b(t) = t$ adopted here.

    [4] Wang, R., Du, Y. *Equilibrium Matching: Generative Modeling with
    Implicit Energy-Based Models.* arXiv:2510.02300, 2025. The autonomous-
    field architecture whose stability analysis the paper [1] formalizes.

    [5] Salimans, T., Ho, J. *Progressive Distillation for Fast Sampling of
    Diffusion Models.* ICLR 2022. An alternative velocity convention,
    $v = \alpha_t\,\epsilon - \sigma_t\,x$, retained for reference as
    `velocity_target_SH` in `src/exact_affine.py` but not used in any
    figure or stability claim in this notebook.

    The full source for this notebook is at `geometry-of-noise-molab/`. See
    `docs/paper_summary.md` for the equation-by-equation extraction and
    `docs/implementation_notes.md` for numerical caveats.
    """)
    return


if __name__ == "__main__":
    app.run()
