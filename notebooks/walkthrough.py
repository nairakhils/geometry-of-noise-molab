import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Singular gradients, conformal flows, and Fourier shrinkage

    *A closed-form reading of arXiv:2602.18428.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Akhil Nair  ·    ·  April 2026  ·
    alphaXiv x marimo competition submission
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The marginal energy $E_{\text{marg}}(u) = -\log \int p(u\mid t)\,p(t)\,dt$
    has a $1/b(t)^2$ singularity at every clean data point, so the raw
    gradient diverges as the noise level shrinks. The figure below shows
    the three-panel "diverges, vanishes, bounded" structure on four
    discrete data points $\{(\pm 1, \pm 1)\}$ at five fixed probes: the
    raw integrand norm tracks the $1/b^2$ envelope (panel 1), the paper's
    conformal factor $\lambda(t)^2$ vanishes at the matching $b^2$ rate
    (panel 2), and their product stays bounded over the divergent regime
    (panel 3). The original-claim part of this notebook is a Fourier-mode
    shrinkage picture for isotropic Gaussian random fields, presented later.
    """)
    return


@app.cell
def _(np, plt, singular_gradient):
    _t = singular_gradient["t_axis"]
    _raw = singular_gradient["raw_grad_norm"]
    _lam = singular_gradient["lambda_bar_curves"]
    _pre = singular_gradient["preconditioned_grad_norm"]
    _env_inv = singular_gradient["envelope_inv_b_squared"]
    _env_b2 = singular_gradient["envelope_b_squared"]
    _labels = list(singular_gradient["probe_labels"])

    # Slope of the bounded curve over the divergent regime
    # (t in [b > sqrt(jitter), 0.5]; below sqrt(jitter)=0.01 the integrand
    # saturates at the jitter floor and the slope is uninformative).
    _fit_mask = (_t >= 0.01) & (_t <= 0.5)
    _slopes = []
    for _i in range(len(_labels)):
        _y = _pre[_i][_fit_mask]
        if (_y > 0).all() and len(_y) > 2:
            _slopes.append(float(np.polyfit(np.log(_t[_fit_mask]), np.log(_y), 1)[0]))
        else:
            _slopes.append(float("nan"))

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)

    # Panel 1: raw per-t integrand of grad E_marg, with 1/b^2 envelope.
    for _i, _lab in enumerate(_labels):
        _y = _raw[_i]
        _mask = _y > 0
        _axes[0].loglog(_t[_mask], _y[_mask], label=_lab, linewidth=1.4)
    _axes[0].loglog(_t, _env_inv, "k--", alpha=0.55, linewidth=1.0,
                    label=r"$1 / b(t)^2$ envelope")
    _axes[0].set_xlabel("t")
    _axes[0].set_ylabel(r"$\|(u - a D_t^*) / \sigma^2\|$")
    _axes[0].set_title("(1) raw: diverges as $1/b^2$")
    _axes[0].legend(fontsize=7, loc="upper right")

    # Panel 2: lambda(t)^2 vs t, with b^2 envelope.
    # Same curve for every probe (no probe dependence in lambda(t)),
    # so a single line covers all.
    _axes[1].loglog(_t, _lam[0], color="C3", linewidth=1.6,
                    label=r"$\lambda(t)^2 = (b + b^2/a)^2$")
    _axes[1].loglog(_t, _env_b2, "k--", alpha=0.55, linewidth=1.0,
                    label=r"$b(t)^2$ envelope")
    _axes[1].set_xlabel("t")
    _axes[1].set_ylabel(r"$\bar\lambda$ proxy")
    _axes[1].set_title(r"(2) $\lambda(t)^2$: vanishes as $b^2$")
    _axes[1].legend(fontsize=7, loc="upper left")

    # Panel 3: bounded product, with measured slope per probe in legend.
    for _i, _lab in enumerate(_labels):
        _y = _pre[_i]
        _mask = _y > 0
        _slope_str = ("flat" if abs(_slopes[_i]) < 0.05
                      else f"slope {_slopes[_i]:+.2f}") if not np.isnan(_slopes[_i]) else "n/a"
        _axes[2].loglog(_t[_mask], _y[_mask], linewidth=1.4,
                        label=f"{_lab}  ({_slope_str})")
    _axes[2].set_xlabel("t")
    _axes[2].set_ylabel(r"$\lambda(t)^2 \cdot \|\nabla\,\mathrm{integrand}\|$")
    _axes[2].set_title("(3) bounded product  $\\approx$  flat over $t \\in [0.01, 0.5]$")
    _axes[2].legend(fontsize=7, loc="upper right")

    _fig.suptitle(
        "Singular gradient (1) and its conformal cancellation (2 → 3)",
        y=1.02, fontsize=12,
    )
    _fig.text(
        0.5, -0.02,
        "Panel 1 diverges as 1 / b(t)^2; panel 2 vanishes at the same rate "
        "(lambda^2 ~ b^2); panel 3 is the bounded product. Slope annotations "
        "in panel 3 are log-log fits over t in [0.01, 0.5]; values near zero "
        "confirm the cancellation. The curves bend up at t -> 1 because "
        "lambda(t) = b + b^2/a inherits the FM singularity at a = 0.",
        ha="center", fontsize=8.5, style="italic",
    )
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
    import sympy as sp

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(REPO_ROOT))

    # The notebook consumes data/*.npz for figures and src.sympy_validation
    # for the symbolic re-derivation cell. The rest of src/ is exercised by
    # tests/ and scripts/precompute_arrays.py.
    from src.sympy_validation import (
        validate_noise_gain_divergence,
        validate_velocity_gain,
    )

    return (
        REPO_ROOT,
        mo,
        np,
        plt,
        sp,
        validate_noise_gain_divergence,
        validate_velocity_gain,
    )


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
    grf_flow_strip = _load_npz("grf_flow_strip.npz")
    return (
        energy,
        gallery,
        grf_flow_strip,
        linear_fit,
        shrinkage,
        singular_gradient,
        stability,
    )


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
    preconditioner still reshapes the flow. Hover over either panel for
    exact $(u_1, u_2, \text{value})$ readouts; pan and zoom each panel
    independently.
    """)
    return


@app.cell
def _(energy, mo):
    import plotly.figure_factory as _pff
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    _u = energy["u_grid"]
    _E = energy["E_marg_grid"]
    _Gr = energy["grad_raw_grid"]
    _Gp = energy["grad_preconditioned_grid"]
    _Lam = energy["conformal_factor_grid"]

    # 1D coordinate axes (u_grid is built with indexing='ij' so axis 0 is u_1)
    _xs = _u[:, 0, 0]
    _ys = _u[0, :, 1]

    # Streamlines on a 60x60 view of the 120x120 grid: cuts trace size in half
    # without losing visual fidelity. Heatmaps stay at 120x120.
    _stride = 2
    _xs_s = _xs[::_stride]
    _ys_s = _ys[::_stride]
    _Gr_x = _Gr[::_stride, ::_stride, 0].T
    _Gr_y = _Gr[::_stride, ::_stride, 1].T
    _Gp_x = _Gp[::_stride, ::_stride, 0].T
    _Gp_y = _Gp[::_stride, ::_stride, 1].T

    _stream_left = _pff.create_streamline(
        _xs_s, _ys_s, _Gr_x, _Gr_y, density=1.0, arrow_scale=0.05,
    ).data[0]
    _stream_left.line.color = "white"
    _stream_left.line.width = 1.0
    _stream_left.showlegend = False
    _stream_left.hoverinfo = "skip"

    _stream_right = _pff.create_streamline(
        _xs_s, _ys_s, _Gp_x, _Gp_y, density=1.0, arrow_scale=0.05,
    ).data[0]
    _stream_right.line.color = "black"
    _stream_right.line.width = 1.0
    _stream_right.showlegend = False
    _stream_right.hoverinfo = "skip"

    _fig_eng = _make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "E_marg(u)  +  raw gradient streamlines",
            "lambda_bar(u)  +  preconditioned field",
        ),
        horizontal_spacing=0.12,
    )
    _fig_eng.add_trace(
        _go.Heatmap(
            x=_xs, y=_ys, z=_E.T,
            colorscale="Viridis",
            colorbar=dict(title="E_marg", x=0.44, thickness=10, len=0.85),
            hovertemplate="u_1=%{x:.2f}  u_2=%{y:.2f}  E_marg=%{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    _fig_eng.add_trace(_stream_left, row=1, col=1)

    _fig_eng.add_trace(
        _go.Heatmap(
            x=_xs, y=_ys, z=_Lam.T,
            colorscale="Magma",
            colorbar=dict(title="lambda_bar", x=1.02, thickness=10, len=0.85),
            hovertemplate="u_1=%{x:.2f}  u_2=%{y:.2f}  lambda_bar=%{z:.3f}<extra></extra>",
        ),
        row=1, col=2,
    )
    _fig_eng.add_trace(_stream_right, row=1, col=2)

    _fig_eng.update_xaxes(title_text="u_1", scaleanchor="y", constrain="domain")
    _fig_eng.update_yaxes(title_text="u_2")
    _fig_eng.update_layout(
        height=460, width=960,
        margin=dict(l=40, r=60, t=50, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )

    mo.ui.plotly(_fig_eng)
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
def _(mo, sp, validate_noise_gain_divergence, validate_velocity_gain):
    _v = validate_velocity_gain()
    _n = validate_noise_gain_divergence()
    mo.md(rf"""
    ### Symbolic verification (sympy)

    The next figure plots the literal Eq. 63 sampler gain $\nu(t)$ for
    noise and velocity prediction. Before reading the plot we re-derive
    the two identities the figure rests on directly in sympy. No
    floating-point arithmetic, no schedule grid; pure symbolic
    simplification of the same formulas listed in
    `docs/paper_summary.md`.

    **Velocity, $(c, d) = (-1, 1)$.** Eq. 63 gives, in symbols,

    $$ \nu(t) \;=\; {sp.latex(_v['nu_general'])}. $$

    Substituting the FM linear schedule $a(t) = 1 - t$, $b(t) = t$,
    $\dot a = -1$, $\dot b = 1$ and asking sympy to simplify yields

    $$ \nu(t)\bigr|_{{\text{{linear FM}}}} \;=\; {sp.latex(_v['nu_linear_FM'])}. $$

    The constant-1 result is paper Eq. 70: under the velocity
    parameterization the sampler gain neither vanishes nor diverges at any
    $t$, which is the closed-form reason velocity prediction is the only
    autonomous parameterization that satisfies the bounded-error
    condition.

    **Noise, $(c, d) = (0, 1)$.** Same machinery gives

    $$ \nu(t)\bigr|_{{\text{{linear FM}}}}
        \;=\; {sp.latex(_n['nu_noise_linear_FM'])},
       \qquad
       \frac{{\dot b}}{{b}}\bigr|_{{\text{{linear FM}}}}
        \;=\; {sp.latex(_n['envelope_linear_FM'])}. $$

    Sympy's $\lim_{{t \to 0^+}} \nu(t) / (\dot b/b) =
    {sp.latex(_n['ratio_limit'])}$. Under FM linear the literal $\nu$ for
    noise prediction is bounded near the manifold; the divergence the
    paper attributes to noise prediction lives in the $\dot b / b$
    envelope of Eq. 66, not in $\nu$ itself. The plot below shows both
    quantities side by side and lets the reader confirm the ratio behaves
    as the symbolic limit predicts.
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

    **Extension result (shrinkage).** The two sliders below select a noise
    level $t$ and a spectral index $n_s \in \{1, 2, 3\}$, and the figure
    panel re-renders the per-mode Wiener signal-fraction
    $W(k, t) = a^2 \sigma_k^2 / (a^2 \sigma_k^2 + b^2)$ accordingly. We
    plot the signal-fraction $W$ rather than the Wiener gain
    $g = a \sigma_k^2 / (a^2 \sigma_k^2 + b^2)$; the two differ by a factor
    of $a(t)$, and $W$ is the bounded $[0, 1]$ quantity that gives the
    variance fraction of $u_t$ at mode $k$ originating from signal. As
    $t \to 0$, $W \to 1$ for every mode, recovering the data; at
    intermediate $t$ the high-$k$ modes fall toward $0$ first because their
    signal-to-noise ratio drops fastest. This is the per-mode picture of
    the same Wiener preconditioning that $\bar\lambda(u)$ implements
    globally.
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
def _(mo, shrinkage):
    n_t = len(shrinkage["t_values"])
    t_slider = mo.ui.slider(
        start=0, stop=n_t - 1, step=1, value=n_t // 2, label="t index"
    )
    n_s_dropdown = mo.ui.dropdown(
        options=["1", "2", "3"], value="2", label="spectral index n_s"
    )
    mo.vstack([t_slider, n_s_dropdown])
    return n_s_dropdown, t_slider


@app.cell
def _(n_s_dropdown, plt, shrinkage, t_slider):
    _ns = int(n_s_dropdown.value)
    _idx = int(t_slider.value)
    _centers = shrinkage["radial_centers"]
    _t_vals = shrinkage["t_values"]
    _rad = shrinkage[f"shrinkage_radial_ns{_ns}"]
    _t_now = float(_t_vals[_idx])

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
    _ax0.axhline(_t_now, color="cyan", lw=1.0, alpha=0.85)
    _ax0.set_xlabel("k"); _ax0.set_ylabel("t")
    _ax0.set_title(f"W(k, t) for n_s = {_ns}")
    plt.colorbar(_im, ax=_ax0, label="W")

    _ax1 = _fig.add_subplot(_gs[0, 1])
    for _ns_overlay, _marker in zip([1, 2, 3], ["o", "s", "^"]):
        _row = shrinkage[f"shrinkage_radial_ns{_ns_overlay}"][_idx]
        _ax1.plot(_centers, _row, _marker + "-",
                  label=f"n_s = {_ns_overlay}", alpha=0.85)
    _ax1.set_xlabel("k"); _ax1.set_ylabel("W")
    _ax1.set_xscale("log"); _ax1.set_ylim(-0.02, 1.02)
    _ax1.set_title(f"slice at t = {_t_now:.4f}")
    _ax1.legend(fontsize=8)

    _ax2 = _fig.add_subplot(_gs[0, 2])
    for _j in [0, 5, 15, 25]:
        if _j < _rad.shape[1]:
            _ax2.plot(_t_vals, _rad[:, _j], "o-",
                      label=f"k = {float(_centers[_j]):.1f}")
    _ax2.set_xlabel("t"); _ax2.set_ylabel("W")
    _ax2.set_xscale("log"); _ax2.set_ylim(-0.02, 1.02)
    _ax2.set_title(f"slice at fixed k  (n_s = {_ns})")
    _ax2.legend(fontsize=8)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Forward corruption and exact reverse flow on one GRF

    We take one clean GRF, draw one forward noise once, and run the exact
    reverse flow on the corrupted field. No neural network is involved;
    every reverse step is a closed-form scalar Wiener-style filter per
    Fourier mode, with per-mode coefficient $c_k(t) = (\dot a\,a\,
    \sigma_k^2 + \dot b\,b) / (a^2 \sigma_k^2 + b^2)$. The reverse-flow
    row visibly approaches the clean field in the lowest-$t$ panel. This
    is the autonomous-field generation process from the paper made fully
    explicit on a substrate where it has a closed form.
    """)
    return


@app.cell
def _(grf_flow_strip, np):
    flow_clean = grf_flow_strip["clean_field"]
    flow_traj = grf_flow_strip["reverse_trajectory"]
    flow_traj_t = grf_flow_strip["reverse_t_values"]
    flow_fwd_at_traj = grf_flow_strip["forward_at_traj_t"]
    flow_strip_t = grf_flow_strip["reverse_strip_t_values"]
    flow_strip_rev = grf_flow_strip["reverse_strip"]
    flow_strip_fwd = grf_flow_strip["forward_fields"]

    flow_n_steps = int(flow_traj.shape[0])
    flow_vmax = float(np.max(np.abs(flow_clean)))
    return (
        flow_clean,
        flow_fwd_at_traj,
        flow_n_steps,
        flow_strip_fwd,
        flow_strip_rev,
        flow_strip_t,
        flow_traj,
        flow_traj_t,
        flow_vmax,
    )


@app.cell
def _(flow_n_steps, mo):
    flow_step_slider = mo.ui.slider(
        start=0, stop=flow_n_steps - 1, step=1, value=0, label="reverse step"
    )
    flow_play_speed = mo.ui.refresh(
        options=["off", "0.5s", "0.25s"], default_interval="off", label="auto-play"
    )
    mo.hstack([flow_step_slider, flow_play_speed], justify="start")
    return flow_play_speed, flow_step_slider


@app.cell
def _(mo):
    get_flow_step, set_flow_step = mo.state(0)
    return get_flow_step, set_flow_step


@app.cell
def _(
    flow_n_steps,
    flow_play_speed,
    flow_step_slider,
    get_flow_step,
    set_flow_step,
):
    # Subscribe to the refresh's value so this cell re-runs on every tick.
    _refresh_value = flow_play_speed.value
    _is_playing = _refresh_value not in ("off", None)
    if _is_playing:
        set_flow_step((get_flow_step() + 1) % flow_n_steps)
    elif flow_step_slider.value != get_flow_step():
        set_flow_step(int(flow_step_slider.value))
    return


@app.cell
def _(flow_fwd_at_traj, flow_traj, flow_traj_t, flow_vmax, get_flow_step, mo):
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    _idx = int(get_flow_step())
    _t_now = float(flow_traj_t[_idx])

    _fig_live = _make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"forward U_t at t = {_t_now:.4f}",
            f"reverse flow at t = {_t_now:.4f}",
        ),
        horizontal_spacing=0.08,
    )
    _fig_live.add_trace(
        _go.Heatmap(
            z=flow_fwd_at_traj[_idx], colorscale="RdBu_r",
            zmin=-flow_vmax, zmax=flow_vmax, showscale=False,
            hovertemplate="i=%{y} j=%{x}<br>U_t=%{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    _fig_live.add_trace(
        _go.Heatmap(
            z=flow_traj[_idx], colorscale="RdBu_r",
            zmin=-flow_vmax, zmax=flow_vmax, showscale=True,
            hovertemplate="i=%{y} j=%{x}<br>U_rev=%{z:.3f}<extra></extra>",
            colorbar=dict(thickness=10, len=0.85),
        ),
        row=1, col=2,
    )
    _fig_live.update_xaxes(showticklabels=False, scaleanchor="y", constrain="domain")
    _fig_live.update_yaxes(showticklabels=False, autorange="reversed")
    _fig_live.update_layout(
        height=380, width=820,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    mo.ui.plotly(_fig_live)
    return


@app.cell
def _(
    flow_clean,
    flow_strip_fwd,
    flow_strip_rev,
    flow_strip_t,
    flow_vmax,
    grf_flow_strip,
    mo,
    np,
    plt,
):
    _fwd_t_static = grf_flow_strip["forward_t_values"]
    _fig_static, _axes = plt.subplots(2, 4, figsize=(13, 6.4))

    for _j in range(4):
        _ax = _axes[0, _j]
        _ax.imshow(flow_strip_fwd[_j], cmap="RdBu_r",
                   vmin=-flow_vmax, vmax=flow_vmax)
        _ax.set_xticks([]); _ax.set_yticks([]); _ax.grid(False)
        _ax.set_title(f"forward, t = {float(_fwd_t_static[_j]):.3f}", fontsize=10)

    # Bottom row read right-to-left so the visual is "motion back toward t=0".
    _rev_order = list(range(3, -1, -1))
    for _col, _src in enumerate(_rev_order):
        _ax = _axes[1, _col]
        _ax.imshow(flow_strip_rev[_src], cmap="RdBu_r",
                   vmin=-flow_vmax, vmax=flow_vmax)
        _ax.set_xticks([]); _ax.set_yticks([]); _ax.grid(False)
        _ax.set_title(f"reverse, t = {float(flow_strip_t[_src]):.3f}", fontsize=10)

    _axes[0, 0].set_ylabel("forward", fontsize=11)
    _axes[1, 0].set_ylabel("reverse", fontsize=11)
    _fig_static.tight_layout()

    mo.accordion(
        {
            "Static snapshot at t in [0.05, 0.2, 0.5, 0.8]  (fallback for PDF / static export)":
            _fig_static
        }
    )
    # silence unused-name warnings for variables only consumed via the
    # accordion's matplotlib payload
    _ = (flow_clean, np)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Half-power cutoff $k_c(t, n_s)$

    The half-power cutoff $k_c(t, n_s)$ is the radial wavenumber at which
    the per-mode signal-fraction $W$ drops to $1/2$. Setting
    $W = a^2 \sigma_k^2 / (a^2 \sigma_k^2 + b^2) = 1/2$ with
    $\sigma_k^2 \propto k^{-n_s}$ collapses to $a^2 \sigma_k^2 = b^2$,
    so $k_c(t, n_s) = (a(t) / b(t))^{2 / n_s}$ in closed form.
    The autonomous flow preserves modes with $k < k_c$ at noise level $t$
    and is dominated by noise above $k_c$, so $k_c(t)$ traces the moving
    boundary between signal and noise as $t$ varies.
    """)
    return


@app.cell
def _(plt, shrinkage):
    _t = shrinkage["t_values"]
    _kc = shrinkage["k_c_curves"]
    _ns_list = shrinkage["k_c_n_s_list"]

    _fig, _ax = plt.subplots(figsize=(8.0, 4.6))
    _cmap = plt.get_cmap("viridis")
    for _j, _ns in enumerate(_ns_list):
        _color = _cmap(_j / max(len(_ns_list) - 1, 1))
        _ax.loglog(_t, _kc[:, _j], "o-", color=_color, linewidth=1.6,
                   label=fr"$n_s = {int(_ns)}$:  $(a/b)^{{2/{int(_ns)}}}$")
    _ax.set_xlabel("t")
    _ax.set_ylabel(r"$k_c(t, n_s)$")
    _ax.set_title("Half-power cutoff $k_c(t, n_s)$")
    _ax.legend()
    _fig.text(
        0.5, -0.02,
        "Solid lines are the closed form k_c = (a/b)^(2/n_s). "
        "Steeper spectra (larger n_s) keep more high-k structure as the noise level grows.",
        ha="center", fontsize=8.5, style="italic",
    )
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
def _(REPO_ROOT, mo):
    import json as _json

    _manifest_path = REPO_ROOT / "data" / "manifest.json"
    if _manifest_path.exists():
        _manifest = _json.loads(_manifest_path.read_text())
        _entries = "\n".join(f"- `{_k}`: `{_v}`" for _k, _v in _manifest.items())
        mo.md(
            "### Data provenance\n\n"
            "Each `.npz` under `data/` was produced by `scripts/reproduce.py` "
            "with fixed seeds. The SHA-256 prefixes below are the committed "
            "reference; rerun `python scripts/reproduce.py` and diff against "
            "the committed `data/manifest.json` for byte-exact verification.\n\n"
            + _entries
        )
    else:
        mo.md(
            "### Data provenance\n\n"
            "`data/manifest.json` not found. Run `python scripts/reproduce.py` "
            "to regenerate every `.npz` and write the manifest."
        )
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

    [3] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M.
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
