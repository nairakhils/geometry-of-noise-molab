import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Singular gradients, conformal flows, and Fourier shrinkage

    *Akhil Nair · April 2026*
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "The paper [Sahraee-Ardakan, Delbracio & Milanfar, 2026] "
            "proves that noise-blind diffusion samplers implicitly run a "
            "Riemannian gradient flow on a singular energy landscape, and "
            "that a posterior-averaged conformal factor cancels the "
            "singularity. The figure below shows the cancellation directly "
            "on a 4-point dataset where the math has a closed form. The "
            "notebook adds an original extension: an exact Fourier-mode "
            "picture of the same flow on Gaussian random fields, with a "
            "closed-form half-power cutoff "
            "$k_c(t, n_s) = (a(t)/b(t))^{2/n_s}$."
        ),
        kind="neutral",
    )
    return


@app.cell
def _(
    a_of_t,
    b_of_t,
    denoiser_discrete,
    np,
    plt,
    probe,
    singular_gradient,
    t_marker_slider,
):
    from matplotlib.lines import Line2D as _Line2D

    _t = singular_gradient["t_axis"]
    _raw = singular_gradient["raw_grad_norm"]
    _lam = singular_gradient["lambda_bar_curves"]
    _pre = singular_gradient["preconditioned_grad_norm"]
    _env_inv = singular_gradient["envelope_inv_b_squared"]
    _env_b2 = singular_gradient["envelope_b_squared"]
    _labels = list(singular_gradient["probe_labels"])
    _centers = singular_gradient["centers"]
    _jitter = float(singular_gradient["jitter"])

    _pv = probe.value if probe is not None else None
    if isinstance(_pv, dict):
        _ux = float(_pv.get("x_value", 1.5))
        _uy = float(_pv.get("y_value", 0.0))
    else:
        _ux = float(getattr(_pv, "x_value", 1.5))
        _uy = float(getattr(_pv, "y_value", 0.0))
    _u_live = np.array([_ux, _uy])

    _a_vals = a_of_t(_t)
    _b_vals = b_of_t(_t)
    _sigma2 = _a_vals * _a_vals * _jitter + _b_vals * _b_vals
    _D_live = denoiser_discrete(_u_live, _t, _centers, _jitter)
    _integrand_live = (_u_live - _a_vals[:, None] * _D_live) / _sigma2[:, None]
    _raw_live = np.linalg.norm(_integrand_live, axis=-1)
    _lam_t = _b_vals + _b_vals * _b_vals / np.where(_a_vals > 0, _a_vals, np.nan)
    _pre_live = _lam_t * _lam_t * _raw_live

    _idx_mark = int(t_marker_slider.value)
    _t_mark = float(_t[_idx_mark])

    # Colorblind-friendly palette: viridis sampled across the probes.
    _cmap = plt.get_cmap("viridis")
    _probe_colors = [_cmap(_i / max(len(_labels) - 1, 1)) for _i in range(len(_labels))]

    # Local style override: high DPI, white background, grid only on log ticks.
    _rc = {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
    }
    with plt.rc_context(_rc):
        _fig, _axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True)

        # Panel (a) -- raw gradient
        for _i, _lab in enumerate(_labels):
            _y = _raw[_i]
            _mask = _y > 0
            _axes[0].loglog(_t[_mask], _y[_mask], color=_probe_colors[_i],
                            linewidth=1.5, alpha=0.9)
        _axes[0].loglog(_t, _env_inv, "k--", alpha=0.6, linewidth=1.2)
        _live_mask = _raw_live > 0
        _axes[0].loglog(_t[_live_mask], _raw_live[_live_mask],
                        color="crimson", linewidth=2.4)
        if _live_mask[_idx_mark]:
            _axes[0].plot(_t_mark, _raw_live[_idx_mark], "o",
                          color="crimson", markersize=10,
                          markeredgecolor="white", zorder=10)
        _axes[0].set_xlabel(r"noise level $t$")
        _axes[0].set_ylabel(r"$\|(u - a D_t^*) / \sigma^2\|$")
        _axes[0].set_title(r"raw gradient", fontsize=12, pad=8)
        _axes[0].text(0.025, 0.965, "(a)", transform=_axes[0].transAxes,
                      fontsize=12, fontweight="bold", va="top", ha="left")

        # Panel (b) -- conformal factor
        _axes[1].loglog(_t, _lam[0], color="#440154", linewidth=2.0)
        _axes[1].loglog(_t, _env_b2, "k--", alpha=0.6, linewidth=1.2)
        _axes[1].axvline(_t_mark, color="crimson", linestyle=":",
                         alpha=0.55, linewidth=1.0)
        _axes[1].set_xlabel(r"noise level $t$")
        _axes[1].set_ylabel(r"$\lambda(t)^2$")
        _axes[1].set_title(r"conformal factor", fontsize=12, pad=8)
        _axes[1].text(0.025, 0.965, "(b)", transform=_axes[1].transAxes,
                      fontsize=12, fontweight="bold", va="top", ha="left")

        # Panel (c) -- bounded product
        for _i, _lab in enumerate(_labels):
            _y = _pre[_i]
            _mask = _y > 0
            _axes[2].loglog(_t[_mask], _y[_mask], color=_probe_colors[_i],
                            linewidth=1.5, alpha=0.9)
        _live_pre_mask = _pre_live > 0
        _axes[2].loglog(_t[_live_pre_mask], _pre_live[_live_pre_mask],
                        color="crimson", linewidth=2.4)
        if _live_pre_mask[_idx_mark]:
            _axes[2].plot(_t_mark, _pre_live[_idx_mark], "o",
                          color="crimson", markersize=10,
                          markeredgecolor="white", zorder=10)
        _axes[2].set_xlabel(r"noise level $t$")
        _axes[2].set_ylabel(r"$\lambda(t)^2 \cdot \|(u - a D_t^*)/\sigma^2\|$")
        _axes[2].set_title(r"bounded product", fontsize=12, pad=8)
        _axes[2].text(0.025, 0.965, "(c)", transform=_axes[2].transAxes,
                      fontsize=12, fontweight="bold", va="top", ha="left")

        # Shared legend at the bottom.
        _handles = [
            _Line2D([0], [0], color=_probe_colors[_i], lw=2, label=_lab)
            for _i, _lab in enumerate(_labels)
        ]
        _handles += [
            _Line2D([0], [0], color="k", linestyle="--", lw=1.2,
                    label="analytic envelope"),
            _Line2D([0], [0], color="crimson", lw=2.4,
                    label=f"live probe ({_ux:.2f}, {_uy:.2f})"),
        ]
        _fig.legend(handles=_handles, loc="lower center", ncol=4,
                    fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

        _fig.tight_layout(rect=[0, 0.06, 1, 1])
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Lead figure.** Five fixed probes plus a live red probe, evaluated at every noise level $t$ on a 4-corner discrete dataset. Panel **(a)** is the per-$t$ integrand of $\nabla E_{\text{marg}}$, which tracks the dashed $1/b(t)^2$ envelope on off-data probes. Panel **(c)** is the product with $\lambda(t)^2$ from panel **(b)**, bounded across the divergent regime. The cancellation is the closed-form mechanism behind the paper's stability theorem.
    """)
    return


@app.cell
def _(TwoDSliderWidget, mo, singular_gradient):
    _n_t = len(singular_gradient["t_axis"])
    probe = mo.ui.anywidget(
        TwoDSliderWidget(x_range=[-3.0, 3.0], y_range=[-3.0, 3.0],
                         x_value=1.5, y_value=0.0)
    )
    t_marker_slider = mo.ui.slider(
        start=0, stop=_n_t - 1, step=1, value=int(_n_t * 0.4),
        label="t marker index",
    )
    mo.vstack([
        mo.md(
            "**Interactive controls** (live on cloud / wasm). Click the "
            "plane to pick a probe; the red curve in the figure above "
            "recomputes in closed form on every click."
        ),
        mo.hstack([probe, t_marker_slider], justify="start", align="start"),
    ])
    return probe, t_marker_slider


@app.cell
def _(mo):
    mo.md(r"""
    ## Contents

    1. [The marginal energy and its preconditioner](#1-the-marginal-energy-and-its-preconditioner)
    2. [Why the raw gradient is singular](#2-why-the-raw-gradient-is-singular)
    3. [Parameterization stability](#3-parameterization-stability)
    4. [OLS scaling check](#4-ols-scaling-linear-regression-recovers-the-closed-form)
    5. [Extension: Fourier-mode shrinkage](#5-extension-fourier-mode-shrinkage-on-gaussian-random-fields)
    6. [Limits](#6-limits)
    7. [References](#7-references)
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## 1. The marginal energy and its preconditioner"),
        mo.callout(
            mo.md(
                "Treating $t$ as random turns the family of conditional "
                "Gaussians into a single mixture density; the conformal "
                "factor $\\bar\\lambda(u)$ shapes its gradient."
            ),
            kind="success",
        ),
    ])
    return


@app.cell
def _():
    import asyncio
    import io

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp

    try:
        import pyodide  # noqa: F401
        IN_WASM = True
    except ImportError:
        IN_WASM = False
    return IN_WASM, asyncio, io, mo, np, plt, sp


@app.cell
def _(np, sp):
    """Inlined helpers from src/. Verbatim copies; src/ stays canonical."""
    import anywidget
    import traitlets
    from scipy.special import logsumexp

    def a_of_t(t):
        t = np.asarray(t, dtype=np.float64)
        return 1.0 - t

    def b_of_t(t):
        t = np.asarray(t, dtype=np.float64)
        return t

    def _component_log_likelihoods_discrete(u, a_vals, b_vals, centers, jitter):
        K, d = centers.shape
        sigma2 = a_vals * a_vals * jitter + b_vals * b_vals
        centers_norm = np.einsum("kd,kd->k", centers, centers)
        u_norm = np.einsum("...d,...d->...", u, u)
        cross = np.einsum("...d,kd->...k", u, centers)
        sq = (
            u_norm[..., None, None]
            - 2.0 * a_vals[:, None] * cross[..., None, :]
            + (a_vals * a_vals)[:, None] * centers_norm[None, :]
        )
        log_w_unnorm = (
            -0.5 * sq / sigma2[:, None]
            - 0.5 * d * np.log(2.0 * np.pi * sigma2)[:, None]
            - np.log(K)
        )
        return log_w_unnorm, sigma2

    def denoiser_discrete(u, t, centers, jitter):
        u = np.asarray(u, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)
        a_vals = np.atleast_1d(np.asarray(a_of_t(t), dtype=np.float64))
        b_vals = np.atleast_1d(np.asarray(b_of_t(t), dtype=np.float64))
        log_w, _ = _component_log_likelihoods_discrete(u, a_vals, b_vals, centers, jitter)
        w = np.exp(log_w - logsumexp(log_w, axis=-1, keepdims=True))
        D = np.einsum("...tk,kd->...td", w, centers)
        if D.shape[-2] == 1:
            D = D[..., 0, :]
        return D

    def validate_velocity_gain():
        a, b, t = sp.symbols("a b t", positive=True, real=True)
        a_dot = sp.symbols("adot", real=True)
        b_dot = sp.symbols("bdot", real=True)
        c = sp.Integer(-1)
        d_ = sp.Integer(1)
        det = a * d_ - b * c
        mu = (a_dot * d_ - b_dot * c) / det
        nu = (b_dot * a - a_dot * b) / det
        sub = {a: 1 - t, b: t, a_dot: -1, b_dot: 1}
        return {
            "mu_general": mu,
            "nu_general": nu,
            "mu_linear_FM": sp.simplify(mu.subs(sub)),
            "nu_linear_FM": sp.simplify(nu.subs(sub)),
        }

    def validate_noise_gain_divergence():
        a, b, t = sp.symbols("a b t", positive=True, real=True)
        a_dot, b_dot = sp.symbols("adot bdot", real=True)
        c = sp.Integer(0)
        d_ = sp.Integer(1)
        det = a * d_ - b * c
        nu = (b_dot * a - a_dot * b) / det
        sub = {a: 1 - t, b: t, a_dot: -1, b_dot: 1}
        nu_lin = sp.simplify(nu.subs(sub))
        envelope = sp.simplify(b_dot / b).subs(sub)
        return {
            "nu_noise_general": nu,
            "nu_noise_linear_FM": nu_lin,
            "envelope_linear_FM": envelope,
            "ratio_limit": sp.limit(nu_lin / envelope, t, 0, "+"),
        }

    _ESM_TWO_D_SLIDER = r"""
    function render({ model, el }) {
        const W = 220, H = 220;
        const ns = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(ns, "svg");
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute("width", W);
        svg.setAttribute("height", H);
        svg.style.border = "1px solid #888";
        svg.style.background = "#fafafa";
        svg.style.cursor = "crosshair";

        const cross_v = document.createElementNS(ns, "line");
        cross_v.setAttribute("y1", 0); cross_v.setAttribute("y2", H);
        cross_v.setAttribute("stroke", "#bbb");
        cross_v.setAttribute("stroke-dasharray", "2,2");
        svg.appendChild(cross_v);

        const cross_h = document.createElementNS(ns, "line");
        cross_h.setAttribute("x1", 0); cross_h.setAttribute("x2", W);
        cross_h.setAttribute("stroke", "#bbb");
        cross_h.setAttribute("stroke-dasharray", "2,2");
        svg.appendChild(cross_h);

        const xRange = model.get("x_range");
        const yRange = model.get("y_range");

        function toPx(x, y) {
            const px = ((x - xRange[0]) / (xRange[1] - xRange[0])) * W;
            const py = H - ((y - yRange[0]) / (yRange[1] - yRange[0])) * H;
            return [px, py];
        }
        function toData(px, py) {
            const x = xRange[0] + (px / W) * (xRange[1] - xRange[0]);
            const y = yRange[0] + (1 - py / H) * (yRange[1] - yRange[0]);
            return [x, y];
        }

        for (const [dx, dy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
            const [cx, cy] = toPx(dx, dy);
            const m = document.createElementNS(ns, "circle");
            m.setAttribute("cx", cx); m.setAttribute("cy", cy);
            m.setAttribute("r", 4); m.setAttribute("fill", "#888");
            svg.appendChild(m);
        }

        const dot = document.createElementNS(ns, "circle");
        dot.setAttribute("r", 6); dot.setAttribute("fill", "red");
        dot.setAttribute("stroke", "white"); dot.setAttribute("stroke-width", 1.5);
        svg.appendChild(dot);

        const label = document.createElement("div");
        label.style.fontFamily = "monospace";
        label.style.fontSize = "11px";
        label.style.marginTop = "4px";

        function refresh() {
            const x = model.get("x_value");
            const y = model.get("y_value");
            const [px, py] = toPx(x, y);
            dot.setAttribute("cx", px); dot.setAttribute("cy", py);
            cross_v.setAttribute("x1", px); cross_v.setAttribute("x2", px);
            cross_h.setAttribute("y1", py); cross_h.setAttribute("y2", py);
            label.textContent = `probe: (u1 = ${x.toFixed(2)}, u2 = ${y.toFixed(2)})`;
        }
        refresh();
        model.on("change:x_value", refresh);
        model.on("change:y_value", refresh);

        function pick(event) {
            const rect = svg.getBoundingClientRect();
            const px = (event.clientX - rect.left) * (W / rect.width);
            const py = (event.clientY - rect.top) * (H / rect.height);
            const [x, y] = toData(px, py);
            model.set("x_value", x);
            model.set("y_value", y);
            model.save_changes();
        }
        svg.addEventListener("click", pick);

        el.appendChild(svg);
        el.appendChild(label);
    }
    export default { render };
    """

    class TwoDSliderWidget(anywidget.AnyWidget):
        _esm = _ESM_TWO_D_SLIDER
        x_value = traitlets.Float(1.5).tag(sync=True)
        y_value = traitlets.Float(0.0).tag(sync=True)
        x_range = traitlets.List(traitlets.Float(), default_value=[-3.0, 3.0]).tag(sync=True)
        y_range = traitlets.List(traitlets.Float(), default_value=[-3.0, 3.0]).tag(sync=True)

        def __init__(self, x_range=None, y_range=None, x_value=1.5, y_value=0.0, **kwargs):
            traits = {"x_value": float(x_value), "y_value": float(y_value)}
            if x_range is not None:
                traits["x_range"] = [float(v) for v in x_range]
            if y_range is not None:
                traits["y_range"] = [float(v) for v in y_range]
            super().__init__(**traits, **kwargs)

    return (
        TwoDSliderWidget,
        a_of_t,
        b_of_t,
        denoiser_discrete,
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
async def _(IN_WASM, asyncio, io, np):
    import json as _json

    DATA_BASE_REMOTE = (
        "https://raw.githubusercontent.com/nairakhils/"
        "geometry-of-noise-molab/main/data"
    )
    DATA_NAMES = (
        "energy_landscape_2d.npz",
        "stability_curves.npz",
        "grf_gallery.npz",
        "shrinkage_heatmap.npz",
        "linear_score_fit.npz",
        "singular_gradient.npz",
        "grf_flow_strip.npz",
    )

    async def _fetch_bytes(name):
        if not IN_WASM:
            for _prefix in ("data", "../data"):
                try:
                    with open(f"{_prefix}/{name}", "rb") as _f:
                        return _f.read()
                except FileNotFoundError:
                    continue
        if IN_WASM:
            import pyodide.http
            _resp = await pyodide.http.pyfetch(f"{DATA_BASE_REMOTE}/{name}")
            return await _resp.bytes()
        import urllib.request
        return urllib.request.urlopen(f"{DATA_BASE_REMOTE}/{name}").read()

    async def _fetch_npz(name):
        _b = await _fetch_bytes(name)
        with np.load(io.BytesIO(_b)) as _z:
            return {_k: np.asarray(_z[_k]) for _k in _z.files}

    _loaded = await asyncio.gather(*[_fetch_npz(_n) for _n in DATA_NAMES])
    energy, stability, gallery, shrinkage, linear_fit, singular_gradient, grf_flow_strip = _loaded

    _manifest_bytes = await _fetch_bytes("manifest.json")
    manifest = _json.loads(_manifest_bytes.decode("utf-8"))
    return (
        energy,
        gallery,
        grf_flow_strip,
        linear_fit,
        manifest,
        shrinkage,
        singular_gradient,
        stability,
    )


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
    _xs = _u[:, 0, 0]
    _ys = _u[0, :, 1]

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
    mo.vstack([
        mo.md(
            "On smooth Gaussian data the marginal energy has no Dirac atoms, "
            "so neither field above is singular. The conformal preconditioner "
            "still reshapes the flow: $\\bar\\lambda(u)$ contracts the field "
            "more where the posterior $p(t \\mid u)$ concentrates on small "
            "$t$. Hover for exact $(u_1, u_2, \\text{value})$ readouts."
        ),
        mo.accordion({
            "Mathematical detail: Tweedie identity (collapse to expand)":
            mo.md(
                "$$\\nabla_u \\log p(u \\mid t) = \\frac{a(t) D_t^*(u) - u}{b(t)^2}"
                "\\qquad \\text{(Tweedie / Robbins, paper Eq. 10)}$$\n\n"
                "$$\\nabla_u E_{\\text{marg}}(u) = \\mathbb{E}_{t \\mid u}"
                "\\!\\left[\\frac{u - a(t)\\,D_t^*(u)}{b(t)^2}\\right]"
                "\\qquad \\text{(paper Eq. 11)}$$"
            ),
        }),
        mo.md("[↑ Top](#singular-gradients-conformal-flows-and-fourier-shrinkage)"),
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## 2. Why the raw gradient is singular"),
        mo.callout(
            mo.md(
                "Near a clean datum the posterior $p(t \\mid u)$ collapses on "
                "small $t$ where $b(t) \\to 0$, so the integrand inherits the "
                "$1/b^2$ blow-up; $\\lambda(t)^2$ vanishes at the matching "
                "$b^2$ rate and the product stays bounded."
            ),
            kind="success",
        ),
    ])
    return


@app.cell
def _(np, plt, singular_gradient):
    _t = singular_gradient["t_axis"]
    _pre = singular_gradient["preconditioned_grad_norm"]
    _labels = list(singular_gradient["probe_labels"])
    _fit_mask = (_t >= 0.01) & (_t <= 0.5)

    _fig, _ax = plt.subplots(figsize=(9.5, 4.4))
    for _i, _lab in enumerate(_labels):
        _y = _pre[_i]
        _mask = _y > 0
        if (_y[_fit_mask] > 0).all() and len(_y[_fit_mask]) > 2:
            _slope = float(np.polyfit(np.log(_t[_fit_mask]),
                                      np.log(_y[_fit_mask]), 1)[0])
            _slope_str = "flat" if abs(_slope) < 0.05 else f"slope {_slope:+.2f}"
        else:
            _slope_str = "n/a"
        _ax.loglog(_t[_mask], _y[_mask], linewidth=1.6,
                   label=f"{_lab}  ({_slope_str})")
    _ax.axvspan(0.01, 0.5, color="#fff4dc", alpha=0.5, zorder=0,
                label="divergent regime $b > \\sqrt{\\mathrm{jitter}}$")
    _ax.set_xlabel("t")
    _ax.set_ylabel(r"$\lambda(t)^2 \cdot \|(u - a D_t^*) / \sigma^2\|$")
    _ax.set_title("Bounded product: the cancellation in close-up")
    _ax.legend(fontsize=8, loc="upper right")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            "On-data probes (at corners or on the symmetry axis) shrink to "
            "zero. Off-data probes plateau at a probe-dependent constant. "
            "The shaded band marks $t \\in [0.01, 0.5]$, where the raw "
            "gradient diverges as $1/b^2$; slopes in the legend are log-log "
            "fits over that band."
        ),
        mo.accordion({
            "Mathematical detail: the $1/b^2$ envelope (collapse to expand)":
            mo.md(
                "For a probe $u$ near a datum $x_k$ and small $t$:\n\n"
                "$$u - a(t)\\,D_t^*(u) \\;\\approx\\; \\|u - x_k\\|"
                "\\qquad (a \\to 1,\\; D_t^* \\to x_k)$$\n\n"
                "$$\\sigma^2(t) \\;\\approx\\; b(t)^2 + a(t)^2 \\cdot "
                "\\mathrm{jitter} \\;\\to\\; b(t)^2"
                "\\qquad (\\mathrm{jitter} \\ll b^2)$$\n\n"
                "$$\\Rightarrow \\quad \\Bigl\\| \\frac{u - a D_t^*}"
                "{\\sigma^2} \\Bigr\\| \\;\\approx\\; \\frac{\\|u - x_k\\|}"
                "{b(t)^2} \\quad \\square$$"
            ),
        }),
        mo.md("[↑ Top](#singular-gradients-conformal-flows-and-fourier-shrinkage)"),
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## 3. Parameterization stability"),
        mo.callout(
            mo.md(
                "Velocity gain is identically 1; the noise envelope diverges "
                "as $|\\dot b / b| = 1/t$ under FM linear."
            ),
            kind="success",
        ),
    ])
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
    _axes[0].set_title(r"(a) sampler gain coefficient $\nu(t)$ vs envelope")
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
def _(mo, sp, validate_noise_gain_divergence, validate_velocity_gain):
    _v = validate_velocity_gain()
    _n = validate_noise_gain_divergence()
    mo.callout(
        mo.md(
            f"**Symbolic verification (sympy).** "
            f"Velocity, $(c, d) = (-1, 1)$ under FM linear: "
            f"$\\nu(t) = {sp.latex(_v['nu_linear_FM'])}$ "
            f"(paper Eq. 70).  "
            f"Noise, $(c, d) = (0, 1)$: "
            f"$\\nu(t) = {sp.latex(_n['nu_noise_linear_FM'])}$, "
            f"envelope $\\dot b / b = {sp.latex(_n['envelope_linear_FM'])}$, "
            f"and $\\lim_{{t \\to 0^+}} \\nu / (\\dot b/b) = "
            f"{sp.latex(_n['ratio_limit'])}$. "
            f"The literal noise $\\nu$ is bounded near the manifold under "
            f"FM linear; the divergence the paper attributes to noise "
            f"prediction lives in $\\dot b / b$, not in $\\nu$ itself."
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    [↑ Top](#singular-gradients-conformal-flows-and-fourier-shrinkage)
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## 4. OLS scaling: linear regression recovers the closed form"),
        mo.callout(
            mo.md(
                "Each decade in sample size $N$ shifts the relative "
                "Frobenius error by $\\sqrt{10}$, exactly as Gaussian "
                "linear-regression theory predicts."
            ),
            kind="success",
        ),
    ])
    return


@app.cell
def _(linear_fit, mo):
    _Ns = linear_fit["n_samples_list"]
    n_highlight_slider = mo.ui.slider(
        start=0, stop=len(_Ns) - 1, step=1, value=2,
        label=f"highlight N = (default 10^{int(round(__import__('math').log10(int(_Ns[2]))))})",
    )
    return (n_highlight_slider,)


@app.cell
def _(linear_fit, mo, n_highlight_slider, plt):
    _t = linear_fit["t_values"]
    _Ns = linear_fit["n_samples_list"]
    _eps_grid = linear_fit["rel_err_eps_grid"]
    _v_grid = linear_fit["rel_err_v_grid"]
    _hi = int(n_highlight_slider.value)

    _fig, _ax = plt.subplots(figsize=(8.5, 5.0))
    _cmap = plt.get_cmap("viridis")
    for _j, _N in enumerate(_Ns):
        _color = _cmap(_j / max(len(_Ns) - 1, 1))
        _alpha = 1.0 if _j == _hi else 0.35
        _lw_eps = 2.4 if _j == _hi else 1.2
        _lw_v = 2.0 if _j == _hi else 1.0
        _ax.semilogy(_t, _eps_grid[_j], "o-", color=_color, linewidth=_lw_eps,
                     alpha=_alpha, label=f"eps, N = {int(_N):>7d}")
        _ax.semilogy(_t, _v_grid[_j], "s--", color=_color, linewidth=_lw_v,
                     alpha=_alpha * 0.85, label=f"v,   N = {int(_N):>7d}")
        _ax.axhline(float(_N) ** -0.5, color=_color, ls=":",
                    alpha=_alpha * 0.55, linewidth=0.9)
    _ax.set_xlabel("t")
    _ax.set_ylabel(r"$\|A_{\mathrm{OLS}} - A_{\mathrm{exact}}\|_F / \|A_{\mathrm{exact}}\|_F$")
    _ax.set_title(f"OLS recovery vs N (highlighted: N = {int(_Ns[_hi])})")
    _ax.legend(ncol=2, fontsize=8, loc="upper right")
    _fig.tight_layout()

    mo.hstack([n_highlight_slider, _fig], justify="start", align="start", widths=[1, 4])
    return


@app.cell
def _(mo):
    mo.md("""
    [↑ Top](#singular-gradients-conformal-flows-and-fourier-shrinkage)
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## 5. Extension: Fourier-mode shrinkage on Gaussian random fields"),
        mo.callout(
            mo.md(
                "**Notebook contribution beyond the paper.** For an isotropic "
                "Gaussian random field with $P(k) \\propto k^{-n_s}$, the "
                "covariance is diagonal in the Fourier basis and the autonomous "
                "flow factorizes mode by mode. The closed-form half-power "
                "cutoff $k_c(t, n_s) = (a/b)^{2/n_s}$ traces the moving "
                "signal-vs-noise boundary."
            ),
            kind="info",
        ),
    ])
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
    gallery_block = _fig
    return (gallery_block,)


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
    psd_block = _fig
    return (psd_block,)


@app.cell
def _(mo, shrinkage):
    _n_t = len(shrinkage["t_values"])
    shrink_t_slider = mo.ui.slider(
        start=0, stop=_n_t - 1, step=1, value=_n_t // 2, label="t index"
    )
    shrink_ns_dropdown = mo.ui.dropdown(
        options=["1", "2", "3"], value="2", label="spectral index n_s"
    )
    return shrink_ns_dropdown, shrink_t_slider


@app.cell
def _(mo, plt, shrink_ns_dropdown, shrink_t_slider, shrinkage):
    _ns = int(shrink_ns_dropdown.value)
    _idx = int(shrink_t_slider.value)
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
    shrinkage_block = mo.hstack(
        [mo.vstack([shrink_t_slider, shrink_ns_dropdown]), _fig],
        justify="start", align="start", widths=[1, 4],
    )
    return (shrinkage_block,)


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
        start=0, stop=flow_n_steps - 1, step=1, value=flow_n_steps // 2,
        label="reverse step",
    )
    flow_play_speed = mo.ui.refresh(
        options=["off", "0.5s", "0.25s"], default_interval="off", label="auto-play",
    )
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
    _refresh_value = flow_play_speed.value
    _is_playing = _refresh_value not in ("off", None)
    if _is_playing:
        set_flow_step((get_flow_step() + 1) % flow_n_steps)
    elif flow_step_slider.value != get_flow_step():
        set_flow_step(int(flow_step_slider.value))
    return


@app.cell
def _(
    flow_fwd_at_traj,
    flow_play_speed,
    flow_step_slider,
    flow_strip_fwd,
    flow_strip_rev,
    flow_strip_t,
    flow_traj,
    flow_traj_t,
    flow_vmax,
    get_flow_step,
    grf_flow_strip,
    mo,
    plt,
):
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
    _live = mo.ui.plotly(_fig_live)

    # Static 2x4 fallback strip
    _fwd_t_static = grf_flow_strip["forward_t_values"]
    _fig_static, _axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for _j in range(4):
        _ax = _axes[0, _j]
        _ax.imshow(flow_strip_fwd[_j], cmap="RdBu_r",
                   vmin=-flow_vmax, vmax=flow_vmax)
        _ax.set_xticks([]); _ax.set_yticks([]); _ax.grid(False)
        _ax.set_title(f"fwd t = {float(_fwd_t_static[_j]):.3f}", fontsize=10)
    _rev_order = list(range(3, -1, -1))
    for _col, _src in enumerate(_rev_order):
        _ax = _axes[1, _col]
        _ax.imshow(flow_strip_rev[_src], cmap="RdBu_r",
                   vmin=-flow_vmax, vmax=flow_vmax)
        _ax.set_xticks([]); _ax.set_yticks([]); _ax.grid(False)
        _ax.set_title(f"rev t = {float(flow_strip_t[_src]):.3f}", fontsize=10)
    _axes[0, 0].set_ylabel("forward", fontsize=11)
    _axes[1, 0].set_ylabel("reverse", fontsize=11)
    _fig_static.tight_layout()

    flow_block = mo.vstack([
        mo.hstack([flow_step_slider, flow_play_speed], justify="start"),
        _live,
        mo.accordion({
            "Static 2x4 strip (fallback for static export / PDF)":
            _fig_static,
        }),
    ])
    return (flow_block,)


@app.cell
def _(mo, plt, shrinkage):
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
    _fig.tight_layout()

    kc_block = mo.vstack([
        _fig,
        mo.callout(
            mo.md(
                "$$k_c(t, n_s) \\;=\\; \\bigl(a(t)/b(t)\\bigr)^{2/n_s}$$"
                "  Setting the per-mode signal-fraction $W = 1/2$ in "
                "$a^2 \\sigma_k^2 = b^2$ with $\\sigma_k^2 = k^{-n_s}$ "
                "gives the closed form."
            ),
            kind="info",
        ),
    ])
    return (kc_block,)


@app.cell
def _(flow_block, gallery_block, kc_block, mo, psd_block, shrinkage_block):
    mo.ui.tabs({
        "Gallery": gallery_block,
        "PSD": psd_block,
        "Shrinkage W(k,t)": shrinkage_block,
        "Forward / reverse flow": flow_block,
        "k_c cutoff": kc_block,
    })
    return


@app.cell
def _(mo):
    mo.md("""
    Each tab is self-contained. The shrinkage tab carries its own $t$ slider and $n_s$ dropdown bound to the heatmap; the forward/reverse-flow tab carries a scrub-and-play widget plus a static 2x4 strip fallback in a collapsed accordion. The $k_c$ closed form is what the paper does not state explicitly: it isolates the moving boundary between signal-preserved and noise-dominated modes as a function of $(t, n_s)$.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    [↑ Top](#singular-gradients-conformal-flows-and-fourier-shrinkage)
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## 6. Limits"),
        mo.callout(
            mo.md(
                "**1.** No image-generation experiments. We do not retrain "
                "DDPM / Flow-Matching U-Nets on CIFAR-10, SVHN, or Fashion-MNIST."
            ),
            kind="warn",
        ),
        mo.callout(
            mo.md(
                "**2.** The exact story assumes linear-Gaussian data; the "
                "paper's manifold + discrete-set analysis (Appendix E) is not "
                "implemented here."
            ),
            kind="warn",
        ),
        mo.callout(
            mo.md(
                "**3.** No neural score network. The OLS sanity check fits a "
                "linear estimator whose closed form is already known."
            ),
            kind="warn",
        ),
        mo.callout(
            mo.md(
                "**4.** The GRF extension is isotropic only. Anisotropic "
                "covariances induce mode coupling we did not implement."
            ),
            kind="warn",
        ),
        mo.md("[↑ Top](#singular-gradients-conformal-flows-and-fourier-shrinkage)"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. References

    - **[1]** Sahraee-Ardakan, M., Delbracio, M., Milanfar, P. *The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning.* arXiv:2602.18428v1, Google, February 2026.
    - **[2]** Sun, Q., Jiang, Z., Zhao, H., He, K. *Is noise conditioning necessary for denoising generative models?* arXiv:2502.13129, 2025.
    - **[3]** Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. *Flow Matching for Generative Modeling.* arXiv:2210.02747, 2023.
    - **[4]** Wang, R., Du, Y. *Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models.* arXiv:2510.02300, 2025.
    - **[5]** Salimans, T., Ho, J. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022.

    Source: `geometry-of-noise-molab/`. See `docs/paper_summary.md` and
    `docs/implementation_notes.md` for the equation extraction and numerical caveats.

    [↑ Top](#singular-gradients-conformal-flows-and-fourier-shrinkage)
    """)
    return


@app.cell
def _(manifest, mo):
    _entries = "\n".join(f"- `{_k}`: `{_v}`" for _k, _v in manifest.items())
    mo.accordion({
        "Data provenance manifest (collapse to expand)":
        mo.md(
            "Each `.npz` under `data/` was produced by `scripts/reproduce.py` "
            "with fixed seeds. SHA-256 prefixes are fetched live alongside "
            "the data; rerun `python scripts/reproduce.py` to regenerate.\n\n"
            + _entries
        ),
    })
    return


if __name__ == "__main__":
    app.run()
