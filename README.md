# Singular gradients, conformal flows, and Fourier shrinkage

[![reproduce](https://github.com/nairakhils/geometry-of-noise-molab/actions/workflows/reproduce.yml/badge.svg)](https://github.com/nairakhils/geometry-of-noise-molab/actions/workflows/reproduce.yml)

A closed-form reading of Sahraee-Ardakan, Delbracio & Milanfar,
*The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning*
([arXiv:2602.18428](https://arxiv.org/abs/2602.18428), February 2026).

The marginal energy `E_marg(u) = -log integral p(u|t) p(t) dt` has a
`1/b(t)^2` singularity at every clean data point, so the raw gradient
diverges as the noise level shrinks. The lead figure of the notebook
evaluates that integrand on four discrete data points at the corners of
`[-1, 1]^2` and shows the raw norm tracking the `1/b^2` envelope while the
paper's conformal factor `lambda(t) = b + b^2/a` trims the divergence.
The original-claim part of the notebook is a Fourier-mode shrinkage picture
for isotropic 2D Gaussian random fields, including the closed-form
half-power cutoff `k_c(t, n_s) = (a/b)^(2/n_s)`, that the paper does not
draw.

## Open in molab (cloud)

[Run on molab](https://molab.marimo.io/github/nairakhils/geometry-of-noise-molab/blob/main/notebooks/walkthrough.py)

WebAssembly preview (Pyodide):
[/wasm variant](https://molab.marimo.io/github/nairakhils/geometry-of-noise-molab/blob/main/notebooks/walkthrough.py/wasm).

## Run locally

```
git clone https://github.com/nairakhils/geometry-of-noise-molab.git
cd geometry-of-noise-molab
uv pip install -e .
marimo edit notebooks/walkthrough.py
```

`numpy`, `scipy`, `matplotlib`, `marimo`, `plotly`, and `sympy` are the
runtime requirements (see `pyproject.toml`). Python 3.11 or newer.

## Reproduce from scratch

```
git clone https://github.com/nairakhils/geometry-of-noise-molab.git
cd geometry-of-noise-molab
uv pip install -e .
python scripts/reproduce.py
```

This regenerates every `.npz`, runs the test suite, and writes
`data/manifest.json`. The committed `manifest.json` is the reference,
generated on the CI runner (Linux, Python 3.11, OpenBLAS); CI re-runs
the same command on every push and asserts the manifest is unchanged.

**Cross-platform note.** Three of the seven files
(`energy_landscape_2d.npz`, `grf_flow_strip.npz`, `shrinkage_heatmap.npz`)
hash identically on macOS and Linux. The remaining four
(`grf_gallery.npz`, `linear_score_fit.npz`, `singular_gradient.npz`,
`stability_curves.npz`) have BLAS / LAPACK / FFT-routine differences
that produce small floating-point variations across platforms; their
SHA-256 prefixes will drift between Apple Accelerate and OpenBLAS even
with identical seeds. The numerical content is consistent (medians and
empirical statistics match to printed precision); only the bytes
differ. The committed manifest tracks the Linux build; macOS users
running `reproduce.py` will see drift in the four sensitive files but
identical figures and test results.

## Layout

- `notebooks/walkthrough.py`: the marimo notebook.
- `src/`: math layer (`exact_affine.py`, `grf_2d.py`, `schedules.py`).
- `tests/`: pytest cases covering the math layer.
- `data/`: precomputed `.npz` arrays produced by
  `scripts/precompute_arrays.py` and `scripts/linear_score_fit.py`.
  Tracked in git so the notebook loads without re-running the precompute
  step.
- `docs/paper_summary.md`: equation-by-equation extraction of the paper.
- `docs/implementation_notes.md`: numerical decisions and a per-function
  confidence ledger.
- ``: prose audit of the notebook.
- ``: pre-submission status report.
- `notebooks/__marimo__/session/`: rendered cell outputs for molab's
  preview.

## Limits

We do not reproduce the paper's CIFAR-10, SVHN, or Fashion-MNIST
experiments. The exact story assumes linear-Gaussian data; the paper's
general results on discrete data sets and smooth manifolds are not
implemented here. The GRF extension is isotropic only.

## Attribution

Akhil Nair, . Submission for the
alphaXiv x marimo notebook competition, April 2026.
