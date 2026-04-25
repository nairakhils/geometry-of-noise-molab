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
`data/manifest.json`. The committed `manifest.json` is a fingerprint
of one specific build (macOS / Python 3.14 / numpy 2.4.4 / Apple
Accelerate). CI re-runs `reproduce.py` on every push and reports any
drift in the manifest, but does **not** fail on drift.

**Cross-platform / cross-run note.** Byte-exact reproduction across
machines turns out to be unachievable for this codebase. Two of the
seven files (`energy_landscape_2d.npz` and one or two more depending
on the run) are stable across macOS and Linux, but the remaining ones
have FFT-plan and LAPACK-reduction order non-determinism that can
shift bytes even between back-to-back Linux CI runs at single-thread
BLAS. The numerical content is consistent across all runs (medians
match to printed precision; pytest passes; sympy gives the same
expressions); only the SHA-256 fingerprints drift. Treat the manifest
as a useful change-detector for *intentional* edits to the precompute
scripts, not as a strict reproducibility lock.

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
