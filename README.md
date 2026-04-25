# Singular gradients, conformal flows, and Fourier shrinkage

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

`numpy`, `scipy`, `matplotlib`, and `marimo` are the only runtime
requirements (see `pyproject.toml`). Python 3.11 or newer.

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
