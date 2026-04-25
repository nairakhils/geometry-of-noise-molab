# geometry-of-noise-molab

A marimo notebook reproducing the analytic results of Sahraee-Ardakan,
Delbracio & Milanfar, *The Geometry of Noise: Why Diffusion Models Don't
Need Noise Conditioning* ([arXiv:2602.18428](https://arxiv.org/abs/2602.18428),
February 2026), on closed-form Gaussian data and isotropic 2D Gaussian random
fields. The notebook computes the marginal energy, its raw and preconditioned
gradients, the per-parameterization sampler gain, and the Drift Perturbation
Error in closed form; a separate linear-OLS sanity check certifies the
conditional-mean derivation against samples without any neural network.

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

`numpy`, `scipy`, `matplotlib`, and `marimo` are the only runtime requirements
(see `pyproject.toml`). Python 3.11 or newer.

## Layout

- `notebooks/walkthrough.py`: the marimo notebook.
- `src/`: math layer (`exact_affine.py`, `grf_2d.py`, `schedules.py`).
- `tests/`: 29 pytest tests covering the math layer.
- `data/`: precomputed `.npz` arrays produced by `scripts/precompute_arrays.py`
  and `scripts/linear_score_fit.py`. Tracked in git so the notebook loads
  without re-running the precompute step.
- `docs/paper_summary.md`: equation-by-equation extraction of the paper.
- `docs/implementation_notes.md`: numerical decisions and a per-function
  confidence ledger.
- ``: prose audit of the notebook.
- `notebooks/__marimo__/session/`: rendered cell outputs for the GitHub
  static preview.

## Limits

We do not reproduce the paper's CIFAR-10, SVHN, or Fashion-MNIST experiments.
The exact story assumes linear-Gaussian data; the paper's general results on
discrete data sets and smooth manifolds are not implemented here. The GRF
extension is isotropic only.

## Attribution

Submission for the alphaXiv × marimo notebook competition, April 2026.
