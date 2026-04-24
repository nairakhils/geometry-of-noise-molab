# Paper summary — "The Geometry of Noise"

**Reference.** Sahraee-Ardakan, M., Delbracio, M., Milanfar, P.
*The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning.*
arXiv:2602.18428v1, Google, 20 February 2026.

All equation numbers below refer to the v1 PDF.

---

## (a) Affine noising family — Eq. (2)

The forward process is the unified affine family of Sun et al. [30]. Let
`t ∈ [0,1]` index the noise level. For clean data `x` and `ε ~ N(0, I)`:

$$ \mathbf{u}_t \;=\; a(t)\,\mathbf{x} \;+\; b(t)\,\boldsymbol{\epsilon}. \tag{2} $$

Signal-to-noise ratio:

$$ \mathrm{SNR}(t) \;=\; \frac{a(t)^2}{b(t)^2}. \tag{3} $$

Common specializations (Table 1, p. 4):

| Model | a(t)            | b(t)            | c(t) | d(t) |
| ----- | --------------- | --------------- | ---- | ---- |
| DDPM  | √ᾱ_t            | √(1 − ᾱ_t)      | 0    | 1    |
| EDM   | 1               | σ_t             | 1    | 0    |
| FM    | 1 − t           | t               | −1   | 1    |
| EqM   | 1 − t           | t               | −t   | t    |

DDPM uses VP coefficients `a² + b² = 1`. The paper's theory is *general* over
this family.

**Confidence: 5/5.**

---

## (b) Marginal energy — Eq. (1), (8)

Treat `t` as a random variable with prior `p(t)` and define the marginal
density and energy:

$$ p(\mathbf{u}) \;=\; \int p(\mathbf{u}\mid t)\,p(t)\,dt, \qquad
   E_{\mathrm{marg}}(\mathbf{u}) \;=\; -\log p(\mathbf{u}). \tag{1, 8} $$

`p(u | t)` is the conditional density of `u_t` given the noise level. For a
discrete data set `X = {x_k}_{k=1}^N` it is a Gaussian mixture (Eq. 37):

$$ p(\mathbf{u}\mid t) \;=\; \frac{1}{N}\sum_{k=1}^N
   \mathcal{N}\!\bigl(\mathbf{u};\, a(t)\mathbf{x}_k,\, b(t)^2 I\bigr). $$

**Confidence: 5/5.**

---

## (c) Raw gradient with the 1/b(t)² singularity — Eq. (9)–(12)

Differentiating the mixture under the integral, using Tweedie's formula:

$$ \nabla_\mathbf{u}\log p(\mathbf{u}\mid t)
   \;=\; \frac{a(t)\,D_t^*(\mathbf{u}) - \mathbf{u}}{b(t)^2}, \tag{10} $$

where `D_t^*(u) ≡ E[x | u, t]` is the optimal denoiser. Substituting:

$$ \nabla_\mathbf{u} E_{\mathrm{marg}}(\mathbf{u})
   \;=\; \mathbb{E}_{t\mid \mathbf{u}}
         \!\left[\frac{\mathbf{u} - a(t)\,D_t^*(\mathbf{u})}{b(t)^2}\right]. \tag{11} $$

The singularity is explicit: the integrand carries a `1/b(t)²` kernel, and on
the data manifold

$$ \lim_{\mathbf{u}\to\mathbf{x}_k}\;
   \bigl\|\nabla_\mathbf{u} E_{\mathrm{marg}}(\mathbf{u})\bigr\| \;=\; \infty. \tag{12} $$

The marginal energy is an infinitely deep potential well at every clean datum
(Fig. 1). The Hessian eigenvalues scale as `1/t_min²` if the time integral is
truncated at `t_min`.

**Confidence: 5/5.**

---

## (d) Autonomous field decomposition and the conformal preconditioner — Eq. (13)–(18), (47)–(56)

The optimal noise-blind target (Lemma 2, Eq. 7 / 24) is

$$ f^*(\mathbf{u}) \;=\; \mathbb{E}_{t\mid \mathbf{u}}\!\left[
     \frac{d(t)}{b(t)}\,\mathbf{u}
     + \left(c(t) - \frac{d(t)\,a(t)}{b(t)}\right) D_t^*(\mathbf{u})
   \right]. \tag{7, 13} $$

Defining the **effective gradient gain**

$$ \boxed{\;\lambda(t) \;\triangleq\;
   \frac{b(t)}{a(t)}\,\bigl(d(t)\,a(t) - c(t)\,b(t)\bigr),\qquad
   \bar\lambda(\mathbf{u}) \;\triangleq\; \mathbb{E}_{t\mid \mathbf{u}}[\lambda(t)]\;} \tag{15, 54} $$

and the linear-drift coefficient `c̄_scale(u) = E_{t|u}[c(t)/a(t)]`, Appendix D
establishes the **General Energy-Aligned Decomposition** (Eq. 14, 56):

$$ f^*(\mathbf{u})
   \;=\; \underbrace{\bar\lambda(\mathbf{u})\,\nabla E_{\mathrm{marg}}(\mathbf{u})}_{\text{natural gradient}}
   \;+\; \underbrace{\mathrm{Cov}_{t\mid\mathbf{u}}\!\bigl(\lambda(t),\,\nabla E_t(\mathbf{u})\bigr)}_{\text{transport correction}}
   \;+\; \underbrace{\bar c_{\mathrm{scale}}(\mathbf{u})\,\mathbf{u}}_{\text{linear drift}}. \tag{14, 56} $$

This is the **Riemannian gradient flow** picture. The scalar field
`ḡ(u) ≡ 1/λ̄(u)` plays the role of a **local conformal metric**: the autonomous
target moves like a *natural gradient* `g⁻¹∇E_marg = λ̄·∇E_marg` with respect to
that metric, plus two corrections that the paper proves vanish in two limits:

- **Regime I — global high-dimensional concentration (§5.2, Eq. 16, 17).**
  When data lies on a `d`-dim manifold inside `R^D` with `D ≫ d`, the noisy
  shells become disjoint, `p(t|u)` collapses to a delta, and the transport
  correction vanishes — the field is `f* ≈ λ̄(u)·∇E_marg(u) + c̄_scale(u)u`.
- **Regime II — local near-manifold proximity (§5.3, Eq. 18; Lemmas 5, 6).**
  As `u → X`, `p(t|u)` again concentrates (weakly to `δ(t)`) provided
  codimension `D − d > 2`. Both `∇E_marg` and `∇E_t` blow up at rate
  `O(1/b(t))`, but `λ̄(u) → 0` at exactly the matching rate, so the product
  stays bounded. This is the **conformal preconditioning** — a Riemannian
  metric whose volume element vanishes precisely where the Euclidean gradient
  diverges.

**Confidence: 5/5.**

---

## (e) Sampler ODE — Eq. (19)–(22), (61)–(64)

Differentiating the forward process gives `u̇ = ȧ(t)x + ḃ(t)ε`. Combined with
the autonomous prediction `f*(u) = c(t)x + d(t)ε` and the observation
`u = a(t)x + b(t)ε`, Appendix F.1 inverts the 2×2 linear system

$$ \begin{bmatrix}\mathbf{u}\\ f^*(\mathbf{u})\end{bmatrix}
   \;=\; \begin{bmatrix}a(t)&b(t)\\ c(t)&d(t)\end{bmatrix}
         \begin{bmatrix}\mathbf{x}\\ \boldsymbol\epsilon\end{bmatrix} \tag{62} $$

to obtain the **unified sampler ODE**

$$ \boxed{\;\frac{d\mathbf{u}}{dt}
   \;=\; \mu(t)\,\mathbf{u} \;+\; \nu(t)\,f^*(\mathbf{u})\;} \tag{19, 63} $$

with **explicit closed forms**

$$ \mu(t) \;=\; \frac{\dot a(t)\,d(t) - \dot b(t)\,c(t)}{a(t)\,d(t) - b(t)\,c(t)},
   \qquad
   \nu(t) \;=\; \frac{\dot b(t)\,a(t) - \dot a(t)\,b(t)}{a(t)\,d(t) - b(t)\,c(t)}. \tag{63} $$

Drift Perturbation Error vs. an oracle that has `t` (Eq. 22, 64):

$$ \Delta\mathbf{v}(\mathbf{u},t) \;=\; |\nu(t)|\cdot
   \bigl\|f^*(\mathbf{u}) - f_t^*(\mathbf{u})\bigr\|. $$

`μ(t)` is an autonomous-schedule property (drift) and `ν(t)` is the
parameterization-dependent **effective gain** that multiplies the estimation
error. Stability is a race between vanishing posterior uncertainty in `f*−f_t*`
and possibly diverging `ν(t)`.

**Confidence: 5/5.**

---

## (f) Three parameterization targets — Table 1, Appendix E

The training target is `r(x, ε, t) = c(t) x + d(t) ε` (Eq. 4). Three canonical
choices:

1. **Noise prediction (DDPM/DDIM): `c = 0, d = 1`.**
   Target is `ε`. Autonomous field collapses to
   $$ f^*(\mathbf{u}) \;=\; \mathbb{E}_{t\mid\mathbf{u}}\!\left[
        \frac{\mathbf{u} - a(t)\,D_t^*(\mathbf{u})}{b(t)}\right]
        \;=\; \mathbb{E}_{t\mid\mathbf{u}}[\boldsymbol\epsilon_t^*(\mathbf{u})]. $$
2. **Signal / data prediction (EDM): `c = 1, d = 0`.**
   Target is `x`. `f*(u) = E_{t|u}[D_t^*(u)]`.
3. **Velocity prediction (Flow Matching): `c = -1, d = 1`.**
   Target in *this paper's convention* is `v = u̇ = ȧ(t)x + ḃ(t)ε`. With FM
   coefficients (a = 1−t, b = t) this gives `v = ε − x`, and
   $$ f^*(\mathbf{u}) \;=\; \mathbb{E}_{t\mid\mathbf{u}}\!\left[
        \frac{\mathbf{u} - D_t^*(\mathbf{u})}{t}\right]. $$

⚠ **Notation flag.** The user prompt references the Salimans–Ho velocity
`v = α_t ε − σ_t x` (Progressive Distillation, 2022). The paper does **not**
use that convention. It defines velocity strictly as the time derivative
`v = u̇`, which corresponds to `(c, d) = (ȧ, ḃ)` of the schedule, not
`(c, d) = (−σ_t, α_t)`. For VP-SDE these two definitions are *not*
algebraically equivalent. The reproduction notebook should use the paper's
convention `v = u̇` and note this difference in a margin remark.

Equilibrium Matching (`c = −t, d = t`) is also covered (Table 1) but is not one
of the user's three requested targets.

**Confidence: 4/5** (paper-side formulas are exact; one notational discrepancy
with the user's prompt flagged above).

---

## (g) Stability claim — Section 6, Table 2, Appendix F.2

The paper does **not** present this as a numbered Proposition / Theorem. It is
a *case-by-case derivation* in §F.2 with the Drift Perturbation Error
`Δv = |ν(t)|·‖f*(u) − f_t^*(u)‖`. Near the manifold (`a → 1`, `b → 0`):

| Parameterization | `ν(t)` scaling                           | Error scaling                     | Verdict           |
| ---------------- | ---------------------------------------- | --------------------------------- | ----------------- |
| Noise (ε)        | `ν(t) ≈ ḃ(t) = O(1/b(t))` for `b ∝ √t`*  | finite Jensen Gap (Eq. 66)        | **Unstable**      |
| Signal (x)       | `ν(t) ≈ 1/b(t)²`                         | `exp(−‖x_j − x_k‖²/(2b(t)²))`     | **Stable**        |
| Velocity (v)     | `ν(t) = 1` (since `ad − bc = 1`)         | bounded posterior dispersion      | **Inherently stable** |

\*For `b(t) ∝ t^k`, the prefactor `ḃ/b = d/dt log b` diverges as `O(1/t)`. For
VP-SDE (`b ∝ √t`) it is `O(t^(−1.5))`. For FM linear (`b = t`) it is `O(1/t)`.

The paper's exact statement (§6, end):

> "Our results prove that velocity-based parameterizations satisfy this
> necessary [bounded-error] condition, whereas noise prediction structurally
> fails for autonomous models."

**Confidence: 3/5.** The substantive content is rigorous (Lemmas 1–6 in
§A–§B; Cases 1–3 in §F.2), but no single named Proposition states "noise gain
= O(1/b), velocity gain = O(1)." That conclusion is assembled from Table 2 +
Eqs. 65–70.

---

## (h) Jensen-gap term — Eq. (65)–(66)

Plugging `ε_t^*(u) ≈ (u − x)/b(t)` into the noise-prediction Drift Perturbation
Error (Eq. 65) and factoring out the geometric direction `‖u − x‖`:

$$ \boxed{\;
   \Delta\mathbf{v}_{\mathrm{noise}}
   \;\approx\;
   \|\mathbf{u} - \mathbf{x}\|\cdot
   \left|\frac{\dot b(t)}{b(t)}\right|\cdot
   \underbrace{\Bigl|\,b(t)\,\mathbb{E}_{\tau\mid\mathbf{u}}\!\left[\tfrac{1}{b(\tau)}\right] - 1\,\Bigr|}_{\text{Jensen Gap}}\;} \tag{66} $$

The gap is the multiplicative excess of the **harmonic mean** of `b(τ)` under
`p(τ|u)` over the true `b(t)`. By strict convexity of `1/x`, the gap is
`> 0` and converges to a non-zero constant whenever the posterior `p(τ|u)`
fails to collapse to `δ(τ−t)`. It is *amplified* by the `O(1/b(t))` prefactor,
giving `lim_{t→0} Δv_noise = ∞`.

For signal prediction (Eq. 67–69) the analogous error decays *exponentially*
with `1/b(t)²` and overwhelms the polynomial gain divergence. For velocity
(Eq. 70) the gain is constant and the error is just the bounded posterior
dispersion.

**Confidence: 5/5** (formula is verbatim Eq. 66).

---

## (i) Noising schedule actually used

The theory is stated for the **general affine family** (Eq. 2) and is
schedule-agnostic. Specific instantiations the paper *names*:

- **DDPM:** VP-SDE with `a² + b² = 1`, `a = √ᾱ_t`. The paper cites Ho et al. and
  Nichol & Dhariwal but does **not** transcribe a closed-form `β(t)`.
- **EDM:** `a = 1`, `b = σ_t` (standard EDM σ-schedule).
- **Flow Matching:** linear, `a(t) = 1 − t`, `b(t) = t`, `t ∈ [0, 1]`.
- **Equilibrium Matching:** same `(1−t, t)`.

The §7 experiments (CIFAR-10, SVHN, Fashion MNIST) report training hyper­
parameters (10 000 steps, EMA = 0.999, batch 128, ResNet-UNet) but do not
print the exact β-schedule used for DDPM Blind / DDPM Conditional. The 2D toy
in §7.2 is concentric circles in `R^D` with `D ∈ {2, 8, 32, 128}`.

⚠ **Flag.** No equation in the body or appendix pins a specific β(t) (linear,
cosine, or otherwise) for the empirical results. For our reproduction we will
adopt **FM-linear** (`a = 1 − t`, `b = t`) throughout — it is the schedule the
paper uses to derive every closed-form near-manifold expression in §E (Eq. 59,
60, 67–70).

**Confidence: 2/5** for "what β(t) the experiments used"; **5/5** for "what
schedule the closed-form theory is written against (FM-linear)".

---

## Notebook scope

**In scope (analytic reproduction).**

- Discrete-data closed forms from §A.3 and §E:
  - GMM marginal `p(u|t)` (Eq. 37) and posterior `p(t|u)` (Eq. 36).
  - Optimal denoiser `D_t^*(u) = Σ_k w_k(u,t) x_k` (Eq. 34, 35).
  - Marginal energy `E_marg(u) = −log p(u)` and its singular gradient (Eq. 11).
  - Effective gain `λ(t)`, `λ̄(u)`, transport correction, linear drift
    (Eq. 14, 15).
  - Riemannian preconditioning visualization: side-by-side plots of
    `‖∇E_marg(u)‖` vs. `‖f*(u)‖ = ‖λ̄(u)·∇E_marg(u) + …‖` showing the
    singularity cancellation.
  - Drift Perturbation Error (Eq. 22) traced numerically along sampler
    trajectories for the three parameterizations under FM-linear schedule;
    verifies `Δv_noise → ∞`, `Δv_signal → 0`, `Δv_velocity` bounded.
  - Jensen Gap (Eq. 66) computed exactly on a small `N`-point dataset.
- 1D linear-Gaussian data (`p_data = N(0, I)`) — admits closed-form
  marginal energy, gradient, and λ̄.
- 2D Gaussian random fields on a small grid — finite-mixture treatment.

**Out of scope.**

- CIFAR-10 / SVHN / Fashion-MNIST training (§7) — full neural-net experiments
  with U-Nets are too heavy for molab; we will not reproduce Figs. 2–4 or
  Table 3.
- Salimans–Ho velocity (`v = α_t ε − σ_t x`) — paper uses `v = u̇`.

**Extension (beyond paper).**

- **Fourier-mode shrinkage picture for stationary GRFs.** For a Gaussian
  random field with covariance diagonalized in the Fourier basis,
  `p(u|t) ∝ N(0, a²Σ + b²I)`, so `D_t^*(u) = a·(a²Σ + b²I)⁻¹·Σ·u` is a
  per-mode Wiener filter. Each Fourier mode independently realizes the
  geometry: the conformal factor `λ̄(u)` becomes a per-mode shrinkage gain
  whose `1/b²` divergence is offset by mode-wise eigenvalue suppression.
  The paper does not draw this picture. The notebook will plot the
  per-mode gain spectrum vs. `b(t)` and visualize that velocity prediction
  keeps every mode bounded while noise prediction blows the high-frequency
  modes up.

---

## Confidence ledger

| Bullet                                              | Confidence (1–5) |
| --------------------------------------------------- | ---------------- |
| (a) affine noising family `u = ax + bε`             | 5                |
| (b) marginal energy `E_marg = −log ∫ p(u\|t)p(t)dt` | 5                |
| (c) raw gradient with `1/b²` singularity            | 5                |
| (d) decomposition + conformal/Riemannian factor     | 5                |
| (e) sampler ODE with explicit μ(t), ν(t)            | 5                |
| (f) ε / x / v parameterizations                     | 4 (velocity convention flagged) |
| (g) "stability theorem"                             | 3 (no formal Proposition/Theorem in paper) |
| (h) Jensen-gap formula (Eq. 66)                     | 5                |
| (i) noising schedule used                           | 2 (theory) / 5 (FM-linear closed forms) |

## Items I could not parse cleanly

1. **Exact β-schedule used in the §7 image experiments.** Section 7 reports
   only training hyperparameters; the underlying DDPM `ᾱ_t` schedule
   (linear-β? cosine?) is not stated.
2. **Salimans–Ho vs. paper velocity.** The user's prompt asks for
   `v = α_t ε − σ_t x`. The paper uses `v = u̇`. These are different except
   in degenerate limits. We use the paper's definition.
3. **Stability "theorem".** There is no numbered Proposition stating
   `ν(t) = O(1/b(t))` for noise vs. `ν(t) = 1` for velocity. The result is the
   conjunction of Table 2 + the Case 1/2/3 derivations in §F.2 (Eq. 65–70).
   Proposition 1 in §C concerns posterior concentration in high `D`, a
   different (though complementary) result.
4. **Sign / direction of `λ(t)`.** Eq. 15 gives `λ(t) = (b/a)(da − cb)`. For
   FM (`a = 1−t, b = t, c = −1, d = 1`), this is `λ(t) = (t/(1−t))(1−t + t) = t/(1−t)`,
   not `≈ t` as stated qualitatively in Eq. 59. Under near-manifold
   linearization `a → 1`, `λ(t) ≈ t`, which is the form Eq. 59 uses.
   This linearization is implicit and worth re-deriving explicitly in the
   notebook.
