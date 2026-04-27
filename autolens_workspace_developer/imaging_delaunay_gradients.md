# JAX Gradient Testing: Imaging Delaunay

## Context

`autolens_workspace_developer/jax_profiling/imaging/mge_gradients.py`
and `pixelization_gradients.py` are step-by-step
`jax.value_and_grad` probes that walk the imaging likelihood
pipeline (ray-trace → blurred image → profile-subtract → mapping
matrix → D → F → H → NNLS → mapped reconstructed image → full
`Fitness.call`) and classify each stage as PASS / FAIL / ERROR with
a gradient-norm diagnostic.

These probes now report **9/9 PASS** on `main` for MGE and
rectangular pixelization, after:

- **PyAutoFit#1222** lifted live JAX-leaf count from 3 to 167 by
  registering `TuplePrior` as a pytree,
- **PyAutoArray#279** added Jacobi preconditioning of the NNLS
  curvature matrix, and
- **PyAutoArray#282** raised `nnls_target_kappa` from jaxnnls's
  NaN-prone `1e-3` default to `1.0e-2`.

**There is no equivalent probe for the Delaunay source model.**
The forward-only `imaging/delaunay.py` script exists, but we have
no per-stage gradient diagnostic, so any NaN / zero-gradient
regression on the Delaunay path would go undiagnosed.

## Task

Create
`autolens_workspace_developer/jax_profiling/imaging/delaunay_gradients.py`,
modelled directly on `pixelization_gradients.py`.

Suggested stages:

1. Ray-trace image-plane grid to source plane.
2. Blurred lens-light image.
3. Profile-subtracted data.
4. Delaunay mapping matrix (mapper → interpolation weights on the
   source-plane Delaunay triangulation).
5. Blurred mapping matrix (PSF-convolved).
6. Data vector D.
7. Curvature matrix F.
8. Regularization matrix H.
9. NNLS reconstruction.
10. Mapped reconstructed image.
11. Full pipeline via `Fitness.call` and
    `jax.value_and_grad(AnalysisImaging.log_likelihood)`.

Each stage wraps a closure in `jax.value_and_grad`, prints gradient
norm / finite-fraction / non-zero-entry count, and classifies the
stage PASS / FAIL / ERROR. Final summary table identical to the
MGE / pixelization versions. Target outcome: **11/11 PASS**.

Include an `_diagnose_kappa`-style loop around the NNLS stage so
that if kappa needs to differ from the `1e-2` used for MGE, the
probe can tell us immediately.

## Why this matters

The Delaunay mapper path has two specific ways it can silently
kill gradients that MGE / rectangular do not share:

- **Triangulation boundary**: `scipy.spatial.Delaunay` is built
  outside the JIT boundary, so the triangulation itself is a frozen
  constant under tracing. If gradients need to flow *through* the
  triangulation (e.g. via source-plane pixel centres that depend on
  the lens model), that path is silently zeroed.
- **Interpolation weights**: the barycentric weights computed on the
  Delaunay cells are a piecewise-linear function of the source-plane
  position and have undefined gradients at cell boundaries. These
  typically manifest as spiky / zero / NaN gradients on a handful of
  leaves rather than catastrophic pipeline failure — exactly the
  kind of thing a stage-by-stage probe is designed to surface.

## Dependencies

- `imaging_delaunay_jax_profiling.md` covers the forward-only
  `delaunay.py` update and the pytree-readiness alignment.
  This gradient probe assumes that update has landed — if
  `delaunay.py` is still on the pre-pytree approach, fix that first
  or the probe will give misleading PASS results (frozen priors
  cannot FAIL because they never enter the gradient path).
