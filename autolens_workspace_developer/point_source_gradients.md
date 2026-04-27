# JAX Gradient Testing: Point Source

## Context

`autolens_workspace_developer/jax_profiling/imaging/mge_gradients.py`
and `pixelization_gradients.py` are step-by-step
`jax.value_and_grad` probes that walk the imaging likelihood
pipeline and report PASS / FAIL / ERROR + gradient-norm
diagnostics at each stage. They have been the primary tool for
isolating JAX gradient regressions (PR #279, #281, #282).

There is no equivalent probe for the point-source likelihoods.
Point-source fitting produces gradient paths that differ
fundamentally from imaging:

- **Source-plane likelihood**: straight chi-squared of ray-traced
  positions vs source position — should be fully differentiable
  and cheap.
- **Image-plane likelihood**: requires a non-linear solver
  (`al.PointSolver` + descendants) to find image-plane arrivals
  from a source-plane guess, so the gradient path passes through
  that solver. Whether that solver traces under JAX at all is an
  open question.

I am confident the image plane likleihood which maps triangles to
and from the source code, cannot prtovide a gradient and dont want
us doing anything too complex, so rreally just decide if there is
any gradient but its follow up work to worry about what to do about it.

## Pytree infrastructure (already shipped — assume present)

These probes exercise
`jax.value_and_grad(AnalysisPoint.log_likelihood)` with priors
flowing as pytree leaves. The library-level groundwork is on `main`:

- **PyAutoFit#1222** — `TuplePrior` registered as a JAX pytree. For
  extended-source imaging models this raised the live JAX-leaf count
  from 3 to 167; for a typical point-source model (lens mass +
  single source position) the uplift is smaller but it is what lets
  position priors flow as differentiable leaves rather than frozen
  constants.
- **PyAutoArray#279 / #282** — NNLS preconditioning and
  `nnls_target_kappa=1.0e-2` default. Not on the point-source
  critical path but present on `main` as part of the same pytree
  readiness work.

For reference, the imaging `mge_gradients.py` and
`pixelization_gradients.py` now report **9/9 PASS** on `main` with
this stack. Point-source probes should expose whether the positions-
residual path matches that standard.

## Task

Create two probe scripts in
`autolens_workspace_developer/jax_profiling/point_source/`:

1. `source_plane_gradients.py`
2. `image_plane_gradients.py`

Each follows the `pixelization_gradients.py` structure:
per-stage `jax.value_and_grad` closures, PASS / FAIL / ERROR
classification, gradient-norm / finite-fraction / nonzero-count
reporting, final summary table.

Suggested stages for `source_plane_gradients.py`:

1. Ray-trace image-plane positions to source plane.
2. Source-plane residual (traced positions − source position).
3. Positions chi-squared.
4. Full pipeline via `Fitness.call` and
   `jax.value_and_grad(AnalysisPoint.log_likelihood)`.

Suggested stages for `image_plane_gradients.py`:

1. Source-plane guess → solver-produced image-plane arrivals.
2. Image-plane residual (solved arrivals − observed positions).
3. Positions chi-squared.
4. Full pipeline via `Fitness.call`.

If the solver does not trace under JAX, log a clean blocker at
stage 1 and record where gradient flow breaks — that is itself
a valuable finding.

## Why this matters

Point-source fitting is expected to gain most from gradient-based
samplers (HMC / NUTS) because the parameter space is small but
highly correlated. If gradients silently zero-out or NaN-poison
the likelihood, a user running NUTS will see indecipherable
sampler failures rather than a clear "this path is not
differentiable" signal. The probe gives us that signal.

## Dependencies

- `point_source_jax_profiling.md` covers the forward-only JIT
  profiling scripts and the dataset auto-simulate pattern. This
  gradient probe assumes those are in place.
