# JAX JIT Profiling: Point Source (Source Plane, Image Plane)

## Context

`autolens_workspace_developer/jax_profiling/imaging/` has mature JAX
JIT profiling scripts for extended source models (MGE, rectangular
pixelization, Delaunay). Point-source model-fitting — where the
source is a single position plus flux, and the likelihood comes from
a positions-residual chi-squared rather than a pixel-residual —
has no equivalent coverage.

There are two distinct point-source likelihoods to profile:

- **Source-plane**: chi-squared of the ray-traced image-plane
  positions compared to the single source-plane position. Cheaper,
  gradient-friendly.
- **Image-plane**: chi-squared of the solved multiple-image
  positions against the observed positions. More expensive — needs
  the non-linear solver to find image-plane arrivals from the
  source-plane guess. The JIT-ability of the solver is the
  interesting question.

## Pytree infrastructure (already shipped — build on top of it)

These scripts target the full pytree approach:
`jax.jit(AnalysisPoint.log_likelihood)` with all priors flowing as
pytree leaves. The library-level groundwork is already on `main`:

- **PyAutoFit#1222** — `TuplePrior` registered as a JAX pytree. For a
  point-source model this matters less than for extended-source
  models (fewer nested `(y, x)` centre priors), but it is still what
  lets the point-source centre/source position priors flow as
  differentiable leaves rather than being frozen constants.
- **PyAutoArray#279 / #282** — NNLS Jacobi preconditioning and
  `nnls_target_kappa=1.0e-2` default. These do not touch the point-
  source path directly but are part of the same pytree-readiness
  series and are present on `main`.

You should not need to modify any library code — if the point-source
path blocks on a pytree/JAX-tracing issue, flag it as a library-level
blocker rather than hacking around it in the script.

## Task

Create two profiling scripts in
`autolens_workspace_developer/jax_profiling/point_source/`:

1. `source_plane.py`
2. `image_plane.py`

Each should:

- Auto-simulate a point-source dataset if missing (via
  `subprocess.run` on a `simulators/point_source.py`, mirror the
  imaging auto-simulate pattern).
- Build the appropriate `al.FitPointDataset` + `al.AnalysisPoint`
  with the right solver type.
- Eager baseline: print `figure_of_merit` / `log_likelihood`.
- JAX path: `jax.jit` around `Fitness.call`, measure compile and
  steady-state timings.
- vmap path: batch N parameter vectors, measure per-likelihood cost.
- Numerical-agreement assertion between eager and JIT paths.
- Results JSON + PNG into `point_source/results/`.

Keep the overall structure identical to `imaging/mge.py` so a
reader can compare apples-to-apples.

## Open questions for the implementer

- For `image_plane.py`, the image-plane solver is probably not
  fully JAX-compatible today — confirm which code path is hit
  (`al.PointSolver` variants) and whether it traces under JIT.
  If it does not, log a clear blocker and profile whatever prefix
  of the pipeline *is* JIT-able.
- Point-source likelihoods are typically so cheap eagerly that the
  compile-time / steady-state ratio is unfavourable. Report this
  honestly — the goal is to understand the JIT profile, not to
  force a speedup.

## Related

- `autolens_workspace_test/scripts/jax_likelihood_functions/point_source/point.py`
  already exercises `fitness._vmap` for point-source models; use
  it as a reference for model construction + parameter vectors.
- `autolens_workspace_test/scripts/point_source/` holds the
  simulator + reference data.
