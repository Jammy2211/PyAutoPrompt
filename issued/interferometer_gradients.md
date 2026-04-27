# JAX Gradient Testing: Interferometer

## Context

`autolens_workspace_developer/jax_profiling/imaging/mge_gradients.py`
and `pixelization_gradients.py` are step-by-step
`jax.value_and_grad` probes that walk the imaging likelihood
pipeline (ray-trace → blurred image → profile-subtract → mapping
matrix → D → F → H → NNLS → mapped reconstruction → full
`Fitness.call`) and report PASS / FAIL / ERROR at each stage with
a gradient-norm diagnostic.

These scripts have been the primary tool for isolating JAX
gradient bugs — they caught the NNLS NaN that PR #279 fixed, the
rectangular-interpolator gradient explosion that PR #281 fixed,
and the NNLS target_kappa NaN that PR #282 fixed.

**There is no equivalent probe for the interferometer pipeline.**
The interferometer likelihood uses a different kernel path
(`transformer_nufft` / sparse Fourier operator, visibilities-space
NNLS) whose gradient health is not yet characterised, and any future
NaN regression there would go undiagnosed.

## Pytree infrastructure (already shipped — assume present)

These probes exercise the full pytree approach:
`jax.value_and_grad(AnalysisInterferometer.log_likelihood)` with all
priors flowing as pytree leaves. Three library-level pieces that make
this work have already landed on `main`:

- **PyAutoFit#1222** — `TuplePrior` registered as a JAX pytree. On a
  typical Isothermal+Shear+MGE model this lifts live JAX-leaf count
  from 3 to 167, so the probe actually differentiates through the
  whole model instead of a handful of free floats.
- **PyAutoArray#279** — Jacobi preconditioning of the NNLS curvature
  matrix.
- **PyAutoArray#282** — `nnls_target_kappa=1.0e-2` config default
  (jaxnnls's own `1e-3` default produces NaN in the relaxed-KKT
  backward pass on real MGE pipelines even *with* Jacobi
  preconditioning).

For reference, the imaging `mge_gradients.py` and
`pixelization_gradients.py` now report **9/9 PASS** on `main` with this
stack. The interferometer probes should aim for the same outcome.

## Task

Create
`autolens_workspace_developer/jax_profiling/interferometer/pixelization_gradients.py`
and `mge_gradients.py`, modelled directly on their imaging
counterparts.

Minimum stages to probe:

1. Ray-trace interferometer grid (pixelization grid).
2. Blurred (`transformer`-applied) lens-light image.
3. Profile-subtracted visibilities.
4. Transformed mapping matrix (mapper → FFT/NUFFT).
5. Data vector D in visibilities space.
6. Curvature matrix F.
7. Regularization matrix H.
8. NNLS reconstruction.
9. Mapped reconstructed visibilities.
10. Full pipeline via `Fitness.call` and
    `jax.value_and_grad(AnalysisInterferometer.log_likelihood)`.

Each stage wraps a closure in `jax.value_and_grad`, prints gradient
norm / finite-fraction / non-zero count, and classifies the stage
as PASS / FAIL / ERROR. Final summary table identical to the
imaging version. Target outcome: 9/9 or 10/10 PASS.

Include an `_diagnose_kappa`-style loop at any stage that exercises
`jaxnnls.solve_nnls_primal` directly, so that if the NNLS NaN
regresses the probe can tell us *which* `target_kappa` value is safe
for the interferometer curvature matrix (imaging empirically needs
≥1e-2; interferometer may need a different value).

## Why this matters

The interferometer path has more moving parts that can silently
kill gradients (NUFFT gridding tables, complex-valued intermediate
arrays, sparse visibility operators). Without a probe we will only
learn about regressions when a downstream fit produces all-zero or
NaN gradients and fails mysteriously. A stage-by-stage probe
isolates the break point immediately.

## Dependencies

- `interferometer_jax_profiling.md` covers the forward-only
  profiling scripts. This gradients probe assumes those exist (so
  the dataset auto-simulate + `Fitness` construction pattern is
  already proven).
- If the interferometer profiling scripts are not yet in place,
  this task is blocked on that one.
