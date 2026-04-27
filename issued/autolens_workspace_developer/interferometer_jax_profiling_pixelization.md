# JAX JIT Profiling: Interferometer Pixelization (Phase 2)

## Context

Phase 1 shipped in `@autolens_workspace_developer/jax_profiling/interferometer/`:
- `simulators/interferometer.py` — procedural SMA/ALMA uv-coverage + lens/source
- `mge.py` — full-pipeline JIT for the MGE source

Phase 2 is the pixelization equivalent, mirroring
`@autolens_workspace_developer/jax_profiling/imaging/pixelization.py`.

Delaunay (Phase 3) is intentionally deferred.

## Task

Create `@autolens_workspace_developer/jax_profiling/interferometer/pixelization.py` for the
`RectangularAdaptDensity` source model. Structure it the same way as
`mge.py` from Phase 1:

1. **PART A — setup**
   - Load the SMA interferometer dataset (same path Phase 1 writes).
   - Build the pytree-registered model:
     lens = `af.Model(al.mp.Isothermal)` + `af.Model(al.mp.ExternalShear)`,
     source = `af.Model(al.Galaxy)` with a `RectangularAdaptDensity`
     pixelization (follow `imaging/pixelization.py` for the mesh/regularization wiring).
   - Eager baseline: `FitInterferometer(dataset, tracer)`; print
     `figure_of_merit` and `log_likelihood`.

2. **PART B — full-pipeline JIT**
   - `analysis = al.AnalysisInterferometer(dataset, use_jax=True)`
   - Wrap `analysis.log_likelihood_function(instance=params_tree)` in
     `jax.jit`, measure lower / compile / first-call / steady-state.
   - Assert eager-vs-JIT agreement at `rtol=1e-4`.

3. **PART C — vmap batched**
   - Batch size 3 (same as Phase 1), assert vs single-JIT at same `rtol`.

4. **PART D — static memory analysis** (same helper as Phase 1).

5. **Results** — write JSON + PNG into
   `jax_profiling/interferometer/results/pixelization_likelihood_summary_<instrument>_<version>.{json,png}`
   with the same schema as the imaging results.

## Infrastructure already in place

- PyAutoFit#1222 — TuplePrior pytree registration
- PyAutoArray#279 — Jacobi preconditioning on the NNLS curvature
- PyAutoArray#282 — `nnls_target_kappa=1.0e-2` default

No library changes should be needed. Raise an issue if any
interferometer-specific step fails to thread `xp=jnp`.

## Expected output

`python pixelization.py` runs end-to-end, matches eager at `rtol=1e-4`,
emits JSON+PNG, reports vmap speedup.

## Explicitly out of scope

- `delaunay.py` — tracked separately as Phase 3
- Changes to the imaging profiling suite
- Any library-side changes
