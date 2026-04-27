# JAX JIT Profiling: Interferometer (MGE, Pixelization, Delaunay)

## Context

`autolens_workspace_developer/jax_profiling/imaging/` contains mature JAX
JIT profiling scripts for the three main source models:

- `mge.py` — Multi-Gaussian Expansion source
- `pixelization.py` — RectangularAdaptDensity pixelization source
- `delaunay.py` — Delaunay pixelization source

Each script builds the log-likelihood function, applies `jax.jit`,
measures compile-time + per-eval runtime, and benchmarks against the
eager path. These scripts have driven many of the recent JAX
performance wins (e.g. the 80–96% runtime reductions in
`smoke-test-optimization`).

**There is no equivalent coverage for interferometer datasets.**
The `FitInterferometer` pipeline exercises a different kernel
(Fourier transform of the mapping matrix, visibilities-space
NNLS) whose JIT behaviour is not yet characterised.

## Pytree infrastructure (already shipped — build on top of it)

These scripts target the full pytree approach:
`jax.jit(AnalysisInterferometer.log_likelihood)` on a real model with
all priors flowing as pytree leaves. Three pieces of library
infrastructure that make this viable have already landed on `main`:

- **PyAutoFit#1222** — `TuplePrior` is now registered as a JAX pytree.
  This is the fix that raised the live JAX-leaf count on a typical
  Isothermal+Shear+MGE model from 3 (only free floats) to 167 (every
  prior in the model), so `jax.value_and_grad` actually flows through
  the whole model rather than freezing most of it.
- **PyAutoArray#279** — Jacobi preconditioning on the NNLS curvature
  matrix. Mitigates ill-conditioning before the relaxed-KKT backward
  pass runs.
- **PyAutoArray#282** — `nnls_target_kappa=1.0e-2` config default
  (was inheriting jaxnnls's `1e-3`), which was producing NaN in the
  NNLS backward pass on real MGE pipelines even *with* Jacobi
  preconditioning.

You should not need to modify any library code to get these scripts
running — they depend on the above being present, and they provide
the signal if any of it regresses.

## Task

Create three profiling scripts in
`autolens_workspace_developer/jax_profiling/interferometer/`:

1. `mge.py` — MGE source, Isothermal + ExternalShear lens.
2. `pixelization.py` — RectangularAdaptDensity source.
3. `delaunay.py` — Delaunay source.

Structure each script like its imaging counterpart:

- Dataset auto-simulation via `al.util.dataset.should_simulate` +
  `subprocess.run` on a matching `simulators/interferometer.py`.
  Instrument presets (ALMA / SMA / …) keyed off an `instrument`
  variable at the top, same pattern as `imaging/mge.py`.
- Eager baseline: build `FitInterferometer`, print
  `figure_of_merit` / `log_likelihood`.
- JAX path: wrap `Fitness.call` in `jax.jit`, measure first-call
  (compile) and steady-state runtimes over N repeats.
- vmap path: batch `batch_size` parameter vectors through
  `fitness._vmap`, measure per-likelihood cost.
- Assertion: numerical agreement between eager and JIT paths within
  a sensible `rtol`.
- Output: write a results JSON + PNG summary into
  `jax_profiling/interferometer/results/` mirroring the imaging
  results schema so they can be compared.

## Dependencies

- An `interferometer` dataset simulator must exist alongside the
  imaging one. Check `autolens_workspace_developer/jax_profiling/`
  for whether a simulator is already in place; if not, copy the
  pattern from `autolens_workspace_test/scripts/interferometer/simulator/`.
- `al.FitInterferometer` and `al.AnalysisInterferometer` are the
  entry points; the xp=jnp path goes through
  `autoarray.inversion.inversion_interferometer.mapper_operator`.

## Expected output

Three working scripts that run end-to-end via:

```bash
cd jax_profiling/interferometer
python mge.py
python pixelization.py
python delaunay.py
```

Each producing a JIT vs eager timing comparison, a vmap batch
throughput measurement, and a results artefact for tracking.

## Likely blockers to raise if encountered

- Interferometer path may not yet thread `xp=jnp` through every
  step (visibilities transform, dirty-image FFT, etc.) — if a step
  fails under JAX tracing, file a separate issue rather than
  hacking around it.
- `jax.jit` may balloon compile time for large `visibilities` —
  document and suggest a `batch_size` knob to split compilation.
- If the NUFFT transformer is not pytree-compatible, flag it as a
  library-level blocker rather than working around it in the script.
