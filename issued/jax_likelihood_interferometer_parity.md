Extend `autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/` to match the
coverage already in `.../jax_likelihood_functions/imaging/`, plus add one variant that exercises
the JAX sparse-operator path introduced for pixelized interferometer fits.

__Baseline already shipped__

The imaging folder is the reference surface:

- `imaging/lp.py` — parametric light profile source
- `imaging/mge.py` — MGE source (already mirrored in `interferometer/mge.py`)
- `imaging/mge_group.py` — MGE + extra galaxies (already mirrored in `interferometer/mge_group.py`)
- `imaging/rectangular.py` — rectangular pixelization (already mirrored in `interferometer/rectangular.py`)
- `imaging/delaunay.py` — Delaunay pixelization with Hilbert image-mesh and edge-zeroed points
- `imaging/delaunay_mge.py` — Delaunay source + MGE lens
- `imaging/rectangular_mge.py` — Rectangular source + MGE lens
- `imaging/rectangular_dspl.py` — Rectangular source on a double source plane
- `imaging/simulator.py` / `simulator_dspl.py` — fixed seeded simulators

Each imaging script follows the same three-step pattern:

1. `fitness._vmap(parameters)` + `np.testing.assert_allclose(..., rtol=1e-4)` against a hardcoded
   expected log-likelihood.
2. `enable_pytrees()` + `register_model(model)`; build `analysis_np` with `use_jax=False`, compute
   `fit_np.log_likelihood`.
3. Build `analysis_jit` with `use_jax=True`; call `fit = jax.jit(analysis_jit.fit_from)(instance)`;
   assert `isinstance(fit.log_likelihood, jnp.ndarray)` and
   `assert_allclose(float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4)`;
   finally print `PASS: jit(fit_from) round-trip matches NumPy scalar.`

__Reference scripts__

- `@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/{lp,delaunay,delaunay_mge,rectangular_mge,rectangular_dspl,simulator_dspl}.py`
  — copy the three-step pattern and model wiring verbatim where possible.
- `@autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/{rectangular,mge,mge_group,simulator}.py`
  — existing interferometer-specific wiring: `al.Interferometer.from_fits(...)` with
  `real_space_mask`, `TransformerDFT`, the `dataset/interferometer/uv_wavelengths/sma.fits`
  baseline, and the `should_simulate(dataset_path)` + `subprocess.run` auto-sim bootstrap.

__Scripts to add__ in `autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/`

1. `lp.py` — parametric Sersic source with `AnalysisInterferometer`.
2. `delaunay.py` — Delaunay pixelization source using `al.image_mesh.Hilbert` + edge-zeroed points
   (copy the image-mesh / `total_mapper_pixels` / `AdaptImages` plumbing from `imaging/delaunay.py`).
3. `delaunay_mge.py` — Delaunay source + MGE lens.
4. `rectangular_mge.py` — Rectangular source + MGE lens.
5. `rectangular_dspl.py` — Double source plane with rectangular pixelization. Reads from the new
   double-source-plane interferometer dataset produced by `simulator_dspl.py` (below).
6. `simulator_dspl.py` — Double source plane interferometer simulator. Mirror
   `imaging/simulator_dspl.py` but use `SimulatorInterferometer` with the same
   `dataset/interferometer/uv_wavelengths/sma.fits` baseline as the existing `simulator.py`,
   `TransformerDFT`, `noise_seed=1`. Writes to
   `dataset/interferometer/dspl/` with the same file layout as
   `dataset/interferometer/simple/` (`data.fits`, `noise_map.fits`, `uv_wavelengths.fits`,
   `positions.json`).
7. `rectangular_sparse.py` — Rectangular pixelization variant that calls

   ```python
   dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=True)
   ```

   after constructing `dataset` from fits (and before building `AnalysisInterferometer`). This
   exercises the JAX sparse NUFFT path used in the `autolens_workspace`
   `notebooks/interferometer/features/pixelization/fit.ipynb` and `delaunay.ipynb` notebooks.

__Sparse-operator caveat__

`apply_sparse_operator(use_jax=True, show_progress=True)` precomputes a sparse NUFFT operator and
attaches it to the dataset. It **must** be called outside the JIT boundary — the sparse operator
is aux state, not a traced pytree leaf. Do not attempt to construct or rebuild it inside
`jax.jit(analysis.fit_from)`. If the three-step pattern fails for this variant, the fallback is to
document the failure in the script's docstring rather than force a workaround — the simpler fix
lives in the library.

__Auto-simulation boilerplate__

Each new script must include the same `if al.util.dataset.should_simulate(dataset_path):` bootstrap
as the existing interferometer scripts, invoking the corresponding `simulator.py` /
`simulator_dspl.py` via `subprocess.run([sys.executable, ...], check=True)`.

__Expected log-likelihood values__

Each script's vmap assertion has a hardcoded expected log-likelihood. On first run the author fills
in the actual value (the pattern used by `interferometer/rectangular.py` and
`interferometer/mge.py`), then the assertion becomes a regression guard.

__Deliverables__

1. Seven new scripts in `autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/`:
   `lp.py`, `delaunay.py`, `delaunay_mge.py`, `rectangular_mge.py`, `rectangular_dspl.py`,
   `simulator_dspl.py`, `rectangular_sparse.py`.
2. One new dataset folder: `dataset/interferometer/dspl/` produced by `simulator_dspl.py`.
3. Each script (except simulators) ends by printing `PASS: jit(fit_from) round-trip matches NumPy scalar.`
4. Update `autolens_workspace_test/scripts/CLAUDE.md` to list the new interferometer scripts in
   the `jax_likelihood_functions/` table.
5. Notes on any sparse-operator × JIT interactions — this variant is the canary for that code path.

__Scope boundary__

- Do not register the sparse NUFFT operator's internal state as a traced pytree — it is aux.
- Do not add a new `autolens_workspace_test/dataset/interferometer/uv_wavelengths/` baseline.
  Reuse `sma.fits` (190 visibilities, fast and deterministic).
- Do not introduce library-side API changes. If a blocker surfaces (e.g. pixelization mapper
  under the sparse operator inside JIT), document and park — file a library prompt separately.
