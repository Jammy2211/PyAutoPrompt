Extend `autolens_workspace_test/scripts/jax_likelihood_functions/point_source/` from the single
`point.py` to cover the two fit-positions modes profiled in
`autolens_workspace_developer/jax_profiling/point_source/` — image-plane and source-plane
chi-squared — while reusing the **existing** simulated dataset in
`autolens_workspace_test/scripts/point_source/simulators/point_source.py`.

__Baseline already shipped__

- `autolens_workspace_test/scripts/jax_likelihood_functions/point_source/point.py` — current
  single-variant jax_likelihood coverage.
- `autolens_workspace_test/scripts/point_source/simulators/point_source.py` — fixed seeded
  `PointDataset` simulator (`noise_seed=1`) that is the canonical input for point-source tests.
- `autolens_workspace_developer/jax_profiling/point_source/{image_plane,source_plane}.py` —
  the two fit modes, with full three-tier numerical assertions and a documented source-plane
  blocker (`Grid2DIrregular.grid_2d_via_deflection_grid_from` `xp`-propagation bug).

__Reference scripts__

- Three-step pattern template: `@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge.py`.
- Fit-mode templates: `@autolens_workspace_developer/jax_profiling/point_source/image_plane.py`
  and `source_plane.py` — use their `AnalysisPoint(..., fit_positions_cls=..., use_jax=True)`
  wiring, `PointSolver.for_grid(..., xp=jnp)` and the ray-trace prefix pattern.
- Dataset source: `@autolens_workspace_test/scripts/point_source/simulators/point_source.py`
  — **do not clone** the simulator into `jax_likelihood_functions/point_source/`. Load the
  already-simulated dataset via the `should_simulate` bootstrap (same pattern as every other
  jax_likelihood_functions script), invoking the simulator in `scripts/point_source/simulators/`
  via `subprocess.run`.

__Scripts to add__ in `autolens_workspace_test/scripts/jax_likelihood_functions/point_source/`

1. `image_plane.py`

   - `AnalysisPoint(dataset=dataset, solver=solver, fit_positions_cls=al.FitPositionsImagePairAll, use_jax=True)`.
   - `PointSolver.for_grid(grid=..., pixel_scale_precision=0.001, magnification_threshold=0.1, xp=jnp)`.
   - Three-step pattern: vmap assertion → Path A `jax.jit(analysis.fit_from)` round-trip →
     `PASS: jit(fit_from) round-trip matches NumPy scalar.`
   - Image-plane fitting is known to JIT end-to-end (the profiling script confirms), so this
     variant should pass cleanly once `FitPointDataset` pytree registration is in place.

2. `source_plane.py`

   - `AnalysisPoint(dataset=dataset, solver=solver, fit_positions_cls=al.FitPositionsSource, use_jax=True)`.
   - Three-step pattern, but wrap Path A (`jax.jit(analysis.fit_from)`) in a
     `try/except jax.errors.TracerArrayConversionError:` that prints a clear BLOCKER line and
     sets a `full_pipeline_jits=False` sentinel, mirroring
     `jax_profiling/point_source/source_plane.py`.
   - Still assert the **eager** NumPy path against a hardcoded log-likelihood regression
     constant so the NumPy side of the pipeline is guarded.
   - If / when the `Grid2DIrregular.grid_2d_via_deflection_grid_from` `xp`-propagation fix
     lands, the script will JIT cleanly without modification.

__Dataset reuse__

Both scripts load from `dataset/point_source/simple/point_dataset_positions_only.json` (or
whatever filename the autolens_workspace_test simulator writes — confirm by reading the existing
simulator). The `should_simulate` bootstrap runs
`scripts/point_source/simulators/point_source.py` if the dataset is missing. **Do not write a
new simulator under `jax_likelihood_functions/point_source/`** — reuse the existing one.

__Pytree registration__

Both scripts call `enable_pytrees()` + `register_model(model)` before the Path A JIT. `FitPointDataset`
registration is covered by the issued `fit_point_pytree.md` prompt — reference it as a dependency.
If that library-side work has not landed yet, Path A will fail with a pytree-registration error;
document and park rather than hand-register in the test scripts.

__Deliverables__

1. Two new scripts in `autolens_workspace_test/scripts/jax_likelihood_functions/point_source/`:
   `image_plane.py`, `source_plane.py`.
2. `image_plane.py` prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
3. `source_plane.py` either prints the PASS line (if the upstream `xp` bug is fixed) or a clear
   BLOCKER line identifying the eager vs JIT divergence — same pattern as the profiling script.
4. Update `autolens_workspace_test/scripts/CLAUDE.md` to list the two new scripts in the
   `jax_likelihood_functions/` table.

__Scope boundary__

- Do not introduce a new simulator — reuse `scripts/point_source/simulators/point_source.py`.
- Do not work around the source-plane `xp`-propagation blocker in the test script. Document it
  and leave the fix for the library task (`fit_point_pytree.md`).
- Do not change `autolens_workspace_developer/jax_profiling/point_source/` — that is the profiling
  reference, not the test surface.
