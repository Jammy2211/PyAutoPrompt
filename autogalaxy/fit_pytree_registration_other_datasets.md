PyAutoGalaxy's `FitImaging` was given JAX pytree registration in PR #364
(2026-04-22, see `complete.md` entry for the imaging port). The remaining
fit types — `FitInterferometer`, `FitEllipse`, `FitQuantity` — have **no
pytree registration** today, so `use_jax_for_visualization=True` on
`AnalysisInterferometer`, `AnalysisEllipse`, and `AnalysisQuantity` either
no-ops or crashes when `fit_for_visualization` tries to lift a fit across
the JIT boundary.

This task ports the imaging registration pattern to the other three dataset
types so that all PyAutoGalaxy analyses can opt into the jit-cached
visualization path.

__Why this matters__

This is **Phase 0c** of `z_features/jax_visualization.md`. Without it,
Phase 1C (`autogalaxy_workspace_test/jax_viz_dataset_coverage.md`) cannot
add JAX viz coverage for interferometer / ellipse / quantity, and Phase 2
(default `use_jax_for_visualization` on whenever `use_jax=True`) would
silently break those dataset types.

__What to register__

For each of the three fit classes, register the fit and every distinct
autoarray / autogalaxy type reachable from a populated instance. Mirror the
pattern shipped in PR #364 for `FitImaging`. The key files:

1. `@PyAutoGalaxy/autogalaxy/interferometer/fit_interferometer.py` —
   `FitInterferometer`. Reachable types include `Visibilities`,
   `VisibilitiesNoiseMap`, `Interferometer`, `Galaxies`, plus the
   `Transformer*` family and any `Inversion` variants used for pixelized
   sources.

2. `@PyAutoGalaxy/autogalaxy/ellipse/fit_ellipse.py` — `FitEllipse`.
   Reachable types include `Ellipse`, `Multipole`, `Array2D`, and the
   `MaskedDataset` analogue used inside the analysis.

3. `@PyAutoGalaxy/autogalaxy/quantity/fit_quantity.py` — `FitQuantity`.
   Reachable types include `DatasetQuantity` and the autogalaxy quantity
   container (`convergence_2d`, `deflections_yx_2d`, `potential_2d`).

For each type follow the autofit `register_instance_pytree` pattern (see
`@PyAutoFit/autofit/jax/pytrees.py` and the imaging analogue in PyAutoLens
`AnalysisImaging._register_fit_imaging_pytrees`):

- `children` = JAX-traced arrays / sub-pytrees (e.g. `visibilities.array`,
  `ellipse.major_axis`, `quantity.convergence_2d._array`).
- `aux` = static Python objects that mustn't change under tracing (masks,
  pixel scales, redshifts, `Inversion` solver state, transformer config).
- The boundary rules from PyAutoLens `CLAUDE.md` apply unchanged:
  `array._array` (or `.array`) is dynamic, `array.mask` and shape metadata
  are static.

__What to test__

For each fit type, add a registration test in
`@PyAutoGalaxy/test_autogalaxy/<dataset>/jax/test_<dataset>_pytree.py`
following the three-step pattern from
`autolens_workspace_test/scripts/hessian_jax.py`:

1. Build a minimal populated `Fit*` instance from synthetic inputs.
2. Round-trip through `jax.tree_util.tree_flatten` + `tree_unflatten` and
   assert the reconstructed fit matches the original on the dynamic fields.
3. Wrap a toy function that returns the fit in `jax.jit` and confirm it
   compiles, runs, and returns a fit whose dynamic leaves are `jax.Array`.

__Verification__

- New unit tests pass: `pytest test_autogalaxy/interferometer/jax`,
  `pytest test_autogalaxy/ellipse/jax`, `pytest test_autogalaxy/quantity/jax`.
- Existing PyAutoGalaxy unit tests still pass: `pytest test_autogalaxy`.
- Run `/smoke_test` on `autogalaxy_workspace`. The non-JAX paths must be
  unchanged — pytree registration only affects JIT, never the eager NumPy
  path.

__Out of scope__

- **No workspace_test JAX visualization scripts in this task.** Those are
  written in the follow-up Phase 1C prompt
  (`autogalaxy_workspace_test/jax_viz_dataset_coverage.md`). That prompt
  is blocked on this one landing.
- **No PyAutoLens equivalent (`AnalysisInterferometer`, no Path A wrap of
  `FitInterferometer`).** PyAutoLens interferometer pytree registration is
  out of scope here; once this lands and Path A on imaging is final, a
  separate PyAutoLens prompt can mirror it.
- **No production workspace adoption.** Tutorials don't get
  `use_jax_for_visualization=True` from this task.
- **No change to the autogalaxy imaging visualizer dispatch.** That's a
  separate pending prompt: `autogalaxy/visualizer_fit_for_visualization_dispatch.md`.

__Reference__

- `@PyAutoFit/autofit/jax/pytrees.py` — autofit pytree machinery
- PR #364 (PyAutoGalaxy imaging pytree port, 2026-04-22) — pattern to mirror
- `@PyAutoLens/autolens/imaging/model/analysis.py` — `AnalysisImaging._register_fit_imaging_pytrees` reference implementation
- `PyAutoPrompt/issued/fit_imaging_pytree.md` — Path A feasibility study (in-flight)
- `PyAutoPrompt/z_features/jax_visualization.md` — sequenced roadmap this task is Phase 0c of
