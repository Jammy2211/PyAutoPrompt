PyAutoGalaxy's `FitImaging` was given JAX pytree registration in PR #364
(2026-04-22). `FitInterferometer` followed in PR #376 (see
`autogalaxy/interferometer/model/analysis.py:165-184` —
`_register_fit_interferometer_pytrees` reuses
`autogalaxy/analysis/jax_pytrees.py::register_galaxies_pytree` as a shared helper).

The remaining gap is `FitEllipse` and `FitQuantity` — neither has pytree
registration today, so `use_jax_for_visualization=True` on `AnalysisEllipse`
or `AnalysisQuantity` either no-ops or crashes when `fit_for_visualization`
tries to lift the fit across the JIT boundary.

This task ports the imaging / interferometer registration pattern to those
two remaining dataset types.

__Why this matters__

This is **Phase 0c** of `z_features/jax_visualization.md`. Without it,
Phase 1C (`autogalaxy_workspace_test/jax_viz_dataset_coverage.md`) cannot
add JAX viz coverage for ellipse / quantity, and Phase 2 (default
`use_jax_for_visualization` on whenever `use_jax=True`) would silently
break those dataset types.

__What to register__

For each fit class, register the fit and every distinct autoarray /
autogalaxy type reachable from a populated instance. Mirror the pattern
from `_register_fit_interferometer_pytrees` — particularly the use of
`autogalaxy/analysis/jax_pytrees.py::register_galaxies_pytree()` as the
shared galaxies hook.

1. `@PyAutoGalaxy/autogalaxy/ellipse/fit_ellipse.py` — `FitEllipse`.
   Reachable types include `Ellipse`, `Multipole`, `Array2D`, and the
   `MaskedDataset` analogue used inside the analysis. Register entry
   point in `@PyAutoGalaxy/autogalaxy/ellipse/model/analysis.py` as
   `_register_fit_ellipse_pytrees`.

2. `@PyAutoGalaxy/autogalaxy/quantity/fit_quantity.py` — `FitQuantity`.
   Reachable types include `DatasetQuantity` and the autogalaxy quantity
   container (`convergence_2d`, `deflections_yx_2d`, `potential_2d`).
   Register entry point in `@PyAutoGalaxy/autogalaxy/quantity/model/analysis.py`
   as `_register_fit_quantity_pytrees`.

For each type follow the autofit `register_instance_pytree` pattern (see
`@PyAutoFit/autofit/jax/pytrees.py` and the imaging / interferometer
analogues in the same repo):

- `children` = JAX-traced arrays / sub-pytrees (e.g. `ellipse.major_axis`,
  `quantity.convergence_2d._array`).
- `aux` = static Python objects that mustn't change under tracing (masks,
  pixel scales, redshifts, transformer config, `Inversion` solver state).
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

- New unit tests pass: `pytest test_autogalaxy/ellipse/jax`,
  `pytest test_autogalaxy/quantity/jax`.
- Existing PyAutoGalaxy unit tests still pass: `pytest test_autogalaxy`.
- Run `/smoke_test` on `autogalaxy_workspace`. The non-JAX paths must be
  unchanged — pytree registration only affects JIT, never the eager NumPy
  path.

__Out of scope__

- **Interferometer pytree registration** — shipped in PR #376; see
  `complete.md` for the entry. Originally this prompt was scoped to
  include interferometer; that work was discovered to be already done
  during the audit on 2026-05-08 and dropped from scope.
- **No workspace_test JAX visualization scripts in this task.** Those are
  written in the follow-up Phase 1C prompt
  (`autogalaxy_workspace_test/jax_viz_dataset_coverage.md`). That prompt
  is blocked on this one (and Phase 0b) landing.
- **No PyAutoLens equivalent.** PyAutoLens equivalents land separately if
  needed.
- **No production workspace adoption.** Tutorials don't get
  `use_jax_for_visualization=True` from this task.

__Reference__

- `@PyAutoFit/autofit/jax/pytrees.py` — autofit pytree machinery
- `@PyAutoGalaxy/autogalaxy/interferometer/model/analysis.py:165-184` —
  recently shipped sibling pattern to mirror (PR #376)
- `@PyAutoGalaxy/autogalaxy/analysis/jax_pytrees.py` — shared
  `register_galaxies_pytree()` helper
- `@PyAutoLens/autolens/imaging/model/analysis.py` —
  `AnalysisImaging._register_fit_imaging_pytrees` reference implementation
- `PyAutoPrompt/issued/fit_imaging_pytree.md` — Path A feasibility study (in-flight)
- `PyAutoPrompt/z_features/jax_visualization.md` — sequenced roadmap (Phase 0c)
