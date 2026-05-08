PyAutoLens visualizers were updated to dispatch through `analysis.fit_for_visualization(instance=instance)`
in PR #443 (commit 761c1778e, 2026-04-19) so that — when `use_jax_for_visualization=True` is set on
the analysis — the fit reused for plotting goes through the cached `jax.jit` wrapper added to the
autofit base `Analysis` in PR #1228.

PyAutoGalaxy was later given matching JAX pytree registration for `FitImaging` + `DatasetModel` +
`Galaxies` (PR #364, 2026-04-22) but the visualizer dispatch was never switched over. As a result
`use_jax_for_visualization=True` on `ag.AnalysisImaging` is currently a no-op for visualization —
the pytree machinery is in place but the call sites still go through the eager
`analysis.fit_from(instance=instance)` path.

__What to change__

1. `@PyAutoGalaxy/autogalaxy/imaging/model/visualizer.py:79` — swap
   `fit = analysis.fit_from(instance=instance)` for
   `fit = analysis.fit_for_visualization(instance=instance)`.
2. `@PyAutoGalaxy/autogalaxy/imaging/model/visualizer.py:176` — same swap inside the
   `visualize_combined` path that builds per-analysis fits in a multi-analysis scenario.
3. `@PyAutoGalaxy/autogalaxy/interferometer/model/visualizer.py:81` — same swap.
   Note: `AnalysisInterferometer` pytree registration shipped in PyAutoGalaxy PR #376
   (see `complete.md` 2026-04-? entry — `_register_fit_interferometer_pytrees` is now
   active in `autogalaxy/interferometer/model/analysis.py:165-184`), so dispatching
   through `fit_for_visualization` is now safe for interferometer. Originally this
   bullet was deferred — it was added to scope on 2026-05-08 once interferometer
   pytree registration landed.

`fit_for_visualization` is defined on the autofit base `Analysis`. It dispatches to a
`jax.jit`-cached wrapper when `use_jax_for_visualization=True` and falls back to plain `fit_from`
otherwise — so this change is safe for the NumPy default path.

__Reference (PyAutoLens equivalent)__

- `@PyAutoLens/autolens/imaging/model/visualizer.py:97` — single-analysis dispatch
- `@PyAutoLens/autolens/imaging/model/visualizer.py:239` — multi-analysis `visualize_combined` dispatch

__Verification__

- `autogalaxy_workspace_test/scripts/imaging/visualization_jax.py` should still pass (uses
  `use_jax_for_visualization=True`) and will now actually exercise the jit-cached path rather
  than silently no-op.
- `autogalaxy_workspace_test/scripts/imaging/visualization.py` should still pass (NumPy path —
  `fit_for_visualization` falls back to `fit_from` when the flag is off).
- Run both via `/smoke_test`.
- For interferometer the equivalent workspace_test scripts don't exist yet — coverage is added
  by `autogalaxy_workspace_test/jax_viz_dataset_coverage.md` (Phase 1C of the JAX
  visualization roadmap). Verify the dispatch swap doesn't break existing
  `autogalaxy_workspace_test/scripts/interferometer/` (NumPy path — should be unchanged).

__Out of scope__

- Production workspace adoption (autogalaxy_workspace scripts opting into
  `use_jax_for_visualization=True`) — defer until Path A from `issued/fit_imaging_pytree.md` lands.
- `AnalysisEllipse` / `AnalysisQuantity` — these have no pytree registration today;
  covered by `autogalaxy/fit_pytree_registration_other_datasets.md` (which is now
  scoped to ellipse + quantity only — interferometer pytree registration shipped in
  PR #376).

__Background__

Original feature: `complete.md` entries `jax-visualization` and `mge-jit-visualization`
(both 2026-04-19). Autogalaxy imaging pytree registration: `complete.md` entry for the
imaging port (2026-04-22).
