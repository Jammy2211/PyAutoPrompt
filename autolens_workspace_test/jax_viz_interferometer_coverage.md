PyAutoLens interferometer dataset coverage in `autolens_workspace_test` is
NumPy-only today. The audit on 2026-05-08 found:

- `scripts/interferometer/visualization.py` — exists (NumPy path)
- `scripts/interferometer/visualization_jax.py` — **missing**
- `scripts/interferometer/modeling_visualization_jit.py` — **missing**

Imaging has all three. This task fills the gap so we have JAX visualization
coverage for both autolens dataset types before Phase 2 of the JAX
visualization roadmap flips the `use_jax_for_visualization` default.

__Why this matters__

This is **Phase 1A** of `z_features/jax_visualization.md`. PyAutoLens's
`AnalysisInterferometer` already dispatches through
`analysis.fit_for_visualization(instance=instance)` (see
`@PyAutoLens/autolens/interferometer/model/visualizer.py`), so the wiring
is in place — we just need the workspace_test scripts that exercise it.

__What to add__

1. `scripts/interferometer/visualization_jax.py` — mirror
   `scripts/imaging/visualization_jax.py` but use the interferometer
   dataset and `AnalysisInterferometer`. The reference script is at
   `@autolens_workspace_test/scripts/imaging/visualization_jax.py`. Reuse
   the existing simulator under `scripts/jax_likelihood_functions/interferometer/`
   for the input visibilities + UV coverage.

   Key construction:
   ```python
   analysis = al.AnalysisInterferometer(
       dataset=dataset,
       use_jax=True,
       use_jax_for_visualization=True,
       title_prefix="JAX_PILOT",
   )
   VisualizerInterferometer.visualize(
       analysis=analysis,
       paths=paths,
       instance=instance,
       during_analysis=False,
   )
   ```

   Assert `subplot_fit.png` lands on disk and the fit was JAX-backed.

2. `scripts/interferometer/modeling_visualization_jit.py` — mirror
   `scripts/imaging/modeling_visualization_jit.py`. Two parts:
   - **Part 1**: caching probe — call `analysis.fit_for_visualization(instance)`
     twice and assert the second call is significantly faster than the first
     (cached `_jitted_fit_from`).
   - **Part 2**: live Nautilus run with `iterations_per_quick_update=500`,
     short `n_like_max`, asserts `subplot_fit.png` files land under the
     output search root.

__Constraints__

- Reuse the existing `scripts/jax_likelihood_functions/interferometer/`
  simulator pattern; do **not** introduce a new dataset path.
- Use a small `n_like_max` (≤ 1500) and `n_live=50` to keep runtime tight.
- Follow the same `config_source/visualize/plots.yaml` constraint as the
  imaging script — limit visualisation to `subplot_fit.png` and
  `subplot_tracer.png` (and `subplot_inversion_0.png` if applicable) so
  runtime stays smoke-test-friendly.
- Both scripts MUST run without `PYAUTO_TEST_MODE=1`. Per
  `autolens_workspace_test/CLAUDE.md`, this repo runs with real (capped)
  searches.

__Verification__

- `python scripts/interferometer/visualization_jax.py` — passes,
  produces expected PNGs.
- `python scripts/interferometer/modeling_visualization_jit.py` — passes,
  cached call faster than first call, ≥ 1 `subplot_fit.png` produced
  during the search.
- `/smoke_test autolens_workspace_test interferometer/visualization_jax.py interferometer/modeling_visualization_jit.py` —
  pass.
- The pre-existing `scripts/interferometer/visualization.py` (NumPy path)
  must still pass — `fit_for_visualization` falls back to `fit_from` when
  the flag is off.

__Out of scope__

- No production `autolens_workspace` adoption — that's Phase 3 of the
  roadmap, blocked on Phase 2.
- No changes to PyAutoLens itself; the dispatch is already wired.
- No new pytree registration; PyAutoLens interferometer pytree work, if
  needed, lands in a separate task — assess only if the script crashes.

__Reference__

- `@autolens_workspace_test/scripts/imaging/visualization_jax.py` — pattern
- `@autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py` — pattern
- `@autolens_workspace_test/scripts/interferometer/visualization.py` — NumPy baseline
- `@PyAutoLens/autolens/interferometer/model/visualizer.py` — already dispatches via `fit_for_visualization`
- `PyAutoPrompt/z_features/jax_visualization.md` — Phase 1A in the sequenced roadmap
