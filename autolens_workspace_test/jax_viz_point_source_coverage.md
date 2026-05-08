Point-source dataset coverage in `autolens_workspace_test` is the weakest
of the three dataset types. The audit on 2026-05-08 found:

- `scripts/point_source/visualization.py` — **missing** (no NumPy baseline either)
- `scripts/point_source/visualization_jax.py` — **missing**
- `scripts/point_source/modeling_visualization_jit.py` — **missing**

Existing point-source scripts under `scripts/jax_likelihood_functions/point_source/`
do exercise JAX likelihoods but **not** the visualizer end-to-end. There is also
a known JIT blocker called out in `scripts/CLAUDE.md`:

> `point_source/source_plane.py` | Point-source source-plane chi-squared
> (`FitPositionsSource`) — JIT currently blocked

That same blocker applies to the visualizer — the source-plane fit cannot
yet round-trip through `jax.jit` cleanly. The task therefore proceeds in
**three sub-steps** with the JIT path gated behind a feasibility check.

__Why this matters__

This is **Phase 1B** of `z_features/jax_visualization.md`. Point-source is
the only autolens dataset type with zero visualization coverage of any
flavour, so it is also the dataset type most at risk of regressing
silently when Phase 2 flips the `use_jax_for_visualization` default.

__Sub-step 1 — NumPy baseline__

Add `scripts/point_source/visualization.py` mirroring
`scripts/imaging/visualization.py` but for point-source:

- Reuse the simulator at `scripts/point_source/simulators/point_source.py`.
- Build a parametric mass + point-source model.
- Call `VisualizerPointSource.visualize` (or whatever the actual class is —
  verify by reading `@PyAutoLens/autolens/point/model/visualizer.py`).
- Assert `subplot_fit.png` (or the equivalent point-source subplot) lands
  on disk.
- `use_jax=False`, `use_jax_for_visualization=False`. NumPy only.

This must land first; without it there's no NumPy regression baseline to
compare the JAX path against.

__Sub-step 2 — JAX viz (image-plane chi-squared only)__

Add `scripts/point_source/visualization_jax.py`:

- Same model + dataset as sub-step 1, but use `FitPositionsImagePairAll`
  (image-plane chi-squared — the variant that **does** JIT, per the
  `point_source/image_plane.py` likelihood-functions script).
- `analysis = al.AnalysisPoint(dataset=dataset, use_jax=True,
  use_jax_for_visualization=True, ...)` — confirm the actual analysis
  class name from PyAutoLens.
- Assert `subplot_fit.png` (or equivalent) lands on disk and the fit was
  JAX-backed.

__Sub-step 3 — Live Nautilus jit-cached visualization__

Add `scripts/point_source/modeling_visualization_jit.py`:

- Mirror `scripts/imaging/modeling_visualization_jit.py` two-part
  structure (caching probe + live Nautilus quick-update).
- Use the image-plane chi-squared variant only (source-plane is currently
  JIT-blocked).
- Short `n_like_max` and `n_live=50` for runtime.

__Source-plane variant — feasibility gate__

Before adding any source-plane variant, verify the JIT blocker has been
resolved:

```python
import jax
analysis_sp = al.AnalysisPoint(dataset=ds, use_jax=True, use_jax_for_visualization=True,
                                fit_positions_cls=al.FitPositionsSource)
fit = analysis_sp.fit_for_visualization(instance)
jax.block_until_ready(fit.log_likelihood)
```

If this still raises (per the CLAUDE.md note), do **not** add a
source-plane visualization script in this task — file a follow-up prompt
and document the blocker. If it works, add `visualization_jax_source_plane.py`
analogously.

__Constraints__

- Real searches, no `PYAUTO_TEST_MODE=1` (per `autolens_workspace_test/CLAUDE.md`).
- Reuse existing point-source dataset / simulator — do not introduce new ones.
- Limit visualization to subplot files needed for assertion via a
  `config_source/visualize/plots.yaml` if the default plot set is too broad.

__Verification__

- All three new scripts pass when run directly:
  - `python scripts/point_source/visualization.py`
  - `python scripts/point_source/visualization_jax.py`
  - `python scripts/point_source/modeling_visualization_jit.py`
- `/smoke_test autolens_workspace_test point_source/visualization.py point_source/visualization_jax.py point_source/modeling_visualization_jit.py` — pass.
- The cached call in `modeling_visualization_jit.py` is significantly
  faster than the first call.
- Existing `scripts/jax_likelihood_functions/point_source/*.py` continue
  to pass.

__Out of scope__

- Source-plane chi-squared JIT — gated behind the feasibility check; if it
  still fails, becomes its own follow-up prompt.
- Production `autolens_workspace` adoption.
- Any change to PyAutoLens point-source code itself; if the visualizer
  isn't correctly dispatched through `fit_for_visualization`, that is a
  PyAutoLens bug to fix in a separate prompt — flag and stop, don't fix
  here.

__Reference__

- `@autolens_workspace_test/scripts/imaging/visualization.py` — NumPy pattern
- `@autolens_workspace_test/scripts/imaging/visualization_jax.py` — JAX pattern
- `@autolens_workspace_test/scripts/imaging/modeling_visualization_jit.py` — JIT live pattern
- `@autolens_workspace_test/scripts/jax_likelihood_functions/point_source/image_plane.py` — image-plane likelihood (JIT works)
- `@autolens_workspace_test/scripts/jax_likelihood_functions/point_source/source_plane.py` — source-plane likelihood (JIT blocked)
- `@autolens_workspace_test/scripts/CLAUDE.md` — JIT blocker note
- `@PyAutoLens/autolens/point/model/visualizer.py` — visualizer dispatch
- `PyAutoPrompt/z_features/jax_visualization.md` — Phase 1B in the sequenced roadmap
