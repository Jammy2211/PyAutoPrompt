PyAutoGalaxy's `autogalaxy_workspace_test` has JAX visualization coverage
only for the `imaging/` dataset type. The audit on 2026-05-08 found:

| Dataset       | NumPy baseline | JAX (`_jax.py`) | jit-cached (`modeling_visualization_jit.py`) |
|---------------|---------------|-----------------|--------------------------------------------|
| imaging       | ✓             | ✓               | ✓                                          |
| interferometer| **missing**   | **missing**     | **missing**                                |
| ellipse       | **missing**   | **missing**     | **missing**                                |
| quantity      | **missing**   | **missing**     | **missing**                                |

This task fills every cell marked **missing**.

__Why this matters__

This is **Phase 1C** of `z_features/jax_visualization.md`. The other three
PyAutoGalaxy dataset types have **no** visualization smoke coverage at all,
so flipping the `use_jax_for_visualization` default in Phase 2 would risk
silent breakage for any user who runs an interferometer, ellipse, or
quantity fit with `use_jax=True`.

__Blockers — status as of 2026-05-14__

All library prerequisites are now green for the **interferometer** sub-step:

- Interferometer pytree registration — shipped in PR #376 ✓
- Interferometer dispatch swap (visualizer → `fit_for_visualization`) — shipped in PyAutoGalaxy #390 ✓ (Phase 0b)
- `ag.AnalysisInterferometer.__init__` `**kwargs` passthrough — shipped in PR #399 ✓

**Ellipse + quantity are blocked, but not by what the original prompt said.**
Phase 0c shipped (PR #401) registering `FitEllipse` + `FitQuantity` pytrees and
adding `**kwargs` passthrough on both analyses. However, that work surfaced a
deeper issue: **both visualizers bypass `analysis.fit_for_visualization`
entirely**:

- `VisualizerEllipse.visualize` calls `analysis.fit_list_from(instance)`
  (returns a `List[FitEllipse]`, not a single fit).
- `VisualizerQuantity.visualize` calls `analysis.fit_quantity_for_instance(instance)`
  (singular but custom-named).

So `use_jax_for_visualization=True` is a no-op for these two analyses despite
the pytree registration. Until the **two visualizer-dispatch follow-ups**
deferred from Phase 0c ship — small for quantity (`fit_from` alias), needs
design for ellipse (list-return contract) — the ellipse + quantity JAX scripts
would silently fall through to the eager path and provide false confidence.

__Scope narrowing — this task ships interferometer only__

Original scope covered all three remaining autogalaxy dataset types. Per the
Phase 0c discovery above, this task is **narrowed to interferometer only**.
Ellipse and quantity coverage moves to a follow-up that ships **after** the
visualizer dispatch fixes for those analyses. Their NumPy baselines are out of
scope here too — the existing `scripts/ellipse/visualization.py` (NumPy) is
sufficient until the JAX side is wireable end-to-end.

__Sub-step 1 — Interferometer (3 scripts)__

Add under `scripts/interferometer/`:

- `visualization.py` — NumPy baseline. Mirror the imaging baseline at
  `@autogalaxy_workspace_test/scripts/imaging/visualization.py`. Use the
  parametric Sersic source for the simplest case; reuse the existing
  interferometer simulator under
  `scripts/jax_likelihood_functions/interferometer/`.
- `visualization_jax.py` — `use_jax=True, use_jax_for_visualization=True`.
  Mirror `@autogalaxy_workspace_test/scripts/imaging/visualization_jax.py`.
- `modeling_visualization_jit.py` — caching probe + live Nautilus.
  Mirror the imaging analogue.

__Sub-step 2 + 3 — Ellipse + Quantity — OUT OF SCOPE (deferred)__

Per the scope narrowing above, ellipse and quantity JAX coverage is deferred
to a follow-up prompt that ships **after** the Phase 0c visualizer-dispatch
follow-ups land. The NumPy ellipse baseline at
`scripts/ellipse/visualization.py` already exists and is unchanged by this
task. Quantity has no NumPy baseline yet either — that arrives with the
follow-up.

__Constraints__

- Real searches, no `PYAUTO_TEST_MODE=1`. Use small `n_like_max` (≤ 1500)
  and `n_live=50` for the `_jit.py` scripts.
- Per-dataset-type `config_source/visualize/plots.yaml` overrides only if
  the default visualization output is too broad for smoke runtime.
- Each NumPy `visualization.py` is the regression baseline for its
  respective `_jax.py` — keep the model + mask + grid identical so the
  numerical comparison is meaningful if a future task adds one.

__Verification__

- All three new scripts pass when run directly with JAX enabled.
- Existing `autogalaxy_workspace_test/scripts/imaging/*` scripts continue to pass.
- The cached call in `interferometer/modeling_visualization_jit.py` is significantly faster than the first call.
- `modeling_visualization_jit.py` includes the explicit `rmtree(output/<path>/<name>/)` before the Nautilus call (lesson from PR #87) so reruns don't silently resume from cached samples.
- `visualization_jax.py` uses `enable_pytrees()` + `register_model(model)` from the start; no `try/except` wrapper (lessons from PR #85).

__Out of scope__

- No production `autogalaxy_workspace` adoption — Phase 3 of the roadmap.
- No PyAutoLens-side coverage (autolens interferometer is its own
  Phase 1A prompt).
- No change to PyAutoGalaxy itself; both the dispatch and the pytree
  registration prerequisites are upstream tasks.

__Reference__

- `@autogalaxy_workspace_test/scripts/imaging/visualization.py` — NumPy pattern
- `@autogalaxy_workspace_test/scripts/imaging/visualization_jax.py` — JAX pattern
- `@autogalaxy_workspace_test/scripts/imaging/modeling_visualization_jit.py` — JIT live pattern
- `PyAutoPrompt/autogalaxy/fit_pytree_registration_other_datasets.md` — blocker
- `PyAutoPrompt/autogalaxy/visualizer_fit_for_visualization_dispatch.md` — blocker
- `PyAutoPrompt/z_features/jax_visualization.md` — Phase 1C in the sequenced roadmap
