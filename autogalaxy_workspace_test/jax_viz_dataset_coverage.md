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

__Blockers — must land first__

State as of 2026-05-08 (post-audit):

- **Interferometer pytree registration is already shipped** (PR #376, see
  `autogalaxy/interferometer/model/analysis.py:165-184`). No blocker for
  the interferometer sub-step's JAX support.
- **Interferometer dispatch swap is still pending** —
  `PyAutoPrompt/autogalaxy/visualizer_fit_for_visualization_dispatch.md`
  (Phase 0b, scope extended 2026-05-08 to cover the interferometer
  visualizer line 81) must land before the interferometer `_jax.py`
  / `_jit.py` scripts will actually exercise the JIT path. Without it
  the scripts will silently fall through to the eager NumPy path.
- **Ellipse and quantity pytree registration is still pending** —
  `PyAutoPrompt/autogalaxy/fit_pytree_registration_other_datasets.md`
  (Phase 0c, now scoped to ellipse + quantity only) must land before
  the ellipse / quantity sub-steps' JAX scripts will run at all.

This task can start the **interferometer NumPy baseline** (sub-step 1's
`visualization.py`) immediately even with both blockers open, since the
NumPy path is unaffected. The `_jax.py` and `_jit.py` scripts must wait
for Phase 0b. The ellipse and quantity sub-steps must wait for Phase 0c.

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

__Sub-step 2 — Ellipse (2 scripts)__

Add under `scripts/ellipse/`:

- `visualization.py` — NumPy baseline. The ellipse fit is simpler than
  imaging (no inversion, no PSF) so the script is shorter. Use a
  pre-canned ellipse dataset; if none exists, simulate a small one via
  `ag.Ellipse(...)` directly inside the script.
- `visualization_jax.py` — `use_jax=True, use_jax_for_visualization=True`.
  No `modeling_visualization_jit.py` required for this dataset type unless
  ellipse fits already use Nautilus quick-update visualization in the
  product — verify by reading `@PyAutoGalaxy/autogalaxy/ellipse/model/`.

__Sub-step 3 — Quantity (2 scripts)__

Add under `scripts/quantity/`:

- `visualization.py` — NumPy baseline. Quantity fits compute residuals on
  derived quantities (`convergence_2d`, `deflections_yx_2d`,
  `potential_2d`). Use the simplest case — convergence — for the script.
- `visualization_jax.py` — `use_jax=True, use_jax_for_visualization=True`.
  Same Nautilus-skip rule as ellipse: no `_jit.py` unless quantity fits
  use quick-update in the product.

__Constraints__

- Real searches, no `PYAUTO_TEST_MODE=1`. Use small `n_like_max` (≤ 1500)
  and `n_live=50` for the `_jit.py` scripts.
- Per-dataset-type `config_source/visualize/plots.yaml` overrides only if
  the default visualization output is too broad for smoke runtime.
- Each NumPy `visualization.py` is the regression baseline for its
  respective `_jax.py` — keep the model + mask + grid identical so the
  numerical comparison is meaningful if a future task adds one.

__Verification__

- All new scripts pass when run directly.
- `/smoke_test autogalaxy_workspace_test interferometer/visualization.py interferometer/visualization_jax.py interferometer/modeling_visualization_jit.py ellipse/visualization.py ellipse/visualization_jax.py quantity/visualization.py quantity/visualization_jax.py` — pass.
- Existing `autogalaxy_workspace_test/scripts/imaging/*` scripts continue
  to pass.
- The cached call in `interferometer/modeling_visualization_jit.py` is
  significantly faster than the first call.

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
