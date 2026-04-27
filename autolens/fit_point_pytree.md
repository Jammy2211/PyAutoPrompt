Extend Path A (jax.jit-wrapped `analysis.fit_from`) to
@PyAutoLens/autolens/point/fit/fit_point_dataset.py `FitPointDataset`, so that
`jax.jit(analysis.fit_from)(instance)` returns a fully-populated `FitPointDataset` with
`jax.Array` leaves whose `log_likelihood` matches the NumPy path.

__Baseline already shipped__

The `FitImaging` + MGE variant is landed — see
@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py. This task is the
point-source analogue.

__Why this variant is structurally different from Imaging / Interferometer__

Point-source fitting does not fit pixel data; it fits the multiple-image positions of a lensed
quasar (and optionally their fluxes and relative time delays). The data types are entirely
different:

- Dataset is `PointDataset` (positions, fluxes, positions_noise_map) — no gridded image, no PSF
- The "model prediction" is the result of solving the lens equation for source-plane positions
  given a lens model, then forward-mapping back — not a pixel-space image
- The `Fit*` object has no `model_image` / `residual_map` / `chi_squared_map` in the usual sense
- Position-solving uses iterative root-finders which historically have been tricky under JAX's
  functional-purity constraints

There is real risk the PoC surfaces a non-differentiable / non-JIT-safe operation buried in the
position-solver. If so, this task's deliverable is a clean statement of the blocker, not a forced
registration.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/point_source/point.py and
@autolens_workspace_test/scripts/jax_likelihood_functions/point_source/simulator.py. Use those for
dataset + model setup. Swap `AnalysisImaging` → `AnalysisPoint`, `FitImaging` → `FitPointDataset`;
keep the three-step `mge_pytree.py` pattern.

__What's likely to surface__

- `FitPositionsImagePairAll` / `FitPositionsSource` / `FitPointDataset` — need
  `register_instance_pytree` with `no_flatten=("dataset", "settings")`. Without this, Path A
  raises `TypeError: function fit_from ... returned a value of type FitPositionsSource ... which
  is not a valid JAX type` — seen right now in the workspace (see __Workspace examples of the
  clunky API__ below).
- `PointDataset` — rides as aux
- `AnalysisPoint._register_fit_point_pytrees` — new wiring, mirror the imaging one
- `PointSolver` / any iterative root-finder state — **likely blocker** for source-plane or any
  path that goes through the solver. If the solver's internal state is stateful or Python-level,
  it cannot be traced through `jax.jit`. Fallback:
  - Document the blocker clearly
  - Propose either (a) refactoring the solver to `jax.lax.while_loop` / `jax.lax.scan`, or
    (b) closing over the solver as a Python callback via `jax.experimental.host_callback` /
    `jax.pure_callback`, or (c) accepting that the point-source path is jit-incompatible and
    documenting that in CLAUDE.md

__Second stacked blocker: `xp` propagation in `Grid2DIrregular`__

Even once `FitPositionsSource` is pytree-registered, the source-plane full-pipeline JIT
(`jax.jit(analysis.fit_from)` with `fit_positions_cls=al.FitPositionsSource`) fails with
`jax.errors.TracerArrayConversionError` at
`Grid2DIrregular.grid_2d_via_deflection_grid_from` because the underlying array module (`xp`)
is not threaded through consistently — somewhere inside that helper the code reaches for
`numpy` directly on tracer values. The image-plane variant (`FitPositionsImagePairAll`) is NOT
affected because it does not exercise the same deflection-grid helper.

Root-cause is documented in
@autolens_workspace_developer/jax_profiling/point_source/source_plane.py. Fix is library-side
in `Grid2DIrregular` (or a helper it calls) — swap the offending `np.…` calls for `xp.…` and
propagate `xp` through the call chain. This is its own small task; treat it as a companion to
the pytree registration rather than a blocker on it.

__Workspace examples of the clunky API (what a user has to write today)__

The current state forces a user to guard Path A with a `try/except` that swallows two distinct
error classes and prints a BLOCKER note instead of a clean PASS. This is ugly, and landing
both fixes above is what makes it go away.

- `autolens_workspace_test/scripts/jax_likelihood_functions/point_source/source_plane.py` —
  ships the workaround live. The relevant shape:

  ```python
  fit_jit_fn = jax.jit(analysis_jit.fit_from)
  try:
      fit = fit_jit_fn(instance)
      # happy path — not currently reachable
      ...
  except (jax.errors.TracerArrayConversionError, TypeError) as e:
      # Two stacked blockers gate the full-pipeline JIT:
      #   1. FitPositionsSource is not pytree-registered (TypeError).
      #   2. Grid2DIrregular.grid_2d_via_deflection_grid_from xp bug (TracerArrayConversionError).
      print(f"BLOCKER: source-plane jit(fit_from) is gated by: {type(e).__name__}: {e}")
  ```

- `autolens_workspace_developer/jax_profiling/point_source/source_plane.py` — the profiling
  mirror of the same pattern, with identical `try/except` shape.

- `autolens_workspace_test/scripts/jax_likelihood_functions/point_source/image_plane.py` — the
  image-plane sibling that shows how this SHOULD look once both blockers are resolved (no
  `try/except`, just `fit = jax.jit(analysis_jit.fit_from)(instance)` → assert scalar equality
  → print `PASS: jit(fit_from) round-trip matches NumPy scalar.`). That is the target shape
  for source-plane.

__Acceptance for this task__: delete the `try/except` BLOCKER guard from `source_plane.py` and
have it print the same `PASS: …` line that `image_plane.py` already does. If that's not
achievable because only one of the two blockers was fixed, update the comment in the
`try/except` to narrate exactly which blocker remains and link to the follow-up.

__Approach__

1. Add `_register_fit_point_pytrees` in
   @PyAutoLens/autolens/point/model/analysis.py matching the imaging pattern.
2. Copy `mge_pytree.py` to `point_source/point_pytree.py`; swap dataset/analysis/fit types.
3. Run, fix, iterate.
4. If the position-solver is the blocker, stop and write up the blocker instead of forcing a
   workaround — the follow-up refactor is a separate task.

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. Point-source fitting does not
use light profiles at all, so this variant is **not** affected by that issue — flagging here
for matrix completeness only.

__Deliverables__

1. One of:
   a. `autolens_workspace_test/scripts/jax_likelihood_functions/point_source/point_pytree.py` that
      prints `PASS: jit(fit_from) round-trip matches NumPy scalar.` (if the solver is JIT-safe), OR
   b. A clear write-up of the blocker (which function, why it's not JIT-safe, proposed
      resolutions) committed as a markdown note inline with the PoC attempt.
2. Library PRs for any new registrations (at minimum: `FitPointDataset`,
   `_register_fit_point_pytrees`).
3. Notes in the workspace PR body on solver JIT-safety status.
