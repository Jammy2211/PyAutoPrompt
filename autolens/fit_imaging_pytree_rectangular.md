Extend Path A (jax.jit-wrapped `analysis.fit_from`) to @PyAutoLens/autolens/imaging/fit_imaging.py
`FitImaging` with a **rectangular pixelization** source, so that `jax.jit(analysis.fit_from)(instance)`
returns a fully-populated `FitImaging` with `jax.Array` leaves whose `log_likelihood` matches the
NumPy path.

__Baseline already shipped__

The MGE parametric variant has already been landed end-to-end:

- Library machinery: @PyAutoArray/autoarray/abstract_ndarray.py `_register_as_pytree` (gated
  auto-registration for `AbstractNDArray` subclasses on the JAX path) and `register_instance_pytree`
  (generic `__dict__`-flatten/unflatten helper with `no_flatten` aux-data escape hatch).
- Wiring: `AnalysisImaging._register_fit_imaging_pytrees` in
  @PyAutoLens/autolens/imaging/model/analysis.py registers `FitImaging`, `Tracer`, and `DatasetModel`
  when `use_jax=True`.
- PoC: @autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py —
  jit-wraps `fit_from`, asserts `fit.log_likelihood` matches the NumPy scalar to rtol=1e-4
  (observed ~1e-8).

This task is that same pattern, one step harder — the source is now a `Pixelization`, which means
`FitImaging` carries an `Inversion` and the pytree cascade reaches into the autoarray
inversion/mapper/mesh layer.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular.py has the full
model setup: `Galaxy(mass=Isothermal, shear=ExternalShear)` + source
`Galaxy(pixelization=Pixelization(mesh=RectangularAdaptImage, regularization=ConstantSplit))`.
Adopt the same model — only replace `fitness._vmap(parameters)` with the `jax.jit(analysis.fit_from)`
round-trip pattern from `mge_pytree.py`.

__What's likely to surface__

Because the lens and source `Galaxy` objects, and every light/mass profile, are already registered
via `autofit.jax.pytrees.register_model(model)`, the new offenders will be on the inversion side of
the fit. Candidates:

- `Inversion` (@PyAutoArray/autoarray/inversion/inversion/) — held as `fit.inversion`
- `Mapper` subclasses (`MapperRectangular`, `MapperRectangularNoInterp`, etc.) — held inside the
  inversion's linear-eqn construction
- `Pixelization` (@PyAutoArray/autoarray/inversion/pixelization/pixelization.py) — the model-level
  container; likely dynamic (mesh params can change per instance)
- Mesh objects (`Rectangular`, `RectangularAdaptImage`) — typically carry shape as static, params as
  dynamic
- `Regularization` (`ConstantSplit`, `Constant`, `AdaptiveBrightnessSplit`) — coefficient is dynamic,
  matrix layout is static
- `LinearEqn` / `LEq*` classes — the objects that actually build the normal equations
- `InversionImaging` (@PyAutoArray/autoarray/inversion/inversion/imaging/) — imaging-specific
  wrapper

The NNLS solver state is a known hard spot. `nnls_invariance.py` already ships NNLS under JAX on the
eager path; whether it round-trips through `jax.jit` without manual intervention is an open question
the PoC will answer.

__Approach__

1. Copy `mge_pytree.py` to `rectangular_pytree.py`; swap in the rectangular-pixelization model from
   `rectangular.py`. Keep the three-step pattern (NumPy reference scalar, `jax.jit(fit_from)` call,
   `rtol=1e-4` assertion).
2. Run it. The first failure identifies the next unregistered type.
3. For each offender: decide dynamic-vs-static (which `__dict__` keys go through `no_flatten`) and
   register via `register_instance_pytree` — ideally at the same registration site used by
   `_register_fit_imaging_pytrees` so the wiring stays in one place.
4. If an offender needs per-subclass flatten logic (not just `__dict__`), add a targeted helper
   rather than forcing `register_instance_pytree` to grow knobs.
5. Repeat until the assertion passes.
6. Ship: library PRs for any new registrations, workspace PR for the new script.

__Scope boundary__

- Do **not** touch non-JIT behaviour. The `xp is np` guard stays; this PoC only exercises the JAX
  path.
- Do **not** generalise to delaunay in the same PR — that's a separate task
  (`fit_imaging_pytree_delaunay.md`). Share registration code if natural; don't block one on the other.
- Do **not** register the NNLS solver's internal state as a traced pytree if it's stateful — if that
  turns out to be a blocker, document it in the PR and fall back to wrapping only up to the
  inversion boundary.

__Starting points__

- @autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py — baseline PoC
- @autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular.py — model setup
- @PyAutoArray/autoarray/abstract_ndarray.py — `register_instance_pytree`
- @PyAutoLens/autolens/imaging/model/analysis.py — `_register_fit_imaging_pytrees`
- @PyAutoArray/autoarray/inversion/ — where new offenders will live

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. This variant's reference model
(`rectangular.py`) uses an `Isothermal` lens and pixelization source with no linear light
profiles, so it is unblocked. If the script is later extended with a linear lens bulge
(e.g. `al.lp_linear.Sersic`) or an MGE basis, that sibling fix must land first.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular_pytree.py` that
   runs green and prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
2. Library PRs (PyAutoArray / PyAutoLens) for any newly registered types, each with an `## API
   Changes` section identifying the class and its `no_flatten` choice.
3. Notes in the workspace PR body on anything that couldn't be registered cleanly (e.g. NNLS solver
   state, if that turns out to be a blocker) so the next variant (delaunay) doesn't re-hit it blind.
