Extend Path A (jax.jit-wrapped `analysis.fit_from`) to @PyAutoLens/autolens/imaging/fit_imaging.py
`FitImaging` with a **Delaunay pixelization** source, so that `jax.jit(analysis.fit_from)(instance)`
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
  jit-wraps `fit_from`, asserts `fit.log_likelihood` matches the NumPy scalar.

__Why delaunay is harder than rectangular__

Rectangular pixelization has a fixed grid geometry: the mesh shape is static, the image-to-source
mapping indices can be precomputed, and the regularization matrix has constant sparsity. Delaunay
adds on top of all that:

- The mesh itself is **data-derived** — source-plane pixel centres come from ray-traced image-plane
  positions (`KMeans` / adapt-image / `image_mesh.Overlay`). The triangulation is recomputed per
  fit.
- The `relocated_grid_from` logic and the Voronoi/Delaunay neighbour lookup are not obvious
  jit-compatible computations and have historically been points of friction for JAX.
- The mapper's `pix_indexes_for_sub_slim_index` is irregular — its shape depends on the mesh
  actually built at runtime.

Expect this task to surface one or more types that `rectangular_pytree.py` (its sibling task) did
not. Share everything you can, but don't force a shared abstraction before both variants are green.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/delaunay.py has the model setup:
same lens (Isothermal + ExternalShear), source is
`Galaxy(pixelization=Pixelization(mesh=DelaunayBrightnessImage, regularization=ConstantSplit))` or
similar. Use whichever delaunay variant is currently jit-safe on the likelihood path today (eager
JAX); the PoC extends that to the `FitImaging` return boundary.

__What's likely to surface__

On top of whatever `rectangular_pytree.py` registers:

- `DelaunayBrightnessImage` / `image_mesh.Overlay` subclasses (@PyAutoArray/autoarray/inversion/pixelization/image_mesh/)
- `MapperDelaunay` / `MapperVoronoi` / `MapperVoronoiNoInterp`
- The Delaunay triangulation object itself, if it ends up stored on the mapper
- Any `scipy.spatial.Delaunay` wrapper — this is **static** (C-level state), must ride as aux_data
  or be reconstructable from the mesh points

__Approach__

Same as the rectangular task:

1. Copy `mge_pytree.py` to `delaunay_pytree.py`; swap in the delaunay model from `delaunay.py`.
2. Run, identify the first offender, decide dynamic-vs-static, register, repeat.
3. If `scipy.spatial.Delaunay` (or equivalent) is held on an instance and can't be pytree-flattened,
   it rides as `no_flatten` aux — it's a derived object, not a parameter — as long as
   `unflatten` doesn't need to rebuild it. If unflatten does need it, add a lazy-rebuild property.
4. Ship: library PRs + workspace PR.

__Scope boundary__

- Do **not** refactor rectangular's registrations to share with delaunay in the same PR. Land
  delaunay independently, then follow-up with dedup if natural.
- Do **not** change `use_jax_for_visualization` dispatch. That flip happens in a later task, once
  all Fit variants are pytree-capable.

__Starting points__

- @autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py — baseline
- @autolens_workspace_test/scripts/jax_likelihood_functions/imaging/delaunay.py — model setup
- @autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular_pytree.py — sibling
  task (may land first; read any registration additions there before starting)
- @PyAutoArray/autoarray/inversion/pixelization/image_mesh/ — where Delaunay-specific offenders live
- @PyAutoArray/autoarray/inversion/pixelization/mappers/ — delaunay/voronoi mappers

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. This variant's reference model
(`delaunay.py`) uses pixelization source and an `Isothermal` lens with no linear light
profiles, so it is unblocked. If the script is later extended with linear lens light profiles
or an MGE basis, that sibling fix must land first.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/imaging/delaunay_pytree.py` that runs
   green.
2. Library PRs for any new registrations with `## API Changes` sections.
3. Notes in the workspace PR body on delaunay-specific surprises vs. rectangular.
