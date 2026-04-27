Extend Path A (jax.jit-wrapped `analysis.fit_from`) to @PyAutoLens/autolens/imaging/fit_imaging.py
`FitImaging` for a **double-source-plane** lens (three-plane system) with a rectangular
pixelization source, so that `jax.jit(analysis.fit_from)(instance)` returns a fully-populated
`FitImaging` with `jax.Array` leaves whose `log_likelihood` matches the NumPy path.

__Depends on__

- `fit_imaging_pytree_rectangular.md` — introduces inversion pytree registrations.
- `fit_imaging_pytree_mge_group.md` — exercises multi-plane `Tracer`.

__Why this variant exists__

Double-source-plane (DSPL) lenses are cosmologically interesting: two sources at different
redshifts are both lensed by the same deflector, giving joint constraints on geometry. The
structural addition over `rectangular_pytree.py`:

- Three planes (lens + two sources) instead of two.
- Cosmology distance calculations across multiple source redshifts.
- `Tracer.traced_grid_2d_list_from` returns 3 grids, not 2.
- The `FitImaging` may build two separate `Inversion`s (one per source plane), or a block-diagonal
  combined inversion — this is the main structural unknown.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular_dspl.py and
@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/simulator_dspl.py. Use that model
(lens at z~0.5, source_1 at z~1.0, source_2 at z~2.0, each source using rectangular pixelization).

__What's likely to surface__

- If `FitImaging` holds a `List[Inversion]` instead of a single `Inversion`, the flatten path needs
  to recurse through Python's default list pytree handling — usually fine, but verify.
- Cosmology-distance cache hits (flagged in the separate `smoke-test-optimization` task as a
  source of 176 calls per fit for a 2-plane system; DSPL will hit harder). These are aux, not
  dynamic — confirm `no_flatten=("cosmology",)` on `Tracer` still suffices and no distance cache
  escapes into traced state.
- Any code that assumes `len(tracer.planes) == 2`.

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. This variant's reference model
uses pixelization sources and an `Isothermal` lens with no linear light profiles, so it is
unblocked. If the DSPL script is later extended with linear lens/source light profiles, that
sibling fix must land first.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular_dspl_pytree.py`
   that prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
2. Library PRs for any new registrations.
3. Notes in the workspace PR body on DSPL-specific observations (multiple inversions, cosmology
   cache interaction).
