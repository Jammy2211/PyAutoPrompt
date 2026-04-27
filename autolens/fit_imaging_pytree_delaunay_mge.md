Extend Path A (jax.jit-wrapped `analysis.fit_from`) to @PyAutoLens/autolens/imaging/fit_imaging.py
`FitImaging` with a **Delaunay pixelization source AND an MGE lens**, so that
`jax.jit(analysis.fit_from)(instance)` returns a fully-populated `FitImaging` with `jax.Array`
leaves whose `log_likelihood` matches the NumPy path.

__Depends on__

- `fit_imaging_pytree_delaunay.md` — introduces delaunay-specific pytree registrations
  (triangulation, delaunay/voronoi mappers, image-mesh objects).
- MGE PoC (already shipped) — covers the `Basis`-of-Gaussians light side.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/delaunay_mge.py — lens uses MGE
bulge + mass, source uses Delaunay pixelization. Copy that model into `delaunay_mge_pytree.py` and
apply the three-step pattern from `mge_pytree.py`.

__What's likely to surface__

If the delaunay and MGE sibling tasks have landed clean, this should be the product of the two —
no new registrations needed. If it doesn't pass, the MGE lens's lensed image is interacting with
the data-derived delaunay mesh construction in a way the simpler variants missed.

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. This variant's MGE lens bulge
**is** a linear light profile, so any `modeling_*.py` script that layers on
`use_jax_for_visualization=True` is blocked on that sibling fix.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/imaging/delaunay_mge_pytree.py` that
   prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
2. Library PRs for any new registrations (likely none).
3. Notes in the workspace PR body on any MGE-lens × delaunay-source interaction surprises.
