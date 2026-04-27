Extend Path A (jax.jit-wrapped `analysis.fit_from`) to @PyAutoLens/autolens/imaging/fit_imaging.py
`FitImaging` for a **group-scale lens** (multiple main-lens galaxies + line-of-sight galaxies) with
an MGE source, so that `jax.jit(analysis.fit_from)(instance)` returns a fully-populated `FitImaging`
with `jax.Array` leaves whose `log_likelihood` matches the NumPy path.

__Baseline already shipped__

The single-lens MGE variant is landed end-to-end — see
@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py. This task extends
the same pattern to a `Tracer` with more than two galaxies (more than two planes in the
multi-plane case, or multiple co-redshift galaxies in the same plane).

__Why this variant exists__

The base MGE PoC has a single lens at z=0.5 and a single source at z=1.0. Group lenses add:

- Multiple `Galaxy` instances at the lens-plane redshift (main + satellites)
- Line-of-sight galaxies at intermediate redshifts (extra planes)
- `Galaxies` containers that now iterate longer

Every `Galaxy` and its profiles are already pytree-registered via `register_model(model)`. The
structural risk is in the `Tracer`'s plane-grouping logic and in anything that closes over the
number of galaxies at trace time (e.g. a cached mapping from galaxy → plane).

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_group.py — same MGE source as
`mge.py`, lens plane carries extras. Copy that model into `mge_group_pytree.py` and apply the
three-step pattern from `mge_pytree.py`.

__What's likely to surface__

- `Galaxies` — pytree-friendly via Python list recursion? Verify.
- `Tracer.planes` / `Tracer.plane_redshifts` — already aux via `no_flatten=("cosmology",)`; confirm
  that plane index lookups don't trace through static structure.
- Any per-plane caching on `Tracer` that could bake a shape into the trace.

If nothing new surfaces, this is a low-cost regression test that locks in group-scale behaviour
under JIT.

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. This variant's MGE basis **is**
a linear light profile, so any `modeling_*.py` script that layers on
`use_jax_for_visualization=True` is blocked on that sibling fix.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_group_pytree.py` that
   prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
2. Library PRs for any new registrations (likely none if the MGE PoC covered the surface).
3. Notes in the workspace PR body on whether `Galaxies` / multi-plane `Tracer` needed explicit
   handling.
