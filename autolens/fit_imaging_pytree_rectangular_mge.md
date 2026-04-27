Extend Path A (jax.jit-wrapped `analysis.fit_from`) to @PyAutoLens/autolens/imaging/fit_imaging.py
`FitImaging` with a **rectangular pixelization source AND an MGE lens**, so that
`jax.jit(analysis.fit_from)(instance)` returns a fully-populated `FitImaging` with `jax.Array`
leaves whose `log_likelihood` matches the NumPy path.

__Depends on two sibling tasks__

- `fit_imaging_pytree_rectangular.md` — must land first; introduces all the inversion /
  mapper / mesh / regularization pytree registrations.
- MGE PoC (already shipped) — covers the `Basis`-of-Gaussians light side.

This task is the intersection — combine an MGE lens bulge with a rectangular-pixelization source.
If the two sibling tasks have landed clean, this should reduce to "pick the model, verify the
assertion passes". If it doesn't pass, the interaction between those two code paths is the bug.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular_mge.py — lens uses MGE
bulge (`al.lp_basis.Basis` of Gaussians) plus NFW/Isothermal mass, source uses rectangular
pixelization. Copy that model into `rectangular_mge_pytree.py` and apply the three-step pattern
from `mge_pytree.py`.

__What's likely to surface__

- Interactions between MGE lens-galaxy pytree flattening and the inversion layer's use of the
  lensed image.
- Any code that caches shape/index info at the first `fit_from` call and would baked-in-trace
  across successive calls.

If the two predecessor tasks are clean, expect this to pass with no new registrations.

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. This variant's MGE lens bulge
**is** a linear light profile, so any `modeling_*.py` script that layers on
`use_jax_for_visualization=True` is blocked on that sibling fix.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/imaging/rectangular_mge_pytree.py` that
   prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
2. Library PRs for any new registrations (likely none if predecessors landed clean).
3. Notes in the workspace PR body on any MGE-lens × pixelization-source interaction surprises.
