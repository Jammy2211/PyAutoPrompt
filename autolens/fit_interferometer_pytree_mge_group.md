Extend Path A (jax.jit-wrapped `analysis.fit_from`) to
@PyAutoLens/autolens/interferometer/fit_interferometer.py `FitInterferometer` for a **group-scale
lens** with an MGE source, so that `jax.jit(analysis.fit_from)(instance)` returns a fully-populated
`FitInterferometer` with `jax.Array` leaves whose `log_likelihood` matches the NumPy path.

__Depends on__

- `fit_interferometer_pytree_mge.md` — introduces all `FitInterferometer`-side pytree
  registrations.
- `fit_imaging_pytree_mge_group.md` — covers the group-scale `Tracer` path.

This task is the intersection. If the two siblings have landed clean, this should be
model-swap-and-verify.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/mge_group.py. Copy its
model into `mge_group_pytree.py` and apply the three-step pattern from the MGE PoC with the
`FitInterferometer` wiring from its sibling task.

__What's likely to surface__

Probably nothing new. Land as a regression test that locks in the group-scale + interferometer
combination.

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. This variant's MGE source **is**
a linear light profile, so any `modeling_*.py` script that layers on
`use_jax_for_visualization=True` is blocked on that sibling fix.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/mge_group_pytree.py`
   that prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
2. Library PRs for any new registrations (likely none).
3. Notes on any interaction between multi-plane `Tracer` and the interferometer transformer.
