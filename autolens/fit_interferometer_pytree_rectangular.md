Extend Path A (jax.jit-wrapped `analysis.fit_from`) to
@PyAutoLens/autolens/interferometer/fit_interferometer.py `FitInterferometer` with a **rectangular
pixelization source**, so that `jax.jit(analysis.fit_from)(instance)` returns a fully-populated
`FitInterferometer` with `jax.Array` leaves whose `log_likelihood` matches the NumPy path.

__Depends on__

- `fit_interferometer_pytree_mge.md` — introduces `FitInterferometer` / `Interferometer` /
  `Visibilities` / `Transformer*` pytree registrations.
- `fit_imaging_pytree_rectangular.md` — introduces inversion / mapper / mesh / regularization
  pytree registrations.

__Why this variant matters__

Interferometer + pixelized source is the hardest combination in the matrix:

- The inversion's model-data → visibility mapping runs through the transformer, so a traced `jax.Array`
  has to flow image-plane → uv-plane cleanly inside the JIT trace.
- Pixelization mapping matrices are typically sparse; combining that with NUFFT (if used) may expose
  aux-vs-dynamic classification bugs that neither sibling task exercises alone.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/rectangular.py — lens
(Isothermal + ExternalShear) + rectangular pixelization source on interferometer data. Copy that
model into `rectangular_pytree.py` and apply the three-step pattern from `mge_pytree.py` with the
`FitInterferometer` wiring from its sibling task.

__What's likely to surface__

If both predecessors landed clean, nothing new. If this fails, the interaction between the
`Transformer` and the `Inversion` is the bug — likely one of:

- `InversionInterferometer` needs its own `register_instance_pytree` entry (it's
  imaging-vs-interferometer-specific in @PyAutoArray/autoarray/inversion/inversion/interferometer/)
- Transformer state accidentally being traced as dynamic when it should be aux
- Complex-dtype leakage between the transformer output and the linear-equation solver

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. This variant's reference model
uses an `Isothermal` lens and pixelization source with no linear light profiles, so it is
unblocked. If the script is later extended with a linear lens bulge or an MGE basis, that
sibling fix must land first.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/rectangular_pytree.py`
   that prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
2. Library PRs for any new registrations (possible: `InversionInterferometer`).
3. Notes on any transformer × inversion interaction surprises — this variant is the canary for
   that code path under JIT.
