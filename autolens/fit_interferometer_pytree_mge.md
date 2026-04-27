Extend Path A (jax.jit-wrapped `analysis.fit_from`) to
@PyAutoLens/autolens/interferometer/fit_interferometer.py `FitInterferometer` with an **MGE source**,
so that `jax.jit(analysis.fit_from)(instance)` returns a fully-populated `FitInterferometer` with
`jax.Array` leaves whose `log_likelihood` matches the NumPy path.

__Baseline already shipped__

The `FitImaging` + MGE variant is landed — see
@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py. This task is the
equivalent on the visibility / uv-plane side instead of CCD imaging.

__Why this variant exists__

Interferometer fitting replaces real-space convolution with complex-visibility transforms. The
structural differences from `FitImaging`:

- Dataset is `Interferometer` (visibilities, noise-map, uv wavelengths) instead of `Imaging`
  (image, noise-map, PSF)
- The data-space arrays are complex-valued `Visibilities` (not real `Array2D`)
- A `TransformerDFT` / `TransformerNUFFT` maps image-plane → visibility-plane
- `FitInterferometer` and `AnalysisInterferometer` inherit from `FitImaging`-analogues but carry a
  different attribute set

The MGE light model and lens mass model are identical to the imaging variant — the interesting
question is what the interferometer's data-side types need for pytree registration.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/mge.py and
@autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/simulator.py. Use those for
dataset + model setup. Swap `AnalysisImaging` → `AnalysisInterferometer` and
`FitImaging` → `FitInterferometer`; keep the rest of the `mge_pytree.py` pattern.

__What's likely to surface__

- `FitInterferometer` — needs `register_instance_pytree` similar to `FitImaging`, with
  `no_flatten=("dataset", "settings", "adapt_images")`
- `Interferometer` dataset — rides as aux (static per analysis)
- `Visibilities`, `VisibilitiesNoiseMap` — complex-valued `AbstractNDArray` subclasses; auto-register
  if constructed with `xp=jnp`, but verify the `instance_flatten` round-trip works for complex
  dtypes
- `TransformerDFT` / `TransformerNUFFT` — likely static (compiled transform plan + uv coords);
  register with all attrs in `no_flatten`
- `AnalysisInterferometer._register_fit_interferometer_pytrees` — new wiring, mirror the imaging
  one in @PyAutoLens/autolens/imaging/model/analysis.py

__Approach__

1. Add `_register_fit_interferometer_pytrees` in
   @PyAutoLens/autolens/interferometer/model/analysis.py matching the imaging pattern.
2. Copy `mge_pytree.py` to `interferometer/mge_pytree.py`; swap dataset/analysis/fit types.
3. Run, fix, ship per the three-step pattern.

__Scope boundary__

- Do **not** register `TransformerNUFFT`'s internal FFT plan as traced pytree — it's
  compile-time state, must ride as aux.
- Do **not** change complex-dtype handling on `Visibilities`. If complex-valued traced arrays
  behave oddly under `jax.jit`, document it rather than working around it — the simpler fix lives
  closer to the autoarray layer.

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. The interferometer visualizer
walks the same dict, so this variant's MGE source **is** affected; any `modeling_*.py` script
that layers on `use_jax_for_visualization=True` is blocked on that sibling fix.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/mge_pytree.py` that
   prints `PASS: jit(fit_from) round-trip matches NumPy scalar.`
2. Library PRs: PyAutoLens `_register_fit_interferometer_pytrees`, plus any PyAutoArray
   registrations for interferometer-side types.
3. Notes on Visibilities complex-dtype round-trip (worked out-of-the-box or needed intervention).
