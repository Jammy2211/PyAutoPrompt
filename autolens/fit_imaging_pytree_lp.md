Extend Path A (jax.jit-wrapped `analysis.fit_from`) to @PyAutoLens/autolens/imaging/fit_imaging.py
`FitImaging` with a **parametric light-profile** source (Sersic / Exponential etc., no MGE, no
pixelization), so that `jax.jit(analysis.fit_from)(instance)` returns a fully-populated `FitImaging`
with `jax.Array` leaves whose `log_likelihood` matches the NumPy path.

__Baseline already shipped__

The MGE parametric variant is landed end-to-end — see
@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py and the wiring in
@PyAutoLens/autolens/imaging/model/analysis.py `_register_fit_imaging_pytrees`. The library
machinery (`register_instance_pytree` and gated `AbstractNDArray` auto-registration) is in
@PyAutoArray/autoarray/abstract_ndarray.py.

__Why this variant exists__

MGE is a `Basis` of many `Gaussian` components — it exercises the "list of profiles" path. A
single-profile parametric source (`Sersic`, `Exponential`, `DevVaucouleurs`) is structurally
simpler and should mostly just work given the MGE PoC passes. The point of this task is to
**confirm** that, and to surface the operated-light-profile code path
(`Gaussian(+PSFOperated)`, `SersicOperated` etc.) which MGE does not exercise.

__Reference script__

@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/lp.py — parametric light profile
model on operated data. Copy that model into `lp_pytree.py` and wrap `analysis.fit_from` with
`jax.jit` per the MGE PoC's three-step pattern.

__What's likely to surface__

Probably nothing new — `Galaxy` + light/mass profiles + `register_model(model)` are already handled.
If anything, it's the operated-profile machinery that's been used less under the full pytree path.
If the PoC passes on first run, the deliverable is a cheap regression test that locks in the
existing behaviour.

__Visualization caveat__

The `fit_from` scalar round-trip this deliverable validates is unaffected. A follow-on task at
@admin_jammy/prompt/autolens/linear_light_profile_intensity_dict_pytree.md blocks
`analysis.fit_for_visualization` (the JIT path Nautilus live-search uses) for any model
containing a linear light profile: `linear_light_profile_intensity_dict` is keyed on profile
object identity and pytree unflatten produces fresh instances. This variant's reference model
(`lp.py`) is parametric and does **not** use linear light profiles, so it is unblocked. If the
script is later extended with `al.lp_linear.*` or an MGE basis, that sibling fix must land
first.

__Deliverables__

1. `autolens_workspace_test/scripts/jax_likelihood_functions/imaging/lp_pytree.py` that prints
   `PASS: jit(fit_from) round-trip matches NumPy scalar.`
2. Library PRs for any new registrations (likely none).
3. Confirmation in the workspace PR body whether this variant needed any new registrations — if
   yes, call that out so sibling tasks know.
