We just fixed a long-standing bug in
@PyAutoLens/autolens/imaging/model/analysis.py where the CPU branch of
`AnalysisImaging.log_likelihood_function` returned `fit.log_likelihood`
instead of `fit.figure_of_merit` (PR #504, merged). For pixelization
inversion fits these differ by the regularization log-det terms of the
Bayesian log evidence, so every CPU-driven nested-sampler search with a
pixelization source was silently optimising the wrong objective —
nested samplers drifted to `outer_coefficient ≈ 0` (noise-overfit
degenerate mode) instead of converging to the physical Bayesian
maximum.

The bug went unnoticed for a long time because the existing unit test
in @PyAutoLens/test_autolens/imaging/model/test_analysis_imaging.py
(`test__figure_of_merit__matches_correct_fit_given_galaxy_profiles`)
used a purely parametric Sersic model — where
`figure_of_merit == log_likelihood` — and therefore didn't exercise the
diverging branch. PR #504 adds one focused regression test for the
pixelization case.

We need to extend the `_test` workspaces with broader, end-to-end
assertions so that any future regression of this kind gets caught early
on realistic configurations, not just the minimal 7x7 unit-test fixture.

What needs adding:

1. Identify the regression-style integration scripts in
   @autolens_workspace_test/scripts/imaging/ (likely `model_fit.py`,
   `modeling_visualization_jit_delaunay.py`,
   `modeling_visualization_jit_rectangular.py` and similar) and in
   @autogalaxy_workspace_test/scripts/imaging/ that fit a pixelization
   source. For each, add a small block (does NOT need its own script —
   can sit inline before the search starts) that:

   - Builds the model's prior-median instance via
     `instance = model.instance_from_prior_medians()`.
   - Computes `analysis_value = analysis.log_likelihood_function(instance=instance)`.
   - Reconstructs the equivalent `fit = al.FitImaging(dataset=..., tracer=..., adapt_images=..., settings=...)`
     directly (via `analysis.tracer_via_instance_from(instance)` and
     `analysis.adapt_images_via_instance_from(instance, galaxies=tracer.galaxies)`).
   - Asserts `analysis_value == pytest.approx(fit.figure_of_merit)`.
   - Asserts `fit.figure_of_merit != pytest.approx(fit.log_likelihood, rel=1e-6)`
     so the test cannot pass tautologically on a pixelization-less model.

2. Add an equivalent `Fitness` assertion. Build the Fitness wrapper the
   way Nautilus would:

   ```python
   from autofit.non_linear.fitness import Fitness
   fitness = Fitness(
       model=model, analysis=analysis, paths=None,
       fom_is_log_likelihood=True, resample_figure_of_merit=-1.0e99,
   )
   parameter_vector = model.physical_values_from_prior_medians
   assert fitness.call_wrap(parameter_vector) == pytest.approx(fit.figure_of_merit)
   ```

   This guards the Nautilus-facing surface specifically. The
   single-instance `log_likelihood_function` check guards the
   AnalysisImaging surface; Fitness.call_wrap guards the conversion
   from parameter vector → instance → log-evidence that Nautilus
   actually invokes per sample.

3. Repeat the same two assertions for both backends if the script
   exercises both — i.e. compute the assertions once with
   `use_jax=False` and once with `use_jax=True` (where applicable on
   the workspace's CI runner). This makes the regression cover any
   future drift between the CPU and JAX branches.

4. Update @autolens_workspace_test/scripts/interferometer/ and
   @autogalaxy_workspace_test/scripts/interferometer/ analogously for
   AnalysisInterferometer + Fitness. The interferometer analysis
   already returns figure_of_merit correctly (no bug there today) but
   we want a guard in case the same asymmetry gets introduced.

5. Do NOT touch the autolens_workspace or autogalaxy_workspace user-
   facing tutorials. These assertions are integration-test
   infrastructure and belong in the `_test` workspaces only.

References (read first):

- @PyAutoLens/test_autolens/imaging/model/test_analysis_imaging.py — see
  the new unit test added in PR #504 (lines around
  `test__log_likelihood_function__returns_figure_of_merit_for_pixelization`)
  for the assertion pattern to mirror at integration scale.
- @PyAutoLens/autolens/imaging/model/analysis.py — the fixed
  `log_likelihood_function`; the assertions here are testing that
  future edits to this method don't reintroduce the asymmetry.
- @PyAutoFit/autofit/non_linear/fitness.py — `Fitness.call_wrap`
  signature and behaviour.

Before starting, run @autolens_workspace_test/scripts/imaging/model_fit.py
and one of the visualization_jit scripts in their current form to see
what they produce. Then propose where the assertion blocks should sit
in each file (before the search? in a dedicated `__Likelihood Sanity__`
section?), confirm with me, then implement.
