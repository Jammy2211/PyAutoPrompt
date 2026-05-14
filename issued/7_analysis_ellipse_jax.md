Step 7 of the ellipse-JAX series — the keystone. Prompts 4-6 made every piece JAX-traceable; this prompt wires `AnalysisEllipse` so `jax.jit(analysis.fit_from)(instance)` works end to end. The template is `AnalysisImaging` in `@PyAutoGalaxy/autogalaxy/imaging/model/analysis.py:30-187`, which has all the moving parts (`use_jax: bool = True`, `_register_fit_imaging_pytrees()`, `super().__init__(use_jax=use_jax)`).

Please:

1. In `@PyAutoGalaxy/autogalaxy/ellipse/model/analysis.py`:
   - Add `use_jax: bool = True` to `AnalysisEllipse.__init__` and pass it through `super().__init__(use_jax=use_jax)`. Default `True` matches `AnalysisImaging`.
   - Add a `fit_from(instance: af.ModelInstance) -> FitEllipse` method (today only `fit_list_from` exists). It should mirror `AnalysisImaging.fit_from`: build the `FitEllipse` (or list of `FitEllipse` collapsed into a sum-figure-of-merit wrapper), call `_register_fit_ellipse_pytrees()` once when `self._use_jax`, return the resulting `FitEllipse`.
   - Update `log_likelihood_function` to call `self.fit_from(instance).figure_of_merit` (or sum the list, matching the existing logic). The existing `fit_list_from` stays — it's used by `VisualizerEllipse.visualize` in `@PyAutoGalaxy/autogalaxy/ellipse/model/visualizer.py:64`.

2. Implement `_register_fit_ellipse_pytrees()` modelled on `AnalysisImaging._register_fit_imaging_pytrees()` (lines 168-187). Register:
   - `FitEllipse` with `no_flatten=("dataset",)`. The `interp` cached property reconstructs from `dataset` so it's safe to skip flattening.
   - `Ellipse` (generic flatten via `register_instance_pytree`).
   - `EllipseMultipole` and `EllipseMultipoleScaled` (generic flatten).
   - Reuse the helper from `autoarray.abstract_ndarray.register_instance_pytree`. Make the function idempotent — match the registry-guard pattern in the imaging analysis.
   - Place a thin shim `@PyAutoGalaxy/autogalaxy/analysis/jax_pytrees.py::register_ellipses_pytree()` if useful to mirror `register_galaxies_pytree`, but it's optional — generic registration may be enough since `Ellipse`s are stored on `instance.ellipses` as a list, not a custom container.

3. Flip the workspace_test scripts from prompt 2 (`@autogalaxy_workspace_test/scripts/jax_likelihood_functions/ellipse/{fit.py, multipoles.py}`) to exercise the JIT path:
   - Replace the `# TODO(7_analysis_ellipse_jax.md)` placeholder with the actual JIT round-trip block, modelled on `@autogalaxy_workspace_test/scripts/jax_likelihood_functions/imaging/lp.py:107-129` — `analysis_jit = ag.AnalysisEllipse(dataset=dataset, use_jax=True); fit_jit_fn = jax.jit(analysis_jit.fit_from); fit = fit_jit_fn(instance)`.
   - Assert `np.testing.assert_allclose(float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4)` against the numpy reference computed earlier in the script.
   - Assert `isinstance(fit.log_likelihood, jnp.ndarray)`.
   - Add a `fitness._vmap` batch-evaluation block too, mirroring `imaging/lp.py:74-98`. This catches issues that only surface under `jax.vmap`.

4. Add a unit test in `@PyAutoGalaxy/test_autogalaxy/ellipse/test_analysis.py` that constructs `AnalysisEllipse(dataset, use_jax=False)` and asserts the existing numpy `log_likelihood_function` value is unchanged for a known instance. The JAX-path checks live in the workspace_test scripts — `@PyAutoGalaxy/CLAUDE.md` "Never use JAX in unit tests".

5. Test bar:
   - `python -m pytest test_autogalaxy/ -v` passes (no regressions in the imaging/interferometer paths).
   - The two workspace_test scripts run cleanly and the JIT path matches the numpy reference to `rtol=1e-4`.
   - `bash run_all_scripts.sh` from `@autogalaxy_workspace_test/` is green.

After this lands, ellipse modeling can run inside `Drawer` / `Nautilus` / any other JAX-compatible search the same way `AnalysisImaging` does today. Note that `Drawer` itself still needs a small fix to pass `use_jax_jit=True` through to `Fitness` (out of scope for this series — see the `z_features/ellipse_fitting_jax.md` "see also" note).
