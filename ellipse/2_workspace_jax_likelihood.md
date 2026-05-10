Step 2 of the ellipse-JAX series. Step 1 (`1_workspace_visualization.md`) added the visualization integration test. This prompt adds the **likelihood-function** integration tests, still on the numpy path. They lock in the reference numbers we'll later assert against once the JAX path lands in `7_analysis_ellipse_jax.md`.

The pattern to mirror is `@autogalaxy_workspace_test/scripts/jax_likelihood_functions/imaging/lp.py`, which (a) auto-simulates a small dataset on first run, (b) builds a model + analysis, (c) computes a baseline `log_likelihood` on the numpy path, and (d) compares the JIT path with `np.testing.assert_allclose(rtol=1e-4)`. For this prompt, only build steps (a)-(c) — leave a `# TODO(7_analysis_ellipse_jax.md)` placeholder where the JIT comparison will go.

Please:

1. Add `@autogalaxy_workspace_test/scripts/jax_likelihood_functions/ellipse/__init__.py`.

2. Add `@autogalaxy_workspace_test/scripts/jax_likelihood_functions/ellipse/simulator.py` modelled on `@autogalaxy_workspace_test/scripts/jax_likelihood_functions/imaging/simulator.py` — small grid (e.g. shape_native=(50, 50), pixel_scales=0.2), single Sersic galaxy, written into `dataset/ellipse/jax_test/`.

3. Add `@autogalaxy_workspace_test/scripts/jax_likelihood_functions/ellipse/fit.py`:
   - Load (or auto-simulate via `simulator.py`) the dataset.
   - Build an `af.Collection(ellipses=af.Collection(ellipse_0=af.Model(ag.Ellipse, major_axis=...)))`-style model (consult `@autogalaxy_workspace/scripts/ellipse/modeling.py` for the exact composition shape).
   - Build `analysis = ag.AnalysisEllipse(dataset=dataset)` (today this defaults to `use_jax=False`).
   - Compute `fit_np = analysis.fit_list_from(instance=model.instance_from_prior_medians())` and print every component — `log_likelihood`, `chi_squared`, `noise_normalization`, `figure_of_merit` — to capture the reference numbers.
   - Leave a `# TODO(7_analysis_ellipse_jax.md): jax.jit(analysis.fit_from) round-trip` placeholder block at the end.

4. Add `@autogalaxy_workspace_test/scripts/jax_likelihood_functions/ellipse/multipoles.py`: same as `fit.py` but with `multipole_list=[ag.EllipseMultipole(m=4, multipole_comps=(0.05, 0.0))]` per ellipse. This locks in the multipole code path which has its own JAX-incompatible `while` loops in `EllipseMultipole.get_shape_angle` (handled in prompt 5).

5. Use the workspace docstring style throughout (see prompt 1 for reference).

6. Test bar: both scripts run cleanly under `bash run_all_scripts.sh` and the printed reference numbers are stable to ~1e-6 across runs. No JAX imports yet — `import jax` should not appear in any of these files.
