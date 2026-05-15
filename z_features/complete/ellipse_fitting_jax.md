Make `AnalysisEllipse.log_likelihood_function` JAX-compatible (analogous to `AnalysisImaging`). Decomposed from the original `autogalaxy/ellipse_fitting_jax.md` meta-prompt — see `issued/ellipse_fitting_jax.md` for the source brief.

**FEATURE COMPLETE — 2026-05-14.** All 7 prompts shipped. `AnalysisEllipse(dataset, use_jax=True)` now traces under `jax.jit` and `jax.vmap`.

see also:
- PyAutoFit `Drawer` search currently does not pass `use_jax_jit=True` to `Fitness` (see `@PyAutoFit/autofit/non_linear/search/mle/drawer/search.py:105` and `@PyAutoFit/autofit/non_linear/fitness.py:121-129`). Independent of this feature — needs a separate prompt under `autofit/`.
- Mass-multipole code in `autogalaxy/profiles/mass/total/power_law_multipole.py` calls `convert.multipole_comps_from` / `multipole_k_m_and_phi_m_from` without `xp` threading. When someone tries to JAX-jit a mass-multipole model, that call site will need the same fix prompt 7 applied to the ellipse-multipole side.

shipped:
- 1_workspace_visualization (`ellipse-visualization-test`, #39 / autogalaxy_workspace_test#40)
- 2_workspace_jax_likelihood (`ellipse-jax-likelihood-tests`, #41 / autogalaxy_workspace_test#42)
- 3_unit_tests_masked_loop (`ellipse-fit-masked-loop-tests`, PyAutoGalaxy#394 / #395)
- 4_jax_interp_2d (`jax-interp-2d`, PyAutoArray#306 / PyAutoArray#308 + PyAutoGalaxy#398)
- 5_ellipse_xp (`ellipse-xp`, PyAutoGalaxy#407 / #408)
- 6_fit_ellipse_masked_jax (`fit-ellipse-jax`, PyAutoGalaxy#409 / #410) — also threw away the 300-iter loop entirely in favour of a unified NaN-marking algorithm
- 7_analysis_ellipse_jax (`analysis-ellipse-jax`, PyAutoGalaxy#411 / #412 + autogalaxy_workspace_test#48) — keystone needed 5 follow-up fix commits to thread `xp` through `convert.py` helpers, `Ellipse` properties, and `EllipseMultipole` inherited-method calls; bug pattern exposed by `vmap(fitness)` not `jit(fit_from)(concrete_instance)`
