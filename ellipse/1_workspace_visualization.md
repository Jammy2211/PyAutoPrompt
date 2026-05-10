Step 1 of the ellipse-JAX series. The end goal is to make `@PyAutoGalaxy/autogalaxy/ellipse/model/analysis.py`, `AnalysisEllipse.log_likelihood_function` JAX-compatible (analogous to `AnalysisImaging` in `@PyAutoGalaxy/autogalaxy/imaging/model/analysis.py`). Before any of that, we need to lock down the existing numpy behaviour with integration tests in `@autogalaxy_workspace_test/scripts`, so when later prompts rewrite the gnarly bits we can spot regressions immediately.

This prompt covers the **ellipse visualization** integration test. The follow-up `2_workspace_jax_likelihood.md` covers the likelihood-function script.

Please:

1. Add `@autogalaxy_workspace_test/scripts/visualization.py`. Pattern it on the existing `@autogalaxy_workspace/scripts/ellipse/fit.py` walkthrough but trimmed to the visualization side: load (or auto-simulate) a small dataset, fit a single `Ellipse` (and an `Ellipse + EllipseMultipole`), then exercise every public plotter path through `@PyAutoGalaxy/autogalaxy/ellipse/model/plotter.py` (`PlotterEllipse.imaging`, `PlotterEllipse.fit_ellipse`) and `aplt.FitEllipsePlotter` if one exists. Use `PYAUTO_OUTPUT_MODE=1` semantics — the script just has to run end-to-end without raising.

2. Cover the multipole code path too: a fit with `multipole_list=[ag.EllipseMultipole(m=4, multipole_comps=(0.05, 0.0))]` and a fit with `EllipseMultipoleScaled` from `@PyAutoGalaxy/autogalaxy/ellipse/ellipse/ellipse_multipole.py`.

3. Cover the masked-data code path: apply a `Mask2D` that the ellipse partially overlaps, so `FitEllipse.points_from_major_axis_from`'s 300-iteration mask-rejection loop in `@PyAutoGalaxy/autogalaxy/ellipse/fit_ellipse.py:81-134` actually fires. Without this, prompt 6 has nothing to compare against.

4. Use the workspace docstring style (`"""..."""` blocks with `__Section Name__` headers, no `#` comments) — see `@autogalaxy_workspace/scripts/ellipse/fit.py` for examples.

5. Test bar: the script runs cleanly under `bash run_all_scripts.sh` from `@autogalaxy_workspace_test/`, and produces output PNGs under `output_mode/visualization/` when `PYAUTO_OUTPUT_MODE=1` is set.

This is numpy-only — no JAX yet. The point is to have a regression target before we touch anything underneath.
