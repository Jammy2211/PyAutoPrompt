Create `scripts/jax_likelihood_functions/multi/` in @autogalaxy_workspace_test with autogalaxy
ports of every autolens multi-dataset JAX-likelihood script, **excluding** `*_dspl.py`.

__Scripts to port__

From @autolens_workspace_test/scripts/jax_likelihood_functions/multi/:

- `simulator.py`
- `lp.py`
- `mge.py`
- `mge_group.py`
- `rectangular.py`
- `rectangular_mge.py`
- `delaunay.py`
- `delaunay_mge.py`

**Skip**: any `*_dspl.py`.

__Context__

The `multi/` scripts exercise `af.FactorGraphModel` / multi-dataset joint fits. The autolens
versions combine an imaging dataset and an interferometer dataset with tied lens-galaxy params.
Autogalaxy versions should tie **galaxy** params across datasets (no lens/source split).

__Pytree prerequisite__

Both `AnalysisImaging` and `AnalysisInterferometer` on autogalaxy need pytree-registered
`fit_from`. If tasks 3 and 4 have landed first, this should follow naturally. If any multi-dataset
registration is missing (e.g. the factor-graph combining analysis), spawn a library task.

__Three-step JAX pattern__

Same as tasks 3 and 4.

__Deliverables__

1. `autogalaxy_workspace_test/scripts/jax_likelihood_functions/multi/__init__.py`
2. Ported scripts.
3. Appended to `smoke_tests.txt`.

__Depends on__

Tasks 3 and 4 (imaging + interferometer pytree registration scaffolding in place).

__Umbrella issue__

Task 5/9. Track under the epic issue on `PyAutoLabs/autogalaxy_workspace_test`.
