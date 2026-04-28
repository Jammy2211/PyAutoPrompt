Create `scripts/jax_likelihood_functions/interferometer/` in @autogalaxy_workspace_test with
autogalaxy ports of every autolens JAX-likelihood interferometer script, **excluding** `*_dspl.py`
and `rectangular_sparse.py` unless it has an autogalaxy analogue (lens-specific; check first).

__Scripts to port__

From @autolens_workspace_test/scripts/jax_likelihood_functions/interferometer/:

- `simulator.py`
- `lp.py`
- `mge.py`
- `mge_group.py`
- `rectangular.py`
- `rectangular_mge.py`
- `delaunay.py`
- `delaunay_mge.py`

**Skip**: `rectangular_dspl.py`, `simulator_dspl.py`, and `rectangular_sparse.py` (confirm with
user if unsure whether the sparse interferometer path has an autogalaxy equivalent).

__Pytree prerequisite — likely blocker__

`autogalaxy/interferometer/model/analysis.py` has no pytree registration method. Compare the
autolens equivalent in @PyAutoLens/autolens/interferometer/model/analysis.py and mirror it on
autogalaxy — register `FitInterferometer`, `DatasetModel`, `Galaxies`.

If the registration is missing, stop and ship a PyAutoGalaxy library PR first (treat as spawn-off
task via `/start_dev`). Same policy as task 3: do not paper over with in-script registrations.

__Three-step JAX pattern__

Same contract as task 3 — NumPy baseline, JIT round-trip, scalar log-likelihood match. Print
`PASS: jit(fit_from) round-trip matches NumPy scalar.`

__Deliverables__

1. `autogalaxy_workspace_test/scripts/jax_likelihood_functions/interferometer/__init__.py`
2. Ported scripts.
3. Appended to `smoke_tests.txt`.
4. Any required PyAutoGalaxy library PRs merged first.

__Depends on__

Task 3 completing the PyAutoGalaxy `AnalysisImaging._register_fit_imaging_pytrees` scaffold —
some of its helpers (e.g. `DatasetModel` registration) will already be in place and should be
reused rather than duplicated.

__Umbrella issue__

Task 4/9. Track under the epic issue on `PyAutoLabs/autogalaxy_workspace_test`.
