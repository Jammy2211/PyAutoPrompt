Create `scripts/imaging/` in @autogalaxy_workspace_test with autogalaxy versions of a **subset**
of the autolens_workspace_test imaging scripts. Per the user's directive, only port four scripts:

- `model_fit.py`
- `modeling_visualization_jit.py`
- `visualization.py`
- `visualization_jax.py`

Do **not** port `convolution.py`, `modeling_visualization_jit_delaunay.py`,
`modeling_visualization_jit_rectangular.py`, or the full `simulator/`, `config/`, `config_source/`,
`images/` trees unless individual scripts fail without them.

__Reference__

@autolens_workspace_test/scripts/imaging/

Strip lens/source split — use `ag.Galaxy` + `ag.Galaxies` + `ag.ImagingAnalysis`. The
`visualization*.py` scripts exercise the plotting API and are mostly mechanical renames
(`al.*` → `ag.*`, remove tracer-specific plots).

__Scripts__

1. `imaging/model_fit.py` — end-to-end model fit on a small imaging dataset. Use the same
   `PYAUTOFIT_TEST_MODE=2` flow as the autolens version.
2. `imaging/modeling_visualization_jit.py` — exercises `analysis.fit_for_visualization` under
   `jax.jit`. Depends on PyAutoGalaxy pytree registration (task 3). If
   `linear_light_profile_intensity_dict_pytree` is needed (autogalaxy side), spawn the library
   fix first.
3. `imaging/visualization.py` — exercises the autogalaxy plotter API end-to-end. NumPy only.
4. `imaging/visualization_jax.py` — same, under JAX.

__Dataset / config__

Reuse an existing autogalaxy imaging dataset (check `autogalaxy_workspace/dataset/imaging/`). Add
a small `config/` directory at `scripts/imaging/config/` only if the default autogalaxy config
doesn't suffice.

__Deliverables__

1. `autogalaxy_workspace_test/scripts/imaging/__init__.py`
2. The four scripts above.
3. Appended to `smoke_tests.txt`.
4. Any PyAutoGalaxy library PRs for missing pytree registration (likely already covered by
   task 3; spawn off if surfacing here).

__Depends on__

Task 3 (PyAutoGalaxy imaging pytree registration). `model_fit.py` and `visualization.py` can run
without it (NumPy path), but `modeling_visualization_jit.py` and `visualization_jax.py` cannot.

__Umbrella issue__

Task 9/9. Track under the epic issue on `PyAutoLabs/autogalaxy_workspace_test`.
