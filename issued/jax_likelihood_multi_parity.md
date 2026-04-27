Extend `autolens_workspace_test/scripts/jax_likelihood_functions/multi/` from the single
`mge.py` to match the coverage already in `.../jax_likelihood_functions/imaging/`, using the
same `FactorGraphModel(*factors, use_jax=True)` wiring already established in `multi/mge.py`.

__Baseline already shipped__

- `multi/mge.py` — two-band (g + r) shared MGE model with `FactorGraphModel(use_jax=True)` and
  the three-step pattern.
- `multi/simulator.py` — two-band seeded simulator writing
  `dataset/multi/lens_sersic/{g,r}_{data,psf,noise_map}.fits`.

Imaging reference surface (each to be mirrored to a two-band `FactorGraphModel` equivalent):

- `imaging/lp.py`
- `imaging/delaunay.py` (Hilbert image-mesh + edge-zeroed points)
- `imaging/rectangular.py`
- `imaging/mge_group.py`
- `imaging/delaunay_mge.py`
- `imaging/rectangular_mge.py`
- `imaging/rectangular_dspl.py`
- `imaging/simulator_dspl.py`

__Reference scripts__

- Imaging templates: `@autolens_workspace_test/scripts/jax_likelihood_functions/imaging/{lp,rectangular,delaunay,mge_group,delaunay_mge,rectangular_mge,rectangular_dspl,simulator_dspl}.py`.
- Existing multi prior art: `@autolens_workspace_test/scripts/jax_likelihood_functions/multi/{mge,simulator}.py`
  — use its `dataset_list` + `analysis_factor_list` + `FactorGraphModel(*factors, use_jax=True)`
  pattern verbatim; swap in the per-variant model.

__Scripts to add__ in `autolens_workspace_test/scripts/jax_likelihood_functions/multi/`

1. `lp.py` — shared parametric Sersic lens + source across g / r bands.
2. `rectangular.py` — shared rectangular pixelization source across bands.
3. `delaunay.py` — shared Delaunay pixelization source across bands, with `al.image_mesh.Hilbert`
   image-mesh and edge-zeroed points (match `imaging/delaunay.py` plumbing).
4. `mge_group.py` — shared MGE + extra galaxies across bands.
5. `delaunay_mge.py` — Delaunay source + MGE lens across bands.
6. `rectangular_mge.py` — Rectangular source + MGE lens across bands.
7. `rectangular_dspl.py` — Double source plane rectangular pixelization across bands.
8. `simulator_dspl.py` — Two-band double source plane simulator. Mirror
   `imaging/simulator_dspl.py` and extend to two bands using the same `for band in ["g", "r"]`
   loop as `multi/simulator.py`. Writes to `dataset/multi/dspl/{g,r}_{data,psf,noise_map}.fits`.

__Wiring__ (copy from `multi/mge.py` verbatim and vary only the model / pixelization)

```python
dataset_list = [al.Imaging.from_fits(...) for band in ("g", "r")]
mask_list = [al.Mask2D.circular(...) for dataset in dataset_list]
dataset_list = [dataset.apply_mask(mask=mask) for dataset, mask in zip(dataset_list, mask_list)]

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))  # fully shared
analysis_list = [al.AnalysisImaging(dataset=dataset, adapt_images=...) for dataset in dataset_list]
analysis_factor_list = [
    af.AnalysisFactor(prior_model=model, analysis=analysis)
    for analysis in analysis_list
]
factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

fitness = Fitness(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)
```

Then the three-step pattern on `fitness._vmap` + Path A `jax.jit(factor_graph.fit_from)` round-trip.

__Adapt images for pixelization variants__

Each `AnalysisImaging` needs its own per-band `AdaptImages` wired in, matching the per-band dataset.
Pattern: per-band `galaxy_name_image_dict` with `dataset.data` as the adapt image, plus the shared
`image_plane_mesh_grid` (computed once from one band's mask + data since the masks are identical).

__Auto-simulation boilerplate__

Each new script must include the `should_simulate(dataset_path)` bootstrap invoking the matching
`simulator.py` or `simulator_dspl.py` via `subprocess.run([sys.executable, ...], check=True)`.

__Expected log-likelihood values__

Each script's vmap assertion has a hardcoded expected log-likelihood. On first run the author
fills in the actual value (same pattern as `multi/mge.py`'s `-2174335.96508048`), then the
assertion becomes a regression guard.

__Deliverables__

1. Eight new scripts in `autolens_workspace_test/scripts/jax_likelihood_functions/multi/`:
   `lp.py`, `rectangular.py`, `delaunay.py`, `mge_group.py`, `delaunay_mge.py`,
   `rectangular_mge.py`, `rectangular_dspl.py`, `simulator_dspl.py`.
2. One new dataset folder: `dataset/multi/dspl/` produced by `simulator_dspl.py`.
3. Each script (except simulators) ends by printing `PASS: jit(fit_from) round-trip matches NumPy scalar.`
4. Update `autolens_workspace_test/scripts/CLAUDE.md` to list the new multi scripts in the
   `jax_likelihood_functions/` table.

__Scope boundary__

- `FactorGraphModel` pytree registration is already exercised by `multi/mge.py` — no library-side
  work expected. If a blocker surfaces on a pixelization variant, it is more likely to be a
  per-band `AdaptImages` wiring issue than a `FactorGraphModel` bug.
- Do not introduce per-band **free** parameters — keep the model fully shared across bands, same
  as `multi/mge.py`. (A per-band intensity split is a follow-up task, not part of this parity
  sweep.)
