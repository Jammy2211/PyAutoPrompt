When running pixelization scripts under `PYAUTO_TEST_MODE=2`, an `AttributeError` is raised
inside `BorderRelocator.relocated_mesh_grid_from` because `source_plane_mesh_grid` arrives as
`None`. This was masked for a long time by the `positions_likelihood_from` zero-size-array
crash (now fixed in `@PyAutoLens` PR #479 / issue #477). With that crash gone, the next test-
mode pixelization run hits this one immediately.

## Reproducer

```bash
cd autolens_workspace
PYAUTO_TEST_MODE=2 python scripts/imaging/features/pixelization/delaunay.py
```

## Traceback

```
File "@PyAutoLens/autolens/imaging/model/analysis.py", line 79, in log_likelihood_function
File "@PyAutoArray/autoarray/fit/fit_dataset.py", line 361, in figure_of_merit
File "@PyAutoLens/autolens/imaging/fit_imaging.py", line 156, in inversion
File "@PyAutoLens/autolens/lens/to_inversion.py", line 479, in inversion
File "@PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py", line 205, in linear_obj_list
File "@PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py", line 188, in linear_obj_galaxy_dict
File "@PyAutoLens/autolens/lens/to_inversion.py", line 436, in mapper_galaxy_dict
File "@PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py", line 489, in mapper_from
File "@PyAutoArray/autoarray/inversion/mesh/mesh/delaunay.py", line 158, in interpolator_from
File "@PyAutoArray/autoarray/inversion/mesh/mesh/abstract.py", line 91, in relocated_mesh_grid_from
File "@PyAutoArray/autoarray/inversion/mesh/border_relocator.py", line 450, in relocated_mesh_grid_from
    grid=mesh_grid.array, origin=origin, a=a, b=b, phi=phi, xp=xp
         ^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'array'
```

## Where the None comes from

`@PyAutoArray/autoarray/inversion/mesh/mesh/abstract.py:90-94` calls
`border_relocator.relocated_mesh_grid_from(grid=source_plane_data_grid, mesh_grid=source_plane_mesh_grid, xp=xp)`
without checking that `source_plane_mesh_grid` is non-None — under `PYAUTO_TEST_MODE=2` the
random/unphysical mass model traces all the data grid into NaN/inf and the upstream pipeline
hands the relocator a `None` mesh grid.

`@PyAutoArray/autoarray/inversion/mesh/border_relocator.py:429-456` then dereferences
`mesh_grid.array` without a None guard.

## What to investigate

1. Walk back up the call chain to find where `source_plane_mesh_grid` becomes `None`. Is it
   produced lazily inside the mesh class? Inside the `to_inversion` machinery on
   `@PyAutoGalaxy`? Or somewhere in the source-plane ray-tracing? The right fix probably
   lives at the point of construction, not at the relocator.
2. Decide between **(a)** a root-cause fix that prevents `None` from being produced (preferred
   — keeps relocator semantics tight), or **(b)** a defensive guard in `BorderRelocator`/the
   abstract mesh layer that returns the unrelocated grid (or a synthetic stand-in) when the
   mesh grid is missing under `is_test_mode()`.
3. Check whether option (b) should follow the same pattern as the recently-shipped
   `Result.positions_likelihood_from` fallback (`is_test_mode()` guard, synthetic stand-in,
   single `logger.warning`) so test-mode behaviour is consistent across the codebase.

## Out of scope

- Don't reintroduce a workaround at the workspace level — the `os.environ.pop` dance was just
  removed in `@autolens_workspace` PR #102.
- Don't change `@PyAutoLens` `Result.positions_likelihood_from` — that fallback already works.
  This is a separate downstream bug in `@PyAutoArray`.

## Acceptance

`PYAUTO_TEST_MODE=2 python scripts/imaging/features/pixelization/delaunay.py` should no
longer crash inside the relocator. Other pixelization scripts (`rectangular.py`,
`voronoi.py`, etc.) should be sanity-checked under the same env vars to confirm they aren't
hiding the same `None` mesh-grid path.
